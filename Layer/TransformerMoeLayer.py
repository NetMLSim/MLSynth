# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Layer.Layer import Layer
from Model.MoeRouting import MoeRouter, RoutingResult
from utils import allreduce, alltoall, compute, send, receive
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)

# unique tag per (layer, microbatch, phase) so send/recv pairs match
_PHASE_FWD_DISPATCH = 0
_PHASE_FWD_COMBINE = 1
_PHASE_BWD_GRAD = 2
_PHASE_BWD_COMBINE = 3


class TransformerMoeLayer(Layer):
    """An implementation of a Transformer Mixture of Experts Layer."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        sequence_len: int,
        vocab_size: int,
        ep_size: int,
        tp_size: int,
        num_experts: int,
        router: MoeRouter,
        layer_index: int = 0,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        bytes_per_val: int = 2,
        scale: float = 1,
        a2a_mode: str = "pairwise",
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.bytes_per_val = bytes_per_val
        self.ep_size = ep_size
        self.tp_size = tp_size
        self.num_experts = num_experts
        self.router = router
        self.layer_index = layer_index
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.scale = scale
        self.a2a_mode = a2a_mode

    # high bit keeps MoE tags off the pipeline send/recv tag (0)
    _TAG_BASE = 1 << 28

    def _tag(self, microbatch: int, phase: int) -> int:
        """Comm tag unique to (layer, microbatch, phase)."""
        return self._TAG_BASE | ((int(self.layer_index) & 0x3FF) << 12) | ((int(microbatch) & 0x3FF) << 2) | (phase & 0x3)

    @staticmethod
    def _stem(name: str) -> str:
        """Strip the orchestrator COMP_NODE_ prefix for comm node names."""
        prefix = "COMP_NODE_"
        return name[len(prefix):] if name.startswith(prefix) else name

    def _pairwise_a2a(
        self,
        routing: RoutingResult,
        ep_local: int,
        ep_member_ids: list[int],
        my_global: int,
        bytes_per_token: float,
        parents: list[ChakraNode],
        tag: int,
        mode: str,
        label: str,
    ) -> list[ChakraNode]:
        """Emit this rank's send/recv nodes for one all-to-all."""
        nodes: list[ChakraNode] = []
        for r in range(self.ep_size):
            if r == ep_local:
                continue
            peer_global = int(ep_member_ids[r])
            if mode == "dispatch":
                send_tokens = int(routing.matrix[ep_local][r])
                recv_tokens = int(routing.matrix[r][ep_local])
            else:
                send_tokens = int(routing.matrix[r][ep_local])
                recv_tokens = int(routing.matrix[ep_local][r])
            if send_tokens > 0:
                nodes.append(send(
                    my_global, peer_global, int(send_tokens * bytes_per_token),
                    name=f"COMM_SEND_NODE_{label}_{my_global}to{peer_global}", parents=parents, tag=tag))
            if recv_tokens > 0:
                nodes.append(receive(
                    peer_global, my_global, int(recv_tokens * bytes_per_token),
                    name=f"COMM_RECV_NODE_{label}_{peer_global}to{my_global}", parents=parents, tag=tag))
        return nodes

    def _tensor_size(self, num_batches) -> int:
        return int((12 * self.hidden_size * self.hidden_size * self.bytes_per_val
                    + num_batches * self.sequence_len * self.hidden_size * self.bytes_per_val) * self.scale)

    def _tp_comm_size(self, num_batches) -> int:
        return int(self.scale * self.bytes_per_val * self.sequence_len * num_batches * self.hidden_size)

    def fwd(
        self,
        name: str = "node_fwd",
        pg_name: str | None = None,
        num_batches: int = 1,
        ep_local: int = 0,
        ep_member_ids: list[int] | None = None,
        my_global: int = 0,
        key=(0,),
        microbatch: int = 0,
        ep_pg_name: str | None = None,
    ) -> list[ChakraNode]:
        tensor_size = self._tensor_size(num_batches)
        bytes_per_token = self.hidden_size * self.bytes_per_val * self.scale
        tokens_per_rank = int(num_batches * self.sequence_len)
        stem = self._stem(name)

        # calculate flops for the attention block
        attention_flops = int(self.scale * (8 * num_batches * self.sequence_len * self.hidden_size * self.hidden_size)
                                            + 4 * num_batches * self.sequence_len * self.sequence_len * self.hidden_size))
        attention_compute = compute(attention_flops, tensor_size, name=f"{name}_attention_compute")

        attention_allreduce = None
        if self.tp_size > 1:
            attention_allreduce = allreduce(self._tp_comm_size(num_batches), pg_name=pg_name,
                                            parents=[attention_compute], name=f"COMM_COLL_NODE_{stem}_attention_allreduce")

        # gating (token -> expert assignment)
        gating_flops = int(self.scale * 2 * num_batches * self.sequence_len * self.hidden_size)
        gating_tensor = int(self.scale * self.bytes_per_val * num_batches * self.sequence_len * self.hidden_size)
        gating_parent = attention_allreduce if attention_allreduce is not None else attention_compute
        gating_compute = compute(gating_flops, gating_tensor, parents=[gating_parent], name=f"{name}_gating_compute")

        nodes: list[ChakraNode] = [attention_compute]
        if attention_allreduce is not None:
            nodes.append(attention_allreduce)
        nodes.append(gating_compute)

        do_a2a = self.ep_size > 1 and ep_member_ids is not None

        if do_a2a and self.a2a_mode == "pairwise":
            routing = self.router.route(key, tokens_per_rank)
            received = int(routing.received_tokens[ep_local])

            dispatch_nodes = self._pairwise_a2a(
                routing, ep_local, ep_member_ids, my_global, bytes_per_token,
                parents=[gating_compute], tag=self._tag(microbatch, _PHASE_FWD_DISPATCH),
                mode="dispatch", label=f"{stem}_ep_dispatch")
            nodes.extend(dispatch_nodes)

            ffwd_flops = int(self.scale * 16 * received * self.hidden_size * self.hidden_size)
            ffwd_parents = dispatch_nodes if dispatch_nodes else [gating_compute]
            ffwd_compute = compute(ffwd_flops, tensor_size, parents=ffwd_parents, name=f"{name}_ffwd_compute")
            nodes.append(ffwd_compute)

            ffwd_allreduce = None
            if self.tp_size > 1:
                ffwd_allreduce = allreduce(self._tp_comm_size(num_batches), pg_name=pg_name,
                                           parents=[ffwd_compute], name=f"COMM_COLL_NODE_{stem}_mlp_allreduce")
                nodes.append(ffwd_allreduce)

            combine_parent = ffwd_allreduce if ffwd_allreduce is not None else ffwd_compute
            combine_nodes = self._pairwise_a2a(
                routing, ep_local, ep_member_ids, my_global, bytes_per_token,
                parents=[combine_parent], tag=self._tag(microbatch, _PHASE_FWD_COMBINE),
                mode="combine", label=f"{stem}_ep_combine")
            nodes.extend(combine_nodes)

            # combine sink so the next layer waits for the full all-to-all
            combine_flops = int(self.scale * 2 * received * self.hidden_size)
            combine_parents = combine_nodes if combine_nodes else [combine_parent]
            combine_compute = compute(combine_flops, gating_tensor, parents=combine_parents,
                                      name=f"{name}_ep_combine_compute")
            nodes.append(combine_compute)
            return nodes

        # collective all-to-all, or no cross-rank traffic when ep_size == 1
        ep_a2a_node = None
        if do_a2a and self.a2a_mode == "collective":
            routing = self.router.route(key, tokens_per_rank)
            received = int(routing.received_tokens[ep_local])
            sent_tokens = int(routing.matrix[ep_local].sum() - routing.matrix[ep_local][ep_local])
            ep_a2a_node = alltoall(int(sent_tokens * bytes_per_token), pg_name=ep_pg_name,
                                   parents=[gating_compute], name=f"COMM_COLL_NODE_{stem}_ep_alltoall")
            nodes.append(ep_a2a_node)
        else:
            received = tokens_per_rank * self.top_k

        ffwd_flops = int(self.scale * 16 * received * self.hidden_size * self.hidden_size)
        ffwd_parents = [ep_a2a_node] if ep_a2a_node is not None else [gating_compute]
        ffwd_compute = compute(ffwd_flops, tensor_size, parents=ffwd_parents, name=f"{name}_ffwd_compute")
        nodes.append(ffwd_compute)

        ffwd_allreduce = None
        if self.tp_size > 1:
            ffwd_allreduce = allreduce(self._tp_comm_size(num_batches), pg_name=pg_name,
                                       parents=[ffwd_compute], name=f"COMM_COLL_NODE_{stem}_mlp_allreduce")
            nodes.append(ffwd_allreduce)

        if ep_a2a_node is not None:
            combine_parent = ffwd_allreduce if ffwd_allreduce is not None else ffwd_compute
            ep_combine = alltoall(int(sent_tokens * bytes_per_token), pg_name=ep_pg_name,
                                  parents=[combine_parent], name=f"COMM_COLL_NODE_{stem}_ep_alltoall_combine")
            nodes.append(ep_combine)
        return nodes

    def bckwd(
        self,
        name: str = "node_bckwd",
        pg_name: str | None = None,
        num_batches: int = 1,
        ep_local: int = 0,
        ep_member_ids: list[int] | None = None,
        my_global: int = 0,
        key=(0,),
        microbatch: int = 0,
        ep_pg_name: str | None = None,
    ) -> list[ChakraNode]:
        tensor_size = self._tensor_size(num_batches)
        bytes_per_token = self.hidden_size * self.bytes_per_val * self.scale
        tokens_per_rank = int(num_batches * self.sequence_len)
        stem = self._stem(name)

        do_a2a = self.ep_size > 1 and ep_member_ids is not None
        routing = None
        if do_a2a and self.a2a_mode in ("pairwise", "collective"):
            routing = self.router.route(key, tokens_per_rank)
            received = int(routing.received_tokens[ep_local])
        else:
            received = tokens_per_rank * self.top_k

        # calculate flops for the mlp block
        ffwd_flops = int(self.scale * 16 * received * self.hidden_size * self.hidden_size)
        ffwd_compute = compute(2 * ffwd_flops, tensor_size, name=f"{name}_ffwd_compute")
        nodes: list[ChakraNode] = [ffwd_compute]

        ffwd_allreduce = None
        if self.tp_size > 1:
            ffwd_allreduce = allreduce(self._tp_comm_size(num_batches), pg_name=pg_name,
                                       parents=[ffwd_compute], name=f"COMM_COLL_NODE_{stem}_mlp_allreduce")
            nodes.append(ffwd_allreduce)

        gating_grad_parent = ffwd_allreduce if ffwd_allreduce is not None else ffwd_compute

        if routing is not None and self.a2a_mode == "pairwise":
            # expert-parallel alltoall backward
            dgrad_nodes = self._pairwise_a2a(
                routing, ep_local, ep_member_ids, my_global, bytes_per_token,
                parents=[gating_grad_parent], tag=self._tag(microbatch, _PHASE_BWD_GRAD),
                mode="dispatch", label=f"{stem}_ep_dgrad")
            nodes.extend(dgrad_nodes)
            gating_grad_parents = dgrad_nodes if dgrad_nodes else [gating_grad_parent]
        elif routing is not None and self.a2a_mode == "collective":
            sent_tokens = int(routing.matrix[ep_local].sum() - routing.matrix[ep_local][ep_local])
            dgrad = alltoall(int(sent_tokens * bytes_per_token), pg_name=ep_pg_name,
                             parents=[gating_grad_parent], name=f"COMM_COLL_NODE_{stem}_ep_alltoall_dgrad")
            nodes.append(dgrad)
            gating_grad_parents = [dgrad]
        else:
            gating_grad_parents = [gating_grad_parent]

        gating_grad_flops = int(self.scale * 2 * num_batches * self.sequence_len * self.hidden_size)
        gating_grad = compute(2 * gating_grad_flops, tensor_size, parents=gating_grad_parents,
                              name=f"{name}_gating_grad")
        nodes.append(gating_grad)

        attention_parent = gating_grad
        if routing is not None and self.a2a_mode == "pairwise":
            cgrad_nodes = self._pairwise_a2a(
                routing, ep_local, ep_member_ids, my_global, bytes_per_token,
                parents=[gating_grad], tag=self._tag(microbatch, _PHASE_BWD_COMBINE),
                mode="combine", label=f"{stem}_ep_cgrad")
            nodes.extend(cgrad_nodes)
            if cgrad_nodes:
                attention_parent = cgrad_nodes

        attention_flops = int(self.scale * (8 * num_batches * self.sequence_len * self.hidden_size * self.hidden_size
                                            + 4 * num_batches * self.sequence_len * self.sequence_len * self.hidden_size))
        attention_parents = attention_parent if isinstance(attention_parent, list) else [attention_parent]
        attention_compute = compute(2 * attention_flops, tensor_size, parents=attention_parents,
                                    name=f"{name}_attention_compute")
        nodes.append(attention_compute)

        if self.tp_size > 1:
            attention_allreduce = allreduce(self._tp_comm_size(num_batches), pg_name=pg_name,
                                            parents=[attention_compute], name=f"COMM_COLL_NODE_{stem}_attention_allreduce")
            nodes.append(attention_allreduce)

        return nodes
