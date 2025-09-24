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
from utils import allreduce, alltoall, compute
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)


class TransformerMoeLayer(Layer):
    """An implementation of a Transformer Mixture of Experts Layer.
       Incomplete.
       TODO: Implement routing
       TODO: Implement imbalanced all-to-all collectives
    """
    
    def __init__(self, 
        num_layers: int,
        hidden_size: int,
        sequence_len: int,
        vocab_size: int,
        ep_size: int,
        tp_size: int,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        bytes_per_val: int = 2,
        scale: float = 1):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.bytes_per_val = bytes_per_val
        self.ep_size = ep_size
        self.tp_size = tp_size
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.scale = scale
    
    def fwd(self, name="node_fwd", pg_name=None, num_batches=1) -> list[ChakraNode]:
        tensor_size = int((12*self.hidden_size*self.hidden_size*self.bytes_per_val + num_batches*self.sequence_len*self.hidden_size*self.bytes_per_val) * self.scale)

        # ------------------------------------------------------------------
        # Attention block (identical to dense transformer layer)
        # ------------------------------------------------------------------
        # Routing & expert parallelism come *after* attention.

        # calculate flops for the attention block
        attention_flops = int(self.scale * (8 * num_batches*self.sequence_len*self.hidden_size*self.hidden_size + 4*num_batches*self.sequence_len*self.sequence_len*self.hidden_size))
        attention_compute = compute(attention_flops, tensor_size, name=f"{name}_attention_compute")

        # ------------------------------------------------------------------
        # Gating (token -> expert assignment)
        # ------------------------------------------------------------------
        gating_flops = int(self.scale * 2 * num_batches * self.sequence_len * self.hidden_size)
        gating_tensor = int(self.scale * self.bytes_per_val * num_batches * self.sequence_len * self.hidden_size)
        gating_compute = compute(gating_flops, gating_tensor, name=f"{name}_gating_compute")
        
        # ------------------------------------------------------------------
        # Expert-parallel all-to-all (token shuffle) – happens between attention
        # and feed-forward experts when ep_size > 1.
        # ------------------------------------------------------------------
        ep_a2a_node = None
        if self.ep_size > 1:
            tokens_per_rank = num_batches * self.sequence_len * self.capacity_factor * self.top_k
            ep_comm_size = int(self.scale * self.bytes_per_val * tokens_per_rank * self.hidden_size)
            ep_a2a_node = alltoall(ep_comm_size, pg_name=pg_name, parents=[gating_compute], name=f"{name}_ep_alltoall")
        
        # TP all-reduce after attention dense projection
        attention_allreduce = None
        if self.tp_size > 1:
            tp_comm_size = int(self.scale * self.bytes_per_val * self.sequence_len * num_batches * self.hidden_size)
            attention_allreduce = allreduce(tp_comm_size, pg_name=pg_name, name=f"{name}_attention_allreduce")
        
        # calculate flops for the mlp block
        ffwd_flops = int(self.scale * 16 * num_batches*self.sequence_len*self.hidden_size*self.hidden_size)
        # Feed-forward experts compute (after token shuffle)
        ffwd_compute_parents = [ep_a2a_node] if ep_a2a_node is not None else [gating_compute]
        ffwd_compute = compute(ffwd_flops, tensor_size, parents=ffwd_compute_parents, name=f"{name}_ffwd_compute")
        
        #layer_flops = int((attention_flops + ffwd_flops) * self.scale)
        
        
        # GPU at 55% capacity
        #slowdown = 1.8182
        #slowdown = 1
        #layer_flops = int(layer_flops * slowdown)

        ffwd_allreduce = None
        if self.tp_size > 1:
            tp_comm_size = int(self.scale * self.bytes_per_val * self.sequence_len * num_batches * self.hidden_size)
            ffwd_allreduce = allreduce(tp_comm_size, pg_name=pg_name, name=f"{name}_mlp_allreduce")

        # Assemble node list (skip Nones to avoid NameErrors)
        nodes = [attention_compute, gating_compute]
        for n in (ep_a2a_node, attention_allreduce, ffwd_compute, ffwd_allreduce):
            if n is not None:
                nodes.append(n)

        return nodes


    def bckwd(self, name="node_bckwd", pg_name=None, num_batches=1) -> list[ChakraNode]:
        tensor_size = int((12*self.hidden_size*self.hidden_size*self.bytes_per_val + num_batches*self.sequence_len*self.hidden_size*self.bytes_per_val) * self.scale)

        # calculate flops for the attention block
        attention_flops = int(self.scale * (8 * num_batches*self.sequence_len*self.hidden_size*self.hidden_size + 4*num_batches*self.sequence_len*self.sequence_len*self.hidden_size))
        attention_compute = compute(2 * attention_flops, tensor_size=tensor_size, name=f"{name}_attention_compute")
        
        attention_allreduce = None
        if self.tp_size > 1:
            tp_comm_size = int(self.scale * self.bytes_per_val * self.sequence_len * num_batches * self.hidden_size)
            attention_allreduce = allreduce(tp_comm_size, pg_name=pg_name, name=f"{name}_attention_allreduce")
        
        # calculate flops for the mlp block
        ffwd_flops = int(self.scale * 16 * num_batches*self.sequence_len*self.hidden_size*self.hidden_size)
        ffwd_compute = compute(2 * ffwd_flops, tensor_size, name=f"{name}_ffwd_compute")

        ffwd_allreduce = None
        if self.tp_size > 1:
            tp_comm_size = int(self.scale * self.bytes_per_val * self.sequence_len * num_batches * self.hidden_size)
            ffwd_allreduce = allreduce(tp_comm_size, pg_name=pg_name, name=f"{name}_mlp_allreduce")

        # ------------------------------------------------------------------
        # Attention + experts backward
        # ------------------------------------------------------------------
        # Expert-parallel alltoall backward (experts -> original ranks)
        ep_a2a_back = None
        if self.ep_size > 1:
            tokens_per_rank = num_batches * self.sequence_len * self.capacity_factor * self.top_k
            ep_comm_size = int(self.scale * self.bytes_per_val * tokens_per_rank * self.hidden_size)
            ep_a2a_back = alltoall(ep_comm_size, pg_name=pg_name, parents=[ffwd_compute], name=f"{name}_ep_alltoall_back")

        # Gating backward compute (grad through softmax & top-k)
        gating_grad_parents = [ep_a2a_back] if ep_a2a_back is not None else [ffwd_compute]
        gating_grad_compute = compute(2 * int(self.scale * 2 * num_batches * self.sequence_len * self.hidden_size), tensor_size, parents=gating_grad_parents, name=f"{name}_gating_grad")

        nodes = [ffwd_compute]
        for n in (ep_a2a_back, gating_grad_compute, ffwd_allreduce, attention_compute, attention_allreduce):
            if n is not None:
                nodes.append(n)

        return nodes