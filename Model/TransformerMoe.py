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

import re

from Model.Model import Model
from Model.MoeRouting import MoeRouter
from Layer.TransformerMoeLayer import TransformerMoeLayer
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)


class TransformerMoe(Model):
    """A Mixture-of-Experts Transformer model."""

    is_moe = True

    def __init__(self, config):
        self.name = config["model"]["name"]
        self.num_layers = int(config["model"]["num_layers"])
        self.hidden_size = int(config["model"]["hidden_size"])
        self.sequence_len = int(config["model"]["sequence_len"])
        self.vocab_size = int(config["model"]["vocab_size"])
        self.batch_size = int(config["model"]["batch_size"])
        self.bytes_per_val = int(config["model"]["bytes_per_val"])
        self.tp_size = int(config["parallelism"]["tp_size"])
        self.pp_size = int(config["parallelism"]["pp_size"])
        self.dp_size = int(config["parallelism"]["dp_size"])
        self.scale = float(config["model"]["scale"])

        moe = config.get("moe", {}) or {}
        self.num_experts = int(moe.get("num_experts", 1))
        self.ep_size = int(moe.get("ep_size", 1))
        self.top_k = int(moe.get("top_k", 1))
        self.capacity_factor = float(moe.get("capacity_factor", 1.25))
        self.a2a_mode = str(moe.get("a2a_mode", "pairwise"))
        if self.a2a_mode not in ("pairwise", "collective"):
            raise ValueError(f"moe.a2a_mode must be 'pairwise' or 'collective' (got {self.a2a_mode})")

        if self.dp_size % self.ep_size != 0:
            raise ValueError(
                f"dp_size ({self.dp_size}) must be divisible by moe.ep_size ({self.ep_size})")
        self.edp_size = self.dp_size // self.ep_size

        self.router = MoeRouter(
            num_experts=self.num_experts,
            ep_size=self.ep_size,
            top_k=self.top_k,
            capacity_factor=self.capacity_factor,
            distribution=moe.get("distribution"),
            placement=moe.get("placement"),
            seed=int(moe.get("seed", 0)),
            resample=str(moe.get("resample", "per_microbatch")),
            drop_tokens=bool(moe.get("drop_tokens", True)),
        )

        # attention/embeddings replicated on DP; expert FFN params sharded on EP
        h, L = self.hidden_size, self.num_layers
        attn_params = 4 * L * h * h
        embed_misc_params = (13 + self.vocab_size + self.sequence_len) * h
        self.dense_params = attn_params + embed_misc_params
        self.expert_params = 8 * L * h * h * self.num_experts
        self.num_params = self.dense_params + self.expert_params

        self.layers = [
            TransformerMoeLayer(
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                sequence_len=self.sequence_len,
                vocab_size=self.vocab_size,
                ep_size=self.ep_size,
                tp_size=self.tp_size,
                num_experts=self.num_experts,
                router=self.router,
                layer_index=i,
                top_k=self.top_k,
                capacity_factor=self.capacity_factor,
                bytes_per_val=self.bytes_per_val,
                scale=self.scale,
                a2a_mode=self.a2a_mode,
            )
            for i in range(self.num_layers)
        ]

    def _ep_context(self, npu_id: int) -> dict:
        pp_tp = self.pp_size * self.tp_size
        dp_group = npu_id // pp_tp
        rem = npu_id % pp_tp
        pp_stage = rem // self.tp_size
        tp_shard = rem % self.tp_size

        ep_block = dp_group // self.ep_size
        ep_local = dp_group % self.ep_size

        def global_id(dpg: int) -> int:
            return dpg * pp_tp + pp_stage * self.tp_size + tp_shard

        ep_members = [global_id(ep_block * self.ep_size + r) for r in range(self.ep_size)]
        edp_members = [global_id(g * self.ep_size + ep_local) for g in range(self.edp_size)]
        return {
            "ep_block": ep_block,
            "ep_local": ep_local,
            "ep_members": ep_members,
            "edp_members": edp_members,
            "ep_pg_name": f"ep_{pp_stage}_{tp_shard}_{ep_block}",
        }

    @staticmethod
    def _microbatch_from_name(name: str) -> int:
        m = re.search(r"_b(\d+)", name or "")
        return int(m.group(1)) if m else 0

    def get_edp_members(self, npu_id: int) -> list[int]:
        return self._ep_context(npu_id)["edp_members"]

    def fwd(self, name, npu_id, layer, num_batches, pg_name=None) -> list[ChakraNode]:
        ctx = self._ep_context(npu_id)
        mb = self._microbatch_from_name(name)
        key = self.router.group_key(layer, mb, ctx["ep_block"])
        return self.layers[layer].fwd(
            name=name, pg_name=pg_name, num_batches=num_batches,
            ep_local=ctx["ep_local"], ep_member_ids=ctx["ep_members"],
            my_global=npu_id, key=key, microbatch=mb, ep_pg_name=ctx["ep_pg_name"])

    def bckwd(self, name, npu_id, layer, num_batches, pg_name=None) -> list[ChakraNode]:
        ctx = self._ep_context(npu_id)
        mb = self._microbatch_from_name(name)
        key = self.router.group_key(layer, mb, ctx["ep_block"])
        return self.layers[layer].bckwd(
            name=name, pg_name=pg_name, num_batches=num_batches,
            ep_local=ctx["ep_local"], ep_member_ids=ctx["ep_members"],
            my_global=npu_id, key=key, microbatch=mb, ep_pg_name=ctx["ep_pg_name"])

    def get_num_params(self) -> int:
        return self.num_params

    def get_dense_params(self) -> int:
        return self.dense_params

    def get_expert_params(self) -> int:
        return self.expert_params

    def get_num_layers(self) -> int:
        return self.num_layers

    def get_name(self) -> str:
        return self.name

    def get_hidden_size(self) -> int:
        return self.hidden_size

    def get_sequence_len(self) -> int:
        return self.sequence_len

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_bytes_per_val(self) -> int:
        return self.bytes_per_val

    def get_tp_size(self) -> int:
        return self.tp_size

    def get_ep_size(self) -> int:
        return self.ep_size

    def get_edp_size(self) -> int:
        return self.edp_size

    def get_scale(self) -> float:
        return self.scale

    def get_layers(self) -> list[TransformerMoeLayer]:
        return self.layers
