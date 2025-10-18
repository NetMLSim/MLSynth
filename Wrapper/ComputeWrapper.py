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

from Wrapper.Wrapper import Wrapper
from utils import compute
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    NodeType as ChakraNodeType,
)

import numpy as np


class ComputeWrapper(Wrapper):
    """A wrapper that inserts compute slowdown nodes into the compute graph.
       Currently incomplete."""
    def __init__(self, model, config):
        self.model = model
        self.num_params = model.num_params

        self.conditions = config["wrapper"]["conditions"]
        # npu_id -> slowdown
        #self.wrapped_npus = {}
        # layer_id -> slowdown
        #self.wrapped_layers = {}

        self.rng = np.random.default_rng(config["wrapper"]["seed"])


    def fwd(self, name, npu_id, layer, num_batches, pg_name=None) -> list[ChakraNode]:
        ops = self.model.fwd(name, npu_id, layer, num_batches, pg_name)
        associated_slowdown_config = self.should_slowdown(npu_id, layer)
        if associated_slowdown_config and ("pass" not in associated_slowdown_config or associated_slowdown_config["pass"] == "forward"):
            ops = self.insert_slowdown(ops, associated_slowdown_config["slowdown"])
        return ops

    def bckwd(self, name, npu_id, layer, num_batches, pg_name=None) -> list[ChakraNode]:
        ops = self.model.bckwd(name, npu_id, layer, num_batches, pg_name)
        associated_slowdown_config = self.should_slowdown(npu_id, layer)
        if associated_slowdown_config and ("pass" not in associated_slowdown_config or associated_slowdown_config["pass"] == "backward"):
            ops = self.insert_slowdown(ops, associated_slowdown_config["slowdown"])
        return ops

    def should_slowdown(self, npu_id: int, layer: int) -> bool:
        for condition in self.conditions:
            npu_id_matches = True
            layer_id_matches = True
            if "npu_id" in condition and condition["npu_id"] != npu_id:
                npu_id_matches = False
            elif "npu_id_range" in condition and (npu_id < condition["npu_id_range"][0] or npu_id > condition["npu_id_range"][1]):
                npu_id_matches = False
            if "layer_id" in condition and condition["layer_id"] != layer:
                layer_id_matches = False
            elif "layer_id_range" in condition and (layer < condition["layer_id_range"][0] or layer > condition["layer_id_range"][1]):
                layer_id_matches = False
            if npu_id_matches and layer_id_matches:
                return condition
        return None

    def insert_slowdown(self, ops: list[ChakraNode], slowdown_config: dict) -> list[ChakraNode]:
        if slowdown_config["type"] == "constant":
            slowdown = slowdown_config["value"]
        elif slowdown_config["type"] == "random":
            slowdown = self.rng.normal(slowdown_config["mean"], slowdown_config["std"])
        
        if slowdown <= 0:
            return ops
        
        # Iterate in reverse order to avoid index issues when inserting elements
        for i in range(len(ops) - 1, -1, -1):
            op = ops[i]
            if op.type == ChakraNodeType.COMP_NODE:
                compute_node = compute(int(op.attr[1].int64_val * slowdown), int(op.attr[2].int64_val * slowdown), parents=[op], name=f"{op.name}_slowdown")
                # Only update dependencies if there's a next element
                if i + 1 < len(ops):
                    ops[i+1].data_deps.remove(op.id)
                    ops[i+1].data_deps.append(compute_node.id)
                ops.insert(i+1, compute_node)
        return ops
    
    def get_name(self) -> str:
        return self.model.get_name()
    
    def get_num_params(self) -> int:
        return self.model.get_num_params()
    
    def get_num_layers(self) -> int:
        return self.model.get_num_layers()

    def get_hidden_size(self) -> int:
        return self.model.get_hidden_size()

    def get_sequence_len(self) -> int:
        return self.model.get_sequence_len()

    def get_batch_size(self) -> int:
        return self.model.get_batch_size()

    def get_bytes_per_val(self) -> int:
        return self.model.get_bytes_per_val()