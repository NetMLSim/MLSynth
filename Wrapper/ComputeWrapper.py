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


class ComputeWrapper(Wrapper):
    """A wrapper that inserts compute slowdown nodes into the compute graph.
       Currently incomplete."""
    def __init__(self, model, config):
        self.model = model
        # layer_id -> compute variation
        self.wrap_layers = {}

    def fwd(self, name, layer, num_batches) -> list[ChakraNode]:
        ops = self.model.fwd(name, layer, num_batches)
        if layer in self.wrap_layers:
            ops = self.insert_slowdown(ops, layer)
        return ops

    def bckwd(self, name, layer, num_batches) -> list[ChakraNode]:
        ops = self.model.bckwd(name, layer, num_batches)
        if layer in self.wrap_layers:
            ops = self.insert_slowdown(ops, layer)
        return ops

    def insert_slowdown(self, ops: list[ChakraNode], layer: int) -> list[ChakraNode]:
        # Iterate in reverse order to avoid index issues when inserting elements
        for i in range(len(ops) - 1, -1, -1):
            op = ops[i]
            if op.type == ChakraNodeType.COMP_NODE:
                compute_node = compute(op.num_ops * self.wrap_layers[layer], op.tensor_size, parents=[op], name=f"{op.name}_slowdown")
                # Only update dependencies if there's a next element
                if i + 1 < len(ops):
                    ops[i+1].data_deps.remove(op.id)
                    ops[i+1].data_deps.append(compute_node.id)
                ops.insert(i+1, compute_node)
        return ops