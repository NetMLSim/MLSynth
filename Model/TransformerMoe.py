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

from Model.Model import Model
from Layer.TransformerMoeLayer import TransformerMoeLayer
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    NodeType as ChakraNodeType,
    AttributeProto as ChakraAttr
)


class TransformerMoe(Model):
    """A concrete implementation of the Model interface."""
    
    def __init__(self, 
        num_layers: int, 
        hidden_size: int, 
        sequence_len: int, 
        vocab_size: int, 
        batch_size: int, 
        bytes_per_val: int,
        ep_size: int,
        tp_size: int, 
        scale: float = 1,
        name="transformer"):
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.bytes_per_val = bytes_per_val
        self.ep_size = ep_size
        self.tp_size = tp_size
        self.scale = scale
        self.num_params = 12 * num_layers * hidden_size * hidden_size * (1 + ((13)/(12*num_layers*hidden_size)) + ((vocab_size + sequence_len)/(12*num_layers*hidden_size)))

        # Model is composed of layers
        self.layers = []

        [self.layers.append(
            TransformerMoeLayer(
                num_layers=num_layers,
                hidden_size=hidden_size,
                sequence_len=sequence_len,
                vocab_size=vocab_size,
                ep_size=ep_size,
                tp_size=tp_size,
                bytes_per_val=bytes_per_val,
                scale=scale)) for i in range(num_layers)]
    
    #def get_next_layer(self) -> list[ChakraNode]:
    #    return self.layers[0]
    
    def fwd(self, name, layer, num_batches) -> list[ChakraNode]:
        return self.layers[layer].fwd(name=name, num_batches=num_batches)
    
    def bckwd(self, name, layer, num_batches) -> list[ChakraNode]:
        return self.layers[layer].bckwd(name=name, num_batches=num_batches)
