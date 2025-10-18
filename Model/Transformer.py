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
from Layer.TransformerLayer import TransformerLayer
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    NodeType as ChakraNodeType,
    AttributeProto as ChakraAttr
)


class Transformer(Model):
    """Implementation of the Transformer model."""
    
    def __init__(self, config):
        self.name = config["model"]["name"]
        self.num_layers = int(config["model"]["num_layers"])
        self.hidden_size = int(config["model"]["hidden_size"])
        self.sequence_len = int(config["model"]["sequence_len"])
        self.vocab_size = int(config["model"]["vocab_size"])
        self.batch_size = int(config["model"]["batch_size"])
        self.bytes_per_val = int(config["model"]["bytes_per_val"])
        self.tp_size = int(config["parallelism"]["tp_size"])
        self.scale = float(config["model"]["scale"])
        self.num_params = 12 * self.num_layers * self.hidden_size * self.hidden_size * (1 + ((13)/(12*self.num_layers*self.hidden_size)) + ((self.vocab_size + self.sequence_len)/(12*self.num_layers*self.hidden_size)))

        # Model is composed of Transformer layers
        self.layers = []

        [self.layers.append(
            TransformerLayer(
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                sequence_len=self.sequence_len,
                vocab_size=self.vocab_size,
                tp_size=self.tp_size,
                bytes_per_val=self.bytes_per_val,
                scale=self.scale)) for i in range(self.num_layers)]
    
    def fwd(self, name, npu_id, layer, num_batches, pg_name=None) -> list[ChakraNode]:
        return self.layers[layer].fwd(name=name, num_batches=num_batches, pg_name=pg_name)
    
    def bckwd(self, name, npu_id, layer, num_batches, pg_name=None) -> list[ChakraNode]:
        return self.layers[layer].bckwd(name=name, num_batches=num_batches, pg_name=pg_name)
    
    def get_num_params(self) -> int:
        return self.num_params
    
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

    def get_scale(self) -> float:
        return self.scale

    def get_layers(self) -> list[TransformerLayer]:
        return self.layers