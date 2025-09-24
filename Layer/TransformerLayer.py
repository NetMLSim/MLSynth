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
from utils import allreduce, compute
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)


class TransformerLayer(Layer):
    """An implementation of a Transformer Layer."""
    
    def __init__(self, 
        num_layers: int,
        hidden_size: int,
        sequence_len: int,
        vocab_size: int,
        tp_size: int,
        bytes_per_val=2,
        scale=1):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.bytes_per_val = bytes_per_val
        self.tp_size = tp_size
        self.scale = scale
    
    def fwd(self, name="node_fwd", pg_name=None, num_batches=1) -> list[ChakraNode]:
        tensor_size = int((12*self.hidden_size*self.hidden_size*self.bytes_per_val + num_batches*self.sequence_len*self.hidden_size*self.bytes_per_val) * self.scale)

        # calculate flops for the attention block
        attention_flops = int(self.scale * (8 * num_batches*self.sequence_len*self.hidden_size*self.hidden_size + 4*num_batches*self.sequence_len*self.sequence_len*self.hidden_size))
        attention_compute = compute(attention_flops, tensor_size, name=f"{name}_attention_compute")
        
        # tensor parallel allreduce
        attention_allreduce = None
        if self.tp_size > 1:
            tp_comm_size = int(self.scale * self.bytes_per_val * self.sequence_len * num_batches * self.hidden_size)
            attention_allreduce = allreduce(tp_comm_size, pg_name=pg_name, parents=[attention_compute], name=f"{name}_attention_allreduce")
        
        # calculate flops for the mlp block
        ffwd_flops = int(self.scale * 16 * num_batches*self.sequence_len*self.hidden_size*self.hidden_size)
        ffwd_parent = [attention_allreduce] if attention_allreduce else [attention_compute]
        ffwd_compute = compute(ffwd_flops, tensor_size, parents=ffwd_parent, name=f"{name}_ffwd_compute")        
        
        # tensor parallel allreduce
        ffwd_allreduce = None
        if self.tp_size > 1:
            tp_comm_size = int(self.scale * self.bytes_per_val * self.sequence_len * num_batches * self.hidden_size)
            ffwd_allreduce = allreduce(tp_comm_size, pg_name=pg_name, parents=[ffwd_compute], name=f"{name}_mlp_allreduce")

        nodes: list[ChakraNode] = [attention_compute]
        if attention_allreduce is not None:
            nodes.append(attention_allreduce)
        nodes.append(ffwd_compute)
        if ffwd_allreduce is not None:
            nodes.append(ffwd_allreduce)
        return nodes


    def bckwd(self, name="node_bckwd", pg_name=None, num_batches=1) -> list[ChakraNode]:
        tensor_size = int((12*self.hidden_size*self.hidden_size*self.bytes_per_val + num_batches*self.sequence_len*self.hidden_size*self.bytes_per_val) * self.scale)

        # calculate flops for the attention block
        attention_flops = int(self.scale * (8 * num_batches*self.sequence_len*self.hidden_size*self.hidden_size + 4*num_batches*self.sequence_len*self.sequence_len*self.hidden_size))
        attention_compute = compute(2 * attention_flops, tensor_size=tensor_size, name=f"{name}_attention_compute")
        
        # tensor parallel allreduce
        attention_allreduce = None
        if self.tp_size > 1:
            tp_comm_size = int(self.scale * self.bytes_per_val * self.sequence_len * num_batches * self.hidden_size)
            attention_allreduce = allreduce(tp_comm_size, pg_name=pg_name, parents=[attention_compute], name=f"{name}_attention_allreduce")
        
        # calculate flops for the mlp block
        ffwd_flops = int(self.scale * 16 * num_batches*self.sequence_len*self.hidden_size*self.hidden_size)
        ffwd_parent = [attention_allreduce] if attention_allreduce else [attention_compute]
        ffwd_compute = compute(2 * ffwd_flops, tensor_size, parents=ffwd_parent, name=f"{name}_ffwd_compute")

        # tensor parallel allreduce
        ffwd_allreduce = None
        if self.tp_size > 1:
            tp_comm_size = int(self.scale * self.bytes_per_val * self.sequence_len * num_batches * self.hidden_size)
            ffwd_allreduce = allreduce(tp_comm_size, pg_name=pg_name, parents=[ffwd_compute], name=f"{name}_mlp_allreduce")

        nodes: list[ChakraNode] = [ffwd_compute]
        if ffwd_allreduce is not None:
            nodes.append(ffwd_allreduce)
        nodes.append(attention_compute)
        if attention_allreduce is not None:
            nodes.append(attention_allreduce)
        return nodes