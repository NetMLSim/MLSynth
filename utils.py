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

from typing import List, Optional
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    NodeType as ChakraNodeType,
    AttributeProto as ChakraAttr,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMM_COLL_NODE,
    ALL_REDUCE,
    ALL_TO_ALL,
    COMP_NODE
)

node_id = 0
def next_id() -> int:
    global node_id
    node_id += 1
    return node_id - 1

def get_node(node_name: str, node_type: ChakraNodeType) -> ChakraNode:
    """Generate a new ChakraNode with a unique ID."""
    node = ChakraNode()
    node.id = next_id()
    node.name = node_name
    node.type = node_type
    return node

def compute(flops: int, tensor_size: int, parents: Optional[List[ChakraNode]] = None, name: str = "COMP_NODE") -> ChakraNode:
    node = get_node(name, COMP_NODE)
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(ChakraAttr(name="num_ops", int64_val=flops))
    node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
    if parents:
        for parent in parents:
            if parent:
                node.data_deps.append(parent.id)
    return node

def send(sender, receiver, size, name="COMM_SEND_NODE", parents: Optional[List[ChakraNode]] = None):    
    node = get_node(name, COMM_SEND_NODE)
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(ChakraAttr(name="comm_size", int64_val=size))
    node.attr.append(ChakraAttr(name="comm_src", int32_val=sender))
    node.attr.append(ChakraAttr(name="comm_dst", int32_val=receiver))
    node.attr.append(ChakraAttr(name="comm_tag", int32_val=0))
    if parents:
        for parent in parents:
            if parent:
                node.data_deps.append(parent.id)
    return node

def receive(sender, receiver, size, name="COMM_RECV_NODE", parents: Optional[List[ChakraNode]] = None):    
    node = get_node(name, COMM_RECV_NODE)
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(ChakraAttr(name="comm_size", int64_val=size))
    node.attr.append(ChakraAttr(name="comm_src", int32_val=sender))
    node.attr.append(ChakraAttr(name="comm_dst", int32_val=receiver))
    node.attr.append(ChakraAttr(name="comm_tag", int32_val=0))
    if parents:
        for parent in parents:
            if parent:
                node.data_deps.append(parent.id)
    return node

def allreduce(coll_size: int, pg_name: Optional[str] = None, name: str = "COMM_COLL_NODE_All-Reduce", parents: Optional[List[ChakraNode]] = None) -> ChakraNode:
    # create Chakra Node    
    node = get_node(name, COMM_COLL_NODE)

    # assign attributes
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(ChakraAttr(name="comm_type", int64_val=ALL_REDUCE))
    node.attr.append(ChakraAttr(name="comm_size", int64_val=coll_size))
    if pg_name:
        node.attr.append(ChakraAttr(name="pg_name", string_val=pg_name))
    if parents:
        for parent in parents:
            if parent:
                node.data_deps.append(parent.id)
    return node

def alltoall(coll_size: int, pg_name: Optional[str] = None, name: str = "COMM_COLL_NODE_All-To-All", parents: Optional[List[ChakraNode]] = None) -> ChakraNode:
    node = get_node(name, COMM_COLL_NODE)
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(ChakraAttr(name="comm_type", int64_val=ALL_TO_ALL))
    node.attr.append(ChakraAttr(name="comm_size", int64_val=coll_size))
    if pg_name:
        node.attr.append(ChakraAttr(name="pg_name", string_val=pg_name))
    if parents:
        for parent in parents:
            if parent:
                node.data_deps.append(parent.id)
    return node

def add_dependencies(child: ChakraNode, parents: List[Optional[ChakraNode]]) -> None:
    for parent in parents:
        if parent:
            child.data_deps.append(parent.id)
    return