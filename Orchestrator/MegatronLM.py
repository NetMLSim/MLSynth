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

from collections import defaultdict
from Model.Model import Model
from Orchestrator.Orchestrator import Orchestrator
from utils import add_dependencies, allreduce, receive, send
from chakra.schema.protobuf.et_def_pb2 import (
    GlobalMetadata,
)


class MegatronLM(Orchestrator):
    def __init__(self, 
        model: Model,
        config):
        self.model = model
        self.dp_size = config["parallelism"]["dp_size"]
        self.pp_size = config["parallelism"]["pp_size"]
        self.tp_size = config["parallelism"]["tp_size"]
        self.num_npus = self.dp_size * self.pp_size * self.tp_size
        self.num_microbatches = config["model"]["num_microbatches"]
        self.scale = config["model"]["scale"]

    def generate_comm_groups(self):
        comm_groups = defaultdict(list)

        # generate comm groups for data parallel groups        
        for dp_group in range(self.dp_size):
            for pp_stage in range(self.pp_size):
                for tp_shard in range(self.tp_size):
                    # dp all-reduce group consists of all ranks that share the same pipeline stage and tensor parallel shard
                    npu_id = dp_group * (self.pp_size * self.tp_size) + (pp_stage * self.tp_size) + tp_shard
                    pp_name = "" if self.pp_size <= 1 else f"pp_{pp_stage}"
                    tp_name = "" if self.tp_size <= 1 else f"_tp_{tp_shard}"
                    comm_groups[f"{pp_name}{tp_name}"].append(npu_id)
        
        # generate comm groups for each tensor parallel group
        if self.tp_size > 1:
            for tp_group in range(self.num_npus // self.tp_size):
                tp_comm_group = []
                base = tp_group * self.tp_size
                for npu in range(self.tp_size):
                    npu_id = base + npu
                    tp_comm_group.append(npu_id)
                comm_groups[f"tp_{tp_group}"] = tp_comm_group
        return comm_groups

    def exec(self) -> dict:
        num_params = self.model.num_params
        B = self.model.get_batch_size()
        S = self.model.get_sequence_len()
        d = self.model.get_hidden_size()
        b = self.model.get_bytes_per_val()

        layers_per_pipeline_stage = self.model.get_num_layers() // self.pp_size
        pp_comm_size = int((B*S*d*b * self.scale) / self.num_microbatches)
        dp_comm_size = int(self.scale * num_params * b / self.tp_size / self.pp_size)

        # print(f"Num params: {num_params:,.2f}")
        # print(f"Pipeline comm size: {pp_comm_size / 1024 / 1024:,.2f} MB")
        # print(f"DP comm size: {dp_comm_size / 1024 / 1024 / 1024:,.2f} GB")

        nodes = defaultdict(list)

        for dp_group in range(self.dp_size):
            #print(f"------------ DP GROUP {dp_group} ------------")
            for pp_stage in range(self.pp_size):
                for tp_shard in range(self.tp_size):
                    npu_id = dp_group * (self.pp_size * self.tp_size) + (pp_stage * self.tp_size) + tp_shard
                    tp_group = npu_id // self.tp_size
                    nodes[npu_id].append(GlobalMetadata(version="0.0.4"))
                    # print(f"NPU {npu_id} - dp group: {dp_group}, pp stage: {pp_stage}, tp shard: {tp_shard}")
                    # -------------
                    # Forward pass
                    # -------------
                    prev_rcv = None
                    prev_comp = None
                    for b in range(self.num_microbatches):
                        rcv_node = None
                        if pp_stage != 0 and self.pp_size > 1:
                            #print(f"RCV ({npu_id - tp_size} -> {npu_id})")
                            rcv_node = receive(npu_id - self.tp_size, npu_id, pp_comm_size, parents=[prev_rcv], name=f"COMM_RECV_NODE_FWD_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                            nodes[npu_id].append(rcv_node)
                            prev_rcv = rcv_node
                        
                        for layer in range(layers_per_pipeline_stage):
                            current_layer = pp_stage * layers_per_pipeline_stage + layer
                            cmp_nodes = self.model.fwd(name=f"COMP_NODE_FWD_b{b}", npu_id=npu_id, layer=current_layer, num_batches=B/self.num_microbatches, pg_name=f"tp_{tp_group}")
                            if layer == 0:
                                add_dependencies(cmp_nodes[0], [rcv_node, prev_comp])
                            else:
                                add_dependencies(cmp_nodes[0], [prev_comp])
                            for node in cmp_nodes:
                                nodes[npu_id].append(node)
                            prev_comp = cmp_nodes[-1]
                        
                        if pp_stage != self.pp_size - 1 and self.pp_size > 1:
                            #print(f"SND ({npu_id} -> {npu_id + tp_size})")
                            snd_node = send(npu_id, npu_id + self.tp_size, pp_comm_size, parents=[prev_comp], name=f"COMM_SEND_NODE_FWD_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                            nodes[npu_id].append(snd_node)                
                    # -------------
                    # Backward pass
                    # -------------
                    #print("Backward pass")
                    for b in range(self.num_microbatches):
                        bck_rcv_node = None
                        if pp_stage != self.pp_size - 1 and self.pp_size > 1:
                            #print(f"RCV ({npu_id + tp_size} -> {npu_id})")
                            bck_rcv_node = receive(npu_id + self.tp_size, npu_id, pp_comm_size, parents=[prev_rcv], name=f"COMM_RECV_NODE_BCKWD_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                            nodes[npu_id].append(bck_rcv_node)
                            prev_rcv = bck_rcv_node
                        
                        for layer in range(layers_per_pipeline_stage):
                            current_layer = pp_stage * layers_per_pipeline_stage + layer
                            bck_cmp_nodes = self.model.bckwd(name=f"COMP_NODE_BCKWD_b{b}", npu_id=npu_id, layer=current_layer, num_batches=B/self.num_microbatches, pg_name=f"tp_{tp_group}")
                            if layer == 0:
                                add_dependencies(bck_cmp_nodes[0], [bck_rcv_node, prev_comp])
                            else:
                                add_dependencies(bck_cmp_nodes[0], [prev_comp])
                            for node in bck_cmp_nodes:
                                nodes[npu_id].append(node)
                            prev_comp = bck_cmp_nodes[-1]
                        
                        
                        if pp_stage != 0 and self.pp_size > 1:
                            #print(f"SND ({npu_id} -> {npu_id - tp_size})")
                            bck_snd_node = send(npu_id, npu_id - self.tp_size, pp_comm_size, parents=[prev_comp], name=f"COMM_SEND_NODE_BCKWD_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                            nodes[npu_id].append(bck_snd_node)

                    if self.dp_size > 1:
                        pp_name = "" if self.pp_size <= 1 else f"pp_{pp_stage}"
                        tp_name = "" if self.tp_size <= 1 else f"_tp_{tp_shard}"
                        dp_comm_node = allreduce(dp_comm_size, parents=[prev_comp], pg_name=f"{pp_name}{tp_name}", name=f"COMM_COLL_NODE_DP_All-Reduce_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                        nodes[npu_id].append(dp_comm_node)
        return nodes
