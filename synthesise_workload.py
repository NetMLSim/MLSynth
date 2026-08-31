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

import json
import os
import sys
from pathlib import Path
import argparse

# Ensure local imports work without modifying system-wide settings
#PROJECT_ROOT = Path(__file__).resolve().parents[1]
#sys.path.insert(0, str(PROJECT_ROOT))

from Wrapper.ComputeWrapper import ComputeWrapper
from Orchestrator.MegatronLM import MegatronLM
from Model.Transformer import Transformer
from Model.TransformerMoe import TransformerMoe
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
import yaml


MOE_MODEL_NAMES = {"transformer_moe", "moe", "transformermoe"}


def is_moe_config(cfg) -> bool:
    """True if the config selects a MoE model."""
    if "moe" in cfg and cfg["moe"]:
        return True
    return str(cfg["model"].get("name", "")).lower() in MOE_MODEL_NAMES


def write_comm_groups(comm_groups, path=""):
    with open(path + "comm_groups.json", "w") as f:
        json.dump(comm_groups, f, indent=2, sort_keys=True)

def write_nodes(nodes, name, path=""):
    for npu_id in nodes.keys():
        with open(path + f"{name}.{npu_id}.et", "wb") as et:
            for node in nodes[npu_id]:
                encode_message(et, node)

def validate_config(cfg):
    if cfg["model"]["batch_size"] < cfg["parallelism"]["dp_size"]:
        raise ValueError(f"num batches (batch_size={cfg['model']['batch_size']}) must be greater than num dp groups (dp_size={cfg['parallelism']['dp_size']})!")
    batch_size = cfg["model"]["batch_size"] // cfg["parallelism"]["dp_size"]

    if batch_size < cfg["model"]["num_microbatches"]:
        raise ValueError(f"num batches (batch_size={cfg['model']['batch_size']}) must be greater than num microbatches (num_microbatches={cfg['model']['num_microbatches']})!")

    if is_moe_config(cfg):
        moe = cfg.get("moe", {}) or {}
        ep_size = int(moe.get("ep_size", 1))
        num_experts = int(moe.get("num_experts", 1))
        top_k = int(moe.get("top_k", 1))
        dp_size = cfg["parallelism"]["dp_size"]
        if ep_size < 1:
            raise ValueError(f"moe.ep_size must be >= 1 (got {ep_size})")
        if dp_size % ep_size != 0:
            raise ValueError(f"dp_size ({dp_size}) must be divisible by moe.ep_size ({ep_size})!")
        if top_k < 1:
            raise ValueError(f"moe.top_k must be >= 1 (got {top_k})")
        strategy = str((moe.get("placement", {}) or {}).get("strategy", "contiguous"))
        if strategy != "custom" and num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by moe.ep_size ({ep_size}) "
                f"for placement strategy '{strategy}'; use placement.strategy='custom' otherwise")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize workload from YAML config")
    parser.add_argument("-c", "--config", default="input.yaml", help="Path to input YAML config file")
    args = parser.parse_args(argv)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    validate_config(cfg)

    name = f"{cfg["model"]["name"]}_{cfg["parallelism"]["dp_size"]}dp_{cfg["parallelism"]["pp_size"]}pp_{cfg["parallelism"]["tp_size"]}tp_{cfg["model"]["batch_size"]}B_{cfg["model"]["sequence_len"]}S_{cfg["model"]["vocab_size"]}V_{cfg["model"]["hidden_size"]}d_{cfg["model"]["bytes_per_val"]}b_{int(cfg["model"]["scale"]*100)}scale"
    os.makedirs(f"output", exist_ok=True)
    # make name directory
    os.makedirs(f"output/{name}", exist_ok=True)
    os.makedirs(f"output/{name}/et", exist_ok=True)

    model = TransformerMoe(cfg) if is_moe_config(cfg) else Transformer(cfg)
    if "wrapper" in cfg:
        model = ComputeWrapper(model, cfg)
    orchestrator = MegatronLM(model, cfg)

    comm_groups = orchestrator.generate_comm_groups()
    write_comm_groups(comm_groups, path=f"output/{name}/")

    nodes = orchestrator.exec()
    write_nodes(nodes, name, path=f"output/{name}/et/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())