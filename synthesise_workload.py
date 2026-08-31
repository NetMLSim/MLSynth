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
import copy

# Ensure local imports work without modifying system-wide settings
#PROJECT_ROOT = Path(__file__).resolve().parents[1]
#sys.path.insert(0, str(PROJECT_ROOT))

from Wrapper.ComputeWrapper import ComputeWrapper
from Orchestrator.MegatronLM import MegatronLM
from Model.Transformer import Transformer
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
import yaml


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
    if cfg["model"]["batch_size"] % cfg["parallelism"]["dp_size"] != 0:
        raise ValueError(f"global batch_size={cfg['model']['batch_size']} must divide evenly across dp_size={cfg['parallelism']['dp_size']}!")
    batch_size = cfg["model"]["batch_size"] // cfg["parallelism"]["dp_size"]

    if batch_size < cfg["model"]["num_microbatches"]:
        raise ValueError(f"num batches (batch_size={cfg['model']['batch_size']}) must be greater than num microbatches (num_microbatches={cfg['model']['num_microbatches']})!")
    if cfg["model"]["num_layers"] % cfg["parallelism"]["pp_size"] != 0:
        raise ValueError(f"num_layers={cfg['model']['num_layers']} must divide evenly across pp_size={cfg['parallelism']['pp_size']}!")


def build_output_name(cfg):
    model = cfg["model"]
    parallelism = cfg["parallelism"]
    scale = int(model["scale"] * 100)
    return (
        f'{model["name"]}_{parallelism["dp_size"]}dp_'
        f'{parallelism["pp_size"]}pp_{parallelism["tp_size"]}tp_'
        f'{model["batch_size"]}B_{model["sequence_len"]}S_'
        f'{model["vocab_size"]}V_{model["hidden_size"]}d_'
        f'{model["bytes_per_val"]}b_{scale}scale'
    )


def model_config_for_rank(cfg):
    model_cfg = copy.deepcopy(cfg)
    model_cfg["model"]["batch_size"] = (
        cfg["model"]["batch_size"] // cfg["parallelism"]["dp_size"]
    )
    return model_cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize workload from YAML config")
    parser.add_argument("-c", "--config", default="input.yaml", help="Path to input YAML config file")
    args = parser.parse_args(argv)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    validate_config(cfg)

    name = build_output_name(cfg)
    os.makedirs(f"output", exist_ok=True)
    # make name directory
    os.makedirs(f"output/{name}", exist_ok=True)
    os.makedirs(f"output/{name}/et", exist_ok=True)

    model = Transformer(model_config_for_rank(cfg))
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
