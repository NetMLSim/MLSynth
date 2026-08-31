# AGENTS.md — MLSynth

MLSynth synthesises Chakra execution traces for ML training workloads. A YAML
config describes a model shape and a 3D parallelism split; `synthesise_workload.py`
emits one `.et` trace per rank plus the process groups those traces reference.
Generation is deterministic: the same config produces the same traces.

Follow `.agents/skills/text-to-chakra-trace/SKILL.md` whenever the task involves
writing, fixing or explaining a config, or reporting what a workload generates.

## Layout

- `synthesise_workload.py` — entry point, `-c/--config` (default `input.yaml`).
  Writes `output/<derived-name>/` relative to the **working directory**.
- `Model/`, `Layer/` — model and layer shapes; emit compute and collective nodes.
- `Orchestrator/MegatronLM.py` — the training step: pipeline schedule, microbatch
  loop, gradient all-reduce.
- `Wrapper/` — optional overlays that perturb a workload (stragglers).
- `utils.py` — Chakra node constructors.
- Dense `transformer` is the only model wired up; the MoE files are scaffolding.

## Conventions

- Package directories are `PascalCase` and match their class; code is `snake_case`.
- Add a layer, model or orchestrator by subclassing the base in its directory.
- Node names are load-bearing. Tensor-parallel all-reduces inherit their compute
  node's name, so find collectives by node type, never by name.
- Sizes are bytes: `int(...)` of a float expression scaled by `scale`. Keep the
  truncation explicit rather than rounding.

## Running it

There is no test suite; a generated trace is the check. Generation needs
`chakra`, `protobuf`, `PyYAML` and `numpy`. Do not `pip install chakra`, which is
an unrelated Python 2 package on PyPI; install from source with `--no-deps` to
skip its unused HolisticTraceAnalysis dependency:

```bash
pip install --no-deps "git+https://github.com/mlcommons/chakra.git"
pip install protobuf PyYAML numpy
python3 synthesise_workload.py -c input.yaml
```
