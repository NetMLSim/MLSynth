---
name: text-to-chakra-trace
description: Turn a described ML training workload into Chakra execution traces with MLSynth — translate the request into a config, check it, generate, and verify. Use when the user describes a transformer training workload (a data/pipeline/tensor parallelism split, layer count, batch or microbatch sizes) and wants traces, or asks to write, fix, or explain an MLSynth input.yaml, why a config was rejected, or how many ranks or collectives a workload produces.
---

# Text to Chakra trace

MLSynth takes a config, not prose. Translate the request, check the rules,
generate, then report. Dense `transformer` only — the MoE files in the repo are
not wired into `synthesise_workload.py`.

## Config

Every key is required; MLSynth raises `KeyError` on a missing one. Use these
values for anything the user did not specify, and add nothing else — there is no
iteration or step count.

```yaml
model:
  name: transformer
  num_layers: 2
  sequence_len: 1024
  vocab_size: 32768
  hidden_size: 1024
  batch_size: 16          # GLOBAL batch, split across data-parallel ranks
  num_microbatches: 1
  bytes_per_val: 2        # 2 for fp16/bf16, 4 for fp32
  scale: 0.01             # scales compute and message sizes; 1 = full size
parallelism:
  dp_size: 2
  pp_size: 1
  tp_size: 1
```

"Global batch" is `batch_size`; "seq len" or "context" is `sequence_len`; "hidden
dim" or "d_model" is `hidden_size`. Given only a GPU count, put it all in
`dp_size` unless pipeline or tensor parallelism was mentioned.

## Rules

Check these before generating and report all failures at once, each with its fix.

- `batch_size` must be a positive multiple of `dp_size`. It is the **global**
  batch, and that one fact causes most rejections.
- `num_microbatches <= batch_size // dp_size`, so the smallest legal batch is
  `dp_size × num_microbatches`.
- `num_layers % pp_size == 0`, else layers cannot split evenly across stages.
- Every value positive, and an integer except `name` and `scale`.

Per-rank compute and pipeline message sizes use the local batch `batch_size //
dp_size`; the output directory name still records the global `batch_size`.

## What to report

Neither number is in the file, so state both:

- **Ranks** = `dp_size × pp_size × tp_size`.
- **Collectives per rank** = `4 × (num_layers // pp_size) × num_microbatches`
  when `tp_size > 1`, plus 1 when `dp_size > 1` for the gradient all-reduce. At
  `dp 1, tp 1` there are none at all. Pipeline stages are send and receive nodes,
  not collectives. At `pp 1, tp 1` the gradient all-reduce carries no `pg_name`,
  and `comm_groups.json` keys that group under `""`.

## Generating

Run it from the directory where the traces should land — `output/` is relative to
the working directory. If `chakra` is missing, see the install note in AGENTS.md.

```bash
python3 synthesise_workload.py -c input.yaml
```

The name is derived, not chosen, so compute it rather than globbing for it
(`scale: 0.01` renders `1scale`):

```
output/<name>_<dp>dp_<pp>pp_<tp>tp_<batch>B_<seq>S_<vocab>V_<hidden>d_<bytes>b_<scale×100>scale/
    et/<derived-name>.<rank>.et     one Chakra trace per rank, 0-indexed
    comm_groups.json                pg_name → member ranks
```

Confirm one `.et` per rank.

## Compute wrapper

Only for straggler or slowdown requests. The presence of `wrapper` activates it;
`type` is never read and `seed` is required.

```yaml
wrapper:
  type: compute
  seed: 42
  conditions:
    - npu_id: 0
      pass: forward
      slowdown: {type: constant, value: 0.1}
    - npu_id_range: [1, 2]
      layer_id_range: [4, 5]
      slowdown: {type: random, mean: 0.2, std: 0.1}
```

- Indices are zero-based and ranges **inclusive**, so `[1, 2]` is NPUs 1 and 2.
  `layer_id` counts globally across pipeline stages, not within one.
- The first matching condition wins; one with no `npu_id`/`layer_id` matches all.
- Omitting `pass` slows **both** passes; otherwise it is exactly `forward` or
  `backward`.
- `slowdown.type` is `constant` with `value` or `random` with `mean` and `std`.
  Any other type crashes, and a slowdown `<= 0` is a no-op.
