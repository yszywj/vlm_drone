# Qwen3-VL Fleet Planner LoRA

This directory implements text-only supervised fine-tuning for the production
contract:

```text
FleetMissionRequest -> FleetMissionPlan | FleetPlanPatch
```

It reuses `fleet_data` validation/evaluation and the production Fleet schemas.
It does not define a training-only planner schema and it never uses images as
Fleet Planner input. The Qwen3-VL vision tower and merger/connector stay frozen;
only reviewed language-backbone LoRA parameters may be trainable.

The committed `configs/lora/fleet_planner_lora.json` remains an inert
`placeholder`: it validates the dataset but never loads Qwen, starts training,
or creates weights. The `*.example.json` active config is a reviewed baseline,
not an automatically approved production run. Training also never changes
`configs/adapters.json` or marks an adapter active.

All model operations are local-only. Keep these settings enabled:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## A. Create and check the isolated environment

Do not install these packages into `r_isaac_sim`, the YOLO environment, or the
vLLM serving environment.

```bash
cd /home/amax/ry/vlm_drones/uav_agent
conda env create -f environment-qwen-lora.yml
conda activate qwen_lora
python scripts/check_qwen_lora_env.py --require-bf16
```

The checker verifies Python/package availability, CUDA/GPU and bf16 support,
the Transformers Qwen3-VL class, and completeness of the local checkpoint. It
does not load the model weights. The default checkpoint is:

```text
/home/amax/ry/vlm_drones/models/initial_model/Qwen3-VL-4B-Instruct
```

## B. Inspect real model modules

Run this in the dedicated environment before accepting any target list:

```bash
python training/lora/inspect_qwen_lora_targets.py \
  --model /home/amax/ry/vlm_drones/models/initial_model/Qwen3-VL-4B-Instruct \
  --language-only \
  --output /home/amax/ry/vlm_drones/outputs/lora/qwen3vl_modules.json
```

The report separates language attention, language MLP, vision, and
connector/merger `Linear` modules. Never use ambiguous suffixes such as only
`q_proj`. The active loader expands fully qualified
`model.language_model...` patterns against the actual model, requires at least
one match, rejects every vision/connector/unsupported match, injects one exact
anchored PEFT expression, and audits PEFT's resulting target names again.

## C. Review and validate an active config

```bash
cp configs/lora/fleet_planner_lora_train.example.json \
  configs/lora/fleet_planner_lora_train.json
# Review paths, inspected language targets, and every hyperparameter.
python training/lora/train_fleet_planner_lora.py \
  --config configs/lora/fleet_planner_lora_train.json \
  --validate-only
```

`--validate-only` checks the strict config, the complete Fleet dataset contract,
local model files, safe target patterns, and writable/non-overlapping output
trees. It does not allocate CUDA, inject PEFT, train, or write weights. The
actual Linear-module expansion is performed by step B and is repeated
fail-closed immediately before PEFT injection in an active run.

To exercise the original safe mode:

```bash
python training/lora/train_fleet_planner_lora.py \
  --config configs/lora/fleet_planner_lora.json
```

Its JSON result must say `training_started=false` and `weights_created=false`.

## D. Run the real one-step smoke

This is opt-in, loads the real local 4B checkpoint, and uses one visible GPU.
Do not run it while that GPU is occupied by another workload.

```bash
QWEN_LORA_CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_fleet_planner_lora_smoke.sh
```

The wrapper derives a new active config with exactly two training examples, one
validation example, batch/accumulation 1, and `max_steps=1`. It uses separate
`fleet_planner_smoke` output/adapter roots, refuses an existing run ID, preserves
`terminal.log`, and requires the final adapter config, safetensors, and both
manifests. A successful smoke proves load/forward/backward/optimizer/save wiring;
it says nothing about planner quality.

## E. Train on one GPU

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN_ID="fleet_planner_$(date -u +%Y%m%d-%H%M%SZ)"
python training/lora/train_fleet_planner_lora.py \
  --config configs/lora/fleet_planner_lora_train.json \
  --run-id "${RUN_ID}"
```

For resume, set `resume_from_checkpoint` to
`<output_dir>/<run_id>/checkpoints/checkpoint-N` in a copy of the original
config and keep all other fields identical. The run ID is inferred from that
path. Checkpoints contain the PEFT adapter plus Trainer optimizer, scheduler,
and state files; recognized full base-model weight files are rejected.

## F. Train with torchrun/DDP

Every rank must receive the same explicit run ID and config:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN_ID="fleet_planner_ddp_$(date -u +%Y%m%d-%H%M%SZ)"
torchrun --nproc_per_node=4 \
  training/lora/train_fleet_planner_lora.py \
  --config configs/lora/fleet_planner_lora_train.json \
  --run-id "${RUN_ID}"
```

Transformers/Accelerate owns optimization and DDP. Only world rank zero
publishes metrics, the final adapter, and manifests; all ranks synchronize
before consuming them. DeepSpeed is intentionally optional and not configured
by this first version.

## G. Generate deterministic Base predictions

```bash
python training/lora/generate_fleet_planner_predictions.py \
  --base-model /home/amax/ry/vlm_drones/models/initial_model/Qwen3-VL-4B-Instruct \
  --base-only \
  --dataset datasets/fleet_planner_v1 \
  --split test \
  --output /home/amax/ry/vlm_drones/outputs/lora/eval/base_test.jsonl
```

Generation uses `do_sample=false`, `num_beams=1`, and no sampling temperature.
Each diagnostic row retains raw output, parse status/error, parsed JSON, and
production-schema status/error. Invalid output is never replaced with Gold.

## H. Generate deterministic LoRA predictions

```bash
python training/lora/generate_fleet_planner_predictions.py \
  --base-model /home/amax/ry/vlm_drones/models/initial_model/Qwen3-VL-4B-Instruct \
  --adapter /home/amax/ry/vlm_drones/models/adapters/fleet_planner/<run_id> \
  --dataset datasets/fleet_planner_v1 \
  --split test \
  --output /home/amax/ry/vlm_drones/outputs/lora/eval/lora_test.jsonl
```

## I. Evaluate Gold vs Base vs LoRA

```bash
python training/lora/evaluate_fleet_planner_lora.py \
  --gold datasets/fleet_planner_v1/test.jsonl \
  --base-predictions /home/amax/ry/vlm_drones/outputs/lora/eval/base_test.jsonl \
  --lora-predictions /home/amax/ry/vlm_drones/outputs/lora/eval/lora_test.jsonl \
  --output /home/amax/ry/vlm_drones/outputs/lora/eval/base_vs_lora.json
```

The evaluator reports JSON parse, production schema, semantic evaluator, exact
output, UAV/target/region/duration/coordination fields, unassigned requirements,
conflict scenarios, and reassignment/replan metrics, plus LoRA-minus-Base
deltas. Model loading remains in the prediction generator.

## J. Verify or regenerate the adapter manifest

Training creates both `run_manifest.json` and `adapter_manifest.json`. To
re-verify and reproduce the adapter manifest explicitly:

```bash
python training/lora/export_adapter_manifest.py \
  --adapter-dir /home/amax/ry/vlm_drones/models/adapters/fleet_planner/<run_id> \
  --base-model-name Qwen3-VL-4B-Instruct \
  --run-manifest /home/amax/ry/vlm_drones/outputs/lora/fleet_planner/<run_id>/run_manifest.json \
  --training-config /home/amax/ry/vlm_drones/outputs/lora/fleet_planner/<run_id>/config.json \
  --base-model-config /home/amax/ry/vlm_drones/models/initial_model/Qwen3-VL-4B-Instruct/config.json \
  --output /home/amax/ry/vlm_drones/models/adapters/fleet_planner/<run_id>/adapter_manifest.json
```

The exporter requires a valid PEFT config and non-empty safetensors envelope,
rejects recognized base weights, and binds model/config/dataset provenance.

## K. Review and deploy through the existing AdapterRegistry

Do not make training activate its own artifact. First complete adapter
verification, offline Base-vs-LoRA evaluation, and a vLLM smoke. Then manually
review a copy of `configs/adapters.json`, set only the intended
`fleet_planner` slot to `active`, and provide its real path, served name, base
lineage, and rank. Validate the existing production arguments with:

```bash
python scripts/build_vllm_lora_args.py \
  --config configs/adapters.json \
  --expected-base-model-name Qwen3-VL-4B-Instruct \
  --format lines
```

An active, verified adapter produces `--enable-lora`, `--lora-modules`, and
bounded rank/count arguments. `scripts/serve_qwen3_vl.sh` remains the serving
entrypoint. A placeholder adapter continues to fall back to the base model and
must not produce LoRA launch arguments.

## Outputs and tests

Formal run metadata is written below
`outputs/lora/fleet_planner/<run_id>/` (`config.json`, `run_manifest.json`,
`metrics/`, `tensorboard/`, and `checkpoints/`). Deployment artifacts go only
below `models/adapters/fleet_planner/<run_id>/`. Neither tree belongs in Git.

CPU/offline tests:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest tests/training/lora tests/fleet/test_lora_training_scaffold.py -q \
  -m 'not qwen_lora_integration'
```

The real integration test is separate and opt-in:

```bash
RUN_QWEN_LORA_INTEGRATION=1 QWEN_LORA_CUDA_VISIBLE_DEVICES=0 \
  pytest -m qwen_lora_integration \
  tests/training/lora/test_qwen_integration.py -q
```

Ordinary pytest collection never loads or downloads the 4B checkpoint.
