# qwen_lora environment

Fleet Planner LoRA inspection and any future training must run in a dedicated
`qwen_lora` environment, not Isaac Sim's `r_isaac_sim` environment and not the
YOLO environment. The committed configuration is intentionally `placeholder`:
the validation command reads configuration and JSONL only, does not load Qwen,
does not start training, and does not create Adapter weights.

The dataset is contract-bound: each `input` is an exact serialized
`FleetMissionRequest`, normal gold `output` values are exact
`FleetMissionPlan` objects, and the execution-failure record in
`test_reassignment.jsonl` uses `output_kind=fleet_plan_patch` plus a production
`FleetPlanPatch`. The planning-time unavailable-UAV record remains a normal
plan. Spatial regions use the Spatial V3 `shape`/`frame` schema; no
training-only region or coordination-policy dialect is accepted. The dataset
validator also verifies the manifest contract, split counts, SHA-256 values and
complete fourteen-scenario catalog before the placeholder command succeeds.

```bash
./python.sh training/lora/train_fleet_planner_lora.py \
  --config configs/lora/fleet_planner_lora.json
```

Before real training, create and document a separate compatible environment,
inspect the actual local checkpoint with `inspect_qwen_lora_targets.py`, review
language-backbone module names, and then choose rank, alpha, dropout, batch size,
learning rate, and target modules. Do not copy speculative Transformers, PEFT,
TRL, or vLLM pins into `environment.yml`; do not enable an adapter in
`configs/adapters.json` until a real path exists and has passed service checks.
