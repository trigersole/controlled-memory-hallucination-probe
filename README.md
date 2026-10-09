# Controlled-memory PEFT probe

This repository implements the pilot experiment discussed in the project chats:

1. Generate fictional facts with known truth values.
2. Assign every non-OOD fact to exactly one of two complementary LoRA adapters.
3. Independently split entities into probe train, validation, and test sets.
4. Generate answers from Adapter A, Adapter B, and the untouched base model.
5. save adapter-on and adapter-disabled **base-replay** hidden-state features.
6. Train matched correctness-only, shuffled-exposure, and genuine-exposure probes.
7. Remove the adapters and test transfer on TriviaQA and TruthfulQA.

Every expensive stage is resumable. LoRA training uses regular Trainer checkpoints, generation is
written in atomic shards, and probe training saves an atomic checkpoint after every epoch.

## Default choices

| Setting | Default | Alternatives |
|---|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` | Set any causal instruction model in YAML; the smoke config uses Qwen2.5-1.5B |
| GPU profile | One 24 GB GPU, 4-bit QLoRA | Set `load_in_4bit: false` on a 40–80 GB GPU; lower batch sizes for <=16 GB |
| Scope | Full pilot, 3,000 facts, five probe seeds | Use `configs/smoke.yaml` or reduce facts/seeds in a copied config |
| Main representation | Base replay | Adapter-on representations are included as a prespecified ablation |
| TruthfulQA | MC1 | `truthfulqa_mode: generation` enables conservative reference matching and excludes ambiguous rows |

Qwen is ungated, so the default does not require accepting a model license. Set `HF_TOKEN` only if
you change to a gated model.

## One-time cluster setup

From a login node in this repository:

```bash
module load python/3.11  # use the equivalent module on your cluster
bash scripts/setup_env.sh
export PYTHON_BIN="$PWD/.venv/bin/python"
export HF_HOME="${SCRATCH:-$PWD}/huggingface"
$PYTHON_BIN -m peft_probe --config configs/pilot.yaml validate
```

The environment script only creates `.venv` and installs this package. If your cluster supplies
PyTorch through a module or Conda, activate that environment and install with `pip install -e .`
instead, then point `PYTHON_BIN` to its Python executable.

## Submit the complete SLURM pipeline

```bash
export PYTHON_BIN="$PWD/.venv/bin/python"
export HF_HOME="${SCRATCH:-$PWD}/huggingface"
export SLURM_ACCOUNT=my_account      # omit if the cluster does not require it
bash slurm/submit_pipeline.sh configs/pilot.yaml
```

The submission script creates one self-requeuing job, suitable for QOS policies that permit only
one submitted job per user. It runs the stages sequentially:

```text
data -> adapters A/B -> synthetic features -> benchmark features -> intervention -> probes -> evaluation
```

Progress is stored in a configuration- and schema-keyed file under `slurm/state/`. The same job
ID requeues before its wall time and resumes the current stage from its application checkpoint.
Edit the resource header in `slurm/serial_pipeline.sbatch` if the cluster partition or limits
change.

### Refresh an older completed run after analysis-schema changes

The corrected-analysis job preserves generated data and LoRA adapters, archives the small prior
result JSON files, removes only regenerable features/probes/results, and rebuilds all affected
artifacts with checkpointed single-job execution:

```bash
sbatch slurm/rerun_corrected_analysis.sbatch configs/pilot_llama2.yaml
```

Use this only for a completed pre-schema-v3 run. Subsequent schema-v3 runs should use the normal
submission script.

### Preemption and wall-time recovery

- LoRA checkpoints are under `outputs/<run>/adapters/adapter_*/checkpoints/`.
- Synthetic and benchmark inference writes `shard_XXXXX.pt` atomically and skips finished shards.
- Probe checkpoints are `last_checkpoint.pt` files saved after every epoch.
- `_SUCCESS.json` is written only after a stage has completely finished.
- `_RUN.json` stores the full configuration and prevents artifacts from different configurations
  from being mixed; change `experiment.output_dir` when changing experiment settings.
- The serial SLURM job requests `--requeue` and requeues on the warning signal before wall time.

Simply resubmitting the pipeline or the failed stage resumes it. Do **not** pass `--force` unless
you intentionally want to recompute completed artifacts under the same configuration.

## Manual or interactive run

The same stages can be run without the submission helper:

```bash
CFG=configs/smoke.yaml
python -m peft_probe --config "$CFG" generate-data
python -m peft_probe --config "$CFG" train-adapter --adapter a
python -m peft_probe --config "$CFG" train-adapter --adapter b
python -m peft_probe --config "$CFG" collect-synthetic --source adapter_a
python -m peft_probe --config "$CFG" collect-synthetic --source adapter_b
python -m peft_probe --config "$CFG" collect-synthetic --source base
python -m peft_probe --config "$CFG" intervention-report
python -m peft_probe --config "$CFG" train-probes
python -m peft_probe --config "$CFG" collect-benchmark --benchmark trivia_qa
python -m peft_probe --config "$CFG" collect-benchmark --benchmark truthful_qa
python -m peft_probe --config "$CFG" evaluate --benchmark trivia_qa
python -m peft_probe --config "$CFG" evaluate --benchmark truthful_qa
```

For a quick integration check, use `configs/smoke.yaml`. It exercises the complete code path with
60 facts, one epoch, one probe seed, and 20 examples from each benchmark. It is not a scientific
run.

## Output layout

```text
outputs/<experiment>/
├── data/                       # truth table, assignments, entity splits
├── adapters/
│   ├── adapter_a/{checkpoints,final}/
│   └── adapter_b/{checkpoints,final}/
├── features/
│   ├── synthetic/{adapter_a,adapter_b,base}/shard_*.pt
│   └── benchmarks/{trivia_qa,truthful_qa}/shard_*.pt
├── probes/<feature_mode>/<variant>/seed_<n>/
└── results/
    ├── intervention_check.json
    ├── trivia_qa/{metrics.json,predictions.jsonl,risk_coverage_*.jsonl}
    └── truthful_qa/{metrics.json,predictions.jsonl,risk_coverage_*.jsonl}
```

`predictions.jsonl` retains every question, generated answer, label, confidence baseline, and probe
risk so analyses can be reproduced without rerunning the LLM.

## Scientific safeguards built into the code

- Adapter assignment and detector entity splits are independent.
- Adapter A/B have balanced alternating assignments within every relation.
- Adapter training questions and collection questions use disjoint templates.
- Withheld exposure is metadata, never a correctness label; answers are graded against the truth table.
- Both probes receive identical correctness examples, including untouched-base generations.
- Exposure loss applies only to A/B examples; base-model exposure is undefined and masked.
- Shuffled exposure is permuted within source, relation, and split.
- The full config stops before probe training if the exposed-minus-withheld accuracy gap is below 0.10.
- Real benchmark labels are used only after probe training.
- Results include log-probability and entropy baselines plus AUROC, AUPRC, Brier, ECE, AURC,
  risk–coverage curves, and paired bootstrap differences.

TruthfulQA MC1 is the default because it gives deterministic labels. Generation mode is available,
but reference matching cannot reliably grade paraphrases; ambiguous generations are deliberately
excluded and grading coverage is reported. A publication-quality generation experiment should add
a prespecified human or validated judge protocol rather than silently treating unmatched text as
false.

## HaloScope baseline

HaloScope should be reported as an external baseline, not as one of the three matched probes. Its
official implementation is built around Llama-2 and OPT, custom attention/MLP hooks, and its own
generation and BLEURT/ROUGE labeling pipeline. This repository therefore does not label a simplified
Qwen reimplementation as “official HaloScope.” The benchmark shards retain base-model hidden states,
labels, and generations needed to add a validated Qwen port. For a faithful reproduction, run the
[official HaloScope repository](https://github.com/deeplearning-wisc/haloscope) with its supported
model and dataset setup and report it in a separate comparison block.

## Prompt-only hidden-space geometry

After a completed controlled-memory run, the optional geometry pipeline maps the representation
space without changing the adapters, probe outputs, or existing result report. It collects
prompt-only activations at multiple layers, forms same-fact paired deltas
`h_exposed - h_withheld`, and fits the memory subspace only on the synthetic training split.

The analysis reports subspace dimensionality, held-out and relation stability, k-nearest-neighbor,
nearest-centroid and Mahalanobis separation, linear CKA, RBF MMD, principal-angle overlap with
benchmark correctness directions, and benchmark error ranking along both the oriented memory
direction and full-subspace energy. Shuffled exposure labels, shuffled pairs, and matched random
directions/subspaces are included as controls.

Submit it only after the normal or corrected probe pipeline has completed:

```bash
sbatch slurm/geometry_pipeline.sbatch configs/pilot_llama2.yaml configs/geometry.yaml
```

It remains compatible with a one-submitted-job QOS. Atomic feature shards and the SLURM state file
make it safe to requeue or resubmit. Results are isolated under:

```text
outputs/<experiment>/geometry/multilayer_v1/results/
├── geometry_metrics.json
├── geometry_summary.md
└── memory_subspaces.pt
```

## Matched HaloScope-style comparison

The optional matched comparison does **not** claim to be an official HaloScope reproduction.
Instead, it implements HaloScope-style centered, singular-value-weighted PCA direct projection and
compares it with the controlled-memory direction under matched conditions. Both methods use the
same Llama model, saved generations, grading labels, layers, stratified wild/validation/test split,
and validation-only layer/rank selection.

Two tracks are reported:

- prompt-only representations for both methods;
- answer-conditioned representations obtained by replaying the exact saved answer text, without
  generating a new answer. The controlled synthetic comparison replays the same base-model answer
  through both adapters so answer wording cannot identify exposure.

Run this only after the geometry pipeline has completed:

```bash
sbatch slurm/haloscope_comparison.sbatch \
  configs/pilot_llama2.yaml \
  configs/geometry.yaml \
  configs/haloscope_comparison.yaml
```

Collection is atomically sharded, the analysis caches its answer-conditioned memory subspaces and
each completed benchmark, and one self-requeuing job remains compatible with a one-job QOS. The
primary report is written to:

```text
outputs/<experiment>/comparisons/haloscope_matched_v1/results/matched_summary.md
```

Official HaloScope should still be reported separately using its authors' repository. In
particular, their full method trains a classifier from PCA-derived pseudo-memberships; the matched
comparison here deliberately reports the direct-projection component so that data access and test
inputs remain explicit.
