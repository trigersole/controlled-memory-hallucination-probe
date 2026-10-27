# Factual LoRA activation-shift experiment

This isolated experiment tests whether the directions most changed by factual LoRA
training predict correctness on unseen answers. It does not reuse the synthetic
adapters or their output directories. Implementation is not evidence that the
hypothesis works; real model runs are required.

## Run

To use the same Llama model as an existing run, generate a new configuration on
the cluster from that run's YAML. This copies the exact model identity/local path
and loading options; it keeps a context limit of at least 512 for the new QA inputs.
It creates a separate output directory and trains fresh TriviaQA adapters without
loading or changing your old synthetic adapters:

```bash
"$PYTHON_BIN" scripts/create_factual_shift_config.py \
  configs/pilot_llama2.yaml configs/factual_shift_llama2.yaml
sbatch slurm/factual_shift.sbatch configs/factual_shift_llama2.yaml
```

Use your actual existing run config filename in place of `configs/pilot_llama2.yaml`.
For a small Llama pipeline check, add `--template configs/factual_shift_smoke.yaml`
and choose `configs/factual_shift_llama2_smoke.yaml` as the target. The helper refuses
to overwrite configurations or reuse existing run output directories. Completed
Qwen outputs remain in their original folders.

Use the repository's normal Linux GPU environment (`scripts/setup_env.sh`). Start
with the small integration configuration:

```bash
python -m peft_probe --config configs/factual_shift_smoke.yaml validate-factual-shift
python -m peft_probe --config configs/factual_shift_smoke.yaml run-factual-shift
```

The smoke configuration is a pipeline check, not a statistically reliable study.
Analysis deliberately fails if its generated probe-training or validation answers
contain only one correctness class; increase those split sizes in a new configuration.

For the default study (Qwen2.5-3B, 4-bit LoRA, one GPU):

```bash
mkdir -p slurm/logs
sbatch slurm/factual_shift.sbatch configs/factual_shift.yaml
```

The job uses the same GPU partition convention as existing jobs; adjust its resource
header for your cluster. Set `PYTHON_BIN` if the environment is elsewhere. The launcher
also honors `HALOSCOPE_PYTHON`, and falls back to the existing
`$HOME/LLM_Haloscope/.venv/bin/python` when the repository has no `.venv`.
Like the older launchers, it adds `${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}`
to `PYTHONPATH`, so an existing PEFT overlay can supply dependencies without changing
the HaloScope environment. Set `PEFT_OVERLAY` explicitly if yours is elsewhere.
The selected Python, overlay path, and package versions are logged and imports are
checked before running stages. Stages skip
completed outputs, training resumes complete Trainer checkpoints, and generation/
replay resume atomic shards. Wall-time warnings stop the active process and requeue
the job. Restart the same command after an ordinary interruption.

If the model stages have finished and only the summary is missing, submit analysis
alone with the original configuration, without `--force`:

```bash
sbatch slurm/factual_shift.sbatch configs/factual_shift_llama2.yaml analyze-factual-shift
```

This uses the existing feature files and does not load model weights or train adapters.
The analysis prints progress for feature loading, layer fitting, and method evaluation.
`results/analysis_progress.json` records its latest step; ordinary Python failures also
write `results/analysis_failure.json` with a traceback. A native crash or scheduler
termination may leave only progress, so the launcher logs the process exit code too.
Fitted scores and each completed method's metrics are checkpointed, allowing an
interrupted analysis to resume. Feature matrices are released after fitting, and
CPU library threads are limited to the allocated CPUs. The final report remains
`results/summary.md`; intermediate checkpoints are not completed experimental results.

If PyTorch reports `failed finding central directory`, a saved archive is unreadable.
The analysis now includes the exact feature filename in its error. You can audit
every feature shard, including directories carrying `_SUCCESS.json`, using
`audit-factual-shift`. To repair and then finish analysis in one GPU job:

```bash
sbatch slurm/factual_shift.sbatch configs/factual_shift_llama2.yaml repair-factual-shift
```

Repair validates all sources, quarantines unreadable/missing or invalid shards, and
replays only affected shards with the existing base model or saved adapter. It does
not train adapters or generate new candidate answers. Readable matching shards are
preserved. Affected derived analysis files are archived under
`results/repair_archive/`, and repaired features are audited again before analysis.
Files with mismatched question/answer provenance are reported but not automatically
overwritten. The audit report is `factual_shift_v1/feature_integrity.json`.
Feature writes are flushed to disk and read back before a new completion marker is
created. A completion marker alone does not protect against later file corruption.
Check storage capacity/quota if damaged writes recur; the archive error alone does
not establish the cause of the damage. Do not use `--force` for this repair.

## Data and controls

All TriviaQA subsets come from `rc.nocontext/train`, with deterministic randomized
question-group order. Repeated evidence rows with the same question ID are collapsed
before splitting, merging answer aliases. Conflicting questions or canonical answers
under the same ID are rejected. Preparation metadata records the deduplication counts.
Normalized duplicate questions and transitively shared answer
aliases cannot cross subsets. Exact sizes are honored by discarding excess members
of a group rather than splitting that group. This is conservative observable
deduplication, not a guarantee against every semantic paraphrase or pretraining leak.

| Subset | Default questions | Purpose |
|---|---:|---|
| adapter_a | 500 requested | Correct-answer LoRA A and matched incorrect-answer control |
| adapter_b | 500 | Correct-answer LoRA B on independent facts |
| direction_fit | 500 | Paired canonical-answer activations; SVD/PCA fitting only |
| probe_train | 500 | Extra-label supervised correctness baseline |
| validation | 500 | Orientation and layer/rank/method selection |
| test | 2,000 | Frozen in-distribution evaluation |

The base model samples potential negative answers for A's training questions. Empty
answers, common abstentions, and answers containing known correct aliases are rejected. Questions with
no accepted negative after the configured attempts are omitted **from both A and
the incorrect control**. They share training order, seed, optimizer settings, and
question prompts. B uses an independent seed. The accepted count is recorded; the
stage fails if fewer than the minimum fraction can be used. No donor-answer fallback
is silently substituted.

Reference rejection does not prove falsehood. Inspect
`factual_shift_v1/negatives/rows.jsonl` for valid unlisted answers and implausible or
abstaining negatives before treating the control as evidence about truth. The first
two commands below allow inspection before training. If you curate this file, keep
A and the incorrect adapter matched and start training before any adapters/features
have been created. Training input hashes protect later reuse.

## Individual stages

```bash
CFG=configs/factual_shift.yaml
python -m peft_probe --config "$CFG" prepare-factual-shift
python -m peft_probe --config "$CFG" prepare-factual-shift-negatives
python -m peft_probe --config "$CFG" train-factual-shift --adapter correct_a
python -m peft_probe --config "$CFG" train-factual-shift --adapter correct_b
python -m peft_probe --config "$CFG" train-factual-shift --adapter incorrect
python -m peft_probe --config "$CFG" collect-factual-shift-candidates
python -m peft_probe --config "$CFG" collect-factual-shift --source base
python -m peft_probe --config "$CFG" collect-factual-shift --source correct_a
python -m peft_probe --config "$CFG" collect-factual-shift --source correct_b
python -m peft_probe --config "$CFG" collect-factual-shift --source incorrect
python -m peft_probe --config "$CFG" analyze-factual-shift
```

The first candidate per question is greedy; subsequent candidates are sampled.
All candidates come from the untouched base model. Every evaluator replays exactly
the same saved question/answer text, tokenized with the base tokenizer. Thus the
A-to-B test measures representation transfer on fixed candidates, not B's own
generation accuracy. Duplicate sampled answers remain in the saved candidate pool.

TriviaQA uses normalized exact match against aliases. An optional TruthfulQA
generation subset provides OOD evaluation, after filtering question/answer overlap
with the selected TriviaQA data. Its reference matching is a conservative proxy:
ambiguous outputs are saved with label -1 and excluded from accuracy/AUROC. Reports
include graded counts, including the denominator after reranking. This is not an
official semantic TruthfulQA evaluation. Set `ood_max_examples: 0` to disable it.

## Frozen analysis

For each configured layer, on **direction_fit only**:

1. Compute `D = h_adapter(q, gold_answer) - h_base(q, gold_answer)` in float32.
2. Fit uncentered SVD of D: the leading right singular vector captures maximum
   squared update magnitude, including the mean shift. Record singular values and
   fractions of total update energy explained. This differs from centered PCA.
3. Save the normalized mean shift and the top-k update subspaces.
4. Fit ordinary centered activation PCA on the same base examples as a baseline.
5. Fit a contrastive update SVD between A and its incorrect-answer control.

Answer-last is the default representation; answer-mean can be configured in a new
run. EOS is excluded from pooling. Empty generated answers explicitly use the last
prompt token, receive very low likelihood, and remain incorrect in TriviaQA. Inputs
that exceed the configured sequence length fail rather than silently truncating.
All requested layer indices must resolve to distinct hidden states.

Scores include signed projection, cosine alignment, projection energy, normalized
subspace energy, and update norm. Score centered base activations, adapter-on
activations, and paired adapter-minus-base changes. Update norm is a control, not
eligible as the primary direction method. The primary method is selected **only among A-derived
scores**, using validation AUROC. B, incorrect-control, contrastive, and ordinary-PCA
families each have their own validation-selected method for comparison. Random
direction and base teacher-forced mean answer log likelihood are baselines. A linear
correctness probe uses additional `probe_train` labels, which is explicitly a larger
label budget than the geometric methods.

Each score's sign is oriented on validation once. A-to-B uses A's direction,
base-derived origin, and A's validation orientation on adapter-on or update features
without fitting to B or using B test labels. A separate transfer method is selected
using A's validation scores only. Adapter overlap diagnostics are descriptive and cannot establish
correctness prediction on their own.

Test and OOD reports include correctness AUROC, 95% cluster-bootstrap intervals,
accuracy at preset coverages, candidate-reranking accuracy, greedy accuracy, and
the rate with at least one correct candidate. Bootstrap clusters preserve shared
question/alias groups and all candidate answers. Coverage values mean accepting
the highest-scoring fraction in the evaluation set; they are not deployed,
validation-calibrated absolute thresholds. Raw scores are not calibrated probabilities.

## Outputs and provenance

### Does the score distinguish answers to the same question?

After the original analysis completes, run this CPU-only diagnostic on the saved
predictions and candidate metadata. The standalone script uses only Python's standard
library and can run without PyTorch, PEFT, model weights, or a GPU:

```bash
python scripts/analyze_within_question.py \
  outputs/factual_shift_llama2/factual_shift_v1/results
```

If submitting through SLURM, the existing launcher also accepts:

```bash
sbatch slurm/factual_shift.sbatch configs/factual_shift_llama2.yaml within-question-factual-shift
```

This launcher retains its normal GPU reservation even though this diagnostic uses
only the CPU. The direct script avoids that reservation. Its `--bootstrap-samples`
and `--seed` default to 500 and 42; the experiment CLI uses the existing config values.

The diagnostic keeps the original validation-selected method and saved score sign.
For each question with both correct and incorrect candidates, it compares all
correct/incorrect pairs, awarding half credit for ties, then averages equally
over questions. Questions with no mixed graded candidates are counted but excluded
from this metric. Ambiguous labels (-1) are excluded from comparisons. Duplicate
sampled answers retain their sampling multiplicity; pairs are not treated as
independent observations. Bootstrap resampling preserves entire question/alias
groups, and comparisons against likelihood use the same draws for paired intervals.

Reports include pooled AUROC, within-question pairwise accuracy, a question-centered
pooled AUROC, eligible question/cluster counts, and the paired difference against
likelihood. A question-mean control assigns every answer to a question the same
mean selected score. Its within-question accuracy must be 0.5 whenever mixed
questions exist. This control uses saved answer scores; it is not a prompt-only
activation measurement. High pooled AUROC and near-chance within-question accuracy
are consistent with a question-level signal, without proving its causal origin.

Outputs are separate from the original report:

- `results/within_question/summary.md`
- `results/within_question/metrics.json`
- `results/within_question/per_question.json`

Original predictions, metrics, adapters, and features are not rewritten. Source
hashes protect against stale report reuse. If intentionally changing diagnostic
settings, the standalone `--force` rebuilds only `within_question/` reports.

Everything is under `outputs/trivia_factual_shift/factual_shift_v1/`:

- `data/rows.jsonl`: split/group assignments and factual references.
- `negatives/rows.jsonl`: accepted control targets and provenance.
- `adapters/{correct_a,correct_b,incorrect}/`: checkpoints and final adapters.
- `candidates/shard_*.jsonl`: shared base-generated answer pool and grades.
- `features/<source>/shard_*.pt`: paired multilayer replay features and likelihoods.
- `results/frozen_methods.pt`: directions, origins, probes, orientations, selection.
- `results/metrics.json`: every method's metrics and geometry diagnostics.
- `results/predictions.jsonl`: selected-method and likelihood predictions.
- `results/summary.md`: compact comparison of validation-selected families.

Configuration and prerequisite manifests reject stale downstream reuse. Change
`experiment.output_dir` for a new configuration. If intentionally recomputing the
same run with global `--force`, rebuild every affected downstream stage as well;
`--force run-factual-shift` rebuilds the whole pipeline. Completion markers are
removed before forced work so failed recomputations cannot appear complete.

For a scientific study, repeat with multiple experiment seeds, training-set sizes,
and model families. Review both the fixed leading-vector result and the selected
subspace result; a successful validation-selected score alone does not show that
the single largest change is a universal correctness direction.
