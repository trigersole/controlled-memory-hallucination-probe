# Factual LoRA activation-shift experiment

This isolated experiment tests whether the directions most changed by factual LoRA
training predict correctness on unseen answers. It does not reuse the synthetic
adapters or their output directories. Implementation is not evidence that the
hypothesis works; real model runs are required.

## Run

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
header for your cluster. Set `PYTHON_BIN` if the environment is elsewhere. Stages skip
completed outputs, training resumes complete Trainer checkpoints, and generation/
replay resume atomic shards. Wall-time warnings stop the active process and requeue
the job. Restart the same command after an ordinary interruption.

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
