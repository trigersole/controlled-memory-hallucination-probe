# Transferable correctness information in factual LoRA activation changes

Status: exploratory pilot completed; prospective replication implementation ready.
The confirmatory Llama cluster experiments have not been run by the local agent.
This is a repository-frozen analysis plan, not a registered preregistration.

## Research question and scope

Does a subspace fitted to base-to-LoRA activation changes on factual QA predict
which answers are correct within the same question, transfer to another factual
adapter, and outperform a matched incorrect-answer training control?

The target claim is predictive information under a specified training and scoring
protocol. It is not universal truth, a causal intervention claim, an improvement in
the model's generation accuracy, or a label-free discovery method. Training uses
factual labels; signs use validation correctness labels. The exploratory study
also used validation labels to select layer, representation and rank.

## Exploratory evidence supplied by the researcher

These are the user's cluster reports, not local reproductions. Model identity was
reported as Llama-2-7B; the exact local checkpoint/loading configuration remains in
the original cluster YAML. Primary: `-17/correct_a/delta/fraction_8`.

| Test method | Pooled correctness AUROC | Within-question pairwise accuracy | 95% question/alias CI |
|---|---:|---:|---|
| Factual A | 0.8115 | 0.8327 | [0.7921, 0.8736] |
| A direction transferred to B | 0.8112 | 0.8200 | [0.7752, 0.8574] |
| Matched incorrect-answer training | 0.7747 | 0.7745 | [0.7291, 0.8244] |
| Base answer likelihood | 0.6445 | 0.4691 | [0.4126, 0.5337] |

Transfer minus native A: -0.0127, paired CI [-0.0468, 0.0160].
A minus incorrect training: 0.0582, paired CI [0.0146, 0.0938].
Only 275/2,000 test questions were mixed, comprising 169 bootstrap clusters.
These are pairwise rankings conditional on the candidate pool, not answer-selection
accuracy on all questions. Duplicate sampled answers retain sampling multiplicity.

The question-mean control reached pooled AUROC 0.8077 but necessarily had 0.5
within-question ranking. It is derived from answer scores, not question-only
activations. The primary's 0.8327 within-question result rules out a purely
question-constant score as the full explanation for the observed ranking.
It does not decompose the pooled AUROC into question and answer contributions.

OOD generated-answer scores had no usable mixed questions out of 300; 749
ambiguous candidates were excluded from grading comparisons. Original OOD pooled
AUROC was 0.4183 for A. Thus the pilot provides neither positive OOD robustness
evidence nor an OOD within-question estimate. These negative/unavailable results
must remain in the report alongside the follow-up reference-ranking results.

The supervised base-activation probe reached test AUROC 0.8623 at layer -17,
above the geometric primary's 0.8115. It uses extra labeled probe-training data;
the pilot does not establish superiority over supervised correctness probing.

## Fixed confirmation protocol

Three fresh full-pipeline seeds: 43, 44, 45. The exploratory source seed is excluded.
Each seed changes grouping/split randomization, negative sampling, training,
candidate sampling and SVD randomness. This studies full-pipeline variability;
it does not isolate optimizer-initialization variability on a fixed dataset.
All runs copy the source's exact model and LoRA configuration into new output roots.

The primary method is fixed to `-17/correct_a/delta/fraction_8`. It is never
reselected on new validation, test or OOD results. Directions are fitted only on
direction-fit canonical answers. Score signs are oriented once using that run's
in-distribution validation labels. Transfer uses A's basis, origin and sign on B
features. Primary control: identical layer/view/rank for the incorrect adapter,
with its own validation sign. The independently validation-selected incorrect
family result is a secondary comparison. Original model and optimization settings
are inherited; the three adapters per seed are newly trained (nine in total).

Primary test endpoint: mean, equally weighted over mixed questions, of the fraction
of correct/incorrect answer pairs ordered correctly; ties receive half credit.
Use 2,000 bootstrap draws of entire question/alias groups, shared across methods.
Record eligibility denominators and ambiguity counts. Pooled AUROC is secondary.

Report native A, A-to-B, matched and validation-selected incorrect controls,
base likelihood, answer length in words, validation-selected base activation PCA,
the primary layer's random-direction control, and its supervised probe. The length
baseline's sign uses validation only, like the other baselines. It addresses a
simple shortcut, not every lexical/style confound. Extra-label supervised probing
must remain identified as a different supervision budget.

Prespecified paired contrasts: A-to-B minus A; A minus matched incorrect;
A minus likelihood; A minus length. Additional PCA/random/probe differences and
the independently selected incorrect control are reported as secondary controls.
Report all intervals; no multiplicity-adjusted claim or best-seed selection is made.
An interval including zero does not establish equivalence. Transfer equivalence
is not tested because no justified noninferiority margin was specified in advance.

Summarize each method with an equal-seed mean and sample standard deviation,
alongside every per-seed estimate and paired interval. Do not pool repeated
seed-question observations as independent data; questions can overlap across seeds.
Question-bootstrap intervals condition on a trained run and do not estimate
training-seed uncertainty. With three runs, report variability descriptively rather
than presenting a falsely precise combined interval. Failed seeds must be resumed,
reported as failed, or the study marked incomplete; never silently omitted.

The confirmation checks replication under this protocol. Because they use the
same broad data sources as the exploratory study, fresh seeds do not create an
entirely untouched benchmark or rule out pretraining contamination.

## Separate OOD reference-ranking task

Use the existing TruthfulQA generation dataset's supplied correct and incorrect
reference answers, up to three distinct normalized answers per class per question.
Remove exact normalized conflicts across classes; fail if no mixed pair remains.
Choose references in dataset order without consulting detector scores, then shuffle
candidate order deterministically. Score every candidate with the same base/A/B/
incorrect teacher-forced replay used for generated TriviaQA candidates.

This guarantees labeled comparisons by construction, without pretending ambiguous
generations are incorrect. It is an OOD reference-answer ranking experiment, not
generated-answer detection and not the official MC1, MC2, or updated two-choice
TruthfulQA metric. Retain this name in every table and conclusion. Do not compare
its accuracy directly against the pilot's generated-answer OOD AUROC as if the task
were unchanged. Reference style and length can be label-correlated. Original
benchmark annotations may be outdated; dataset provenance and a label audit remain
necessary when making stronger truthfulness claims.

The dataset authors distinguish generation and multiple-choice tasks, describe the
new two-choice setting, and note dated/invalid questions in their change log:
[TruthfulQA official repository](https://github.com/sylinrl/TruthfulQA).

## Interpretation after all planned runs finish

Report results irrespective of sign. A replicating above-chance within-question
effect supports answer-specific predictive information for the evaluated pool.
Above-chance A-to-B ranking supports transfer to the second adapter; a small paired
change is descriptive unless equivalence has separately been established.
A consistent positive factual-minus-matched-incorrect effect supports specificity
to the factual training protocol beyond generic adaptation. A strong incorrect
control must still be acknowledged. If the length/PCA/random controls explain the
ranking equally well, weaken the interpretation accordingly. If OOD reference
ranking fails, state the method's domain/task limitation rather than reverse its
sign or select another layer after observing OOD labels.

No causal claim follows without interventions. Prompt-only activations have not
been measured by the question-mean control. Multi-model generality, truly novel
facts, multiple training sizes, audited generated-answer OOD labels and independent
benchmark replication remain separate extensions, not completed findings.

## Related work and limits of novelty

This targeted literature check was performed on 2026-10-27. It is not evidence of
exhaustive priority coverage, and the study should not claim to be the first to
discover truth-related activation directions, use LoRA for truthfulness, or use
activation geometry for hallucination detection.

- [Burns et al., Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827)
  extract activation-space information through logical consistency, without the
  factual adapter-change construction used here.
- [Du et al., HaloScope](https://arxiv.org/abs/2409.17504) use unlabeled generated
  mixtures to estimate truthfulness membership and train a classifier. Our experiment
  uses factual LoRA changes, validation orientation and matched training controls;
  it is not an annotation-free replacement for HaloScope.
- [Li et al., Alleviating Hallucinations in Large Language Models via Truthfulness-driven Rank-adaptive LoRA](https://aclanthology.org/2025.findings-acl.103/)
  connect truthfulness probing to module-dependent LoRA rank allocation. This is
  direct overlap with truthfulness and low-rank adaptation, although its training
  objective differs from the answer-ranking transfer diagnostic studied here.
- [Mahmoud et al., The Unintended Trade-off of AI Alignment](https://aclanthology.org/2026.findings-eacl.53/)
  analyze LoRA-based hallucination mitigation and associated safety trade-offs.
  This further limits any broad novelty claim about LoRA changing truth-related
  activation behavior.
- [Challenges with unsupervised LLM knowledge discovery](https://deepmind.google/research/publications/66937/)
  motivates sanity checks against prominent features unrelated to truth.

A defensible contribution to investigate is the combined evidence from
factual-adapter activation-change subspaces, independently trained adapter transfer,
within-question ranking, and matched incorrect-training controls. Whether this exact
combination is novel requires a fuller related-work comparison before submission.

## Reproduction and remaining work

See `FACTUAL_SHIFT.md` for the cluster commands. The generated `study.json` records
planned seeds, copied config hashes and output directories. The SLURM array runs
one seed at a time and uses existing requeue/resume behavior. Completed pilot runs
are never inputs for new training or overwritten. Each seed writes the original
analysis plus its separate final within-question report. Study aggregation refuses
missing, modified, differently selected or uninformative planned runs.

The empirical study is complete only after cluster execution, inspection of every
planned report, resolution of failures/label issues, and a final write-up including
unfavorable controls and the original OOD limitation. This repository document is
a protocol and pilot account; it does not fill pending result tables with estimates.
