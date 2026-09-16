# Ranker audit and ranked experiment plan

> This document describes the experiments. See
> [next-steps.md](next-steps.md) for unfinished work.


Audit date: 2026-09-06. The audit is complete. See
[ranker-experiments.md](ranker-experiments.md) for code and results. The proposed
experiments remain here for reference.

Source line citations below describe the pre-implementation files at repository
revision `6956830b84866806f76dd5fdce67ac89c44de873`. Working-tree line numbers move
as the fixes are implemented; use that revision to inspect historical evidence.

The repo does **not** establish a 0.695 representation ceiling, does **not** establish that nonlinear heads cannot help, and does **not** establish that pooled training labels caused the plateau.

For this audit I reproduced the benchmark counts, linear capacity sweep,
human-only/distilled comparisons, and rater agreements using local labels, cached
embeddings, and the shipped artifact. Diagnostic retraining used one Torch CPU
thread. Subsequent new Gemini collection and cache refreshes belong to the
implementation log; they do not retroactively verify historical teacher settings
or the recorded 95 ms latency.

The chosen priority is **live–Gemini agreement**. Human agreement remains a separately reported quality measure.

## 1. Ceiling decomposition

### What actually reproduces

The benchmark contains **62 unique pairs involving 25 images**. Both humans gave decisive judgments on every included pair.

| Quantity | Recomputed result |
|---|---:|
| AC–DP agreement | 54/62 = **0.871** |
| Gemini agreement, averaged over both raters | 94/124 = **0.758** |
| Shipped ranker agreement, averaged over both raters | 80/124 = **0.645** |
| Student–Gemini agreement | 41/62 = **0.661** |
| Gemini on the 54 human-consensus pairs | 43/54 = **0.796** |
| Ranker on those consensus pairs | 36/54 = **0.667** |

These are the harness’s calculations, including its different denominators for humans and models. [benchmark_judges.py:235](../services/inference/scripts/benchmark_judges.py#L235)

The **22.58 percentage-point** gap in the original question decomposes arithmetically as:

- Ranker → Gemini: **11.29 points**.
- Gemini → AC–DP agreement: **11.29 points**.

That is **not a causal allocation to representation and teacher ceilings**.

On the 54 consensus pairs, the joint outcomes are:

| | Student correct | Student wrong |
|---|---:|---:|
| Gemini correct | 30 | 13 |
| Gemini wrong | 6 | 5 |

Replacing the student with Gemini would repair **13** decisions and break **6**, giving the net improvement of **7/62 = 11.29 points**. The remaining two student–Gemini disagreements occur where the humans disagree; changing the model’s answer cannot improve its average agreement with those two labels.

Consequently:

- The student already corrects **six teacher errors**. The claim that it “cannot exceed Gemini’s 0.758” is false. [handoff.md:315](handoff.md#L315)
- The observed teacher advantage is not statistically established by these counts alone: an exact paired test on the 13 versus 6 discordances gives **p = 0.167**, even before accounting for shared images.

### The label “ceiling” uses the wrong estimand

**0.871 is inter-rater agreement, not the maximum accuracy attainable against their recorded labels.**

On eight pairs, AC and DP disagree. Any binary prediction necessarily gets one of their two labels wrong. On the other 54, a prediction can match both. Therefore the unrestricted per-pair maximum on this particular benchmark is:

**(54 × 2 + 8) / 124 = 0.9355.**

This provides an exact accounting of the student’s observed error:

| Component | Error contribution |
|---|---:|
| Contradictory human labels: unavoidable against this recorded target | 8/124 = **6.45 points** |
| Gemini’s errors above that empirical optimum | 22/124 = **17.74 points** |
| Student’s additional errors relative to Gemini, net of corrections | 14/124 = **11.29 points** |
| Total student error | 44/124 = **35.48 points** |

This is a decomposition of **recorded outcomes**, not three population ceilings. The 0.9355 optimum permits arbitrary per-pair predictions; a pointwise ranker need not attain it.

The pooled agreement claim also needs correction. The recorded pooled inter-rater agreement is **458/668 = 0.686**, not approximately 0.5. The near-chance figures describe FW’s agreement with individual raters:

- AC–DP: **326/419 = 0.778**.
- AC–FW: **56/114 = 0.491**.
- DP–FW: **76/135 = 0.563**.

These reproduce from the [AC](../data/labelling/decisions.ac.jsonl#L1), [DP](../data/labelling/decisions.dp.jsonl#L1), and [FW](../data/labelling/decisions.fw.jsonl#L1) files. On the **same 106 three-rater decisive pairs**, agreement remains different: AC–DP **0.792**, AC–FW **0.491**, DP–FW **0.585**.

### The claimed representation ceiling fails a direct control

`fit_projection()` uses an economy SVD. With 128 training images, requesting 384 dimensions returns a **128×384 basis**, with centered training-data rank **127**. Thus the 128- and 384-dimensional sweep entries were the **same transformation**. [train_ranker.py:276](../services/inference/scripts/train_ranker.py#L276)

I reproduced the recorded sweep. I then used an identity 384-coordinate basis, retaining the existing centering and L2 normalization:

| True 384-coordinate linear fit | In-sample teacher agreement |
|---|---:|
| L2 = 0.01 | **0.701** |
| L2 = 0.001 | **0.739** |
| L2 = 0.0001 | **0.771** |
| L2 = 0 | **0.811** |

All use the same **1,926 decisive teacher labels**, with ties still included in training.

**0.811 is training fit, not a new generalisation result.** It nevertheless disproves “the frozen representation cannot fit this teacher beyond 0.695.” The original experiment jointly restricted dimensions and regularisation.

### What cannot be separated

The available evidence cannot allocate the student–teacher gap among:

- missing encoder information;
- PCA and normalization losses;
- regularisation and optimisation;
- Fashion144k versus application-image distribution;
- teacher randomness, presentation-order effects, or opponent-dependent judgments;
- changes caused by human fine-tuning.

The 0.645 benchmark evaluates a **human-fine-tuned artifact**, whereas the capacity and head-fidelity tables evaluate the **teacher-pretrained head**. Those are different stages and datasets. [train_ranker.py:830](../services/inference/scripts/train_ranker.py#L830)

Separating these requires matched model-stage predictions, true full-coordinate controls, image-disjoint teacher evaluation, repeated/swapped teacher judgments, and prospective human repeats. There are no blind within-rater repeats establishing whether disagreements reflect stable taste or inconsistent decisions.

## 2. Results to withdraw or qualify

The following covers the ranker-related quantitative claims in the repo. **Reproducible arithmetic can remain as descriptive evidence; the corresponding generalisation or causal claim often cannot.**

| Results | Verdict and reason |
|---|---|
| **0.871, 0.758, 0.645, 0.661; consensus-only 0.796/0.667** | Counts reproduce. Withdraw their interpretation as independent population estimates, established ceilings, or real-battle flip rates. The cohort and benchmark are selected, images repeat, and the benchmark uses single photographs. [handoff.md:26](handoff.md#L26) |
| Benchmark model CIs **[0.676, 0.825]**, **[0.558, 0.724]** | The harness passes **124** correlated rater comparisons from **62 pairs** into a binomial interval. Both labels share the same model prediction; pairs also share images. These are not valid independence-based uncertainty estimates for deployment. [benchmark_judges.py:239](../services/inference/scripts/benchmark_judges.py#L239) |
| AC+DP **0.643 [0.561, 0.717]**, pooled **0.553 [0.485, 0.619]**, human-only **0.636 [0.553, 0.711]** | Point estimates reproduce. The denominators are **140 decisions over 78 unique decisive pairs**, and **208 over 79**, respectively—not independent pairs. Test reuse and post-hoc cohort selection further invalidate confirmatory interpretations. [train_ranker.py:887](../services/inference/scripts/train_ranker.py#L887), [PRD.md:549](PRD.md#L549) |
| Initial single-rater **0.791 validation / 0.622 test**, CI **[0.508, 0.724]** | Reproduced. Validation selected the model; test pairs share images. “Better than chance” is not established by that naive CI. These are superseded exploratory results. [PRD.md:401](PRD.md#L401) |
| Capacity fits **0.674/0.681/0.696/0.695/0.695**, zero-shot **0.709/0.709/0.715/0.703/0.703**, fine-tuned **0.734/0.728/0.741/0.741/0.741** | Reproduced, but the teacher column is training accuracy, and the last two configurations are identical. Neither a representation ceiling nor a generalisation benefit is established. “Outside noise” for 16→64 is unsupported without an appropriate paired, clustered comparison. [handoff.md:60](handoff.md#L60) |
| ReLU table: training **0.725/0.784/0.856**, teacher holdout **0.661/0.663/0.655**, human validation **0.684/0.684/0.677** | **Not exactly reproduced.** My current-code, one-thread rerun produced training **0.723/0.768/0.879**, teacher holdout **0.645/0.658/0.681**, human validation **0.703/0.665/0.696**. Linear reproduced **0.677/0.676/0.734**. Historical artifacts, execution settings, and per-seed outputs are missing. Keep the historical MLP figures as notes, not reproducible exclusions of a function class. [handoff.md:90](handoff.md#L90) |
| All teacher “held-out” fidelity values and CIs | They measure unseen **pairs**, mostly involving seen **images**. Of 398 held-out pairs, **247** have both endpoints seen in training, **134** have one, and only **17** have neither. Among 386 decisive pairs, those counts are **241/128/17**. Valid for that specific pair-holdout task; invalid as image-disjoint generalisation evidence. [train_ranker.py:606](../services/inference/scripts/train_ranker.py#L606) |
| Pooled/cohort validation **0.638/0.715**, zero-shot **0.621/0.709**, distilled **0.660/0.734**; teacher-basis **0.613** | Reproduced. These are selected validation results. They do not establish label independence, universal domain-transfer failure, or a causal benefit from removing inconsistent training raters. [PRD.md:477](PRD.md#L477) |
| “Distillation does not improve accuracy,” **0.643 versus 0.636** | Observed difference is **one additional correct decision out of 140**. This establishes neither improvement nor equivalence. It also measures human agreement, not the primary live–Gemini objective. [PRD.md:535](PRD.md#L535) |
| “Within four points of the ceiling” | Compares selected validation accuracy with an all-split inter-rater statistic. The populations and estimands differ. Withdraw it. [PRD.md:493](PRD.md#L493) |
| Tanh correlation **0.991**, pair agreement **96%** | Historical notes only; the corresponding artifact is absent. They cannot be remeasured here. The implementation does support the stated collapse mechanism. [handoff.md:103](handoff.md#L103) |
| “Five test reads” | Verified as a statement in the handoff, not as an independently auditable execution count. There is no immutable evaluation ledger. Its advice to treat the results as reliable is too generous after adaptive selection. [handoff.md:217](handoff.md#L217) |
| **~95 ms/frame**, **~$5**, **~1 hour**, **~15 minutes** | Historical measurements/estimates without preserved benchmark or billing evidence. Do not treat them as reproduced costs. The pair endpoint performs two encoder calls sequentially, so per-frame latency is not pair latency. [handoff.md:249](handoff.md#L249), [main.py:237](../services/inference/src/fitted_inference/main.py#L237) |

Two additional bookkeeping corrections:

- **1,263 raw human decisions** is correct; **1,252** are usable. The **761 training decisions** represent **423 unique pairs**, with **704 decisive decisions over 412 pairs**.
- The stored teacher labels reference **1,992 images**, not 2,001. The graph contains 1,972 edges in 20 path components. There are **no cycles**, so these decisions cannot test whether teacher preferences violate a pointwise ordering. [handoff.md:178](handoff.md#L178), [distil_teacher.py:71](../services/inference/scripts/distil_teacher.py#L71)

## 3. Confounds in the rejected approaches

**Pooled versus coherent-cohort training never tested the claimed training-label effect.** FW has **zero training rows**: all 177 decisions are validation or test. Pooled and AC+DP training inputs are identical; in the reproduced distilled runs, validation also selects the same L2, **0.03**. The pooled test adds FW’s **25/68** agreement to AC+DP’s **90/140**, producing **115/208 = 0.553**. This is an evaluation-population change, not evidence that pooling FW corrupted learning. [decisions.fw.jsonl:1](../data/labelling/decisions.fw.jsonl#L1)

**The capacity sweep held the teacher regulariser fixed at 0.01.** That is not a capacity-limit experiment, especially when projected vectors are normalized and extra dimensions change their geometry. The actual full-coordinate control above demonstrates the confound. [train_ranker.py:287](../services/inference/scripts/train_ranker.py#L287), [train_ranker.py:830](../services/inference/scripts/train_ranker.py#L830)

**The ReLU comparison changed more than the function class.**

- Linear uses LBFGS; MLP uses 1,500 Adam steps without early stopping.
- MLP first-layer weights and biases receive **no regularisation**. Only the output weights are penalized.
- In teacher pretraining, ReLU permits scaling the first layer up and output weights down without changing predictions, while reducing that penalty. Thus “same L2 sweep” does not mean comparable functional regularisation.
- During human fine-tuning, the unanchored first layer can change the function while the output weights remain at their teacher anchor.

These tests establish regression for particular training recipes, not that nonlinearity has no value. [train_ranker.py:396](../services/inference/scripts/train_ranker.py#L396)

**The MLP teacher prior is selected using human validation labels.** Calling its human-validation score “BEFORE any human label” is false as a model-selection claim. Fine-tuning “restarts” also begin from the identical prior with no subsequent random operation, so changing the seed does not create independent fine-tuning runs. [train_ranker.py:450](../services/inference/scripts/train_ranker.py#L450)

**Teacher settings may differ from production.** Argument defaults are evaluated before `load_dotenv()`. With the current local settings, an ordinary invocation can choose prompt `v2` while production uses `v1`. Teacher rows record the model name but omit prompt/media provenance; benchmark rows record only pair ID and margin. Existing caches are reused without checking those settings. Historical mismatch is therefore **possible but not provable**, not something I can assert happened. [distil_teacher.py:118](../services/inference/scripts/distil_teacher.py#L118), [distil_teacher.py:226](../services/inference/scripts/distil_teacher.py#L226), [benchmark_judges.py:84](../services/inference/scripts/benchmark_judges.py#L84)

**The benchmark does not reproduce the final-round protocol.** It forces every Gemini margin into a binary direction. Production declares a draw below an absolute margin of **2**; **three cached benchmark judgments** meet that condition. Production also accepts image sequences, while this benchmark sends one photograph per player. Therefore **1−0.661 is not a measured battle-flip rate**. [benchmark_judges.py:222](../services/inference/scripts/benchmark_judges.py#L222), [engine.py:294](../services/inference/src/fitted_inference/engine.py#L294)

**Leakage and provenance need precise qualifications.**

- I found **no human image-ID overlap across splits**, no exact decoded-image duplicates in the 183-image human pool, and no exact decoded-image overlap with the 1,992 used teacher images.
- That does **not** establish person/outfit disjointness: ingestion normally hashes each filename into its own “subject.” Different photographs of the same person need explicit grouping. [ingest-label-pool.mjs:68](../scripts/ingest-label-pool.mjs#L68)
- Teacher-pool PCA is fitted before the teacher holdout split, including held-out image features. Its evaluation is transductive. [train_ranker.py:792](../services/inference/scripts/train_ranker.py#L792)
- Embedding caches are keyed by image ID, without encoder, preprocessing, or content hashes. Their historical provenance cannot be verified from the cache itself. [train_ranker.py:205](../services/inference/scripts/train_ranker.py#L205)
- The documented rebuild command omits `--teacher-holdout 0`, whereas the shipped artifact records **0.0** and the current default is **0.2**. It is not an exact rebuild command. [ranker.json:29](../models/ranker/ranker.json#L29), [train_ranker.py:736](../services/inference/scripts/train_ranker.py#L736)

## 4. Ranked experiments

There is no defensible numerical estimate of future held-out gain in this repo. Inventing one would violate the instruction not to speculate. The ordering below uses **measured defects, demonstrated training headroom, and concrete cost**. Where expected gain is unidentified, it is stated explicitly and a prespecified success threshold is given instead.

All changes remain offline until validated. Photos and captures stay in local, gitignored storage.

### 0. Repair evaluation and establish a fresh benchmark — Makes the metric trustworthy

- Implement and validate experiment 0 against the criteria below.

- **Hypothesis / ceiling:** Current measurement cannot distinguish label inconsistency, teacher inconsistency, and student generalisation.
- **Concrete change:** Add an evaluation-manifest module shared by `train_ranker.py`, `distil_teacher.py`, and `benchmark_judges.py`. Record content/group/split IDs, code/config hashes, actual basis dimensions, model stage, and teacher prompt/media provenance. Split images/groups before constructing teacher pairs. Load environment settings before argument defaults. Add cached-only evaluation and production draw handling.
- **Measurement:** Zero cross-split content/group overlap; reject mismatched caches; deterministic counts; paired subject-aware bootstrap intervals retaining all rater votes together. Retire existing tests to exploratory status. Collect an initial **200 fresh paired capture cases across at least 100 subject/outfit groups**, divided equally into disjoint validation and sealed test groups. Report decisive-final fidelity with coverage, displayed-leader/final-outcome agreement including draws, and human agreement separately.
- **Cost:** Harness work; **200 Gemini calls**, **400 human judgments plus 100 blind repeats**, and capture effort. No additional live inference.
- **Expected gain / confidence:** **0 accuracy points by construction**; high confidence that validity improves. This pilot does not guarantee power for small gains.
- **Falsification:** Any overlap, stale-cache acceptance, protocol discrepancy, or irreproducible count fails the repair. Wide intervals mean “unresolved,” not “no effect.”

### 1. Test genuine full-coordinate capacity with appropriate regularisation — Raises the metric

- Implement and validate experiment 1 against the criteria below.

- **Hypothesis / ceiling:** Compression and the fixed teacher penalty account for recoverable student–teacher disagreement.
- **Concrete change:** In `fit_projection()`, add an explicit identity-basis mode and report effective dimensions. In `project()`, expose normalization as a recorded option. Sweep 16/64/128/identity-384 with L2 **0, 0.0001, 0.001, 0.01**, selecting on image-disjoint **teacher validation**, not human validation.
- **Measurement:** At least **+3 percentage points** decisive teacher fidelity, with a paired interval excluding zero; report human agreement and raw/live latency separately. Examine training loss and fit to distinguish optimisation from generalisation.
- **Cost:** At most **32 cached-embedding linear fits** for normalized and unnormalized controls; no labels or encoder forwards.
- **Expected gain / confidence:** Held-out gain unidentified. Demonstrated training headroom is **+13.7 points** versus the shipped 16-dimensional teacher fit, or **+11.6** versus the claimed 0.695 ceiling. Confidence that the original exclusion was invalid: high; confidence in deployment gain: unknown.
- **Falsification:** Training improves but the held-out improvement’s upper confidence bound is below **+3 points**. A wider interval remains inconclusive.

### 2. Measure teacher repeatability, order sensitivity, and pointwise compatibility — Makes the metric trustworthy

- Implement and validate experiment 2 against the criteria below.

- **Hypothesis / ceiling:** Some apparent representation loss is an inconsistent or context-dependent target.
- **Concrete change:** Extend `distil_teacher.py::judge()` and `benchmark_judges.py::run_gemini()` to preserve individual repetitions, orientation, complete scores, and exact configuration. Judge **100 development pairs three times in each orientation**, plus **30 disjoint image triples** with all three edges.
- **Measurement:** Repeat agreement, swap consistency, draw transitions, and reproducible preference cycles. Separate repeated identical-input variability from changes when the opponent changes.
- **Cost:** **690 Gemini calls**; no human labels or live compute.
- **Expected gain / confidence:** **0 direct accuracy gain**. The existing 1,952 twice-used images have a median **2.7-point** score change between opponents, but the data cannot separate context from randomness. Confidence that those causes are currently unidentified: high.
- **Falsification:** Near-perfect repeat/swap agreement and no reproducible cycles would weaken teacher inconsistency as an important explanation. Absence of cycles in this small sample would not prove global transitivity.

### 3. Compare matched in-domain and Fashion144k distillation — Raises the metric

- Implement and validate experiment 3 against the criteria below.

- **Hypothesis / ceiling:** Training-image distribution, rather than head capacity alone, limits fidelity.
- **Concrete change:** Make `distil_teacher.py::sample_pairs()` accept an explicit training-image manifest. Generate **2,000 unique pairs** only from the existing **128 training images**, using the verified production configuration. Train human-only, teacher-only, and teacher-plus-human stages under identical projection and selection rules.
- **Measurement:** At least **+3 points** teacher fidelity on independent application-image validation/test; separately compare teacher-only and human-fine-tuned predictions. Existing-photo improvement must not be presented as webcam-domain improvement.
- **Cost:** **2,000 Gemini calls**, cached student embeddings, linear fits; no new human training labels. Live architecture unchanged.
- **Expected gain / confidence:** Unidentified; no measured in-domain ablation exists. Confidence in a positive gain: unknown.
- **Falsification:** An adequately precise paired interval rules out **+3 points** over matched Fashion144k training. Training-only improvement does not count.

### 4. Test information discarded by CLS-only extraction — Raises the metric

- Implement and validate experiment 4 against the criteria below.

- **Hypothesis / ceiling:** The frozen encoder contains useful information that its current single CLS descriptor does not expose.
- **Concrete change:** In `embed_pool()` and `Dinov2FitRanker.score()`, compare CLS, mean patch tokens, and their concatenation from the **same forward pass**. Version the descriptor and cache; apply the validated projection/regularisation procedure.
- **Measurement:** At least **+3 points** image-disjoint teacher fidelity, with paired uncertainty and identical input preprocessing.
- **Cost:** One embedding pass over the used images and cached head fits; no labels.
- **Expected gain / confidence:** Unidentified; the repo tests only CLS. Confidence in improvement: unknown.
- **Falsification:** Neither alternative produces a material held-out gain after comparable selection.

**CPU budget:** Encoder forwards remain unchanged, but pooling/projection overhead must be measured. Reject deployment if warm per-frame latency exceeds the **95 ms** budget on the target CPU.

### 5. Re-test nonlinear learning with controlled regularisation and a data curve — Raises the metric

- Implement and validate experiment 5 against the criteria below.

- **Hypothesis / ceiling:** The rejected MLP recipe overfit because its function was poorly constrained and its teacher dataset was small.
- **Concrete change:** In `fit_head()`, regularise both ReLU layers during teacher training; during human fine-tuning, constrain function drift using teacher-training examples instead of anchoring only `w2`. Add validation-selected early stopping and report every seed. Compare linear and width-16 ReLU on nested **500/1,000/2,000/4,000** teacher-pair sets.
- **Measurement:** At least **+3 points** independent teacher fidelity; decreasing generalisation gap with more unique images; report each stage before and after human fine-tuning.
- **Cost:** Up to roughly **2,000 additional teacher labels** beyond the existing pool, refreshed sampling over more images, and **24 head fits** for two heads × four sizes × three seeds.
- **Expected gain / confidence:** Unidentified. Existing training-fit gains and human-validation regression support running a controlled learning-curve test, not forecasting success.
- **Falsification:** Training fit increases while held-out fidelity remains flat, and uncertainty excludes a **+3-point** benefit. More repeated comparisons over the same tiny image set do not establish a useful data-scaling result.

### 6. Conditional last-block encoder adaptation — Raises the metric

- Implement and validate experiment 6 only after its prerequisite controls justify it.

- **Hypothesis / ceiling:** After valid descriptor/head controls, frozen features remain the limiting factor.
- **Concrete change:** Add an offline training path that unfreezes only DINOv2’s last transformer block and the head. Export the adapted encoder checkpoint with preprocessing/version metadata; load it explicitly in `Dinov2FitRanker.__init__()`.
- **Measurement:** Compare frozen versus adapted encoders on nested **2,000/4,000/8,000** training pairs spanning at least **2,000 distinct training images**, with three seeds and the same image-disjoint evaluation. Require at least **+3 points** held-out teacher fidelity.
- **Cost:** Up to **8,000 teacher labels total** for the adaptation dataset, image collection, and **nine adaptation fits capped at 20 epochs**, plus frozen controls. This requires backpropagation rather than cached-feature training.
- **Expected gain / confidence:** Unidentified and presently low evidential support. **8,000 is an experiment budget, not a scientifically established minimum label count**; the repo cannot identify such a minimum. Do not start until the cheaper controls and learning curve justify it.
- **Falsification:** No material advantage over the frozen model at the largest budget, or gains disappear on new subjects/outfits.

**CPU budget:** Last-block adaptation preserves the inference graph and forward count. Rebenchmark against **95 ms/frame**; do not assume unchanged architecture guarantees unchanged latency.

For every metric-raising experiment, choose configurations on validation, preserve all candidate outputs, and open the fresh test only for the final selected candidate. Do not repeatedly use it to adjudicate this list.

## 5. What not to try

- **Another `--dims 384` run through the existing PCA function.** It repeats the 128-coordinate transformation.
- **The same tanh recipe as evidence about nonlinear capacity.** Its known collapse invalidates that interpretation. GELU is also approximately linear near zero; the code’s claim otherwise is incorrect. [train_ranker.py:346](../services/inference/scripts/train_ranker.py#L346)
- **Blindly widening the current unregularised-first-layer MLP.** The observed human-validation regressions reject that recipe. They do not reject all nonlinear models.
- **An intercept to repair pointwise ranking.** A per-image intercept cancels in the score difference; a pair-margin intercept breaks the structural antisymmetry.
- **Temperature or percentile remapping as a raw ranking improvement.** Monotone remapping cannot repair incorrect orderings. Display stability is a different product measurement.
- **Removing FW and calling the resulting metric increase a training improvement.** FW contributed no training examples.
- **More human fine-tuning solely because human validation rises.** That is not the chosen objective. For reference, the existing teacher-only 16-dimensional head agrees with the cached benchmark Gemini on **38/62**, versus the shipped model’s **41/62**; this one exploratory comparison does not establish a general policy.
- **Teacher fine-tuning now.** It changes the target being imitated, and this repo has no clean human evaluation or label-count evidence supporting that change.
- **A larger encoder deployed without a latency gate.** The repo has not benchmarked alternatives against the CPU budget.
- **Any claim that more teacher data, every alternative encoder, or every nonlinear head has already been ruled out.** The evidence does not support those exclusions.
