# Beyond Ratings
## Course-Aware Instructor Effectiveness with Reliability and Tier Stability

A reproducible research project using the supplied 2,000-batch EdTech dataset. It combines a transparent descriptive score with a separate classical ML experiment and asks whether the resulting tiers are stable enough to support contextual coaching review.

**AI assistance disclosure:** this project was prepared with AI assistance. The original assessment explicitly prohibits LLM assistance. Use this as a learning/reference project unless your evaluator authorizes an AI-assisted submission. The analysis itself calls no LLMs, external APIs, external datasets or AutoML services.

## Executed findings

| Measure | Result |
|---|---:|
| Supplied batches / instructors / courses | 2,000 / 120 / 25 |
| Development / held-out instructors | 90 / 30 |
| Selected estimator | Logistic regression |
| Development three-fold mean macro F1 | 0.686 |
| Held-out macro F1 | 0.533 |
| Held-out balanced accuracy | 0.548 |
| Circular full-score reconstruction macro F1 | 1.000 — formula reproduction only |
| Instructors changing descriptive tier under any sensitivity scenario | 17 / 120 |
| Conditional 90% intervals crossing a descriptive boundary | 49 / 120 |

These are concurrent associations with constructed proxy labels. There is no independent effectiveness ground truth, causal claim, future prediction claim, or production-readiness claim. Three-fold model-selection results are not an unbiased substitute for the held-out result.

![Held-out evaluation and development validation](outputs/evaluation.png)

## How it works

1. Validate schema, numeric ranges, missingness and batch uniqueness. Inspect distributions, correlations and instructor/course coverage. Keep the original input unchanged.
2. Fit per-course metric means and residual standard deviations using development instructors only. An unseen course uses the development global metric mean.
3. Aggregate standardized batch metrics into instructor means/SDs, batch counts and distinct-course counts. Equal batch weights are necessary because cohort sizes are absent.
4. For descriptive full scoring, combine outcomes (70%), engagement (20%) and feedback (10%). Within outcomes, completion/improvement/quiz receive 45/40/15%. These are documented judgment calls.
5. Shrink the composite toward zero by `n / (n + 5)` and report `50 + 10 * shrunk_composite`. This is a relative index, not a bounded grade. Tertiles define relative tiers.
6. For the primary ML task, define an outcome-only tier and predict it using engagement means/SDs and counts. Exclude outcome ingredients, IDs and feedback. A separate circular experiment demonstrates why giving a model its label ingredients can produce misleadingly high results.
7. Compare dummy, logistic and shallow random-forest models. Refit scoring references and label cuts inside every development fold. Select by mean macro F1, then evaluate once on unseen instructors.
8. Examine permutation importance, alternative weighting/course/shrinkage policies, nearby cutoffs and conditional within-instructor batch-resampling intervals.

The largest positive permutation contributions on this holdout were mean forum activity (~0.071 macro-F1 decrease on shuffling) and mean assignment submission (~0.063). Repeat SDs were ~0.065 and ~0.058 respectively: the estimates are noisy. Neither establishes a causal driver. Negative permutation importances elsewhere do not justify post-hoc feature deletion on this test set.

## Run the notebook

Open `Beyond_Ratings.ipynb` in Jupyter or upload it to Google Colab. Upload the supplied CSV alongside it, or retain `data/instructor_batches.csv`. Run all cells. The notebook embeds its own analysis functions and does not need `effectiveness.py` to be imported. Colab's file pane can upload the CSV without notebook API calls.

For a local environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Open the notebook using your existing Jupyter environment.
```

The pinned requirements cover analysis/inference. Jupyter/Colab provides the notebook host; notebook-host packages are not analysis dependencies.

## Train and predict from Python

```bash
python train.py --data data/instructor_batches.csv
python predict.py --input data/instructor_batches.csv --output outputs/new_predictions.csv
```

Run from the repository root. Prediction produces one row per instructor and requires the complete input schema. The classifier itself consumes only engagement/count features. A newly supplied instructor needs observed batch data; the model cannot predict from an ID alone.

## Files

| File/folder | Purpose |
|---|---|
| `Beyond_Ratings.ipynb` | Single self-contained notebook with explanations, executed outputs and all five mandatory answers |
| `effectiveness.py` | Reusable scoring, training and inference functions matching the notebook |
| `train.py` / `predict.py` | Command-line entry points |
| `data/instructor_batches.csv` | Unmodified supplied CSV |
| `models/evaluated_model.pkl` | Development-fitted estimator and reference policy used for the held-out result |
| `models/instructor_model.pkl` | Selected estimator and policy refit on all supplied instructors for research inference |
| `outputs/metrics.json` | Recorded evaluation metrics and caveats |
| `outputs/cv_folds.csv` / `cv_summary.csv` | Development validation results |
| `outputs/heldout_predictions.csv` | Actual constructed labels, predictions, probabilities and review flags |
| `outputs/descriptive_instructor_scores.csv` | Full-composite scores, tiers and evidence counts |
| `outputs/tier_stability.csv` | Per-instructor sensitivity results |
| `outputs/conditional_score_intervals.csv` | Conditional batch-resampling intervals |
| `outputs/*.png` | EDA, evaluation, importance, sensitivity and uncertainty figures |
| `MODEL_CARD.md` | Intended use and limitations |
| `publish.sh` | Optional GitHub CLI creation/push script; private repository by default |

The two model files serve different purposes. The full-data refit has no separate measured performance; do not attach the held-out metrics to it as an independent evaluation. Pickle artifacts include scikit-learn estimators, pandas references, feature order and tier policies; load only trusted artifacts using compatible package versions.

## Verification

All 12 Python code cells were executed in order in one shared Python process, with stdout and matplotlib outputs embedded. Jupyter kernel socket binding was prohibited in the build environment, so execution used an in-process Python runner rather than a live Jupyter kernel. The notebook was schema-validated. It contains plain Python without magics or shell cells. A fresh Python process verified serialized-model inference. Additional checks verify instructor split separation, finite features, valid output tiers and exact reproduction of evaluated-model predictions after reload.

## Limits and responsible use

- Course means confound course characteristics, instructor allocation and learner mix. Adjustment is contextual normalization, not causal deconfounding. Sparse course references are not regularized.
- Neither the composite weights nor the tier boundaries are externally validated. Shrinkage is heuristic rather than a fitted hierarchical posterior.
- Batch counts provide imperfect evidence strength; learner counts and within-batch variance are absent. Standardized dispersion can hide learner subgroup disparities.
- Resampling intervals condition on the chosen policy and course references. They ignore reference uncertainty, learner-level uncertainty, time dependence and selection bias.
- Probabilities are not calibrated. Review flags (`max_probability < 0.6`, fewer than five batches, or unseen-course exposure) are illustrative and unvalidated.
- No timestamps, pre-assessment distributions or independent teaching rubrics are available. Prospective deployment requires additional data and validation.
- Use only for optional support/coaching exploration with contextual human review. Do not use for employment decisions, compensation or termination.

## GitHub repository

Repository: https://github.com/ayaan2304/beyond-ratings-instructor-effectiveness

The repository contains the executed notebook, source code, supplied dataset, saved research models and generated outputs. No redistribution license was provided for the supplied data; this project does not assign one. The repository's visibility is controlled by its owner.

After cloning, follow the run and prediction instructions above. Retraining rewrites local model/output files; review changes before committing them.
