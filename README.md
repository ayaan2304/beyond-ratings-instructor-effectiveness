# Beyond Ratings
## Course-Aware Instructor Effectiveness with Reliability and Tier Stability

A reproducible research project using the supplied 2,000-batch EdTech dataset. It combines a transparent descriptive score with a separate classical ML experiment and asks whether the resulting tiers are stable enough to support contextual coaching review.

**AI assistance disclosure:** this project was prepared with AI assistance. The original assessment explicitly prohibits LLM assistance. Use this as a learning/reference project unless your evaluator authorizes an AI-assisted submission. The analysis itself calls no LLMs, external APIs, external datasets or AutoML services.

## Public repository contents

This repository currently contains the source code and a notebook without stored cell outputs. The original CSV, trained model files, generated tables/charts and executed notebook outputs are withheld pending explicit approval to publish dataset-derived information. See OUTPUTS_NOTICE.md. The full executed package is available separately to the owner.

## How it works

1. Validate schema, numeric ranges, missingness and batch uniqueness. Inspect distributions, correlations and instructor/course coverage. Keep the original input unchanged.
2. Fit per-course metric means and residual standard deviations using development instructors only. An unseen course uses the development global metric mean.
3. Aggregate standardized batch metrics into instructor means/SDs, batch counts and distinct-course counts. Equal batch weights are necessary because cohort sizes are absent.
4. For descriptive full scoring, combine outcomes (70%), engagement (20%) and feedback (10%). Within outcomes, completion/improvement/quiz receive 45/40/15%. These are documented judgment calls.
5. Shrink the composite toward zero by `n / (n + 5)` and report `50 + 10 * shrunk_composite`. This is a relative index, not a bounded grade. Tertiles define relative tiers.
6. For the primary ML task, define an outcome-only tier and predict it using engagement means/SDs and counts. Exclude outcome ingredients, IDs and feedback. A separate circular experiment demonstrates why giving a model its label ingredients can produce misleadingly high results.
7. Compare dummy, logistic and shallow random-forest models. Refit scoring references and label cuts inside every development fold. Select by mean macro F1, then evaluate once on unseen instructors.
8. Examine permutation importance, alternative weighting/course/shrinkage policies, nearby cutoffs and conditional within-instructor batch-resampling intervals.

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

## Included files

- `Beyond_Ratings.ipynb`: self-contained analysis code and methodological explanations; no stored outputs.
- `effectiveness.py`: reusable scoring, training and prediction functions.
- `train.py` and `predict.py`: command-line entry points.
- `requirements.txt`: pinned analysis dependencies.
- `OUTPUTS_NOTICE.md`: status of dataset, trained models and results publication.

Create `data/` and provide an authorized `instructor_batches.csv` before running. The analysis recreates `models/` and `outputs/` locally. Load only trusted pickle model artifacts.

## Verification

The complete private working copy was executed in a shared Python process, schema-validated and checked for serialization round trips. The public copy removes stored outputs and data-specific recorded findings. A local run requires an authorized input CSV. No hosted inference service or production deployment is included.

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

The repository contains the source code and code-only notebook. Dataset and output publication is pending explicit approval. No redistribution license was provided for the supplied data; this project does not assign one. The repository's visibility is controlled by its owner.

After cloning, follow the run and prediction instructions above. Retraining rewrites local model/output files; review changes before committing them.
