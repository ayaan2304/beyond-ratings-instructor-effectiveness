"""Reproducible, course-aware instructor analysis; no API or LLM calls."""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

SEED = 42
LEVELS = ['Low', 'Medium', 'High']
OUTCOMES = ['completion_rate', 'avg_score_improvement', 'avg_quiz_score']
ENGAGEMENT = ['avg_watch_time', 'assignment_submission_rate', 'forum_activity_rate']
FEEDBACK = ['avg_feedback_score']
METRICS = OUTCOMES + ENGAGEMENT + FEEDBACK
FEATURES = [f'{m}_{s}' for m in ENGAGEMENT for s in ['mean', 'std']]
FEATURES += ['n_batches', 'n_courses']


def validate(df):
    """Fail explicitly on invalid records rather than silently deleting them."""
    required = ['batch_id', 'instructor_id', 'course_id'] + METRICS + ['dropout_rate', 'feedback_response_rate']
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if df[required].isna().any().any():
        raise ValueError('Missing data found: review before training/scoring.')
    if df.batch_id.duplicated().any():
        raise ValueError('Batch IDs must be unique.')
    for m in ENGAGEMENT + ['completion_rate', 'dropout_rate', 'feedback_response_rate']:
        if not df[m].between(0, 1).all():
            raise ValueError(f'Out-of-range rate: {m}')
    if not df.avg_feedback_score.between(1, 5).all():
        raise ValueError('Feedback outside 1–5.')
    # Assessment scales are not documented: do not invent 0–100 bounds.
    if not np.isfinite(df[METRICS + ['dropout_rate', 'feedback_response_rate']].to_numpy()).all():
        raise ValueError('Non-finite numeric measurements.')


def fit_reference(df, adjusted=True):
    """Training-only course baselines, residual scale and fixed score origin."""
    center = df[METRICS].mean()
    course = df.groupby('course_id')[METRICS].mean()
    residual = df[METRICS] - center
    if adjusted:
        baseline = course.reindex(df.course_id).set_axis(df.index)
        residual = df[METRICS] - baseline
    scale = residual.std(ddof=0).replace(0, 1)
    return {'center': center, 'course': course, 'scale': scale, 'adjusted': adjusted,
            'origin': 50.0, 'score_unit': 10.0}


def transform_batches(df, ref):
    """Unseen courses use training global means; held-out data never fit baselines."""
    baseline = pd.DataFrame(np.tile(ref['center'].to_numpy(), (len(df), 1)), index=df.index, columns=METRICS)
    if ref['adjusted']:
        baseline = ref['course'].reindex(df.course_id).set_axis(df.index).fillna(ref['center'])
    z = (df[METRICS] - baseline) / ref['scale']
    z['instructor_id'] = df.instructor_id.to_numpy()
    z['course_id'] = df.course_id.to_numpy()
    return z


def aggregate(df, ref):
    """Equal batch weights: learner counts are unavailable. SD preserves consistency."""
    z = transform_batches(df, ref)
    result = z.groupby('instructor_id')[METRICS].agg(['mean', 'std'])
    result.columns = ['_'.join(c) for c in result.columns]
    result = result.fillna(0)  # Single-batch SD undefined; count preserves that distinction.
    result['n_batches'] = z.groupby('instructor_id').size()
    result['n_courses'] = z.groupby('instructor_id').course_id.nunique()
    return result


def score(a, ref, weights=(.7, .2, .1), k=5):
    """A relative index, not a probability or a validated 0–100 teaching rating."""
    outcome = .45*a.completion_rate_mean + .40*a.avg_score_improvement_mean + .15*a.avg_quiz_score_mean
    engagement = a[[f'{m}_mean' for m in ENGAGEMENT]].mean(axis=1)
    feedback = a.avg_feedback_score_mean
    composite = weights[0]*outcome + weights[1]*engagement + weights[2]*feedback
    reliability = a.n_batches / (a.n_batches+k)
    return ref['origin'] + ref['score_unit']*reliability*composite


def fit_policy(df, adjusted=True, weights=(1., 0., 0.), k=5):
    ref = fit_reference(df, adjusted)
    a = aggregate(df, ref)
    s = score(a, ref, weights, k)
    cuts = np.quantile(s, [1/3, 2/3])
    if cuts[0] >= cuts[1]:
        raise ValueError('Scores cannot support three tiers.')
    return {'reference': ref, 'weights': weights, 'k': k, 'cuts': cuts}


def apply_policy(df, policy):
    a = aggregate(df, policy['reference'])
    s = score(a, policy['reference'], policy['weights'], policy['k'])
    y = pd.Series(np.array(LEVELS)[np.digitize(s, policy['cuts'])], index=a.index, name='tier')
    return a, s, y


def candidates():
    return {
        'Dummy': make_pipeline(SimpleImputer(), DummyClassifier(strategy='prior')),
        'Logistic': make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=1., max_iter=2000, class_weight='balanced', random_state=SEED)),
        'RandomForest': make_pipeline(SimpleImputer(), RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=5, class_weight='balanced', random_state=SEED, n_jobs=1))
    }


def evaluate(y, pred):
    return {'macro_f1': float(f1_score(y, pred, labels=LEVELS, average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred))}


def train_experiment(df, output_dir='outputs', model_dir='models'):
    """Select on inner folds; evaluate once on unseen instructors. No chronology implied."""
    out, models = Path(output_dir), Path(model_dir)
    out.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    validate(df)
    ids = np.sort(df.instructor_id.unique())
    train_ids, test_ids = train_test_split(ids, test_size=.25, random_state=SEED)
    train = df[df.instructor_id.isin(train_ids)]
    test = df[df.instructor_id.isin(test_ids)]
    rows = []
    # Each fold rebuilds course baselines, scales, shrinkage labels and boundaries.
    for fold, (fit_idx, val_idx) in enumerate(KFold(3, shuffle=True, random_state=SEED).split(train_ids), 1):
        fit = train[train.instructor_id.isin(train_ids[fit_idx])]
        val = train[train.instructor_id.isin(train_ids[val_idx])]
        p = fit_policy(fit)
        a, _, y = apply_policy(fit, p); b, _, v = apply_policy(val, p)
        for name, estimator in candidates().items():
            estimator.fit(a[FEATURES], y)
            rows.append({'fold': fold, 'model': name, **evaluate(v, estimator.predict(b[FEATURES]))})
    cv = pd.DataFrame(rows)
    summary = cv.groupby('model')[['macro_f1', 'balanced_accuracy']].agg(['mean', 'std'])
    winner = cv.groupby('model').macro_f1.mean().idxmax()  # Dummy may legitimately win.
    policy = fit_policy(train)
    a, s, y = apply_policy(train, policy); b, t, v = apply_policy(test, policy)
    selected = candidates()[winner].fit(a[FEATURES], y)
    pred = selected.predict(b[FEATURES])
    result = {'selected_model': winner, 'train_instructors': len(a), 'test_instructors': len(b),
              'test': evaluate(v, pred), 'target': 'course-adjusted outcome-only policy tier',
              'interpretation': 'Concurrent association with a constructed proxy; not future or causal effectiveness.'}
    report = pd.DataFrame(classification_report(v, pred, labels=LEVELS, output_dict=True, zero_division=0)).T
    predictions = pd.DataFrame({'actual_tier': v, 'predicted_tier': pred, 'outcome_score': t})
    prob = selected.predict_proba(b[FEATURES])
    for j, level in enumerate(selected.classes_):
        predictions[f'probability_{level}'] = prob[:, j]
    predictions['max_probability'] = prob.max(axis=1)
    predictions['review_flag'] = (predictions.max_probability < .6) | (b.n_batches < 5)
    # A review heuristic, NOT calibrated uncertainty or a validated rejection rule.
    importance = permutation_importance(selected, b[FEATURES], v, scoring='f1_macro', n_repeats=20, random_state=SEED)
    imp = pd.DataFrame({'feature': FEATURES, 'importance': importance.importances_mean, 'std': importance.importances_std}).sort_values('importance', ascending=False)
    # Explicitly circular auxiliary task: reconstruct full composite tiers.
    full = fit_policy(train, weights=(.7, .2, .1))
    fa, _, fy = apply_policy(train, full); fb, _, fv = apply_policy(test, full)
    full_features = [f'{m}_mean' for m in METRICS] + ['n_batches']
    reconstruct = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(fa[full_features], fy)
    result['circular_reconstruction_test'] = evaluate(fv, reconstruct.predict(fb[full_features]))
    result['circular_warning'] = 'These inputs define their own target. This result is formula reproduction only.'
    cv.to_csv(out/'cv_folds.csv', index=False); summary.to_csv(out/'cv_summary.csv')
    report.to_csv(out/'classification_report.csv'); predictions.to_csv(out/'heldout_predictions.csv')
    imp.to_csv(out/'permutation_importance.csv', index=False)
    pd.DataFrame(confusion_matrix(v, pred, labels=LEVELS), index=LEVELS, columns=LEVELS).to_csv(out/'confusion_matrix.csv')
    (out/'metrics.json').write_text(json.dumps(result, indent=2))
    pd.DataFrame({'instructor_id': ids, 'split': ['train' if i in train_ids else 'test' for i in ids]}).to_csv(out/'split_manifest.csv', index=False)
    with (models/'evaluated_model.pkl').open('wb') as f:
        pickle.dump({'model': selected, 'policy': policy, 'features': FEATURES, 'selected_name': winner}, f)
    # Refit for research inference on all supplied instructors; does not change holdout metrics.
    final_policy = fit_policy(df)
    all_a, all_s, all_y = apply_policy(df, final_policy)
    final = candidates()[winner].fit(all_a[FEATURES], all_y)
    with (models/'instructor_model.pkl').open('wb') as f:
        pickle.dump({'model': final, 'policy': final_policy, 'features': FEATURES, 'selected_name': winner}, f)
    return result, cv, predictions, imp


def predict_batches(df, artifact='models/instructor_model.pkl'):
    """Research inference: input is batch rows, output one prediction per instructor."""
    validate(df)
    with open(artifact, 'rb') as f:
        bundle = pickle.load(f)  # Load only your own trusted artifact.
    a, _, _ = apply_policy(df, bundle['policy'])
    model = bundle['model']; p = model.predict_proba(a[bundle['features']])
    result = pd.DataFrame({'predicted_tier': model.predict(a[bundle['features']]), 'n_batches': a.n_batches}, index=a.index)
    for j, level in enumerate(model.classes_):
        result[f'probability_{level}'] = p[:, j]
    result['review_flag'] = (p.max(axis=1) < .6) | (a.n_batches.to_numpy() < 5)
    known = bundle['policy']['reference']['course'].index
    unseen = df.assign(unseen=~df.course_id.isin(known)).groupby('instructor_id').unseen.mean()
    result['unseen_course_fraction'] = unseen
    result['review_flag'] |= unseen > 0
    return result
