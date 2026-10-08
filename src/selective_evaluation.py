"""Validation-only risk/coverage diagnostics; no threshold fitting or locked-test access."""
import argparse
import json
from pathlib import Path
import numpy as np


def selective_summary(labels, predictions, accepted):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    accepted = np.asarray(accepted, dtype=bool)
    if labels.ndim != 1 or not len(labels) or predictions.shape != labels.shape or accepted.shape != labels.shape:
        raise ValueError('Expected equal, nonempty one-dimensional arrays')
    if not np.isin(labels, [0, 1, 2]).all() or not np.isin(predictions, [0, 1, 2]).all():
        raise ValueError('Unknown class index')
    n = int(accepted.sum())
    correct = labels == predictions
    # Empty accepted sets have undefined accuracy/risk, never a misleading 100%.
    return {'n_images': len(labels), 'accepted': n, 'reviewed': int((~accepted).sum()),
            'coverage': float(accepted.mean()),
            'accepted_accuracy': float(correct[accepted].mean()) if n else None,
            'selective_risk': float((~correct[accepted]).mean()) if n else None,
            'errors_accepted': int((accepted & ~correct).sum()),
            'errors_referred': int((~accepted & ~correct).sum()),
            'per_class_coverage': {str(c): float(accepted[labels == c].mean()) if (labels == c).any() else None for c in range(3)}}


def evaluate(run, output):
    from src.config import CLASS_NAMES, resolve_path
    from src.manifest import load_bundle
    from src.predict import Predictor
    from src.artifacts import atomic_json
    run, output = resolve_path(run), resolve_path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    frame, metadata = load_bundle(run / 'dataset')
    validation = frame[frame.split.eq('val')].copy()
    predictor = Predictor(run)
    details = [predictor.predict(path) for path in validation.filepath]
    labels = validation.label_idx.to_numpy()
    predicted = np.array([CLASS_NAMES.index(row['label']) for row in details])
    confidences = np.array([row['confidence'] for row in details])
    fixed = predictor.review_threshold
    policies = {
        'classify_all': np.ones(len(labels), dtype=bool),
        'confidence_only_fixed': confidences >= fixed,
        'confidence_quality_consistency_fixed': np.array([not row['needs_review'] for row in details])}
    result = {'evaluation_split':'validation', 'dataset_fingerprint':metadata['fingerprint'],
              'model_run':str(run), 'classes':CLASS_NAMES, 'fixed_threshold':fixed,
              'threshold_fitted':False, 'locked_test_read':False,
              'n_groups':int(validation.group_id.nunique()),
              'policies':{name:selective_summary(labels,predicted,mask) for name,mask in policies.items()},
              'risk_coverage_curve':[{'threshold':float(t), **selective_summary(labels,predicted,confidences>=t)} for t in np.linspace(0,1,21)],
              'limitations':['Validation also drives early stopping and model selection.',
                             'Six validation images from two sessions cannot validate safety or generalization.',
                             'Threshold grid is descriptive, not an optimization or test result.'],
              'predictions':[{'filepath':row.filepath,'actual_label':row.label, **prediction} for row,prediction in zip(validation.itertuples(),details)]}
    atomic_json(output,result)
    print(json.dumps(result['policies'],indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    evaluate(args.run,args.output)
