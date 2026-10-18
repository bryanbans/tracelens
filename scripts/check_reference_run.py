"""Cross-check released metrics, counts, and live inference on real BGL-derived artifacts."""
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from tracelens.api import create_app
from tracelens.evaluation import metrics

root = Path(__file__).resolve().parents[1]
artifacts = root / 'artifacts' / 'portfolio'
report = json.loads((artifacts / 'report.json').read_text(encoding='utf-8'))
traces = json.loads((artifacts / 'traces.json').read_text(encoding='utf-8'))
labels = np.array([t['label'] for t in traces])
inference_checks = 0
largest_error = 0.0
with TestClient(create_app(artifacts)) as client:
    assert client.get('/api/docs').status_code == 200
    assert client.get('/openapi.json').status_code == 200
    for entry in report['runs']:
        name = entry['id']
        scores = np.array([t['scores'][name] for t in traces])
        checked = metrics(labels, scores, entry['metrics']['threshold'])
        assert checked['confusion'] == entry['metrics']['confusion']
        assert abs(checked['average_precision'] - entry['metrics']['average_precision']) < 1e-12
        for kind, expected in [('alerts', checked['confusion']['tp'] + checked['confusion']['fp']),
                               ('missed', checked['confusion']['fn']), ('false', checked['confusion']['fp'])]:
            actual = client.get('/api/traces', params={'model':name, 'kind':kind, 'limit':1}).json()['total']
            assert actual == expected, (name, kind, actual, expected)
        # A normal window, a flagged incident, and a later window exercise reloaded checkpoints.
        for trace_id in [0, 583, 7500]:
            detail = client.get(f'/api/traces/{trace_id}', params={'model':name}).json()
            decomposition = float(np.mean(detail['event_scores'][1:]))
            expected = detail['scores'][name]
            assert np.isclose(decomposition, expected, rtol=2e-5, atol=2e-6)
            content = '\n'.join(e['template'] for e in detail['events'])
            response = client.post('/api/analyze', json={'text':content,'model':name})
            assert response.status_code == 200
            score = response.json()['windows'][0]['score']
            error = abs(score - expected)
            largest_error = max(largest_error, error)
            assert np.isclose(score, expected, rtol=2e-5, atol=2e-6), (name, trace_id, score, expected)
            inference_checks += 1
print(json.dumps({'models_checked':len(report['runs']), 'real_sequence_round_trips':inference_checks,
                  'maximum_absolute_score_error':largest_error, 'all_metrics_and_filter_counts_match':True}, indent=2))
