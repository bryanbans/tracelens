import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tracelens.api import create_app
from tracelens.train import run


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp('pipeline')
    source = root / 'BGL.log'
    # Generated fixture exercises plumbing. It is never portfolio benchmark evidence.
    lines = []
    for i in range(400):
        label = 'ALERT' if i % 47 == 0 else '-'
        content = 'adapter link failed' if label != '-' else ['job started', 'job accepted', 'job completed'][i % 3]
        lines.append(f'{label} {1117838570+i} 2005.06.03 R01 time R01 RAS KERNEL INFO {content}\n')
    source.write_text(''.join(lines), encoding='utf-8')
    config = json.loads((Path(__file__).parent.parent / 'configs/portfolio.json').read_text())
    config.update(max_lines=400, window_size=10, epochs=2, seeds=[42], embedding=4, hidden=8, threads=1)
    config['experiments'] = [config['experiments'][0], config['experiments'][1]]
    output = root / 'artifacts'
    report = run(source, output, config)
    return output, report


def test_real_training_export_and_reloaded_inference_agree(built):
    output, report = built
    with TestClient(create_app(output)) as client:
        assert client.get('/api/health').json()['status'] == 'ok'
        assert client.get('/').status_code == 200
        for entry in report['runs']:
            model = entry['id']
            detail = client.get(f'/api/traces/0?model={model}').json()
            # The displayed decomposition must reproduce the saved benchmark score.
            assert np.mean(detail['event_scores'][1:]) == pytest.approx(detail['scores'][model], rel=1e-5)
            text = '\n'.join(e['template'] for e in detail['events'])
            response = client.post('/api/analyze', json={'text': text, 'model': model})
            assert response.status_code == 200
            inferred = response.json()
            assert inferred['windows'][0]['score'] == pytest.approx(detail['scores'][model], rel=1e-5)
            assert inferred['discarded_tail_lines'] == 0


def test_api_errors_filters_and_unknown_formats(built):
    output, report = built
    model = report['selected_model']
    with TestClient(create_app(output)) as client:
        assert client.get('/api/timeline?model=not-a-checkpoint').status_code == 404
        assert client.get(f'/api/traces/-1?model={model}').status_code == 404
        assert client.post('/api/analyze', json={'text':'short', 'model':model}).status_code == 422
        payload = client.post('/api/analyze', json={'text':'\n'.join(['unfamiliar unicorn process']*13), 'model':model}).json()
        assert payload['unknown_fraction'] == 1
        assert payload['discarded_tail_lines'] == 3
        assert client.get(f'/api/traces?model={model}&q=nothing-will-match').json()['total'] == 0
        assert client.get(f'/api/traces?model={model}&offset=-1').status_code == 422


def test_missing_artifacts_fail_with_recovery_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match='train'):
        create_app(tmp_path)
