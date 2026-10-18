"""Regression checks for review findings; frozen detector scores stay unchanged."""
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from tracelens.api import create_app
from tracelens.calibration import choose_threshold
from tracelens.data import read_lines

ROOT = Path(__file__).parents[1] / 'artifacts' / 'validated'

@pytest.fixture(scope='module')
def client():
    with TestClient(create_app(ROOT)) as c:
        yield c


def test_empty_bgl_content_is_valid_and_metadata_is_removed():
    row = read_lines(['- 111 2005.06.03 R01 time R01 RAS KERNEL INFO'], 1)[0]
    assert row['template'] == '<empty>'
    assert row['label'] == 0


@pytest.mark.parametrize('scores,labels,budget', [
    ([float('nan'), .2], [0, 1], .01),
    ([.1, .2], [0, 2], .01),
    ([.1, .2], [0], .01),
    ([.1, .2], [0, 1], -1),
    ([], [], .01),
])
def test_calibration_rejects_invalid_inputs(scores, labels, budget):
    with pytest.raises(ValueError):
        choose_threshold(scores, labels, 'f2_budget', budget)


def test_frozen_audit_rejects_override_and_invalid_filters(client):
    for params in [{'threshold': .9}, {'kind': 'typo'}, {'split': 'typo'}]:
        assert client.get('/api/traces', params={'model': 'calibrated', **params}).status_code == 422


def test_replay_no_decision_on_unencodable_input_and_source_lines_preserved(client):
    text = '\n\n'.join(['...'] * 20) + '\nextra'
    result = client.post('/api/analyze', json={'model': 'calibrated', 'text': text}).json()
    window = result['windows'][0]
    assert window['alert'] is None
    assert window['coverage']['status'] == 'unscorable'
    assert window['coverage']['empty_token_events'] == 20
    assert window['events'][-1]['line'] == 39
    assert result['discarded_tail_lines'] == 1
    assert result['ignored_blank_lines'] == 19


def test_semantic_coverage_and_truncation_are_separate_from_legacy_templates(client):
    text = '\n'.join(['instruction cache parity error corrected ' * 5] * 20)
    result = client.post('/api/analyze', json={'model': 'calibrated', 'text': text}).json()
    coverage = result['windows'][0]['coverage']
    assert coverage['truncated_events'] == 20
    assert coverage['known_word_fraction'] > 0
    assert coverage['status'] == 'domain_unverified'
    assert result['legacy_template_unknown_fraction'] == 1


def test_listing_preview_does_not_consult_ground_truth(client):
    item = client.get('/api/traces?model=calibrated&limit=1').json()['items'][0]
    detail = client.get(f"/api/traces/{item['id']}?model=calibrated").json()
    assert item['preview'] == detail['events'][0]['template']


def test_chunked_request_body_is_bounded(client):
    response = client.post('/api/analyze', content=iter([b'x' * 600000, b'x' * 600000]),
                           headers={'content-type': 'application/json'})
    assert response.status_code == 413


def test_startup_refuses_corrupt_release_files(tmp_path):
    import json
    for name in ('report.json', 'vocab.json', 'calibration.json', 'baseline.json'):
        (tmp_path/name).write_text('{}')
    (tmp_path/'manifest.json').write_text(json.dumps({'files': {'report.json': 'wrong'}}))
    with pytest.raises(ValueError, match='checksum mismatch'):
        create_app(tmp_path)


def test_archive_cannot_silently_shorten_requested_audit_range(tmp_path):
    import zipfile
    from tracelens.audit import ranges_from_archive
    path = tmp_path/'short.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('BGL.log', '\n'.join(['- 111 date node time node RAS KERNEL INFO message'] * 20))
    with pytest.raises(ValueError, match='Archive ended'):
        ranges_from_archive(path, {'test': [0, 40]}, {}, window_size=20)
