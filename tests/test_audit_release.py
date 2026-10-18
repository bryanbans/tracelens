"""Validate the packed real-data release without reopening/tuning the audit."""
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from tracelens.api import create_app
from tracelens.data import sha256
from tracelens.evaluation import metrics

ROOT=Path(__file__).parents[1]/'artifacts'/'validated'


@pytest.fixture(scope='module')
def release():
    if not (ROOT/'report.json').exists():pytest.skip('Validated release not bundled')
    report=json.loads((ROOT/'report.json').read_text())
    with gzip.open(ROOT/'traces.json.gz','rt',encoding='utf-8') as source:packed=json.load(source)
    return report,packed


def test_frozen_pipeline_and_audit_boundaries(release):
    report,_=release
    assert report['freeze_sha256']==sha256(ROOT/'frozen.json')
    frozen=json.loads((ROOT/'frozen.json').read_text())
    assert frozen['config']['deployment_calibration'][1]<=min(v[0] for v in frozen['config']['audit_ranges'].values())
    for name,digest in frozen['checkpoint_sha256'].items():assert sha256(ROOT/name)==digest
    for name,digest in frozen['code_sha256'].items():assert sha256(ROOT/'frozen_sources'/name)==digest
    assert (ROOT/'word_vocab.json').read_bytes()==(ROOT.parent/'semantic/word_vocab.json').read_bytes()
    assert report['audits']['confirmation']['discarded_tail_records']==3


def test_all_packed_audit_scores_reproduce_reported_metrics(release):
    report,packed=release
    y=np.array([r[2] for r in packed['windows']])
    for column,model in enumerate(packed['models']):
        entry=next(r for r in report['runs'] if r['id']==model)
        scores=np.array([r[5][column] for r in packed['windows']])
        result=metrics(y,scores,entry['metrics']['threshold'])
        assert result['confusion']==entry['metrics']['confusion']
        assert result['average_precision']==pytest.approx(entry['metrics']['average_precision'],abs=1e-12)


def test_real_ensemble_inference_and_split_filters(release):
    report,_=release
    with TestClient(create_app(ROOT)) as client:
        assert 'id="page-overview"' in client.get('/').text
        for entry in report['runs']:
            model=entry['id']
            detail=client.get(f'/api/traces/9332?model={model}').json()
            text='\n'.join(e['template'] for e in detail['events'])
            result=client.post('/api/analyze',json={'model':model,'text':text}).json()
            assert result['windows'][0]['score']==pytest.approx(detail['scores'][model],abs=3e-6)
            response=client.get('/api/traces',params={'model':model,'split':'confirmation','kind':'missed','limit':1}).json()
            assert response['total']==report['audits']['confirmation']['metrics'][model]['confusion']['fn']


def test_budget_failure_is_reported_not_hidden(release):
    report,_=release
    assert all(a['metrics']['calibrated']['normal_false_positive_rate']>report['config']['normal_fpr_budget'] for a in report['audits'].values())
    assert all(a['metrics']['calibrated']['recall']>a['metrics']['legacy']['recall'] for a in report['audits'].values())
