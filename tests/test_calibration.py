import numpy as np
import pytest
import torch
from tracelens.calibration import choose_threshold,aggregate,select_candidate,f2_from_metrics,paired_block_intervals
from tracelens.evaluation import metrics
from tracelens.semantic import TextCNN,encode_groups,word_vocabulary,score_text


def test_budget_optimizer_matches_brute_force_with_ties():
    rng=np.random.default_rng(7)
    scores=rng.integers(0,8,size=300).astype(np.float32)
    labels=rng.integers(0,2,size=300)
    threshold=choose_threshold(scores,labels,'f2_budget',.05)
    result=metrics(labels,scores,threshold)
    feasible=[metrics(labels,scores,float(np.nextafter(float(s),-np.inf))) for s in np.unique(scores)]
    feasible=[m for m in feasible if m['normal_false_positive_rate']<=.05]
    assert result['normal_false_positive_rate']<=.05
    assert f2_from_metrics(result)==pytest.approx(max([f2_from_metrics(m) for m in feasible]+[0]))


def test_float32_scores_preserve_strict_threshold_semantics():
    scores=np.array([.2,.2,.8,.8],dtype=np.float32);labels=np.array([0,0,1,1])
    threshold=choose_threshold(scores,labels,'f2_budget',0)
    assert metrics(labels,scores,threshold)['confusion']=={'tn':2,'fp':0,'fn':0,'tp':2}


def test_selection_uses_development_only_and_respects_budget():
    def candidate(name,p,r,fpr):
        return {'id':name,'selection_metrics':{'precision':p,'recall':r,'normal_false_positive_rate':fpr,'average_precision':.5}}
    safe=candidate('safe',.8,.6,.005);unsafe=candidate('unsafe',.99,.99,.2)
    winner,feasible=select_candidate([unsafe,safe])
    assert winner is safe and feasible


def test_semantic_word_features_exclude_future_words_and_scores_match_pooling():
    group=[[{'template':'normal operation finished'}]*20]
    vocab=word_vocabulary(group)
    future=[[{'template':'unseen-signal in operation'}]*20]
    assert 'unseen' not in vocab
    x=encode_groups(future,vocab)
    assert 1 in x
    model=TextCNN(len(vocab)).eval()
    scores,events=score_text(model,x)
    assert scores[0]==pytest.approx(events[0].max())


def test_paired_bootstrap_of_identical_predictions_has_zero_difference():
    y=np.array([0,1]*100);p=y.copy()
    result=paired_block_intervals(y,p,p,repetitions=30,block_size=20)
    assert result['recall_difference_95']==[0,0]
    assert result['normal_fpr_difference_95']==[0,0]


def test_score_rules_are_explicit_and_labels_are_not_inputs():
    losses=np.array([[1,2,3,4]])
    assert aggregate(losses,'top3')[0]==3
    assert aggregate(losses,'max')[0]==4
    with pytest.raises(ValueError):aggregate(losses,'unknown')


@pytest.mark.parametrize('missing',[True,False])
def test_new_audit_refuses_unbound_or_changed_word_mapping(tmp_path,missing):
    import json
    from tracelens.audit import audit
    from tracelens.data import sha256
    source=tmp_path/'archive';source.write_bytes(b'not reached: vocabulary must fail first')
    (tmp_path/'vocab.json').write_text('{}')
    (tmp_path/'word_vocab.json').write_text('{"changed": 1}')
    frozen={'config':{},'selected':{'family':'semantic'},'archive_sha256':sha256(source),
            'code_sha256':{},'checkpoint_sha256':{},'vocab_sha256':sha256(tmp_path/'vocab.json')}
    if not missing:frozen['word_vocab_sha256']='a deliberately wrong digest'
    (tmp_path/'frozen.json').write_text(json.dumps(frozen))
    (tmp_path/'frozen.sha256').write_text(sha256(tmp_path/'frozen.json'))
    with pytest.raises(ValueError,match='vocabulary'):
        audit(source,tmp_path)
