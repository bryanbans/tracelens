import numpy as np
import pytest

from tracelens.data import encode, fit_vocab, normalize, read_lines, split_records, windows
from tracelens.evaluation import calibrate, metrics


def test_parser_removes_label_and_metadata():
    normal = '- 1117838570 2005.06.03 R01 2005-06-03-15.42.50 R01 RAS KERNEL INFO error at 0xF00D node 123\n'
    alert = normal.replace('- ', 'KERNFAULT ', 1).replace('INFO', 'FATAL')
    rows = read_lines([normal, alert], 10)
    assert rows[0]['template'] == rows[1]['template']
    assert [r['label'] for r in rows] == [0, 1]
    assert 'kernfault' not in rows[1]['template']
    assert 'fatal' not in rows[1]['template']


@pytest.mark.parametrize('content', ['Error at 0xABCD host 192.168.2.1', 'node R02-M1-N0 failed 123', 'register deadbeef value=20', 'instruction cache parity error corrected'])
def test_normalization_is_idempotent(content):
    assert normalize(normalize(content)) == normalize(content)


def test_future_templates_cannot_enter_training_vocab():
    records = [{'line': i+1, 'template': 'old' if i < 60 else 'future', 'label': 0} for i in range(100)]
    splits = split_records(records)
    vocab = fit_vocab(splits['train'])
    assert 'future' not in vocab
    assert np.all(encode(splits['test'], vocab) == 0)
    boundaries = []
    for key in ['train', 'validation', 'test']:
        _, _, groups = windows(splits[key], vocab, 10)
        boundaries.append({r['line'] for g in groups for r in g})
    assert not boundaries[0] & boundaries[1]
    assert not boundaries[1] & boundaries[2]
    assert not boundaries[0] & boundaries[2]


def test_threshold_uses_normal_validation_only_and_ties_do_not_alert():
    threshold = calibrate([1, 2, 1000], [0, 0, 1], .99)
    assert threshold == 2
    result = metrics([0, 0, 1, 1], [1, 2, 2, 3], threshold)
    assert result['confusion'] == {'tn': 2, 'fp': 0, 'fn': 1, 'tp': 1}
    assert result['recall'] == .5


def test_missing_normal_validation_is_explicit_error():
    with pytest.raises(ValueError, match='normal'):
        calibrate([1, 2], [1, 1])
