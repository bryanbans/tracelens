"""Post-hoc uncertainty sensitivity; no fitting, selection or threshold changes."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from tracelens.calibration import paired_block_intervals
from tracelens.data import sha256, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/validated'))
    parser.add_argument('--output', type=Path, default=Path('docs/audit-sensitivity.json'))
    args = parser.parse_args()
    report = json.loads((args.artifacts/'report.json').read_text(encoding='utf-8'))
    with gzip.open(args.artifacts/'traces.json.gz', 'rt', encoding='utf-8') as f:
        packed = json.load(f)
    thresholds = {r['id']: r['metrics']['threshold'] for r in report['runs']}
    old_column = packed['models'].index('legacy')
    new_column = packed['models'].index('calibrated')
    output = {'purpose': 'Post-hoc sensitivity of conditional intervals, not new validation or model selection.',
              'report_sha256': sha256(args.artifacts/'report.json'),
              'freeze_sha256': sha256(args.artifacts/'frozen.json'),
              'seed': 2026, 'repetitions': 500, 'ranges': {}}
    for i, name in enumerate(packed['splits']):
        rows = [r for r in packed['windows'] if r[0] == i]
        y = np.array([r[2] for r in rows])
        old = np.array([r[5][old_column] > thresholds['legacy'] for r in rows])
        new = np.array([r[5][new_column] > thresholds['calibrated'] for r in rows])
        output['ranges'][name] = {str(b): paired_block_intervals(y, old, new, block_size=b)
                                  for b in (100, 250, 1000, 2500)}
    save_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
