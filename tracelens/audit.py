"""Two-stage temporal audit. Develop/freeze first; unlock audit predictions separately."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

from .calibration import aggregate, choose_threshold, f2_from_metrics, select_candidate, paired_block_intervals
from .data import normalize, read_lines, save_json, sha256
from .evaluation import metrics, bootstrap_ap
from .models import SequenceModel
from .train import score_model
from .semantic import TextCNN,encode_groups,score_text


def ranges_from_archive(path, ranges, vocab, window_size=20, include_events=False):
    """Stream only requested ranges; no extraction and no giant Python record list."""
    result={}
    for name,(lo,hi) in ranges.items():
        result[name]={"tokens":[],"labels":[],"groups":[],"bounds":[lo,hi]}
    stop=max(hi for lo,hi in ranges.values())
    if window_size < 1 or any(lo < 0 or hi <= lo for lo, hi in ranges.values()):
        raise ValueError('Window size and range boundaries must be valid.')
    seen = 0
    pending={k:[] for k in ranges}
    with zipfile.ZipFile(path) as z, z.open('BGL.log') as binary:
        for i,line in enumerate(io.TextIOWrapper(binary,encoding='utf-8',errors='replace')):
            if i>=stop: break
            seen = i + 1
            for name,(lo,hi) in ranges.items():
                if lo<=i<hi:
                    parts=line.rstrip().split(maxsplit=9)
                    if len(parts)<9 or not parts[1].isdigit(): raise ValueError(f'Malformed line {i+1}')
                    template=normalize(parts[9]) if len(parts)==10 else '<empty>'
                    token=vocab.get(template,0)
                    row={"line":i+1,"template":template,"token":token,"known":bool(token),"label":int(parts[0]!='-')}
                    pending[name].append(row)
                    if len(pending[name])==window_size:
                        group=pending[name]; pending[name]=[]
                        result[name]['tokens'].append([e['token'] for e in group])
                        result[name]['labels'].append(int(any(e['label'] for e in group)))
                        if include_events: result[name]['groups'].append(group)
                    break
    if seen < stop:
        raise ValueError(f'Archive ended at {seen} records; requested {stop}.')
    for name,value in result.items():
        value['tokens']=np.array(value['tokens'],dtype=np.int64)
        value['labels']=np.array(value['labels'],dtype=np.int64)
        value['discarded_tail_records']=len(pending[name])
        if not len(value['tokens']): raise ValueError(f'Empty split: {name}')
    return result


def checkpoint_losses(reference, config, datasets):
    result={}
    for family in config['families']:
        seed_losses=[]
        for seed in config['seeds']:
            path=(reference.parent/'semantic' if family=='semantic' else reference)/f'{family}-s{seed}.pt'
            payload=torch.load(path,map_location='cpu',weights_only=True)
            if family=='semantic':
                word_vocab=json.loads((path.parent/'word_vocab.json').read_text(encoding='utf-8'))
                net=TextCNN(payload['vocab_size']);net.load_state_dict(payload['state_dict'])
                values={name:score_text(net,encode_groups(d['groups'],word_vocab))[1] for name,d in datasets.items()}
            else:
                net=SequenceModel(payload['vocab_size'],**payload['model_config']); net.load_state_dict(payload['state_dict'])
                values={name:score_model(net,d['tokens'])[1] for name,d in datasets.items()}
            seed_losses.append(values)
            if family=='lstm' and seed==17: result['legacy']=values
        result[family]={name:np.mean([v[name] for v in seed_losses],axis=0) for name in datasets}
    return result


def develop(data, reference, output, plan_path):
    torch.set_num_threads(4)
    config=json.loads(plan_path.read_text(encoding='utf-8')); output.mkdir(parents=True,exist_ok=True)
    if (output/'frozen.json').exists(): raise ValueError('Refusing to overwrite a frozen run. Use a new directory.')
    vocab=json.loads((reference/'vocab.json').read_text(encoding='utf-8'))
    development={k:config[k] for k in ('development_calibration','development_selection','deployment_calibration')}
    datasets=ranges_from_archive(data,development,vocab,config['window_size'],True)
    losses=checkpoint_losses(reference,config,datasets)
    candidates=[]
    for family in config['families']:
        for mode in (['max'] if family=='semantic' else config['aggregation_candidates']):
            cal=aggregate(losses[family]['development_calibration'],mode)
            selection=aggregate(losses[family]['development_selection'],mode)
            for policy in config['threshold_candidates']:
                threshold=choose_threshold(cal,datasets['development_calibration']['labels'],policy,config['normal_fpr_budget'])
                candidates.append({'id':f'{family}-{mode}-{policy}','family':family,'aggregation':mode,'policy':policy,
                                   'development_threshold':threshold,
                                   'selection_metrics':metrics(datasets['development_selection']['labels'],selection,threshold)})
    chosen,feasible=select_candidate(candidates,config['normal_fpr_budget'])
    deploy_scores=aggregate(losses[chosen['family']]['deployment_calibration'],chosen['aggregation'])
    threshold=choose_threshold(deploy_scores,datasets['deployment_calibration']['labels'],chosen['policy'],config['normal_fpr_budget'])
    old=json.loads((reference/'report.json').read_text(encoding='utf-8'))
    legacy=next(r for r in old['runs'] if r['id']=='lstm-s17')
    calibration={
        'legacy':sorted(aggregate(losses['legacy']['deployment_calibration'])[datasets['deployment_calibration']['labels']==0].tolist()),
        'calibrated':sorted(deploy_scores[datasets['deployment_calibration']['labels']==0].tolist())}
    # Isolate recalibration from the ensemble/scoring changes on the same frozen legacy model.
    legacy_cal=aggregate(losses['legacy']['deployment_calibration'])
    recalibrated_threshold=choose_threshold(legacy_cal,datasets['deployment_calibration']['labels'],'f2_budget',config['normal_fpr_budget'])
    calibration['recalibrated']=calibration['legacy']
    checkpoint_names=[f'{chosen["family"]}-s{s}.pt' for s in config['seeds']]
    for name in set(checkpoint_names+['lstm-s17.pt']):
        origin=reference.parent/'semantic' if name.startswith('semantic') else reference
        shutil.copy2(origin/name,output/name)
    if chosen['family']=='semantic':shutil.copy2(reference.parent/'semantic/word_vocab.json',output/'word_vocab.json')
    shutil.copy2(reference/'vocab.json',output/'vocab.json'); shutil.copy2(reference/'baseline.json',output/'baseline.json')
    save_json(output/'calibration.json',calibration)
    frozen={'config':config,'selected':chosen,'selection_budget_feasible':feasible,'candidates':candidates,
            'deployment_threshold':threshold,'recalibrated_legacy_threshold':recalibrated_threshold,
            'legacy_threshold':legacy['metrics']['threshold'],
            'checkpoint_files':checkpoint_names,'legacy_checkpoint':'lstm-s17.pt',
            'checkpoint_sha256':{n:sha256(output/n) for n in set(checkpoint_names+['lstm-s17.pt'])},
            'archive_sha256':sha256(data),'vocab_sha256':sha256(output/'vocab.json'),
            'code_sha256':{n:sha256(Path(__file__).parent/n) for n in ['audit.py','calibration.py','models.py','semantic.py','data.py','evaluation.py','train.py']},
            'frozen_at':datetime.now(timezone.utc).isoformat(),
            'calibration_metrics':metrics(datasets['deployment_calibration']['labels'],deploy_scores,threshold)}
    if (output/'word_vocab.json').exists():
        frozen['word_vocab_sha256']=sha256(output/'word_vocab.json')
    save_json(output/'frozen.json',frozen)
    (output/'frozen.sha256').write_text(sha256(output/'frozen.json'))
    print(json.dumps({'selected':chosen,'feasible':feasible,'deployment_calibration':frozen['calibration_metrics'],
                      'frozen_sha256':sha256(output/'frozen.json')},indent=2),flush=True)


def audit(data, output):
    torch.set_num_threads(4)
    if (output/'report.json').exists(): raise ValueError('Audit already opened. Refusing to re-score in place.')
    if not (sha256(output/'frozen.json')==(output/'frozen.sha256').read_text(encoding='utf-8').strip()):
        raise ValueError('Frozen plan changed')
    frozen=json.loads((output/'frozen.json').read_text(encoding='utf-8')); config=frozen['config']; selected=frozen['selected']
    if not (sha256(data)==frozen['archive_sha256']):
        raise ValueError('Archive changed')
    for n,h in frozen['code_sha256'].items():
        if not (sha256(Path(__file__).parent/n)==h):
            raise ValueError(f'Frozen code changed: {n}')
    for n,h in frozen['checkpoint_sha256'].items():
        if not (sha256(output/n)==h):
            raise ValueError(f'Checkpoint changed: {n}')
    if not (sha256(output/'vocab.json')==frozen['vocab_sha256']):
        raise ValueError('Event vocabulary changed')
    if selected['family']=='semantic':
        if not (frozen.get('word_vocab_sha256')):
            raise ValueError('Semantic vocabulary is not independently bound by this freeze. Use a new output; do not retrofit an opened audit.')
        if not (sha256(output/'word_vocab.json')==frozen['word_vocab_sha256']):
            raise ValueError('Word vocabulary changed')
    vocab=json.loads((output/'vocab.json').read_text(encoding='utf-8'))
    data_splits=ranges_from_archive(data,config['audit_ranges'],vocab,config['window_size'],True)
    # Only selected family and the old comparator are scored after the freeze.
    losses={}
    for name in frozen['checkpoint_sha256']:
        payload=torch.load(output/name,map_location='cpu',weights_only=True)
        if payload.get('model_type')=='semantic':
            net=TextCNN(payload['vocab_size']);net.load_state_dict(payload['state_dict'])
            word_vocab=json.loads((output/'word_vocab.json').read_text(encoding='utf-8'))
            losses[name]={k:score_text(net,encode_groups(d['groups'],word_vocab))[1] for k,d in data_splits.items()}
        else:
            net=SequenceModel(payload['vocab_size'],**payload['model_config']);net.load_state_dict(payload['state_dict'])
            losses[name]={k:score_model(net,d['tokens'])[1] for k,d in data_splits.items()}
    all_labels=[]; all_scores={k:[] for k in ('legacy','recalibrated','calibrated')}; traces=[]; audits={}
    thresholds={'legacy':frozen['legacy_threshold'],'recalibrated':frozen['recalibrated_legacy_threshold'],'calibrated':frozen['deployment_threshold']}
    for split,d in data_splits.items():
        legacy_scores=aggregate(losses['lstm-s17.pt'][split])
        new_scores=aggregate(np.mean([losses[n][split] for n in frozen['checkpoint_files']],axis=0),selected['aggregation'])
        scores={'legacy':legacy_scores,'recalibrated':legacy_scores,'calibrated':new_scores}
        audits[split]={'first_line':d['bounds'][0]+1,'last_line':d['bounds'][1],
                       'windows':len(d['labels']),'anomalous_windows':int(d['labels'].sum()),
                       'discarded_tail_records':d['discarded_tail_records'],
                       'unknown_event_fraction':float((d['tokens']==0).mean()),
                       'metrics':{k:metrics(d['labels'],v,thresholds[k]) for k,v in scores.items()},
                       'paired_intervals':paired_block_intervals(d['labels'],legacy_scores>thresholds['legacy'],new_scores>thresholds['calibrated'],**config['bootstrap'])}
        all_labels.extend(d['labels'].tolist())
        for k in scores: all_scores[k].extend(scores[k].tolist())
        for i,group in enumerate(d['groups']):
            traces.append({'id':len(traces),'split':split,'start_line':group[0]['line'],'end_line':group[-1]['line'],
                           'label':int(d['labels'][i]),'events':group,'scores':{k:float(v[i]) for k,v in scores.items()}})
    entries=[]
    for key,name in [('legacy','Original detector'),('recalibrated','Recalibration only'),('calibrated','Validated ensemble')]:
        m=metrics(all_labels,all_scores[key],thresholds[key]);m['f2']=f2_from_metrics(m)
        files=frozen['checkpoint_files'] if key=='calibrated' else ['lstm-s17.pt']
        mode=selected['aggregation'] if key=='calibrated' else 'mean'
        entries.append({'id':key,'name':name,'architecture':selected['family'] if key=='calibrated' else 'lstm',
                        'aggregation':mode,'checkpoint':files[0],'checkpoints':files,'seed':None,'parameters':None,
                        'model_type':'semantic' if key=='calibrated' and selected['family']=='semantic' else 'sequence',
                        'history':[],'metrics':m,'description':f'{len(files)} checkpoint(s), {mode} target surprise; frozen threshold.'})
    report={'schema_version':2,'created_at':datetime.now(timezone.utc).isoformat(),'selected_model':'calibrated',
            'dataset':{'name':'Loghub Â· Blue Gene/L','source':'https://github.com/logpai/loghub/tree/master/BGL',
                       'lines':4747963,'vocab_size':len(vocab),'window_size':20,'sha256':frozen['archive_sha256'],
                       'splits':{k:{'records':v[1]-v[0],'first_line':v[0]+1,'last_line':v[1]} for k,v in config['audit_ranges'].items()}},
            'config':config,'freeze_sha256':sha256(output/'frozen.json'),'frozen_at':frozen['frozen_at'],
            'selection':selected,'selection_budget_feasible':frozen['selection_budget_feasible'],
            'runs':entries,'audits':audits,
            'protocol':'Frozen historical weights/vocabulary; disjoint temporal development selection, deployment calibration, and prediction-blind audits; 1% normal-FPR calibration budget, no point adjustment; no inference labels.',
            'disclosure':config['disclosure'],
            'environment':{'python':__import__('platform').python_version(),'torch':str(torch.__version__),'device':'cpu'}}
    save_json(output/'report.json',report)
    (output/'traces.json').write_text(json.dumps(traces,separators=(',',':'),allow_nan=False),encoding='utf-8')
    manifest={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}
    save_json(output/'manifest.json',{'algorithm':'sha256','files':manifest})
    print(json.dumps({'runs':entries,'audits':audits},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['develop','audit']);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=Path('artifacts/portfolio'));parser.add_argument('--output',type=Path,default=Path('artifacts/validated'))
    parser.add_argument('--plan',type=Path,default=Path('configs/validation-v2.json'));args=parser.parse_args()
    if args.stage=='develop': develop(args.data,args.reference,args.output,args.plan)
    else: audit(args.data,args.output)


if __name__=='__main__':main()
