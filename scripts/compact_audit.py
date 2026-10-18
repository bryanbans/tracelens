"""Lossless UI artifact packing; does not change labels, scores, thresholds, or model weights."""
import gzip
import argparse
import json
from pathlib import Path
from tracelens.data import sha256,save_json

parser=argparse.ArgumentParser(description='Pack the raw audit trace export for the UI without changing numerical scores.')
parser.add_argument('--artifacts',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts'/'validated')
root=parser.parse_args().artifacts
raw=json.loads((root/'traces.json').read_text(encoding='utf-8'))
catalog={};models=list(raw[0]['scores']);splits=list(dict.fromkeys(t['split'] for t in raw));windows=[]
for trace in raw:
    ids=[];mask=0
    for i,event in enumerate(trace['events']):
        text=event['template']
        if text not in catalog:catalog[text]=len(catalog)
        ids.append(catalog[text]);mask|=event['label']<<i
    windows.append([splits.index(trace['split']),trace['start_line'],trace['label'],ids,mask,[trace['scores'][m] for m in models]])
packed={'format':'compact-v2','templates':list(catalog),'models':models,'splits':splits,'windows':windows}
with gzip.open(root/'traces.json.gz','wt',encoding='utf-8',compresslevel=6) as out:json.dump(packed,out,separators=(',',':'),allow_nan=False)
with gzip.open(root/'traces.json.gz','rt',encoding='utf-8') as source:round_trip=json.load(source)
assert len(round_trip['windows'])==len(raw)
for index in (0,len(raw)//2,len(raw)-1):
    row=round_trip['windows'][index];trace=raw[index]
    assert [round_trip['templates'][i] for i in row[3]]==[e['template'] for e in trace['events']]
    assert row[5]==[trace['scores'][m] for m in models]
print(f'Packed {len(raw)} windows and {len(catalog)} unique templates. All numerical scores retained.')
