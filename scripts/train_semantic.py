import json
import argparse
from pathlib import Path
import torch
from tracelens.audit import ranges_from_archive
from tracelens.semantic import word_vocabulary,encode_groups,train_text
from tracelens.data import save_json

torch.set_num_threads(4)
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Fit word CNNs on historical training data, with a separate checkpoint-selection range.')
parser.add_argument('--data',type=Path,default=root/'data'/'BGL.zip')
parser.add_argument('--output',type=Path,default=root/'artifacts'/'semantic')
parser.add_argument('--overwrite',action='store_true')
args=parser.parse_args()
data=args.data;output=args.output
if not data.exists():parser.error('BGL archive missing. Run tracelens download or pass --data.')
if output.exists() and any(output.glob('*.pt')) and not args.overwrite:parser.error('Checkpoints exist. Choose a fresh --output, or explicitly use --overwrite.')
output.mkdir(parents=True,exist_ok=True)
old_vocab=json.loads((root/'artifacts/portfolio/vocab.json').read_text())
splits=ranges_from_archive(data,{'train':[0,600000],'validation':[600000,1000000]},old_vocab,20,True)
vocab=word_vocabulary(splits['train']['groups'])
save_json(output/'word_vocab.json',vocab)
x={k:encode_groups(v['groups'],vocab) for k,v in splits.items()}
for seed in [17,42,73]:
 net,history=train_text(x['train'],splits['train']['labels'],x['validation'],splits['validation']['labels'],len(vocab),seed)
 torch.save({'state_dict':net.state_dict(),'vocab_size':len(vocab),'model_type':'semantic','model_config':{}},output/f'semantic-s{seed}.pt')
 save_json(output/f'history-s{seed}.json',history)
print('Semantic training complete.',flush=True)
