"""Run a fixed small search; select with validation before touching test metrics."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from src.config import ExperimentConfig, OUTPUTS_DIR, ROOT_DIR, resolve_path
from src.manifest import relative, load_bundle
from src.artifacts import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--samples-per-class',type=int,default=300)
    args=parser.parse_args()
    _,dataset=load_bundle(args.dataset)
    root=resolve_path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    base=ExperimentConfig(epochs=args.epochs,train_samples_per_class=args.samples_per_class)
    configs={
        'baseline':replace(base,architecture='baseline'),
        'regularized_lr001':base,
        'regularized_lr0003':replace(base,learning_rate=.0003,dropout=.5),
    }
    records=[]
    for name,config in configs.items():
        config_path=root/(name+'.json')
        config_path.write_text(json.dumps(config.to_dict(),indent=2))
        run=root/name
        subprocess.run([sys.executable,'-m','src.train','--dataset',str(resolve_path(args.dataset)),
                        '--run',str(run),'--config',str(config_path)],cwd=ROOT_DIR,check=True)
        meta=json.loads((run/'run.json').read_text())
        records.append({'name':name,'run':relative(run),'validation':meta['validation'],
                        'train_val_accuracy_gap':meta['train_val_accuracy_gap']})
    winner=max(records,key=lambda r:(r['validation']['macro_f1'],-r['validation']['loss']))
    selection={'run':winner['run'],'selection_rule':'validation macro F1, then validation loss',
               'dataset_fingerprint':dataset['fingerprint'],'mode':dataset['mode'],
               'candidates':records}
    (root/'selection.json').write_text(json.dumps(selection,indent=2))
    # Test is read only after the winner has been frozen. Never iterate on test.
    for name in dict.fromkeys(['baseline',winner['name']]):
        subprocess.run([sys.executable,'-m','src.evaluate','--run',str(root/name)],cwd=ROOT_DIR,check=True)
    selection['test']={name:json.loads((root/name/'evaluation.json').read_text())
                       for name in dict.fromkeys(['baseline',winner['name']])}
    (root/'comparison.json').write_text(json.dumps(selection,indent=2))
    OUTPUTS_DIR.mkdir(parents=True,exist_ok=True)
    atomic_json(OUTPUTS_DIR/'selected_run.json',{'run':winner['run'],'mode':dataset['mode']})
    print(json.dumps({'selected':winner['run'],'mode':dataset['mode']},indent=2))


if __name__=='__main__':
    main()
