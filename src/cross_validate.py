"""Bounded three-fold group development study. Locked test never used for scoring."""
import argparse
from dataclasses import replace
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from src.config import ExperimentConfig, CLASS_NAMES, ROOT_DIR, resolve_path
from src.dataset_split import assign_group_splits, validate_splits
from src.manifest import load_bundle, save_bundle, relative
from src.evaluate import metrics

def development_folds(frame,mode,seed=42):
    development=frame[frame.split.ne('test')].copy()
    allocation=assign_group_splits(development,seed,ratios=(1/3,1/3,1/3))
    validation_groups=[set(allocation[allocation.split==part].group_id) for part in ['train','val','test']]
    locked=frame[frame.split=='test'].copy()
    outputs=[]
    for groups in validation_groups:
        fold=frame.copy()
        fold.loc[fold.split.ne('test'),'split']='train'
        fold.loc[fold.group_id.isin(groups),'split']='val'
        validate_splits(fold)
        pd.testing.assert_frame_equal(fold[fold.split=='test'],locked)
        outputs.append(fold)
    return outputs

def run_study(dataset,output,epochs=30,samples=60):
    frame,meta=load_bundle(dataset)
    root=resolve_path(output)
    root.mkdir(parents=True,exist_ok=False)
    base=ExperimentConfig(epochs=epochs,train_samples_per_class=samples)
    candidates={'reference_128':base,
                'texture_160':replace(base,image_size=160,learning_rate=.0007,dropout=.2,weight_decay=.00005)}
    protocol={'parent_dataset':meta['fingerprint'],'mode':meta['mode'],
              'candidates':{name:c.to_dict() for name,c in candidates.items()},
              'folds':3,'locked_test_sha256':sorted(frame[frame.split=='test'].sha256.tolist()),
              'promotion_rule':'Candidate OOF macro F1 >= reference + 0.02, and worst-fold F1 not lower. No automatic deployment.',
              'limitations':'Development folds also select checkpoints; not nested CV, not unbiased final accuracy.',
              'test_scoring':'Prohibited in this study; test image pixels are not loaded for model fitting or prediction.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    folders=[]
    settings={k:v for k,v in meta.items() if k not in {'fingerprint','counts','groups','unique_images','near_duplicate_merges'}}
    for idx,fold in enumerate(development_folds(frame,meta['mode'])):
        folders.append(save_bundle(fold,{**settings,'development_parent':meta['fingerprint'],
                                         'development_fold':idx,'test_usage':'locked, do not score'},root/f'data_fold{idx}'))
    summaries={}
    for name,config in candidates.items():
        records=[]
        fold_metrics=[]
        for idx,folder in enumerate(folders):
            actual=replace(config,seed=42+idx)
            config_path=root/f'{name}_fold{idx}.json'
            config_path.write_text(json.dumps(actual.to_dict(),indent=2))
            run=root/f'{name}_fold{idx}'
            subprocess.run([sys.executable,'-u','-m','src.train','--dataset',str(folder),
                            '--run',str(run),'--config',str(config_path)],cwd=ROOT_DIR,check=True)
            rows=pd.read_csv(run/'validation_predictions.csv')
            rows['fold']=idx
            records.append(rows)
            report=json.loads((run/'run.json').read_text())
            fold_metrics.append(report['validation'])
        out=pd.concat(records,ignore_index=True)
        expected=frame[frame.split.ne('test')]
        if out.sha256.duplicated().any() or set(out.sha256)!=set(expected.sha256):
            raise ValueError('OOF coverage is incorrect or repeated')
        if set(out.sha256)&set(protocol['locked_test_sha256']):
            raise ValueError('Locked test entered development scores')
        out.to_csv(root/f'{name}_oof.csv',index=False)
        probs=out[['prob_'+c for c in CLASS_NAMES]].to_numpy()
        scores=[m['macro_f1'] for m in fold_metrics]
        summaries[name]={'oof':metrics(out.label_idx.to_numpy(),probs),'folds':fold_metrics,
                         'mean_fold_f1':float(np.mean(scores)),'std_fold_f1':float(np.std(scores)),
                         'worst_fold_f1':float(min(scores))}
        (root/'development_progress.json').write_text(json.dumps(summaries,indent=2))
    reference=summaries['reference_128'];candidate=summaries['texture_160']
    approved=(candidate['oof']['macro_f1']>=reference['oof']['macro_f1']+.02 and
              candidate['worst_fold_f1']>=reference['worst_fold_f1'])
    result={'protocol':protocol,'candidates':summaries,'candidate_passed':bool(approved),
            'recommendation':'Eligible for a separately trained candidate; no real-world claim' if approved else
                             'Keep current model; candidate lacks consistent development improvement',
            'deployment_changed':False,'test_evaluated':False}
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'study':relative(root),'candidate_passed':bool(approved),
                     'oof':{k:v['oof'] for k,v in summaries.items()}},indent=2))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--samples-per-class',type=int,default=60)
    a=p.parse_args()
    run_study(a.dataset,a.output,a.epochs,a.samples_per_class)

if __name__=='__main__':
    main()
