"""Project diagnostics. Distinguish a usable demo from a scientifically ready dataset."""
import argparse
import importlib.metadata
import json
import platform
import sys
from src.config import ROOT_DIR
from src.workspace import dataset_readiness
from src.predict import Predictor, selected_run
from src.manifest import load_bundle

def check(deep=False):
    report={'python':platform.python_version(),'packages':{},'errors':[],
            'real_world_validated':False,'model':None}
    for package in ['tensorflow','keras','numpy','pandas','Pillow','scikit-learn','matplotlib']:
        try:
            report['packages'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['errors'].append('Missing dependency: '+package)
    for path in ['web/index.html','web/app.js','web/style.css','configs/default.json']:
        if not (ROOT_DIR/path).exists():
            report['errors'].append('Missing project asset: '+path)
    try:
        report['dataset']=dataset_readiness()
    except (ValueError,OSError,KeyError) as error:
        report['errors'].append('Dataset: '+str(error))
    try:
        run=selected_run()
        meta=json.loads((run/'run.json').read_text())
        _,bundle=load_bundle(run/'dataset')
        if meta['dataset_fingerprint']!=bundle['fingerprint']:
            raise ValueError('Run and bundle fingerprints disagree')
        if not (run/'model.keras').exists():
            raise ValueError('model.keras missing')
        report['model']={'run':str(run.relative_to(ROOT_DIR)),'mode':meta['mode'],
                         'production_ready':meta['production_ready'],'bundle_verified':True}
        if deep:
            import numpy as np
            model=Predictor(run)
            # Neutral tensor tests execution only, not recognition quality.
            size=meta['config']['image_size']
            probs=model.model(np.full((1,size,size,3),.5,dtype=np.float32),training=False).numpy()
            if not np.isfinite(probs).all() or not np.allclose(probs.sum(),1,atol=1e-4):
                raise ValueError('Non-finite/invalid inference probabilities')
            report['model']['inference_smoke']='passed; neutral tensor, not accuracy test'
    except (ValueError,OSError,KeyError,ImportError) as error:
        report['errors'].append('Model: '+str(error))
    report['demo_ready']=not report['errors']
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--deep',action='store_true',help='Load model and run one numerical inference')
    parser.add_argument('--require-real',action='store_true',help='Fail when own dataset is not ready for training')
    args=parser.parse_args()
    report=check(args.deep)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if report['errors'] or (args.require_real and not report.get('dataset',{}).get('can_train')):
        sys.exit(2)

if __name__=='__main__':
    main()
