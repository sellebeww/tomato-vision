import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import numpy as np
import pandas as pd
from src.config import CLASS_NAMES
from src.dataset_split import assign_group_splits
from src.manifest import group_near_duplicates, save_bundle, load_bundle
from src.cross_validate import development_folds
from src.evaluate import calibration_policy, metrics, evaluate_run
from src.quality import assess_quality, review_decision
from src.artifacts import atomic_json

def fixture():
    return pd.DataFrame([dict(filepath=f'data/{g}_{c}.png',sha256=f'{g}_{c}',pixel_sha256=f'p{g}_{c}',
                              group_id=f'fruit{g}',label=name,label_idx=c,source='own')
                         for g in range(12) for c,name in enumerate(CLASS_NAMES)])

class ReliabilityTests(unittest.TestCase):
    def test_csv_probability_rounding_does_not_warn_or_change_classes(self):
        probabilities=np.array([[.70000004,.2,.1],[.1,.70000004,.2],[.1,.2,.70000004]])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result=metrics(np.array([0,1,2]),probabilities)
        self.assertEqual(result['accuracy'],1)
        with self.assertRaises(ValueError):
            metrics(np.array([0,1,2]),probabilities*2)

    def test_development_run_cannot_score_locked_test(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'run.json').write_text('{}')
            with patch('src.evaluate.load_bundle',return_value=(None,{'development_fold':0})):
                with self.assertRaisesRegex(ValueError,'locked test'):
                    evaluate_run(directory)

    def test_failed_atomic_publication_preserves_previous_model_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'selected.json'
            atomic_json(path,{'run':'previous'})
            with self.assertRaises(ValueError):
                atomic_json(path,{'run':float('nan')})
            self.assertEqual(json.loads(path.read_text()),{'run':'previous'})
            self.assertEqual(len(list(Path(directory).iterdir())),1)

    def test_exact_duplicate_links_all_other_views_before_dedup(self):
        frame=pd.DataFrame([
            dict(filepath='a',group_id='A',pixel_sha256='same',dhash='0000000000000000'),
            dict(filepath='b',group_id='B',pixel_sha256='same',dhash='0000000000000000'),
            dict(filepath='b2',group_id='B',pixel_sha256='other',dhash='ffffffffffff0000'),
            dict(filepath='c',group_id='C',pixel_sha256='third',dhash='aaaaaaaaaaaaaaaa')])
        result,_=group_near_duplicates(frame)
        self.assertEqual(len(result),3)
        self.assertEqual(result[result.filepath=='a'].group_id.iloc[0],result[result.filepath=='b2'].group_id.iloc[0])

    def test_fold_validation_covers_development_only_once(self):
        original=assign_group_splits(fixture())
        folds=development_folds(original,'own')
        locked=set(original[original.split=='test'].sha256)
        held=[]
        for fold in folds:
            self.assertEqual(set(fold[fold.split=='test'].sha256),locked)
            held.extend(fold[fold.split=='val'].sha256)
        self.assertEqual(len(held),len(set(held)))
        self.assertEqual(set(held),set(original[original.split!='test'].sha256))

    def test_immutable_bundle_and_tamper_detection(self):
        frame=assign_group_splits(fixture())
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'bundle'
            save_bundle(frame,{'mode':'own'},folder)
            loaded,_=load_bundle(folder,verify=False)
            self.assertEqual(len(loaded),len(frame))
            self.assertEqual(save_bundle(frame,{'mode':'own'},folder),folder)
            with self.assertRaises(ValueError):
                save_bundle(frame,{'mode':'own','seed':2},folder)
            metadata=json.loads((folder/'dataset.json').read_text())
            metadata['mode']='other'
            (folder/'dataset.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'modified'):
                load_bundle(folder,verify=False)

    def test_tiny_validation_does_not_fit_temperature(self):
        labels=np.tile([0,1,2],4)
        result=calibration_policy(labels,np.eye(3)[labels],['group1']*6+['group2']*6)
        self.assertFalse(result['fitted'])
        self.assertEqual(result['temperature'],1)

    def test_bundle_summary_cannot_misreport_sample_count(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'bundle'
            save_bundle(assign_group_splits(fixture()),{'mode':'own'},folder)
            path=folder/'dataset.json'
            metadata=json.loads(path.read_text())
            metadata['unique_images']=99999
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'summary'):
                load_bundle(folder,verify=False)
            with self.assertRaisesRegex(ValueError,'summary'):
                save_bundle(assign_group_splits(fixture()),{'mode':'own'},folder)

    def test_flat_image_gets_quality_warnings(self):
        quality=assess_quality(np.full((64,64,3),.5,dtype=np.float32))
        self.assertGreaterEqual(len(quality['warnings']),2)
        result=review_decision(np.array([.99,.005,.005]),np.array([.99,.005,.005]),[0]*4,quality,'own')
        self.assertTrue(result['needs_review'])

    def test_inconsistent_views_are_not_confidently_accepted(self):
        result=review_decision(np.array([.9,.05,.05]),np.array([.9,.05,.05]),[0,1,0,0],{'warnings':[]},'own')
        self.assertTrue(result['needs_review'])
        self.assertEqual(result['stability']['agreement'],.75)

    def test_calibration_cannot_hide_low_raw_confidence(self):
        result=review_decision(np.array([.9,.05,.05]),np.array([.6,.2,.2]),[0]*4,{'warnings':[]},'own')
        self.assertTrue(result['needs_review'])

if __name__=='__main__':
    unittest.main()
