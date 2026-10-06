import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
from src.config import CLASS_NAMES, ExperimentConfig
from src.dataset_split import assign_group_splits, validate_splits, split_two_tomatoes
from src.preprocessing import load_image, balanced_indices
from src.evaluate import metrics, temperature_scale
from src.manifest import group_near_duplicates


def example_manifest():
    return pd.DataFrame([dict(filepath=f'{g}_{label}.png',group_id=f'fruit{g}',
                             label=label,label_idx=i,sha256=f'{g}-{i}',pixel_sha256=f'p{g}-{i}',source='own')
                         for g in range(12) for i,label in enumerate(CLASS_NAMES)])


class PipelineTests(unittest.TestCase):
    def test_group_split_reproducible_and_disjoint(self):
        df=example_manifest()
        first=assign_group_splits(df)
        pd.testing.assert_frame_equal(first,assign_group_splits(df))
        validate_splits(first)
        self.assertEqual(first.groupby('group_id').split.nunique().max(),1)

    def test_single_and_two_group_rejected(self):
        df=example_manifest()
        with self.assertRaises(ValueError):
            assign_group_splits(df[df.group_id=='fruit0'])
        with self.assertRaises(ValueError):
            split_two_tomatoes(df,'fruit0','fruit0')

    def test_cross_split_duplicate_rejected(self):
        df=assign_group_splits(example_manifest())
        a=df.index[df.split=='train'][0]
        b=df.index[df.split=='test'][0]
        df.loc[b,'pixel_sha256']=df.loc[a,'pixel_sha256']
        with self.assertRaisesRegex(ValueError,'Leakage'):
            validate_splits(df)

    def test_near_duplicate_groups_merge(self):
        df=pd.DataFrame([dict(group_id='a',filepath='a',dhash='0000000000000000'),
                         dict(group_id='b',filepath='b',dhash='0000000000000001')])
        result,pairs=group_near_duplicates(df)
        self.assertEqual(result.group_id.nunique(),1)
        self.assertEqual(len(pairs),1)

    def test_missing_class_rejected(self):
        df=example_manifest()
        with self.assertRaises(ValueError):
            assign_group_splits(df[df.label!='busuk'])

    def test_incorrect_class_index_rejected(self):
        df=assign_group_splits(example_manifest())
        df.loc[0,'label_idx']=2
        with self.assertRaisesRegex(ValueError,'indexes'):
            validate_splits(df)

    def test_preprocess_formats_and_range(self):
        for mode,fmt in [('RGB','JPEG'),('RGBA','PNG'),('L','PNG')]:
            image=Image.new(mode,(80,40))
            buf=io.BytesIO()
            image.save(buf,format=fmt)
            buf.seek(0)
            arr=load_image(buf,64)
            self.assertEqual(arr.shape,(64,64,3))
            self.assertEqual(arr.dtype,np.float32)
            self.assertTrue(0<=arr.min()<=arr.max()<=1)

    def test_balance_exact_and_reproducible(self):
        labels=np.array([0,1,1,2,2,2])
        a=balanced_indices(labels,11,np.random.default_rng(42))
        b=balanced_indices(labels,11,np.random.default_rng(42))
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(np.bincount(labels[a]),[11,11,11])

    def test_metrics_missing_prediction_class(self):
        result=metrics(np.array([0,1,2]),np.array([[.8,.1,.1]]*3))
        self.assertAlmostEqual(result['accuracy'],1/3)
        self.assertEqual(np.shape(result['confusion_matrix']),(3,3))
        self.assertTrue(np.isfinite(result['loss']))

    def test_temperature_probabilities(self):
        p=temperature_scale(np.array([[.8,.1,.1]]),2)
        self.assertAlmostEqual(p.sum(),1)
        self.assertLess(p[0,0],.8)

    def test_invalid_probabilities_and_temperature_rejected(self):
        for temperature in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):
                temperature_scale(np.array([[.8,.1,.1]]),temperature)
        for probs in [np.array([[1.,1.,1.]]),np.array([[float('nan'),0.,0.]]),np.array([[.5,.5]])]:
            with self.assertRaises(ValueError):
                temperature_scale(probs)

    def test_rename_preserves_and_converts(self):
        from src.rename_helper import _copy_photo
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'input.png',Path(d)/'output.jpg'
            Image.new('RGBA',(20,30)).save(a)
            _copy_photo(a,b)
            with Image.open(b) as im:
                self.assertEqual(im.format,'JPEG')
            with self.assertRaises(FileExistsError):
                _copy_photo(a,b)

    def test_model_roundtrip_and_inference_determinism(self):
        import tensorflow as tf
        from src.model import build_model
        tf.keras.utils.set_random_seed(42)
        x=np.random.default_rng(42).random((2,64,64,3),dtype=np.float32)
        model=build_model(config=ExperimentConfig(image_size=64))
        p=model(x,training=False).numpy()
        np.testing.assert_array_equal(p,model(x,training=False).numpy())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.keras'
            model.save(path)
            restored=tf.keras.models.load_model(path,compile=False)
            np.testing.assert_allclose(p,restored(x,training=False).numpy(),rtol=1e-5)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(epochs=0)
        with self.assertRaises(ValueError):
            ExperimentConfig(architecture='pretrained')


if __name__=='__main__':
    unittest.main()
