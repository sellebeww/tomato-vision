"""Isolated fixtures only: never modify the user's real labels or photo inventory."""
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from PIL import Image
from src import workspace, manifest
from src.jobs import TrainingJobs

class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve()
        self.data=self.root/'data'
        self.raw=self.data/'raw'
        self.raw.mkdir(parents=True)
        self.stack=ExitStack()
        for module,values in [(workspace,{'DATA_DIR':self.data,'RAW_DATA_DIR':self.raw}),
                              (manifest,{'DATA_DIR':self.data,'RAW_DATA_DIR':self.raw,'ROOT_DIR':self.root})]:
            for name,value in values.items():
                self.stack.enter_context(patch.object(module,name,value))
        self.stack.enter_context(patch.object(manifest,'resolve_path',lambda p:self.root/Path(p)))
        self.rows=[]
        self.add_photo(1)
        self.save_fixture()

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def add_photo(self,index,group='',label='',approved=False):
        p=self.raw/f'photo{index}.png'
        arr=np.random.default_rng(index).integers(0,256,(40,40,3),dtype=np.uint8)
        Image.fromarray(arr).save(p)
        self.rows.append(dict(filepath=str(p.relative_to(self.root)),sha256=manifest.sha256(p),
                              group_id=group,label=label,approved=approved,source='own',notes=''))

    def save_fixture(self):
        pd.DataFrame(self.rows,columns=manifest.LABEL_COLUMNS).to_csv(self.data/'annotations.csv',index=False)

    def test_revision_backup_and_immutable_fields(self):
        snap=workspace.annotation_snapshot()
        edit=snap['rows'][0]
        edit.update(label='segar',group_id='T1',approved=True,source='other',filepath='outside.png')
        saved=workspace.save_annotations(snap['rows'],snap['revision'])
        self.assertNotEqual(saved['revision'],snap['revision'])
        self.assertEqual(saved['rows'][0]['source'],'own')
        self.assertEqual(saved['rows'][0]['filepath'],'data/raw/photo1.png')
        self.assertEqual(len(list((self.data/'annotation_backups').glob('*.csv'))),1)
        with self.assertRaisesRegex(ValueError,'tab lain'):
            workspace.save_annotations(snap['rows'],snap['revision'])

    def test_approval_requires_actual_group_and_label(self):
        snap=workspace.annotation_snapshot()
        snap['rows'][0]['approved']=True
        with self.assertRaisesRegex(ValueError,'fruit group'):
            workspace.save_annotations(snap['rows'],snap['revision'])
        self.assertEqual(workspace.annotation_snapshot()['revision'],snap['revision'])

    def test_pending_data_cannot_train(self):
        report=workspace.dataset_readiness()
        self.assertFalse(report['can_train'])
        self.assertEqual(report['unreviewed'],1)

    def test_fruit_identity_case_and_spaces_are_not_new_groups(self):
        self.rows=[]
        for g,group in enumerate(['Fruit1','fruit1 ',' FRUIT1']):
            for c,label in enumerate(['segar','tidak_segar','busuk']):
                self.add_photo(100+g*3+c,group,label,True)
        self.save_fixture()
        report=workspace.dataset_readiness()
        self.assertFalse(report['can_train'])
        self.assertEqual(report['groups_per_class'],{'segar':1,'tidak_segar':1,'busuk':1})

    def test_ready_grouped_fixture(self):
        self.rows=[]
        for g in range(9):
            for c,label in enumerate(['segar','tidak_segar','busuk']):
                self.add_photo(100+g*3+c,f'fruit{g}',label,True)
        self.save_fixture()
        report=workspace.dataset_readiness()
        self.assertTrue(report['can_train'],report['blockers'])
        self.assertEqual(report['approved'],27)
        self.assertEqual(set(report['split_preview']),{'train','val','test'})

    def test_import_original_bytes_and_deduplicate(self):
        buf=io.BytesIO()
        Image.new('RGB',(40,40),'green').save(buf,format='PNG')
        first=workspace.import_own_photo(buf.getvalue(),'../../untrusted-name.jpeg')
        second=workspace.import_own_photo(buf.getvalue(),'copy.png')
        self.assertFalse(first['duplicate'])
        self.assertTrue(second['duplicate'])
        target=self.raw/f"UPLOAD_{first['sha256']}.png"
        self.assertEqual(target.read_bytes(),buf.getvalue())
        self.assertEqual(len(workspace.annotation_snapshot()['rows']),2)

    def test_import_rejects_invalid_and_small_image(self):
        with self.assertRaises(OSError):
            workspace.import_own_photo(b'not an image','a.png')
        buf=io.BytesIO()
        Image.new('RGB',(4,4)).save(buf,format='PNG')
        with self.assertRaises(ValueError):
            workspace.import_own_photo(buf.getvalue(),'small.png')

    def test_inventory_supports_uppercase_extension(self):
        Image.new('RGB',(40,40),'blue').save(self.raw/'uppercase.PNG')
        manifest.init_labels()
        self.assertEqual(len(workspace.annotation_snapshot()['rows']),2)

    def test_training_blocked_before_process_creation(self):
        with patch('src.jobs.dataset_readiness',return_value={'can_train':False,'blockers':['label missing']}),patch('src.jobs.subprocess.Popen') as popen:
            with self.assertRaisesRegex(ValueError,'label missing'):
                TrainingJobs().start()
            popen.assert_not_called()

    def test_training_arguments_are_bounded(self):
        for epochs in [-1,0,101,True,'40']:
            with self.assertRaises(ValueError):
                TrainingJobs().start(epochs)

    def test_training_lifecycle(self):
        process=MagicMock()
        process.poll.return_value=None
        with patch('src.jobs.dataset_readiness',return_value={'can_train':True}),patch('src.jobs.prepare',return_value=self.data/'prepared'),patch('src.jobs.OUTPUTS_DIR',self.root/'outputs'),patch('src.jobs.ROOT_DIR',self.root),patch('src.jobs.relative',side_effect=lambda p:str(p.relative_to(self.root))),patch('src.jobs.subprocess.Popen',return_value=process) as popen:
            manager=TrainingJobs()
            job=manager.start(2)
            self.assertEqual(job['state'],'running')
            with self.assertRaisesRegex(ValueError,'masih berjalan'):
                manager.start()
            self.assertIn('src.experiment',popen.call_args.args[0])
            process.poll.return_value=0
            self.assertEqual(manager.status()['state'],'succeeded')
            self.assertTrue(manager.log.closed)

    @unittest.skipIf(os.name=='nt','POSIX process-group assertion')
    def test_cancel_targets_only_managed_group(self):
        process=MagicMock(pid=123456)
        process.poll.return_value=None
        with patch('src.jobs.dataset_readiness',return_value={'can_train':True}),patch('src.jobs.prepare',return_value=self.data/'prepared'),patch('src.jobs.OUTPUTS_DIR',self.root/'outputs'),patch('src.jobs.ROOT_DIR',self.root),patch('src.jobs.relative',side_effect=lambda p:str(p.relative_to(self.root))),patch('src.jobs.subprocess.Popen',return_value=process),patch('src.jobs.os.killpg') as killpg:
            manager=TrainingJobs()
            manager.start(1)
            self.assertEqual(manager.cancel()['state'],'cancelling')
            self.assertEqual(killpg.call_args.args[0],123456)
            process.poll.return_value=-15
            self.assertEqual(manager.status()['state'],'cancelled')
            self.assertTrue(manager.log.closed)

if __name__=='__main__':
    unittest.main()
