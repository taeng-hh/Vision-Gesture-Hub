"""Small neural save/load and holdout integration test; CPU only."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from ksl.data import SPEC,write_clip


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional PyTorch training dependency not installed')
class TrainingTest(unittest.TestCase):
    def test_training_holdout_and_saved_event(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';rows=[];registry=[]
            for i in range(2):
                registry.append(dict(class_id=f'WORD{i}',kind='word',text=f'word{i}',game_action='start' if i==0 else None))
                for group in ('REAL01','REAL02'):
                    a=np.ones((10,67,3),dtype=np.float32);a[:,:,:2]=0;a[:,44,:2]=[0,0];a[:,47,:2]=[2,0]
                    for offset in (0,21):
                        for j in range(21):a[:,offset+j,:2]=[j*.1,(j%4)*(.1+i*.05)]
                    rows.append(write_clip(data,f'w{i}_{group}',a,class_id=f'WORD{i}',performer_group=group,
                                           recording_id=f'w{i}_{group}',view='F',extractor='synthetic-test'))
            (data/'manifest.json').write_text(json.dumps(dict(feature_spec=SPEC,clips=rows)),encoding='utf-8')
            config=root/'classes.json';config.write_text(json.dumps(dict(classes=registry)),encoding='utf-8')
            result=subprocess.run([sys.executable,'-B','-X','utf8','-m','ksl.train','--data',str(data),'--classes',str(config),
                                   '--validation-group','REAL02','--output',str(root/'model'),'--epochs','2'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((root/'model/metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(report['train_clips'],2);self.assertEqual(report['validation_clips'],2)
            from ksl.predict import Recognizer
            event=Recognizer(root/'model',min_score=0,min_margin=0).predict(a)
            self.assertTrue(event['accepted']);self.assertIn(event['class_id'],{'WORD0','WORD1'})
            self.assertEqual(event['kind'],'word')


if __name__=='__main__':unittest.main()
