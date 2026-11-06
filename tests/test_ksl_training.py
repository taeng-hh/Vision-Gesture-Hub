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
                                   '--validation-group','REAL02','--output',str(root/'model'),'--epochs','2'],capture_output=True,text=True,encoding="utf-8",errors="replace")
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((root/'model/metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(report['train_clips'],2);self.assertEqual(report['validation_clips'],2)
            from ksl.predict import Recognizer
            event=Recognizer(root/'model',min_score=0,min_margin=0).predict(a)
            self.assertTrue(event['accepted']);self.assertIn(event['class_id'],{'WORD0','WORD1'})
            self.assertEqual(event['kind'],'word')


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional PyTorch dependency not installed')
class PublishedModelTest(unittest.TestCase):
    def test_published_weights_load_without_dataset(self):
        import hashlib
        from ksl.predict import Recognizer
        folder=Path(__file__).resolve().parents[1]/'models/ksl30_robust_v2'
        self.assertEqual(hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest(),'a947122ead740ada68f68c5e0bca881028cd071f1d1cc99afe53aad16decf8b9')
        model=Recognizer(folder)
        self.assertEqual(len(model.classes),30)
        self.assertEqual(sum(c['kind']=='word' for c in model.classes),23)
        a=np.ones((8,67,3),dtype=np.float32);a[:,:,:2]=0
        a[:,47,0]=2
        for offset in (0,21):
            for j in range(21):a[:,offset+j,:2]=[j*.1,(j%4)*.1]
        event=model.predict(a)
        self.assertTrue(np.isfinite(event['score']))
        self.assertIn(event['candidate_id'],{c['class_id'] for c in model.classes})
        comparison=model.compare_windows([a,a])
        self.assertEqual(len(comparison),2)
        for pair in comparison:
            self.assertEqual(pair['normal']['candidate_id'],event['candidate_id'])
            self.assertAlmostEqual(pair['normal']['score'],event['score'],places=5)
            game=model.predict_game(a)
            self.assertEqual(pair['game']['candidate_id'],game['candidate_id'])
            self.assertAlmostEqual(pair['game']['global_score'],game['global_score'],places=5)



if __name__=='__main__':unittest.main()
