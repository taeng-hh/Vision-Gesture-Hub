"""Check export from a supplied data path, without any private data."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np

class SkeletonExportTests(unittest.TestCase):
    def test_export_synthetic_clip_with_custom_paths(self):
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'dataset';data.mkdir()
            frames=np.zeros((4,67,3),dtype=np.float32)
            np.savez_compressed(data/'one.npz',frames=frames)
            (data/'manifest.json').write_text(json.dumps({'clips':[{'path':'one.npz','class_id':'WORD1157','fps':30}]}),encoding='utf-8')
            output=root/'viewer.html'
            subprocess.run([sys.executable,str(repo/'scripts/export_skeleton_viewer.py'),'--data',str(data),'--output',str(output)],check=True,capture_output=True)
            html=output.read_text(encoding='utf-8')
            self.assertIn('WORD1157',html)
            self.assertNotIn('__CLIPS_JSON__',html)
