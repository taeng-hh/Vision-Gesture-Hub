import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
from ksl.archive import JoinedParts
from ksl.data import SPEC,encode,write_clip,load_dataset,split


def frames(kind=0):
    a=np.ones((10,67,3),dtype=np.float32)
    a[:,:,:2]=0
    a[:,44,:2]=[0,0];a[:,47,:2]=[2,0]
    for offset in (0,21):
        for j in range(21):a[:,offset+j,:2]=[j*.1,(j%4)*(.1+kind*.05)]
    return a


class DataTests(unittest.TestCase):
    def test_normalization_and_confidence_channel(self):
        a=frames();b=a.copy();b[:,:,:2]=b[:,:,:2]*100+[500,300]
        np.testing.assert_allclose(encode(a),encode(b),atol=1e-5)
        a[:,:,2]=-1
        with self.assertRaises(ValueError):encode(a)

    def test_split_blocks_same_source_and_missing_classes(self):
        rows=[dict(class_id='A',performer_group='01',recording_id='a'),
              dict(class_id='A',performer_group='02',recording_id='a')]
        with self.assertRaisesRegex(ValueError,'leakage'):split(rows,'02')
        with self.assertRaises(ValueError):split(rows,'03')

    def test_exchange_writer_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            rows=[write_clip(root,'a',frames(),class_id='WORD1',performer_group='REAL01',recording_id='a',view='F',extractor='test')]
            (root/'manifest.json').write_text(json.dumps(dict(feature_spec=SPEC,clips=rows)))
            x,read=load_dataset(root,{'WORD1'});self.assertEqual(x.shape,(1,32,90))
            with self.assertRaises(FileExistsError):write_clip(root,'a',frames(),class_id='WORD1',performer_group='REAL01',recording_id='a',view='F',extractor='test')

    def test_split_zip_random_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            memory=io.BytesIO()
            with zipfile.ZipFile(memory,'w') as z:z.writestr('sample.json','{"ok":true}')
            raw=memory.getvalue();root=Path(temp);parts=[]
            for offset in range(0,len(raw),31):
                path=root/f'test.zip.part{offset}';path.write_bytes(raw[offset:offset+31]);parts.append(path)
            with JoinedParts(parts[::-1]) as stream:
                with zipfile.ZipFile(stream) as z:self.assertEqual(json.loads(z.read('sample.json')),{'ok':True})


if __name__=='__main__':unittest.main()
