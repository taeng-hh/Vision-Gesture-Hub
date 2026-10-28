"""Canonical exchange format for AI Hub and future video extractors."""
import json
from pathlib import Path
import numpy as np

SPEC = 'ksl-xyc67-unmirrored-v1'
STEPS = 32


def classes(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))['classes']
    ids = [c['class_id'] for c in data]
    if len(ids) != len(set(ids)) or any(c['kind'] not in ('word', 'sentence') for c in data):
        raise ValueError('Invalid or duplicate classes')
    return data


def encode(frames):
    a = np.asarray(frames, dtype=np.float32)
    if a.ndim != 3 or a.shape[1:] != (67, 3) or len(a) < 4 or not np.isfinite(a).all():
        raise ValueError('Expected finite T>=4 x 67 x 3 XY/confidence frames')
    if ((a[:, :, 2] < 0) | (a[:, :, 2] > 1)).any():
        raise ValueError('Third channel must be confidence in [0,1], not depth')
    if (a[:, [44, 47], 2] < .2).any():
        raise ValueError('Both shoulders required in every frame')
    center = (a[:, 44, :2] + a[:, 47, :2]) / 2
    body_scale = np.linalg.norm(a[:, 44, :2] - a[:, 47, :2], axis=-1)
    if (body_scale < 1e-6).any(): raise ValueError('Invalid shoulder width')
    parts = []
    for hand in (a[:, :21], a[:, 21:42]):
        valid = hand[:, :, 2] >= .2
        scale = np.linalg.norm(hand[:, 9, :2] - hand[:, 0, :2], axis=-1)
        present = valid[:, 0] & valid[:, 9] & (valid.sum(axis=1) >= 15) & (scale > 1e-6)
        local = (hand[:, :, :2] - hand[:, :1, :2]) / np.maximum(scale[:, None, None], 1e-6)
        local *= valid[:, :, None] & present[:, None, None]
        wrist = (hand[:, 0, :2] - center) / body_scale[:, None]
        wrist *= present[:, None]
        parts.append(np.concatenate([local.reshape(len(a),42), wrist, present[:,None]],axis=1))
    f = np.concatenate(parts, axis=1)
    if not np.any(f[:, [44,89]]): raise ValueError('No usable hands')
    x = np.linspace(0,1,len(f)); target = np.linspace(0,1,STEPS)
    return np.stack([np.interp(target,x,c) for c in f.T],axis=1).astype(np.float32)


def write_clip(root, clip_id, frames, *, class_id, performer_group, recording_id, view, extractor):
    """Public writer for video teammates. All views of one take share recording_id."""
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', clip_id): raise ValueError('Unsafe clip ID')
    if not all(isinstance(s,str) and s for s in (class_id,performer_group,recording_id,view,extractor)):
        raise ValueError('All provenance fields are required')
    frames=np.asarray(frames,dtype=np.float32)
    encode(frames)
    root=Path(root); (root/'clips').mkdir(parents=True,exist_ok=True)
    relative=f'clips/{clip_id}.npz'
    path=root/relative
    if path.exists(): raise FileExistsError(path)
    np.savez_compressed(path,frames=frames)
    return dict(path=relative,class_id=class_id,performer_group=performer_group,
                recording_id=recording_id,view=view,extractor=extractor,feature_spec=SPEC)


def load_dataset(root, allowed):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('feature_spec')!=SPEC: raise ValueError('Incompatible feature specification')
    rows=manifest['clips']; xs=[]; seen=set(); group_by_record={}
    for row in rows:
        if row['class_id'] not in allowed: raise ValueError('Class not present in registry')
        if row.get('feature_spec')!=SPEC: raise ValueError('Clip feature specification mismatch')
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root) or path in seen: raise ValueError('Unsafe or duplicate clip path')
        seen.add(path)
        if not all(row.get(k) for k in ('performer_group','recording_id','extractor','view')):
            raise ValueError('Missing provenance')
        prior=group_by_record.setdefault(row['recording_id'],row['performer_group'])
        if prior!=row['performer_group']: raise ValueError('Recording split across performer groups')
        with np.load(path,allow_pickle=False) as d: xs.append(encode(d['frames']))
    if not xs: raise ValueError('No usable clips')
    return np.stack(xs),rows


def split(rows, validation_group):
    train=[i for i,r in enumerate(rows) if r['performer_group']!=validation_group]
    val=[i for i,r in enumerate(rows) if r['performer_group']==validation_group]
    if not train or not val: raise ValueError('Need separate training and validation performer groups')
    if {rows[i]['recording_id'] for i in train}&{rows[i]['recording_id'] for i in val}:
        raise ValueError('Recording leakage')
    if {rows[i]['class_id'] for i in train}!={rows[i]['class_id'] for i in val}:
        raise ValueError('Train and validation must cover the same class IDs')
    return train,val
