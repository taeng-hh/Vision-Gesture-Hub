"""Select registry classes from WORD/SEN keypoint and morpheme ZIPs."""
import argparse
import json
import math
import re
import zipfile
from contextlib import contextmanager
from pathlib import Path
from .archive import JoinedParts
from .data import SPEC, classes, write_clip


@contextmanager
def archive(path):
    path=Path(path)
    if path.name.endswith('.part0'):
        with JoinedParts(path.parent.glob(path.name[:-1]+'*')) as stream:
            with zipfile.ZipFile(stream) as z:yield z
    else:
        with zipfile.ZipFile(path) as z:yield z


def find_archives(source,token):
    return sorted(p for p in Path(source).rglob('*') if token in p.name.lower()
                  and (p.name.endswith('.zip') or p.name.endswith('.zip.part0')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--classes',type=Path,default=Path('config/ksl_classes.json'))
    parser.add_argument('--views',default='F',help='Comma separated; keep same take views grouped')
    parser.add_argument('--fps',type=float,default=30)
    args=parser.parse_args()
    if not math.isfinite(args.fps) or args.fps<=0:parser.error('FPS must be positive')
    if (args.output/'manifest.json').exists():parser.error('Use a new output folder; dataset already exists')
    registry=classes(args.classes); wanted={r['class_id']:r for r in registry}; views=set(args.views.split(','))
    annotations={}; conflicts=[]
    pattern=re.compile(r'(NIA_SL_((?:WORD|SEN)\d+)_REAL(\d+)_([A-Z]))')
    for path in find_archives(args.source,'morpheme'):
        with archive(path) as z:
            for n in z.namelist():
                m=pattern.search(n)
                if not m or m[2] not in wanted or m[4] not in views or not n.endswith('.json'):continue
                d=json.loads(z.read(n)); gloss=[a.get('name','') for seg in d.get('data',[]) for a in seg.get('attributes',[])]
                if gloss!=wanted[m[2]]['expected_gloss']:
                    conflicts.append(dict(clip_id=m[1],expected=wanted[m[2]]['expected_gloss'],actual=gloss));continue
                if m[1] in annotations and annotations[m[1]]!=d:raise ValueError('Conflicting duplicate annotation')
                annotations[m[1]]=d
    clips=[];seen=set();invalid=[]
    for path in find_archives(args.source,'keypoint'):
        print('Reading',path.name,flush=True)
        with archive(path) as z:
            groups={}
            for n in z.namelist():
                m=pattern.search(n)
                if m and m[1] in annotations and n.endswith('_keypoints.json'):
                    index=int(re.search(r'_(\d+)_keypoints.json$',n)[1])
                    groups.setdefault(m[1],[]).append((index,n))
            for clip_id,entries in sorted(groups.items()):
                if clip_id in seen:raise ValueError(f'Duplicate clip across archives: {clip_id}')
                seen.add(clip_id);m=pattern.search(clip_id);d=annotations[clip_id];segments=d['data']
                start=min(float(s['start']) for s in segments);end=max(float(s['end']) for s in segments)
                if not 0<=start<end:raise ValueError('Invalid annotation interval')
                frames=[]
                for index,n in sorted(entries):
                    if not start<=index/args.fps<end:continue
                    person=json.loads(z.read(n))['people'];person=person[0] if isinstance(person,list) else person
                    pts=[]
                    for key,count in [('hand_left_keypoints_2d',21),('hand_right_keypoints_2d',21),('pose_keypoints_2d',25)]:
                        values=person[key]
                        if len(values)!=3*count:raise ValueError(f'Invalid keypoint dimensions: {n}')
                        pts.extend(values[i:i+3] for i in range(0,len(values),3))
                    frames.append(pts)
                try:
                    row=write_clip(args.output,clip_id,frames,class_id=m[2],performer_group='REAL'+m[3],
                                   recording_id=clip_id.rsplit('_',1)[0],view=m[4],extractor='AIHub-provided-keypoints')
                    row.update(start_s=start,end_s=end,fps=args.fps)
                    clips.append(row)
                except ValueError as exc:invalid.append(dict(clip_id=clip_id,error=str(exc)))
    present={r['class_id'] for r in clips}
    report=dict(schema_version=1,feature_spec=SPEC,clips=clips,missing_classes=sorted(set(wanted)-present),
                label_conflicts=conflicts,invalid_clips=invalid,source_note='Single-view default; no independent evaluation implied.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(clips=len(clips),missing=report['missing_classes'],label_conflicts=len(conflicts),invalid=len(invalid)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
