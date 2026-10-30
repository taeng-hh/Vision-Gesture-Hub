"""Export the actual training keypoints as an offline animated HTML viewer."""
import json
import argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/ksl30_complete_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'학습동작_골격보기.html')
    args = parser.parse_args()
    data = args.data
    registry = json.loads((ROOT/'config/ksl_classes.json').read_text(encoding='utf-8'))['classes']
    names = {c['class_id']:c for c in registry}
    if not (data/'manifest.json').is_file():
        parser.error(f'Missing {data}/manifest.json. Extract the shared training ZIP first or use --data.')
    rows = json.loads((data/'manifest.json').read_text(encoding='utf-8'))['clips']
    clips = []
    for row in rows:
        with np.load(data/row['path'], allow_pickle=False) as d:
            frames = d['frames']
        if not np.isfinite(frames).all(): raise ValueError('Invalid coordinates')
        clips.append(dict(id=row['class_id'],text=names[row['class_id']]['text'],kind=names[row['class_id']]['kind'],fps=row['fps'],frames=np.round(frames,3).tolist()))
    template = (ROOT/'scripts/skeleton_viewer_template.html').read_text(encoding='utf-8')
    output = args.output
    output.write_text(template.replace('__CLIPS_JSON__',json.dumps(clips,ensure_ascii=False,separators=(',',':'))),encoding='utf-8')
    print(f'Exported {len(clips)} clips to {output}')
if __name__ == '__main__': main()
