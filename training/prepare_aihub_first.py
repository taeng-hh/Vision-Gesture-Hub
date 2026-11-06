"""Extract selected clips directly from split ZIP files, train an offline baseline."""
import argparse
import bisect
import io
import json
import math
import re
import zipfile
from pathlib import Path
from core.aihub_words import AIHubWordModel

WORDS = {'0943': ('GO', '가다'), '1152': ('HURT', '아프다'),
         '1157': ('I', '나'), '1249': ('HEAD', '머리'),
         '1496': ('HOSPITAL', '병원'), '1534': ('FOOD', '밥'),
         '1589': ('REQUEST', '부탁'), '2388': ('HELP', '돕다')}
UNKNOWN = {'0001', '1501', '1514'}  # Fixed unseen-word diagnostic, never used to tune thresholds.


class JoinedParts(io.RawIOBase):
    def __init__(self, parts):
        self.parts = sorted(parts, key=lambda p: int(p.suffix[5:]))
        self.starts, self.pos, self.total = [], 0, 0
        self.handle, self.index = None, -1
        for path in self.parts:
            if int(path.suffix[5:]) != self.total:
                raise ValueError('Missing or misordered ZIP part')
            self.starts.append(self.total)
            self.total += path.stat().st_size
        if not self.parts:
            raise ValueError('No keypoint ZIP parts found')

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.total + offset
        if pos < 0:
            raise ValueError('Negative position')
        self.pos = pos
        return pos

    def read(self, size=-1):
        size = self.total - self.pos if size < 0 else min(size, self.total - self.pos)
        chunks = []
        while size > 0:
            idx = bisect.bisect_right(self.starts, self.pos) - 1
            if idx != self.index:
                if self.handle: self.handle.close()
                self.handle, self.index = self.parts[idx].open('rb'), idx
            self.handle.seek(self.pos - self.starts[idx])
            end = self.starts[idx + 1] if idx + 1 < len(self.starts) else self.total
            chunk = self.handle.read(min(size, end - self.pos))
            if not chunk: break
            chunks.append(chunk)
            size -= len(chunk)
            self.pos += len(chunk)
        return b''.join(chunks)

    def close(self):
        if self.handle: self.handle.close()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('.'))
    args = parser.parse_args()
    out = args.output.resolve()
    label_path = next(args.source.rglob('01_real_word_morpheme.zip.part0'))
    selected = set(WORDS) | UNKNOWN
    labels = {}
    with zipfile.ZipFile(label_path) as z:
        for name in z.namelist():
            m = re.search(r'(NIA_SL_WORD(\d+)_REAL01_([A-Z]))_morpheme.json$', name)
            if m and m[2] in selected:
                labels[m[1]] = json.loads(z.read(name))
    clips = []
    with JoinedParts(list(args.source.rglob('01_real_word_keypoint.zip.part*'))) as stream:
        with zipfile.ZipFile(stream) as z:
            grouped = {}
            for name in z.namelist():
                m = re.search(r'(NIA_SL_WORD(\d+)_REAL01_([A-Z]))_(\d+)_keypoints.json$', name)
                if m and m[2] in selected:
                    grouped.setdefault(m[1], []).append((int(m[4]), name))
            for clip_id, entries in sorted(grouped.items()):
                word_id, angle = re.search(r'WORD(\d+)_REAL01_([A-Z])', clip_id).groups()
                annotation = labels[clip_id]
                segments = annotation['data']
                if len(segments) != 1:
                    raise ValueError(f'Expected one word segment: {clip_id}')
                segment = segments[0]
                name = segment['attributes'][0]['name']
                if word_id in WORDS and name != WORDS[word_id][1]:
                    raise ValueError(f'Unexpected word label: {clip_id}: {name}')
                start, end = float(segment['start']), float(segment['end'])
                frames = []
                for index, path in sorted(entries):
                    # Dataset documentation specifies 30fps, indices start at zero.
                    if not start <= index / 30 < end:
                        continue
                    person = json.loads(z.read(path))['people']
                    if isinstance(person, list): person = person[0]
                    points = []
                    for key, count in [('hand_left_keypoints_2d', 21),
                                       ('hand_right_keypoints_2d', 21), ('pose_keypoints_2d', 25)]:
                        values = person[key]
                        if len(values) != count * 3: raise ValueError(f'Bad dimensions: {path}')
                        points.extend([values[i:i+3] for i in range(0, len(values), 3)])
                    frames.append(points)
                clip = {'format': 'aihub-openpose-2d-v1', 'clip_id': clip_id,
                        'word_id': word_id, 'label': WORDS[word_id][0] if word_id in WORDS else name,
                        'korean': name, 'signer_code': 'REAL01', 'view': angle,
                        'fps': 30, 'start_s': start, 'end_s': end, 'frames': frames,
                        'source': str(args.source), 'is_training': word_id in WORDS}
                folder = out / 'data/aihub_first' / ('train' if clip['is_training'] else 'unknown')
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f'{clip_id}.json').write_text(json.dumps(clip, ensure_ascii=False), encoding='utf-8')
                clips.append(clip)
                print('EXTRACTED', clip_id, name, len(frames), flush=True)
    train = [c for c in clips if c['is_training']]
    unknown = [c for c in clips if not c['is_training']]
    if len(train) != len(WORDS) * 5:
        raise ValueError(f'Expected 40 selected clips, got {len(train)}')
    model = AIHubWordModel.fit(train)
    path = out / 'models/ksl_aihub_first.npz'
    model.save(path)
    restored = AIHubWordModel.load(path)
    known_results = [{'clip_id': c['clip_id'], 'expected': c['label'],
                      **restored.predict(c['frames'])} for c in train]
    unknown_results = [{'clip_id': c['clip_id'], 'expected': '<UNKNOWN>',
                        **restored.predict(c['frames'])} for c in unknown]
    report = {'model_format': 'aihub-openpose-2d-v1', 'model_type': 'nearest-sequence-template',
              'words': WORDS, 'training_clips': len(train), 'independent_signer_codes': ['REAL01'],
              'independent_recordings_per_word': 1, 'views_per_word': 5,
              'held_out_accuracy': None, 'resubstitution_accepted_correct':
              sum(r['label'] == r['expected'] for r in known_results),
              'unknown_diagnostic_count': len(unknown_results),
              'unknown_diagnostic_rejected': sum(r['label'] is None for r in unknown_results),
              'threshold_policy': 'Fixed heuristic distance=.35 margin=.03; not calibrated',
              'limitations': ['Five camera views of the same performance are correlated.',
                             'No independent signer or recording evaluation is available.',
                             'Unknown-word diagnostic uses the same source signer.',
                             'Not compatible with MediaPipe XYZ or main.py --sign-model.',
                             'Sentence output is a semantic template, not verified KSL grammar.'],
              'training_replay': known_results, 'unknown_diagnostic': unknown_results}
    path.with_suffix('.metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('MODEL', path, 'BYTES', path.stat().st_size, flush=True)
    print('REPLAY_ONLY', report['resubstitution_accepted_correct'], '/', len(train),
          'UNKNOWN_REJECTED', report['unknown_diagnostic_rejected'], '/', len(unknown), flush=True)


if __name__ == '__main__': main()
