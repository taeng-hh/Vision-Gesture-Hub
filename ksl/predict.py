"""Stable recognition-event API for subtitle, webcam, and future game code."""
import argparse
import json
from pathlib import Path
from .data import encode


class Recognizer:
    def __init__(self,folder,min_score=.75,min_margin=.15):
        import torch
        from .network import SequenceGRU
        self.torch=torch;folder=Path(folder)
        d=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        if d['format']!='ksl-gru-v1':raise ValueError('Unsupported model')
        self.classes=json.loads((folder/'classes.json').read_text(encoding='utf-8'))
        if [c['class_id'] for c in self.classes]!=d['class_ids']:raise ValueError('Class order mismatch')
        self.model=SequenceGRU(len(self.classes),d['hidden']);self.model.load_state_dict(d['state_dict']);self.model.eval()
        self.mean=d['mean'];self.std=d['std'];self.min_score=min_score;self.min_margin=min_margin

    def predict(self,frames):
        t=self.torch;x=t.tensor(encode(frames));x=(x-self.mean)/self.std
        with t.no_grad():scores=self.model(x[None]).softmax(dim=1)[0];values,indices=scores.topk(2)
        score=float(values[0]);margin=float(values[0]-values[1]);candidate=self.classes[int(indices[0])]
        accepted=score>=self.min_score and margin>=self.min_margin
        return dict(accepted=accepted,class_id=candidate['class_id'] if accepted else None,
                    kind=candidate['kind'] if accepted else None,text=candidate['text'] if accepted else None,
                    game_action=candidate.get('game_action') if accepted else None,
                    candidate_id=candidate['class_id'],score=score,margin=margin)


def main():
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--clip',required=True)
    args=p.parse_args()
    with np.load(args.clip,allow_pickle=False) as d: frames=d['frames']
    print(json.dumps(Recognizer(args.model).predict(frames),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
