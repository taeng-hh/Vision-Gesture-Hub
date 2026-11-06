"""Stable recognition-event API for subtitle, webcam, and future game code."""
import argparse
import json
from pathlib import Path
from .data import encode
import numpy as np


class Recognizer:
    def __init__(self,folder,min_score=.75,min_margin=.15):
        import torch
        from .network import SequenceGRU
        self.torch=torch;torch.set_num_threads(2);folder=Path(folder)
        d=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        if d['format']!='ksl-gru-v1':raise ValueError('Unsupported model')
        self.classes=json.loads((folder/'classes.json').read_text(encoding='utf-8'))
        if [c['class_id'] for c in self.classes]!=d['class_ids']:raise ValueError('Class order mismatch')
        self.model=SequenceGRU(len(self.classes),d['hidden']);self.model.load_state_dict(d['state_dict']);self.model.eval()
        self.mean=d['mean'];self.std=d['std'];self.min_score=min_score;self.min_margin=min_margin

    def scores(self,frames):
        t=self.torch;x=t.tensor(encode(frames));x=(x-self.mean)/self.std
        with t.no_grad():
            return self.model(x[None]).softmax(dim=1)[0].numpy()

    def predict_proba(self,frames):
        """Full class-order probabilities for the optional fusion UI."""
        return self.scores(frames)

    def predict(self,frames):
        return prediction_event(self.classes,self.scores(frames),self.min_score,self.min_margin)

    def compare_windows(self,clips):
        """One batched forward pass; both decisions use identical scores per clip."""
        t=self.torch
        x=np.stack([encode(clip) for clip in clips])
        tensor=(t.tensor(x)-self.mean)/self.std
        with t.no_grad():scores=self.model(tensor).softmax(dim=1).numpy()
        return [dict(probabilities=s.tolist(),normal=prediction_event(self.classes,s,self.min_score,self.min_margin),
                     game=prediction_event(self.classes,s,self.min_score,self.min_margin,game_only=True))
                for s in scores]

    def predict_game(self,frames):
        # A game-only comparison must not turn arbitrary hands into a command.
        return prediction_event(self.classes,self.scores(frames),self.min_score,self.min_margin,game_only=True)


def prediction_event(classes,scores,min_score=.75,min_margin=.15,game_only=False):
    full=np.asarray(scores,dtype=np.float64)
    if full.shape!=(len(classes),) or not np.isfinite(full).all() or (full<0).any() or not np.isclose(full.sum(),1,atol=1e-4):
        raise ValueError('Expected finite full-class probabilities')
    indices=np.asarray([i for i,c in enumerate(classes) if not game_only or c.get('game_action')],dtype=int)
    if len(indices)<2:raise ValueError('Need at least two candidate classes')
    mass=float(full[indices].sum())
    ranked=full[indices]/max(mass,1e-12) if game_only else full[indices]
    order=np.argsort(-ranked,kind='stable');best=int(indices[order[0]])
    score=float(ranked[order[0]]);margin=score-float(ranked[order[1]])
    global_score=float(full[best]);top=int(np.argmax(full))
    accepted=score>=min_score and margin>=min_margin
    reason=None
    if game_only:
        # Keep absolute evidence and reject strong non-game predictions even if
        # one tiny game probability dominates the other tiny probabilities.
        if global_score<.35 or mass<.40:
            accepted=False;reason='weak_game_evidence'
        elif not classes[top].get('game_action') and float(full[top])>=.65:
            accepted=False;reason='strong_non_game_prediction'
    if not accepted and reason is None:reason='low_score_or_margin'
    c=classes[best]
    return dict(accepted=bool(accepted),class_id=c['class_id'] if accepted else None,
                kind=c['kind'] if accepted else None,text=c['text'] if accepted else None,
                game_action=c.get('game_action') if accepted else None,candidate_id=c['class_id'],
                score=score,margin=margin,global_score=global_score,game_mass=mass if game_only else None,
                full_candidate_id=classes[top]['class_id'],reason=reason,scope='game' if game_only else 'all',
                alternatives=[dict(class_id=classes[int(indices[j])]['class_id'],text=classes[int(indices[j])]['text'],score=float(ranked[j])) for j in order[:3]])



def main():
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--clip',required=True)
    args=p.parse_args()
    with np.load(args.clip,allow_pickle=False) as d: frames=d['frames']
    print(json.dumps(Recognizer(args.model).predict(frames),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
