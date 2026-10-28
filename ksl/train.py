"""Train mixed word/sentence classes with performer-group holdout."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
from .data import SPEC,classes,load_dataset,split


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--classes',type=Path,default=Path('config/ksl_classes.json'))
    p.add_argument('--output',type=Path,default=Path('models/ksl30'))
    p.add_argument('--validation-group')
    p.add_argument('--smoke-only',action='store_true',help='Training replay only; no validation accuracy')
    p.add_argument('--allow-partial',action='store_true',help='Train present classes, explicitly marked partial')
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.epochs<1:p.error('Epochs must be positive')
    if args.smoke_only==bool(args.validation_group):p.error('Choose validation-group OR smoke-only')
    registry=classes(args.classes);allowed={c['class_id'] for c in registry}
    x,rows=load_dataset(args.data,allowed);present={r['class_id'] for r in rows};missing=sorted(allowed-present)
    if missing and not args.allow_partial:p.error(f'Missing classes: {missing}; add data or explicitly use --allow-partial')
    selected=[c for c in registry if c['class_id'] in present]
    if len(selected)<2:p.error('Need at least two classes')
    train_ids,val_ids=(list(range(len(rows))),[]) if args.smoke_only else split(rows,args.validation_group)
    import torch
    from .network import SequenceGRU
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
    ids={c['class_id']:i for i,c in enumerate(selected)}
    y=np.asarray([ids[r['class_id']] for r in rows])
    # Normalization fitted exclusively on training data.
    mean=x[train_ids].mean(axis=(0,1));std=np.maximum(x[train_ids].std(axis=(0,1)),.05)
    tx=torch.tensor((x-mean)/std);ty=torch.tensor(y,dtype=torch.long)
    model=SequenceGRU(len(selected));optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    loss_fn=torch.nn.CrossEntropyLoss();rng=np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train();loss_sum=0
        order=rng.permutation(train_ids)
        for offset in range(0,len(order),32):
            batch=order[offset:offset+32];optim.zero_grad();loss=loss_fn(model(tx[batch]),ty[batch]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();loss_sum+=loss.item()*len(batch)
        if epoch==0 or (epoch+1)%10==0:print(f'Epoch {epoch+1}: train loss {loss_sum/len(order):.4f}',flush=True)
    model.eval()
    with torch.no_grad():probs=model(tx).softmax(dim=1);pred=probs.argmax(dim=1).numpy()
    confusion={}
    for i in val_ids:
        row=confusion.setdefault(rows[i]['class_id'],{});label=selected[int(pred[i])]['class_id'];row[label]=row.get(label,0)+1
    # Closed-set accuracy is separate from uncalibrated live rejection.
    report=dict(partial=bool(missing),missing_classes=missing,train_clips=len(train_ids),validation_clips=len(val_ids),
                validation_group=args.validation_group,validation_accuracy=float(np.mean(pred[val_ids]==y[val_ids])) if val_ids else None,
                training_replay_accuracy=float(np.mean(pred[train_ids]==y[train_ids])),confusion_matrix=confusion,
                seed=args.seed,epochs=args.epochs,feature_spec=SPEC,
                warning='Smoke replay is NOT generalization. Live score thresholds are not calibrated. WORD1290/SEN0355 may overlap.')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.save(dict(format='ksl-gru-v1',state_dict=model.state_dict(),hidden=64,class_ids=[c['class_id'] for c in selected],
                    mean=torch.tensor(mean),std=torch.tensor(std)),args.output/'model.pt')
    (args.output/'classes.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'provenance.json').write_text(json.dumps(dict(train=[rows[i] for i in train_ids],validation=[rows[i] for i in val_ids]),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
