"""Train with synthetic timing/tracking perturbations; never call this user validation."""
import sys,json,random,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ksl.data import encode,SPEC
from ksl.network import SequenceGRU
import torch

def perturb(raw,rng):
    a=raw.copy(); n=len(a)
    # Keep chronological order, vary performance pace and segmentation boundaries.
    source=np.linspace(0,1,n)
    target=np.linspace(0,1,n)**rng.uniform(.7,1.4)
    a=np.stack([np.interp(target,source,c) for c in a.reshape(n,-1).T],axis=1).reshape(a.shape)
    shoulder=np.linalg.norm(a[:,44,:2]-a[:,47,:2],axis=1)
    for start in (0,21):
        hand=a[:,start:start+21];present=hand[:,:,2]>=.2
        wrist=hand[:,:1,:2].copy()
        hand[:,:,:2]=wrist+(hand[:,:,:2]-wrist)*rng.uniform(.9,1.1)
        shift=rng.normal(0,.025,(1,1,2))*shoulder[:,None,None]
        hand[:,:,:2]+=shift+ rng.normal(0,.0025,hand[:,:,:2].shape)*shoulder[:,None,None]
        hand[:,:,:2]*=present[:,:,None]
        if n>8 and rng.random()<.35:
            j=int(rng.integers(1,n-2));hand[j:j+2]=0
    lo=int(rng.integers(0,max(1,n//12)));hi=n-int(rng.integers(0,max(1,n//12)))
    a=a[lo:hi]
    before=int(rng.integers(0,max(1,n//5)));after=int(rng.integers(0,max(1,n//5)))
    return np.concatenate([np.repeat(a[:1],before,axis=0),a,np.repeat(a[-1:],after,axis=0)])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/ksl30_complete_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'models/ksl30_robust_v2')
    args=parser.parse_args();folder=args.output
    if folder.exists():raise FileExistsError(folder)
    torch.set_num_threads(2);torch.manual_seed(42);random.seed(42)
    root=args.data;manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    registry=json.loads((ROOT/'config/ksl_classes.json').read_text(encoding='utf-8'))['classes']
    labels={c['class_id']:i for i,c in enumerate(registry)}
    raw=[];ys=[]
    for r in manifest['clips']:
        with np.load(root/r['path'],allow_pickle=False) as d:raw.append(d['frames'])
        ys.append(labels[r['class_id']])
    def samples(seed,count):
        rng=np.random.default_rng(seed);xs=[];yy=[]
        for a,y in zip(raw,ys):
            xs.append(encode(a));yy.append(y)
            for _ in range(count):xs.append(encode(perturb(a,rng)));yy.append(y)
        return np.asarray(xs),np.asarray(yy)
    x,y=samples(42,24);probe,py=samples(8453,8)
    mean=x.mean(axis=(0,1));std=np.maximum(x.std(axis=(0,1)),.05)
    tx=torch.tensor((x-mean)/std);ty=torch.tensor(y,dtype=torch.long)
    model=SequenceGRU(len(registry));optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    rng=np.random.default_rng(42)
    for epoch in range(180):
        model.train();total=0
        for batch in np.array_split(rng.permutation(len(y)),int(np.ceil(len(y)/64))):
            optim.zero_grad();loss=torch.nn.functional.cross_entropy(model(tx[batch]),ty[batch]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();total+=loss.item()*len(batch)
        if (epoch+1)%30==0:print(f'Epoch {epoch+1}: loss {total/len(y):.4f}',flush=True)
    model.eval()
    def evaluate(net,mean,std):
        with torch.no_grad():
            probs=net(torch.tensor((probe-mean)/std)).softmax(1);pred=probs.argmax(1);top=probs.topk(2,dim=1).values
        return dict(synthetic_top1_accuracy=float((pred==torch.tensor(py)).float().mean()),synthetic_accepted_correct=float(((pred==torch.tensor(py))&(top[:,0]>=.75)&((top[:,0]-top[:,1])>=.15)).float().mean()))
    baseline=ROOT/'models/ksl30_complete_v1/model.pt'
    old_result=None
    if baseline.is_file():
        old=torch.load(baseline,weights_only=True,map_location='cpu');oldnet=SequenceGRU(len(registry));oldnet.load_state_dict(old['state_dict']);oldnet.eval()
        old_result=evaluate(oldnet,old['mean'].numpy(),old['std'].numpy())
    report=dict(source_recordings=30,classes=30,synthetic_train_samples=len(y),epochs=180,seed=42,feature_spec=SPEC,old=old_result,new=evaluate(model,mean,std),warning='Synthetic perturbation check from the SAME 30 recordings, NOT independent validation or webcam accuracy. Rejection thresholds unchanged.')
    folder.mkdir(parents=True);torch.save(dict(format='ksl-gru-v1',state_dict=model.state_dict(),hidden=64,class_ids=[c['class_id'] for c in registry],mean=torch.tensor(mean),std=torch.tensor(std)),folder/'model.pt')
    (folder/'classes.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
