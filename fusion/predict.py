"""Class-aligned late fusion; raw full probability vectors are combined once."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from ksl.data import encode
from ksl.network import SequenceGRU

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_KEYPOINT_FOLDER=ROOT/"models/ksl30_robust_v2"
DEFAULT_KEYPOINT_WEIGHT=.1
ASSIST_KEYPOINT_WEIGHT=.8


def align_probabilities(scores,source_ids,target_ids):
    source_ids=list(source_ids);target_ids=list(target_ids)
    if len(set(source_ids))!=len(source_ids) or len(set(target_ids))!=len(target_ids) or set(source_ids)!=set(target_ids):
        raise ValueError("Class IDs must be unique and identical across branches")
    a=np.asarray(scores,dtype=np.float64)
    if a.shape!=(len(source_ids),) or not np.isfinite(a).all() or np.any(a<0) or a.sum()<=0:
        raise ValueError("Invalid probability vector")
    lookup={cid:i for i,cid in enumerate(source_ids)}
    aligned=a[[lookup[cid] for cid in target_ids]]
    return aligned/aligned.sum()


def fuse_probabilities(keypoint,rgb,keypoint_weight=.5):
    if not 0<=keypoint_weight<=1:raise ValueError("Keypoint weight must be in [0,1]")
    if keypoint is None and rgb is None:raise ValueError("Neither model has a usable input")
    if keypoint is None:return np.asarray(rgb),["rgb"]
    if rgb is None:return np.asarray(keypoint),["keypoint"]
    if np.asarray(keypoint).shape!=np.asarray(rgb).shape:raise ValueError("Class dimensions differ")
    return keypoint_weight*np.asarray(keypoint)+(1-keypoint_weight)*np.asarray(rgb),["keypoint","rgb"]


def recognition_event(scores,classes,min_score=.75,min_margin=.15):
    a=np.asarray(scores);order=np.argsort(-a);top=int(order[0]);score=float(a[top]);margin=float(a[top]-a[int(order[1])])
    accepted=score>=min_score and margin>=min_margin
    c=classes[top]
    return dict(accepted=accepted,class_id=c["class_id"] if accepted else None,
        kind=c["kind"] if accepted else None,text=c["text"] if accepted else None,
        game_action=c.get("game_action") if accepted else None,candidate_id=c["class_id"],
        score=score,margin=margin,alternatives=[dict(class_id=classes[int(i)]["class_id"],
            text=classes[int(i)]["text"],score=float(a[int(i)])) for i in order[:3]])


MODE_LABELS={"keypoint":"키포인트", "rgb":"영상", "fusion":"결합", "assisted":"키포인트+CNN보조"}


def select_prediction(result,classes,mode="fusion",min_score=.75,min_margin=.15):
    """Compare the same clip without letting the other branch change its decision."""
    if mode not in MODE_LABELS:raise ValueError("Unknown recognition mode")
    ids=[c["class_id"] for c in classes]
    def branch(name):
        values=result.get(name+"_probabilities")
        return None if values is None else align_probabilities(values,ids,ids)
    keypoint=branch("keypoint");rgb=branch("rgb")
    event=None;status=None;assistance_needed=False;assistance_used=False
    if mode=="assisted":
        if keypoint is None:raise ValueError("키포인트 입력 부족: CNN만으로 확정하지 않습니다.")
        primary=recognition_event(keypoint,classes,min_score,min_margin)
        scores=keypoint;used=["keypoint"];effective_weight=1.
        event=primary
        if primary["accepted"]:status="keypoint_confident"
        elif rgb is None:
            assistance_needed="rgb" not in result.get("computed_modalities",())
            status="awaiting_cnn" if assistance_needed else "cnn_unavailable"
        else:
            used=["keypoint","rgb"]
            auxiliary=recognition_event(rgb,classes,min_score,min_margin)
            if auxiliary["candidate_id"]!=primary["candidate_id"]:status="cnn_disagrees"
            elif not auxiliary["accepted"]:status="cnn_uncertain"
            else:
                scores=ASSIST_KEYPOINT_WEIGHT*keypoint+(1-ASSIST_KEYPOINT_WEIGHT)*rgb
                effective_weight=ASSIST_KEYPOINT_WEIGHT
                event=recognition_event(scores,classes,min_score,min_margin)
                status="cnn_confirmed" if event["accepted"] else "still_uncertain"
                assistance_used=True
    elif mode=="fusion":
        scores,used=fuse_probabilities(keypoint,rgb,result["keypoint_weight"])
        effective_weight=result["keypoint_weight"]
    else:
        scores=keypoint if mode=="keypoint" else rgb
        if scores is None:raise ValueError(MODE_LABELS[mode]+" 입력 부족: 선택한 모델로 판정할 수 없습니다.")
        used=[mode];effective_weight=1. if mode=="keypoint" else 0.
    selected=dict(result)
    selected.update(event if event is not None else recognition_event(scores,classes,min_score,min_margin))
    selected.update(probabilities=scores.tolist(),decision_mode=mode,decision_modalities=used,
                    decision_keypoint_weight=effective_weight,assist_status=status,
                    assistance_needed=assistance_needed,assistance_used=assistance_used)
    return selected


class KeypointBranch:
    def __init__(self,folder):
        from ksl.predict import Recognizer
        self.recognizer=Recognizer(folder)
        self.classes=self.recognizer.classes;self.ids=[c["class_id"] for c in self.classes]
        if len(self.ids)!=30:raise ValueError("Expected 30-class teammate-compatible model")
    def predict_proba(self,frames):return self.recognizer.predict_proba(frames)


class TemporalRGB(nn.Module):
    def __init__(self,dim,classes,config):
        super().__init__();self.kind=config["temporal"]
        self.rnn=(nn.GRU if self.kind=="GRU" else nn.LSTM)(dim,config["hidden"],num_layers=2,batch_first=True,dropout=config["dropout"])
        self.head=nn.Linear(config["hidden"],classes)
    def forward(self,x):
        _,h=self.rnn(x)
        return self.head((h[0] if self.kind=="LSTM" else h)[-1])


class RGBBranch:
    def __init__(self,folder):
        from torchvision.models import mobilenet_v3_small
        folder=Path(folder);self.manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format")!="rgb-ksl30-ensemble-v1":raise ValueError("Unsupported RGB bundle")
        self.ids=self.manifest["class_ids"];self.members=[]
        cnn=mobilenet_v3_small(weights=None)
        cnn.load_state_dict(torch.load(folder/self.manifest["cnn_weights"],map_location="cpu",weights_only=True))
        self.cnn=nn.Sequential(cnn.features,cnn.avgpool).eval();self.cnn.requires_grad_(False)
        for name in self.manifest["checkpoints"]:
            d=torch.load(folder/name,map_location="cpu",weights_only=True)
            if d["format"] not in {"ksl-night-search-v1","ksl-final-refit-v1"}:raise ValueError("Unsupported RGB checkpoint")
            ids=[c.get("class_id",c.get("id")) for c in d["classes"]]
            if len(ids)!=30 or d["feature_dim"]!=1736:raise ValueError("Expected 30-class MobileNet ROI model")
            net=TemporalRGB(1736,30,d["config"]);net.load_state_dict(d["state_dict"]);net.eval()
            self.members.append((net,d["mean"].float(),d["std"].float(),ids))
    def features(self,rois):
        images=rois["images"];positions=rois["sample_positions"];quality=rois["quality"]
        if images.ndim!=5 or images.shape[1:]!=(3,224,224,3) or images.dtype!=np.uint8:raise ValueError("Invalid RGB ROI images")
        x=torch.from_numpy(images.reshape(-1,224,224,3)).permute(0,3,1,2).float()/255
        x=(x-torch.tensor([.485,.456,.406]).view(1,3,1,1))/torch.tensor([.229,.224,.225]).view(1,3,1,1)
        with torch.inference_mode():a=torch.cat([self.cnn(b).flatten(1) for b in x.split(8)]).numpy()
        a=a.reshape(len(images),3,576)
        a[:,0]*=(quality[:,0]>0)[:,None];a[:,1]*=(quality[:,1]>0)[:,None]
        values=np.concatenate([a.reshape(len(images),-1),rois["motion"]],axis=1)[positions].astype(np.float32)
        if values.shape!=(64,1736) or not np.isfinite(values).all():raise ValueError("Invalid RGB features")
        return values
    def predict_features(self,features):
        x=torch.from_numpy(np.asarray(features,dtype=np.float32));values=[]
        for net,mean,std,ids in self.members:
            with torch.inference_mode():p=net(((x-mean)/std)[None]).softmax(1)[0].numpy()
            values.append(align_probabilities(p,ids,self.ids))
        return np.mean(values,axis=0)
    def predict_proba(self,rois):return self.predict_features(self.features(rois))


class FusionRecognizer:
    def __init__(self,keypoint_model=None,rgb_model=None,keypoint_weight=DEFAULT_KEYPOINT_WEIGHT,min_score=.75,min_margin=.15,registry=None):
        if not 0<=keypoint_weight<=1 or not 0<=min_score<=1 or not 0<=min_margin<=1:
            raise ValueError("Weights and thresholds must be in [0,1]")
        torch.set_num_threads(2)
        self.classes=json.loads(Path(registry or ROOT/"config/ksl_classes.json").read_text(encoding="utf-8"))["classes"]
        self.ids=[c["class_id"] for c in self.classes]
        if len(self.ids)!=30 or len(set(self.ids))!=30:raise ValueError("Expected canonical 30 classes")
        self.keypoint=KeypointBranch(keypoint_model or DEFAULT_KEYPOINT_FOLDER)
        self._rgb_folder=Path(rgb_model or ROOT/"models/rgb_ksl30");self._rgb=None
        # Validate IDs before any recognition; list order is aligned by ID.
        align_probabilities(np.ones(30),self.keypoint.ids,self.ids)
        self.weight=keypoint_weight;self.min_score=min_score;self.min_margin=min_margin
    @property
    def rgb(self):
        if self._rgb is None:
            branch=RGBBranch(self._rgb_folder)
            align_probabilities(np.ones(30),branch.ids,self.ids)
            self._rgb=branch
        return self._rgb
    def predict_rois(self,rois,keypoint_frames=None,mode="fusion"):
        if mode not in MODE_LABELS:raise ValueError("Unknown recognition mode")
        errors={};keypoint=None;rgb=None;computed=[]
        if mode!="rgb":
            computed.append("keypoint")
            frames=rois["keypoints"][rois["sample_positions"]] if keypoint_frames is None else keypoint_frames
            try:keypoint=align_probabilities(self.keypoint.predict_proba(frames),self.keypoint.ids,self.ids)
            except ValueError as e:errors["keypoint"]=str(e)
        want_rgb=mode in ("rgb","fusion")
        if mode=="assisted":
            if keypoint is None:raise ValueError("키포인트 입력 부족: CNN만으로 확정하지 않습니다.")
            want_rgb=not recognition_event(keypoint,self.classes,self.min_score,self.min_margin)["accepted"]
        if want_rgb:
            computed.append("rgb")
            try:rgb=align_probabilities(self.rgb.predict_proba(rois),self.rgb.ids,self.ids)
            except (ValueError,RuntimeError,OSError) as e:
                if mode!="assisted" and not isinstance(e,ValueError):raise
                errors["rgb"]=str(e)
        scores,available=fuse_probabilities(keypoint,rgb,self.weight)
        result=recognition_event(scores,self.classes,self.min_score,self.min_margin)
        result.update(available_modalities=available,computed_modalities=computed,component_errors=errors,keypoint_weight=self.weight,
            keypoint_probabilities=keypoint.tolist() if keypoint is not None else None,
            rgb_probabilities=rgb.tolist() if rgb is not None else None,probabilities=scores.tolist())
        return select_prediction(result,self.classes,mode,self.min_score,self.min_margin)
