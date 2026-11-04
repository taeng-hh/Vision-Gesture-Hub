"""Default sign translator: keypoint GRU first, RGB CNN only for uncertain cases."""
import argparse,json
from pathlib import Path
from .predict import FusionRecognizer,DEFAULT_KEYPOINT_WEIGHT,DEFAULT_KEYPOINT_FOLDER,select_prediction,ASSIST_KEYPOINT_WEIGHT
from .shared import SharedExtractor
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera",type=int,default=0)
    p.add_argument("--video",type=Path)
    p.add_argument("--start",type=float,default=0.)
    p.add_argument("--end",type=float)
    p.add_argument("--keypoint-model",type=Path,default=DEFAULT_KEYPOINT_FOLDER)
    p.add_argument("--rgb-model",type=Path,default=ROOT/"models/rgb_ksl30")
    p.add_argument("--keypoint-weight",type=float,default=DEFAULT_KEYPOINT_WEIGHT)
    p.add_argument("--mode",choices=["assisted","keypoint","rgb","fusion"],default="assisted",help="Keypoint-first; CNN checks uncertain cases only")
    p.add_argument("--no-live-reuse",action="store_true",help="Recompute landmarks after input for diagnostic comparison")
    p.add_argument("--diagnose",action="store_true")
    p.add_argument("--no-game",action="store_true",help="Disable the V hand shape -> Galaga shortcut")
    args=p.parse_args()
    model=FusionRecognizer(args.keypoint_model,args.rgb_model,args.keypoint_weight)
    if args.diagnose:
        if args.mode in ("rgb","fusion"):model.rgb
        print(json.dumps(dict(ready=True,classes=len(model.classes),keypoint_model=str(args.keypoint_model),
            rgb_model=str(args.rgb_model),keypoint_weight=model.weight,decision_mode=args.mode,live_keypoint_reuse=not args.no_live_reuse,rgb_loaded=model._rgb is not None,
            decision_keypoint_weight=None if args.mode=="assisted" else 1. if args.mode=="keypoint" else 0. if args.mode=="rgb" else model.weight,
            assisted_keypoint_weight=ASSIST_KEYPOINT_WEIGHT,
            original_teammate_weights=(args.keypoint_model/ "provenance.json").exists() and
                json.loads((args.keypoint_model/"provenance.json").read_text(encoding="utf-8")).get("original_teammate_weights_used",False)),
            ensure_ascii=False,indent=2));return
    missing=[name for name in ("hand_landmarker.task","pose_landmarker_lite.task") if not (ROOT/"models"/name).is_file()]
    if missing:p.error("Missing models/"+", models/".join(missing)+". Run: python scripts/setup_assets.py")
    if args.video:
        rois,quality=SharedExtractor(ROOT/"models").extract(args.video,args.start,args.end)
        result=select_prediction(model.predict_rois(rois,mode=args.mode),model.classes,args.mode,model.min_score,model.min_margin);result["quality"]=dict(any_hand_roi_fraction=quality["any_hand_roi_fraction"])
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    from .live import run_camera
    game=None
    if not args.no_game:
        from .game_switch import GameSwitch
        game=GameSwitch(args.camera,args.keypoint_model,ROOT/"models")
    try:run_camera(model,ROOT/"models",args.camera,mode=args.mode,reuse_live=not args.no_live_reuse,game=game)
    finally:
        if game is not None:game.close()

if __name__=="__main__":main()
