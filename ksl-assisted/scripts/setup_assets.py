"""Download official runtime assets with pinned SHA256 checks."""
from pathlib import Path
from urllib.request import urlopen
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path);p.add_argument("--check-only",action="store_true")
    args=p.parse_args()
    for asset in json.loads((ROOT/"assets_manifest.json").read_text(encoding="utf-8")):
        path=ROOT/asset["path"]
        if path.is_file() and digest(path)==asset["sha256"]:print("Ready:",asset["path"]);continue
        if args.check_only:raise RuntimeError("Missing or invalid asset: "+asset["path"])
        path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+".download")
        try:
            if args.source:
                source=args.source/asset["source"]
                if not source.is_file() or digest(source)!=asset["sha256"]:raise RuntimeError("Invalid source asset: "+asset["path"])
                shutil.copyfile(source,temp)
            else:
                print("Downloading:",asset["path"])
                with urlopen(asset["url"],timeout=120) as response,temp.open("wb") as out:
                    shutil.copyfileobj(response,out,1024*1024)
            if digest(temp)!=asset["sha256"]:raise RuntimeError("SHA256 mismatch: "+asset["path"])
            temp.replace(path);print("Ready:",asset["path"])
        finally:temp.unlink(missing_ok=True)
if __name__=="__main__":main()
