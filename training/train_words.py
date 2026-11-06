"""Train a lightweight sequence baseline and evaluate on a held-out signer."""
import argparse
import json
from pathlib import Path
from core.ksl import SequenceModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ksl_clips"))
    parser.add_argument("--validation-signer", required=True)
    parser.add_argument("--output", type=Path, default=Path("models/ksl_words.npz"))
    args = parser.parse_args()
    samples = []
    for path in sorted(args.data.glob("*.json")):
        sample = json.loads(path.read_text(encoding="utf-8"))
        if (sample.get("version") != 1 or not isinstance(sample.get("label"), str)
                or not sample["label"].strip() or not isinstance(sample.get("signer"), str)
                or not sample["signer"].strip() or sample.get("mirrored") is not True):
            raise ValueError(f"Invalid clip metadata: {path}")
        samples.append(sample)
    train = [s for s in samples if s["signer"] != args.validation_signer]
    validation = [s for s in samples if s["signer"] == args.validation_signer]
    if not validation:
        parser.error("No clips for validation signer")
    model = SequenceModel.fit(train)
    labels = set(model.labels)
    if {s["label"] for s in validation} != labels:
        parser.error("Validation signer must have clips for every training class (and no extra classes)")
    confusion = {}
    correct = rejected = 0
    for sample in validation:
        prediction = model.predict(sample["frames"])["label"]
        correct += prediction == sample["label"]
        rejected += prediction is None
        row = confusion.setdefault(sample["label"], {})
        name = prediction or "<REJECT>"
        row[name] = row.get(name, 0) + 1
    report = {"validation_signer": args.validation_signer,
              "train_clips": len(train), "validation_clips": len(validation),
              "accuracy_including_rejections": correct / len(validation),
              "rejected": rejected, "confusion_matrix": confusion,
              "warning": "Known-word evaluation only; rejection thresholds require unknown-gesture testing."}
    model.save(args.output)
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
