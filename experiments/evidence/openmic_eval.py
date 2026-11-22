"""Both audio models against OpenMIC-2018's human guitar answers, each model on its own.

OpenMIC clips are ten seconds of Free Music Archive tracks; neither model was trained on
them. Reference: clips where guitar was put to annotators and they agreed (see
``gmv.evidence.openmic`` for the rule). Clips where guitar was not asked about are not used.

Reported for the whole set and for the dataset's own test partition, and, among guitar-free
clips, separately for those where annotators agreed another instrument is present (piano,
synthesizer, bass, violin, cello, organ, banjo, mandolin, ukulele): that is where a guitar
detector is expected to be fooled.

Runs in the application environment.

    python experiments/evidence/openmic_eval.py --root E:/datasets/openmic-2018 \
        --panns research-local/evidence-v1/panns/openmic_guitar.npz --report OUT.json
"""

import argparse
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from compare_models import auc, recall_at  # noqa: E402

from gmv.evidence import openmic  # noqa: E402
from gmv.recognition import downmix, levelled, load_backend  # noqa: E402

CONFUSERS = ("piano", "synthesizer", "organ", "bass", "violin", "cello", "banjo", "mandolin", "ukulele")


def own_scores(backend, path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = downmix(audio)
    divisor = gcd(rate, backend.sample_rate)
    mono = resample_poly(mono, backend.sample_rate // divisor, rate // divisor).astype(np.float32)
    size = round(backend.context_s * backend.sample_rate)
    mono = mono[-size:] if len(mono) >= size else np.pad(mono, (size - len(mono), 0))
    return backend.infer(levelled(mono))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--panns", required=True)
    parser.add_argument("--model", default=".model-cache/guitar_mn10_v1.json")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    clips = openmic.read(args.root)
    backend = load_backend(args.model)
    thresholds = json.loads(Path(args.model).read_text(encoding="utf-8"))["thresholds"]["guitar"]
    with np.load(args.panns, allow_pickle=False) as data:
        classes = [str(c) for c in data["classes"]]
        panns = dict(zip((str(k) for k in data["keys"]), data["mean"], strict=True))
    cache = Path(args.report).with_suffix(".gmv.npz")
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            own = dict(zip((str(k) for k in data["keys"]), data["scores"], strict=True))
    else:
        own = {key: own_scores(backend, openmic.audio_path(args.root, key)) for key in panns}
        np.savez_compressed(cache, keys=np.array(list(own)), scores=np.array(list(own.values())))
    names = {
        "gmv_guitar": lambda k: own[k][0],
        "panns_guitar": lambda k: panns[k][classes.index("guitar")],
        "panns_electric_guitar": lambda k: panns[k][classes.index("electric_guitar")],
        "panns_acoustic_guitar": lambda k: panns[k][classes.index("acoustic_guitar")],
        "panns_plucked_string": lambda k: panns[k][classes.index("plucked_string")],
    }

    def describe(keys):
        pos = [k for k in keys if clips[k]["labels"]["guitar"]["verdict"] == "present"]
        neg = [k for k in keys if clips[k]["labels"]["guitar"]["verdict"] == "absent"]
        out = {"guitar_clips": len(pos), "guitar_free_clips": len(neg), "models": {}}
        for name, value in names.items():
            p, n = np.array([value(k) for k in pos]), np.array([value(k) for k in neg])
            out["models"][name] = {
                "auc": round(auc(p, n), 3),
                "recall_at_5pct_false_alarms": round(recall_at(p, n, 0.05), 3),
                "recall_at_10pct_false_alarms": round(recall_at(p, n, 0.10), 3),
            }
        p = np.array([own[k][0] for k in pos])
        n = np.array([own[k][0] for k in neg])
        out["gmv_at_shipped_thresholds"] = {
            "detected": {
                "recall": round(float((p >= thresholds["detected"]).mean()), 3),
                "false_alarms": round(float((n >= thresholds["detected"]).mean()), 3),
            },
            "detected_or_candidate": {
                "recall": round(float((p >= thresholds["candidate"]).mean()), 3),
                "false_alarms": round(float((n >= thresholds["candidate"]).mean()), 3),
            },
        }
        return out, pos, neg

    keys = sorted(panns)
    report = {"rule": openmic.__doc__.split("What a label means here:")[1].split("The dataset")[0].strip()}
    report["all_clips"], pos, neg = describe(keys)
    report["test_partition"], _, _ = describe([k for k in keys if clips[k]["split"] == "test"])
    # Which other instruments make each model fire without a guitar?
    confusion = {}
    for instrument in CONFUSERS:
        some = [k for k in neg if clips[k]["labels"].get(instrument, {}).get("verdict") == "present"]
        if len(some) < 8:
            continue
        confusion[instrument] = {
            "guitar_free_clips_with_it": len(some),
            "gmv_mean_score": round(float(np.mean([own[k][0] for k in some])), 3),
            "gmv_share_detected_or_candidate": round(
                float(np.mean([own[k][0] >= thresholds["candidate"] for k in some])), 3
            ),
            "panns_guitar_mean_score": round(
                float(np.mean([panns[k][classes.index("guitar")] for k in some])), 3
            ),
        }
    report["guitar_free_clips_by_other_instrument"] = confusion
    # Agreement as a selector: how pure are the clips both models are sure about?
    g = np.array([own[k][0] for k in keys])
    q = np.array([panns[k][classes.index("guitar")] for k in keys])
    truth = np.array([clips[k]["labels"]["guitar"]["verdict"] == "present" for k in keys])
    agreement = {}
    for name, own_cut, panns_cut in (("strict", 0.926, 0.5), ("moderate", 0.627, 0.3)):
        both = (g >= own_cut) & (q >= panns_cut)
        agreement[f"both_say_guitar_{name}"] = {
            "gmv_at_least": own_cut,
            "panns_at_least": panns_cut,
            "clips": int(both.sum()),
            "share_of_guitar_clips_covered": round(float(both[truth].mean()), 3),
            "share_truly_guitar": round(float(truth[both].mean()), 3) if both.any() else None,
        }
    for name, own_cut, panns_cut in (("strict", 0.05, 0.02), ("moderate", 0.2, 0.05)):
        both = (g <= own_cut) & (q <= panns_cut)
        agreement[f"both_say_no_guitar_{name}"] = {
            "gmv_at_most": own_cut,
            "panns_at_most": panns_cut,
            "clips": int(both.sum()),
            "share_of_guitar_free_clips_covered": round(float(both[~truth].mean()), 3),
            "share_truly_guitar_free": round(float((~truth[both]).mean()), 3) if both.any() else None,
        }
    report["agreement_between_models"] = agreement
    Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rule"}, indent=1))


if __name__ == "__main__":
    main()
