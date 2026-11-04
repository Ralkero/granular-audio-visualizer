import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline, candidate = (json.loads(Path(p).read_text()) for p in (args.baseline, args.candidate))
    old = {r["fixture"]["id"]: r for r in baseline["fixtures"]}
    rows = []
    for result in candidate["fixtures"]:
        ref = old[result["fixture"]["id"]]
        if result["fixture"]["sha256_float32_le"] != ref["fixture"]["sha256_float32_le"]:
            raise ValueError("Fixture mismatch; comparison is invalid")
        rows.append((result["fixture"]["category"], ref, result))
    lines = [
        "# Frozen baseline versus candidate",
        "",
        f"Split: {candidate['split']}. Synthetic scores do not establish real-music instrument accuracy.",
        "",
        "| Fixture | F1 ±50 ms baseline → candidate | F1 ±25 ms baseline → candidate | FP baseline → candidate | FN baseline → candidate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, old, new in rows:
        a, b = old["onsets"], new["onsets"]
        lines.append(
            f"| {name} | {a['50']['f1']:.4f} → {b['50']['f1']:.4f} | {a['25']['f1']:.4f} → {b['25']['f1']:.4f} | {a['50']['fp']} → {b['50']['fp']} | {a['50']['fn']} → {b['50']['fn']} |"
        )
    lines.extend(
        [
            "",
            "Aggregate counts and timing errors:",
            "```json",
            json.dumps(
                {"baseline": baseline["aggregate"], "candidate": candidate["aggregate"]}, indent=2
            ),
            "```",
            "",
            "Per-class negatives, tempo ambiguity and beat continuity are in the corresponding JSON files; aggregate onset F1 is not a substitute for those checks.",
        ]
    )
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
