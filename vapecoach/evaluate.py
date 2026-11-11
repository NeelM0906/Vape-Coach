"""Event-level evaluation: labels CSV vs ``hermes vape watch --replay ... --out pred.json``.

Labels CSV columns: ``clip,start_s,end_s,type,notes``. Positive types: ``hit``, ``stealth_hit``.
``unsure``/``ignore`` rows are don't-care spans. Everything else (``sip``, ``chin``, ``snack``, ``phone``, ``scratch``, ``other``) is a look-alike
used to explain false positives. A prediction matches a positive label when its ``t_start`` falls
in ``[start_s - tol, end_s + tol]``; matching is one-to-one, earliest first.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

POSITIVE = {"hit", "stealth_hit"}
DONT_CARE = {"unsure", "ignore"}   # ambiguous spans: a prediction there is neither a TP nor an FP


def load_labels(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("clip") or row["clip"].startswith("#"):
                continue
            out[Path(row["clip"]).stem].append({
                "start": float(row["start_s"]), "end": float(row["end_s"]),
                "type": (row.get("type") or "hit").strip().lower(), "notes": row.get("notes", "")})
    return out


def score_clip(labels: list[dict], preds: list[float], tol: float) -> dict:
    pos = sorted([l for l in labels if l["type"] in POSITIVE], key=lambda l: l["start"])
    neg = [l for l in labels if l["type"] not in POSITIVE and l["type"] not in DONT_CARE]
    dont = [l for l in labels if l["type"] in DONT_CARE]
    used = set()
    tp, fp_times, matched_types, ignored = 0, [], Counter(), 0
    for t in sorted(preds):
        hit = next((i for i, l in enumerate(pos) if i not in used and l["start"] - tol <= t <= l["end"] + tol), None)
        if hit is None and any(l["start"] - tol <= t <= l["end"] + tol for l in dont):
            ignored += 1
        elif hit is None:
            fp_times.append(t)
        else:
            used.add(hit)
            tp += 1
            matched_types[pos[hit]["type"]] += 1
    fp_cause = Counter()
    for t in fp_times:
        cause = next((l["type"] for l in neg if l["start"] - tol <= t <= l["end"] + tol), "unlabeled")
        fp_cause[cause] += 1
    return {"tp": tp, "fp": len(fp_times), "fn": len(pos) - tp, "n_pos": len(pos), "n_pred": len(preds) - ignored,
            "ignored_dont_care": ignored, "fp_times": [round(t, 1) for t in fp_times],
            "missed": [[l["start"], l["end"], l["type"]] for i, l in enumerate(pos) if i not in used],
            "pos_by_type": dict(Counter(l["type"] for l in pos)), "tp_by_type": dict(matched_types),
            "fp_by_cause": dict(fp_cause), "lookalikes": dict(Counter(l["type"] for l in neg))}


def evaluate(labels: dict[str, list[dict]], predictions: dict[str, dict], tolerance: float = 2.0) -> dict:
    per_clip, totals, seconds = {}, Counter(), 0.0
    tp_type, pos_type, fp_cause, look = Counter(), Counter(), Counter(), Counter()
    abs_count_err = []
    for clip, pred in predictions.items():
        r = score_clip(labels.get(clip, []), [e["t_start"] for e in pred.get("events", [])], tolerance)
        per_clip[clip] = r
        for k in ("tp", "fp", "fn", "n_pos", "n_pred"):
            totals[k] += r[k]
        tp_type.update(r["tp_by_type"]); pos_type.update(r["pos_by_type"])
        fp_cause.update(r["fp_by_cause"]); look.update(r["lookalikes"])
        abs_count_err.append(abs(r["n_pred"] - r["n_pos"]))
        seconds += float((pred.get("metrics") or {}).get("media_s") or 0.0)
    missing = sorted(set(labels) - set(predictions))
    p = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else None
    r_ = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else None
    f1 = 2 * p * r_ / (p + r_) if p and r_ else None
    return {
        "clips": len(predictions), "hours": round(seconds / 3600, 3),
        "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"],
        "precision": _r(p), "recall": _r(r_), "f1": _r(f1),
        "fp_per_hour": _r(totals["fp"] / (seconds / 3600)) if seconds else None,
        "count_error_total": totals["n_pred"] - totals["n_pos"],
        "count_error_rel": _r((totals["n_pred"] - totals["n_pos"]) / totals["n_pos"]) if totals["n_pos"] else None,
        "mean_abs_count_error_per_clip": _r(sum(abs_count_err) / len(abs_count_err)) if abs_count_err else None,
        "recall_by_type": {t: _r(tp_type[t] / n) for t, n in pos_type.items()},
        "fp_by_cause": dict(fp_cause), "lookalikes_in_labels": dict(look),
        "unlabeled_or_unpredicted_clips": missing, "per_clip": per_clip, "tolerance_s": tolerance,
    }


def _r(x):
    return None if x is None else round(x, 3)


def rescore(pred: dict, threshold: float) -> dict:
    """Re-threshold one replay offline: accepted + rejected candidates with score >= threshold.

    Candidates rejected for a hard reason (dwell too short/long, wide jaw) have score 0 and stay out.
    """
    cands = list(pred.get("events", [])) + list(pred.get("rejected", []))
    return {**pred, "events": [c for c in cands if c.get("score", 0) >= threshold and c.get("score", 0) > 0]}


def sweep(labels: dict, preds: dict, tolerance: float, thresholds=None) -> list[dict]:
    rows = []
    for thr in thresholds or [x / 100 for x in range(35, 90, 5)]:
        r = evaluate(labels, {k: rescore(v, thr) for k, v in preds.items()}, tolerance)
        rows.append({"threshold": thr, **{k: r[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1", "fp_per_hour")}})
    return rows


def load_predictions(pred_files: Iterable[Path]) -> dict:
    preds = {}
    for f in pred_files:
        d = json.loads(Path(f).read_text())
        preds[Path(d.get("source") or f).stem] = d
    return preds


def evaluate_files(labels_csv: Path, pred_files: Iterable[Path], tolerance: float = 2.0,
                   do_sweep: bool = False) -> dict:
    labels = load_labels(labels_csv)
    preds = load_predictions(pred_files)
    out = evaluate(labels, preds, tolerance)
    if do_sweep:
        out["threshold_sweep"] = sweep(labels, preds, tolerance)
    return out
