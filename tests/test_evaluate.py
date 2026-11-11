import json

from vapecoach.evaluate import evaluate, evaluate_files, load_labels


def test_event_level_metrics(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "clip,start_s,end_s,type,notes\n"
        "clipA.mp4,10,13,hit,\n"
        "clipA.mp4,40,42,stealth_hit,\n"
        "clipA.mp4,60,63,sip,coffee\n"
        "clipA.mp4,90,95,hit,\n")
    pred = tmp_path / "clipA.json"
    pred.write_text(json.dumps({"source": "/x/clipA.mp4", "metrics": {"media_s": 1800},
                                "events": [{"t_start": 11.0}, {"t_start": 61.5}, {"t_start": 92.0},
                                           {"t_start": 93.0}]}))
    r = evaluate_files(labels, [pred], tolerance=2.0)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 2, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.667
    assert r["fp_per_hour"] == 4.0
    assert r["fp_by_cause"] == {"sip": 1, "unlabeled": 1}
    assert r["recall_by_type"] == {"hit": 1.0, "stealth_hit": 0.0}
    assert r["count_error_total"] == 1


def test_template_parses():
    from pathlib import Path
    tpl = Path(__file__).resolve().parent.parent / "eval" / "labels_template.csv"
    labels = load_labels(tpl)
    assert labels and all(l["end"] >= l["start"] for v in labels.values() for l in v)


def test_dont_care_span_is_neither_tp_nor_fp():
    from vapecoach.evaluate import score_clip
    labels = [{"start": 10, "end": 12, "type": "hit", "notes": ""},
              {"start": 30, "end": 32, "type": "unsure", "notes": ""}]
    r = score_clip(labels, [11.0, 31.0, 50.0], tol=2.0)
    assert (r["tp"], r["fp"], r["fn"], r["ignored_dont_care"]) == (1, 1, 0, 1)
    assert r["fp_times"] == [50.0]

