import json

import pandas as pd

from physionet_mi.evaluation.rebuild_holdout_csvs import rebuild_repeated_holdout_artifacts


def _write_run(tmp_path, name: str, *, bal_acc: float, use_ea: bool, repeat_id: int):
    run = tmp_path / name
    run.mkdir()
    (run / "metrics.json").write_text(
        json.dumps({
            "accuracy": bal_acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": bal_acc,
            "kappa": 0.1,
            "model": "fbcsp_lda",
            "use_ea": use_ea,
            "dataset": "physionet",
            "status": "ok",
        }),
        encoding="utf-8",
    )
    (run / "split_metadata.json").write_text(
        json.dumps({
            "dataset": "physionet",
            "master_seed": 42,
            "n_repeats": 2,
            "repeat_id": repeat_id,
            "split_seed": 100 + repeat_id,
            "model_seed": 100 + repeat_id,
            "train": {"n_subjects": 10, "n_trials": 100},
            "test": {"n_subjects": 2, "n_trials": 20},
        }),
        encoding="utf-8",
    )


def test_rebuild_from_run_folders(tmp_path):
    _write_run(tmp_path, "fbcsp_lda_ea_repeat0", bal_acc=0.7, use_ea=True, repeat_id=0)
    _write_run(tmp_path, "fbcsp_lda_no_ea_repeat0", bal_acc=0.6, use_ea=False, repeat_id=0)
    _write_run(tmp_path, "fbcsp_lda_ea_repeat1", bal_acc=0.8, use_ea=True, repeat_id=1)

    df = rebuild_repeated_holdout_artifacts(tmp_path)
    assert len(df) == 3
    assert set(df["model"]) == {"fbcsp_lda"}
    assert (tmp_path / "repeated_holdout_results.csv").exists()
    summary = pd.read_csv(tmp_path / "repeated_holdout_summary.csv")
    assert len(summary) == 2
    assert not (tmp_path / "failed_runs.csv").exists()


def test_rebuild_clears_stale_failed_runs(tmp_path):
    _write_run(tmp_path, "fbcsp_lda_ea_repeat0", bal_acc=0.7, use_ea=True, repeat_id=0)
    stale = tmp_path / "failed_runs.csv"
    stale.write_text("model,status\neegnet,error\n", encoding="utf-8")

    rebuild_repeated_holdout_artifacts(tmp_path)
    assert not stale.exists()
