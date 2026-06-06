import os

from physionet_mi.evaluation.parallel_runner import (
    effective_inner_n_jobs,
    partition_models,
)


def test_partition_models():
    classical, deep = partition_models(
        ["fbcsp_lda", "csp_svm", "eegnet", "riemann_mdm", "eegme"]
    )
    assert classical == ["fbcsp_lda", "csp_svm", "riemann_mdm"]
    assert deep == ["eegnet", "eegme"]


def test_effective_inner_n_jobs_serial():
    assert effective_inner_n_jobs(1) == -1


def test_effective_inner_n_jobs_parallel():
    n = effective_inner_n_jobs(4)
    assert n >= 1
    assert n <= (os.cpu_count() or 1)
