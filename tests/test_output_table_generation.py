import pandas as pd

from physionet_mi.paper.latex_tables import dataframe_to_latex


def test_latex_table_generation():
    df = pd.DataFrame({
        "model": ["fbcsp_lda", "eegnet"],
        "accuracy": [0.71, 0.68],
    })
    tex = dataframe_to_latex(df, "Test table", "tab:test", bold_best_col="accuracy")
    assert "\\begin{table}" in tex
    assert "fbcsp_lda" in tex
