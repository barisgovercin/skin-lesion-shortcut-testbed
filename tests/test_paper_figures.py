"""The manuscript's shortcut figures (Figs 2 to 4) regenerate from the shipped results."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

PAPER_FIGURES = ["Fig2shortcut_ham_vs_isic_dose", "Fig3shortcut_ham_vs_isic_bakeoff",
                 "Fig4a_shortcut_ham10000_llm_vs_template_dose",
                 "Fig4b_shortcut_ham10000_llm_vs_template_bakeoff"]


def test_paper_figures_are_written_as_pdf_and_png(tmp_path):
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_paper_figures.py"),
                          "--out", str(tmp_path)], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    for name in PAPER_FIGURES:
        for ext in ("pdf", "png"):
            path = tmp_path / f"{name}.{ext}"
            assert path.exists() and path.stat().st_size > 5000, path
