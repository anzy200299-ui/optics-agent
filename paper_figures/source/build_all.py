"""Build all six paper figures in the hand-drawn (FEAST) style.

Run:  cd paper_figures/source && python3 build_all.py
Outputs land in /tmp/paper_figures/ (style.save); copy the PDFs into
paper_figures/ afterwards (see build_all.py tail / Makefile notes).
"""

from __future__ import annotations

import fig1
import fig2
import fig3
import fig4
import fig5
import fig6
from style import setup


def main() -> None:
    setup()
    fig1.build()
    fig2.build()
    fig3.build()
    fig4.build()
    fig5.build()
    fig6.build()
    print("built fig1..fig6 into /tmp/paper_figures/")


if __name__ == "__main__":
    main()
