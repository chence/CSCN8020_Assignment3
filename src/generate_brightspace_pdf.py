from __future__ import annotations

import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT = Path("output/pdf/CSCN8020_Assignment3_Brightspace.pdf")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        "This project implements a student-written PyTorch Deep Q-Network to control the Unitree G1 "
        "humanoid robot's left elbow in MuJoCo. The agent observes elbow angle, velocity, goal angle, "
        "and signed error, then selects decrease, hold, or increase actions. Separate online and target "
        "networks, experience replay, epsilon-greedy exploration, Huber loss, and checkpoint loading are "
        "implemented directly. Two epsilon-decay schedules were compared under controlled conditions. "
        "Both trained agents achieved 20 out of 20 successful greedy evaluation episodes across four "
        "target angles. Configuration A was selected and compared fairly with the supplied rule-based "
        "policy. Reproducible code, metrics, plots, checkpoint, report, and execution instructions are included."
    )
    figure = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    figure.text(0.09, 0.91, "CSCN8020 Assignment 3", fontsize=25, weight="bold", color="#17365d")
    figure.text(0.09, 0.865, "Deep Q-Network Control of the Unitree G1 Left Elbow", fontsize=14, color="#1f4e79")
    figure.text(0.09, 0.78, "Student", fontsize=12, weight="bold", color="#1f4e79")
    figure.text(0.30, 0.78, "Ce Chen", fontsize=12)
    figure.text(0.09, 0.735, "Student ID", fontsize=12, weight="bold", color="#1f4e79")
    figure.text(0.30, 0.735, "9007166", fontsize=12)
    figure.text(0.09, 0.65, "Project Summary", fontsize=14, weight="bold", color="#1f4e79")
    figure.text(0.09, 0.605, "\n".join(textwrap.wrap(summary, 88)), fontsize=11, va="top", linespacing=1.55)
    figure.text(0.09, 0.34, "GitHub Repository", fontsize=12, weight="bold", color="#1f4e79")
    figure.text(0.09, 0.30, "https://github.com/chence/CSCN8020_Assignment3", fontsize=11)
    figure.text(0.09, 0.235, "Cloneable URL", fontsize=12, weight="bold", color="#1f4e79")
    figure.text(0.09, 0.195, "https://github.com/chence/CSCN8020_Assignment3.git", fontsize=11)
    figure.text(0.09, 0.09, "Submitted to Brightspace | July 2026", fontsize=9, color="#666666")
    figure.savefig(OUTPUT, format="pdf", bbox_inches=None)
    plt.close(figure)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
