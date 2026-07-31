from __future__ import annotations

import csv
import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


OUTPUT = Path("report/DQN_Assignment_Report.pdf")


def add_text_page(pdf: PdfPages, title: str, sections: list[tuple[str, str]]) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.patch.set_facecolor("white")
    figure.text(0.09, 0.94, title, fontsize=20, weight="bold", color="#17365d")
    y = 0.89
    for heading, body in sections:
        figure.text(0.09, y, heading, fontsize=13, weight="bold", color="#1f4e79")
        y -= 0.035
        wrapped = "\n".join(textwrap.wrap(body, width=98))
        figure.text(0.09, y, wrapped, fontsize=10.2, va="top", linespacing=1.45)
        y -= 0.029 * (wrapped.count("\n") + 1) + 0.035
    figure.text(0.5, 0.035, "CSCN8020 Assignment 3 - Unitree G1 DQN", ha="center", fontsize=8)
    pdf.savefig(figure)
    plt.close(figure)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, caption: str) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.09, 0.94, title, fontsize=20, weight="bold", color="#17365d")
    axis = figure.add_axes((0.08, 0.22, 0.84, 0.64))
    axis.imshow(plt.imread(image_path))
    axis.axis("off")
    figure.text(0.09, 0.16, "\n".join(textwrap.wrap(caption, 100)), fontsize=10, va="top")
    figure.text(0.5, 0.035, "CSCN8020 Assignment 3 - Unitree G1 DQN", ha="center", fontsize=8)
    pdf.savefig(figure)
    plt.close(figure)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add_table_page(
    pdf: PdfPages,
    title: str,
    columns: list[str],
    rows: list[list[str]],
    discussion: str,
) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.09, 0.94, title, fontsize=20, weight="bold", color="#17365d")
    axis = figure.add_axes((0.06, 0.48, 0.88, 0.34))
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1, 1.8)
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
    figure.text(0.09, 0.39, "Interpretation", fontsize=13, weight="bold", color="#1f4e79")
    figure.text(
        0.09,
        0.355,
        "\n".join(textwrap.wrap(discussion, 100)),
        fontsize=10.2,
        va="top",
        linespacing=1.45,
    )
    figure.text(0.5, 0.035, "CSCN8020 Assignment 3 - Unitree G1 DQN", ha="center", fontsize=8)
    pdf.savefig(figure)
    plt.close(figure)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    comparison = read_rows(Path("results/epsilon_decay_comparison.csv"))
    evaluation = read_rows(Path("results/config_a/dqn_evaluation_summary.csv"))
    policy = read_rows(Path("results/rule_based_vs_selected_dqn.csv"))

    with PdfPages(OUTPUT) as pdf:
        add_text_page(
            pdf,
            "Deep Q-Network Control of the\nUnitree G1 Left Elbow",
            [
                ("Assignment", "CSCN8020 Reinforcement Learning - Assignment 3"),
                ("Scope", "A student-written PyTorch DQN for multi-goal discrete control in the supplied fixed-base Unitree G1 MuJoCo environment."),
                ("Outcome", "Both controlled epsilon-decay experiments achieved 20/20 successful greedy evaluation episodes. Configuration A (decay 0.995) was selected because it combined 100% success with the highest mean evaluation reward and lowest final absolute error."),
                ("Student information", "Name: Ce Chen    Student ID: 9007166    Date: July 23, 2026"),
                ("AI assistance disclosure", "I used OpenAI Codex as a development aid for code organization, debugging, automated testing, plot generation, and editorial review. I reviewed and executed all submitted code, verified the reported results, and remain responsible for understanding and explaining the DQN implementation, experimental methodology, interpretation, and final demonstration."),
            ],
        )
        add_text_page(
            pdf,
            "1. Introduction and Environment",
            [
                ("Primer connection", "The G1 Primer established the fixed-base model, left-elbow PD controller, MuJoCo bias-force compensation, Gymnasium interface, success condition, and rule-based validation policy. This assignment replaces only the high-level three-action decision rule with a learned action-value policy."),
                ("Observation", "The state is [current elbow angle, current elbow velocity, goal angle, goal minus current angle]. Supplying both the goal and signed error allows one network to learn across the full training goal range of -0.8 to +0.8 radians."),
                ("Actions", "Action 0 decreases the internal controller target, action 1 holds it, and action 2 increases it. The approved low-level controller converts the target into bounded torque while compensating qfrc_bias."),
                ("Reward and success", "The reward is negative absolute error, plus a success-region bonus and a small penalty for changing the target while close. Success requires error within 0.04 radians for eight consecutive environment steps and adds a terminal bonus of 10."),
                ("Termination", "A successful episode produces terminated=True. Reaching the 150-step time limit produces truncated=True. Evaluation success is taken only from the true success termination."),
            ],
        )
        add_text_page(
            pdf,
            "2. DQN Architecture and Bellman Update",
            [
                ("Network", "The online and target networks each use Linear(4,64), ReLU, Linear(64,64), ReLU, and Linear(64,3). No softmax is applied because Q-values are unconstrained estimates of discounted return."),
                ("Replay", "A bounded deque stores 50,000 transitions. Sampling uses uniformly random batches of 64. Learning begins only after a 500-transition warm-up, reducing correlation and satisfying the required minimum."),
                ("Bellman target", "For a sampled transition, the selected online value Q(s,a) is compared with y = r + gamma(1-terminated) max_a' Q_target(s',a'), where gamma is 0.95. Targets are calculated under no_grad so gradients cannot flow through the target network."),
                ("Truncation treatment", "Time-limit truncation does not mask bootstrap because it is an artificial horizon rather than a terminal condition of the robot task. True successful termination does mask bootstrap. Both flags still end the data-collection episode."),
                ("Optimization", "Adam uses learning rate 0.001 and Smooth L1 (Huber) loss. Gradients are clipped to norm 10. The target network starts as a copy of the online network and is synchronized every 250 optimization steps."),
            ],
        )
        add_text_page(
            pdf,
            "3. Training and Reproducibility",
            [
                ("Protocol", "Both configurations trained headlessly for 600 episodes on CPU. Python, NumPy, PyTorch, replay sampling, and Gymnasium resets were seeded from 42. Each episode used a deterministic reset seed of 42 plus the zero-based episode index."),
                ("Controlled study", "Configuration A used epsilon decay 0.995 and Configuration B used 0.985. Both began at 1.0 and were bounded below by 0.05. Gamma, learning rate, batch size, replay capacity, warm-up, network shape, optimizer, target-update interval, goal distribution, episode horizon, and seed policy were identical."),
                ("Metrics", "The training CSV records episode, seed, sampled goal, cumulative reward, success, length, final error, epsilon, mean episode loss, optimization steps, and elapsed seconds. Checkpoints contain model weights, optimizer state, configuration, dimensions, steps, and experiment metadata."),
                ("Commands", "Training is run with PYTHONPATH=src python src/train_dqn.py --config-name config_a (or config_b) --episodes 600 --seed 42. Evaluation uses evaluate_dqn.py and epsilon 0.0. Exact commands are preserved in README.md."),
            ],
        )
        add_text_page(
            pdf,
            "4. Finding the Submitted Outputs",
            [
                ("Fastest performance check", "Open results/config_a/dqn_evaluation_summary.csv. The overall row is the shortest proof of final performance: 20 successes in 20 greedy episodes, mean reward 13.278, mean episode length 19.75, and mean final absolute error 0.00676 radians. The four preceding rows show that success covers every required goal rather than one favorable target."),
                ("Training evidence", "Open results/config_a/training_plots.png for reward, rolling success, epsilon, and Huber-loss diagnostics. The underlying per-episode values are in training_metrics.csv. The matching files under results/config_b document the controlled alternative epsilon schedule."),
                ("Comparisons", "Open results/epsilon_decay_comparison.csv or .png to compare the two training configurations. Open results/rule_based_vs_selected_dqn.csv to compare the selected learned policy with the supplied rule-based policy under the same goals and seed schedule."),
                ("Model and demonstration", "models/selected_dqn.pt is the reloadable Configuration A checkpoint used for final evaluation and rendering. src/evaluate_dqn.py reproduces the numeric evaluation, and src/render_dqn_policy.py loads the same checkpoint for the MuJoCo Viewer without retraining. README.md provides platform-specific commands and the rendered-video link."),
            ],
        )
        add_text_page(
            pdf,
            "5. How to Read the Metrics",
            [
                ("Reward and success", "Cumulative reward should increase as the policy spends less time far from the target and earns the success bonuses. The primary task metric is success rate: the elbow must remain within 0.04 radians of the goal for eight consecutive environment steps. A stable 100% rolling rate indicates consistent training behavior."),
                ("Speed and precision", "Episode length is meaningful only together with success: among successful policies, a shorter episode means the goal was reached and held sooner. Final absolute error reports terminal precision in radians; lower is better."),
                ("Exploration", "Epsilon is the probability of selecting a random action during training. Its curve verifies the intended exploration schedule. Final evaluation uses epsilon 0.0, so the reported 20/20 result measures the learned greedy policy rather than lucky exploratory actions."),
                ("Why loss can rise", "Huber loss measures temporal-difference error, not task success. DQN targets are non-stationary because the online policy, target network, replay-buffer distribution, and terminal-bonus samples change during training. Therefore, a later loss increase is not automatically degradation. Here, stable reward, stable 100% rolling success, and the separate 20/20 greedy evaluation provide the behavioral evidence that the controller works."),
            ],
        )
        rows = []
        for row in comparison:
            rows.append([
                row["configuration"], row["epsilon_decay"], row["training_episodes"],
                f"{float(row['training_seconds']):.2f}", f"{float(row['final_epsilon']):.2f}",
                f"{float(row['final_20_mean_training_reward']):.3f}",
                f"{float(row['final_50_training_success_rate']):.0%}",
                f"{float(row['evaluation_success_rate']):.0%}",
            ])
        add_table_page(
            pdf,
            "6. Exploration-Decay Study",
            ["Config", "Decay", "Episodes", "Seconds", "Final eps", "Reward-20", "Success-50", "Greedy eval"],
            rows,
            "Both settings converged to a 100% rolling training success rate and 100% greedy evaluation success. B reached minimum exploration much earlier and trained faster, while A retained diverse exploration longer and finished with a slightly higher final-20 mean training reward. The absence of late failures suggests stable learning for both settings.",
        )
        add_image_page(
            pdf,
            "7. Training Curves",
            Path("results/epsilon_decay_comparison.png"),
            "The moving averages show the controlled comparison. Faster decay improves early exploitation, while both curves stabilize at full success. Individual reward, epsilon, loss, and success plots for each configuration are included in the results directories.",
        )
        evaluation_rows = [
            [r["goal"], r["episodes"], r["successes"], f"{float(r['success_rate']):.0%}", f"{float(r['mean_reward']):.3f}"]
            for r in evaluation
        ]
        add_table_page(
            pdf,
            "8. Selected DQN Evaluation",
            ["Goal (rad)", "Episodes", "Successes", "Success rate", "Mean reward"],
            evaluation_rows,
            "Configuration A succeeded in all 20 required greedy episodes, exceeding the 80% threshold. It generalized symmetrically to positive and negative goals. Overall mean reward was 13.278, mean episode length was 19.75, and mean final absolute error was 0.00676 radians. Greedy action counts are retained in the episode-level CSV for HOLD and oscillation analysis.",
        )
        policy_rows = [
            [r["policy"], f"{r['successes']}/{r['episodes']}", f"{float(r['success_rate']):.0%}", f"{float(r['mean_cumulative_reward']):.3f}", f"{float(r['mean_episode_length']):.2f}", f"{float(r['mean_final_absolute_error']):.5f}"]
            for r in policy
        ]
        add_table_page(
            pdf,
            "9. Rule-Based Baseline and Discussion",
            ["Policy", "Successes", "Rate", "Mean reward", "Mean length", "Mean final error"],
            policy_rows,
            "Both policies were perfectly successful. The selected DQN was faster on average, earned higher reward, and ended closer to the goal. The rule-based controller is nevertheless far more sample efficient because it requires no training and directly encodes target direction. The DQN learned the same useful pattern: move the controller target toward the goal, then use HOLD so the physical joint can settle. Remaining limitations include evaluation on only four benchmark goals, one training seed, simulation-only evidence, and sensitivity to reward shaping. Future work should repeat multiple seeds, test denser unseen goals, study Double DQN or prioritized replay, and quantify action switching near the target. Configuration A is recommended because its equal success, higher reward, and lower final error outweigh B's modest speed advantage.",
        )

    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
