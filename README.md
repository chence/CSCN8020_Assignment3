# CSCN8020 Assignment 3 - Unitree G1 DQN

**Student:** Ce Chen<br>
**Student ID:** 9007166<br>
**Repository:** https://github.com/chence/CSCN8020_Assignment3<br>
**Clone URL:** https://github.com/chence/CSCN8020_Assignment3.git<br>
**Validated environment:** Python 3.12.13 on macOS (Apple Silicon), CPU execution

## Project Summary

This project implements a student-written PyTorch Deep Q-Network to control the
Unitree G1 humanoid robot's left elbow in MuJoCo. The agent observes elbow angle,
velocity, goal angle, and signed error, then selects decrease, hold, or increase
actions. Separate online and target networks, experience replay, epsilon-greedy
exploration, Huber loss, and checkpoint loading are implemented directly. Two
epsilon-decay schedules are compared under controlled conditions. Both trained
agents achieved 20/20 successful greedy evaluation episodes across four target
angles. Configuration A was selected and compared fairly with the supplied
rule-based policy.

## Rendered Evaluation Video

[Watch or download the Unitree G1 DQN evaluation video](https://raw.githubusercontent.com/chence/CSCN8020_Assignment3_Video/refs/heads/master/CSCN8020_Assignment3.mp4)

The video loads `models/selected_dqn.pt` without retraining, reviews the
experiment and benchmark evidence, and demonstrates the greedy DQN policy in
the MuJoCo Viewer across the required target angles.

## Start Here: Assignment Evidence

No retraining is required to inspect the submitted evidence.

| Question | Open this file | What to look for |
|---|---|---|
| Did the selected DQN succeed? | `results/config_a/dqn_evaluation_summary.csv` | The `overall` row reports 20/20 successes, a 1.0 success rate, mean reward 13.278, and mean final error 0.00676 rad. |
| Did learning improve during training? | `results/config_a/training_plots.png` | Reward rises and stabilizes; the rolling success rate reaches 1.0 and remains there. |
| Which epsilon schedule was selected? | `results/epsilon_decay_comparison.csv` and `.png` | Both configurations reach 100% evaluation success; Configuration A has slightly higher reward and lower final error. |
| Is the DQN better than the supplied baseline? | `results/rule_based_vs_selected_dqn.csv` | Both succeed 20/20, while the DQN is faster, earns more reward, and finishes closer to the goal. |
| Can the trained model be reloaded? | `models/selected_dqn.pt` | This is the selected Configuration A checkpoint used by evaluation and rendering. |
| Can I see the robot move? | [Rendered evaluation video](https://raw.githubusercontent.com/chence/CSCN8020_Assignment3_Video/refs/heads/master/CSCN8020_Assignment3.mp4) | The saved checkpoint controls the G1 left elbow across the benchmark goals without retraining. |
| Where is the full explanation? | `report/DQN_Assignment_Report.pdf` | Architecture, Bellman update, reproducibility, output guide, metric interpretation, results, and limitations. |

### How the evidence supports the conclusion

The training curves are diagnostic evidence, not the final test. Increasing
reward and rolling success show that the policy improved on the training
distribution. The decisive evidence is the separate greedy evaluation with
epsilon set to `0.0`: the selected checkpoint succeeded in all 20 held-out
evaluation episodes across goals `-0.8`, `-0.4`, `+0.4`, and `+0.8` radians.
The comparison with the rule-based policy uses the same goals and seed schedule,
so differences in reward, episode length, and final error are directly
interpretable.

The Huber-loss curve should not be treated like a supervised-learning validation
loss. DQN targets change while the policy and target network change, and the
replay buffer increasingly contains successful transitions with terminal
bonuses. A later increase in mean loss therefore does not by itself indicate
policy degradation. Here, the stable 100% rolling success rate, stable reward,
and independent 20/20 greedy evaluation provide the behavioral evidence that
the trained controller works.

## Primer Foundation

The supporting workshop introduces control of a Unitree G1 humanoid robot using
MuJoCo and Gymnasium. It covers model inspection, fixed-base model generation,
single-joint PD control, bias-force compensation, CSV logging, and a custom
Gymnasium environment with deterministic rule-based validation. Assignment 3
extends that validated foundation with the student-written DQN documented below.

## Project Overview

This project develops a reproducible instructional workflow for working with the Unitree G1 humanoid robot in MuJoCo.

The workshop begins with environment preparation and model inspection, then progresses through:

1. MuJoCo installation and viewer validation
2. Unitree G1 model inspection
3. Joint, actuator, sensor, `qpos`, and `qvel` analysis
4. Single-joint proportional-derivative control
5. Whole-body joint stabilization
6. Gravity and bias-force compensation
7. Creation of a course-owned fixed-base G1 model
8. CSV logging and deterministic validation
9. Construction of a custom Gymnasium environment
10. Rule-based environment validation
11. Optional interactive visualization before reinforcement learning

The primer portion stops before reinforcement learning so the conventional
robotics foundation can be inspected separately. The Assignment 3 portion then
adds the complete student-written DQN, controlled experiments, saved
checkpoints, greedy evaluation, plots, and rendered demonstration.

---

## Educational Purpose

The workshop is intended for college-level students studying:

- Reinforcement learning
- Robotics
- Machine learning
- Simulation
- Control systems
- Artificial intelligence
- Python programming

The material emphasizes conceptual understanding and reproducibility rather than only presenting finished code.

Students are expected to understand the relationship between:

```text
High-level discrete action
        ↓
Internal joint-position target
        ↓
PD controller
        ↓
Bias-force compensation
        ↓
Actuator torque
        ↓
Simulated physical movement
```

The workshop separates conventional low-level control from high-level reinforcement-learning decisions.

This allows students to focus on the reinforcement-learning problem without first needing to solve full humanoid balance, locomotion, inverse kinematics, and whole-body torque control.

---

## Learning Outcomes

After completing the workshop, students should be able to:

1. Explain the role of MuJoCo in robot simulation.
2. Distinguish between bodies, joints, actuators, sensors, and degrees of freedom.
3. Explain the purpose of `qpos` and `qvel`.
4. Load and inspect the Unitree G1 29-DOF model.
5. Identify a joint and actuator by name.
6. Read joint position and velocity data.
7. Apply bounded actuator torque.
8. Implement a proportional-derivative controller.
9. Explain the effect of gravity and bias forces.
10. Create a fixed-base instructional robot model.
11. Record simulation results in CSV format.
12. Build a Gymnasium-compatible environment.
13. Explain the difference between `terminated` and `truncated`.
14. Define observations, actions, rewards, and success conditions.
15. Validate an environment with a rule-based policy.
16. Confirm deterministic simulation behaviour.
17. Prepare the environment for a future student-written DQN agent.

---

## Current Project Status

| Milestone | Status |
|---|---|
| WSL 2 and Ubuntu setup | Complete |
| MuJoCo installation | Complete |
| MuJoCo viewer test | Complete |
| Unitree G1 repository integration | Complete |
| G1 model inspection | Complete |
| Fixed-base G1 generation | Complete |
| Left-elbow PD control | Complete |
| Whole-body joint stabilization | Complete |
| Bias-force compensation | Complete |
| CSV logging | Complete |
| Deterministic controller validation | Complete |
| Gymnasium environment | Complete |
| Gymnasium environment checker | Complete |
| Rule-based validation policy | Complete |
| Five-run determinism test | Complete |
| Optional rendered validation | Complete |
| Interactive camera-preparation demo | Complete |
| Student-written DQN | Complete |
| Two epsilon-decay experiments | Complete |
| Greedy evaluation | 20/20 success for both configurations |
| Selected checkpoint | Complete |
| Physical G1 deployment | Future work |

---

## Requirements

- Python 3.12
- Packages pinned in `requirements.txt`
- CPU execution for training and evaluation
- A graphical desktop (macOS, Linux, or WSLg) only for the rendered demonstration

## Setup

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The workshop also uses the official Unitree MuJoCo repository as an external
dependency:

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco
git -C external/unitree_mujoco checkout ae6a8403e272733e9996ef59990880330496177f
```

## Workshop

Open `Unitree_MuJoCo_G1_Primer_Workshop.ipynb` and follow its sections in
order. Runtime source is under `src/`, and the course-owned fixed-base model is
under `assets/g1_fixed_base/`.

## Headless validation

```bash
source .venv/bin/activate
python -m compileall src
python src/inspect_g1_model.py \
  assets/g1_fixed_base/scene_29dof_fixed_base.xml \
  --no-viewer
python src/control_single_joint.py \
  --scene assets/g1_fixed_base/scene_29dof_fixed_base.xml \
  --target -0.8 \
  --duration 2 \
  --no-viewer
PYTHONPATH=src python src/test_g1_elbow_env.py
```

Headless execution is authoritative. Rendered demonstrations are optional and
require WSLg.

## Assignment 3: Deep Q-Network

The repository now contains a student-written PyTorch DQN with separate online
and target networks, experience replay, epsilon-greedy action selection, Huber
loss, gradient clipping, checkpoint support, evaluation, and plotting scripts.
Training uses the assignment goal range `[-0.8, 0.8]` and remains headless.

The Bellman target masks true `terminated` transitions. A time-limit
`truncated` transition remains eligible for bootstrapping because it is an
artificial episode boundary rather than a terminal state of the control task.

### Major files

- `CSCN8020_Assignment3.ipynb`: completed assignment notebook and result summary.
- `src/dqn/`: Q-network, replay buffer, and DQN agent implementation.
- `src/train_dqn.py`: headless controlled training workflow.
- `src/evaluate_dqn.py`: greedy DQN and rule-based benchmark evaluation.
- `src/plot_dqn_results.py`: required training and evaluation plots.
- `src/render_dqn_policy.py`: saved-checkpoint viewer demonstration.
- `models/selected_dqn.pt`: selected Configuration A checkpoint.
- `results/`: training metrics, evaluation metrics, tables, and plots.
- `report/DQN_Assignment_Report.pdf`: technical report.
- `output/pdf/CSCN8020_Assignment3_Brightspace.pdf`: one-page submission sheet.
- [Rendered evaluation video](https://raw.githubusercontent.com/chence/CSCN8020_Assignment3_Video/refs/heads/master/CSCN8020_Assignment3.mp4): saved-model MuJoCo demonstration.

### Results directory map

Each configuration directory has the same structure:

| Path | Contents and interpretation |
|---|---|
| `results/config_a/training_metrics.csv` | One row per training episode: seed, goal, cumulative reward, success, length, final error, epsilon, mean Huber loss, optimization steps, and elapsed time. Use this for detailed or reproducibility checks. |
| `results/config_a/training_plots.png` | Four training diagnostics: raw and rolling reward, 50-episode success rate, epsilon, and mean Huber loss. |
| `results/config_a/dqn_evaluation_episodes.csv` | One row per greedy evaluation episode, including goal, seed, reward, length, final error, and action counts. |
| `results/config_a/dqn_evaluation_summary.csv` | Results grouped by goal plus an `overall` row. This is the shortest proof of final DQN performance. |
| `results/config_a/evaluation_success_by_goal.png` | Visual confirmation that success is not concentrated at only one target angle. |
| `results/config_a/rule_based_episodes.csv` | Episode-level results for the supplied rule-based policy under the matching evaluation schedule. |
| `results/config_a/rule_based_summary.csv` | Rule-based results grouped by goal and overall. |
| `results/config_a/training_summary.json` | Experiment identity, seed, epsilon decay, completed episodes, duration, goal range, and truncation treatment. |

`results/config_b/` contains the equivalent files for the faster epsilon-decay
experiment. Repository-level comparison files combine the configurations:

- `results/epsilon_decay_comparison.csv` and `.png` compare Configurations A
  and B under the controlled protocol.
- `results/rule_based_vs_selected_dqn.csv` compares the selected DQN with the
  rule-based policy on the same 20 evaluation episodes.

### Metric reading guide

- **Cumulative reward:** Higher is better. It combines distance-to-goal
  penalties, the success-region bonus, and the terminal success bonus.
- **Success rate:** The primary task metric. Success means the absolute elbow
  error remains within `0.04` rad for eight consecutive environment steps.
- **Episode length:** Lower is better only when success is maintained; it
  indicates how quickly the controller reaches and holds the goal.
- **Final absolute error:** Lower is better and measures terminal precision in
  radians.
- **Epsilon:** The probability of a random training action. It documents the
  exploration schedule and is set to `0.0` for final evaluation.
- **Mean Huber loss:** A learning diagnostic measuring temporal-difference
  error. It is not a standalone measure of task success because DQN targets are
  non-stationary.

### Run the Notebook

```bash
source .venv/bin/activate
jupyter notebook CSCN8020_Assignment3.ipynb
```

### Reproducibility

All commands below run from the repository root. The scripts seed Python,
NumPy, PyTorch, replay sampling, and each Gymnasium reset. CPU is the required
and default device.

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m compileall src
PYTHONPATH=src python src/test_g1_elbow_env.py
```

Run the quick DQN integration test before full training:

```bash
PYTHONPATH=src python src/train_dqn.py \
  --config-name config_a --episodes 2 --smoke-test \
  --output-root results/smoke --model-root models/smoke
```

### Required controlled experiments

Configuration A uses epsilon decay `0.995`; Configuration B uses `0.985`.
Every other baseline hyperparameter and the seed are identical.

```bash
PYTHONPATH=src python src/train_dqn.py \
  --config-name config_a --episodes 600 --seed 42
PYTHONPATH=src python src/train_dqn.py \
  --config-name config_b --episodes 600 --seed 42
```

Each command saves per-episode metrics under `results/<config>/` and a reloadable
checkpoint under `models/`. Adjust `--episodes` only if the documented training
budget requires it; each command also has a wall-clock stop controlled by
`--max-hours`.

### Greedy evaluation and rule-based baseline

Evaluate each checkpoint on five episodes at each required goal. The DQN uses
epsilon `0.0`, and the rule-based policy receives exactly the same goal and seed
schedule.

```bash
PYTHONPATH=src python src/evaluate_dqn.py \
  --checkpoint models/config_a.pt --output-dir results/config_a
PYTHONPATH=src python src/evaluate_dqn.py \
  --checkpoint models/config_b.pt --output-dir results/config_b
```

Generate the required plots:

```bash
PYTHONPATH=src python src/plot_dqn_results.py \
  results/config_a results/config_b
```

After comparing the two evaluation summaries, copy the selected checkpoint to
`models/selected_dqn.pt` and record the evidence-based selection in the report.

### Render the saved policy

Rendering loads the checkpoint and never retrains. The action policy is greedy
with epsilon set to `0.0`.

#### Platform-specific viewer commands

The final development and validation were completed on macOS. MuJoCo's passive
viewer requires Python scripts to be launched through `mjpython` on macOS. This
affects only viewer startup; every platform loads the same saved DQN checkpoint.

**macOS:**

```bash
source .venv/bin/activate
PYTHONPATH=src mjpython src/render_dqn_policy.py \
  --checkpoint models/selected_dqn.pt --goals -0.8 0.8
```

**Linux or WSLg:**

```bash
source .venv/bin/activate
PYTHONPATH=src python src/render_dqn_policy.py \
  --checkpoint models/selected_dqn.pt --goals -0.8 0.8
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python src/render_dqn_policy.py `
  --checkpoint models/selected_dqn.pt `
  --goals -0.8 0.8
```

On Windows, the MuJoCo Viewer requires a working desktop OpenGL environment.
WSL users should use WSLg and follow the Linux command. These commands load
`models/selected_dqn.pt`; they do not train or modify the submitted model.


#### Rendered evaluation demonstration

##### Step 1 - Clone the submitted repository

```bash
git clone https://github.com/chence/CSCN8020_Assignment3.git CSCN8020_Assignment3
cd CSCN8020_Assignment3
```

##### Step 2 - Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

##### Step 3 - Verify the selected model

```bash
python --version
ls -lh models/selected_dqn.pt
```

##### Step 4 - Review the selected model and results in Jupyter

```bash
jupyter notebook CSCN8020_Assignment3.ipynb
```

The Notebook loads the submitted checkpoint and displays its source
configuration, model-selection evidence, and final evaluation results. It does
not retrain or modify the model.

##### Step 5 - Run the continuous MuJoCo demonstration

```bash
PYTHONPATH=src mjpython src/render_dqn_policy.py \
  --checkpoint models/selected_dqn.pt \
  --goals -0.8 -0.4 0.8 0.4 0.0 \
  --continuous \
  --interactive \
  --step-delay 0.08
```

The elbow starts at neutral and follows `-0.8`, `-0.4`, `+0.8`, `+0.4`, and
finally `0.0` radians. This continuous sequence includes all four required
benchmark goals and finishes at the neutral position.

The console reports `goal`, `step`, selected `action`, physical `angle`, signed
`error`, and step `reward`. A successful `SEGMENT` line confirms that the elbow
remained within the required tolerance for the required consecutive steps.
