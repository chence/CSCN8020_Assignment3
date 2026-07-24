# CSCN8020 Assignment 3 - Unitree G1 DQN

**Student:** Ce Chen<br>
**Student ID:** 9007166<br>
**GitHub Link:** https://github.com/chence/CSCN8020_Assignment3.git<br>
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

This beginner-friendly workshop introduces control of a Unitree G1 humanoid
robot using MuJoCo and Gymnasium. It covers model inspection, fixed-base model
generation, single-joint PD control, bias-force compensation, CSV logging, and
a custom Gymnasium environment with deterministic rule-based validation.

The workshop intentionally stops before implementation of a DQN agent.

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

The project intentionally stops before the student-written Deep Q-Network implementation. The validated Gymnasium environment is designed to become the foundation for that next phase.

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

Rendering loads the checkpoint and never retrains. The action policy is greedy.

```bash
PYTHONPATH=src python src/render_dqn_policy.py \
  --checkpoint models/selected_dqn.pt --goals -0.8 0.8
```

The final submission must also contain the two experiment metric sets, the
selected checkpoint, plots, a 6-10 page report, and a 2-3 minute rendered video.
AI assistance used for code or writing must be disclosed according to the
course and institutional requirements.
