# Retraining the PPO Actor-Critic Model

Run all commands from the repository root. The standard retraining commands use separate model and result paths, so they do not overwrite `models/inspire_actor_critic.pt`.

## 1. Retrain the model

Run 100 PPO updates with 1,024 environment steps per update.

### macOS

```bash
python final_project/train_actor_critic.py \
  --updates 100 \
  --steps-per-update 1024 \
  --seed 42 \
  --device cpu \
  --model models/inspire_actor_critic_retrained.pt \
  --results results/inspire_actor_critic_retrained
```

### Windows PowerShell

```powershell
python final_project/train_actor_critic.py `
  --updates 100 `
  --steps-per-update 1024 `
  --seed 42 `
  --device cpu `
  --model models/inspire_actor_critic_retrained.pt `
  --results results/inspire_actor_critic_retrained
```

## 2. Evaluate the retrained model

Run a deterministic evaluation on 100 randomized reachable targets.

### macOS

```bash
python final_project/evaluate_actor_critic.py \
  --model models/inspire_actor_critic_retrained.pt \
  --episodes 100 \
  --seed 1000 \
  --csv results/inspire_actor_critic_retrained/evaluation.csv
```

### Windows PowerShell

```powershell
python final_project/evaluate_actor_critic.py `
  --model models/inspire_actor_critic_retrained.pt `
  --episodes 100 `
  --seed 1000 `
  --csv results/inspire_actor_critic_retrained/evaluation.csv
```

## 3. Demonstrate the retrained model with the Viewer

Open the MuJoCo Viewer and run five evaluation episodes.

### macOS

MuJoCo Viewer scripts must be launched with `mjpython` on macOS:

```bash
mjpython final_project/evaluate_actor_critic.py \
  --model models/inspire_actor_critic_retrained.pt \
  --episodes 5 \
  --viewer \
  --keep-viewer-open \
  --csv results/inspire_actor_critic_retrained/demo_evaluation.csv
```

### Windows PowerShell

Use standard Python on Windows:

```powershell
python final_project/evaluate_actor_critic.py `
  --model models/inspire_actor_critic_retrained.pt `
  --episodes 5 `
  --viewer `
  --keep-viewer-open `
  --csv results/inspire_actor_critic_retrained/demo_evaluation.csv
```

With `--keep-viewer-open`, the first episode waits until Enter is pressed. After evaluation, close the MuJoCo Viewer or press Enter in the terminal to exit.

## 4. Optional: visualize training progress

The separate visualized-training script trains headlessly and displays one deterministic episode every five updates. Its model and metrics are written under the Git-ignored `tmp/actor_critic_visualized/` directory, and it cannot overwrite `models/inspire_actor_critic.pt`.

### macOS

```bash
mjpython final_project/train_actor_critic_visualized.py \
  --updates 100 \
  --steps-per-update 1024 \
  --seed 42 \
  --device cpu \
  --viewer-interval 5 \
  --viewer-episodes 1 \
  --model tmp/actor_critic_visualized/model.pt \
  --results tmp/actor_critic_visualized/results
```

### Windows PowerShell

```powershell
python final_project/train_actor_critic_visualized.py `
  --updates 100 `
  --steps-per-update 1024 `
  --seed 42 `
  --device cpu `
  --viewer-interval 5 `
  --viewer-episodes 1 `
  --model tmp/actor_critic_visualized/model.pt `
  --results tmp/actor_critic_visualized/results
```

Closing the Viewer disables later visualizations but allows the remaining training to continue headlessly.
