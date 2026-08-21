# Retraining the PPO Actor-Critic Model

The commands below train and evaluate a new model using separate output paths. They do not overwrite the existing model at `models/inspire_actor_critic.pt`.

## 1. Retrain the model

Run 100 PPO updates with 1,024 environment steps per update:

```bash
python final_project/train_actor_critic.py \
  --updates 100 \
  --steps-per-update 1024 \
  --seed 42 \
  --device cpu \
  --model models/inspire_actor_critic_retrained.pt \
  --results results/inspire_actor_critic_retrained
```

## 2. Evaluate the retrained model

Run a deterministic evaluation on 100 randomized reachable targets:

```bash
python final_project/evaluate_actor_critic.py \
  --model models/inspire_actor_critic_retrained.pt \
  --episodes 100 \
  --seed 1000 \
  --csv results/inspire_actor_critic_retrained/evaluation.csv
```

## 3. Demonstrate the retrained model on macOS

Open the MuJoCo Viewer and run five evaluation episodes:

```bash
mjpython final_project/evaluate_actor_critic.py \
  --model models/inspire_actor_critic_retrained.pt \
  --episodes 5 \
  --viewer \
  --keep-viewer-open \
  --csv results/inspire_actor_critic_retrained/demo_evaluation.csv
```

With `--keep-viewer-open`, the first episode waits until Enter is pressed. After evaluation, close the MuJoCo Viewer or press Enter in the terminal to exit.
