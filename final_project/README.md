# Unitree G1 Reach-and-Touch Dry-Run Demo

This is a deterministic scripted PD-controller baseline for the proposed final
project. It is deliberately labelled as **not a trained RL policy**. It proves
the MuJoCo task mechanics before the high-level controller is replaced by DQN
or PPO.

The sequence is:

```text
REST -> REACH -> TOUCH/HOLD -> RETURN
```

The red sphere is the virtual target. It turns green while the left hand is
inside the success region. The terminal output reports the current phase,
hand-to-target distance, touch state, minimum distance, and hold duration.

## Run the visual demo

From the repository root on macOS:

```bash
source .venv/bin/activate
mjpython final_project/reach_touch_demo.py
```

Close the MuJoCo viewer after the sequence finishes. To save measurement data:

```bash
mjpython final_project/reach_touch_demo.py \
  --csv final_project/reach_touch_demo_metrics.csv
```

## Headless verification

```bash
.venv/bin/python final_project/reach_touch_demo.py \
  --no-viewer \
  --csv final_project/reach_touch_demo_metrics.csv
```

A successful run exits with status 0 and should report at least 1.5 seconds of
continuous accumulated contact during the TOUCH/HOLD phase.

## What to say during the dry run

> This is our deterministic environment-validation baseline, not our learned
> policy. It verifies the target visualization, arm actuation, endpoint-distance
> observation, touch threshold, hold-success rule, phase transitions, and
> return motion. The next step is to replace the scripted arm targets with the
> DQN action output and compare it against this baseline.

