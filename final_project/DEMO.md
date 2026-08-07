# Unitree G1 Reach-and-Touch Demo

This document contains the commands required to run and verify the dry-run
demo. Run all commands from the repository root:

```text
CSCN8020_Assignment3/
```

The current demo is a deterministic scripted PD-controller baseline. It is
**not yet a trained reinforcement-learning policy**.

## 1. Activate the existing environment

```bash
source .venv/bin/activate
```

Optional dependency check:

```bash
python -c "import mujoco, numpy; print('MuJoCo:', mujoco.__version__)"
```

## 2. Run the visual demo

On macOS, use `mjpython` when opening the MuJoCo viewer:

```bash
mjpython final_project/reach_touch_demo.py
```

The sequence shown in the viewer is:

```text
REST -> REACH -> TOUCH/HOLD -> RETURN
```

The sphere represents the virtual target:

- Red: the hand is outside the target region.
- Green: the hand is inside the target region.

The terminal displays the phase, hand-to-target distance, and touch state.
Close the viewer after the sequence finishes.

## 3. Run the demo and save metrics

```bash
mjpython final_project/reach_touch_demo.py \
  --csv final_project/reach_touch_demo_metrics.csv
```

The CSV contains:

- Simulation time and phase
- Hand position
- Target position
- Hand-to-target distance
- Touch indicator

## 4. Run the automated headless verification

This command does not open a viewer and is useful for checking the demo before
the presentation:

```bash
python final_project/reach_touch_demo.py \
  --no-viewer \
  --csv final_project/reach_touch_demo_metrics.csv
```

The tested demo should finish with output similar to:

```text
DEMO RESULT
success=True
minimum_distance=0.0012m
touch_hold_duration=2.002s
```

A successful run exits with status code `0`. To display the exit code:

```bash
python final_project/reach_touch_demo.py --no-viewer
echo $?
```

## 5. Optional parameter overrides

### `--target-radius`

`--target-radius` defines the radius of the successful target region in
metres. Its default value is `0.045`, or 4.5 cm.

The hand is considered to be touching the target when:

```text
hand-to-target distance <= target-radius
```

When this condition is true, the terminal reports `touched=True` and the
target sphere changes from red to green. A larger radius makes the target
easier to reach, while a smaller radius requires greater endpoint accuracy.

Change the radius of the successful target region:

```bash
mjpython final_project/reach_touch_demo.py --target-radius 0.05
```

In this example, the hand must be within 5 cm of the target centre.

### `--required-hold`

`--required-hold` defines the minimum accumulated time, in seconds, for which
the hand must remain inside the target region during the `TOUCH/HOLD` phase.
Its default value is `1.5` seconds.

The complete demo is considered successful when:

```text
touch_hold_duration >= required-hold
```

A larger value requires a longer stable touch. A smaller value makes the hold
criterion easier to satisfy.

Change the required hold duration used for the final success result:

```bash
mjpython final_project/reach_touch_demo.py --required-hold 1.0
```

In this example, the hand must remain in the target region for at least one
second.

Use both overrides and save the measurements:

```bash
mjpython final_project/reach_touch_demo.py \
  --target-radius 0.05 \
  --required-hold 1.0 \
  --csv final_project/reach_touch_demo_metrics.csv
```

This combined example means that the hand must enter a target region with a
5 cm radius and stay there for at least 1 second.

## 6. Quick pre-presentation checklist

Run these commands before the dry run:

```bash
source .venv/bin/activate
python -m py_compile final_project/reach_touch_demo.py
python final_project/reach_touch_demo.py --no-viewer
mjpython final_project/reach_touch_demo.py
```

Confirm that:

- The viewer opens and shows the full upper body and left arm.
- The target sphere is visible.
- The target turns green during `TOUCH/HOLD`.
- The terminal reports `success=True`.
- The robot returns toward its neutral pose.

## 7. Suggested dry-run explanation

> This is our deterministic environment-validation baseline, not our learned
> policy. It verifies the virtual-target visualization, multi-joint arm
> actuation, endpoint-distance observation, touch threshold, hold-success rule,
> phase transitions, and return motion. Our next step is to replace the
> scripted arm targets with DQN action outputs and evaluate the learned policy
> against this baseline.
