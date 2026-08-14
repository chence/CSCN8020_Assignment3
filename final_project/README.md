# Unitree G1 + Inspire Five-Finger Demo

This directory contains the Inspire-hand version of the reach-and-touch demo.

The model is based on Unitree's official
`g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf`. MuJoCo exposes 29 body joints
and 24 modeled finger joints, for 53 joints in total. The script adds actuator
mappings at load time and uses a site on the left index fingertip as the task
endpoint.

## Run the visual demo

From the repository root:

```bash
source .venv/bin/activate
mjpython final_project/reach_touch_inspire_demo.py
```

The blue gradient sky, checkerboard floor, lighting, and camera framing are
matched to the original fixed-base demo. The default hand uses a lightly
curled relaxed pose instead of the URDF's mechanically flat zero pose. Both
elbows and wrists are also initialized in a natural hanging pose. Both wrist
roll and pitch remain zero; yaw compensates for the Inspire mounting frame so
neither palm turns upward or outward.

The inactive right side uses the original G1 human-shaped rubber-hand visual,
while the animated left side retains the articulated Inspire hand. This gives
the unused right hand a clean, anatomically familiar silhouette without
affecting the 53-joint model dimensions. The hidden right Inspire joints remain
fixed internally. All left-hand REST and pointing settings are unchanged.

The sequence is:

```text
REST -> POINT -> REACH -> TOUCH/HOLD -> RETURN -> RELAX
```

During `POINT`, the thumb, middle, ring, and little fingers curl moderately
while the index finger stays almost straight. During `REACH`, the arm moves
the actual index fingertip toward the virtual target.

The reach pose keeps wrist roll and wrist pitch at zero. Wrist yaw is set to
`+1.5 rad` to compensate for the Inspire hand's approximately 90-degree mount
rotation; visually this aligns the palm and index finger with the forearm. The
target position is recalculated from this straight-wrist fingertip pose, so it
appears in front of the left hand rather than beside the upper arm.

## Headless verification

```bash
python final_project/reach_touch_inspire_demo.py --no-viewer
```

Save fingertip metrics:

```bash
python final_project/reach_touch_inspire_demo.py \
  --no-viewer \
  --csv final_project/reach_touch_inspire_metrics.csv
```

## Optional parameters

`--target-radius` is the numerical success radius in metres. The default is
`0.045`, meaning the index fingertip must be within 4.5 cm of the target
centre.

`--target-visual-radius` changes only the solid target sphere displayed in the
viewer. Its default is `0.018` metres.

`--required-hold` is the minimum accumulated time that the index fingertip
must stay inside the success region during `TOUCH/HOLD`. Its default is 1.5
seconds.

Example:

```bash
mjpython final_project/reach_touch_inspire_demo.py \
  --target-radius 0.04 \
  --target-visual-radius 0.015 \
  --required-hold 1.5
```

## What this demo proves

- The rendered hand is an articulated five-finger Inspire DFQ model.
- The pointing pose is produced by actuated finger joints, not a rigid-hand
  coordinate offset.
- The task distance is measured from the modeled left index fingertip.
- The controller is a deterministic scripted joint-position baseline, not a
  trained RL policy. Direct position tracking is used because the imported
  finger links have very small inertias and body-scale torque gains make their
  simulation numerically unstable.

Finger contact is disabled in this baseline because detailed imported hand
collision meshes can interfere with the scripted pose. The virtual target
touch is evaluated using fingertip-to-target distance.

## Actor-Critic learned-policy version

The project also includes a PPO Actor-Critic policy that learns to reach
randomized reachable targets. Unlike the scripted sequence, the target
position is part of the observation, so moving the blue target changes the
actions selected by the Actor.

The 19-value state is:

```text
[q(4), qdot(4), fingertip_xyz(3), target_xyz(3),
 target_minus_fingertip(3), distance(1), hold_progress(1)]
```

The Actor produces four continuous joint-position increments for left
shoulder pitch, shoulder roll, shoulder yaw, and elbow. The Critic estimates
the state value `V(s)`. The articulated fingers remain in the validated
pointing pose while the learned policy controls the arm.

Train from the repository root:

```bash
source .venv/bin/activate
python final_project/train_actor_critic.py
```

The trainer evaluates the deterministic policy on a fixed validation set every
five updates and saves the best checkpoint, rather than assuming that the last
PPO update is the best one.

Evaluate the saved checkpoint on new randomized targets:

```bash
python final_project/evaluate_actor_critic.py --episodes 100
```

Show the learned policy in the MuJoCo viewer:

```bash
# macOS
mjpython final_project/evaluate_actor_critic.py --episodes 5 --viewer

# Windows / Linux
python final_project/evaluate_actor_critic.py --episodes 5 --viewer
```

The evaluation command loads the saved checkpoint, generates a different blue
target each episode, prints the deterministic Actor actions and Critic values,
and reports touch success. It does not retrain the model. macOS uses `mjpython`
because the MuJoCo Viewer must run on the main thread; Windows and Linux can
normally use standard Python. For a cross-platform headless fallback, run:

```bash
python final_project/evaluate_actor_critic.py --episodes 5
```

The current saved checkpoint achieved 100 successes in 100 deterministic
evaluation episodes with a mean minimum fingertip distance of 0.0185 m. These
episodes use seeds separate from the validation targets used for checkpoint
selection.

Implementation files:

- `inspire_reach_env.py`: randomized-target Gymnasium environment.
- `actor_critic.py`: Actor, Critic, PPO update, GAE, and checkpoint code.
- `train_actor_critic.py`: training, metrics, validation, and best-model selection.
- `evaluate_actor_critic.py`: deterministic saved-policy evaluation and viewer.
- `test_actor_critic.py`: headless environment and PPO update checks.

See `ACTOR_CRITIC.md` for the mathematical task definition and the distinction
between the earlier DQN and this Actor-Critic implementation.
