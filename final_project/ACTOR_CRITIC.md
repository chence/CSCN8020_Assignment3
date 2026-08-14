# Inspire Actor-Critic Technical Definition

## What the robot learns

The learned policy maps the current robot and target state to bounded arm-joint
increments. Its objective is to move the modeled left index fingertip into the
target region, remain there for eight consecutive control steps, and avoid
unnecessarily large actions.

Targets are sampled from forward kinematics of reachable arm poses. This makes
the target move between episodes while ensuring that every training target has
a valid solution within the controlled joint limits.

## State

The observation is the 19-dimensional vector

```text
s_t = [q_t, qdot_t, p_tip,t, p_target, delta_p_t, d_t, h_t]
```

where:

- `q_t` contains four controlled joint angles.
- `qdot_t` contains their four joint velocities.
- `p_tip,t` is the three-dimensional fingertip position.
- `p_target` is the three-dimensional blue-target position.
- `delta_p_t = p_target - p_tip,t` is the signed Cartesian error.
- `d_t = ||delta_p_t||_2` is the fingertip distance.
- `h_t` is normalized touch-hold progress.

The target and relative error make this a target-conditioned policy: different
blue-dot locations produce different observations and therefore different
Actor outputs.

## Action

The Actor outputs

```text
a_t in [-1, 1]^4
```

for shoulder pitch, shoulder roll, shoulder yaw, and elbow. The environment
applies each component as a bounded joint-position increment:

```text
q_cmd,t+1 = clip(q_cmd,t + 0.055 a_t, q_min, q_max)
```

The hand stays in the pointing pose established by the scripted baseline. This
keeps the first learned task focused on target-conditioned arm reaching.

## Reward

The implemented dense reward is

```text
r_t = 25 (d_t-1 - d_t) - 1.5 d_t - 0.01 ||a_t||^2
      + 0.5 I[d_t <= 0.045] + 10 I[success]
```

The progress term rewards motion toward the target, the distance term penalizes
remaining far away, the action term discourages unnecessarily large commands,
and the two bonuses reward touching and completing the hold requirement.

## Actor and Critic

The Actor is a Gaussian policy whose sampled output is passed through `tanh`:

```text
a_t ~ pi_theta(a | s_t)
```

The Critic estimates the state value:

```text
V_phi(s_t)
```

Generalized Advantage Estimation starts from the temporal-difference error:

```text
delta_t = r_t + gamma V_phi(s_t+1) - V_phi(s_t)
```

PPO updates the Actor with a clipped probability ratio and trains the Critic
against the bootstrapped return. Time-limit truncation still permits value
bootstrapping, while true successful termination does not.

## Difference from the Assignment 3 DQN

The earlier DQN uses an online Q-network and a target Q-network, then selects a
discrete action with `argmax Q(s, a)`. Those two networks are not an Actor and
a Critic.

This version has an explicit Actor that outputs continuous actions and an
explicit Critic that estimates `V(s)`. The Critic's advantage estimate guides
the Actor update.

## Verified result

The selected checkpoint is chosen by deterministic validation during training.
On a separate evaluation set of 100 randomized targets, it achieved:

```text
successes:                 100 / 100
success rate:              100.0%
mean minimum distance:     0.0185 m
touch tolerance:           0.0450 m
```

The detailed results are stored in
`results/inspire_actor_critic/evaluation.csv`.
