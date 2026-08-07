# Group 2 Dry-Run Speaker Notes

Estimated presentation length: 6–8 minutes, excluding the live demo and Q&A.

## Slide 1 — Reach-and-touch is now executable

**Speaker:**

Good morning. We are Group 2, and our final project extends our Assignment 3
Deep Q-Network controller into a Unitree G1 reach-and-touch task.

In Assignment 3, we successfully trained a DQN to control one left-elbow joint.
For the final project, we want the robot to perform a measurable sequence: start
from rest, reach a virtual target, touch and hold it, and return to a neutral
pose.

Our main progress for this dry run is that this complete task is now executable
in MuJoCo. Today, we will show the validated task mechanics, the evidence from
our demo, and how we will replace the current scripted controller with a
learned policy.

**中文提示：** 开场要主动说明项目目标。不要说当前 reach-and-touch 已经由 RL
学会，只说完整任务已经可以执行。

**Transition:** First, let us explain how the current demo connects our A3 work
to the final learned policy.

## Slide 2 — We are extending a proven controller into a measurable task

**Speaker:**

Our development has three stages.

The first stage is our A3 baseline. The DQN observes the left-elbow state and
selects one of three actions: decrease, hold, or increase the controller target.

The second stage is our current dry-run demo. It uses scripted PD targets to
control the left shoulder and elbow through the full reach, touch, hold, and
return sequence. This validates the task before we spend time training.

The third stage is the final learned policy. We will keep the same task and
measurements, but replace the scripted target sequence with actions selected by
DQN or, if necessary, PPO.

The important distinction is that the dry run proves the middle stage. The
multi-joint task works, but policy training remains our next milestone.

**中文提示：** 这一页是最重要的范围说明。强调三个阶段，防止老师误认为 demo
是训练结果。

**Transition:** Our confidence in using DQN comes from the results we already
achieved in Assignment 3.

## Slide 3 — The A3 DQN already provides a reliable learning baseline

**Speaker:**

This slide summarizes our validated A3 baseline.

The training reward improved and stabilized over 500 episodes. The rolling
success rate reached one hundred percent and remained stable. We then evaluated
the saved checkpoint greedily, with exploration disabled.

The selected model succeeded in all 20 evaluation episodes across four target
angles. The proposal reports a 100 percent evaluation success rate and a mean
final absolute error of approximately 0.00491 radians for Configuration A.

This result does not prove the reach-and-touch task yet, because that task has a
larger state and action space. However, it gives us a working DQN
implementation, replay buffer, target network, checkpoint workflow, and
evaluation procedure that we can extend instead of rebuilding from zero.

**中文提示：** 不要花时间逐张解释四个图。重点是训练改善、独立 greedy
evaluation，以及可以复用的 DQN 基础设施。

**Transition:** To extend this baseline, we first need to define touching as a
clear reinforcement-learning problem.

## Slide 4 — Touch becomes a measurable RL objective

**Speaker:**

The new observation will include the controlled joint angles and velocities,
the hand and target positions, the hand-to-target error, and the current task
phase.

Our first action design remains discrete so that it is compatible with our DQN.
Each action changes or holds a shoulder or elbow controller target. We will keep
the action set small enough to train reliably, while still allowing the hand to
move toward the target.

Touching is defined quantitatively. The hand must enter the target region and
remain there for a required duration. Returning to the neutral pose can then be
measured as a separate completion criterion.

The planned reward has three main parts. It penalizes the distance between the
hand and the target, provides a bonus for touching and holding, and penalizes
unnecessary or unstable motion. Therefore, the agent must learn both accuracy
and stability rather than only moving the arm quickly.

**中文提示：** 指公式时只解释三部分，不要推导。可能被问 state 中放 phase
是否会限制端到端学习，可以回答它让初版任务更稳定、也更容易解释。

**Transition:** We used exactly these measurable definitions to build and test
the current demo sequence.

## Slide 5 — The demo validates one complete reach-touch-return cycle

**Speaker:**

The demo contains four phases.

During REST, the controller stabilizes the robot. During REACH, it smoothly
interpolates the shoulder and elbow targets. During TOUCH AND HOLD, the hand
must remain inside the virtual target. Finally, during RETURN, the arm moves
back toward its neutral pose.

We use two explicit success parameters. The target radius is 0.045 metres, so
the hand is considered to be touching when its endpoint is within 4.5
centimetres of the target centre. The required hold duration is 1.5 seconds, so
briefly passing through the target does not count as success.

These parameters are command-line options, which makes the criterion easy to
reproduce and allows us to test stricter accuracy or hold requirements later.

**中文提示：** 讲到这里可以准备切换到 MuJoCo。若现场时间紧，可以先播放
demo，再回来讲第 6 页结果。

**Live demo introduction:** We will now run the visual demo. Please notice the
four phase labels in the terminal. The virtual target is red while the hand is
outside the success region and turns green while the hand is touching it.

**After the demo:** The sequence completed successfully. We also ran the same
controller headlessly and saved every time step to a CSV file for quantitative
verification.

## Slide 6 — The demo succeeds with 1.2 mm endpoint accuracy

**Speaker:**

This chart shows the measured hand-to-target distance throughout the complete
sequence.

The distance begins at approximately 0.23 metres during rest. It then decreases
during the reach phase and remains near zero during touch and hold. During the
return phase, the distance increases again, which is expected because the hand
is moving back to its neutral pose.

The minimum measured distance was approximately 0.0012 metres, or 1.2
millimetres. The hand accumulated 2.002 seconds inside the target region, which
exceeded our required 1.5-second hold threshold. The run therefore returned
SUCCESS.

The visual and headless versions use the same target radius and hold criteria,
so the reported result is reproducible and not based only on what the movement
looks like.

**中文提示：** 不要说 1.2 mm 是 learned accuracy。这只是 scripted baseline
在本次固定目标上的结果。

**Transition:** These results are useful, but we need to be precise about what
they do and do not prove.

## Slide 7 — The demo proves task mechanics, not learned behavior

**Speaker:**

The current demo proves six important components.

It proves that we can actuate multiple arm joints, display a virtual target,
measure the endpoint distance, apply a touch-and-hold success rule, execute all
four task phases, and export reproducible measurements.

However, the action sequence is still scripted. We have not yet shown that a
policy can select actions from observations, improve through reward, or
generalize across target positions.

The final experiments must compare the learned policy against this scripted
baseline, evaluate multiple target positions and random seeds, and measure
motion smoothness. This separation gives us an honest dry-run result and a
clear baseline for the learned controller.

**中文提示：** 老师如果指出“这不是 RL”，直接同意并说明这是 environment
validation baseline；不要试图模糊 scripted 和 learned 的区别。

**Transition:** With the environment mechanics validated, our remaining work is
focused on learning and evaluation.

## Slide 8 — Next, the learned policy must beat the scripted baseline

**Speaker:**

Our next work has four steps.

First, we will expose the new observation, discrete action space, reward, and
termination rules through a Gymnasium environment.

Second, we will train DQN across randomized but reachable virtual target
positions and tune the reward weights.

Third, we will evaluate the saved policy using fixed seeds and exploration
disabled. We will report success rate, hand-to-target distance, hold duration,
cumulative reward, and motion smoothness.

Finally, we will compare the results against the scripted baseline. If DQN is
reliable and sufficiently smooth, we will retain it as our main approach. If
the discrete actions produce unstable or jerky motion, we will compare it with
PPO as a continuous-control extension.

For this dry run, the feedback we need is whether our task definition and
evaluation criteria are appropriate before we begin full training.

Thank you. We are ready for questions and feedback.

**中文提示：** 结尾不要只说 Thank you。明确向老师询问 task definition 和
evaluation criteria 是否合理。

## Short Q&A Answers

### Is this demo controlled by reinforcement learning?

Not yet. It is a deterministic scripted PD baseline used to validate the task
mechanics and success measurements. The A3 elbow controller is learned by DQN;
the next project milestone is to replace this scripted multi-joint sequence
with learned actions.

### Why use DQN for a multi-joint task?

DQN gives us a direct extension of our validated A3 implementation. We will
start with a carefully limited discrete action space. If that action space
becomes inefficient or produces jerky motion, PPO is our planned continuous
control comparison.

### How is touch detected?

We calculate the Euclidean distance between the hand endpoint and the virtual
target centre. Touch is true when that distance is less than or equal to the
target radius.

### Why require a hold duration?

Without a hold requirement, the hand could pass through the target briefly and
still be counted as successful. The hold duration requires stable, sustained
contact.

### Why does the distance increase at the end of the graph?

That is the RETURN phase. After completing the touch requirement, the hand
moves away from the target and returns toward the neutral pose.

### How will you show that the policy learned?

We will train on randomized reachable targets, save the checkpoint, disable
exploration during evaluation, and test across fixed held-out targets and
multiple seeds. We will compare those results with the untrained and scripted
baselines.

