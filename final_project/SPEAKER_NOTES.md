# Inspire Demo Speaker Notes / Inspire Demo 演讲稿

Recommended length: 7–9 minutes, excluding the live demo and Q&A.

建议时长：7–9 分钟，不包括现场 demo 和问答。

## Slide 1 — Five-finger pointing is now executable

### English script

Good morning. We are Group 2. This version of our project uses a Unitree G1 model with an articulated Inspire five-finger hand.

Our dry-run result is an executable pointing and touching sequence. The robot begins in a relaxed pose, forms a pointing gesture with its left hand, moves the actual modeled index fingertip to a virtual target, holds there, and returns to rest.

The scope is important: this is a deterministic scripted joint-position baseline, not a trained reach-and-touch policy. It validates the model, task mechanics, endpoint measurement, and success rule needed before adaptive control and reinforcement-learning training.

### 中文提示

开场立刻说明：五指模型和完整动作已经跑通，但当前不是训练出来的策略。不要把 scripted baseline 说成 RL 成果。

**Transition:** First, let us show how this demo fits into our project path.

## Slide 2 — The Inspire demo connects our proven DQN to a richer task

### English script

Our work has three stages.

In Assignment 3, our DQN learned discrete control of one left-elbow joint. That work gave us a functioning replay buffer, online and target networks, checkpoint saving, and greedy evaluation.

The current Inspire demo validates a richer physical task. It commands seven left-arm joint targets and twelve left-hand joint targets to produce a natural pointing motion and a measurable touch.

The next stage is adaptation. We first need inverse kinematics, or another target-conditioned controller, so that the fingertip responds when the target moves. Then we can train DQN or PPO across multiple reachable target positions.

### 中文提示

强调顺序：A3 DQN 基础 → Inspire 环境和动作验证 → IK 自适应 → RL 训练。IK 和 RL 不是同一件事。

**Transition:** Our confidence in the learning pipeline comes from the A3 evaluation.

## Slide 3 — The A3 DQN still provides our learning foundation

### English script

The A3 controller achieved 20 successes in 20 greedy evaluation episodes across four target angles. Its reported evaluation success rate was 100 percent, and Configuration A had a mean final absolute error of approximately 0.00491 radians.

These numbers apply only to the earlier one-joint elbow task. They do not prove that the new five-finger task has already been learned. Their value is that we can reuse a tested DQN implementation and evaluation workflow instead of starting again from zero.

### 中文提示

明确这些数字属于 A3 的单关节实验。不要称它们为 Inspire demo 的训练结果。

**Transition:** The new model changes the task endpoint from a hand approximation to a real fingertip site.

## Slide 4 — The endpoint is now an actuated index fingertip

### English script

The MuJoCo model exposes 53 joints: 29 body joints and 24 modeled hand joints. The animated left side retains the articulated Inspire hand. The unused right side uses a simpler human-shaped visual, while its internal joints remain in the model for consistent dimensions.

The current baseline commands seven left-arm targets and twelve left-finger targets. The endpoint is a site attached to the modeled left index fingertip, so success is based on the actual finger geometry rather than a rigid hand offset.

For the future reinforcement-learning objective, the main reward term will reduce fingertip-to-target distance. Additional terms will reward touching and holding and penalize unstable motion.

### 中文提示

如果被问为什么右手看起来不同：右手在任务中不活动，为了展示自然，使用了原始 G1 rubber-hand visual；左手仍是可动 Inspire 手。

**Transition:** With that endpoint defined, the demo runs a six-phase sequence.

## Slide 5 — Six phases produce one complete point-touch-return cycle

### English script

The implementation has six phases: REST, POINT, REACH, TOUCH/HOLD, RETURN, and RELAX.

During POINT, the thumb and three non-pointing fingers curl while the index finger remains almost straight. During REACH, the left-arm targets move the actual fingertip to the virtual target. During TOUCH/HOLD, valid time accumulates only while the fingertip is inside the success region. The arm then returns and the hand relaxes.

Two command-line parameters define success. `target-radius` is the maximum fingertip-to-centre distance that counts as touching. Its default is 0.045 metres. `required-hold` is the minimum accumulated valid time, and its default is 1.5 seconds.

### 中文提示

准备切到现场 demo。指出目标外为蓝色、触碰时变绿；终端会打印每个 phase。

**Live-demo command:**

```bash
source .venv/bin/activate
mjpython final_project/reach_touch_inspire_demo.py --keep-viewer-open
```

## Slide 6 — The Inspire baseline reaches the target and holds

### English script

We also ran the same sequence headlessly and recorded the index-tip distance at every simulation step.

The verified run reached a minimum distance of 0.0000 metres at the displayed precision. It accumulated 2.002 seconds inside the target region, which exceeds the required 1.5 seconds, and returned `success=True`.

The curve starts near 0.20 metres, falls during REACH, remains at the target during TOUCH/HOLD, and increases again during RETURN. The final increase is expected because the hand intentionally leaves the target.

These values describe a fixed-target scripted run, not learned-policy accuracy.

### 中文提示

重点说 0.0000 m、2.002 s、SUCCESS。一定补充“不是 learned accuracy”。

**Transition:** This gives strong environment evidence, but it also has a clear boundary.

## Slide 7 — The Inspire demo proves mechanics, not adaptation

### English script

The demo proves that the articulated hand loads correctly, the index fingertip is measurable, a natural pointing gesture can be commanded, the six phases run, the touch-and-hold rule works, and the results can be reproduced from CSV output.

It does not yet adapt to a moved target. The current target is calculated from the fixed `TOUCH_ARM` pose. If its distance or left-right position changes, the scripted arm pose will not automatically follow it.

That limitation is exactly why inverse kinematics or a target-conditioned learned controller is the next engineering step.

### 中文提示

如果老师问“球换位置能不能追踪”，直接回答当前不能；目标由固定 TOUCH_ARM 姿态计算。后续 IK 或 RL 才能自适应。

**Transition:** Our remaining work follows directly from this limitation.

## Slide 8 — Next, make the fingertip adapt before full RL training

### English script

Our next work has four steps.

First, we will expose fingertip error, actions, reward, termination, and success information through a Gymnasium environment.

Second, we will add inverse kinematics so the arm can respond to target-position changes. This gives us a reliable adaptive reference controller and helps confirm which targets are reachable.

Third, we will train DQN or PPO across randomized reachable targets.

Finally, we will compare the learned policy against both the scripted baseline and the IK reference using success rate, minimum distance, hold time, reward, and motion smoothness.

For this dry run, we would like feedback on whether this Inspire task scope and evaluation plan are appropriate before full training.

### 中文提示

结尾提出明确问题，不要只说 Thank you。等待老师对 task scope、IK baseline 和评价指标给反馈。
