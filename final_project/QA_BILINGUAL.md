# Group 2 Dry-Run Q&A Preparation / Group 2 Dry Run 问答准备

This document provides concise English answers for the presentation and Chinese
explanations for group preparation.

本文档提供可在演示现场直接使用的英文回答，以及供组员理解和统一口径的中文版本。

## 1. Is the current reach-and-touch demo controlled by reinforcement learning? / 当前的 reach-and-touch demo 是由强化学习控制的吗？

### English answer

Not yet. The current reach-and-touch demo is a deterministic scripted PD
baseline used to validate the environment, multi-joint actuation, target
visualization, distance measurement, and success criteria. Our Assignment 3
left-elbow controller was trained with DQN. The next milestone is to replace the
scripted multi-joint targets with actions selected by a trained policy.

### 中文回答

还不是。当前 reach-and-touch demo 是确定性的 scripted PD baseline，用来验证
环境、多关节控制、虚拟目标显示、距离测量和成功判定。Assignment 3 的左肘控制器
是用 DQN 训练的。下一阶段会用训练后的策略输出替换当前预设的多关节 target。

---

## 2. Why did you build a scripted demo before training the model? / 为什么在训练模型之前先制作 scripted demo？

### English answer

Reinforcement learning cannot compensate for an incorrectly defined or unstable
environment. The scripted baseline lets us verify that the target is reachable,
the observation is measurable, the reward inputs are valid, and the success
condition works before spending time on training. It also provides a baseline
for comparing the learned policy.

### 中文回答

如果环境定义错误或控制不稳定，强化学习训练也无法解决问题。Scripted baseline
让我们在训练之前确认目标可达、observation 可测量、reward 所需数据有效、成功判定
正确。同时它也为后续 learned policy 提供比较基线。

---

## 3. What was learned in Assignment 3? / Assignment 3 中模型学习了什么？

### English answer

In Assignment 3, a DQN learned to control the Unitree G1 left elbow. The state
contained the elbow angle, elbow velocity, goal angle, and signed error. The
agent selected decrease, hold, or increase actions. The selected checkpoint
achieved 20 successes in 20 greedy evaluation episodes across four target
angles.

### 中文回答

Assignment 3 中，DQN 学习控制 Unitree G1 左肘。State 包含肘部角度、角速度、
目标角度和有符号误差。Agent 从 decrease、hold、increase 三个动作中选择。
最终 checkpoint 在四个目标角度上的 20 次 greedy evaluation 中全部成功。

---

## 4. What exactly is the final reinforcement-learning task? / 最终的强化学习任务具体是什么？

### English answer

The final task is to control the G1 left arm through a short sequence: begin at
rest, reach a virtual target, remain inside the target region for a required
duration, and return toward a neutral pose. The final behavior will be evaluated
using success rate, endpoint distance, hold duration, episode reward, episode
length, and motion smoothness.

### 中文回答

最终任务是控制 G1 左臂完成一个短序列：从静止姿态开始，接近虚拟目标，在目标区域
内保持规定时间，然后返回 neutral pose。最终将使用成功率、手端距离、保持时间、
episode reward、episode length 和动作平滑度进行评估。

---

## 5. What will the agent observe? / Agent 将观察哪些信息？

### English answer

The initial observation will include the controlled shoulder and elbow angles
and velocities, the hand position, the target position, the hand-to-target
error, and the current task phase. We will normalize the values where necessary
and keep the observation small enough for stable training.

### 中文回答

初始 observation 将包括受控肩部和肘部的角度及速度、手部位置、目标位置、
hand-to-target error 和当前任务 phase。必要时会进行归一化，并尽量保持 state
维度适中，以提高训练稳定性。

---

## 6. What is the action space? / Action space 是如何定义的？

### English answer

Our first DQN baseline will use a limited discrete action space. Actions will
increment, decrement, or hold the controller targets for selected shoulder and
elbow joints. We will avoid unnecessary joint-action combinations because they
would make the discrete action space too large.

### 中文回答

第一版 DQN baseline 将使用规模有限的离散 action space。动作会增加、减少或保持
指定肩部和肘部关节的 controller target。我们会避免加入过多关节动作组合，因为
这会使离散动作空间迅速变大。

---

## 7. Why are you using DQN instead of PPO? / 为什么使用 DQN 而不是 PPO？

### English answer

DQN is our first choice because it directly extends our validated Assignment 3
implementation. We already have replay memory, online and target networks,
checkpoint loading, and greedy evaluation. This gives us an interpretable
baseline. If the discrete action space becomes inefficient or produces jerky
motion, PPO is our planned continuous-control comparison.

### 中文回答

DQN 是首选，因为它可以直接复用已经验证过的 Assignment 3 实现，包括 replay
memory、online/target network、checkpoint 和 greedy evaluation。这样可以得到
清晰且容易解释的 baseline。如果离散动作效率低或动作抖动明显，我们会使用 PPO
作为连续控制比较方案。

---

## 8. How do you define a touch? / 如何定义一次触碰？

### English answer

We calculate the Euclidean distance between the hand endpoint and the centre of
the virtual target. A touch is detected when this distance is less than or equal
to the target radius. The current baseline uses a target radius of 0.045 metres,
or 4.5 centimetres.

### 中文回答

我们计算手端点与虚拟目标中心之间的欧氏距离。当距离小于或等于 target radius
时，判定为触碰。当前 baseline 使用 `0.045 m`，即 4.5 cm 的目标半径。

---

## 9. Why do you require a hold duration? / 为什么要求保持一定时间？

### English answer

Without a hold requirement, the hand could briefly pass through the target and
still be counted as successful. Requiring sustained contact tests stability and
makes the task more meaningful. The current baseline requires at least 1.5
seconds of accumulated contact during the touch-and-hold phase.

### 中文回答

如果没有 hold requirement，手部只需要短暂穿过目标也可能被判定成功。要求持续
保持可以测试动作稳定性，使任务更有意义。当前 baseline 要求在 TOUCH/HOLD 阶段
累计触碰至少 1.5 秒。

---

## 10. What do `target-radius` and `required-hold` mean? / `target-radius` 和 `required-hold` 分别是什么意思？

### English answer

`target-radius` defines how close the hand must be to the target centre to count
as touching. `required-hold` defines how long the hand must remain inside that
region before the complete run is considered successful. The current defaults
are 0.045 metres and 1.5 seconds.

### 中文回答

`target-radius` 定义手部必须距离目标中心多近才算触碰；`required-hold` 定义手部
必须在该区域内保持多久才算整次任务成功。当前默认值分别是 0.045 m 和 1.5 s。

---

## 11. What is your reward function? / Reward function 是如何设计的？

### English answer

Our initial reward will encourage reduction in hand-to-target distance, provide
bonuses for entering and remaining inside the target region, and provide a
larger bonus for completing the task. It will also include small penalties for
unnecessary actions, oscillation, or unstable motion. We will tune the reward
weights through controlled experiments.

### 中文回答

初始 reward 会奖励 hand-to-target distance 的减少，奖励进入目标和在目标中保持，
并在任务完成时提供较大的 bonus。同时会对不必要动作、振荡和不稳定运动施加小的
惩罚。各项权重会通过受控实验进行调整。

---

## 12. How will you prevent the agent from only approaching the target without touching it? / 如何防止 agent 只接近目标却不真正触碰？

### English answer

Distance shaping alone may allow the agent to stop near the target. We will add
separate rewards for entering the success region, remaining there, and meeting
the full hold requirement. The episode will only be marked successful after the
hold criterion is satisfied.

### 中文回答

仅使用距离 reward 可能导致 agent 停在目标附近。因此我们会分别奖励进入成功区域、
在区域内保持以及满足完整 hold requirement。只有达到保持标准，episode 才会被标记
为成功。

---

## 13. Why does the distance increase at the end of the demo graph? / 为什么 demo 曲线最后的距离会上升？

### English answer

The increase corresponds to the RETURN phase. After the touch-and-hold
requirement is completed, the hand intentionally moves away from the target and
returns toward the neutral pose. Therefore, increasing target distance at the
end is expected behavior rather than a failure.

### 中文回答

距离增加对应 RETURN 阶段。在完成 touch-and-hold 后，手部会主动离开目标并返回
neutral pose。因此曲线最后上升是预期行为，不代表失败。

---

## 14. What did the current demo achieve? / 当前 demo 实现了什么结果？

### English answer

The validated run completed all four phases, reached a minimum endpoint distance
of approximately 0.0012 metres, and accumulated approximately 2.002 seconds of
contact. It exceeded the 1.5-second requirement and returned `success=True`.
These values describe the scripted baseline on the current fixed target, not a
learned-policy result.

### 中文回答

当前验证运行完成了四个阶段，最小手端距离约为 0.0012 m，累计触碰约 2.002 s，
超过 1.5 s 的要求并返回 `success=True`。这些数据描述的是当前固定目标上的
scripted baseline，不是 learned policy 的训练结果。

---

## 15. How will you show that the model actually learned? / 如何证明模型确实学到了策略？

### English answer

We will separate training from evaluation, save the trained checkpoint, disable
exploration during evaluation, and test the policy on multiple fixed target
positions and random seeds. We will compare it with an untrained policy and the
scripted baseline. A single successful animation will not be treated as
sufficient evidence.

### 中文回答

我们会分开 training 和 evaluation，保存训练后的 checkpoint，在评估时关闭
exploration，并在多个固定目标位置和随机种子上测试。结果会与未训练策略和
scripted baseline 比较。单次成功动画不会被视为充分证据。

---

## 16. How many evaluation episodes will you run? / 计划运行多少个 evaluation episodes？

### English answer

Our initial plan is to evaluate at least five reachable target positions across
five fixed seeds, giving at least 25 greedy evaluation episodes. We will use the
same targets and seeds for every policy being compared so that the results are
fair.

### 中文回答

初步计划是至少使用 5 个可达目标位置和 5 个固定随机种子，共至少 25 次 greedy
evaluation。比较不同策略时会使用相同的目标和 seed，以保证公平。

---

## 17. What metrics will you report? / 将报告哪些评估指标？

### English answer

We will report success rate, cumulative reward, minimum and final endpoint
distance, hold duration, episode length, and a smoothness measure based on
changes in joint targets or joint velocity. We will report means and variation
across evaluation episodes rather than only the best run.

### 中文回答

我们会报告成功率、累计 reward、最小和最终手端距离、保持时间、episode length，
以及基于 joint target 变化或 joint velocity 的平滑度指标。报告会包含多次评估的
均值和波动，而不是只展示最好的一次。

---

## 18. How will you measure smoothness? / 如何衡量动作平滑度？

### English answer

We can measure smoothness using the average magnitude of consecutive changes in
joint targets, actions, joint velocities, or accelerations. A smoother policy
should complete the task with less oscillation while maintaining a similar or
better success rate.

### 中文回答

可以使用相邻时间步 joint target、action、joint velocity 或 acceleration 的平均
变化幅度衡量平滑度。更平滑的策略应在保持相近或更高成功率的同时产生更少振荡。

---

## 19. Will the target always be in the same position? / 目标是否始终位于同一个位置？

### English answer

The current validation demo uses one fixed reachable target so that the mechanics
are deterministic and easy to debug. Training will use randomized targets from
a reachable region. Evaluation will use a fixed set of target positions so that
different policies can be compared fairly.

### 中文回答

当前 validation demo 使用一个固定可达目标，以便保持确定性并方便调试。训练阶段
会从可达区域内随机生成目标；评估阶段会使用固定目标集合，以便公平比较不同策略。

---

## 20. How do you know that the target positions are reachable? / 如何确认目标位置是机械臂可以到达的？

### English answer

We will define the target sampling region using positions that can be reached by
the controlled shoulder and elbow joints within their MuJoCo joint limits. We
can validate candidate targets using forward kinematics or the scripted
controller before including them in the training distribution.

### 中文回答

目标采样区域会根据受控肩部和肘部在 MuJoCo joint limits 内能够到达的位置定义。
在加入训练分布之前，可以使用 forward kinematics 或 scripted controller 验证候选
目标是否可达。

---

## 21. Why are you using a fixed-base robot? / 为什么使用固定底座机器人？

### English answer

The fixed base isolates the arm-control and reinforcement-learning problem from
whole-body balance and locomotion. This keeps the project scope achievable and
allows us to evaluate the reach-and-touch policy clearly. Whole-body balance is
outside the current project scope.

### 中文回答

固定底座可以把机械臂控制和强化学习问题与全身平衡、行走问题分离，使项目范围可控，
也能更清楚地评估 reach-and-touch policy。全身平衡不属于当前项目范围。

---

## 22. Why use task phases instead of learning the whole sequence end to end? / 为什么使用任务阶段，而不是端到端学习整个序列？

### English answer

Task phases make the initial problem easier to define, debug, and evaluate. They
also prevent the meanings of reach, hold, and return rewards from becoming
ambiguous. After obtaining a reliable baseline, we can investigate whether a
single end-to-end policy can complete the full sequence without an explicit
phase variable.

### 中文回答

Task phase 让初始问题更容易定义、调试和评估，也可以避免 reach、hold 和 return
阶段的 reward 含义混淆。在获得可靠 baseline 后，可以进一步研究不使用显式 phase
变量的端到端策略。

---

## 23. What happens if DQN fails or produces jerky movement? / 如果 DQN 失败或产生抖动动作怎么办？

### English answer

We will first inspect the action design, reward scaling, observation
normalization, and training stability. If the main limitation is the discrete
action space, we will compare DQN with PPO, which can output continuous or
smoother joint-target changes. DQN will still remain a useful baseline.

### 中文回答

我们会先检查 action design、reward scaling、observation normalization 和训练
稳定性。如果主要限制来自离散动作空间，就会使用 PPO 进行连续或更平滑的 joint
target 控制比较。即使如此，DQN 仍然是有价值的 baseline。

---

## 24. What are the main risks in the project? / 项目的主要风险是什么？

### English answer

The main risks are an action space that is too large, sparse or poorly scaled
rewards, unstable multi-joint motion, and overfitting to one target position.
We reduce these risks by starting with a small discrete action set, using dense
distance feedback plus explicit success bonuses, validating reachable targets,
and evaluating across multiple targets and seeds.

### 中文回答

主要风险包括 action space 过大、reward 稀疏或缩放不合理、多关节运动不稳定，以及
对单一目标位置过拟合。我们会从较小的离散动作集开始，使用距离反馈和明确成功奖励，
验证目标可达性，并在多个目标和 seed 上进行评估。

---

## 25. What feedback do you need from the dry run? / 希望通过 dry run 获得哪些反馈？

### English answer

We would like confirmation that our touch definition, hold criterion, task
phases, and evaluation metrics are appropriate before full training. We would
also like guidance on whether returning to neutral should be part of the main
success condition or reported as a separate metric.

### 中文回答

我们希望在全面训练前确认 touch definition、hold criterion、task phases 和评估
指标是否合理。同时希望老师建议 return to neutral 应作为主要成功条件的一部分，
还是单独报告的指标。

## Numbers Everyone Should Remember / 全员需要记住的数字

| Item / 项目 | Value / 数值 | Meaning / 含义 |
|---|---:|---|
| A3 evaluation | 20 / 20 | All greedy evaluation episodes succeeded / Greedy evaluation 全部成功 |
| A3 success rate | 100% | Validated single-elbow DQN result / 已验证的单肘 DQN 结果 |
| Target radius | 0.045 m | Touch region radius, equal to 4.5 cm / 触碰区域半径 4.5 cm |
| Required hold | 1.5 s | Minimum accumulated touch time / 最短累计触碰时间 |
| Demo minimum distance | 0.0012 m | Approximately 1.2 mm for the scripted run / Scripted run 约 1.2 mm |
| Demo hold duration | 2.002 s | Exceeded the required hold threshold / 超过规定保持时间 |

## Phrases to Use / 推荐说法

- **English:** The current demo is an environment-validation baseline, not the
  final learned policy.
- **中文：** 当前 demo 是环境验证 baseline，不是最终 learned policy。
- **English:** Assignment 3 validated our DQN implementation; the final project
  extends it to a multi-joint task.
- **中文：** Assignment 3 验证了 DQN 实现，最终项目将它扩展到多关节任务。
- **English:** We will use repeated greedy evaluation rather than one successful
  animation as evidence.
- **中文：** 我们将使用多次 greedy evaluation，而不是用一次成功动画作为证据。

## Phrases to Avoid / 避免使用的说法

- Avoid: **Our RL agent has learned the reach-and-touch task.**
- 原因：当前 reach-and-touch demo 仍是 scripted controller。
- Avoid: **The demo proves that DQN works for multi-joint control.**
- 原因：当前结果只证明任务 mechanics 和 baseline controller 可以运行。
- Avoid: **A 1.2 mm distance is the accuracy of our trained model.**
- 原因：1.2 mm 是 scripted baseline 在当前固定目标上的结果。
