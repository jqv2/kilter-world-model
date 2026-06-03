# Learning to Climb: World Models vs. Reinforcement Learning for Pose Prediction on the Kilter Board

**Jerry Qu** | jerryrqu@stanford.edu

See presentation for a higher-level overview: https://docs.google.com/presentation/d/1twbGcTAWz-KbsykhDjvjRQxSfE_BA91dbVHieJnn33I/edit?usp=sharing

---

## Background and Setup

Rock climbing is an incredibly varied and complex sport. Despite what many think, climbing isn't just about relying on one's arm strength. There are many techniques climbers must employ (based on hold positions, orientations, and types; also even based on climbers' body proportions and climbing styles) to increase stability and reach, offload weight onto feet, etc. From a physics standpoint, there are so many different aspects of climbing that one must consider (friction forces depending on how one holds a sloper vs a crimp, the way the full body must coordinate to generate momentum for dynamic moves, the way foot and ankle position and engagement can change how likely a foot will slip from a bad foothold). This makes the problem of predicting "beta" (the climbers' term for "how to do a climb": the sequence of holds and body positions needed to get to the finish) interesting and also quite challenging.

The **goal** of this project is to (at least attempt to) tackle this problem, constrained to a training board called the [Kilter Board](https://settercloset.com/collections/kilter-board), a standardized training board with a set grid of holds. Two different methods were compared: a **world model** that learns body dynamics from videos of a human climber (myself), and an **RL agent** that discovers movement through a reward-driven search inside of a simple 2D physics simulator. Both methods are given information about the route and produce a sequence of frame-by-frame skeleton poses from (hopefully) the start of the climb to the end of the climb.

**Inputs:** 
- A climbing route definition
    - hold coordinates (in the board's coordinate system)
    - hold roles (start, finish, hand or foot, foot-only)
- Starting pose

**Output:** A variable-length sequence of 12-keypoint (including shoulders, arms, wrists, torso, hips, legs, ankles) 2D skeleton poses in board coordinates, one pose per frame, representing a climb attempt.

To keep the problem tractable, there are a number of **constraints**:
- Instead of trying to predict climbing poses for any kind of climb (indoor, outdoor, bouldering, lead, competition), we focus our attention on climbs on the Kilter Board at a fixed angle. Climbs were chosen to be of lower difficulty and to be more static (not requiring big dynamic moves and "cutting feet", where the feet leave the wall while the arms hold the swing after a big move, as is common for harder Kilter Board climbs), so that a smaller set of poses and unique techniques would be needed.
- We project positions and movement onto 2D
- The dataset contains videos of the same person (me) climbing: fixed body proportions and style of climbing.
- All methods will be given the order to grab hand holds (as well as the starting pose). This closely matches with how a lot of climbers will plan out a shorter climb (boulder) from the ground, at least on the first pass: it is often intuitive even without climbing the what the *hand* sequence will be. However, the foot sequence will often depend on climber height and style. So assuming the hand sequence planning has already been done, the methods will have to figure out what to do with the rest of the body to achieve that target hand sequence.
- We ignore hold type and direction. Most of the Kilter Board holds are known to be pretty ergonomic (especially the lower-difficulty climbs). The original plan was to also encode their direction and type, but that was abandoned due to lack of time.

**Prior work:** Existing work includes using computer vision to visualize climbs, detect holds, and analyze technique from videos [1]. Also see [Belay AI](https://belay.ai/) and [an earlier project of mine](https://cs231n.stanford.edu/2024/papers/using-pose-estimation-to-analyze-rock-climbing-technique.pdf). These are fundamentally different from climbing sequence *prediction*.

## Approach

### Data pipeline

The pipeline has two sources of information merged via homography + matching climb names to a database:

1. **Route definitions** come directly from the Kilter Board's SQLite database via [BoardLib](https://github.com/lemeryfertitta/BoardLib). Each route is a structured list of holds with exact positions and roles. No computer vision is needed for hold detection.
2. **Pose extraction** is done with a two-stage approach: RT-DETR [3] for person detection, then ViTPose (chosen due to its reliability in my previous computer vision climbing project) [4] for 17-keypoint COCO skeleton extraction. Per-video homography calibration transforms pixel-space keypoints into board-space coordinates. Pose cleaning handles missing frames and keypoint glitches via interpolation.

The 5 head keypoints (nose, eyes, ears) are ignored during training and prediction, leaving 12 climbing-relevant keypoints (shoulders, elbows, wrists, hips, knees, ankles). A simple circular head was added in visualizations simply based on shoulder positions.

I manually created a dataset by choosing 89 routes on the Kilter board and climbing them myself, recording a video of each one. I was aiming for more, but was limited by board availability and strength (to reduce noise in the dataset, I had to make sure my climbs did not involve too much hesitation, unnecessary movement, or backtracking, so I ended up having to do some of the climbs 2-3 times to get a good video). I split the dataset into a training set (71 routes) and a test set (18 routes).

I implemented auto-detection of hand hold order (manually edited as needed). I also ended up deciding to try to further improve the signal-to-noise ratio by editing the route definitions for each climb to exclude any holds I may have skipped when I climbed them myself.

### World model architecture

**Why?:** In principle, a world model, *with enough data*, would be able to learn climbing dynamics directly from the keypoints from video. There is no need to hand-craft a complex climbing simulator and/or carefully design a reward landscape for Reinforcement Learning.

**Architecture:** The model is a small transformer encoder (2 layers, 128 hidden dim, 4 heads). It was trained for 100 epochs with cosine-annealed learning rate (1e-3 to 1e-5) and batch size of 256, and the best (validation loss + qualitative checking) model was chosen (it ended up being best at aorund 30 epochs). Pose frames and route hold positions are independently projected into the transformer's embedding space, concatenated as a single token sequence, and processed with self-attention. The output head predicts a pose *delta* (change from current frame), not an absolute pose, so the model does not learn to simply output "stay still" for every frame. Pose tokens are fed in with a 30-frame context window, and multi-stride sampling at strides of 1, 3, 6, and 10 frames exposes the model to a wider range of movement timescales. In addition to the context window of poses and the route definition, the model receives the current target hand hold (position + which hand) as an explicit conditioning token. During autoregressive rollout, the target advances via proximity-based arrival detection, with a timeout fallback if the predicted pose doesn't reach the target hold.

The model is trained with teacher forcing (ground-truth frames as input) using displacement-weighted MSE loss (upweighting rare movement frames), limb-length regularization to encourage anatomically plausible poses, and scheduled sampling that gradually ramped up during training to bridge the teacher-forcing and autoregressive rollout gap. Gaussian noise augmentation is applied to all context frames except the last.

Many training configurations were explored (varying stride sets, displacement weighting, limb constraint methods, hold proximity loss, keypoint subsets, etc.), but autoregressive rollout error consistently remained much higher than the teacher-forcing error across, confirming that the data scale was by far the bottleneck. The initial hypothesis was that a few hundred videos/climbs (representing on the order of a few thousand transitions between holds and several tens of thousands of frames) on a constrained board would provide enough transitions to learn at least some basic sense of dynamics, or at the very least produce plausible trajectories. This turned out to be wildly over-optimistic, as was the goal of collecting that much data in the limited time I had.

### RL baseline

**Why?:** This method is not limited by data scale. In principle, an RL agent can learn to progress through climbs and use stable climbing poses if given a good enough environment and well-designed reward landscape.

**Simulator:** The agent controls a 2D ragdoll skeleton (9 rigid segments, 8 joints) built in [Pymunk](http://www.pymunk.org), a 2D rigid-body physics engine. Gravity is projected onto the wall plane to simulate a 30-degree overhang. Segment masses, centers of gravity, and moments of inertia are based on De Leva [2] body-segment proportions, and segment lengths are derived from the training data's distribution. Limbs attach to holds via pivot joints at the wrist or ankle keypoints.

The action space is hybrid: 8 continuous joint angle deltas plus 4 binary grab/release commands (one per limb), issued at 10 Hz while physics runs at 60 Hz. Joint angle deltas are not applied directly. Instead, a proportional controller converts each delta into a target angle, computes the error between the current and target angle, and sets the joint motor's angular velocity proportional to that error (gain = 10), clamped to a maximum angular speed of 4 rad/s. Torque and angle limits vary by the type of joint.

The agent receives the same ground-truth-derived hand hold order as the world model. The reward landscape includes a bonus that scales with hand proximity to the target hold, fixed bonuses for reaching target and finish holds, a per-step penalty, and stability bonuses/penalties based on center-of-gravity position relative to the support polygon of anchored limbs (the further away the center-of-gravity is from the contact polygon or line of support, the higher the penalty; hanging by the arms alone is more heavily penalized). Training uses PPO with GAE, implemented directly in PyTorch.

Key hyperparameters: learning rate 1e-4, clip epsilon 0.2, gamma 0.993, GAE lambda 0.95, 4096 frames collected per PPO update. The policy and value networks are both 2-layer MLPs with 128 hidden units.

The RL agent trains to fit a separate policy for each route. Each policy trains for ~2 million frames.

Two RL configurations were tested:

- **Current version (more plausible hip and shoulder joint limits, higher penalties based on stability, start poses that more closely match the ground truth's, excluded unused holds to be consistent with what the world model received as context):** I only had time to train this version on 7 of the 18 test routes. Movement is more physically plausible due to joint limits on shoulders and hips as well as higher stability penalties, but the agent would get stuck 1-2 holds in to the climb.
- **Earlier version (less plausible hip and shoulder joint limits, auto-generated start pose, did not exclude unused holds):** I tested this earlier version on 5 routes. Unsurprisingly, this version was more successful than the current version because the RL agent could find ways to exploit less plausible movement to flail its way up the holds. On one route, this version was able to consistently complete the full climb after the ~2 million frames of learning.

### Hands-only baseline

This was a geometry-based baseline against which both learned methods are compared. Given the same ground-truth-derived hand hold order as the other two methods, it constructs a hands-only pose at each step (with torso and legs perfectly vertical) and linearly interpolates between these poses. The arm's joint angles are solved using inverse kinematics. This baseline is meant to mimic a strong climber slowly and statically "campusing" up a boulder, relying fully on arm strength. As this is a very naive baseline, ideally a better method would be able to produce more reasonable climbing poses (and potentially more reasonable trajectories than just a piecewise linear path).

## Evaluation and Results

### Definition of success

I decided to evaluate success based on three main dimensions: trajectory, pose correctness, and pose quality.

The reason why I differentiated between pose correctness and pose quality is because there are often multiple ways to perform a climb, and the RL agent especially would be more likely to find a different way to the top than the ground truth. So while pose correctness aims to quantify how closely the sequence of poses predicted matched the sequence of poses of the ground truth, pose quality aims to quantify how plausible in general the sequence of poses predicted were. If a predicted sequence used a completely different set of techniques that were still plausible and relatively stable, it may have worse "pose correctness" but still have a high "pose quality."

My guess was that the world model would perform significantly worse than the RL agent in terms of trajectory, but that it might outperform the RL agent in terms of pose correctness and pose quality, since it is incredibly difficult to shape the reward landscape for the RL agent to penalize unstable positions while still encouraging it to explore ways to reach the next hold without getting stuck.

If the world model wasn't so data-limited, success would look like getting low errors in trajectory, pose correctness, and pose quality, especially compared to the baseline. If the RL agent had access to a more robust environment, success would look like getting moderately low errors in trajectory and low errors in pose quality, while accepting potentially higher errors in pose correctness as the RL could find other alternative "good" sequences.

### Metrics

All metrics operate in the 2D board coordinate system.

**Teacher-forcing error (world model only):** Per-frame mean keypoint prediction error when the model receives ground-truth context. Measures single-step prediction accuracy and provides a very rough lower bound on what the model has learned.

**Trajectory-aligned metrics (all methods):** These are computed after aligning the sequences by normalizing by 100 spatial progress points through the trajectory (cumulative torso centroid arc length) rather than by frame index, so temporal differences are not penalized (e.g. a frame where the torso centroid of the RL agent is 1/3 of the way through its own trajectory will be compared to a frame where the torso centroid of the ground-truth skeleton is 1/3 of the way through the ground truth trajectory). These metrics can only be used to compare full rollouts: the world model full rollout (even if it drifts in the wrong direction, since it rolls out on the full target sequence) and the hands-only baseline. These metrics can't apply to the RL agent's predictions unless the RL agent successfully makes it to the finish hold because otherwise it will not receive the full target sequence as context.

- **Mean centroid distance (trajectory):** Average distance between torso centroids at matched progress points along the trajectories, normalized by ground-truth trajectory length. Measures how closely the predicted trajectory follows the ground-truth trajectory.
- **Procrustes-aligned keypoint error (pose correctness):** At each matched progress point along the trajectories, [full-Procrustes](https://en.wikipedia.org/wiki/Procrustes_analysis)-align the predicted and ground-truth skeletons, then compute mean per-keypoint error. Then, take the mean of these values over all progress points. Measures pose shape accuracy independent of spatial drift. Value is in board units.

**Nearest-neighbor pose distance (all methods):** For each predicted frame, calculate the [Procrustes distance](https://en.wikipedia.org/wiki/Procrustes_analysis#Shape_comparison) to the closest pose in the training set, and then normalize to between 0 and 1. Then, take the mean over all frames. Measures whether predicted poses look like real climbing poses. Does not require trajectory-alignment, so it can be applied even if the RL agent does not complete the route.

**Hold visit rate (HVR):** Fraction of target hand holds reached (hand holds visited / total hand holds in sequence). Requires a visit by the correct hand (if the next target is for the left hand but a method reaches it with the right hand, that doesn't count). Counted by each method's own target-advancement mechanism (world model: predicted wrist within threshold; RL: successful anchor in the simulator). Reported as a rate between 0 and 1 since different climbs have different hand sequence lengths.

### Teacher-forcing vs. autoregressive error

*For reference, adjacent hand holds are 8 board units apart and form a grid, with footholds filling in some of the diagonals (so the closest spacing is between an adjacent hand and foothold, which is 4√2 board units).*

| | Mean keypoint error (board units) |
|---|---|
| Teacher forcing | **0.58** |
| Autoregressive (median over test set) | **58.41** |
| Ratio | ~100x |

The model can make somewhat reasonable single-step predictions but cannot maintain accuracy over hundreds of frames with such a small training set.

### Full test set: world model vs. baseline (18 climbs)

*I did not have enough time to run the RL training on all 18 climbs in the test set, so it is excluded from this comparison.*

| Method | Trajectory error (median [IQR]) | Pose correctness error (median [IQR]) | Pose plausibility error (median [IQR]) | HVR (mean) |
|---|---|---|---|---|
| Hands-Only | 0.032 [0.026, 0.037] | 7.351 [6.791, 7.827] | 0.099 [0.097, 0.103] | 1.000+/-0.000 |
| World Model | 0.482 [0.312, 5.275] | 10.598 [9.552, 11.707] | 0.176 [0.149, 0.214] | 0.091 |

The world model visited at least one hold on 8 of 18 test routes (HVR 0.12-0.33 on those routes; 0 on the remaining 10). On most routes, the model's spatial drift (as seen by the very high trajectory errors) prevented it from reaching any target holds via actual wrist proximity. Surprisingly, it even scored worse on both pose correctness and pose plausibility than the hands-only baseline. My suspicion is that the hands-only baseline poses are more plausible than I expected. There are likely many ground-truth frames where it does briefly look like I'm just hanging from my arms with legs straight down. This shows a limitation in the pose correctness and plausibility metrics, which is that they do not take into account which limbs are actually in contact with holds: when making the dataset, I made sure to not choose any climbs where I had to ever fully rely on my arms, so any poses that looked like that likely involved a foothold right below the hand-holds and me being in the middle of a move.

<img src="https://github.com/user-attachments/assets/3b96dcae-5f2b-4b69-a8b1-02a3b8b11c47" />

The metric distributions are also very skewed. There are some climbs where the world model performs "less bad", and some climbs where the world model performs very badly.


### 7-climb subset: world model vs. RL (current version) vs baseline

*This is the primary comparison. On the 7 test routes where the RL agent (current version, with joint limits) was also trained, all three methods are evaluated side by side. The RL agent reached only 1-2 holds on each route, so its trajectory and pose correctness metrics are omitted.*

| Method | Trajectory | Pose correctness | Pose plausibility (NN dist) | HVR (mean) |
|---|---|---|---|---|
| Hands-Only | 0.037 [0.031, 0.046] | 6.625 [6.488, 7.076]  | 0.103 [0.097, 0.104] | 1.000+/-0.000 |
| World Model | 2.340 [0.431, 3.897] | 10.314 [9.451, 10.947] | 0.169 [0.144, 0.194] | 0.147+/-0.118 |
| RL | N/A | N/A | 0.191 [0.134, 0.221] | 0.184+/-0.073 |

The world model visited at least one hold on 5 of these 7 test routes (HVR 0.12-0.33 on those routes; 0 on the remaining 2). As expected, the world model produced more plausible poses than the RL agent (although not by much, and both were still significantly worse at producing plausible poses than the baseline).

### 1-climb comparison: world model vs RL (earlier version) vs baseline

*This was the only setting where the RL agent completed a route. Although this isn't an apples-to-apples comparison since I excluded unused holds from the context for the world model but not for this particular RL version (for this climb, the RL agent had access to 1 foothold the world model didn't see, although in the end the RL agent opted to not use it anyways), I just wanted to show an example of a full comparison between the three methods on all metrics.*

| Method | Trajectory | Pose correctness | Pose plausibility (NN dist) | HVR |
|---|---|---|---|---|
| Hands-Only | 0.02 | 7.80 | 0.1032 | 1 (8/8) |
| World Model | 2.83 | 11.22 | 0.1988 | 0.125 (1/8) |
| RL (early version) | 0.06 | 11.24 | 0.1751 | 1 (8/8) |

For this particular climb, the world model performed very poorly (as can be seen by the extremely high trajectory error). Pose correctness and plausibility for the world model and RL are comparable and both significantly worse than the hands-only model. RL pose correctness and pose plausibility were understandably high as the RL agent tried its best to exploit the physics in the simulator to reach the top, despite my best efforts to tune the penalties.

### Qualitative results

**Video and trajectory comparison between ground-truth, baseline, world model, and RL:**

<video src="https://github.com/user-attachments/assets/210c2c25-c0d4-484c-ba06-f855e536f197" controls="controls" width="100%"></video>
<img width="2400" height="750" alt="Image" src="https://github.com/user-attachments/assets/3b96dcae-5f2b-4b69-a8b1-02a3b8b11c47" />
This is a comparison between the predicted sequences for the climb referenced in the "1-climb comparison" section above. The RL agent manages to flail its way up to the top, while the world model completely drifts in the wrong direction and outputs nonsense.

<video src="https://github.com/user-attachments/assets/ae121078-58f1-472b-b421-48a9cc3eb899" controls="controls" width="100%"></video>
<img src="https://github.com/user-attachments/assets/afefcb16-fbdd-4fa8-9d36-3c01429a2159" />
This is a comparison between the predicted sequences for one of the climbs in the "7-climb subset" section mentioned above. The RL agent gets stuck pretty quickly, while the world model actually makes two moves with somewhat plausible poses before its trajectory quickly drifts far away.

**Technique emergence in the world model:**

Despite very poor aggregate trajectory metrics, upon visual inspection, the world model occasionally produces recognizable climbing techniques (flagging and hip turns). Very occasionally (like the example above), it will produce these techniques at contextually appropriate points, but more often than not it will attempt climbing techniques randomly.

**RL agent movement style:**

Unlike the ground truth and even the world model, the RL agent moves stiffly and jerkily. It has no exposure to actual human movement data so its movements don't look as "natural." Furthermore, because we are constraining the problem to 2D, it cannot represent certain techniques that require out-of-plane motion (hip turns, drop knees).

<video src="https://github.com/user-attachments/assets/afeaba6a-0ffe-4b8e-b1a3-e505b97d3b62" controls="controls" width="100%"></video>

Furthermore, even after spending a lot of time trying to tune the reward landscape, the RL agent still acts very "greedily" and will try to find the fastest way to the next hold (some iterations of the agent on certain climbs would bypass footholds completely and monkey bar to the next holds). While this often works for the first couple of target holds which are relatively close to the start, this makes it difficult for the RL agent to learn to progress, as to get further, the RL agent would need to learn to use footholds, often in a non-greedy way (i.e. swapping feet, using feet that seemingly take you further from the hold but are necessary for stability, etc.), which is very difficult to incentivize.

## Conclusion

*World Model:* without enough data, it...
- sometimes attempts climbing techniques it sees in training data, but often in the wrong contexts
- often has no idea what to do
- almost always ends up drifting in the wrong direction partway through the climb, sometimes even right from the start

*RL Agent:* without a more robust simulator and a *very* carefully designed reward landscape, it...
- can often reach 1-2 target holds, but usually gets stuck afterwards; there is a constant push-and-pull between encouraging it with enough rewards to get it to make the effort to explore ways to reach the targets, while also penalizing it for exploring unstable/implausible poses
- will often try to find exploits (swinging/flailing to the next hold)
- will not be able to replicate some climbing techniques, especially ones that require out-of-plane motion

## Future Work

**World model:** The primary bottleneck is data scale. Collecting data at a much larger scale (probably thousands to tens of thousands of videos) would test whether the approach is at all viable with sufficient data. If trajectory quality improves with more data, further extensions would include a learned hold sequence model (removing the need for ground-truth hand sequence as input) and encoding hold geometry (direction and shape) into the route context.

**RL agent:** The simulator needs to be more robust to produce realistic movement. A 3D environment would allow representation of hip turns and drop knees. Adding friction, normal force directions, more realistic grip strength, fatigue modeling, hold geometry, etc., and switching from joint-angle-delta control to direct torque control would likely make poses and movement more realistic if an appropriate reward landscape was designed.

## Team Responsibilities

Solo project.

## References
[1] S. Ekaireb et al., “Computer Vision Based Indoor Rock Climbing Analysis,” kastner.ucsd.edu, https://kastner.ucsd.edu/ryan/wp-content/uploads/sites/5/2022/06/admin/rock-climbing-coach.pdf. 

[2] P. de Leva, “Adjustments to Zatsiorsky-Seluyanov’s segment inertia parameters,” Journal of Biomechanics, https://pubmed.ncbi.nlm.nih.gov/8872282/. 

[3] Y. Zhao et al., “DETRs Beat YOLOs on Real-time Object Detection,” arXiv.org, https://arxiv.org/abs/2304.08069. 

[4] Y. Xu, J. Zhang, Q. Zhang, and D. Tao, “ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation,” arXiv.org, https://arxiv.org/abs/2204.12484. 