# Summary
## Goal
I aim to train a world model that learns how a climber’s body moves during boulder problems on the Kilter Board (a rock climbing training board with standardized holds + hold locations). Given a boulder problem defined as a subset of these holds, the goal is for the model to be able to autoregressively predict a sequence of full-body poses representing a (hopefully) plausible (and even more hopefully, good) way to climb the problem, outputting a pose overlay animation on the wall. The model will be trained on pose sequences from videos of many different boulders on the Kilter Board, and evaluated against a nearest-neighbor baseline to test whether the world model actually learned dynamics successfully and can generalize to new hold configurations.

## Inputs/Outputs
- **Input:** A boulder problem (a set of holds on the Kilter Board, each with board coordinates and role: start/middle/finish/foot-only).
- **Output:** A sequence of full-body poses (17 COCO keypoints per frame) representing a plausible way to climb the problem, rendered as a skeleton overlay animation on the board.

## Deliverables
1. A data pipeline from raw climbing video -> cleaned pose sequences in board coordinates (ViTPose + homography calibration).
2. A trained autoregressive world model that predicts pose sequences given a boulder problem.
3. A quantitative comparison against a greedy climber + inverse kinematics baseline on held-out problems.
4. Skeleton overlay visualizations of predicted climbs.

# Important changes since second checkpoint
- Ignoring head keypoints
- Option to ignore unused holds in climbs
- Removed climbs that may be causing issues:
    - Traverses
    - Any climb with beta that includes moves that aren't the expected standard static moves (cutting feet + bumping feet, more dynamic moves, etc.)

# Second checkpoint (5/22, sorry this is a couple hours late)
## Progress since first checkpoint
**Completed:**
- Homography calibration pipeline (pixel space -> board space for both main board and kickboard)
- Pose cleaning (detecting any brief anomalies or gaps and interpolating)
- Dataset construction with train/test split by problem
- Rough greedy IK baseline implementation
- Rough transformer world model trained on pose sequences with problem context (hold positions + roles)
- Structured model variant that receives explicit target hold conditioning
- Autoregressive rollout animation
- Evaluation script comparing all methods side-by-side

**Current dataset:** 87 videos (69 train, 18 test), 64750 total frames (51150 train, 13600 test)

## Evaluation

### Evaluation plan

**Quantitative (automated, on held-out routes):**
- **Teacher forcing:** Feed ground truth previous pose, predict next frame. Reports mean keypoint error across all test frames. Measures single-step accuracy: can the model predict the immediate next pose given perfect input?
- **Autoregressive rollout:** Seed with the first ~1 second of ground truth frames, then predict using the model's own outputs. Reports error at 25/50/75/100% through the climb to measure how errors accumulate.

**Qualitative (manual inspection):**
- Skeleton overlay videos of predicted climbs: do they look like plausible climbing movement?
- Specific things to look for: does the model reach for the right holds? Are limb lengths stable? Does it produce biomechanically plausible poses?

### Methods compared

| Method | Description |
|---|---|
| Static baseline | Predict the first frame's pose for all subsequent frames. Lower bound, no movement at all. |
| Greedy IK | From current pose, move the nearest limb to the nearest unused hold. Interpolation between keyframe poses. |
| World model (direct) | Transformer autoregressively predicts next-frame pose delta from a context window of recent poses + all route hold positions/roles. |
| World model (structured) | Same architecture but receives explicit target hold position as a conditioning token. |

### Model details

The world model is a transformer encoder that takes two kinds of tokens as input: a context window of recent poses (each a flattened 17-keypoint COCO skeleton) and the route's hold positions with role embeddings (start/middle/finish/foot-only). Self-attention lets pose tokens attend to hold tokens, so the model can learn which holds are relevant to the current movement. The model predicts a pose delta, which is added to the last context frame to produce the next absolute pose.

Training uses Gaussian noise injection on context frames (to prevent overfitting to exact ground-truth inputs), a displacement-weighted MSE loss (to weight rare frames with actual movement over the many slower-moving frames), a limb-length penalty (to discourage limbs from stretching unreasonably), and scheduled sampling (to expose the model to its own drift during training).

The structured variant adds a conditioning token encoding the current target hold position, derived from the ground truth hold visit order. The idea is that this would let the model focus just on motion execution.

### Quantitative results

All metrics are mean keypoint position error in board-space units, computed on the held-out test set.

TF: Teacher-forcing. AR: Autoregressive rollout.

| Method | TF Mean | TF Med | AR p25 | AR p50 | AR p75 | AR p100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Static (repeat frame 0) | 58.26 | 59.43 | 67.70 | 98.23 | 78.97 | 127.08 |
| Greedy IK | 34.98 | 21.79 | 46.32 | 96.17 | 8.51 | 125.56 |
| best_structured (structured) | 0.55 | 0.51 | 71.06 | 203.13 | 101.07 | 1495.32 |
| best_unstructured (direct) | 0.58 | 0.54 | 51.56 | 65.09 | 262.31 | 597.52 |

Not promising... will be coming to office hours soon.

### Qualitative results
Overall, the world model seems to have a few fundamental issues:
- It has a hard time generating distinct "moves." It just continuously warps and drifts. After some changes and improvements it seems to sometimes emulate climbing movements/poses while drifting in the general direction.
- Even with hold conditioning, it's very hard to force the world model to actually use the holds and follow the obvious trajectory. Often times, a limb will vaguely try to reach toward a hold while the whole skeleton is drifting away.
- With ~70 training videos, frame-level autoregressive prediction may not have enough data to learn climbing dynamics. A move-level approach (predict the next stable pose at each hold, rather than every intermediate frame) could reduce the prediction horizon and make the problem more feasible.

### What still needs to be done
- Go to office hours and ask for advice.
- Refine existing models.
- Think about move-level prediction instead of frame-level prediction.


# First checkpoint (5/8)
## Progress
**Completed:**
- Project scope and plan defined
- Pipeline for extracting keypoints from training videos (and verifying that it works on the kinds of videos I'm taking)
- Explored SQLite database from boardlib to see how to parse climb information, hold positions, etc.
- Basic evaluation code with incredibly naive baseline (static climber).

**In Progress:**
- Collecting enough training videos
- Handling occasional frames with faulty/nonexistent keypoints/person detection.
- Homography to convert from camera space to Kilter board space (for both main board and kickboard).

## Evaluation
### What are the questions your project aims to answer?
**Overarching Question:**
Is it possible for an autoregressive world model to learn climbing (a complex sport with intricate technique, movement, and interaction with holds) dynamics from keypoints extracted from climbing videos?

If so, there are additional questions I want to answer:
- How does it compare to naive baselines? The baselines I currently have in mind are a "greedy climber + inverse kinematics" model and a "hold sequence + inverse kinematics" model (neither of these are the baseline that is currently implemented in the evaluation code).
- Is encoding the positions of holds enough? What about encoding the type of hold as well? The direction of the hold?
- Is it better to have the model learn to directly predict poses or first predict the sequence of holds, then derive poses between steps of the sequence? I.e., does decomposing the prediction improve the model?
    - A lot of existing work only predicts hold sequence, not pose.

### What experiments should be done to answer that question, and how will you know from the outcome of the experiment that you have succeeded?

**Primary comparison:**

| Model | Description |
|---|---|
| Greedy climber + Inverse Kinematics | From current pose, move the nearest limb to the nearest unused hold. Derive body pose geometrically via simple inverse kinematics. Interpolate between stable states. |
| World model (direct) | Autoregressively predict the next frame's pose from the current frame's pose + problem context (all holds and their positions). The model must learn both which hold to target and how to move toward it. |

**Ablations on hold encoding:**

| Encoding | Features per hold |
|---|---|
| Position only | (x, y) |
| + hold type | (x, y, hold_type) |
| + direction (8 cardinal directions) | (x, y, hold_type, direction_8) |

**Potential additional ablations (if time permits):**

| Model | Description | Question to Answer |
|---|---|---|
| World model (predict sequence -> predict pose) | The model first predicts the overall sequence of the climb (in what order do the limbs use the different holds). Then, the model autoregressively predicts the next frame's pose from the current frame's pose + problem context + sequence context. | Does separating the sequence learning from the movement dynamics improve performance? |

**Quantitative metrics (all computed on held-out problems):**
- Per-frame mean keypoint position error under teacher forcing (input ground truth previous pose, predict next frame: measures single-step accuracy)
- Per-problem accumulated error (input model's own predictions autoregressively for the full climb: measures whether errors accumulate or stay bounded)

**Qualitative evaluation (by me, acting as the "expert"):**
- Visual inspection of the generated beta (the climber's term for "how to climb a problem") to see if it's plausible
- On-the-wall testing: attempt the generated betas on the actual board and evaluate their plausibility and quality

**What success looks like:**
- The world model should outperform the greedy climber baseline, demonstrating that learned dynamics beat naive heuristics
- The world model should produce somewhat plausible continuous motion.
- If the world model does not beat the baseline: either the data scale or the representation is insufficient to learn dynamics

### Current Evaluation Status

I created a rough evaluation script (run_eval_baselines.py) and tested it on a very naive baseline (climber doesn't move). You can run it yourself on the one json file containing keypoints extracted from a video of me climbing a problem called "Floats your boat" that I have uploaded to the github: `python scripts/run_eval_baselines.py data/poses/floats_your_boat.json`

Here are the results on a handful of boulder problems (most of my data is gitignored due to space, so you won't see them in the data folder right now):
```
=== Static Pose Baseline ===
Baseline: predict first frame's pose for all subsequent frames

centurion (505 frames):
  Teacher forcing >>> mean: 287.15, median: 285.09, max: 562.79
  Autoregressive  >>> error at 25%: 149.36, 50%: 289.44, 75%: 443.03, 100%: 561.04
cheers (1225 frames):
  Teacher forcing >>> mean: 352.34, median: 315.84, max: 682.15
  Autoregressive  >>> error at 25%: 301.35, 50%: 306.00, 75%: 471.67, 100%: 675.14
contact_wombat (423 frames):
  Teacher forcing >>> mean: 214.38, median: 194.91, max: 424.00
  Autoregressive  >>> error at 25%: 158.07, 50%: 196.74, 75%: 300.99, 100%: 419.77
doobies_bounce (702 frames):
  Teacher forcing >>> mean: 277.63, median: 257.08, max: 553.98
  Autoregressive  >>> error at 25%: 112.42, 50%: 257.08, 75%: 426.23, 100%: 550.95
joint_efforts (817 frames):
  Teacher forcing >>> mean: 335.70, median: 303.92, max: 679.87
  Autoregressive  >>> error at 25%: 195.98, 50%: 303.62, 75%: 536.48, 100%: 677.46
mr_goobler (767 frames):
  Teacher forcing >>> mean: 354.37, median: 334.47, max: 634.09
  Autoregressive  >>> error at 25%: 233.43, 50%: 330.25, 75%: 507.72, 100%: 614.99
preetty_legit_4a (674 frames):
  Teacher forcing >>> mean: 258.56, median: 249.75, max: 520.02
  Autoregressive  >>> error at 25%: 117.57, 50%: 248.40, 75%: 405.24, 100%: 515.58
spider_boy (603 frames):
  Teacher forcing >>> mean: 269.48, median: 248.90, max: 511.12
  Autoregressive  >>> error at 25%: 152.25, 50%: 245.13, 75%: 402.44, 100%: 505.13
tap_tap_tap (845 frames):
  Teacher forcing >>> mean: 308.00, median: 333.01, max: 634.31
  Autoregressive  >>> error at 25%: 155.58, 50%: 335.99, 75%: 454.89, 100%: 632.64
warm_meh_2 (659 frames):
  Teacher forcing >>> mean: 247.40, median: 254.61, max: 545.16
  Autoregressive  >>> error at 25%: 107.58, 50%: 258.27, 75%: 360.99, 100%: 543.28
you_dont_know_me (640 frames):
  Teacher forcing >>> mean: 299.32, median: 270.53, max: 639.82
  Autoregressive  >>> error at 25%: 110.07, 50%: 270.53, 75%: 489.53, 100%: 639.12
bump_it (884 frames):
  Teacher forcing >>> mean: 321.85, median: 321.05, max: 602.34
  Autoregressive  >>> error at 25%: 198.49, 50%: 327.18, 75%: 442.51, 100%: 595.56
cchhhhooooccccoooolllaaatteee (771 frames):
  Teacher forcing >>> mean: 224.31, median: 217.08, max: 463.30
  Autoregressive  >>> error at 25%: 139.11, 50%: 216.01, 75%: 302.68, 100%: 458.35
chalk_the_chalk (805 frames):
  Teacher forcing >>> mean: 290.98, median: 291.54, max: 634.60
  Autoregressive  >>> error at 25%: 100.99, 50%: 289.98, 75%: 432.60, 100%: 630.04
chanel (692 frames):
  Teacher forcing >>> mean: 317.32, median: 301.45, max: 611.09
  Autoregressive  >>> error at 25%: 168.34, 50%: 296.70, 75%: 474.17, 100%: 606.42
easy_living (982 frames):
  Teacher forcing >>> mean: 371.37, median: 379.18, max: 655.28
  Autoregressive  >>> error at 25%: 251.25, 50%: 379.21, 75%: 511.81, 100%: 652.23
easy_pinch (852 frames):
  Teacher forcing >>> mean: 316.85, median: 301.49, max: 645.20
  Autoregressive  >>> error at 25%: 151.36, 50%: 298.58, 75%: 462.45, 100%: 623.60
elephant_trunk (749 frames):
  Teacher forcing >>> mean: 334.12, median: 331.00, max: 642.12
  Autoregressive  >>> error at 25%: 180.19, 50%: 331.69, 75%: 488.34, 100%: 632.83
floats_your_boat (647 frames):
  Teacher forcing >>> mean: 276.41, median: 255.19, max: 549.08
  Autoregressive  >>> error at 25%: 145.50, 50%: 254.81, 75%: 407.40, 100%: 541.88
goly (731 frames):
  Teacher forcing >>> mean: 314.64, median: 326.11, max: 588.27
  Autoregressive  >>> error at 25%: 215.92, 50%: 324.35, 75%: 448.07, 100%: 583.01
jugs (753 frames):
  Teacher forcing >>> mean: 298.62, median: 294.34, max: 530.22
  Autoregressive  >>> error at 25%: 188.72, 50%: 299.48, 75%: 420.80, 100%: 522.27
lowballin (1007 frames):
  Teacher forcing >>> mean: 308.85, median: 323.17, max: 574.61
  Autoregressive  >>> error at 25%: 138.93, 50%: 323.30, 75%: 502.77, 100%: 562.15
one_step_two_step_red_step_blue_step (878 frames):
  Teacher forcing >>> mean: 293.46, median: 294.07, max: 586.12
  Autoregressive  >>> error at 25%: 134.75, 50%: 294.07, 75%: 451.50, 100%: 581.25
pinch_me_im_dreaming (909 frames):
  Teacher forcing >>> mean: 270.97, median: 291.15, max: 505.32
  Autoregressive  >>> error at 25%: 156.23, 50%: 294.39, 75%: 359.28, 100%: 500.40
sloth_twistie_twist (585 frames):
  Teacher forcing >>> mean: 305.02, median: 284.51, max: 572.95
  Autoregressive  >>> error at 25%: 170.53, 50%: 284.37, 75%: 460.35, 100%: 572.95
sure_2 (933 frames):
  Teacher forcing >>> mean: 358.79, median: 372.58, max: 683.92
  Autoregressive  >>> error at 25%: 186.51, 50%: 373.03, 75%: 509.03, 100%: 681.45
taco_bell (938 frames):
  Teacher forcing >>> mean: 309.16, median: 319.13, max: 599.17
  Autoregressive  >>> error at 25%: 151.56, 50%: 319.43, 75%: 464.36, 100%: 598.29
tension_tension (672 frames):
  Teacher forcing >>> mean: 281.07, median: 272.83, max: 589.11
  Autoregressive  >>> error at 25%: 144.23, 50%: 275.57, 75%: 388.17, 100%: 587.86

--- Aggregate (28 videos) ---
  Teacher forcing >>> mean: 304.74, median: 299.85
  Total frames evaluated: 21620
```