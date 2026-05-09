# First checkpoint (5/8)
## Progress
**Completed:**
- Project scope and plan defined
- Pipeline for extracting keypoints from training videos (and verifying that it works on the kinds of videos I'm taking)
- Explored SQLite database from boardlib to see how to parse climb information, hold positions, etc.
- TODO: Basic evaluation code with incredibly naive baseline.

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