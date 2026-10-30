# PA1: NetVLAD

- Due: 23:59 (KST) 31st October, 2026
- TA: Seungho Park(gomi0904@yonsei.ac.kr)
- Kaggle competition: https://www.kaggle.com/t/f6eb5c5317ba414c8730508597bdb4a4
- Dataset: https://drive.google.com/file/d/1pDErcDJLocm_T-kd_cpOZSk8ipX8Wyup/view?usp=sharing

**NO PLAGIARISM, NO DELAY**

## Overview

### Before you start, make sure you understand the code flow by reading base.py

| Task | File | Evaluated by |
|---|---|---|
| 1. VGG16-based NetVLAD structure | `model.py` | code |
| 2. Learning with ranking loss | `train.py` | code |
| 3. Image retrieval, Recall@K | `eval.py` | code + **Kaggle leaderboard** |
| 4. Camera pose estimation | `test.py` | `pose_submission.csv` on LearnUS (hidden GT) |
| Report | | LearnUS |

## Data

Download `PA1_data.zip` (about 650 MB) from the Google Drive link above and unzip it next to the code.
This creates `./data`. If you put it somewhere else, change `DATA_ROOT` in `config.py`.

```
data/
├── train/                      # Task 1-2: training (GPS ground truth public)
│   ├── query/  index/
│   ├── query_train.txt  index_train.txt
│   └── gt/pitts30k_train.npz
├── val/                        # Task 3: local Recall@K check (GPS ground truth public)
│   ├── query/  index/
│   ├── query_val.txt  index_val.txt
│   └── gt/pitts30k_val.npz
├── test/                       # Task 3: Kaggle submission (ground truth hidden)
│   ├── query/q_XXXX.jpg  index/db_XXXX.jpg
│   └── query_test.txt  index_test.txt
├── camera/                     # Task 4
│   ├── dev/<scene>/database/{images/, poses.csv}
│   ├── dev/<scene>/query/images/
│   ├── dev/dev_query_gt.csv    # query poses of dev scenes (for local evaluation)
│   ├── test/<scene>/database/{images/, poses.csv}
│   ├── test/<scene>/query/images/          # query poses hidden
│   └── sample_pose_submission.csv
└── sample_submission.csv
```

train / val / test cover different parts of the city (no shared locations).

## Requirements

- **Python Version:** 3.9 or above
- **Libraries:** numpy, pillow, tqdm, torch, torchvision (>= 0.13)

```
pip install numpy pillow tqdm torch torchvision
```

## Usage

```
python base.py        # Task 1-3: train, Recall@K on val, writes output/submission.csv
python test.py dev    # Task 4: pose estimation on dev scenes, prints errors (GT public)
python test.py test   # Task 4: pose estimation on test scenes, writes output/pose_submission.csv
```

Run the code on any machine you like (your own GPU, a lab server, Colab, ...). A GPU is strongly
recommended. Kaggle is used only for scoring: upload `output/submission.csv` on the competition
page (**Submit Prediction**). Only the submitted files are scored.

## Task 2: Ranking loss

Training tuples come from `LoadDataset` in `base.py`:

- Positives of a query `q`: database images within **25 m** of `q` (GPS).
- Negatives of `q`: database images farther than 25 m.
- A location is photographed in every direction, so a random positive often shows a different scene.
  `LoadDataset.mine()` therefore re-selects tuples with the current model at the start of every epoch.
  For each query it keeps the positives closest in descriptor space and the hardest negatives (NetVLAD paper, Sec. 4).
- Before training, `init_centroids()` in `base.py` initializes the NetVLAD centroids and soft-assignment with k-means
  (NetVLAD paper, Sec. 3.1). It needs the attribute names given in `model.py` (`backbone`, `centroids`, `assign`, `num_clusters`).

Implement the loss from the lecture / paper:

$$L = \sum_j l\left(\min_i d^2(q, p_i) + m - d^2(q, n_j)\right), \quad l(x) = \max(0, x)$$

## Task 3: Kaggle leaderboard

- For each test query, submit the ids of the **top-10** database images, best match first.
- A retrieved image is correct if it lies within **25 m** of the query.
- $\text{hit}_q@K = 1$ if the top-K contains a correct image. $R@K = \frac{1}{N}\sum_q \text{hit}_q@K$.
- **Score = mRecall = (R@1 + R@5 + R@10) / 3**
- All test queries are scored. The leaderboard score you see is your final score.
- Set your Kaggle team name to your **student ID**.

Submission format (`sample_submission.csv`). `generate_submission()` in `eval.py` writes it for you.
```
query_id,top1,top2,top3,top4,top5,top6,top7,top8,top9,top10
q_0000,db_0697,db_0752,db_0708,db_1299,db_0693,db_0579,db_1653,db_0494,db_0893,db_0593
```

## Task 4: Camera pose estimation

- Each scene has database images with known poses (`database/poses.csv`) and query images without poses.
- Estimate the pose of every query image by using retrieval results and matching information. **You can add any ideas** (top-K pose interpolation, local feature matching, geometric verification, PnP, SfM, ...).
- Open libraries are allowed in Task 4 only.
- `camera/dev`: query GT is given for your own evaluation. `camera/test`: query GT is hidden and used for grading.

**Pose format (TartanAir): `tx ty tz qx qy qz qw`**
- `tx ty tz`: position of the camera optical center in the world frame (NED), meters.
- `qx qy qz qw`: orientation quaternion of the camera w.r.t. the world frame. **`w` is the LAST element.**
- Details: https://github.com/castacks/tartanair_tools/blob/master/data_type.md

Metrics (averaged over all test queries):
- Translation error $E_t = \lVert \hat t - t \rVert_2$ [m]
- Rotation error $E_R = 2\cos^{-1}(|\hat q^\top q|)$ [deg]

`pose_submission.csv` (written by `python test.py test`):
```
query_id,tx,ty,tz,qx,qy,qz,qw
hospital_P000_q000,-12.254475,22.780603,-7.539531,-0.252453,-0.177866,-0.947394,0.084115
```

## Grading

| Item | Points |
|---|---|
| Task 1. NetVLAD structure | 3 |
| Task 2. Ranking loss + training | 3 |
| Task 3. Recall@K implementation + valid Kaggle submission | 3 |
| Task 3. Good performance (Kaggle leaderboard mRecall) | up to +4 (thresholds TBD) |
| Task 4. Camera pose estimation | 3 |
| Task 4. Good performance (hidden test scenes) | up to +4 (thresholds TBD) |
| Report | 3 |
| Own idea | up to +2 |

### Rules
- Train only on the provided train/val splits. Do not train on test images or label them by hand.
- The only pretrained weights allowed for Task 1-3 are torchvision's ImageNet VGG16 weights.
- The NetVLAD structure (Task 1) must remain. You may change the training details (hyper-parameters, number of clusters, mining, augmentation, image size, ...).

## Submission

- **Kaggle**: `submission.csv`
- **LearnUS**: `<student_id>.zip`
  ```
  <student_id>.zip
  ├── base.py, config.py, model.py, train.py, eval.py, test.py
  ├── pose_submission.csv      # python test.py test
  └── report.pdf               # under 3 pages: method, results (incl. Kaggle score screenshot), analysis
  ```
