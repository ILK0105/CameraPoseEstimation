import os
import sys
import csv
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from glob import glob
from model import NetVLADModel
from config import DATA_ROOT, OUTPUT_DIR, CKPT_PATH
'''
Newly defined libraries should be listed here.
'''

# TartanAir pose format: tx ty tz qx qy qz qw
#   (tx, ty, tz)     : camera optical center in the world frame (NED), meters
#   (qx, qy, qz, qw) : camera orientation w.r.t. the world frame  (NOTE: w is LAST)
POSE_COLS = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]


def load_pose_csv(path):
    with open(path) as f:
        return {r["image_id"]: np.array([float(r[c]) for c in POSE_COLS]) for r in csv.DictReader(f)}


class SceneImages(Dataset):
    """
    Images of one scene.
      kind="database": pose is known  -> self.poses (N, 7)
      kind="query"   : pose is unknown -> self.poses is None
    """
    def __init__(self, scene_dir, kind, transform=None):
        self.paths = sorted(glob(os.path.join(scene_dir, kind, "images", "*.png")))
        self.ids = [os.path.splitext(os.path.basename(p))[0] for p in self.paths]
        self.poses = None
        if kind == "database":
            poses = load_pose_csv(os.path.join(scene_dir, "database", "poses.csv"))
            self.poses = np.stack([poses[i] for i in self.ids])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, idx

@torch.no_grad()
def extract_embeddings(model, dataset, device):
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    return torch.cat([model(imgs.to(device)).cpu() for imgs, _ in loader], dim=0)

def pose_error(pose_q, pose_db):
    p1 = np.asarray(pose_q, dtype=np.float64)
    p2 = np.asarray(pose_db, dtype=np.float64)

    t_err = float(np.linalg.norm(p1[:3] - p2[:3]))

    q1, q2 = p1[3:7].copy(), p2[3:7].copy()
    n1, n2 = np.linalg.norm(q1), np.linalg.norm(q2)
    if n1 == 0 or n2 == 0:
        r_err = np.nan
    else:
        q1 /= n1; q2 /= n2
        dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
        r_err = float(2.0 * np.degrees(np.arccos(dot)))

    return t_err, r_err

def estimate_poses(q_vecs, db_vecs, query_ds, db_ds):
    """
    Your Task) Estimate the camera pose of every query image of ONE scene.

    Inputs:
        q_vecs  (Nq, D) : NetVLAD descriptors of the query images
        db_vecs (Nd, D) : NetVLAD descriptors of the database images
        query_ds        : query images   (query_ds.paths, query_ds.ids)
        db_ds           : database images (db_ds.paths, db_ds.ids, db_ds.poses (Nd, 7))

    Returns:
        np.ndarray (Nq, 7) : [tx, ty, tz, qx, qy, qz, qw] for each query, in query_ds order

    You can add any ideas (top-K pose averaging/interpolation, local feature matching,
    geometric verification, PnP, SfM, ...).
    """
    # -------------------------------
    # TASK4 : Camera Pose Estimation
    # -------------------------------
    # example code: copy the pose of the top-1 retrieved database image
    # dist_mat = torch.cdist(q_vecs, db_vecs, p=2)
    # best_db_idxs = dist_mat.argmin(dim=1).numpy()
    # return db_ds.poses[best_db_idxs]
    # -------------------------------

    return ...

if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"   # "dev": local check with GT | "test": submission
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NetVLADModel(num_clusters=16).to(device)
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    split_dir = os.path.join(DATA_ROOT, "camera", split)
    scenes = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))

    predictions = {}
    for scene in scenes:
        query_ds = SceneImages(os.path.join(split_dir, scene), "query", transform)
        db_ds = SceneImages(os.path.join(split_dir, scene), "database", transform)
        print(f"[{scene}] Query: {len(query_ds)} | Database: {len(db_ds)}")

        q_vecs = extract_embeddings(model, query_ds, device)
        db_vecs = extract_embeddings(model, db_ds, device)
        est = np.asarray(estimate_poses(q_vecs, db_vecs, query_ds, db_ds), dtype=np.float64)
        assert est.shape == (len(query_ds), 7), f"estimate_poses must return ({len(query_ds)}, 7)"
        predictions.update(zip(query_ds.ids, est))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = "pose_submission.csv" if split == "test" else f"pose_submission_{split}.csv"
    with open(os.path.join(OUTPUT_DIR, out_name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id"] + POSE_COLS)
        for qid, pose in predictions.items():
            w.writerow([qid] + [f"{v:.6f}" for v in pose])
    print(f"Saved {len(predictions)} query poses -> {os.path.join(OUTPUT_DIR, out_name)}")

    # dev only: the query GT is public, evaluate locally
    if split == "dev":
        gt = load_pose_csv(os.path.join(split_dir, "dev_query_gt.csv"))
        rows = [(qid, *pose_error(predictions[qid], gt[qid])) for qid in gt]
        with open(os.path.join(OUTPUT_DIR, "result_pose_eval.txt"), "w") as f:
            f.write(f"Pose evaluation on camera/dev (queries={len(rows)})\n\n")
            for qid, t_err, r_err in rows:
                f.write(f"{qid}: Translation Error: {t_err:.4f} m, Rotation Error: {r_err:.2f} deg\n")
        mean_t = float(np.nanmean([r[1] for r in rows]))
        mean_r = float(np.nanmean([r[2] for r in rows]))
        print(f"Mean Translation Error: {mean_t:.4f} m")
        print(f"Mean Rotation Error: {mean_r:.2f} deg")
