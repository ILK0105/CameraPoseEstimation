import os
import math
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from config import DATA_ROOT, OUTPUT_DIR, CKPT_PATH
from model import NetVLADModel
from train import train, validate
from eval import compute_embeddings, evaluate_recall, generate_submission

# =====================================================================================
# [읽는 순서]  이 파일은 위에서 아래로 읽지 말고, 아래 순서로 따라가면 이해가 쉽다.
#   [1] main()                      : 전체 흐름 (데이터 준비 -> 모델 -> 학습 -> 평가)
#   [2] LoadDataset.__init__        : 학습용 데이터셋이 무엇을 들고 있는지
#   [3] LoadDataset.mine            : 매 epoch마다 positive / negative 를 고르는 핵심 로직
#   [4] LoadDataset.__getitem__     : 고른 번호대로 이미지를 불러와 (q, P, N) 반환
#   [5] ImageListDataset            : "이미지 한 장씩" 불러오는 단순 데이터셋 (임베딩 추출용)
#   [6] init_centroids              : NetVLAD 클러스터 중심 k-means 초기화 (가장 수학적, 마지막에)
#
# 용어
#   query      : "여기가 어디냐?" 하고 묻는 사진
#   database   : 위치(GPS/UTM 좌표)를 이미 알고 있는 사진들 (= index)
#   UTM        : 위경도를 미터 단위 평면 좌표 (x, y) 로 바꾼 것 -> 거리 계산이 그냥 유클리드 거리
#   descriptor : 모델이 이미지 한 장을 요약한 벡터 (= embedding, 크기 K*D = 16*512)
# =====================================================================================


# [2]~[4] -----------------------------------------------------------------------------
# 학습(ranking loss)에 쓰는 데이터셋. 샘플 하나 = (쿼리 1장, positive n_pos장, negative n_neg장).
# 사용 흐름:  dataset.mine(model) 으로 번호를 골라두고 -> DataLoader 가 __getitem__ 을 호출.
class LoadDataset(Dataset):
    """
    Training tuples {query, positives, negatives} with hard mining (NetVLAD paper, Sec. 4).

    Positives: database images within posDistThr (25 m) of the query.
    Negatives: database images farther than posDistThr.
    A place has views in every direction, so a random positive often does not show the
    same scene as the query. mine() therefore picks, with the CURRENT model,
      - the n_pos positives closest to the query in descriptor space, and
      - the n_neg hardest negatives out of n_sample random negatives.
    Call mine(model, device) at the start of every epoch (see train.py).

    __getitem__ returns q (3,H,W), P (n_pos,3,H,W), N (n_neg,3,H,W).
    """
    # [2] 생성자: 이미지 "경로 목록"과 "GPS 정답 좌표"만 읽어 둔다. 이미지 자체는 아직 안 읽음.
    #   n_pos    : 쿼리 하나당 쓸 positive 개수
    #   n_neg    : 쿼리 하나당 쓸 negative 개수
    #   n_sample : hard negative 를 찾을 때 후보로 무작위 추출할 negative 개수
    #              (DB 전체와 비교하면 너무 느리므로 1000장만 뽑아 그중 가장 어려운 n_neg 장을 고른다)
    def __init__(self, query_file, db_file, root_dir, gt_npz, transform=None,
                 n_pos=4, n_neg=5, n_sample=1000):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform    # 이미지 -> 텐서 변환 (main()의 Resize/ToTensor/Normalize). None 이면 __getitem__ 에서 에러남
        self.n_pos, self.n_neg, self.n_sample = n_pos, n_neg, n_sample

        # .txt 파일 한 줄 = 이미지 상대경로 하나. root_dir 를 붙여 전체 경로 리스트로 만든다.
        # 리스트에서의 위치(인덱스)가 곧 그 이미지의 "번호"이고, 아래 utmQ / utmDb 의 행 번호와 1:1 대응한다.
        with open(query_file, 'r') as f:
            self.query_images = [os.path.join(root_dir, line.strip()) for line in f if line.strip()]
        with open(db_file, 'r') as f:
            self.db_images = [os.path.join(root_dir, line.strip()) for line in f if line.strip()]

        # 정답(GT) 파일: 각 이미지가 찍힌 위치
        gt = np.load(gt_npz, allow_pickle=True)
        self.utmQ = gt["utmQ"]                       # (쿼리 수, 2)  각 쿼리의 UTM (x, y), 단위 m
        self.utmDb = gt["utmDb"]                     # (DB 수, 2)    각 DB 이미지의 UTM (x, y)
        self.posDistThr = float(gt["posDistThr"])    # 25.0 m. 이 거리 안이면 "같은 장소"
        self.pos_idx = self.neg_idx = None           # mine() 이 채운다. 채우기 전에는 __getitem__ 사용 불가

    # [3] Hard mining: "현재 모델" 기준으로 쿼리마다 학습에 쓸 positive / negative 번호를 고른다.
    #     train.py 의 train()/validate() 가 매 epoch 시작 때 호출한다.
    #     (모델이 학습되면 descriptor 가 바뀌어서 '가까운/어려운' 이미지도 바뀌므로 매번 다시 고른다)
    #     결과: self.pos_idx[i], self.neg_idx[i] = 쿼리 i 에 대한 DB 이미지 번호 배열
    @torch.no_grad()    # 고르기만 하므로 gradient 필요 없음
    def mine(self, model, device):
        # 1단계) 모든 쿼리와 모든 DB 이미지의 descriptor 를 현재 모델로 계산
        #        q_embs: (쿼리 수, K*D), db_embs: (DB 수, K*D)
        #        root_dir 에 "" 를 주는 이유: query_images/db_images 는 이미 전체 경로라서
        q_embs, _, _ = compute_embeddings(model, ImageListDataset(self.query_images, "", self.transform), device)
        db_embs, _, _ = compute_embeddings(model, ImageListDataset(self.db_images, "", self.transform), device)
        q_embs, db_embs = torch.from_numpy(q_embs).to(device), torch.from_numpy(db_embs).to(device)

        self.pos_idx, self.neg_idx = [], []
        for i in range(len(self.query_images)):
            # 2단계) GPS 로 후보 나누기 (모델과 무관, 위치 정보만 사용)
            dists = np.linalg.norm(self.utmDb - self.utmQ[i], axis=1)   # 쿼리 i 와 모든 DB 이미지 간 실제 거리(m), shape (DB 수,)
            pos = np.where(dists < self.posDistThr)[0]                 # 25m 이내인 DB 번호들 = positive 후보
            if len(pos) == 0: pos = np.array([np.argmin(dists)])       # 25m 안에 아무것도 없으면 가장 가까운 1장으로 대체
            neg = np.where(dists > self.posDistThr)[0]                 # 25m 밖인 DB 번호들 = negative 후보 (보통 수천 장)
            neg = np.random.choice(neg, min(self.n_sample, len(neg)), replace=False)  # 그중 n_sample 장만 무작위 추출

            # 3단계) descriptor 공간에서의 거리로 최종 선택 (여기서 모델이 쓰인다)
            #        q_embs[i:i+1] 은 (1, K*D). cdist 결과 (1, 후보 수) 에서 [0] 으로 (후보 수,) 로 만든다.
            d_pos = torch.cdist(q_embs[i:i+1], db_embs[pos])[0]
            d_neg = torch.cdist(q_embs[i:i+1], db_embs[neg])[0]
            # positive: descriptor 가 쿼리와 가장 가까운 n_pos 개
            #           -> 같은 25m 안이라도 "같은 방향을 찍은 사진"일 가능성이 높은 것만 남긴다
            #           argsort() 는 거리 오름차순 정렬된 "후보 내 위치"를 주므로 pos[...] 로 실제 DB 번호로 변환
            best_pos = pos[d_pos.argsort()[:self.n_pos].cpu().numpy()]
            # 후보가 n_pos 보다 적으면 반복해서 개수를 맞춘다. 예) [a, b] -> [a, b, a, b]
            # (샘플마다 P 의 크기가 같아야 DataLoader 가 배치로 쌓을 수 있기 때문)
            self.pos_idx.append(np.resize(best_pos, self.n_pos))       # repeat if fewer than n_pos
            # negative: 실제로는 다른 장소인데 descriptor 가 쿼리와 가장 가까운 n_neg 개 = hard negative
            #           -> 모델이 헷갈려하는 것만 골라야 loss 가 0 이 아니어서 학습 신호가 생긴다
            self.neg_idx.append(neg[d_neg.argsort()[:self.n_neg].cpu().numpy()])

    def __len__(self): return len(self.query_images)    # 샘플 수 = 쿼리 수

    # [4] DataLoader 가 샘플 하나를 요청할 때 호출된다. 판단은 하지 않고, mine() 이 골라둔 번호대로 이미지를 읽기만 한다.
    def __getitem__(self, idx):
        if self.pos_idx is None:
            raise RuntimeError("Call dataset.mine(model, device) before iterating (see train.py)")
        load = lambda p: self.transform(Image.open(p).convert("RGB"))    # 경로 -> (3, 224, 224) 텐서
        q_img = load(self.query_images[idx])                                     # (3, H, W)
        # P 와 N 은 코드가 같고 "어떤 번호 목록을 쓰느냐"만 다르다. 둘 다 같은 DB 이미지 풀에서 꺼낸다.
        # 리스트로 읽은 뒤 torch.stack 으로 맨 앞에 차원을 하나 추가해 쌓는다.
        P = torch.stack([load(self.db_images[j]) for j in self.pos_idx[idx]])   # (n_pos, 3, H, W)
        N = torch.stack([load(self.db_images[j]) for j in self.neg_idx[idx]])   # (n_neg, 3, H, W)
        # DataLoader 가 batch_size 개를 다시 쌓으므로 train.py 에서는
        #   q: (B, 3, H, W),  P: (B, n_pos, 3, H, W),  N: (B, n_neg, 3, H, W)
        return q_img, P, N


# [5] ---------------------------------------------------------------------------------
# 이미지 목록을 "한 장씩" 돌려주는 단순 데이터셋. positive/negative 개념 없음.
# 쓰이는 곳: mine() 과 평가에서 descriptor 계산(compute_embeddings), init_centroids, test 제출.
class ImageListDataset(Dataset):
    def __init__(self, list_file, root_dir, transform=None):
        # list_file: path to a .txt file, or a list of image paths
        # (.txt 경로를 주면 파일을 읽어 목록을 만들고, 이미 리스트면 그대로 쓴다)
        if isinstance(list_file, str):
            with open(list_file, "r") as f:
                list_file = [line.strip() for line in f if line.strip()]
        self.paths = [os.path.join(root_dir, p) for p in list_file]   # root_dir="" 이면 경로가 그대로 유지됨
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, idx, p    # (이미지 텐서, 번호, 경로). 번호/경로는 제출 파일 작성 등에 쓰인다


# [6] ---------------------------------------------------------------------------------
# NetVLAD 는 각 지역 특징(local descriptor)을 K 개 클러스터 중심에 "부드럽게" 배정한다.
# 이 함수는 학습 전에 그 중심(centroids)과 배정 층(assign)을 k-means 결과로 맞춰 놓는다.
# 처음 읽을 때는 "학습이 잘 시작되도록 초기값을 잡아주는 함수" 정도로만 이해하고 넘어가도 된다.
@torch.no_grad()
def init_centroids(model, image_paths, transform, device, n_images=500, per_image=100, iters=30, seed=0):
    """
    (Provided) k-means initialization of NetVLAD (NetVLAD paper, Sec. 3.1).
    Clusters conv5 descriptors of `n_images` training images and sets
        model.centroids = c_k,   model.assign: w_k = 2*alpha*c_k,  b_k = -alpha*||c_k||^2
    so that the soft-assignment starts as a soft k-means. Without it, randomly initialized
    centroids make all descriptors almost identical and training collapses.
    """
    # (a) 학습 DB 이미지 중 500장을 무작위로 고른다 (seed 고정 -> 매번 같은 결과)
    rng = np.random.default_rng(seed)
    paths = list(rng.choice(image_paths, min(n_images, len(image_paths)), replace=False))
    loader = DataLoader(ImageListDataset(paths, "", transform), batch_size=32, num_workers=0)
    g = torch.Generator(device=device).manual_seed(seed)

    # (b) VGG16 backbone 으로 지역 특징 추출. 224x224 입력이면 feature map 은 (B, 512, 14, 14).
    #     14x14=196 개 위치 각각이 512차원 지역 특징 하나다. 이미지마다 그중 100개만 무작위로 뽑는다.
    model.eval()
    feats = []
    for imgs, _, _ in tqdm(loader, desc="k-means init"):
        # 채널 방향 L2 정규화 후 (B, D, h, w) -> (B, h*w, D): "위치별 특징 벡터 목록"으로 모양 변경
        f = F.normalize(model.backbone(imgs.to(device)), p=2, dim=1).flatten(2).transpose(1, 2)  # (B, HW, D)
        idx = torch.randint(0, f.shape[1], (f.shape[0], per_image), device=device, generator=g)  # 이미지마다 위치 100개 번호
        # gather 로 그 위치의 특징만 꺼내 (B*100, D) 로 펼친다
        feats.append(torch.gather(f, 1, idx[..., None].expand(-1, -1, f.shape[2])).reshape(-1, f.shape[2]))
    x = torch.cat(feats)    # (약 50000, 512): k-means 에 넣을 전체 특징

    # (c) 평범한 k-means: 무작위 K 개로 시작해 "가장 가까운 중심에 배정 -> 중심을 평균으로 갱신"을 30번 반복
    K = model.num_clusters
    c = x[torch.randperm(len(x), device=device, generator=g)[:K]].clone()   # (K, D) 초기 중심
    for _ in range(iters):
        assign = torch.cdist(x, c).argmin(1)          # 각 특징이 가장 가까운 중심 번호
        for k in range(K):
            if (assign == k).any():                   # 비어 있는 클러스터는 그대로 둔다
                c[k] = x[assign == k].mean(0)

    # (d) alpha(배정의 "날카로움") 결정
    #     soft-assignment 는 softmax(-alpha * ||x - c_k||^2). 가장 가까운 중심과 두 번째 중심의
    #     거리제곱 차이를 Δ 라 하면 두 배정 확률의 비는 exp(alpha * Δ).
    #     alpha = ln(100) / 평균 Δ  로 두면, 평균적으로 1등 중심이 2등보다 100배 강하게 배정된다.
    d2 = torch.cdist(x, c).pow(2).topk(2, dim=1, largest=False).values   # (N, 2) 1등/2등 중심까지 거리제곱
    alpha = -math.log(0.01) / (d2[:, 1] - d2[:, 0]).mean().item()

    # (e) 모델 파라미터에 복사
    #     -alpha*||x - c_k||^2 = 2*alpha*c_k·x - alpha*||c_k||^2 - alpha*||x||^2
    #     마지막 항은 모든 k 에 공통이라 softmax 에서 사라진다. 따라서 1x1 conv (w·x + b) 로
    #     w_k = 2*alpha*c_k, b_k = -alpha*||c_k||^2 를 넣으면 soft k-means 와 똑같이 동작한다.
    model.centroids.data.copy_(c)
    model.assign.weight.data.copy_((2.0 * alpha * c)[..., None, None])   # conv 가중치 모양 (K, D, 1, 1) 로 맞춤
    model.assign.bias.data.copy_(-alpha * c.pow(2).sum(1))
    print(f"NetVLAD k-means init: {len(x)} descriptors, {K} clusters, alpha={alpha:.1f}")


# [1] ---------------------------------------------------------------------------------
# 프로그램 시작점. 전체 파이프라인을 순서대로 실행한다. 여기서부터 읽기 시작할 것.
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DATA_ROOT = {DATA_ROOT} | OUTPUT_DIR = {OUTPUT_DIR} | device = {device}")

    # 모든 이미지에 공통으로 적용하는 전처리: 224x224 로 크기 통일 -> 텐서 -> ImageNet 평균/표준편차로 정규화
    # (VGG16 이 ImageNet 으로 사전학습되었기 때문에 같은 정규화를 써야 한다)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]) #전처리를 담당하는 인스턴스라고 보면 됨

    # split("train" / "val") 에 해당하는 파일 경로 4개를 LoadDataset 인자 이름에 맞춰 dict 로 묶는다
    def split_files(split):
        return dict(
            query_file=os.path.join(DATA_ROOT, split, f"query_{split}.txt"),   # 쿼리 이미지 목록 경로
            db_file=os.path.join(DATA_ROOT, split, f"index_{split}.txt"),      # DB 이미지 목록 경로
            root_dir=DATA_ROOT,
            gt_npz=os.path.join(DATA_ROOT, split, "gt", f"pitts30k_{split}.npz"),  # 각 이미지의 UTM 좌표
        )

    # 학습용/검증용 데이터셋 (-> [2] LoadDataset 참고). 이 시점에는 경로와 좌표만 읽는다.
    train_set = LoadDataset(**split_files("train"), transform=transform)
    val_set = LoadDataset(**split_files("val"), transform=transform)

    # 배치 단위로 (q, P, N) 을 꺼내주는 로더. 실제 이미지 로딩은 반복할 때 __getitem__ 에서 일어난다.
    # 주의: mine() 이 먼저 호출되어야 반복 가능 -> train.py 가 매 epoch 시작 때 호출한다.
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    #train set에서 batch size만큼 shuffle해서 가져오고 PyTorch가 지원하는 worker(프로세스)를 4개 호출해서 데이터를 로딩한다.
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4)
    #val set은 shuffle하지 않고 순서대로 가져온다. (평가할 때는 순서가 중요하므로)

    # -------------------------------
    # TASK1 : VGG16-based NetVLAD Structure
    # -------------------------------
    # 모델 구조는 model.py 에서 구현. 만든 직후 k-means 로 클러스터 중심을 초기화 (-> [6] init_centroids)
    model = NetVLADModel(num_clusters=16).to(device) #구현 완
    init_centroids(model, train_set.db_images, transform, device)   # (Provided)


    # -------------------------------
    # TASK2 : Learning with Ranking Loss
    # -------------------------------
    # train.py 에서 구현. 매 epoch: mine() -> (q, P, N) 배치로 ranking loss 학습 -> validate()
    train(model, train_loader, val_loader, device, epochs=5)

    # 학습된 가중치 저장
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)


    # -------------------------------
    # TASK3 : Evaluation with Recall@K
    # -------------------------------
    # 평가는 (q, P, N) 튜플이 필요 없고 이미지 한 장씩 descriptor 만 뽑으면 되므로 ImageListDataset 사용.
    # val_set 에서는 경로 목록과 정답 좌표(utmQ, utmDb, posDistThr)만 빌려 쓴다.
    # (a) Validation: GT is public, so you can compute Recall@K yourself.
    query_ds = ImageListDataset(val_set.query_images, "", transform)
    db_ds = ImageListDataset(val_set.db_images, "", transform)
    evaluate_recall(model, query_ds, db_ds, val_set.utmQ, val_set.utmDb,
                    val_set.posDistThr, device, recall_values=[1,5,10])

    # (b) Test: GT is hidden. Submit the top-10 retrievals to Kaggle.
    # test 는 정답 좌표가 없으므로 LoadDataset 을 만들 수 없다 -> .txt 목록에서 바로 ImageListDataset 생성
    query_ds = ImageListDataset(os.path.join(DATA_ROOT, "test", "query_test.txt"), DATA_ROOT, transform)
    db_ds = ImageListDataset(os.path.join(DATA_ROOT, "test", "index_test.txt"), DATA_ROOT, transform)
    generate_submission(model, query_ds, db_ds, device, os.path.join(OUTPUT_DIR, "submission.csv"), k=10)


if __name__ == "__main__":
    main()
