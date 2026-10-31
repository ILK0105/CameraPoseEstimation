# PA1 NetVLAD 구현 가이드 (RTX 4090 서버용)

> 읽는 사람: 서버에서 구현하고 실행할 Claude Code 또는 학생 본인.
> 마감: 2026-10-31 23:59 KST. 시간이 부족하면 필수 단계(1~4)를 먼저 끝내고 제출합니다.

## 0. 먼저 알아둘 점

- 과제 명세와 규칙은 `README (1).md`가 기준입니다. 이 문서와 다르면 README를 따릅니다.
- **이 문서의 수치는 전부 실행 전 가설입니다.** 시간, 메모리, 성능 향상 폭은 측정한 적이 없습니다.
- Kaggle 1등은 보장할 수 없습니다. 목표는 규칙 안에서 val mRecall을 최대로 올리는 것입니다.
- 코드 스타일은 스타터 코드에 맞춥니다(간결한 코드, 적은 주석).

### 현재 저장소 상태

| 파일 | 상태 |
|---|---|
| `model.py` | TASK 1 구현 완료. 38번째 줄의 `pass`는 지워도 됨. 실행 검증은 아직 안 함 |
| `train.py` | `ranking_loss`, `validate`, `train` 미구현 |
| `eval.py` | `evaluate_recall` 미구현 |
| `test.py` | `estimate_poses` 미구현. `num_clusters=16`, `Resize((224,224))` 하드코딩 |
| `base.py` | 코드는 스타터와 동일. 한국어 설명 주석이 추가된 상태 |

### 규칙 요약

- 학습은 train/val만 사용합니다. test 이미지로 학습하거나 PCA를 학습하지 않습니다.
- TASK 1~3의 사전학습 가중치는 torchvision ImageNet VGG16만 허용됩니다.
- TASK 1~3의 라이브러리는 numpy, pillow, tqdm, torch, torchvision으로 제한합니다. cv2 등 외부 라이브러리는 TASK 4에서만 씁니다.
- NetVLAD 구조는 유지합니다. 클러스터 수, 이미지 크기, mining, augmentation은 바꿔도 됩니다.

## 1. 서버 준비

```bash
git clone https://github.com/ILK0105/CameraPoseEstimation.git
cd CameraPoseEstimation
pip install gdown numpy pillow tqdm scipy opencv-python
# torch, torchvision은 서버의 CUDA 버전에 맞게 설치
gdown 1pDErcDJLocm_T-kd_cpOZSk8ipX8Wyup -O PA1_data.zip
unzip -q PA1_data.zip    # 저장소 루트에서 풀면 ./data/train 등이 생김. config.py 수정 불필요
python -c "import torch;print(torch.cuda.get_device_name(0))"
```

- zip 내부 경로가 `data/...`이므로 반드시 저장소 루트에서 풉니다.
- `data/`와 `*.zip`, `*.pth`는 `.gitignore`에 있습니다. `git add -f`를 쓰지 않습니다.
- 학습은 `tmux`나 `nohup`으로 돌리고 로그를 파일로 남깁니다. 보고서에 필요합니다.
- `base.py`의 한국어 설명 주석을 제거합니다. 영어 docstring과 스타터의 주석은 남깁니다. 제거 전후의 AST가 같은지 확인합니다.

## 2. 필수 구현

### `train.py`

```python
def ranking_loss(q_emb, P_emb, N_emb, margin=0.1):
    d_pos = ((q_emb.unsqueeze(1) - P_emb) ** 2).sum(2).min(1)[0]
    d_neg = ((q_emb.unsqueeze(1) - N_emb) ** 2).sum(2)
    return torch.clamp(d_pos.unsqueeze(1) + margin - d_neg, min=0).sum(1).mean()
```

- `validate`: 제공된 `mine` 호출 뒤 `model.eval()`, `torch.no_grad()`로 평균 loss를 구해 반환합니다.
- `train`: 주석의 TODO대로 구현합니다. 옵티마이저는 README에 따라 Adam이고, `requires_grad`인 파라미터만 넘깁니다.
- 매 epoch의 체크포인트를 저장합니다.

### `eval.py`의 `evaluate_recall`

- descriptor를 코사인 유사도로 정렬합니다.
- positive는 UTM 거리 `posDistThr` 이내이고, 없으면 가장 가까운 DB 이미지로 대체합니다(docstring 기준).
- 함수 끝의 출력 코드와 `recalls` 딕셔너리(개수)를 그대로 유지합니다.
- `(R@1 + R@5 + R@10) / 3`도 함께 출력합니다. 이 값이 모델 선택 기준입니다.

### 스모크 테스트 (본 학습 전 필수)

1. `model(torch.randn(1,3,224,224))`의 shape이 `(1, K*512)`이고 norm이 약 1인지 확인합니다.
2. `ranking_loss`를 작은 텐서로 손계산과 비교합니다.
3. 쿼리 수십 장의 부분 집합으로 전체 파이프라인을 끝까지 돌립니다. `submission.csv` 형식이 `sample_submission.csv`와 같은지 확인합니다.

## 3. 학습에서 빠지기 쉬운 함정

- **`persistent_workers=True`를 쓰지 않습니다.** 워커는 데이터셋 복사본을 들고 있어서 `mine()`이 갱신한 `pos_idx`, `neg_idx`를 보지 못합니다.
- **모델 선택은 val loss가 아니라 val mRecall로 합니다.** `validate`는 매번 mining을 다시 하므로 epoch 간 loss를 비교할 수 없습니다. 스타터는 마지막 가중치만 저장하므로 best 저장으로 바꿉니다.
- **K와 해상도를 한곳에서 관리합니다.** `base.py`와 `test.py`에 각각 하드코딩되어 있어 어긋나면 체크포인트 로드나 검색 품질이 깨집니다. `config.py`에 `NUM_CLUSTERS`, `IMG_SIZE`를 두고 양쪽에서 import합니다.
- `compute_embeddings`는 `num_workers`가 기본값 0이라 `mine()`이 느립니다. `num_workers`와 `pin_memory`를 추가합니다. 반환값이 바뀌지 않게 합니다.
- `mine()`은 전체 descriptor를 GPU에 올립니다. K=64에서는 32768차원이므로 메모리를 확인합니다.
- `Resize((224,224))`는 640x480 원본의 비율을 왜곡합니다. 해상도 실험을 먼저 합니다.

## 4. Kaggle 성능 전략 (TASK 2~3)

기준선을 만든 뒤 한 번에 하나씩 바꾸고 val mRecall이 오를 때만 채택합니다. val 쿼리가 612장이므로 쿼리 1장이 약 0.16%입니다. 0.5% 미만의 차이는 노이즈로 봅니다.

| 순서 | 변경 | 시작 값 | 근거 |
|---|---|---|---|
| 1 | 기준선 | K=16, 224, lr 1e-4, epoch 5 | 스타터 설정. 시간 측정용 |
| 2 | 해상도 | `Resize((480,640))` | 비율 유지, conv5 위치 수 증가 |
| 3 | 클러스터 수 | K=64 | 논문 설정 |
| 4 | 층 고정 | `backbone[:24]` 고정 (conv5만 학습) | 논문 방식, 메모리 절감 |
| 5 | AMP | `torch.autocast` + `GradScaler` | 메모리와 속도 |
| 6 | epoch, lr | epoch 15~30, lr {1e-4, 1e-5}, 5 epoch마다 절반 | 논문의 학습 길이 참고 |
| 7 | mining | `n_neg=10`, `n_sample`을 음성 전체로 | DB가 5126장뿐이라 전수 탐색 가능 |
| 8 | 배치 | OOM이 나지 않는 범위에서 증가 | 샘플당 이미지가 `1+n_pos+n_neg`장 |

### 학습 후 기법 (순서대로 시도)

1. **PCA whitening:** train DB descriptor로만 학습하고 4096차원으로 줄인 뒤 L2 정규화합니다. N이 D보다 작으므로 Gram 행렬의 고유값 분해를 씁니다.
2. **멀티스케일 평균:** 2~3개 해상도의 descriptor를 평균하고 L2 정규화합니다.
3. **모델 앙상블:** 시드나 K가 다른 체크포인트의 유사도를 평균합니다.
4. **train+val 재학습:** 하이퍼파라미터를 확정한 뒤 train과 val을 합쳐 같은 epoch 수로 학습합니다. 지역이 분리되어 있으므로 두 파일의 목록과 UTM을 이어 붙이면 됩니다. val로 검증할 수 없으므로 기존 모델의 제출 파일을 먼저 보관합니다.
5. **지역 특징 재정렬:** 상위 100장을 conv5 지역 특징의 상호 최근접 매칭 수로 재정렬합니다. torch만 사용합니다. 구현 비용이 크므로 시간이 남을 때만 합니다.

### 제출 전 확인이 필요한 기법

query expansion과 database-side augmentation은 test descriptor를 추론 때 사용합니다. 학습은 아니지만 규칙의 경계에 있으므로 쓰려면 TA에게 먼저 확인하고 보고서에 밝힙니다.

### 코드 수정 시 주의

- `generate_submission`의 출력 형식은 바꾸지 않습니다. 후처리가 필요하면 descriptor 계산 부분만 바꿉니다.
- Kaggle의 하루 제출 횟수를 먼저 확인합니다. README에 따르면 리더보드 점수가 최종 점수입니다.
- Kaggle 팀 이름을 학번으로 설정합니다.

## 5. 카메라 자세 추정 (TASK 4)

평가는 평균 오차이므로 **큰 실패 몇 개가 점수를 좌우합니다.** 정밀도보다 실패 방지를 우선합니다.

### 데이터와 검증 방법

- dev는 4개 장면, 쿼리 24장입니다. test는 4개 장면, DB 30장, 쿼리 40장입니다. 깊이와 내부 파라미터 파일은 없습니다.
- DB가 장면당 최대 30장이므로 NetVLAD 상위 K 대신 전체 DB와 매칭해도 비용이 작습니다.
- **dev 쿼리만으로 튜닝하면 과적합되기 쉽습니다.** DB 이미지를 하나씩 빼고 쿼리로 쓰는 leave-one-out 평가를 함께 합니다. 포즈가 제공되어 있으므로 dev와 test 장면 모두 가능합니다. 다만 test 쿼리의 포즈로 쓰는 것이 아니라 파이프라인 검증에만 씁니다.

### 좌표계 (구현 전 반드시 검증)

- TartanAir의 공개 내부 파라미터는 `fx=fy=320, cx=320, cy=240`입니다.
- 카메라 프레임은 x 전방, y 오른쪽, z 아래로 알려져 있습니다. OpenCV 광학 프레임과 다르므로 변환이 필요합니다.
  ```python
  R_bo = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=float)   # optical -> body
  R_w_optical = R_w_body @ R_bo
  ```
- **먼저 dev의 DB-DB 쌍으로 검증합니다.** 매칭으로 구한 상대 자세를 GT 상대 자세와 비교해서 회전 오차가 작은지 봅니다. 맞지 않으면 다른 축 배치를 시험합니다.
- 쿼터니언은 `x y z w` 순서입니다. `scipy.spatial.transform.Rotation.from_quat`의 순서와 같습니다.

### 단계별 구현

단계마다 `python test.py dev`와 leave-one-out으로 평가하고, 나아진 경우에만 채택합니다.

1. **기준선:** NetVLAD 상위 1장의 자세를 복사합니다. 항상 동작하는 최종 대체 방법입니다.
2. **가중 평균:** 상위 K장의 위치와 회전을 매칭 점수로 가중 평균합니다. 회전은 `Rotation.mean(weights=...)`를 씁니다.
3. **매칭 기반 선택:** SIFT와 ratio test, `cv2.findEssentialMat(..., cv2.USAC_MAGSAC)`로 전체 DB를 검증하고 inlier 수로 순위를 정합니다.
4. **상대 자세 기반 추정:** inlier가 충분한 DB 여러 장에서 회전을 구해 평균합니다. 위치는 각 DB에서 쿼리로 향하는 광선의 최소제곱 교점으로 구합니다. 2장 이상의 DB가 필요합니다. 스케일은 DB 사이의 기준선으로 정해집니다.
5. **삼각측량 + PnP:** DB 쌍의 매칭을 알려진 포즈로 삼각측량한 뒤 `cv2.solvePnPRansac`을 적용합니다. 재투영 오차가 작고 inlier가 많을 때만 채택합니다.
6. **학습 기반 매처 (선택):** SuperPoint+LightGlue나 LoFTR로 3~5단계의 매칭을 교체합니다. 넓은 시점 차이나 질감이 적은 장면에서 도움이 될 수 있습니다. 라이브러리와 가중치 라이선스를 확인합니다.

### 실패 방지

- 결과가 DB 포즈의 범위에서 크게 벗어나거나, inlier가 임계값 미만이거나, 추정값들이 서로 일치하지 않으면 한 단계 아래 방법으로 내려갑니다.
- 임계값은 소수만 두고 leave-one-out으로 정합니다.
- 새로 쓰는 라이브러리는 `test.py` 위쪽의 `Newly defined libraries should be listed here` 위치에 적습니다.
- `test.py`의 `Resize`와 `num_clusters`를 학습 설정과 맞춥니다.

## 6. 시간이 부족할 때의 우선순위

1. **필수:** 2장 구현과 스모크 테스트.
2. **필수:** 해상도, K=64, 층 고정으로 1회 학습한 뒤 Kaggle에 제출합니다.
3. **필수:** TASK 4를 1~3단계까지 구현하고 `pose_submission.csv`를 만듭니다.
4. **필수:** 보고서를 작성합니다. 실험 로그와 Kaggle 점수 스크린샷을 씁니다.
5. **여유 시:** 4장의 학습 후 기법과 5장의 4~6단계를 진행합니다. 각 기법은 "Own idea" 항목에 쓸 수 있습니다.

첫 epoch의 소요 시간을 측정한 뒤 남은 시간에 맞춰 epoch 수를 정합니다. 마감 최소 2시간 전에는 유효한 제출본을 만들어 둡니다.

## 7. 제출 체크리스트

- [ ] Kaggle: `output/submission.csv` 업로드, 팀 이름은 학번
- [ ] `python test.py test`로 `pose_submission.csv` 생성 (40행과 헤더)
- [ ] LearnUS: `<학번>.zip`에 `base.py, config.py, model.py, train.py, eval.py, test.py, pose_submission.csv, report.pdf`
- [ ] `report.pdf`는 3쪽 미만: 방법, 결과(Kaggle 점수 스크린샷 포함), 분석
- [ ] 수업의 AI 사용 규정에 따라 사용 사실 표기가 필요한지 확인
- [ ] 제출한 코드로 `base.py`, `test.py`가 실행되는지 확인 (K, 해상도, 체크포인트 일치)
