import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

#

# [읽는 순서] base.py의 main() -> train() -> embed_tuples() -> ranking_loss() -> validate()
# q: 기준 사진(query), P: 같은 장소의 사진들(positive), N: 다른 장소의 사진들(negative)
# B: 배치의 query 수, Np/Nn: query당 P/N 개수, D: 특징 벡터 길이, H/W: 이미지 높이/너비
# descriptor와 embedding은 여기서 모델이 사진 한 장을 요약한 특징 벡터를 뜻한다.
# 현재 embed_tuples만 구현되어 있다. 나머지 함수의 TODO는 구현 안내다.
# return ... / return total_loss / pass 부분은 아직 실제 손실 계산·검증·학습을 수행하지 않는다.

def embed_tuples(model, q, P, N, device):
    """
    (Provided) One forward pass for a batch of training tuples.
        q (B,3,H,W), P (B,Np,3,H,W), N (B,Nn,3,H,W)
    Returns descriptors  q_emb (B,D), P_emb (B,Np,D), N_emb (B,Nn,D)
    """
    # P/N의 (배치, 후보 수) 두 축을 flatten으로 합쳐 이미지 목록으로 만든다.
    # query -> positive -> negative 순으로 붙여 모델에 한 번에 넣는다.
    B, Np, Nn = q.shape[0], P.shape[1], N.shape[1]
    x = torch.cat([q, P.flatten(0, 1), N.flatten(0, 1)]).to(device)
    # x: (B + B*Np + B*Nn, 3, H, W). to(device)는 모델과 같은 CPU/GPU로 옮긴다.
    # 모델 출력 emb는 이미지마다 하나의 D차원 벡터를 갖는다.
    emb = model(x)
    # 붙인 순서대로 나누고 P/N을 query별 묶음으로 복원한다. -1은 D를 자동 계산한다.
    # 반환 크기: (B, D), (B, Np, D), (B, Nn, D).
    return emb[:B], emb[B:B + B * Np].view(B, Np, -1), emb[B + B * Np:].view(B, Nn, -1)


# [손실 함수] 가장 가까운 positive보다 negative가 margin만큼 더 멀도록 유도한다.
# 입력은 이미지가 아닌 특징 벡터이며, 목표 출력은 스칼라 텐서 하나다.
# q를 (B, 1, D)로 바꾸고 P/N과 빼면 각 후보와의 차이를 한 번에 계산할 수 있다.
# 차이를 제곱하고 D축으로 더하면 제곱 유클리드 거리: P는 (B, Np), N은 (B, Nn).
# P 거리의 최솟값 (B,)을 고른 뒤 (B, 1)로 바꿔 N 거리와 비교한다.
# 예: min_pos=0.2, neg=0.25, margin=0.1이면 손실은 max(0, 0.05)=0.05.
# 충분히 먼 negative의 손실은 0. negative축으로 더하고 배치축으로 평균낸다.
# 현재 return ...은 Ellipsis를 반환하는 자리 표시자이므로 구현이 필요하다.
def ranking_loss(q_emb, P_emb, N_emb, margin=0.1):
    """
    Your Task) Complete the NetVLAD ranking loss (lecture slides / NetVLAD paper):

        L = sum_j  l( min_i d^2(q, p_i) + m - d^2(q, n_j) ),     l(x) = max(0, x)

        d^2 : squared Euclidean distance between descriptors
        p_i : positives of q  (P_emb[:, i]),   n_j : negatives of q  (N_emb[:, j])

    Parameters:
        q_emb (B, D), P_emb (B, Np, D), N_emb (B, Nn, D)
    Returns:
        scalar Tensor: sum over negatives, averaged over the batch
    """
    # TODO:
    # 1. Squared distances to all positives, keep the minimum for each query  -> (B,)
    # 2. Squared distances to all negatives                                    -> (B, Nn)
    # 3. Hinge: max(0, min_pos + margin - neg), sum over negatives, mean over batch

    return ...


# [검증 함수] 가중치를 업데이트하지 않고 검증 데이터의 평균 ranking loss를 측정한다.
# mine은 현재 모델로 사용할 P/N 이미지 번호를 다시 고른다. loader 순회 전에 필요하다.
# model.eval()은 평가 모드로 바꾸고, torch.no_grad()는 gradient 기록을 끈다.
# 두 기능은 서로 다르므로 검증에서는 함께 사용한다.
# 각 배치에서 embed_tuples -> ranking_loss 순서로 계산한다.
# 전체 query 평균: 배치 평균 손실 * 실제 배치 크기를 누적한 뒤 전체 query 수로 나눈다.
# 이렇게 하면 마지막 배치가 작아도 각 query의 비중이 같다.
# 현재는 mine만 실행하고 0.0을 반환한다. 아래 TODO를 구현해야 검증값을 얻는다.
def validate(model, loader, device, margin=0.1):
    """
    Your Task) Complete the function in train.py.

    - Compute the average ranking loss (same loss as in training) over the validation set.

    Returns:
        float: Average validation loss.
    """
    loader.dataset.mine(model, device)   # (Provided) mine tuples with the current model
    total_loss = 0.0


    # TODO:
    # 1. Set the model to evaluation mode.
    # 2. Disable gradient computation (torch.no_grad).
    # 3. For each batch (q, P, N): embed_tuples(...), then ranking_loss(...).
    # 4. Accumulate the loss and return the average.

    return total_loss


# [학습 함수] base.py에서 준비한 모델과 학습/검증 loader를 받아 학습 전체를 진행한다.
# epochs: 전체 데이터 반복 횟수, lr: 학습률, margin: 손실 함수에 쓰는 거리 여유 값.
# Adam은 requires_grad=True인 파라미터로 만들며 epoch 반복문 밖에서 한 번 생성한다.
# 매 epoch: mine -> model.train() -> 배치별 학습 -> 평균 학습 손실 출력 -> validate.
# mine이 평가 모드로 바꾸므로 이후 model.train()으로 학습 모드로 되돌려야 한다.
# 배치별 순서:
#   1. optimizer.zero_grad(): 이전 배치의 gradient 제거. gradient는 기본적으로 누적된다.
#   2. embed_tuples -> ranking_loss: 특징 벡터와 손실 계산.
#   3. loss.backward(): 각 파라미터의 gradient 계산.
#   4. optimizer.step(): gradient를 이용해 실제 가중치 업데이트.
# 로그용 손실은 loss.item()으로 숫자를 꺼내고, 검증처럼 배치 크기로 가중 평균한다.
# 현재 본체는 pass이므로 아직 학습하지 않는다. 아래 TODO는 구현 순서 안내다.
def train(model, train_loader, val_loader, device, epochs=5, lr=1e-4, margin=0.1):
    """
    Your Task) Complete the function in train.py.

    - Implement the training loop with the ranking loss above.
    - Optimize the model using Adam optimizer.
    - After each epoch:
        * Print the average training loss.
        * Evaluate on the validation set using `validate`.
    """
    # TODO:
    # 1. Define an optim.Adam over the parameters that require grad.
    # 2. For each epoch:
    #    - (Provided) train_loader.dataset.mine(model, device)
    #          hard mining with the current model -- call it at the START of every epoch
    #    - Set model to train mode (mine() switches the model to eval mode).
    #    - Loop over training batches (q, P, N):
    #        * q_emb, P_emb, N_emb = embed_tuples(model, q, P, N, device)
    #        * loss = ranking_loss(q_emb, P_emb, N_emb, margin)
    #        * Backpropagate and update parameters.
    #        * Track the training loss.
    #    - Print average training loss.
    #    - Call validate(model, val_loader, device, margin) and print it.

    pass
