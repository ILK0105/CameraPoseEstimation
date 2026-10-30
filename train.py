import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def embed_tuples(model, q, P, N, device):
    """
    (Provided) One forward pass for a batch of training tuples.
        q (B,3,H,W), P (B,Np,3,H,W), N (B,Nn,3,H,W)
    Returns descriptors  q_emb (B,D), P_emb (B,Np,D), N_emb (B,Nn,D)
    """
    B, Np, Nn = q.shape[0], P.shape[1], N.shape[1]
    x = torch.cat([q, P.flatten(0, 1), N.flatten(0, 1)]).to(device)
    emb = model(x)
    return emb[:B], emb[B:B + B * Np].view(B, Np, -1), emb[B + B * Np:].view(B, Nn, -1)


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
