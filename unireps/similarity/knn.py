import torch

def embs_knn(emb_mat, k):

    if len(emb_mat.shape) == 3:
        return torch.stack([embs_knn(emb, k) for emb in emb_mat])

    gram = emb_mat @ emb_mat.T
    gram.fill_diagonal_(-torch.inf)

    return gram.topk(k, sorted=False).indices

def _mutual_knn_tensor(knn_mat_1, knn_mat_2):
    if len(knn_mat_1.shape) == 3:
        return torch.stack([_mutual_knn_tensor(knn_1, knn_mat_2) for knn_1 in knn_mat_1])
    
    if len(knn_mat_2.shape) == 3:
        return torch.stack([_mutual_knn_tensor(knn_mat_1, knn_2) for knn_2 in knn_mat_2])

    assert knn_mat_1.shape == knn_mat_2.shape
    n = knn_mat_1.shape[0]
    k = knn_mat_1.shape[1]

    total = 0
    for i in range(n):
        total += torch.isin(knn_mat_1[i], knn_mat_2[i], assume_unique=True).sum()
    
    return (total / (k * n))

def mutual_knn(knn_mat_1, knn_mat_2):
    out = _mutual_knn_tensor(knn_mat_1, knn_mat_2)
    if out.dim() == 0:
        return out.item()
    else:
        return out

def mutual_knn_baseline(n, k):
    """
    Calculate the mutual k-nearest neighbors baseline.

    This function returns the baseline probability of a mutual k-nearest neighbor
    in a dataset with `n` elements and `k` nearest neighbors.

    Parameters:
    n (int): The total number of elements in the dataset.
    k (int): The number of nearest neighbors to consider.

    Returns:
    float: The baseline probability of a mutual k-nearest neighbor.
    """
    return k / (n-1)