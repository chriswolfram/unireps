import torch

def embs_knn(emb_mat, k):

    if len(emb_mat.shape) == 3:
        return torch.stack([embs_knn(emb, k) for emb in emb_mat])

    gram = emb_mat @ emb_mat.T
    gram.fill_diagonal_(-torch.inf)

    return gram.topk(k, sorted=False).indices

# def _mutual_knn_tensor(knn_mat_1, knn_mat_2):
#     if len(knn_mat_1.shape) == 3:
#         return torch.stack([_mutual_knn_tensor(knn_1, knn_mat_2) for knn_1 in knn_mat_1])
    
#     if len(knn_mat_2.shape) == 3:
#         return torch.stack([_mutual_knn_tensor(knn_mat_1, knn_2) for knn_2 in knn_mat_2])

#     assert knn_mat_1.shape == knn_mat_2.shape
#     n = knn_mat_1.shape[0]
#     k = knn_mat_1.shape[1]

#     # This is a more explicit version that is unfortunately much slower:
#     # total = 0
#     # for i in range(n):
#     #     total += torch.isin(knn_mat_1[i], knn_mat_2[i], assume_unique=True).sum()

#     m1 = torch.zeros(n, n)
#     m2 = torch.zeros(n, n)

#     r = torch.arange(n).unsqueeze(1)
#     m1[r, knn_mat_1] = 1
#     m2[r, knn_mat_2] = 1

#     total = (m1 * m2).sum()
    
#     return (total / (k * n))

# def mutual_knn(knn_mat_1, knn_mat_2):
#     out = _mutual_knn_tensor(knn_mat_1, knn_mat_2)
#     if out.dim() == 0:
#         return out.item()
#     else:
#         return out

def knn_mask(knn):
    n = knn.shape[0]
    k = knn.shape[1]
    m = torch.zeros(n, n)
    r = torch.arange(n).unsqueeze(1)
    m[r, knn] = 1

    return m

def _mutual_knn_tensor(knn_mat_1, knn_mat_2):
    n = knn_mat_1.shape[-2]
    k = knn_mat_1.shape[-1]

    if len(knn_mat_1.shape) == 3 and len(knn_mat_2.shape) == 3:
        knn_masks_1 = torch.stack([knn_mask(knn) for knn in knn_mat_1])
        knn_masks_2 = torch.stack([knn_mask(knn) for knn in knn_mat_2])
        return torch.stack([(knn_1 * knn_masks_2).sum(dim=1).sum(dim=1) / n / k for knn_1 in knn_masks_1])
    elif len(knn_mat_1.shape) == 3:
        knn_masks_1 = torch.stack([knn_mask(knn) for knn in knn_mat_1])
        knn_masks_2 = knn_mask(knn_mat_2)
        return (knn_masks_1 * knn_masks_2).sum(dim=1).sum(dim=1) / n / k
    elif len(knn_mat_2.shape) == 3:
        knn_masks_1 = knn_mask(knn_mat_1)
        knn_masks_2 = torch.stack([knn_mask(knn) for knn in knn_mat_2])
        return (knn_masks_1 * knn_masks_2).sum(dim=1).sum(dim=1) / n / k
    else:
        return (knn_mask(knn_mat_1) * knn_mask(knn_mat_2)).sum() / n / k

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