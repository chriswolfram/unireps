import torch

def embs_knn(emb_mat, k):
    """
    Computes the k-nearest neighbors for each embedding in the given matrix using the dot product as the distance metric.
    Args:
        emb_mat (torch.Tensor): A 2D tensor where each row represents an embedding.
        k (int): The number of nearest neighbors to find for each embedding.
    Returns:
        torch.Tensor: A 2D tensor of shape (n, k) where n is the number of embeddings. Each row contains the indices of the k-nearest neighbors for the corresponding embedding.
    """

    gram = emb_mat @ emb_mat.T
    gram.fill_diagonal_(-torch.inf)

    return gram.topk(k, sorted=False).indices

def mutual_knn(knn_mat_1, knn_mat_2):
    """
    Computes the mutual k-nearest neighbors between two k-nearest neighbor matrices.

    Args:
        knn_mat_1 (torch.Tensor): A 2D tensor where each row contains the indices of the k-nearest neighbors for the first set of embeddings.
        knn_mat_2 (torch.Tensor): A 2D tensor where each row contains the indices of the k-nearest neighbors for the second set of embeddings.

    Returns:
        float: The proportion of mutual k-nearest neighbors between the two matrices.
    """

    assert knn_mat_1.shape == knn_mat_2.shape
    n = knn_mat_1.shape[0]
    k = knn_mat_1.shape[1]

    total = 0
    for i in range(n):
        total += torch.isin(knn_mat_1[i], knn_mat_2[i], assume_unique=True).sum()
    
    return (total / (k * n)).item()

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