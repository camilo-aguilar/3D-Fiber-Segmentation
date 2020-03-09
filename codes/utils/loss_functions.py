import torch.nn.functional as F
import torch.nn as nn
import torch


# Semantic Segmentation

def dice_loss(true, logits, eps=1e-7):
    """.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W, Z].
        logits: a tensor of shape [B, C, H, W, Z]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
    """
    num_classes = true.max().item() + 1
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()
        probas = F.softmax(logits, dim=1)

    true_1_hot = true_1_hot.type(logits.type()).to(true.device)
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def cross_entropy(true, logits, weights=None):
    num_classes = true.max().item()
    true_masks = true.contiguous().view(-1)
    segmentation_output = nn.functional.softmax(logits, dim=1)
    masks_seg_probs = segmentation_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
    criterion = nn.CrossEntropyLoss().to(true.device)
    s_loss = criterion(masks_seg_probs, true_masks)
    return s_loss


# Instance Segmentation
def embedded_loss(outputs, labels, t_params):
    outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, t_params.n_embeddings)
    labels = labels.contiguous().view(-1)
    device = outputs.device

    N_embedded = outputs.shape[1]

    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, N_embedded))

    labels = torch.unique(labeled_pixels, sorted=True)
    N_objects = len(labels)
    if(N_objects < 2):
        return None
    mu_vector = torch.zeros(N_objects, N_embedded).to(device)

    # Find mu vector
    Lv = 0  # Lv_same_object_proximity
    Lr = 0  # Ld_minimal_distance_clusters
    Ld = 0  # Lr_cluster_center_regularizer
    for c in range(N_objects):
        fiber_id = labels[c]
        idx_c = (labeled_pixels == fiber_id).nonzero()

        # xi vector
        x_i = torch.gather(object_pixels, 0, idx_c.repeat(1, N_embedded))
        # get mu
        mu = x_i.mean(0)
        mu_vector[c, :] = mu

        # get Lv
        Nc = len(idx_c)

        lv_term = torch.norm(mu_vector[c, :] - x_i, p=2, dim=1) - t_params.delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = lv_term**2
        lv_term = torch.sum(lv_term, dim=0)
        Lv += (lv_term / Nc)
        Lr += torch.norm(mu, 2)

    for ca in range(N_objects):
        for cb in range(ca + 1, N_objects):
            ld_term = t_params.delta_d - torch.norm(mu_vector[ca, :] - mu_vector[cb, :], 2)
            ld_term = torch.clamp(ld_term, 0, 10000000)
            ld_term = ld_term ** 2
            Ld += ld_term

    N_objects = float(N_objects)
    Lv /= N_objects
    Lr /= N_objects
    Ld /= (N_objects * (N_objects - 1))

    loss = t_params.alpha_i * Lv + t_params.beta_i * Ld + t_params.gamma_i * Lr
    return loss
