import torch.nn.functional as F
import torch.nn as nn
import torch
# from .jacard_loss import lovasz_hinge_flat


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
    num_classes = true.max().item() + 1
    true_masks = true.contiguous().view(-1)
    segmentation_output = nn.functional.softmax(logits, dim=1)
    masks_seg_probs = segmentation_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
    if(weights is None):
        criterion = nn.CrossEntropyLoss().to(true.device)
    else:
        criterion = nn.CrossEntropyLoss(weights).to(true.device)
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


def coordinate_loss(outputs, labels, t_params):
    delta_v = 1
    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)

    # copy 3D labels
    labels3D = labels[0, 0, ...].clone()

    # Flatten Everything and make offset_vectors 3 dimensions at the end
    labels = labels.contiguous().view(-1)
    offset_vectors = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

    # idx at fibers
    idx_array = labels.nonzero()
    if(len(idx_array) > 0):
        idx_tuple = idx_array.split(1, dim=1)
    else:
        return None

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)

    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)
    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(offset_vectors, 0, idx_array.repeat(1, 3))
    offset_loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]
        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        Nc = len(idx_c)
        # Get output vectors at those dimensions
        o_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 3))

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        # Get Center Pixel
        center_pixel = coordinates.mean(0)
        # Get offset vector
        o_hat = coordinates - center_pixel

        # Regression
        lv_vector = torch.norm(o_i - o_hat, p=2, dim=1)
        lv_term = lv_vector - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        offset_loss += (lv_term / Nc)
    return offset_loss


# #################################################
# Multitask Loss
def multi_task_loss(outputs, labels, t_params):
    segmentation = outputs[0][:, 0:2, ...]
    sigma0_output = outputs[0][:, 2, ...].unsqueeze(0)
    offset_vectors = outputs[1][:, 0:3, ...]
    sigma1_output = outputs[1][:, 3, ...].unsqueeze(0)
    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]

    N_objects = len(labels_ids)
    # copy 3D labels
    labels3D = labels[0, 0, ...].clone()

    # Flatten Everything and make offset_vectors 3 dimensions at the end
    labels = labels.contiguous().view(-1)

    offset_vectors = offset_vectors.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

    # idx at fibers
    idx_array = labels.nonzero()
    if(len(idx_array) > 0):
        idx_tuple = idx_array.split(1, dim=1)
    else:
        return [None, None, None]
    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(offset_vectors, 0, idx_array.repeat(1, 3))
    offset_loss = 0

    for c in range(N_objects):
        fiber_id = labels_ids[c]
        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        Nc = len(idx_c)
        # Get output vectors at those dimensions
        o_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 3))

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        # Get Center Pixel
        center_pixel = coordinates.mean(0)
        # Get offset vector
        o_hat = coordinates - center_pixel

        # Regression
        lv_vector = torch.norm(o_i - o_hat, p=2, dim=1)
        lv_term = lv_vector - t_params.delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        offset_loss += (lv_term / Nc)

    # ############# Sigma Vector Loss ##############
    sigma0 = sigma0_output.mean()
    sigma1_output = sigma1_output.contiguous().view(-1)
    sigma1_pixels = sigma1_output[idx_tuple].squeeze(1)
    sigma1 = sigma1_pixels.mean()

    segmentation_loss = cross_entropy((labels > 0).long(), segmentation)

    Total_Loss = torch.exp(-sigma1) * offset_loss + torch.exp(-sigma0) * segmentation_loss + 1 / 2 * (sigma1 + sigma0)

    return [Total_Loss, segmentation_loss, offset_loss]


# #################################################
# Multitask Loss
def multi_task_loss_learned_center(outputs, labels, t_params):
    segmentation = outputs[0][:, 0:2, ...]
    sigma0_output = outputs[0][:, 2, ...].unsqueeze(0)
    offset_vectors = outputs[1][:, 0:3, ...]
    sigma1_output = outputs[1][:, 3, ...].unsqueeze(0)

    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)

    # copy 3D labels
    labels3D = labels[0, 0, ...].clone()
    segmentation_loss = cross_entropy((labels > 0).long(), segmentation)

    # Flatten Everything and make offset_vectors 3 dimensions at the end
    labels = labels.contiguous().view(-1)

    offset_vectors = offset_vectors.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

    # idx at fibers
    idx_array = labels.nonzero()
    if(len(idx_array) > 0):
        idx_tuple = idx_array.split(1, dim=1)
    else:
        return [None, None, None]

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)

    object_pixels = torch.gather(offset_vectors, 0, idx_array.repeat(1, 3))
    offset_loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]
        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        Nc = len(idx_c)
        # Get output vectors at those dimensions
        o_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 3))

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        # Get Center Pixel
        center_pixel = o_i.mean(0)  # coordinates.mean(0)
        # Get offset vector
        o_hat = coordinates - center_pixel

        # Regression
        lv_vector = torch.norm(o_i - o_hat, p=2, dim=1)
        lv_term = lv_vector - t_params.delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        offset_loss += (lv_term / Nc)

    # ############# Sigma Vector Loss ##############
    sigma0 = sigma0_output.mean()
    sigma1_output = sigma1_output.contiguous().view(-1)
    sigma1_pixels = sigma1_output[idx_tuple].squeeze(1)
    sigma1 = sigma1_pixels.mean()

    Total_Loss = torch.exp(-sigma1) * offset_loss + torch.exp(-sigma0) * segmentation_loss + 1 / 2 * (sigma1 + sigma0)

    return [Total_Loss, segmentation_loss, offset_loss]


# ##################################################
# Bandwidh Loss
def joint_spatial_bandwidth_loss(outputs, labels, t_params):
    device = labels.device
    seed_branch = outputs[0]
    tan_ac = torch.nn.Tanh()
    offset_vectors = tan_ac(outputs[1][:, 0:3, ...])
    sigma_output = outputs[1][:, 3, ...].unsqueeze(0)
    sigma_output = torch.exp(-sigma_output)

    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)

    # copy 3D labels
    labels3D = labels[0, 0, ...].clone()

    # Flatten Everything and make offset_vectors 3 dimensions at the end
    labels = labels.contiguous().view(-1)

    N_pixels = len(labels)
    sigma_output = sigma_output.contiguous().view(-1)
    offset_vectors = offset_vectors.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

    # idx at fibers
    idx_array = labels.nonzero()
    if(len(idx_array) > 0):
        idx_tuple = idx_array.split(1, dim=1)
    else:
        return [None, None, None]

    probs = torch.zeros(labels.shape).to(device)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    sigma_pixels = sigma_output[idx_tuple].squeeze(1)
    probs_results = torch.zeros_like(labeled_pixels).float()
    object_pixels = torch.gather(offset_vectors, 0, idx_array.repeat(1, 3))
    centroid_loss = 0
    sigma_loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]
        # Get indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        Nc = len(idx_c)
        # Get output vectors at those dimensions
        o_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 3))

        # Get coordinates of objects
        x_i = (labels3D == fiber_id).nonzero().float()

        # Get offset vector
        e_i = x_i + o_i

        # xi vector
        mu = e_i.mean(0)

        # ############# Sigma ##############
        # Regression of sigma
        sigma_i = sigma_pixels[idx_c[:, 0]]
        sigma_k = torch.sum(sigma_i) / Nc
        l_smooth = torch.norm(sigma_i - sigma_k, p=2)
        sigma_loss += torch.sum(l_smooth, dim=0) / Nc

        # ############ Phi ################
        # Exponential becomes a probability
        phi_k = torch.exp(- torch.norm(e_i - mu, p=2, dim=1)**2 / (2 * sigma_k))
        probs_results[idx_c[:, 0]] = phi_k

        # Make a mask of object
        instance_binary_mask = torch.zeros(labels.shape).to(device)
        instance_predicted_mask = torch.zeros(labels.shape).to(device)

        # Fill the binary and predicted masks
        instance_binary_mask[idx_array[idx_c.split(1, dim=1)]] = 1
        instance_predicted_mask[idx_array[idx_c.split(1, dim=1)]] = phi_k.unsqueeze(1).unsqueeze(1)
        instance_predicted_mask = instance_predicted_mask.unsqueeze(0)
        instance_predicted_mask = torch.cat((1 - instance_predicted_mask, instance_predicted_mask), 0)

        # ############# Phi Loss ##############
        # centroid_loss += lovasz_hinge_flat(instance_predicted_mask.contiguous().view(-1), instance_binary_mask.contiguous().view(-1))
        centroid_loss += cross_entropy(instance_predicted_mask.contiguous().view(-1), instance_binary_mask.contiguous().view(-1))

    # ############# Seed Loss ##############
    # Estiamte phi directly
    probs[idx_array[:, 0]] = probs_results
    seed_branch = seed_branch.contiguous().view(-1)
    seed_loss = torch.sum(torch.norm(seed_branch - probs, p=2) ** 2) / N_pixels

    # ############# Instance Vector Loss ##############
    Total_Loss = centroid_loss + seed_loss + sigma_loss
    return [Total_Loss, seed_loss, centroid_loss]
