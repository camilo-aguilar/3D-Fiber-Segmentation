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

    true_1_hot = true_1_hot.type(logits.type())
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


def embedded_geometric_loss(outputs, labels, mini_v):
    delta_v = 0.2
    delta_d = 5

    alpha = 2
    beta = 2
    gamma = 0.0000001
    GPU_YES = torch.cuda.is_available()
    device = torch.device("cuda" if GPU_YES else "cpu")

    N_embedded = outputs.shape[1]
    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, N_embedded))

    labels = torch.unique(labeled_pixels, sorted=True)
    N_objects = len(labels)

    mu_vector = torch.zeros(N_objects, N_embedded).to(device)

    # Find mu vector
    Lv = 0  # Lv_same_object_proximity
    Lr = 0  # Ld_minimal_distance_clusters
    Ld = 0  # Lr_cluster_center_regularizer
    for c in range(N_objects):
        fiber_id = labels[c]
        idx_c = (labeled_pixels == fiber_id).nonzero()
        coordinates = (mini_v == fiber_id).nonzero().float()
        # xi vector
        x_i = torch.gather(object_pixels, 0, idx_c.repeat(1, N_embedded))
        # get mu
        mu = x_i.mean(0)
        mu_vector[c, :] = mu

        # get Lv
        Nc = len(idx_c)
        weights = r_individual(coordinates)
        if (1 in torch.isnan(weights)):
            weights = torch.ones(Nc, device=coordinates.device)
        else:
            weights = 1 / (1 + torch.exp(- weights))

        lv_term = torch.norm(mu_vector[c, :] - x_i, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = lv_term**2

        lv_term = lv_term * weights
        lv_term = torch.sum(lv_term, dim=0)
        Lv += (lv_term / Nc)
        Lr += torch.norm(mu, 2)

    for ca in range(N_objects):
        for cb in range(ca + 1, N_objects):
            ld_term = delta_d - torch.norm(mu_vector[ca, :] - mu_vector[cb, :], 2)
            ld_term = torch.clamp(ld_term, 0, 10000000)
            ld_term = ld_term ** 2
            Ld += ld_term

    N_objects = float(N_objects)
    Lv /= N_objects
    Lr /= N_objects
    Ld /= (N_objects * (N_objects - 1))

    loss = alpha * Lv + beta * Ld + gamma * Lr
    return loss


def embedded_geometric_loss(outputs, labels, mini_v):
    delta_v = 0.2
    delta_d = 5

    alpha = 2
    beta = 2
    gamma = 0.0000001
    GPU_YES = torch.cuda.is_available()
    device = torch.device("cuda" if GPU_YES else "cpu")

    N_embedded = outputs.shape[1]
    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, N_embedded))

    labels = torch.unique(labeled_pixels, sorted=True)
    N_objects = len(labels)

    mu_vector = torch.zeros(N_objects, N_embedded).to(device)

    # Find mu vector
    Lv = 0  # Lv_same_object_proximity
    Lr = 0  # Ld_minimal_distance_clusters
    Ld = 0  # Lr_cluster_center_regularizer
    for c in range(N_objects):
        fiber_id = labels[c]
        idx_c = (labeled_pixels == fiber_id).nonzero()
        coordinates = (mini_v == fiber_id).nonzero().float()
        # xi vector
        x_i = torch.gather(object_pixels, 0, idx_c.repeat(1, N_embedded))
        # get mu
        mu = x_i.mean(0)
        mu_vector[c, :] = mu

        # get Lv
        Nc = len(idx_c)
        weights = r_individual(coordinates)
        if (1 in torch.isnan(weights)):
            weights = torch.ones(Nc, device=coordinates.device)
        else:
            weights = 1 / (1 + torch.exp(- weights))

        lv_term = torch.norm(mu_vector[c, :] - x_i, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = lv_term**2

        lv_term = lv_term * weights
        lv_term = torch.sum(lv_term, dim=0)
        Lv += (lv_term / Nc)
        Lr += torch.norm(mu, 2)

    for ca in range(N_objects):
        for cb in range(ca + 1, N_objects):
            ld_term = delta_d - torch.norm(mu_vector[ca, :] - mu_vector[cb, :], 2)
            ld_term = torch.clamp(ld_term, 0, 10000000)
            ld_term = ld_term ** 2
            Ld += ld_term

    N_objects = float(N_objects)
    Lv /= N_objects
    Lr /= N_objects
    Ld /= (N_objects * (N_objects - 1))

    loss = alpha * Lv + beta * Ld + gamma * Lr
    return loss


def embedded_directional_loss(network_outputs, labels):
    delta_v = 0.2
    delta_d = 5

    alpha = 2
    beta = 2
    gamma = 0.0000001


    GPU_YES = torch.cuda.is_available()
    device = torch.device("cuda:0" if GPU_YES else "cpu")

    outputs = network_outputs[0]
    directions = network_outputs[1]

    N_embedded = outputs.shape[1]
    N_directions = directions.shape[1]
    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, N_embedded))
    direction_pixels = torch.gather(outputs, 0, idx_array.repeat(1, N_directions))

    labels = torch.unique(labeled_pixels, sorted=True)
    N_objects = len(labels)

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

        lv_term = torch.norm(mu_vector[c, :] - x_i, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = lv_term**2
        lv_term = torch.sum(lv_term, dim=0)
        Lv += (lv_term / Nc)
        Lr += torch.norm(mu, 2)

    for ca in range(N_objects):
        for cb in range(ca + 1, N_objects):
            ld_term = delta_d - torch.norm(mu_vector[ca, :] - mu_vector[cb, :], 2)
            ld_term = torch.clamp(ld_term, 0, 10000000)
            ld_term = ld_term ** 2
            Ld += ld_term

    N_objects = float(N_objects)
    Lv /= N_objects
    Lr /= N_objects
    Ld /= (N_objects * (N_objects - 1))

    loss = alpha * Lv + beta * Ld + gamma * Lr
    return loss

def distance_loss(network_outputs, labels):
    labels = labels.contiguous().view(-1)
    network_outputs = network_outputs.contiguous().view(-1)

    idx_array = (labels).nonzero().split(1, dim=1)



    object_pixels = network_outputs[idx_array].double()
    label_pixels = labels[idx_array].double()
    # idx at fibers
    # Get only the non-zero indexes
    loss = torch.norm(label_pixels - object_pixels, p=2)
    return loss

def direction_loss(network_outputs, labels, device=None):
    idx_array = (labels[:, 0] + labels[:, 1] + labels[:, 2]).nonzero()

    object_pixels = torch.gather(network_outputs, 0, idx_array.repeat(1,  3))
    label_pixels = torch.gather(labels, 0, idx_array.repeat(1, 3))
    # idx at fibers
    # Get only the non-zero indexes
    loss = torch.norm(label_pixels - object_pixels, p=1)
    return loss



def refine_connected(labels):
    device = labels.device
    labels = labels.cpu().numpy()
    labels_t = np.zeros(labels.shape)
    num_labels = labels.max().astype(np.int)
    counter = 2

    for c in range(2, num_labels + 1):
        im = labels == c
        temp_labels, temp_nums = measure.label(im, return_num=True)
        for points in range(1, temp_nums + 1):
            idx_c = np.where(temp_labels == points)
            if(len(idx_c[0]) < 30):
                labels_t[idx_c] = 1
            else:
                labels_t[idx_c] = counter
                counter = counter + 1
    labels_t[np.where(labels == 1)] = 1
    labels = torch.from_numpy(labels_t).long().to(device)
    return (labels)

def refine_watershed_end_points(labels):
    num_labels = labels.max().astype(np.int)
    end_points = []
    label_points = []
    for c in range(2, num_labels + 1):
        im = labels == c
        temp_labels, temp_nums = measure.label(im, return_num=True)
        for end_point_idx in range(1, temp_nums + 1):
            idx_c = np.where(temp_labels == end_point_idx)
            mean =[X.mean() for X in idx_c]
            end_points.append(mean)
            label_points.append(c)
    return (end_points, label_points)


def refine_watershed(labels, segmentation=None):
    device = labels.device
    labels = labels.cpu().numpy()

    markers = np.copy(labels)
    markers[np.where(labels == 1)] = 0

    energy = np.zeros(labels.shape)
    energy[np.where(labels==1)] = 1
    distance = ndi.distance_transform_edt(energy)
    distance[np.where(labels > 1)] = 1

    mask = np.zeros(labels.shape)
    mask[np.where(labels > 0)] = 1
    labels = watershed(-distance, markers, mask=mask)

    labels= torch.from_numpy(labels).long().to(device)

    return labels


def save_data(embeddings, labels, iteration=None, detected_labels=None):
    N_embedded = embeddings.shape[1]
    # num_dims = N_embedded
    # num_display_dims = 2
    # tsne_lr = 20.0

    # Get only the non-zero indexes
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(embeddings, 0, idx_array.repeat(1, N_embedded))

    from sklearn.manifold import TSNE
    # from tsne import tsne
    X = object_pixels.cpu().detach().numpy()
    Y = TSNE(n_components=2, perplexity=40, learning_rate=50).fit_transform(X)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(Y[:, 0], Y[:, 1], 5, labeled_pixels, cmap='tab20b')
    if iteration is None:
        iteration = 0
    plt.savefig("low_dim_embeeding/embedded_%d.png" % iteration)
    plt.close(fig)


    if detected_labels is not None:
        detected_labels_pixels = detected_labels[idx_tuple].squeeze(1)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        pylab.scatter(Y[:, 0], Y[:, 1], 5, detected_labels_pixels, cmap='tab20b')
        pylab.savefig("low_dim_embeeding/embedded_%d.png" % (iteration + 1))
        pylab.close(fig)

def get_single_fiber_property(space_labels, fiber_id):
    idx = (space_labels == fiber_id).nonzero().float()
     
    center = idx.mean(0)
    rs0 = torch.norm(idx - center, p=2, dim=1)

    # Find farthest distance from center
    end_point0_idx = (rs0 == rs0.max()).nonzero()
        
    # Find points close to EP1
    end_point0_idx = end_point0_idx[0, 0]
    idx_split = idx.split(1, dim=1)

    end_point0 = torch.tensor([idx_split[0][end_point0_idx], idx_split[1][end_point0_idx], idx_split[2][end_point0_idx]], device=idx.device)
        
    # Find closes points from end point 0
    rs1 = torch.norm(idx - end_point0, p=2, dim=1)
    end_point1_idx = (rs1 < 3).nonzero()
    end_point1_idx = end_point1_idx[:, 0]
    end_point1 = torch.tensor([idx_split[i][end_point1_idx][:, 0].mean() for i in range(3)])
          
    # Find farthest point from end point 1
    rs2 = torch.norm(idx - end_point0, p=2, dim=1)
    # Find farthest point from end point 1
    end_point2_idx = (rs2 > rs2.max() - 3).nonzero()
    end_point2_idx = end_point2_idx[:, 0]
    end_point2 = torch.tensor([idx_split[i][end_point2_idx][:, 0].mean() for i in range(3)])

    c_np = center.cpu().numpy()

    length = torch.norm(end_point1 - end_point2, p=2).cpu().item()
    direction = (end_point1 - end_point2)
    direction = direction / torch.norm(direction, p=2)

    R = 1.5 # rr[1]

    direction = direction.cpu().numpy()
    return  [fiber_id, c_np[0], c_np[1], c_np[2], R, length, direction[0], direction[1], direction[2]]

def get_fiber_properties(space_labels, large_volume=False):
    end_points = []
    fiber_ids = []
    centers = {}
    fiber_list = {}
    # id center1, center2, center3, L, R, Ty, Tz, error
    for fiber_id in torch.unique(space_labels):
        if(fiber_id.cpu().item() == 0 or fiber_id.cpu().item() == 1):
            continue
        idx = (space_labels == fiber_id).nonzero().float()
        if(large_volume is True):
            if(len(idx) < 10):
                space_labels[idx.long().split(1, dim=1)] = 0
                continue
        # idx is shape [N, 3]

        center = idx.mean(0)
        rs0 = torch.norm(idx - center, p=2, dim=1)

        # Find farthest distance from center
        end_point0_idx = (rs0 == rs0.max()).nonzero()
        
        # Find points close to EP1
        end_point0_idx = end_point0_idx[0, 0]
        idx_split = idx.split(1, dim=1)

        end_point0 = torch.tensor([idx_split[0][end_point0_idx], idx_split[1][end_point0_idx], idx_split[2][end_point0_idx]], device=idx.device)
        
        # Find closes points from end point 0
        rs1 = torch.norm(idx - end_point0, p=2, dim=1)
        end_point1_idx = (rs1 < 3).nonzero()
        end_point1_idx = end_point1_idx[:, 0]
        end_point1 = torch.tensor([idx_split[i][end_point1_idx][:, 0].mean() for i in range(3)])
         #, idx_split[1][end_point1_idx], idx_split[2][end_point1_idx]], device=idx.device)
     
        # Find farthest point from end point 1
        rs2 = torch.norm(idx - end_point0, p=2, dim=1)
        # Find farthest point from end point 1
        end_point2_idx = (rs2 > rs2.max() - 3).nonzero()
        end_point2_idx = end_point2_idx[:, 0]
        end_point2 = torch.tensor([idx_split[i][end_point2_idx][:, 0].mean() for i in range(3)])


        '''
        close_to_end_point1 = (rs2 < 3).nonzero().long()
        close_to_end_point1 = close_to_end_point1[:, 0]
        end_points_image[idx_split[0][close_to_end_point1].long(), idx_split[1][close_to_end_point1].long(), idx_split[2][close_to_end_point1].long()] = fiber_id
        close_to_end_point2 = (rs2 > rs2.max() - 3).nonzero().long()
        close_to_end_point2 = close_to_end_point2[:, 0]
        end_points_image[idx_split[0][close_to_end_point2].long(), idx_split[1][close_to_end_point2].long(), idx_split[2][close_to_end_point2].long()] = fiber_id
        '''
        end_points.append(end_point1.cpu().numpy())
        end_points.append(end_point2.cpu().numpy())

        fiber_ids.append(fiber_id.cpu().item())
        fiber_ids.append(fiber_id.cpu().item())

        c_np = center.cpu().numpy()
        centers[fiber_id.cpu().item()] = c_np

        length = torch.norm(end_point1 - end_point2, p=2).cpu().item()
        direction = (end_point1 - end_point2)
        direction = direction / torch.norm(direction, p=2)

        '''
        if(math.isnan(direction)):
            idx = (space_labels == fiber_id).nonzero().split(1, dim=1)
            space_labels[idx] = 1
        '''
       #  rr = fit_t.r2(direction.unsqueeze(1), idx, center.unsqueeze(1))
        R = 1.5 # rr[1]
        # G = fit_t.G(direction.unsqueeze(1), idx)

        direction = direction.cpu().numpy()
        fiber_list[fiber_id.cpu().item()] = [fiber_id.cpu().item(), c_np[0], c_np[1], c_np[2], R, length, direction[0], direction[1], direction[2]]

    return centers, fiber_ids, end_points, fiber_list


def evaluate_iou(Vf, Vgt):
    Vf[np.where(Vf == 1)] = 0
    # Vf[np.where(Vgt == 0)] = 0
    labels_gt = np.unique(Vgt)
    num_fibers = len(labels_gt) - 1

    labels_f = np.unique(Vf)
    num_fibers_f = len(labels_f) - 1

    set_f = set(labels_f)
    set_gt = set(labels_gt)

    set_f.remove(0)
    set_gt.remove(0)

    Vf = Vf[0:150, 0:150, 0:150]
    print("Num Fibers Gt: {}".format(num_fibers))
    print("Labels Vf:{}".format(num_fibers_f))

    fibers_corrected_detected = 0
    fibers_splitted_but_detected = 0

    fibers_in_v_detected_double = 0
    flag_match_detected = 0
    for Lgt in set_gt:

        Vgt_temp = np.zeros(Vgt.shape)
        idxs_gt = np.where(Vgt == Lgt)
        Vgt_temp[idxs_gt] = 1

        labels_in_V = set(np.unique(Vf[idxs_gt]))
        labels_in_V = labels_in_V.intersection(set_f)

        IOU = 0.0
        total_intersection = 0
        set_broken_fibers = set()
        for Lf in labels_in_V:
            Vf_temp = np.zeros(Vgt.shape)
            idxs_f = np.where(Vf == Lf)
            Vf_temp[idxs_f] = 1

            #num_detected_gt_fibers = len(np.unique(Vgt[idxs_f]))
            #if(num_detected_gt_fibers > 2):
            #    fibers_in_v_detected_double += 1

            intersection = np.logical_and(Vgt_temp, Vf_temp).sum().astype(np.float)
            union = np.logical_or(Vgt_temp, Vf_temp).sum().astype(np.float)
            area = (Vgt_temp).sum().astype(np.float)

            ind_IOU = intersection / union

            # print(area)
            if(ind_IOU > 0.5):
                flag_match_detected = 1
                fibers_corrected_detected += 1
                set_f.remove(Lf)
                IOU = ind_IOU
                # print("IOU", IOU)
                break
            else:
                total_intersection += intersection
                set_broken_fibers.add(Lf)

        if(not flag_match_detected):
            print("Cheking again")
            IOU = total_intersection / union
            if(IOU > 0.5):
                fibers_splitted_but_detected += 1
                for item in set_broken_fibers:
                    set_f.remove(item)
    print("")
    print("Total Fibers: {}, Fibers Detected and Splitted {}, Fibers Correctly Detected {}".format(num_fibers, fibers_splitted_but_detected, fibers_corrected_detected))
    print("Percent of Fibers Correctly Detected: {}".format(float(fibers_corrected_detected) / float(num_fibers)))
    print("Percent of Fibers Detected ans splitted: {}".format(float(fibers_splitted_but_detected) / float(num_fibers)))

    print("")
    print("Fibers Missed: {}".format(num_fibers - fibers_corrected_detected - fibers_splitted_but_detected))
    print("Fibers in V that detected double {}".format(fibers_in_v_detected_double))


def merge_inner_fibers(end_points, fiber_list, fiber_ids, space_labels, debug=0, radius=10, angle_threshold=10):
    neigh = NearestNeighbors(n_neighbors=4, radius=2).fit(end_points)
    A = neigh.kneighbors_graph(end_points, mode='distance')        # A = A.toarray()
    sure_neighbors = []
    lost_ids_dict = {}
    mini_fibers = {}
    A = find_sparse(A)
    fiber_merged_dict = defaultdict(list)



    if(debug):
        f = open("instances/debug/post_merged/merging_fibers.txt", "w")

    for i in range(len(A[0])):
        i_entry = A[0][i]
        j_entry = A[1][i]
        distance = A[2][i]
        nan_flag = 0
        nan_flag_a = 0
        nan_flag_b = 0

        if(i_entry == j_entry):
            continue
        if(fiber_ids[i_entry] == fiber_ids[j_entry] or distance > radius):
            continue

        if( ((fiber_ids[j_entry], fiber_ids[i_entry]) in sure_neighbors) or ((fiber_ids[i_entry], fiber_ids[j_entry]) in sure_neighbors)):
            continue

        fiber_a = fiber_list[fiber_ids[i_entry]]
        fiber_b = fiber_list[fiber_ids[j_entry]]

        dir_a = np.array([fiber_a[6], fiber_a[7], fiber_a[8]])
        dir_b = np.array([fiber_b[6], fiber_b[7], fiber_b[8]])
        

        center_a = np.array([fiber_a[1], fiber_a[2], fiber_a[3]])
        center_b = np.array([fiber_b[1], fiber_b[2], fiber_b[3]])
      
        # Vector between centers 
        vector_between_centers = center_a - center_b
        vector_between_centers = vector_between_centers / np.sqrt(np.dot(vector_between_centers, vector_between_centers))
 
        angle_between = np.arccos(np.abs(np.dot(dir_a, dir_b))) * 180 / np.pi
        if(math.isnan(angle_between)):
            angle_between = 0
            nan_flag = 1

        angle_between_a_center_v = np.arccos(np.abs(np.dot(dir_a, vector_between_centers))) * 180 / np.pi
        if(math.isnan(angle_between_a_center_v)):
            angle_between_a_center_v = 0
            nan_flag_a = 1

        angle_between_b_center_v = np.arccos(np.abs(np.dot(dir_b, vector_between_centers))) * 180 / np.pi

        if(math.isnan(angle_between_b_center_v)):
            angle_between_b_center_v = 0
            nan_flag_b = 1

        angle_total = max(angle_between_a_center_v, angle_between_b_center_v)


        if(debug):

            f.write("Considering {} {}\n".format(fiber_ids[i_entry], fiber_ids[j_entry]))

            Txya = np.arctan2(dir_a[1], dir_a[0]) * 180 / np.pi
            if(Txya < 0):
                Txya = 180 + Txya
            Tza = np.arccos(np.dot(dir_a, np.array([0, 0, 1])) / np.linalg.norm(dir_a, 2)) * 180 / np.pi

            Txyb = np.arctan2(dir_b[1], dir_b[0]) * 180 / np.pi
            if(Txyb < 0):
                Txyb = 180 + Txyb
            Tzb = np.arccos(np.dot(dir_b, np.array([0, 0, 1])) / np.linalg.norm(dir_b, 2)) * 180 / np.pi


            Txyc = np.arctan2(vector_between_centers[1], vector_between_centers[0]) * 180 / np.pi
            if(Txyc < 0):
                Txyc = 180 + Txyc
            Tzc = np.arccos(np.dot(vector_between_centers, np.array([0, 0, 1])) / np.linalg.norm(vector_between_centers, 2)) * 180 / np.pi

            f.write("angle_between {}\n".format(angle_between))
            f.write("angle_between_a_center_v {}\n".format(angle_between_a_center_v))
            f.write("angle_between_b_center_v {}\n".format(angle_between_b_center_v))
            f.write("Center a: {}, {}, {}\n".format(center_a[0], center_a[1], center_a[2]))
            f.write("Center b: {}, {}, {}\n".format(center_b[0], center_b[1], center_b[2]))

            f.write("Txy_a, Tz_a: {}, {}\n".format(Txya, Tza))
            f.write("Txy_b, Tz_b: {}, {}\n".format(Txyb, Tzb))
            f.write("Txy_c, Tz_c: {}, {}\n".format(Txyc, Tzc))


        if(angle_total < 5 and angle_between < 10 and nan_flag == 0):
            if((fiber_ids[j_entry], fiber_ids[i_entry]) not in sure_neighbors and (fiber_ids[i_entry], fiber_ids[j_entry]) not in sure_neighbors):
                sure_neighbors.append((fiber_ids[j_entry], fiber_ids[i_entry]))
                fiber_merged_dict[fiber_ids[j_entry]].append(fiber_ids[i_entry])
        
        # fibers that are too small to get an angle:
        elif(angle_total < angle_threshold and nan_flag == 1):
            # Consider fiber a is a very very small
            if(nan_flag_a == 1 and nan_flag_b == 0):
                # If fiber a has not been seen yet
                if(fiber_ids[i_entry] not in mini_fibers.keys()):
                    mini_fibers[fiber_ids[i_entry]] = (fiber_ids[j_entry], distance)
                else:
                # Check if the new candidate is better for fiber_a
                    if distance < mini_fibers[fiber_ids[i_entry]][1]:
                        mini_fibers[fiber_ids[i_entry]] = (fiber_ids[j_entry], distance)

            # Consider fiber b is a very very small
            elif(nan_flag_b == 1 and nan_flag_a == 0):
                # If fiber b has not been seen yet
                if(fiber_ids[j_entry] not in mini_fibers.keys()):
                    mini_fibers[fiber_ids[j_entry]] = (fiber_ids[i_entry], distance)
                else:
                # Check if the new candidate is better for fiber_b
                    if distance < mini_fibers[fiber_ids[j_entry]][1]:
                        mini_fibers[fiber_ids[j_entry]] = (fiber_ids[i_entry], distance)

            #if both fibers are very very small
            else:
                if(fiber_ids[i_entry] not in mini_fibers.keys()):
                    mini_fibers[fiber_ids[i_entry]] = (fiber_ids[j_entry], 10000)
                
                if(fiber_ids[j_entry] not in mini_fibers.keys()):
                    mini_fibers[fiber_ids[j_entry]] = (fiber_ids[i_entry], 10000)


    for (fa, fb) in sure_neighbors:
        if(debug):
            f.write("Merging {} {}\n".format(fa, fb))

        while(fb in lost_ids_dict.keys()):
            #print(lost_ids_dict[fb])
            fb = lost_ids_dict[fb]
            #print("Saved one lost id")

        while(fa in lost_ids_dict.keys()):
            # print(lost_ids_dict[fa])
            fa = lost_ids_dict[fa]
            # print("Saved another lost id")

        if(fa != fb):
            lost_ids_dict[fa] = fb
        else:
            continue

        idx1 = (space_labels == fa).nonzero()
        if(len(idx1) == 0):
            continue
        idx1 = idx1.split(1, dim=1)
        space_labels[idx1] = fb

        new_entry = get_single_fiber_property(space_labels, fb)
        fiber_list[fb] = new_entry
        del fiber_list[fa]



    # Take Care of Very Small Fibers
    for fa in mini_fibers.keys():
        '''
        fb = mini_fibers[fa][0]
        dist = mini_fibers[fa][1]
        while(fb in lost_ids_dict.keys()):
            fb = lost_ids_dict[fb]
        
        if(debug):
            f.write("Merging Very Small {} {} {}\n".format(fa, fb, dist ))

        if(fa != fb):
            lost_ids_dict[fa] = fb
        '''
        idx1 = (space_labels == fa).nonzero()
        if(len(idx1) == 0):
            continue
        idx1 = idx1.split(1, dim=1)
        space_labels[idx1] = 0

    if(debug):
        f.close()
    return len(lost_ids_dict)


def merge_outer_fibers(end_points, centers, fiber_ids, space_labels, debug=0, radius=10):
    neigh = NearestNeighbors(n_neighbors=8, radius=radius).fit(end_points)
    A = neigh.kneighbors_graph(end_points, mode='distance')        # A = A.toarray()
    possible_neighbors = set()
    sure_neighbors = []
    lost_ids_dict = {}

    A = find_sparse(A)
    fiber_merged_dict = defaultdict(list)

    device = space_labels.device
    space_labels = space_labels.cpu().numpy()

    for i in range(len(A[0])):
        i_entry = A[0][i]
        j_entry = A[1][i]

        if(i_entry == j_entry):
            continue
        if(fiber_ids[i_entry] == fiber_ids[j_entry] or A[2][i] > radius):
            continue

        # [L, C_fit[0], C_fit[1], C_fit[2], r_fit, h_fit, Txy, Tz, fit_err]
        # properties1 = guess_cylinder_parameters_merged(fiber_ids[i_entry], -1, space_labels)
        #properties2 = guess_cylinder_parameters_merged(fiber_ids[j_entry], -1, space_labels)

        properties_together = guess_cylinder_parameters_merged(fiber_ids[i_entry], fiber_ids[j_entry], space_labels)
        err = properties_together[-1]
        if(err < 100):
            if((fiber_ids[j_entry], fiber_ids[i_entry]) not in possible_neighbors and (fiber_ids[i_entry], fiber_ids[j_entry]) not in possible_neighbors):
                sure_neighbors.append((fiber_ids[j_entry], fiber_ids[i_entry]))
                fiber_merged_dict[fiber_ids[j_entry]].append(fiber_ids[i_entry])

    space_labels = torch.from_numpy(space_labels).to(device)

    for (fa, fb) in sure_neighbors:
        if(0):
            print("Merging {} {}".format(fa, fb))

        while(fb in lost_ids_dict.keys()):
            #print(lost_ids_dict[fb])
            fb = lost_ids_dict[fb]
            #print("Saved one lost id")

        while(fa in lost_ids_dict.keys()):
            # print(lost_ids_dict[fa])
            fa = lost_ids_dict[fa]
            # print("Saved another lost id")

        if(fa != fb):
            lost_ids_dict[fa] = fb

        idx1 = (space_labels == fa).nonzero()
        if(len(idx1) == 0):
            continue
        idx1 = idx1.split(1, dim=1)
        space_labels[idx1] = fb

    return len(sure_neighbors)


# Regression to instance center
def embedded_geometric_loss_coords(outputs, labels):
    delta_v = 2
    device = outputs.device
    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)

    # copy 3D
    labels3D = labels[0, 0, ...].clone()

    # Flatten Everything adn make outputs 3 dimensions at the end
    labels = labels.contiguous().view(-1)
    outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, 3))
    Loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]

        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        # Get output vectors at those dimensions
        o_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 3))

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        
        # Get Center Pixel
        center_pixel = coordinates.mean(0)

        # Get offset vector
        o_hat = coordinates - center_pixel
        
        Nc = len(idx_c)

        # Regression 
        lv_term = torch.norm(o_i - o_hat, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        Loss += (lv_term / Nc)

    return Loss


def embedded_geometric_loss_coords22(outputs, labels):
    delta_v = 2
    device = outputs.device
    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)

    # copy 3D
    labels3D = labels[0, 0, ...].clone()

    # Flatten Everything adn make outputs 3 dimensions at the end
    labels = labels.contiguous().view(-1)
    outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    mu_vector = torch.zeros(N_objects, N_embedded).to(device)
    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, 3))
    Loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]

        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        # Get output vectors at those dimensions
        o_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 3))

        # xi vector
        mu = o_i.mean(0)
        mu_vector[c, :] = mu

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        
        # Get Center Pixel
        center_pixel = coordinates.mean(0)

        # Get offset vector
        o_hat = coordinates - mu
        
        Nc = len(idx_c)

        # Regression 
        lv_term = torch.norm(o_i - o_hat, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        Loss += (lv_term / Nc)

    return Loss

def embedded_geometric_loss_r(outputs, labels):
    delta_v = 2
    device = outputs.device
    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)

    # copy 3D
    labels3D = labels[0, 0, ...].clone()

    centers, fiber_ids, end_points, fiber_list = get_fiber_properties(labels3D)
    # Flatten Everything adn make outputs 3 dimensions at the end
    labels = labels.contiguous().view(-1)
    outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, 3))
    Loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]

        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        # Get output vectors at those dimensions
        o_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 3))

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        
        # Get Center Pixel
        center_pixel = coordinates.mean(0)

        # Get offset vector
        o_hat = coordinates - center_pixel

        if(fiber_id == 1):
            continue
        properties = fiber_list[fiber_id.cpu().item()]
        W = torch.tensor(np.array([properties[6], properties[7], properties[8]]).astype(np.float)).to(device)
        W = W.unsqueeze(1).cpu().numpy()
        P = np.identity(3) - np.dot(np.reshape(W, (3, 1)), np.reshape(W, (1, 3)))
        
        P = torch.from_numpy(P).float().to(device)
        o_hat = torch.mm(P, o_hat.t())
        o_hat = o_hat.t()

        Nc = len(idx_c)

        # Regression 
        lv_term = torch.norm(o_i - o_hat, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        Loss += (lv_term / Nc)

    return Loss

def embedded_geometric_loss_r(outputs, labels):
    delta_v = 2
    device = outputs.device
    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)

    # copy 3D
    labels3D = labels[0, 0, ...].clone()

    centers, fiber_ids, end_points, fiber_list = get_fiber_properties(labels3D)
    # Flatten Everything adn make outputs 3 dimensions at the end
    labels = labels.contiguous().view(-1)
    outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, 3))
    Loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]

        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        # Get output vectors at those dimensions
        o_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 3))

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        
        # Get Center Pixel
        center_pixel = coordinates.mean(0)

        # Get offset vector
        o_hat = coordinates - center_pixel

        if(fiber_id == 1):
            continue
        properties = fiber_list[fiber_id.cpu().item()]
        W = torch.tensor(np.array([properties[6], properties[7], properties[8]]).astype(np.float)).to(device)
        W = W.unsqueeze(1).cpu().numpy()
        P = np.identity(3) - np.dot(np.reshape(W, (3, 1)), np.reshape(W, (1, 3)))
        
        P = torch.from_numpy(P).float().to(device)
        o_hat = torch.mm(P, o_hat.t())
        o_hat = o_hat.t()

        Nc = len(idx_c)

        # Regression 
        lv_term = torch.norm(o_i - o_hat, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        Loss += (lv_term / Nc)

    return Loss


def projection_matrix(w):
    I_m = torch.eye(3)
    device = w.device
    I_m = I_m.to(device)
    mult = torch.mm(w, w.t())
    return I_m - mult

# Regression to instance center
def embedded_geometric_loss_radious(outputs, labels):
    delta_v = 0
    device = outputs.device
    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)

    # copy 3D
    labels3D = labels[0, 0, ...].clone()

    # Flatten Everything adn make outputs 3 dimensions at the end
    labels = labels.contiguous().view(-1)
    outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, 1)

    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, 1))
    Loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]

        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        # Get output vectors at those dimensions
        r_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 1))

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        
        # Get Radious
        if(len(coordinates) < 30):
            continue
        radii = r_individual(coordinates.clone())
        raddii = radii.detach()
        
        Nc = len(idx_c)

        # Regression 
        lv_term = torch.norm(r_i - radii, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        Loss += (lv_term / Nc)

    return Loss
if __name__ == '__main__':
    Vol = tensors_io.read_volume_h5('final_fibers_single','final_fibers_single','../h5_files')
    Vol = torch.from_numpy(Vol).unsqueeze(0)
    Vol = Vol.unsqueeze(0)

    [_, _, rows, cols, slices] = Vol.shape
    embedded_geometric_loss_coords(torch.zeros(1, 3, rows, cols, slices), Vol)