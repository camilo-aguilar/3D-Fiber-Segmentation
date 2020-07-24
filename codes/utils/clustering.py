try:
    from geometry import fit, point_line_distance, get_spatial_properties, point_line_distance_torch, get_ref_angle
    from mpp import Tube
    from rjmcmc import parameters as mpp_pam
    from rjmcmc import rj_mcmc
    import tensors_io
except:
    from .geometry import fit, point_line_distance, get_spatial_properties, point_line_distance_torch, get_ref_angle
    from .mpp import Tube
    from .rjmcmc import parameters as mpp_pam
    from .rjmcmc import rj_mcmc
    import codes.utils.tensors_io as tensors_io
import numpy as np
import torch

from sklearn.cluster import DBSCAN


def clustering_algorithm(offset_vectors, final_pred, mask=None, data_image=None, params_t=None):
    print("Starting Clustering Algo with parameter r: {}".format(params_t.eps_param))
    params_t.min_points = 10
    # params_t.eps_param = 2
    params_t.mpp_min_r = params_t.mpp_min_r
    params_t.mpp_max_r = params_t.mpp_min_r


    # Get offset magnitudes
    magnitudes = torch.norm(offset_vectors, dim=1)  # [N_foreground]
    magnitudes = magnitudes / magnitudes.max()
    _, initial_indexes0 = torch.sort(magnitudes)  # [N_foreground]

    labels_array = torch.zeros(magnitudes.shape, device=offset_vectors.device, dtype=torch.long) - 1
    soft_labels_array = torch.zeros(magnitudes.shape, device=offset_vectors.device, dtype=torch.long) - 1
    soft_distances = torch.zeros(magnitudes.shape, device=offset_vectors.device, dtype=torch.float) + 10000
    # Get coordinates of real pixels
    coordinates_pixels = (final_pred[0, ...] == 1).nonzero().float()  # [N_foreground, 3]

    conected_component_labels = DBSCAN(eps=2, min_samples=4).fit_predict(coordinates_pixels.cpu())
    conected_component_labels = torch.from_numpy(conected_component_labels).to(offset_vectors.device)

    # Get coordinates of pixels with offset
    offset_pixels = (coordinates_pixels - offset_vectors).long().detach().float()  # [N_foreground, 3]

    # Get Center Proposals
    counter_image, initial_indexes = get_center_pixels(offset_pixels, dims=params_t.cube_size)
    # Rescale magnitudes to make them probabilities
    embedded_volume_marks = torch.zeros(params_t.cube_size, params_t.cube_size, params_t.cube_size).detach().long()

    list_of_objects = []
    label = 1

    for pixel_idx in initial_indexes:
        off_index = offset_vectors[pixel_idx, :].long()
        if(labels_array[pixel_idx] > - 1):
            continue
        if(embedded_volume_marks[off_index[0], off_index[1], off_index[2]] > 0):
            labels_array[pixel_idx] = embedded_volume_marks[off_index[0], off_index[1], off_index[2]]
            continue

        neighbor_label, marks, absolute_fiber_indexes = propose_cluster4(pixel_idx, labels_array, offset_pixels, coordinates_pixels, embedded_volume_marks, list_of_objects, params_t)
        if(neighbor_label > -1):
            labels_array[pixel_idx] = neighbor_label
            continue

        if(marks is None):
            continue
        # label = expand_cluster3(marks, absolute_fiber_indexes, conected_component_labels, labels_offset, soft_distances, coordinates_pixels, offset_pixels, labels_array, soft_labels_array, embedded_volume_marks, label, params_t)
        label = expand_cluster4(marks, absolute_fiber_indexes, conected_component_labels, soft_distances, coordinates_pixels, labels_array, soft_labels_array, embedded_volume_marks, label, list_of_objects, params_t, data_image)

    # print("Refining")
    labels_array = refine_cluster4(list_of_objects, offset_pixels, coordinates_pixels, labels_array, params_t)
    # print("The number of non-classified pixel percentage is: {}".format(initial_indexes.shape[0] / (initial_indexes == -1).sum() * 100))

    #for el in list_of_objects:
    #    print(el.label)
    #    print(el.data_energy)
    #    print(el.prior_energy)
    #    print("")
    # labels_array = assign_unclassified_clusters(list_of_objects, initial_indexes, labels_array, coordinates_pixels)

    # for pixel_idx in initial_indexes:
    #    if(labels_array[pixel_idx] == - 1):
    #        labels_array[pixel_idx] = soft_distances[pixel_idx]

    '''
    DEBUG
    '''
    # labels_array = soft_labels_array
    if(True):
        from .evaluation import evaluate_iou_volume

        # Vectorize and make it numpy
        labels_offset = DBSCAN(eps=1.5, min_samples=4).fit_predict(offset_pixels.cpu())  # DBSCAN(eps=1, min_samples=1).fit_predict(offset_pixels.cpu().numpy())
        labels_offset = torch.from_numpy(labels_offset).to(offset_pixels.device)
        coordinates_pixels = coordinates_pixels.long()
        offset_pixels = offset_pixels.long().clamp(0, params_t.cube_size - 1)
        
        # Create Embedded Volume
        embedded_volume = torch.zeros(params_t.cube_size, params_t.cube_size, params_t.cube_size).detach()
        for pix in offset_pixels.long():
            embedded_volume[pix[0], pix[1], pix[2]] += 1.0
        embedded_volume = embedded_volume / (0.01 * embedded_volume.max())
        tensors_io.save_subvolume(embedded_volume.unsqueeze(0), "debug_cluster/points")
        embedded_volume[coordinates_pixels.split(1, dim=1)] = magnitudes.cpu().unsqueeze(1)

        # Create Real Image
        embedded_volume_labels = torch.zeros(params_t.cube_size, params_t.cube_size, params_t.cube_size).detach().long()
        embedded_volume_labels[coordinates_pixels.split(1, dim=1)] = labels_array.unsqueeze(1).cpu() + 2

        embedded_volume_DBSCAN = torch.zeros(params_t.cube_size, params_t.cube_size, params_t.cube_size).detach().long()
        embedded_volume_DBSCAN[coordinates_pixels.split(1, dim=1)] = conected_component_labels.unsqueeze(1).cpu() + 2

        # Create Clustered Image
        space_clusters = torch.zeros_like(final_pred[0, ...])
        if(mask is not None):
            t_mask = mask.clone()
            t_mask = t_mask[0, 0, ...].to(space_clusters.device)
        else:
            t_mask = embedded_volume_marks
        
        space_clusters[offset_pixels.split(1, dim=1)] = t_mask[offset_pixels.long().split(1, dim=1)]

        space_clusters_MARKS = torch.zeros_like(final_pred[0, ...]).cpu()
        space_clusters_MARKS[coordinates_pixels.split(1, dim=1)] = labels_offset.unsqueeze(1).cpu() + 2

        # tensors_io.save_subplots(embedded_volume.unsqueeze(0) * 0, (space_clusters).long().unsqueeze(0).unsqueeze(0), torch.max((embedded_volume_marks).unsqueeze(0).cpu(), (0 * space_clusters).unsqueeze(0).unsqueeze(0).cpu()), t_mask.unsqueeze(0), (embedded_volume_labels).unsqueeze(0), "debug_cluster/side_to_side")
        tensors_io.save_subplots_6(data_image.unsqueeze(0), (space_clusters).long().unsqueeze(0).unsqueeze(0), (embedded_volume_marks).unsqueeze(0).cpu(), space_clusters_MARKS.long().unsqueeze(0).unsqueeze(0),
                                   t_mask.unsqueeze(0), (embedded_volume_labels).unsqueeze(0), embedded_volume_DBSCAN.unsqueeze(0), "debug_cluster/side_to_side")
        V = evaluate_iou_volume(embedded_volume_labels, t_mask.cpu())
        V = torch.from_numpy(V).unsqueeze(0)
        tensors_io.save_volume_h5(t_mask.cpu(), name='mask', directory='debug_cluster/h5')
        tensors_io.save_volume_h5(space_clusters.cpu(), name='space_clusters', directory='debug_cluster/h5')
        tensors_io.save_volume_h5(embedded_volume_marks, name='marks', directory='debug_cluster/h5')
        tensors_io.save_volume_h5(embedded_volume_labels, name='labels', directory='debug_cluster/h5')
        tensors_io.save_volume_h5(final_pred[0, ...].cpu(), name='seg_only', directory='debug_cluster/h5')
        tensors_io.save_volume_h5(counter_image.cpu(), name='counter_image', directory='debug_cluster/h5')
    return labels_array, embedded_volume_marks


def propose_cluster4(idx, hard_labels, offset_pixels, coordinates_pixels, image_w_objects, list_of_objects, parameters):
    threshold = parameters.mpp_min_r + 0.5

    # ############################ FIND BALL AROUND CENTER IN COORD SPACE ###########################
    new_center_real = coordinates_pixels[idx, :]
    # get distances to center coordinate
    distances_real = torch.norm(coordinates_pixels - new_center_real, dim=1)

    # find nearby pixels to cluster center and make them absolute pixels
    neighbors = (distances_real < threshold).nonzero()
    neighbor_label = get_counts(hard_labels[neighbors[:, 0]])
    if(neighbor_label > 0):
        return neighbor_label, None, None
    '''
    # ############################ FIND BALL AROUND CENTER IN OFFSET SPACE ###########################
    new_center_off = offset_pixels[idx, :]
    # get distances to center coordinate
    distances_offset = torch.norm(offset_pixels - new_center_off, dim=1)

    # find nearby pixels to cluster center and make them absolute pixels
    neighbors = (distances_offset < threshold).nonzero()
    neighbor_label = get_counts(hard_labels[neighbors[:, 0]])
    if(neighbor_label > 0):
        return neighbor_label, None, None
    '''

    # ############################ FIND BALL AROUND CENTER #########################################
    new_center_offsets = offset_pixels[idx, :]
    # get distances to center coordinate
    distances_offset = torch.norm(offset_pixels - new_center_offsets, dim=1)

    # find nearby pixels to cluster center and make them absolute pixels
    absolute_fiber_indexes = ((hard_labels == -1) & (distances_offset < threshold)).nonzero()
    absolute_fiber_indexes = absolute_fiber_indexes[:, 0]
    if(absolute_fiber_indexes.shape[0] < parameters.min_points):
        return -1, None, None
    ############################# Get Initial Marks #########################################
    proposed_fiber_real_pixels = coordinates_pixels[absolute_fiber_indexes, :]
    proposed_fiber_offset_pixels = offset_pixels[absolute_fiber_indexes, :]
    marks = get_spatial_properties(proposed_fiber_real_pixels, proposed_fiber_offset_pixels, parameters)

    #proposed_cluster = Tube(marks=marks, label=1, parameters=parameters)
    #overlap_percent, overlap_ids = proposed_cluster.get_prior_and_overlap_ids(image_w_objects, list_of_objects)
    # if(overlap_percent > 0.3):
    #    return -1, None, None
    return -1, marks, absolute_fiber_indexes


def get_counts(array):
    counts = {}
    for x in array:
        if(x.item() in counts.keys()):
            counts[x.item()] += 1
        else:
            counts[x.item()] = 1
    max_counter = 0
    for x in counts.keys():
        if(counts[x] > max_counter):
            max_counter = counts[x]
            value = x
    return value


def expand_cluster4(marks, absolute_fiber_indexes, connected_components, soft_distances, real_pixels, hard_labels, soft_labels, image_w_objects, label, list_of_objects, parameters, data_image):
    # Know to which connected component element these items belong to. 
    connected_components_sets = torch.unique(connected_components[absolute_fiber_indexes])
    candidates = torch.unique(connected_components_sets)
    if(len(candidates) > 1):
        connected_component_of_interest = get_counts(connected_components[absolute_fiber_indexes])
    else:
        connected_component_of_interest = connected_components_sets[0]

    R_hat = parameters.mpp_min_r
    # In theory we want only one with the most votes
    # Get the pixels as ABSOLUTE indexes of interest
    absolute_idx_of_interests_cc = (connected_components == connected_component_of_interest).nonzero()  # [N_cc, 1]
    absolute_idx_of_interests_cc = absolute_idx_of_interests_cc[:, 0]

    # Ger radious from all points to cylinder axis. RELATIVE idxs to connected_component_of_interest
    relative_distance_array = point_line_distance_torch(real_pixels[absolute_idx_of_interests_cc, :], marks[0], marks[3])

    # These pixels are surely part of the cylinder. They are ABSOLUTE indexes
    absolute_expanded_fiber_indexes = absolute_idx_of_interests_cc[relative_distance_array <= R_hat]

    if(absolute_expanded_fiber_indexes.shape[0] < 10):
        return label

    # These are the soft distances and soft labels
    relative_shorter_dist_idx = relative_distance_array < soft_distances[absolute_idx_of_interests_cc]
    absolute_shorter_dist_idx = absolute_idx_of_interests_cc[relative_shorter_dist_idx]

    # Update in the overall array with the shorter distances
    soft_distances[absolute_shorter_dist_idx] = relative_distance_array[relative_shorter_dist_idx]

    # Update the soft labels
    soft_labels[absolute_shorter_dist_idx] = label

    # Update the hard labels
    hard_labels[absolute_expanded_fiber_indexes] = label
    marks = get_spatial_properties(real_pixels[absolute_expanded_fiber_indexes, :], real_pixels[absolute_expanded_fiber_indexes, :], parameters)

    proposed_cluster = Tube(marks=marks, label=label + 2, parameters=parameters)
    proposed_cluster.get_energy(data_image, image_w_objects, list_of_objects, parameters)
    proposed_cluster.Area = max(1, proposed_cluster.Area)
    # print(proposed_cluster.data_energy / proposed_cluster.Area)
    if(proposed_cluster.prior_energy / proposed_cluster.Area > parameters.mpp_T_ov or proposed_cluster.data_energy / proposed_cluster.Area < parameters.Vo_t):
        return label

    list_of_objects.append(proposed_cluster)
    proposed_cluster.draw(image_w_objects)
    # Ger radious from all points to cylinder axis. RELATIVE idxs to connected_component_of_interest
    relative_distance_array = point_line_distance_torch(real_pixels[absolute_idx_of_interests_cc, :], marks[0], marks[3])
    # These pixels are surely part of the cylinder. They are ABSOLUTE indexes
    absolute_expanded_fiber_indexes = absolute_idx_of_interests_cc[relative_distance_array <= R_hat]
    # These are the soft distances and soft labels
    relative_shorter_dist_idx = relative_distance_array < soft_distances[absolute_idx_of_interests_cc]
    absolute_shorter_dist_idx = absolute_idx_of_interests_cc[relative_shorter_dist_idx]
    # Update the soft labels
    hard_labels[absolute_expanded_fiber_indexes] = label

    return label + 1



def refine_cluster4(list_of_objects, cluster_offs, real_pixels, hard_labels, parameters):
    # ############################ FIND BALL AROUND CENTER IN COORD SPACE ###########################
    # Death Process
    to_skip = set()
    '''
    for el in list_of_objects:
        print(el.label)
        print(el.prior_energy / el.Area)
        print(el.data_energy / el.Area)
        print(el.C)
        print("")
        if(el.prior_energy / el.Area > 0.8):
            to_skip.add(el.label)
    '''
    new_labels = torch.zeros_like(hard_labels) - 1
    soft_labels = torch.zeros_like(hard_labels) - 1
    soft_distances = torch.zeros_like(hard_labels).float() + 10000

    for el in list_of_objects:
        if(el.label in to_skip):
            continue
        el.C = torch.from_numpy(el.C).to(real_pixels.device)

        length_array = torch.norm(real_pixels - el.C, dim=1)
        r_array = point_line_distance_torch(real_pixels, el.C, el.w)

        R_hat = parameters.mpp_max_r
        L_hat = (((el.H / 2) ** 2 + R_hat ** 2) ** 0.5)

        indexes = (length_array <= L_hat) & (r_array <= parameters.mpp_curvature * R_hat)
        indexes_abs = indexes.nonzero()
        # These are the soft distances and soft labels
        relative_shorter_dist_idx = r_array[indexes] < soft_distances[indexes]
        absolute_shorter_dist_idx = indexes_abs[relative_shorter_dist_idx]

        # Update in the overall array with the shorter distances
        soft_distances[absolute_shorter_dist_idx] = r_array[absolute_shorter_dist_idx]
        soft_labels[absolute_shorter_dist_idx] = el.label - 2

    return soft_labels

def assign_unclassified_clusters(list_of_objects, initial_indexes, labels_array, coordinates_pixels):
    best_label = torch.ones(initial_indexes.shape).to(labels_array.device) - 2
    best_distances = 1000 * torch.ones(initial_indexes.shape).to(labels_array.device)

    for obj in list_of_objects:
        center = torch.from_numpy(obj.C).to(labels_array.device)
        radious_array = point_line_distance_torch(coordinates_pixels[initial_indexes, :], center, obj.w)
        length_array = torch.norm(coordinates_pixels[initial_indexes, :] - center, dim=1)

        idx_of_interest = (radious_array < best_distances) & (length_array < (1.41 * float(obj.H) / 2.0))
        best_label[idx_of_interest] = obj.label
        best_distances[idx_of_interest] = radious_array[idx_of_interest]

    best_label = best_label.long()
    labels_array[initial_indexes] = best_label
    return labels_array

def check_marks(marks, parameters):
    r = marks[1]
    if(r < parameters.mpp_min_r or r > parameters.mpp_max_r):
        return False
    length = marks[2]
    if(length < parameters.mpp_min_l or length > parameters.mpp_max_l):
        return False

    theta = marks[3]
    if(theta > parameters.mpp_min_t and theta < parameters.mpp_max_t):
        return False

    phi = marks[4]
    if(phi < parameters.mpp_min_p or phi > parameters.mpp_max_p):
        return False

    return True

def get_center_pixels(offset_pixels, dims=64):
    # torch.Size([2281, 3])
    # torch.Size([20890])
    counter_image = torch.zeros(dims, dims, dims)
    for pix in offset_pixels.long():
        out_of_bunds_flag = 0
        for i in range(3):
            if(pix[i] < 0 or pix[i] >= dims):
                out_of_bunds_flag = 1

        if(out_of_bunds_flag == 0):
            counter_image[pix[0], pix[1], pix[2]] += 1.0

    center_values = counter_image[offset_pixels.long().clamp(0, dims-1).split(1, dim=1)]
    center_values = center_values[:, 0]
    _, initial_indexes = torch.sort(-center_values)  # [N_foreground] 
    return counter_image, initial_indexes




def refine_cluster3(hard_labels, image_w_objects, parameters):
    # ############################ FIND BALL AROUND CENTER IN COORD SPACE ###########################
    new_labels = torch.zeros_like(hard_labels) - 1
    for el in list_of_objects:
        if(el.prior_energy > 0.3):
            continue
        else:
            el.C = torch.from_numpy(el.C).to(real_pixels.device)
            length_array = torch.norm(real_pixels - el.C, dim=1)

            r_array = point_line_distance_torch(real_pixels, el.C, el.w)

            R_hat = parameters.mpp_max_r + 0.5
            L_hat = (((el.H / 2) ** 2 + R_hat ** 2) ** 0.5)

            indexes = (length_array <= L_hat) & (r_array <= parameters.mpp_curvature * R_hat)
            new_labels[indexes] = el.label - 2

    return new_labels