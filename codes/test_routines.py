import codes.utils.evaluation as evaluation
import codes.utils.tensors_io as tensors_io
from codes.utils.geometry import get_angle_w
import torch.nn as nn
import numpy as np
import time
import torch
import scipy.ndimage as ndi
import os
from scipy import ndimage as ndi
from skimage.morphology import watershed
from collections import defaultdict


#################################################################################################################
# Semantic Segmentation
#################################################################################################################
def get_only_segmentation(net_s, data_volume, n_classes, cube_size, device=None):
    if(device is None):
        device = data_volume.device

    (batch_size, channels, rows, cols, depth) = data_volume.shape

    final_probs = torch.zeros((batch_size, n_classes, rows, cols, depth), requires_grad=False)

    final_probs = test_net_one_pass_segmentation(net_s, data_volume, final_probs, n_classes, cube_size, start_offset=0, device=device)

    # Make a second pass
    # final_probs = test_net_one_pass_segmentation(net_s, data_volume, final_probs, n_classes, cube_size, start_offset=cube_size / 2, device=device)
    _, final_pred = final_probs.max(1)
    final_pred = final_pred.unsqueeze(0)
    return final_pred


def test_net_one_pass_segmentation(net, data_volume, final_probs, n_classes=2, cube_size=192, start_offset=0, device=None):
    first_pass = len(final_probs.nonzero()) > 0
    if(device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (batch_size, channels, rows, cols, depth) = data_volume.shape
    st = start_offset
    starting_points_x = []
    starting_points_y = []
    starting_points_z = []
    while(st + cube_size < rows):
        starting_points_x.append(st)
        st = st + cube_size
    starting_points_x.append(rows - cube_size)

    st = start_offset
    while(st + cube_size < cols):
        starting_points_y.append(st)
        st = st + cube_size
    starting_points_y.append(cols - cube_size)

    st = start_offset
    while(st + cube_size < depth):
        starting_points_z.append(st)
        st = st + cube_size
    starting_points_z.append(depth - cube_size)

    net = net.to(device)
    net.eval()
    counter = 0
    print("Segmenting", end=".")
    with torch.no_grad():
        for lb_z in starting_points_z:
            for lb_y in starting_points_y:
                for lb_x in starting_points_x:
                    counter = counter + 1
                    (mini_V, mini_M) = tensors_io.full_crop_3D_image_batched(data_volume, data_volume, lb_x, lb_y, lb_z, cube_size)
                    mini_V = mini_V.to(device)
                    # mini_M = mini_M.to(device)
                    masks_pred_temp = net(mini_V)
                    masks_probs_temp = nn.functional.softmax(masks_pred_temp, dim=1).float()
                    _, final_pred_temp = masks_probs_temp.max(1)
                    final_pred_temp = final_pred_temp.float()
                    final_pred_temp = final_pred_temp.to(torch.device("cpu"))
                    torch.cuda.empty_cache()
                    if(first_pass):
                        final_probs[:, :, lb_x:lb_x + cube_size, lb_y:lb_y + cube_size, lb_z:lb_z + cube_size] = torch.max(final_probs[:, :, lb_x:lb_x + cube_size, lb_y:lb_y + cube_size, lb_z:lb_z + cube_size], masks_probs_temp.cpu())
                    else:
                        final_probs[:, :, lb_x:lb_x + cube_size, lb_y:lb_y + cube_size, lb_z:lb_z + cube_size] = (final_probs[:, :, lb_x:lb_x + cube_size, lb_y:lb_y + cube_size, lb_z:lb_z + cube_size] + masks_probs_temp.cpu()) / 2
                    print(".", end="")
    print("")
    return final_probs


#################################################################################################################
# Instance Segmentation Helper Functions
#################################################################################################################
def test_net_one_pass_embedding(params_t, data_volume, final_fibers, final_pred, masks):
    import time
    device = params_t.device
    cube_size = params_t.cube_size

    _, _, rows, cols, depth = final_fibers.shape

    overlap = int((1 - params_t.percent_overlap) * cube_size)

    st = 0
    starting_points_x = []
    starting_points_y = []
    starting_points_z = []
    while(st + cube_size < rows):
        starting_points_x.append(st)
        st = st + overlap
    starting_points_x.append(rows - cube_size)

    st = 0
    while(st + cube_size < cols):
        starting_points_y.append(st)
        st = st + overlap
    starting_points_y.append(cols - cube_size)

    st = 0
    while(st + cube_size < depth):
        starting_points_z.append(st)
        st = st + overlap
    starting_points_z.append(depth - cube_size)

    if(params_t.net_i is None):
        net_i = params_t.net.to(device)
    else:
        net_i = params_t.net_i.to(device)
    net_i.eval()

    final_fibers = final_fibers.to(device)
    counter = 0
    total_fibers = 0
    total_volumes = len(starting_points_x) * len(starting_points_y) * len(starting_points_z)
    volume_counter = 0
    with torch.no_grad():
        print("Total Iterations: {}".format(len(starting_points_z) * len(starting_points_y) * len(starting_points_x)))
        for lb_z in starting_points_z:
            for lb_y in starting_points_y:
                for lb_x in starting_points_x:
                    torch.cuda.empty_cache()
                    (mini_V, mini_M) = tensors_io.full_crop_3D_image_batched(data_volume, final_pred, lb_x, lb_y, lb_z, params_t.cube_size)
                    mini_M = mini_M.long().to(device)
                    mini_V = mini_V.to(device)

                    start = time.time()

                    if(params_t.net_i is None):
                        if(params_t.debug):
                            space_pred, space_labels, final_clusters = net_i.forward_inference(mini_V, params_t, mini_M)
                        else:
                            space_pred, space_labels = net_i.forward_inference(mini_V, params_t, mini_M)
                    else:
                        if(params_t.debug_cluster_unet_double is False):
                            space_labels = net_i.forward_inference(mini_V, mini_M, params_t)
                        else:
                            space_labels, marks = net_i.forward_inference(mini_V, mini_M, params_t)

                    if(space_labels is None):
                        continue
                    merge_outputs, total_fibers = merge_volume(final_fibers[0, 0, lb_x:lb_x + cube_size, lb_y:lb_y + cube_size, lb_z:lb_z + cube_size], space_labels[0, 0, ...].to(device), total_fibers)

                    final_fibers[0, 0, lb_x:lb_x + cube_size, lb_y:lb_y + cube_size, lb_z:lb_z + cube_size] = merge_outputs
                    volume_counter += 1
                    # if((volume_counter + 1) % 50 == 0):
                    #    tensors_io.save_subvolume_instances((data_volume * params_t.std_d) + params_t.mu_d, (final_fibers), "total_volume")
                    #    tensors_io.save_volume_h5(final_fibers[0, 0, ...].cpu().numpy(), name='Volume', directory='./h5_files')
    torch.cuda.empty_cache()
    return final_fibers


def merge_volume(Vol_a, Vol_b, total_fibers=0):
    added_ids_b = set()
    added_ids_a = set()
    new_id_map = defaultdict(list)
    result = Vol_a * Vol_b
    overlapping_indices = result.nonzero()
    merged_fibers = 0
    # If there are overlping indices
    if(len(overlapping_indices) > 0):
        overlapping_indices = overlapping_indices.split(1, dim=1)
        overlapping_fiber_ids = torch.unique(Vol_a[overlapping_indices].view(-1))
        for fiber_id_a in overlapping_fiber_ids:
            if(fiber_id_a.item() == 0 or fiber_id_a.item() == 1):
                continue
            # Find indices in Va that overlap
            idx1 = (Vol_a == fiber_id_a).nonzero().split(1, dim=1)

            # Find which ids in Volb coincide with the ones in VolB
            ids_in_b = torch.unique(Vol_b[idx1])
            for fiber_id_b in ids_in_b:
                fiber_id_b = fiber_id_b.item()
                if(fiber_id_b == 0 or fiber_id_b == 1):
                    continue
                # Change fiber id 
                new_id_map[fiber_id_b].append(fiber_id_a)
                # Add to merged fibers
                added_ids_b.add(fiber_id_b)

    # Update volume A and fibers dictionary 
    new_fiber_id = total_fibers + 1
    ids_in_vol_b = torch.unique(Vol_b)
    # for each fiber in the new list
    for fid_b in ids_in_vol_b:
        if(fid_b.item() == 0 or fid_b.item() == 1):
            continue
        idxb = (Vol_b == fid_b).nonzero()
        # variables to find best candidate
        best_candidate = -1
        max_volume = -1000
        temp_vol = torch.zeros_like(Vol_a)
        temp_vol[idxb.split(1, dim=1)] = 1
        # check if the fiber is to be merged
        if(fid_b.item() in added_ids_b):
            # check which candidate in a is best fit according to angle difference and delta 10 degrees
            if(len(new_id_map[fid_b.item()]) > 1):
                for fid_a in new_id_map[fid_b.item()]:
                    idxa = (Vol_a == fid_a).nonzero()
                    overlap = temp_vol[idxa.split(1, dim=1)].sum()

                    # Find the volume that overlaps the most
                    if(overlap > max_volume):
                        best_candidate = fid_a
                        max_volume = overlap
            else:
                best_candidate = new_id_map[fid_b.item()][0]

        if(best_candidate > 0):
            # if a good match was found
            Vol_a[idxb.split(1, dim=1)] = best_candidate
        else:
            # if it is a brand new fiber
            Vol_a[idxb.split(1, dim=1)] = new_fiber_id
            new_fiber_id += 1

    return (Vol_a, new_fiber_id - 1)


#################################################################################################################
# Instance Segmentation
#################################################################################################################

def test_semantic_w_instance(params_t, length=-1):
    print("Starting testing Quick Semantic and Instance for " + params_t.network_name + " in device " + str(params_t.device))
    print("~~~~Using GPU~~~~")

    device = params_t.device
    data_volume, masks, V_or = tensors_io.load_data_and_masks(params_t)

    if(params_t.dataset_name == "2016_s"):
        data_volume = data_volume[:, :, 60:, 42:, 70:]
        masks = masks[:, :, 60:, 42:, 70:]

    if(params_t.dataset_name == "AFRL"):
        data_volume = data_volume[:, :, :, 60:, :]

    '''
    if(length > 32):
        data_volume = data_volume[:, :, 0:length, 0:length, 0:length]
        if(masks is not None):
            masks = masks[:, :, 0:length, 0:length, 0:length]
        if(V_or is not None):
            V_or = V_or[:, :, 0:length, 0:length, 0:length]
    '''
    #tensors_io.save_subvolume_instances((data_volume * params_t.std_d) + params_t.mu_d, masks * 0, 'medium_vols')
    #exit()

    _, _, rows, cols, depth = data_volume.shape
    final_fibers = torch.zeros((params_t.batch_size, 1, rows, cols, depth), requires_grad=False, dtype=torch.long)

    if(params_t.net_i is not None):
        # ###############################################Semantic Segmentation ############################################################
        final_pred = get_only_segmentation(params_t.net.to(device), data_volume.to(device), params_t.n_classes, params_t.cube_size)
        if(params_t.dataset_name == "AFRL"):
            final_pred = (((data_volume * params_t.std_d) + params_t.mu_d) > 0.65).long()

        if(params_t.dataset_name == "voids"):
            # tensors_io.save_subvolume_instances((data_volume * params_t.std_d) + params_t.mu_d, final_pred, "test_voids")
            final_pred = final_pred[..., 0:75]
            masks = masks[..., 0:75]
            data_volume = data_volume[..., 0:75]
            seg_eval1 = evaluation.evaluate_segmentation(final_pred.cpu(), masks.cpu(), n_class=1)
            seg_eval2 = evaluation.evaluate_segmentation(final_pred.cpu(), masks.cpu(), n_class=2)
            params_t.save_quick_results((data_volume * params_t.std_d) + params_t.mu_d, final_pred, final_pred, masks, seg_eval1, seg_eval2)
            exit()
        # ################################################Intance Segmentation ############################################################
    else:
        final_pred = torch.zeros(final_fibers.shape).long()

    final_fibers = test_net_one_pass_embedding(params_t, data_volume.to(device), final_fibers, final_pred, masks)
    ###################################################################################################################################

    if(params_t.testing_mask is not None and False):
        seg_eval = evaluation.evaluate_segmentation(final_pred.cpu(), masks.cpu())
        inst_eval = evaluation.evaluate_iou(final_fibers.cpu().numpy(), masks.cpu().numpy(), params_t)
        print(evaluation.evaluate_adjusted_rand_index_torch(final_fibers.to(device), masks.to(device)))
    else:
        seg_eval = [1, 1, 1]
        inst_eval = [1, 1, 1]

    if(params_t.cleaning is True):
        data_volume = (data_volume * params_t.std_d) + params_t.mu_d

    if(params_t.cleaning_sangids is True):
        data_volume = V_or

    print("FINISHED TESTING")
    return data_volume, final_pred, final_fibers, masks, seg_eval, inst_eval

#################################################################################################################
# Quick Test
#################################################################################################################
def quick_seg_inst_test(params_t, start_point=[0, 0, 0]):
    print("Starting testing Quick Semantic and Instance for " + params_t.network_name + " in device " + str(params_t.device))
    print("Window Size: {}. N Classes: {} N Embeddings: {} N Dim{}".format(params_t.cube_size, params_t.n_classes, params_t.n_embeddings, params_t.ndims))
    print("eps: {}. min_points: {}".format(params_t.eps_param, params_t.min_samples_param))
    print("For dataset " + params_t.dataset_name + " version " + str(params_t.dataset_version))
    print("~~~~Using GPU~~~~")

    device = params_t.device
    data_volume, masks, V_or = tensors_io.load_data_and_masks(params_t)

    if(params_t.cleaning_sangids is True):
        (mini_V_or, mini_M) = tensors_io.full_crop_3D_image_batched(V_or, masks, start_point[0], start_point[1], start_point[2], (params_t.cube_size))

    print("Cropping Mini Volume and Mask")
    max_fibers = 0
    while(max_fibers < 2):
        print("Testing Cropping at", start_point)
        (mini_V, mini_M) = tensors_io.full_crop_3D_image_batched(data_volume, masks, start_point[0], start_point[1], start_point[2], (params_t.cube_size))
        start_point = [(1 + int(np.random.rand(1, 1) * (masks.shape[i + 2] - 2))) for i in range(3)]
        if(params_t.testing_mask is not None):
            max_fibers = len(torch.unique(mini_M))
        else:
            max_fibers = 100
    # If networks have single output
    if(params_t.net_i is None):
        # ######################################## Semantic and Instance Ensemble ###########################################
        net = params_t.net
        net = net.to(device)
        if(params_t.debug):
            final_pred, final_fibers, final_clusters = net.forward_inference(mini_V.to(device), params_t, mini_M)
        else:
            final_pred, final_fibers = net.forward_inference(mini_V.to(device), params_t)
    else:
        net_s = params_t.net
        net_e = params_t.net_i

        # ###############################################Semantic Segmentation ############################################################
        print("Getting Semantic Seg")
        final_pred = get_only_segmentation(net_s.to(device), mini_V.to(device), params_t.n_classes, params_t.cube_size)


        if(params_t.dataset_name == "voids"):
            # tensors_io.save_subvolume_instances((data_volume * params_t.std_d) + params_t.mu_d, final_pred, "test_voids")
            print(final_pred.shape)
            final_pred = final_pred[:, :, 0:64, 0:64, 0:64]
            mini_V = (final_pred * 0).float()
            masks = final_pred * 0;
            params_t.save_quick_results((mini_V * params_t.std_d) + params_t.mu_d, final_pred, final_pred, masks, [1, 1, 1], [1, 1, 1])
            exit()


        # ################################################Intance Segmentation ############################################################
        print("Getting Instance Seg")
        net_i = net_e.to(device)
        if(params_t.debug):
            final_fibers, final_clusters = net_i.forward_inference(mini_V.to(device), final_pred.to(device), params_t, mini_M)
        else:
            final_fibers = net_i.forward_inference(mini_V.to(device), final_pred.to(device), params_t)

    mini_V = (mini_V * params_t.std_d) + params_t.mu_d

    # final_fibers = refine_watershed(final_fibers, final_pred, dbscan=True)
    if(params_t.testing_mask is not None):
        seg_eval = evaluation.evaluate_segmentation(final_pred.cpu(), mini_M.cpu())
        # inst_eval = evaluation.evaluate_iou(final_fibers.cpu().numpy(), mini_M.cpu().numpy(), params_t)
        inst_eval = evaluation.evaluate_iou_pixelwise(final_fibers.cpu().numpy(), mini_M.cpu().numpy(), params_t)
        inst_eval_object_wise = evaluation.evaluate_iou(final_fibers.cpu().numpy(), mini_M.cpu().numpy(), params_t)

        Ra = evaluation.evaluate_adjusted_rand_index_torch(final_fibers.to(device), mini_M.to(device))
    else:
        seg_eval = [1, 1, 1]
        inst_eval = [1, 1, 1]
        inst_eval_object_wise = [1, 1, 1]
        Ra = 1
        mini_M = mini_M * 0

    if(params_t.cleaning_sangids is False):
        mini_V_or = mini_V

    print("FINISHED TESTING")
    if(params_t.debug):
        return mini_V_or, final_pred, final_fibers, final_clusters, mini_M, seg_eval, inst_eval, inst_eval_object_wise, Ra
    else:
        return mini_V_or, final_pred, final_fibers, mini_M, seg_eval, inst_eval, inst_eval_object_wise, Ra


def refine_watershed(labels, mask, dbscan=False):
    device = labels.device
    labels = labels[0, 0, ...].cpu().numpy()
    print("Regining Watershed")
    markers = np.copy(labels)
    markers[np.where(labels == 1)] = 0

    distance = ndi.distance_transform_edt(mask[0, 0, ...].cpu().numpy())
    # distance[np.where(labels > 1)] = 1

    labels = watershed(-distance, markers, mask=mask[0, 0, ...].cpu().numpy())

    labels = torch.from_numpy(labels).long().to(device).unsqueeze(0).unsqueeze(0)

    return labels

