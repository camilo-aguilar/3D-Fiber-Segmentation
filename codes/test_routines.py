import codes.utils.evaluation as evaluation
import codes.utils.tensors_io as tensors_io
import torch.nn as nn
import numpy as np
import time
import torch
import scipy.ndimage as ndi
import os
from scipy import ndimage as ndi


#################################################################################################################
# Semantic Segmentation
#################################################################################################################
def test_segmentation(params_t, data_path, mask_path=None):
    print("Starting testing instance for " + params_t.network_name + " in device " + str(t_params.device))
    if(params_t.device is None):
        device = torch.device("cuda:0")

    net_s = params_t.net
    net_s.load_state_dict(torch.load(params_t.net_weights_dir[0]))

    if(params_t.uint_16 is False):
        data_volume = tensors_io.load_volume(data_path, scale=params_t.scale_p).unsqueeze(0)
    else:
        data_volume = tensors_io.load_fibers_uint16(data_path, scale=params_t.scale_p).unsqueeze(0)

    if(params_t.cleaning is True):
        data_volume, mu, std = tensors_io.normalize_dataset_w_info(data_volume)

    result = get_only_segmentation(net_s, data_volume, params_t.n_classes, params_t.cube_size, device=device)

    if(mask_path is not None):
        mask_volume = tensors_io.load_volume_uint16(mask_path, scale=params_t.scale_p).long().unsqueeze(0)
        precision, recall, f1 = evaluation.evaluate_segmentation(result.cpu(), mask_volume.cpu())
    else:
        precision = 0
        recall = 0
        f1 = 0

    return ((data_volume * std) + mu), result, precision, recall, f1


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
# Instance Segmentation
#################################################################################################################

def test_semantic_w_instance(params_t, data_path, mask_path=None):
    print("Starting testing Instance for " + params_t.network_name)
    print("~~~~Using GPU~~~~")

    device = params_t.device

    net_s.load_state_dict(torch.load(params_t.net_weights_dir[0]))
    net_e.load_state_dict(torch.load(params_t.net_weights_dir[1]))

    if(mask_path is not None):
        masks = tensors_io.load_volume_uint16(mask_path, scale=scale_p).long().unsqueeze(0)
        print("MASKS CONTAINS: {} unique fibers".format(len(torch.unique(masks))))

    if(params_t.uint_16 is False):
        data_volume = tensors_io.load_volume(data_path, scale=params_t.scale_p).unsqueeze(0)
    else:
        data_volume = tensors_io.load_fibers_uint16(data_path, scale=params_t.scale_p).unsqueeze(0)

    if(params_t.cleaning is True):
        data_volume, mu, std = tensors_io.normalize_dataset_w_info(data_volume)

    final_fibers = torch.zeros((batch_size, 1, rows, cols, depth), requires_grad=False, dtype=torch.long)

    # ###############################################Semantic Segmentation ############################################################
    final_pred = get_only_segmentation(params_t.net.to(device), data_volume.to(device), params_t.n_classes, params_t.cube_size)
    if(mask_path is not None):
        precision, recall, f1 = evaluation.evaluate_segmentation(final_pred.cpu(), mask_volume.cpu())

    # ################################################Intance Segmentation ############################################################
    start = time.time()
    (final_fibers, _, volume_fibers) = test_net_one_pass_embedding(params_t, data_volume, final_fibers, final_pred)
    print("Embedding took {}".format(time.time() - start))
    ###################################################################################################################################

    if(mask_path is not None):
        evaluate_results.evaluate_fiber_detection(final_fibers, masks)
    print("FINISHED TESTING")
    return final_fibers


def test_net_one_pass_embedding(params_t, data_volume, final_fibers, final_pred):
    device = params_t.device
    cube_size = params_t.cube_size

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

    net_e = params_t.net_i.to(device)
    net_e.eval()

    final_fibers = final_fibers.to(device)
    counter = 0
    total_volumes = len(starting_points_x) * len(starting_points_y) * len(starting_points_z)
    with torch.no_grad():
        for lb_z in starting_points_z:
            for lb_y in starting_points_y:
                for lb_x in starting_points_x:
                    torch.cuda.empty_cache()
                    counter = counter + 1
                    (mini_V, mini_M) = tensors_io.full_crop_3D_image_batched(data_volume, final_pred, lb_x, lb_y, lb_z, params_t.cube_size)
                    mini_M = mini_M.long().to(device)
                    mini_V = mini_V.to(device)

                    space_labels = net_e.forward_inference(mini_V, mini_M, params_t)
                    merge_outputs = merge_volume_vector(final_fibers[0, 0, lb_x:lb_x + cube_size, lb_y:lb_y + cube_size, lb_z:lb_z + cube_size], space_labels.to(device))
                    final_fibers[0, 0, lb_x:lb_x + cube_size, lb_y:lb_y + cube_size, lb_z:lb_z + cube_size] = merge_outputs
    torch.cuda.empty_cache()
    return final_fibers


def merge_volume(Vol_a, Vol_b):
    added_ids_b = set()

    update_in_a = {}

    new_id_map = {}
    new_angle_id = {}
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

            # Find which ids in Volb coincide with the one sin VolB
            ids_in_b = torch.unique(Vol_b[idx1])
            for fiber_id_b in ids_in_b:
                fiber_id_b = fiber_id_b.item()
                if(fiber_id_b == 0 or fiber_id_b == 1):
                    continue
                # Get the fiber information
                f_a = dict_ids[fiber_id_a.item()]
                f_b = new_list_ids[fiber_id_b]
                # Merge fibers based on angles
                if(True):
                    # If a fiber in B was detected as two fibers in A
                    if(fiber_id_b in added_ids_b):
                            new_id_map[fiber_id_b] = fiber_id_a
                            new_angle_id[fiber_id_b] = angle
                            update_in_a[fiber_id_a.item()] = new_id_map[fiber_id_b]
                    else:
                        #  Change fiber id
                        new_id_map[fiber_id_b] = fiber_id_a
                        new_angle_id[fiber_id_b] = angle_between_a_b(f_a, f_b)
                        merged_fibers = merged_fibers + 1

                        # Add to merged fibers
                        added_ids_b.add(fiber_id_b)

        if(old_fiber_id not in added_ids_b):
            # Look where Vol_b was the old id and update Vol_a
            idxb = (Vol_b == old_fiber_id).nonzero().split(1, dim=1)
            Vol_a[idxb] = new_fiber_id
        # if fiber is to merge with a fiber in Vol_a
        else:
            idxb = (Vol_b == old_fiber_id).nonzero().split(1, dim=1)
            Vol_a[idxb] = new_id_map[old_fiber_id]

    return Vol_a


#################################################################################################################
# Quick Test
#################################################################################################################
def quick_seg_inst_test(params_t, data_path, mask_path=None, start_point=[100, 100, 50]):
    print("Starting testing Quick Semantic and Instance for " + params_t.network_name + " in device " + str(params_t.device))
    print("~~~~Using GPU~~~~")

    device = params_t.device

    if(params_t.uint_16 is False):
        data_volume = tensors_io.load_volume(data_path, scale=params_t.scale_p).unsqueeze(0)
    else:
        data_volume = tensors_io.load_fibers_uint16(data_path, scale=params_t.scale_p).unsqueeze(0)

    if(mask_path is not None):
        masks = tensors_io.load_volume_uint16(mask_path, scale=params_t.scale_p).long().unsqueeze(0)
        print("MASKS CONTAINS: {} unique fiber(s)".format(len(torch.unique(masks)) - 1))
    else:
        masks = torch.zeros(data_volume.shape).long()
    if(params_t.cleaning_sangids is True):
        (mini_V, mini_M) = tensors_io.full_crop_3D_image_batched(data_volume, masks, start_point[0], start_point[1], start_point[2], (params_t.cube_size))
        mini_V_or = mini_V.clone()
        data_volume[0, 0, ...] = tensors_io.clean_noise(data_volume[0, 0, ...])

    if(params_t.cleaning is True):
        data_volume, mu, std = tensors_io.normalize_dataset_w_info(data_volume)
        params_t.mu_d = mu
        params_t.std_d = std
    else:
        mu = 0
        std = 1

    print("Cropping Mini Volume and Mask")
    max_fibers = 0
    while(max_fibers < 2):
        print("Cropping at", start_point)
        (mini_V, mini_M) = tensors_io.full_crop_3D_image_batched(data_volume, masks, start_point[0], start_point[1], start_point[2], (params_t.cube_size))
        if(mask_path is not None):
            max_fibers = len(torch.unique(mini_M))
        else:
            max_fibers = 100
    # If networks have single output
    if(params_t.net_i is None):
        # ######################################## Semantic and Instance Ensemble ###########################################
        net = params_t.net
        if(params_t.device == torch.device('cuda:0')):
            net.load_state_dict(torch.load(params_t.net_weights_dir[0], map_location='cuda:0'))
        else:
            net.load_state_dict(torch.load(params_t.net_weights_dir[0], map_location='cuda:1'))
        net = net.to(device)
        if(params_t.debug):
            final_pred, final_fibers, final_clusters = net.forward_inference(mini_V.to(device), params_t, mini_M)
        else:
            final_pred, final_fibers = net.forward_inference(mini_V.to(device), params_t)
    else:
        net_s = params_t.net
        net_e = params_t.net_i

        if(params_t.device == torch.device('cuda:0')):
            net_s.load_state_dict(torch.load(params_t.net_weights_dir[0], map_location='cuda:0'))
            net_e.load_state_dict(torch.load(params_t.net_weights_dir[1], map_location='cuda:0'))
        else:
            net_s.load_state_dict(torch.load(params_t.net_weights_dir[0], map_location='cuda:1'))
            net_e.load_state_dict(torch.load(params_t.net_weights_dir[1], map_location='cuda:1'))
        # ###############################################Semantic Segmentation ############################################################
        print("Getting Semantic Seg")
        final_pred = get_only_segmentation(net_s.to(device), mini_V.to(device), params_t.n_classes, params_t.cube_size)
        # ################################################Intance Segmentation ############################################################
        print("Getting Instance Seg")
        net_i = net_e.to(device)
        if(params_t.debug):
            final_fibers = net_i.forward_inference(mini_V.to(device), final_pred.to(device), params_t, mini_M)
        else:
            final_fibers = net_i.forward_inference(mini_V.to(device), final_pred.to(device), params_t)

    mini_V = (mini_V * std) + mu

    if(mask_path is not None):
        seg_eval = evaluation.evaluate_segmentation(final_pred.cpu(), mini_M.cpu())
        inst_eval = evaluation.evaluate_iou(final_fibers.cpu().numpy(), mini_M.cpu().numpy(), params_t)
    else:
        seg_eval = [1, 1, 1]
        inst_eval = [1, 1, 1]
        mini_M = mini_M * 0

    if(params_t.cleaning_sangids is False):
        mini_V_or = mini_V

    print("FINISHED TESTING")
    if(params_t.debug):
        return mini_V_or, final_pred, final_fibers, final_clusters, mini_M, seg_eval, inst_eval
    else:
        return mini_V_or, final_pred, final_fibers, mini_M, seg_eval, inst_eval
