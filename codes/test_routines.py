import codes.utils.evaluation as evaluation
import codes.utils.tensors_io as tensors_io
import torch.nn as nn
import numpy as np
import time
import torch
import scipy.ndimage as ndi
import os
from scipy import ndimage as ndi


def test_segmentation(params_t, data_path, mask_path=None):
    print("Starting testing for " + params_t.network_name)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def test_net_one_pass_embedding(net_e, data_volume, final_fibers, final_pred, n_embedded=12, cube_size=32, percent_overlap=0.1, start_offset=[0, 0, 0], fibers_before=0, fiber_dict={}, volume_n=0, device=None):
    start = time.time()
    (batch_size, channels, rows, cols, depth) = data_volume.shape
    if(device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    overlap = int( (1 - percent_overlap) * cube_size)

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

    net_e.to(device)
    net_e.eval()

    final_fibers = final_fibers.to(device)
    counter = 0
    total_volumes = len(starting_points_x) * len(starting_points_y) * len(starting_points_z)
    num_fibers = fibers_before

    with torch.no_grad():
        for lb_z in starting_points_z:
            for lb_y in starting_points_y:
                for lb_x in starting_points_x:
                    torch.cuda.empty_cache()
                    counter = counter + 1
                    (mini_V, mini_M) = tensors_io.full_crop_3D_image_batched(data_volume, final_pred, lb_x, lb_y, lb_z, cube_size)
                    mini_M = mini_M.long()
                    mini_V = mini_V.to(device)
                    mini_M = mini_M.to(device)

                    outputs2 = net_e.forward_inference_fast(mini_V, mini_M, eps_param=0.20, min_samples_param=40)
                    if(outputs2 is None):
                        print('V: {}.  SubVolume {} out of {}. Found {} fibers'.format(volume_n, counter, total_volumes, 0))
                        continue
                    space_labels = outputs2[0].to(device)
                    list_of_ids = outputs2[1]

                    merge_outputs = merge_volume_vector(final_fibers[0, 0, lb_x:lb_x + cube_size, lb_y:lb_y + cube_size, lb_z:lb_z + cube_size], space_labels, dict_ids=fiber_dict, new_list_ids=list_of_ids, lb_x=lb_x + start_offset[0], lb_y=lb_y + start_offset[1], lb_z=lb_z + start_offset[2], total_fibers=num_fibers)
                    final_fibers[0, 0, lb_x:lb_x + cube_size, lb_y:lb_y + cube_size, lb_z:lb_z + cube_size] = merge_outputs[0]
                    num_fibers = merge_outputs[1]

                    print('V: {}.  SubVolume {} out of {}. Found {} fibers'.format(volume_n, counter, total_volumes, num_fibers))

    print("Estimating Fiber Properties")
    torch.cuda.empty_cache()
    outputs = get_fiber_properties(final_fibers[0, 0, :, :, :], large_volume=True)
    centers, fiber_ids, end_points, fiber_list = outputs

    print("Merging Neighbors")
    merged_n = merge_inner_fibers(end_points, fiber_list, fiber_ids, final_fibers[0, 0, :, :, :], debug=0, angle_threshold=5)
    print("Merged {} neighbors first pass".format(merged_n))
    
    print("Estimating Fiber Properties Second Pass")
    torch.cuda.empty_cache()
    outputs = get_fiber_properties(final_fibers[0, 0, :, :, :], large_volume=True)
    centers, fiber_ids, end_points, fiber_list = outputs

    print("Merging Neighbors Second Pass")
    merged_n = merge_inner_fibers(end_points, fiber_list, fiber_ids, final_fibers[0, 0, :, :, :], debug=0, angle_threshold=5)

    # merged_n += merge_inner_fibers(end_points, fiber_list, fiber_ids, final_fibers[0, 0, :, :, :], debug=0, angle_threshold=5)
    print("Merged {} neighbors in second pass".format(merged_n))

    print("Appliying Offset")
    for dict_id in fiber_list:
        fiber_list[dict_id][3] += start_offset[2]

    fiber_dict.update(fiber_list)
    end = time.time() - start
    print("Subvolume fibers finished: {} seconds".format(end))
    return (final_fibers, fiber_list, num_fibers)


def merge_volume_vector(Vol_a, Vol_b, dict_ids={}, new_list_ids={}, lb_x=0, lb_y=0, lb_z=0, total_fibers=0):
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
                        angle = angle_between_a_b(f_a, f_b)
                        if(angle < new_angle_id[fiber_id_b]):
                            new_id_map[fiber_id_b] = fiber_id_a
                            new_angle_id[fiber_id_b] = angle
                        update_in_a[fiber_id_a.item()] = new_id_map[fiber_id_b]
                    else:
                        # Change fiber id 
                        new_id_map[fiber_id_b] = fiber_id_a
                        new_angle_id[fiber_id_b] = angle_between_a_b(f_a, f_b)
                        merged_fibers = merged_fibers + 1
                        
                        # Add to merged fibers
                        added_ids_b.add(fiber_id_b)

    # Update volume A and fibers dictionary 
    new_fiber_id = total_fibers + 1
    # new_list_ids = torch.unique(Vol_b)
    # for each fiber in the new list
    for k in new_list_ids.keys():
        el = new_list_ids[k]
        #if(el == 0 or el == 1):
        #    continue
        old_fiber_id = el[0]
        #old_fiber_id = el
        # if it is a brand new fiber
        if(old_fiber_id not in added_ids_b):

            el[0] = new_fiber_id
            el[1] = el[1] + lb_x
            el[2] = el[2] + lb_y
            el[3] = el[3] + lb_z

            # Look where Vol_b was the old id and update Vol_a
            idxb = (Vol_b == old_fiber_id).nonzero().split(1, dim=1)
            Vol_a[idxb] = new_fiber_id

            # Update dictionary id and fiber # 
            dict_ids[new_fiber_id] = el
            new_fiber_id = new_fiber_id + 1
        # if fiber is to merge with a fiber in Vol_a
        else:
            idxb = (Vol_b == old_fiber_id).nonzero().split(1, dim=1)
            Vol_a[idxb] = new_id_map[old_fiber_id]

            key1= new_id_map[old_fiber_id]
            old_f = dict_ids[key1.item()]
            old_f[1] = (old_f[1] + el[1] + lb_x) / 2
            old_f[2] = (old_f[2] + el[2] + lb_y) / 2
            old_f[3] = (old_f[3] + el[3] + lb_z) / 2
            old_f[4] = (old_f[4] + el[4]) / 2
            old_f[5] = (old_f[5] + el[5])
            old_f[6] = (old_f[6] + el[6]) / 2
            old_f[7] = (old_f[7] + el[7]) / 2
            old_f[8] = (old_f[8] + el[8]) / 2
            dict_ids[key1.item()] = old_f

    # to keep unlabeled pixels
    '''
    temp_vol_a = (Vol_a > 0).clone().long()
    temp_vol_a = 1 - temp_vol_a
    idxb = ((Vol_b * temp_vol_a) == 1).nonzero()
    if(len(idxb) > 0):
        idxb = idxb.split(1, dim=1)
        Vol_a[idxb] = 1
    '''
    return (Vol_a, new_fiber_id - 1, update_in_a)


def angles_are_close_vector(f_a, f_b):
    dir_a = np.array([f_a[6], f_a[7], f_a[8]])
    dir_b = np.array([f_b[6], f_b[7], f_b[8]])

    L_a = f_a[5]
    L_b = f_b[5]
    if(L_a < 10 or L_b < 10):
        return True

    angle_between_a_b = np.arccos(np.abs(np.dot(dir_b, dir_a))) * 180 / np.pi
    #print(angle_between_a_b)
    if(angle_between_a_b < 20):
        return True
    else:
        return False

def angle_between_a_b(f_a, f_b):
    dir_a = np.array([f_a[6], f_a[7], f_a[8]])
    dir_b = np.array([f_b[6], f_b[7], f_b[8]])

    angle_between_a_b = np.arccos(np.abs(np.dot(dir_b, dir_a))) * 180 / np.pi
    #print(angle_between_a_b)
    return angle_between_a_b



def fill_watershed(labels, segmentation=None):
    device = labels.device
    labels = labels.cpu().numpy()
    segmentation = segmentation.cpu().numpy()
    segmentation[np.where(segmentation == 2)] = 0

    markers = np.copy(labels)

    distance = ndi.distance_transform_edt(segmentation)
    distance[np.where(labels > 0)] = 1
    labels = watershed(-distance, markers, mask=segmentation)

    labels= torch.from_numpy(labels).long().to(device)

    return labels



def test_full_embedded_net(net_e, net_s, data_path, mask_path=None, n_classes=3, n_embedded=16, cube_size=32, epochs=10, batch_size=1, Patches_per_Epoch=30, scale_p=2, net_weights_dir=None, cleaning=True, eps_param=0.40, min_samples=5):
    '''
        Test Net on a 450x450x450 Volume
    '''

    device = torch.device('cuda:0')
    print("Starting Testing Embedded.......")
    print("~~~~Using GPU~~~~")
    if(net_weights_dir is None):
        net_e.load_state_dict(torch.load('info_files/net_fibers_e.pth'))
        net_s.load_state_dict(torch.load('info_files/net_fibers_s.pth'))
    else:
        net_s.load_state_dict(torch.load(net_weights_dir[0]))
        net_e.load_state_dict(torch.load(net_weights_dir[1]))

    # net_s.load_state_dict(torch.load('info_files/u_net_3D_k.pth'))

    if(mask_path is not None):
        masks = tensors_io.load_volume_uint16(mask_path, scale=scale_p).long().unsqueeze(0)
        print("MASKS CONTAINS: {} unique fibers".format(len(torch.unique(masks))))

    if(cleaning is True):
        data_volume = tensors_io.load_volume(data_path, scale=scale_p)
        data_volume[0, ...] = tensors_io.clean_noise(data_volume[0, ...], data_path)
    else:
        data_volume = tensors_io.load_volume(data_path, scale=scale_p)
        #data_volume = tensors_io.load_fibers_uint16(data_path, scale=scale_p)

    data_volume = data_volume.unsqueeze(0)

    (batch_size, channels, rows, cols, depth) = data_volume.shape

    final_fibers = torch.zeros((batch_size, 1, rows, cols, depth), requires_grad=False, dtype=torch.long)
    list_of_fibers = {}

    ################################################Semantic Segmentation ############################################################
    cube_size_s = cube_size  #192
    n_classes_s = 2
    net_s = net_s.to(device)
    data_volume = data_volume.to(device)
    print("Segmenting")
    final_pred = get_only_segmentation(net_s, data_volume, n_classes_s, cube_size_s)
    if(mask_path is not None):
        evaluate_results.evaluate_segmentation(final_pred.int().to(device), masks.int().to(device))
    tensors_io.save_subvolume_instances(data_volume, final_pred.long(), 'process_all_instances_seg2')
    #################################################Intance Segmentation ############################################################
    start = time.time()
    (final_fibers, _, volume_fibers) = test_net_one_pass_embedding(net_e, data_volume, final_fibers, final_pred, n_embedded, cube_size, percent_overlap=0.2, start_offset=[0, 0, 0], fibers_before=0, fiber_dict=list_of_fibers)
    print("Embedding took {}".format(time.time() - start))
    ###################################################################################################################################
    tensors_io.save_subvolume_instances(data_volume, final_fibers.long(), 'process_all_instances')

    if(mask_path is not None):
        evaluate_results.evaluate_fiber_detection(final_fibers, masks)
    # unknown_idx = (final_fibers == 1).nonzero().split(1, dim=1)
    # final_fibers[unknown_idx] = 0
    
    #voids_idx = (final_pred == 2).nonzero().split(1, dim=1)
    #final_fibers[voids_idx] = 1
    # camilo
    tensors_io.save_volume_h5(final_fibers[0, 0, ...].cpu().numpy(), directory='./h5_files', name='final_fibers_single_unet', dataset_name='final_fibers_single_unet')

    
    # print(list_of_fibers)
    f = open("dict_single.txt","w")
    for k in list_of_fibers.keys():
        el = list_of_fibers[k]
        # print(el[8])
        f.write("{},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f},{:.0f},{:.0f},{:.3f}\n".format(el[0],el[1],el[2],el[3],el[4],el[5],el[6],el[7],el[8]))
        # print(len(el))
        # f.write("{},{:.0f},{:.0f},{:.0f},{:.0f},{:.02f},{:.2f},{:.0f},{:.0f},{:.3f}".format(el[0],el[1],el[2],el[3],el[4],el[5],el[6],el[7]))#,el[8]))
    f.close()

    # tensors_io.save_subvolume_instances(data_volume * 0, ((final_pred.long() - final_fibers.long()) > 0).float(), 'process_all_instances_diff')
    # tensors_io.save_subvolume_instances(data_volume , ((final_pred.float() - final_fibers.float()) > 0).float(), 'process_all_instances_diff_mask')
    print("FINISHED TESTING")
    return final_fibers
