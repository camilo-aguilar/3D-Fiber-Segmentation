from __future__ import print_function
import numpy as np
import math
import torch
import scipy.ndimage as ndi
from scipy.special import comb
from sklearn.metrics.cluster import adjusted_rand_score


def evaluate_segmentation(Vfo, V_or, n_class=0):
    if(n_class == 0):
        Vgt = (V_or > 0).int()
        Vf = (Vfo > 0).int()
    else:
        Vgt = (V_or == n_class).int()
        Vf = (Vfo == n_class).int()
    TP = (Vgt * Vf).sum().float()
    FN = (Vgt * (1 - Vf)).sum().float()
    FP = ((1 - Vgt) * Vf).sum().float()

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2 * TP / (2 * TP + FP + FN)
    print("Semantic: Re:{} Pre:{} f1: {}".format(recall, precision, f1))
    return precision.item(), recall.item(), f1.item()


def evaluate_segmentation_flexible(Vfo, V_or):
    Vgt = V_or.clone()
    Vgt = (Vgt > 0).int()
    distance = ndi.distance_transform_edt(1 - Vgt)

    Vf = Vfo.clone()
    Vf = (Vf > 0).int()

    Vgt_d = Vgt.clone()
    Vgt_d[np.where(distance < 2)] = 1
    TP = (Vgt_d * Vf).sum().float()

    distance = ndi.distance_transform_edt(1 - Vf)
    Vf_d = Vf.clone()
    Vf_d[np.where(distance < 2)] = 1
    FN = (Vgt * (1 - Vf_d)).sum().float()

    FP = ((1 - Vgt_d) * Vf).sum().float()

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2 * TP / (2 * TP + FP + FN)
    print("Recall {}".format(recall))
    print("Precision {}".format(precision))
    print("f1 is {}".format(f1))
    return f1


def evaluate_iou(Vf, Vgt, params_t=False):

    if(params_t.debug):
        labels_to_see = np.zeros(Vf.shape)
    labels_gt = np.unique(Vgt)
    labels_f = np.unique(Vf)

    set_f = set(labels_f)
    set_gt = set(labels_gt)

    # Remove background from label sets
    set_f.remove(0)
    if(1 in set_f):
        set_f.remove(1)
    set_gt.remove(0)

    TP = 0
    FP = 0
    FN = 0

    for Lgt in set_gt:
        # draw gt fiber alone
        Vgt_temp = np.zeros(Vgt.shape)
        idxs_gt = np.where(Vgt == Lgt)
        Vgt_temp[idxs_gt] = 1

        # get overlapping labels in detected fibers (that have not been counted yet)
        labels_in_V = set(np.unique(Vf[idxs_gt])).intersection(set_f)

        if(len(labels_in_V) == 0):
            FN += 1
            if(params_t.debug):
                labels_to_see[idxs_gt] = 2
            continue

        if(len(idxs_gt[0]) < 20):
            continue
        # Check if one of the detected labels is worth checking further
        fiber_detected_flag = False
        for Lf in labels_in_V:
            if(fiber_detected_flag is True):
                FP += 1
                if(params_t.debug):
                    labels_to_see[np.where(Vf == Lf)] = 3
                continue
            # Draw detected object
            Vf_temp = np.zeros(Vgt.shape)
            Vf_temp[np.where(Vf == Lf)] = 1

            intersection = np.logical_and(Vgt_temp, Vf_temp).sum().astype(float)
            union = np.logical_or(Vgt_temp, Vf_temp).sum().astype(float)

            IOU = intersection / union

            if(IOU > 0.4):
                TP += 1
                set_f.remove(Lf)
                if(params_t.debug):
                    labels_to_see[idxs_gt] = 1
                fiber_detected_flag = True
            else:
                FP += 1
                # set_f.remove(Lf)
                if(params_t.debug):
                    labels_to_see[np.where(Vf == Lf)] = 3

        if(fiber_detected_flag is False):
            FN += 1
            if(params_t.debug):
                labels_to_see[idxs_gt] = 2

    if(params_t.debug):
        for Lf in set_f:
            labels_to_see[np.where(Vf == Lf)] = 3
    TP = float(TP)
    FN = float(FN)
    FP = float(FP)

    if(params_t.debug):
        from .tensors_io import save_subvolume_IoU
        import os
        labels_to_see = torch.from_numpy(labels_to_see).long()
        vol = torch.zeros(labels_to_see.shape)
        results_directory = "results"
        path = results_directory + "/" + params_t.dataset_name + "/" + "v_" + str(params_t.dataset_version) + "/" + params_t.network_string + "/IOU_checking"
        path_temp = ""
        for local_path in path.split("/"):
            path_temp = path_temp + local_path
            if not os.path.isdir(path_temp):
                os.mkdir(path_temp)
            path_temp = path_temp + "/"
        
        save_subvolume_IoU(vol, labels_to_see, path)

    recall = TP / (TP + FN + 0.001)
    precision = TP / (TP + FP + 0.0001)
    f1 = 2 * TP / (2 * TP + FP + FN + 0.0001)
    print("Instace: Re:{} Pre:{} f1: {}".format(recall, precision, f1))
    return precision, recall, f1


def evaluate_iou_pixelwise(Vf, Vgt, params_t=False):
    labels_gt = np.unique(Vgt)
    labels_f = np.unique(Vf)

    set_f = set(labels_f)
    set_gt = set(labels_gt)

    # Remove background from label sets
    set_f.remove(0)
    if(1 in set_f):
        set_f.remove(1)
    set_gt.remove(0)

    TP = 0
    FP = 0
    FN = (Vf == 1).sum()

    used_vf_ids = {}
    for Lgt in set_gt:
        max_intersection = 0
        max_intersection_id = None
        intersection_pixels = 0

        # draw gt fiber alone
        Vgt_temp = np.zeros(Vgt.shape)
        idxs_gt = np.where(Vgt == Lgt)
        Vgt_temp[idxs_gt] = 1

        # get overlapping labels in detected fibers (that have not been counted yet)
        labels_in_V = set(np.unique(Vf[idxs_gt])).intersection(set_f)

        if(len(labels_in_V) == 0):
            FN += Vgt_temp.sum()

        # Check if one of the detected labels is worth checking further
        for Lf in labels_in_V:
            # Draw detected object
            Vf_temp = np.zeros(Vgt.shape)
            Vf_temp[np.where(Vf == Lf)] = 1

            intersection = np.logical_and(Vgt_temp, Vf_temp).sum().astype(float)
            intersection_pixels += intersection
            if(intersection > max_intersection):
                max_intersection = intersection
                max_intersection_id = Lf

        FP += intersection_pixels - max_intersection
        if(max_intersection_id not in used_vf_ids.keys()):
            used_vf_ids[max_intersection_id] = max_intersection
            TP += max_intersection
        else:
            if(used_vf_ids[max_intersection_id] < max_intersection):
                FP += used_vf_ids[max_intersection_id]
                TP -= used_vf_ids[max_intersection_id]
                used_vf_ids[max_intersection_id] = max_intersection
                TP += max_intersection
            else:
                FP += max_intersection

    TP = float(TP)
    FN = float(FN)
    FP = float(FP)

    # print(TP)
    # print(FN)
    # print(FP)

    recall = TP / (TP + FN + 0.001)
    precision = TP / (TP + FP + 0.0001)
    f1 = 2 * TP / (2 * TP + FP + FN + 0.0001)
    print("Instace: Re:{} Pre:{} f1: {}".format(recall, precision, f1))
    return precision, recall, f1

def evaluate_iou_volume(Vf, Vgt, params_t=False):
    labels_to_see = np.zeros(Vf.shape)
    labels_gt = np.unique(Vgt)
    labels_f = np.unique(Vf)

    set_f = set(labels_f)
    set_gt = set(labels_gt)

    # Remove background from label sets
    set_f.remove(0)
    if(1 in set_f):
        set_f.remove(1)
    set_gt.remove(0)

    for Lgt in set_gt:
        # draw gt fiber alone
        Vgt_temp = np.zeros(Vgt.shape)
        idxs_gt = np.where(Vgt == Lgt)
        Vgt_temp[idxs_gt] = 1

        # get overlapping labels in detected fibers (that have not been counted yet)
        labels_in_V = set(np.unique(Vf[idxs_gt])).intersection(set_f)

        if(len(labels_in_V) == 0):
            # False Negative
            labels_to_see[idxs_gt] = 2
            continue

        # Check if one of the detected labels is worth checking further
        fiber_detected_flag = False
        for Lf in labels_in_V:
            if(fiber_detected_flag is True):
                # False Positive
                labels_to_see[np.where(Vf == Lf)] = 3
                continue

            # Draw detected object
            Vf_temp = np.zeros(Vgt.shape)
            Vf_temp[np.where(Vf == Lf)] = 1

            intersection = np.logical_and(Vgt_temp, Vf_temp).sum().astype(float)
            union = np.logical_or(Vgt_temp, Vf_temp).sum().astype(float)

            IOU = intersection / union

            if(IOU > 0.8):
                # True Positive
                set_f.remove(Lf)
                labels_to_see[idxs_gt] = 1
                fiber_detected_flag = True
            else:
                # False Positive
                labels_to_see[np.where(Vf == Lf)] = 3

        if(fiber_detected_flag is False):
            # False Negative
            labels_to_see[idxs_gt] = 2

    for Lf in set_f:
        # False Positive
        labels_to_see[np.where(Vf == Lf)] = 3
    return labels_to_see



def evaluate_fiber_detection(Vf, Vgt, params_t=None):
    labels_gt = np.unique(Vgt)
    num_fibers = 0

    labels_f = np.unique(Vf)
    num_fibers_f = len(labels_f) - 1

    set_f = set(labels_f)
    set_gt = set(labels_gt)

    set_f.remove(0)
    set_gt.remove(0)

    broken_fibers = 0
    detected_fibers = 0
    missed_fibers = 0
    counter = 0
    for Lgt in set_gt:
        counter += 1
        print("{} out of {}".format(counter, num_fibers))
        # get gt fiber
        Vgt_temp = np.zeros(Vgt.shape)
        idxs_gt = np.where(Vgt == Lgt)
        Vgt_temp[idxs_gt] = 1
        Vgt_temp, num_fs = ndi.measurements.label(Vgt_temp)
        for gt_fiber in np.unique(Vgt_temp):
            total_intersection = 0
            detected_flag = 0
            if(gt_fiber == 0):
                continue


            Vgt_temp2 = np.zeros(Vgt.shape)
            idxs_gt2 = np.where(Vgt_temp == gt_fiber)
            volume_gt = len(idxs_gt2[0])
            # get labels in detected fibers
            labels_in_V = set(np.unique(Vf[idxs_gt2]))

            Vgt_temp2[idxs_gt2] = 1
            # pick only from the remaining labels
            # labels_in_V = labels_in_V.intersection(set_f)
            num_fibers += 1
            # if(len(labels_in_V) > 1):
            #    print(Lgt, labels_in_V)
            T_good_fiber = 0.3
            if(len(labels_in_V) == 0):
                continue
            for Lf in labels_in_V:
                if(Lf == 0):
                    continue
                # set_f.remove(Lf)
                Vf_temp = np.zeros(Vgt.shape)
                idxs_f = np.where(Vf == Lf)
                Vf_temp[idxs_f] = 1
                volume_f = len(idxs_f[0])
                intersection = np.logical_and(Vgt_temp2, Vf_temp).sum().astype(float)
                if(float(intersection) / float(volume_gt) > T_good_fiber):
                    detected_fibers += 1
                    detected_flag = 1
                    break
                # else:
                    # print(Lgt, Lf, float(intersection) / float(volume_gt), volume_gt)
                if(float(intersection) / float(volume_f) > T_good_fiber):
                    total_intersection += intersection

            if(detected_flag == 0):
                if(float(total_intersection) / float(volume_gt) > T_good_fiber):
                    broken_fibers += 1
                    print("Broken", Lgt, num_fs, labels_in_V, total_intersection, volume_gt)
                else:
                    missed_fibers += 1
                    print("Missed", Lgt, num_fs, labels_in_V, total_intersection, volume_gt)

    print("Labels Vg:{}. Detected: {}. Broken {}. Missed {}".format(num_fibers, detected_fibers, broken_fibers, missed_fibers))
    return(num_fibers, detected_fibers, missed_fibers)


def evaluate_adjusted_rand_index(Vf, Vgt):

    def nCr(n, r):
        f = math.factorial
        return f(n) / f(r) / f(n - r)

    labels_gt = np.unique(Vgt)
    num_fibers_gt = len(labels_gt) - 1

    labels_f = np.unique(Vf)
    num_fibers_f = len(labels_f) - 1

    t1 = nCr(num_fibers_gt, 2)
    t2 = nCr(num_fibers_f, 2)

    print("Fibers in Gt: {}".format(num_fibers_gt))
    print("Fibers in Vf: {}".format(num_fibers_f))
    n = Vgt.shape[0] * Vgt.shape[1] * Vgt.shape[2]
    n = float(n)

    t3 = (t1 * t2) / (n * (n + 1.0))

    print(t1, t2, t3)
    print(((t1 + t2) / 2) - t3)
    exit()
    print(n)
    set_f = set(labels_f)
    set_gt = set(labels_gt)

    set_f.remove(0)
    set_gt.remove(0)
    numerator = 0
    for Lgt in set_gt:
        for Lf in set_f:
            Vf_temp = np.zeros(Vgt.shape)
            Vg_temp = np.zeros(Vgt.shape)

            Vf_temp[np.where(Vf == Lf)] = 1
            Vg_temp[np.where(Vgt == Lgt)] = 1

            mij = np.sum(Vf_temp * Vg_temp)
            if(mij > 2):
                numerator += nCr(mij, 2) - t3
                print(numerator)

    denominator = ((t1 + t2) / 2) - t3

    Ra = numerator / denominator
    return Ra


def evaluate_adjusted_rand_index_torch(Vf, Vgt):
    Vf_o = Vf
    Vgt_o = Vgt
    clusters_gt = Vf_o[Vgt_o.nonzero().split(1, dim=1)]
    clusters_f = Vgt_o[Vgt_o.nonzero().split(1, dim=1)]

    clusters_gt = clusters_gt[clusters_f > 0]
    clusters_f = clusters_f[clusters_f > 0]

    # r = adjusted_rand_score(clusters_gt[:, 0].cpu().numpy(), clusters_f[:, 0].cpu().numpy())
    r = adjusted_rand_score(clusters_gt.cpu().numpy(), clusters_f.cpu().numpy())
    print("Adjusted Rand Index: {}".format(r))
    return r
    '''
    def nCr(n, r):
        f = math.factorial
        return f(n) / f(r) / f(n - r)

    labels_gt = torch.unique(Vgt)
    num_fibers_gt = labels_gt.shape[0] - 1

    labels_f = torch.unique(Vf)
    num_fibers_f = labels_f.shape[0] - 1

    print("Fibers in Gt: {}".format(num_fibers_gt))
    print("Fibers in Vf: {}".format(num_fibers_f))
    # n = Vgt.shape[-1] ** 3
    n = len(Vgt.nonzero())
    n = float(n)

    t3 = 0

    set_f = set(labels_f)
    set_gt = set(labels_gt)

    t1 = 0.0
    for Lf in set_f:
        if(Lf == 0):
            continue
        Vf_temp = torch.zeros(Vgt.shape).to(Vf.device)
        idx1 = (Vf == Lf).nonzero()
        t1 += float(comb(len(idx1), 2))

    t2 = 0.0
    for Lgt in set_gt:
        if(Lgt == 0):
            continue
        Vg_temp = torch.zeros(Vgt.shape).to(Vf.device)
        idx2 = (Vgt == Lgt).nonzero()
        t2 += float(comb(len(idx2), 2))

    t3 = (2.0 * t1 * t2) / (n * (n - 1))
    numerator = 0.0
    for Lgt in set_gt:
        if(Lgt == 0):
            continue
        for Lf in set_f:
            if(Lf == 0):
                continue
            Vf_temp = torch.zeros(Vgt.shape).to(Vf.device)
            Vg_temp = torch.zeros(Vgt.shape).to(Vf.device)
            idx1 = (Vf == Lf).nonzero().split(1, dim=1)
            idx2 = (Vgt == Lgt).nonzero().split(1, dim=1)
            Vf_temp[idx1] = 1
            Vg_temp[idx2] = 1

            mij = (Vf_temp * Vg_temp).sum().cpu().int().item()
            if(mij > 2):
                numerator += float(comb(mij, 2))
    numerator = numerator - t3
    denominator = ((t1 + t2) / 2) - t3

    Ra = numerator / denominator
    return Ra
    '''

if __name__ == '__main__':
    import tensors_io
    volume = tensors_io.load_volume_uint16("/Storage/2020/test_restults_transformed", scale=2).long().unsqueeze(0)
    masks = tensors_io.load_volume_uint16("/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_2_anno", scale=1).long().unsqueeze(0)
    (mini_V, mini_M) = tensors_io.full_crop_3D_image_batched(masks, masks, 128, 128, 40, (64))
    mini_M = mini_M[..., 0:63]
    seg_eval = evaluate_segmentation((volume > 0).cpu(), mini_M.cpu())
    print(evaluate_adjusted_rand_index_torch(volume, mini_M))
    tensors_io.save_subvolume_instances( (volume * 0).float(), volume, "wow")
    tensors_io.save_subvolume_instances( (mini_M * 0).float(), mini_M, "wow_gt")
