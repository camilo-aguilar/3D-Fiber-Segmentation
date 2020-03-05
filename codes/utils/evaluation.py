from __future__ import print_function
import numpy as np
import math
import scipy.ndimage as ndi


def evaluate_segmentation(Vfo, V_or):
    Vgt = (V_or > 0).int()
    Vf = (Vfo > 0).int()
    TP = (Vgt * Vf).sum().float()
    FN = (Vgt * (1 - Vf)).sum().float()
    FP = ((1 - Vgt) * Vf).sum().float()

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2 * TP / (2 * TP + FP + FN)
    print("Re:{} Pre:{} f1: {}".format(recall, precision, f1))
    return f1


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


def evaluate_iou(Vf, Vgt):
    labels_gt = np.unique(Vgt)
    num_fibers = len(labels_gt) - 1

    labels_f = np.unique(Vf)
    num_fibers_f = len(labels_f) - 1

    set_f = set(labels_f)
    set_gt = set(labels_gt)

    set_f.remove(0)
    set_gt.remove(0)

    fibers_corrected_detected = 0
    fibers_splitted_but_detected = 0

    fibers_in_v_detected_double = 0
    flag_match_detected = 0
    mean_iou = 0
    counter = 0
    for Lgt in set_gt:

        # get gt fiber 
        Vgt_temp = np.zeros(Vgt.shape)
        idxs_gt = np.where(Vgt == Lgt)
        Vgt_temp[idxs_gt] = 1

        # get labels in detected fibers
        labels_in_V = set(np.unique(Vf[idxs_gt]))
        labels_in_V = labels_in_V.intersection(set_f)

        if(len(labels_in_V) == 0):
            continue

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
            
            ind_IOU = intersection / union

            if(ind_IOU > 0.5):
                flag_match_detected = 1
                fibers_corrected_detected += 1
                set_f.remove(Lf)
                IOU = ind_IOU
                break
            else:
                total_intersection += intersection
                set_broken_fibers.add(Lf)
            '''
            if(not flag_match_detected):
                IOU = total_intersection / union
                if(IOU > 0.5):
                    fibers_splitted_but_detected += 1
                    for item in set_broken_fibers:
                        set_f.remove(item)
            '''
        mean_iou += IOU
        counter += 1

    print("Num Fibers Gt: {}".format(num_fibers))
    print("Labels Vf:{}".format(num_fibers_f))
    TP = fibers_corrected_detected
    FN = num_fibers - fibers_corrected_detected
    FP = num_fibers_f - fibers_corrected_detected

    TP = float(TP)
    FN = float(FN)
    FP = float(FP)

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2 * TP / (2 * TP + FP + FN)
    print("Recall {}".format(recall))
    print("Precision {}".format(precision))
    print("f1 is {}".format(f1))
    return f1


def evaluate_fiber_detection(Vf, Vgt):
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
            if(volume_gt < 500):
                continue
            num_fibers += 1
            # if(len(labels_in_V) > 1):
            #    print(Lgt, labels_in_V)
            T_good_fiber = 0.5
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
                intersection = np.logical_and(Vgt_temp2, Vf_temp).sum().astype(np.float)
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