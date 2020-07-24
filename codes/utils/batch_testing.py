from codes.test_routines import quick_seg_inst_test
from matplotlib import pyplot as plt 
from sklearn.cluster import DBSCAN
import numpy as np
import os
import codes.utils.tensors_io as tensors_io


def batch_test(parameters, args):
    print("Starting Batch Testing")
    if(parameters.offset_clustering is True):
        eps_parameters = [0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10, 15]
        #eps_parameters = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10, 15]
    else:
        eps_parameters = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3]

    if(parameters.debug_cluster_unet_double):
        if(parameters.network_number == 4):
            parameters.network_string = "METHOD"
        else:
            parameters.network_string = "METHOD_double"

    recall_array = np.zeros(len(eps_parameters))
    precision_array = np.zeros(len(eps_parameters))
    f1_array = np.zeros(len(eps_parameters))

    recall_array_objectwise = np.zeros(len(eps_parameters))
    precision_array_objectwise = np.zeros(len(eps_parameters))
    f1_array_objectwise = np.zeros(len(eps_parameters))

    Ra_arary = np.zeros(len(eps_parameters))

    counter = 0
    for eps_p in eps_parameters:
        print("Starting Testing for: {}".format(eps_p))
        parameters.eps_param = eps_p
        # parameters.min_samples_param = 30
        parameters.clustering = DBSCAN(eps=parameters.eps_param, min_samples=parameters.min_samples_param).fit_predict

        if(args.debug):
            mini_V, final_pred, final_fibers, final_clusters, mini_gt, seg_f1, ins_f1, ins_f1_object, Ra = quick_seg_inst_test(parameters, start_point=args.start_point)
        else:
            mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1, ins_f1_object, Ra = quick_seg_inst_test(parameters, start_point=args.start_point)

        precision_array[counter] = ins_f1[0]
        recall_array[counter] = ins_f1[1]
        f1_array[counter] = ins_f1[2]

        precision_array_objectwise[counter] = ins_f1_object[0]
        recall_array_objectwise[counter] = ins_f1_object[1]
        f1_array_objectwise[counter] = ins_f1_object[2]

        Ra_arary[counter] = Ra
        counter += 1

    path = "results/" + parameters.dataset_name + "/" + "v_" + str(parameters.dataset_version) + "/" + parameters.network_string
    print("Saving in:")
    print(path)
    path_temp = ""
    for local_path in path.split("/"):
        path_temp = path_temp + local_path
        if not os.path.isdir(path_temp):
            os.mkdir(path_temp)
        path_temp = path_temp + "/"

    to_write = np.array([precision_array, recall_array, f1_array, precision_array_objectwise, recall_array_objectwise, f1_array_objectwise, Ra_arary])
    directory = "results/" + parameters.dataset_name + "/" + "v_" + str(parameters.dataset_version) + "/" + parameters.network_string + "/ROC_results.csv"
    np.savetxt(directory, to_write, delimiter=',')

    fig, ax = plt.subplots()
    plt.title("Operator Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(recall_array, precision_array)
    for i, v in enumerate(eps_parameters):
        ax.text(i, v + 25, "%d" % v, ha="center")
    plt.savefig("results/" + parameters.dataset_name + "/" + "v_" + str(parameters.dataset_version) + "/" + parameters.network_string + "/ROC_results.png")

    fig, ax = plt.subplots()
    plt.title("f1 Results")
    plt.xlabel("eps parameter")
    plt.ylabel("f1 score")
    plt.plot(f1_array)
    plt.xticks(range(len(eps_parameters)), [str(ep) for ep in eps_parameters])
    # ax.set_xticklabels([str(el) for el in eps_parameters[::2]])
    plt.savefig("results/" + parameters.dataset_name + "/" + "v_" + str(parameters.dataset_version) + "/" + parameters.network_string + "/f1_array.png")


    fig, ax = plt.subplots()
    plt.title("Operator Curve Objectwise")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(recall_array_objectwise, precision_array_objectwise)
    for i, v in enumerate(eps_parameters):
        ax.text(i, v + 25, "%d" % v, ha="center")
    plt.savefig("results/" + parameters.dataset_name + "/" + "v_" + str(parameters.dataset_version) + "/" + parameters.network_string + "/ROC_results_object_wise.png")

    fig, ax = plt.subplots()
    plt.title("f1 Results Objectwise")
    plt.xlabel("eps parameter")
    plt.ylabel("f1 score")
    plt.plot(f1_array_objectwise)
    plt.xticks(range(len(eps_parameters)), [str(ep) for ep in eps_parameters])
    # ax.set_xticklabels([str(el) for el in eps_parameters[::2]])
    plt.savefig("results/" + parameters.dataset_name + "/" + "v_" + str(parameters.dataset_version) + "/" + parameters.network_string + "/f1_array_objectwise.png")


    fig, ax = plt.subplots()
    plt.title("Ra Score")
    plt.xlabel("eps parameter")
    plt.ylabel("Ra")
    plt.plot(Ra_arary)
    plt.xticks(range(len(eps_parameters)), [str(ep) for ep in eps_parameters])
    # ax.set_xticklabels([str(el) for el in eps_parameters[::2]])
    plt.savefig("results/" + parameters.dataset_name + "/" + "v_" + str(parameters.dataset_version) + "/" + parameters.network_string + "/ra_score.png")

    # plt.show()


def read_plots(dataset_name, dataset_version=[0, 1, 2]):
    a = [0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10, 15]
    # b = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3]

    eps_rad = [str(a[i]) for i in range(len(a))]
    dataset_version = [0, 1, 2]
    network_strings = ['unet_2_channels__64_dims__wz_64',
                       'unet_2_channels__64_dims__wz_64_offset',
                       'unet_double_multi_2_channels__64_dims__wz_64',
                       'METHOD_double']

    legend_strings = ['2 Nets & 12 embeddings',
                      '2 Nets & center Regression',
                      'Multi Task Learning & center regression',
                      'Proposed Method & center regression']
    # network_strings = network_strings[0:2]
    # legend_strings = legend_strings[0:2]
    data = {}
    for network in network_strings:
        for dataset_v in dataset_version:
            directory = "results/" + dataset_name + "/" + "v_" + str(dataset_v) + "/" + network + "/ROC_results.csv"
            print("reading " + directory)
            results = np.loadtxt(directory, delimiter=',')
            if(network in data.keys()):
                data[network] += results / len(dataset_version)
            else:
                data[network] = results / len(dataset_version)

    fig, ax = plt.subplots()
    plt.title("f1 Results")
    plt.ylabel("f1 score")

    plt.xlabel("eps paramaeter")
    plt.xticks(np.arange(len(eps_rad)),
               eps_rad)

    print("f1 array")
    for net in data.keys():
        print(net, data[net][2, :].max())
        plt.plot(data[net][2, :])
    plt.legend(legend_strings)

    plt.savefig("results/" + dataset_name + "/f1_array_pixelwise.png")


    fig, ax = plt.subplots()
    plt.title("f1 Results Objectwise")
    plt.ylabel("f1 score")

    plt.xlabel("eps paramaeter")
    plt.xticks(np.arange(len(eps_rad)),
               eps_rad)

    for net in data.keys():
        plt.plot(data[net][5, :])
    plt.legend(legend_strings)

    plt.savefig("results/" + dataset_name + "/f1_array_objectwise.png")

    fig, ax = plt.subplots()
    plt.title("Operator Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    for net in data.keys():
        plt.plot(data[net][1, :], data[net][0, :])
    plt.legend(legend_strings)

    plt.savefig("results/" + dataset_name + "/ROC_curve.png")

    fig, ax = plt.subplots()
    plt.title("Ra Score")
    plt.ylabel("Ra Score")

    print("Ra Score")
    for net in data.keys():
        print(net, 3 *  data[net][6, :].max())
        plt.plot(3 * data[net][6, :])
    plt.legend(legend_strings)

    plt.xlabel("eps paramaeter")
    plt.xticks(np.arange(len(eps_rad)),
               eps_rad)

    plt.savefig("results/" + dataset_name + "/ra_curve.png")


    plt.show()

import torch
def show_side_results(dataset_name, dataset_version=0):
    network_strings = ['unet_2_channels__64_dims__wz_64',
                       'unet_2_channels__64_dims__wz_64_offset',
                        'METHOD_double']

    string_of_names = ['Gt', 'U Net 12 Embs', 'Multi Offset', 'Proposed']


    data = {}
    counter = 1
    for network in network_strings:
        directory = "results/" + dataset_name + "/" + "v_" + str(dataset_version) + "/" + network + "/mini_inst"
        print("reading " + directory)
        results = tensors_io.load_volume(directory)
        data[counter] = results
        counter += 1
    data[0] = tensors_io.load_volume("results/" + dataset_name + "/" + "v_" + str(dataset_version) + "/gt")
    '''
    tensors_io.save_subplots_compare(data[0], data[1], data[2], data[3], "RESULTS_COMPARISON", string_of_names=string_of_names)

    og = tensors_io.load_volume("results/" + dataset_name + "/" + "v_" + str(dataset_version) + "/original")
    og = torch.transpose(og, 3, 2)
    tensors_io.save_subvolume(torch.transpose(data[0], 3, 2), 'compare/gt')
    tensors_io.save_subvolume(og, 'compare/side_og')
    tensors_io.save_subvolume(torch.transpose(data[1], 3, 2), 'compare/side_unet')
    tensors_io.save_subvolume(torch.transpose(data[2], 3, 2), 'compare/side_unet_of')
    tensors_io.save_subvolume(torch.transpose(data[3], 3, 2), 'compare/side_proposed')
    '''
    multi = tensors_io.load_volume("results/" + dataset_name + "/" + "v_" + str(dataset_version) + "/unet_double_multi_2_channels__64_dims__wz_64/mini_inst")
    tensors_io.save_subvolume(torch.transpose(multi, 1, 3), 'compare/multi')
