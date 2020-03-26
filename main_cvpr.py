from codes.train_routines import train_semantic_segmentation_net, train_instance_segmentation_net, train_multitask_loss_net
from codes.models.fcn_tomasz.RNet_model import RNet_model as RNet
from codes.models.encoder_decoder.unet_double import UNet_double
from codes.models.encoder_decoder.unet_single import UNet
# from codes.models.deeplabv3_plus.deeplabv3 import DeepLabV3_single
from codes.test_routines import test_segmentation, quick_seg_inst_test
import codes.utils.loss_functions as loss_fns
import codes.utils.tensors_io as tensors_io
import argparse
import torch
import os

from sklearn.cluster import DBSCAN  # , OPTICS


'''  #############################################################   PARAMETERS   ######################################################################################
     #############################################################   PARAMETERS   ######################################################################################
'''


class training_parameters:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.dataset_version = args.dataset_version
        self.network_name = args.network_list[args.network]
        # IO Parameters
        self.scale_p = args.scale_p
        self.uint_16 = args.uint16
        self.cleaning = args.cleaning
        self.cleaning_sangids = args.cleaning_sangids

        # Network parameters
        self.n_classes = 2
        self.n_embeddings = 12
        self.intput_channels = 1
        self.ndims = 64

        # Training Parameters
        self.epochs = 1000
        self.epochs_instance = 10000
        self.batch_size = 20
        self.cube_size = 64
        self.lr = 0.0001
        self.criterion_s = loss_fns.dice_loss
        if(args.loss == 0):
            self.criterion_i = loss_fns.embedded_loss
        else:
            self.criterion_i = loss_fns.coordinate_loss

        # Instance Segmentation Training Parameters
        self.delta_v = 0.2
        self.delta_d = 5

        self.alpha_i = 2
        self.beta_i = 2
        self.gamma_i = 0.0000001

        # Instance Segmentation Inference parameters
        self.eps_param = 0.4
        self.min_samples_param = 10
        self.clustering = DBSCAN(eps=self.eps_param, min_samples=self.min_samples_param).fit_predict
        self.offset_clustering = False

        # Double Net Parameters
        self.alpha_seg = 0.1
        self.alpha_emb = 2
        # Volume split/merge
        self.percent_overlap = 0.2

        self.network_string = self.network_name + '_{}_channels__{}_dims__wz_{}'.format(self.n_classes, self.ndims, self.cube_size)
        if(args.loss == 1):
            self.network_string = self.network_string + "_offset"
            self.n_embeddings = 3
            self.eps_param = 1

        if(args.train_dataset_number > 0):
            self.network_string = self.network_string + "_tdt_" + str(args.train_dataset_number)
        # Weights Directories
        self.net_weights_dir = ['info_files/' + self.network_string + '_seg.pth']
        self.net_weights_dir.append('info_files/' + self.network_string + '_inst.pth'.format(self.n_embeddings, self.ndims, self.cube_size))

        # Network
        if(self.network_name == "unet"):
            self.net = UNet(self.intput_channels, self.n_classes, self.ndims)
            self.net_i = UNet(self.intput_channels, self.n_embeddings, self.ndims)
        elif(self.network_name == "rnet"):
            self.net = RNet(self.intput_channels, self.n_classes, self.ndims)
            self.net_i = RNet(self.intput_channels, self.n_embeddings, self.ndims)
        elif(self.network_name == "dlv3_net"):
            print("In Developing but not quite ready yet. Tough life bro")
            exit()
            # self.net = DeepLabV3_single(self.intput_channels, self.n_classes)
            # self.net_i = DeepLabV3_single(self.intput_channels, self.n_embeddings)
        elif(self.network_name == "unet_double"):
            self.net = UNet_double(self.intput_channels, self.n_classes, self.n_embeddings, self.ndims)
            self.net_i = None
        elif(self.network_name == "unet_double_multi" or self.network_name == "unet_double_multi_learned_center"):
            self.net = UNet_double(self.intput_channels, self.n_classes + 1, 3 + 1, self.ndims)
            self.net_i = None
            if(self.network_name == "unet_double_multi"):
                self.criterion = loss_fns.multi_task_loss
            else:
                self.criterion = loss_fns.multi_task_loss_learned_center
            self.offset_clustering = True
            self.delta_v = 0.5
            self.eps_param = 0.7
            self.min_samples_param = 20
            self.clustering = DBSCAN(eps=self.eps_param, min_samples=self.min_samples_param).fit_predict
            self.n_embeddings = 3
        elif(self.network_name == "unet_double_multi_fixed"):
            self.net = UNet_double(self.intput_channels, self.n_classes, 3, self.ndims)
            self.net_i = None
            if(self.network_name == "unet_double_multi_fixed"):
                self.criterion = loss_fns.multi_task_fixed_loss
            self.offset_clustering = True
            self.delta_v = 0.5
            self.eps_param = 0.7
            self.min_samples_param = 20
            self.clustering = DBSCAN(eps=self.eps_param, min_samples=self.min_samples_param).fit_predict
            self.n_embeddings = 3
        elif(self.network_name == "unet_double_bandwidth"):
            self.net = UNet_double(self.intput_channels, 1, 3 + 1, self.ndims)
            self.net_i = None
            self.criterion = loss_fns.joint_spatial_bandwidth_loss
        else:
            print("Select a correct network name")
            exit()
        # debug
        self.debug = args.debug
        if(args.debug_cluster == 1):
            self.debug_cluster_unet_double = True
            print("============================================================")
            print("==================     Alert Unet Double Deebug   ===========")
            print("============================================================")
        else:
            self.debug_cluster_unet_double = False

        self.debug_display_cluster = True
        self.save_side = args.save_side
        # Network weights IO
        self.device = torch.device("cuda:" + str(args.device))

        if(args.pre_trained):
            print("Loading pre-trained-weights")
            try:
                self.net.load_state_dict(torch.load(self.net_weights_dir[0]))
                if(self.net_i is not None):
                    self.net_i.load_state_dict(torch.load(self.net_weights_dir[1]))
            except:
                print("Weights not found")

    def save_segmentation_results(self, data_volume, results, precision, recall, f1, results_directory="results", dataset_name="segmentation"):
        path = results_directory + "/" + self.network_string
        if not os.path.isdir(path):
            os.mkdir(path)
        tensors_io.save_subvolume_instances(data_volume, result, path + "/" + dataset_name + "_seg_results")

        file1 = open(path + "/" + dataset_name + "_seg_results.txt", "a")
        str1 = dataset_name + ": precision: {}, recall: {}, f1: {}".format(precision, recall, f1)
        file1.write(str1)
        file1.close()

    def save_quick_results(self, mini_volume, mini_seg, mini_inst, mini_gt, seg_eval, inst_eval, masks=None, final_clusters=None, results_directory="results", dataset_name="mini"):
        if(self.debug_cluster_unet_double):
            self.network_string = "METHOD"
        path = results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/" + self.network_string
        path_temp = ""
        for local_path in path.split("/"):
            path_temp = path_temp + local_path
            if not os.path.isdir(path_temp):
                os.mkdir(path_temp)
            path_temp = path_temp + "/"

        # mini_gt = 0 * mini_gt
        seg_f1 = int(seg_eval[2] * 1000)
        seg_f1 = float(seg_f1) / 1000

        inst_f1 = int(inst_eval[2] * 1000)
        inst_f1 = float(inst_f1) / 1000
        tensors_io.save_subvolume(mini_volume, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/original")
        tensors_io.save_subvolume_instances(mini_volume, mini_gt, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/gt")
        tensors_io.save_subvolume_instances(mini_volume * 0, mini_gt, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/gt_only")

        if(self.save_side):
            tensors_io.save_subvolume_instances_side(mini_volume, mini_gt * 0, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/original_side1", top=1)
            tensors_io.save_subvolume_instances_side(mini_volume, mini_gt * 0, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/original_side2", top=2)
            tensors_io.save_subvolume_instances_side(mini_volume, mini_gt, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/gt_side1", top=1)
            tensors_io.save_subvolume_instances_side(mini_volume, mini_gt, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/gt_side2", top=2)

        tensors_io.save_subvolume_instances(mini_volume, mini_seg.cpu().long().cpu(), path + "/" + dataset_name + "_seg_" + str(seg_f1))
        tensors_io.save_subvolume_instances(mini_volume * 0, mini_seg.cpu(), path + "/" + dataset_name + "_seg_only_" + str(seg_f1))
        tensors_io.save_subvolume_instances(mini_volume, mini_inst, path + "/" + dataset_name + "_inst_" + str(inst_f1))
        tensors_io.save_subvolume_instances(mini_volume * 0, mini_inst, path + "/" + dataset_name + "_inst_only_" + str(inst_f1))
        if(final_clusters is not None):
            tensors_io.save_subvolume_instances(mini_volume, final_clusters, path + "/" + dataset_name + "_clusters")
            tensors_io.save_subvolume_instances(mini_volume * 0, final_clusters, path + "/" + dataset_name + "_clusters_only")

        seg_results = []
        inst_results = []
        for i in range(3):
            v = int(seg_eval[i] * 1000)
            seg_results.append(float(v) / 1000)

            v = int(inst_eval[i] * 1000)
            inst_results.append(float(v) / 1000)

        file1 = open(path + "/results.txt", "w")
        str1 = "{},{},{}\n".format(seg_results[0], seg_results[1], seg_results[2])
        file1.write(str1)
        str1 = "{},{},{}\n".format(inst_results[0], inst_results[1], inst_results[2])
        file1.write(str1)
        file1.close()

        if(self.save_side):
            tensors_io.save_subvolume_instances_side(mini_volume, mini_inst, path + "/" + dataset_name + "_inst_side1", top=1)
            tensors_io.save_subvolume_instances_side(mini_volume, mini_inst, path + "/" + dataset_name + "_inst_side2", top=1)


'''  #############################################################   Networks   ######################################################################################
     #############################################################   Datasets   ######################################################################################
'''

network_list = ["unet", "rnet", "dlv3_net", "unet_double", "unet_double_multi", "unet_double_multi_learned_center", "unet_double_bandwidth", "unet_double_multi_fixed"]
datasets = ["2016_r", "AFRL", "Sangids", "Tianyu_syn"]
'''  #############################################################   ARGUMENTS   ######################################################################################
     #############################################################   ARGUMENTS   ######################################################################################
'''

help_network_str = "["
for i in range(len(network_list)):
    help_network_str += str(i) + ":" + network_list[i] + ", "

help_datasets_str = "["
for i in range(len(datasets)):
    help_datasets_str += str(i) + ":" + datasets[i] + ", "

help_network_str = help_network_str[:-2] + "]"
help_datasets_str = help_datasets_str[:-2] + "]"
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str, default='quick',
                    help="[train/test/test_s/both/quick]")

parser.add_argument("-n", "--network", type=int, default=4,
                    help=help_network_str)

parser.add_argument("-d", "--device", type=int, default=0,
                    help="[0: cuda:0, 1: cuda:1]")
parser.add_argument("-l", "--loss", type=int, default=0,
                    help="[0: embedded, 1: offset]")

parser.add_argument("-dg", "--debug", type=int, default=0,
                    help="[0: Flase, 1:True]")
parser.add_argument("-pt", "--pre_trained", type=int, default=0,
                    help="[0: False, 1: True]")
parser.add_argument("-dt", "--dataset_number", type=int, default=0,
                    help=help_datasets_str)

parser.add_argument("-t_dt", "--train_dataset_number", type=int, default=0,
                    help=help_datasets_str)

parser.add_argument("-dt_v", "--dataset_version", type=int, default=0,
                    help=help_datasets_str)

parser.add_argument("-dg_c", "--debug_cluster", type=int, default=0,
                    help=help_datasets_str)

args = parser.parse_args()

'''  #############################################################   DATASETS   ######################################################################################
     #############################################################   DATASETS   ######################################################################################
'''
args.dataset_name = datasets[args.dataset_number]
args.network_list = network_list
args.cleaning_sangids = False
args.cleaning = True
args.save_side = False

if(args.dataset_name == "2016_r"):
    args.uint16 = True
    args.scale_p = 1

    if(args.dataset_version == 0):
        start_point = [0, 0, 40]
        start_point = [100, 100, 50]
    elif(args.dataset_version == 1):
        start_point = [64, 64, 40]
    else:
        start_point = [128, 128, 40]

    if(torch.cuda.device_count() == 1):
        training_dir = ["/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_1"]
        training_masks = ["/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_1_anno"]
        testing_dir = "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_2"  # "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_s/5.35/HR/HR_5.35p"
        testing_mask = "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_2_anno"  # "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_s/5.35/HR/HR_5.35p_anno"
    else:
        training_dir = ["/pub2/aguilarh/DATASETS/Tomasz/HR3_1"]
        training_masks = ["/pub2/aguilarh/DATASETS/Tomasz/HR3_1_anno"]
        testing_dir = "/pub2/aguilarh/DATASETS/Tomasz/HR3_2"  # "/pub2/aguilarh/DATASETS/Tomasz/HR_5.35p"
        testing_mask = "/pub2/aguilarh/DATASETS/Tomasz/HR3_2_anno"  # "/pub2/aguilarh/DATASETS/Tomasz/HR_5.35p_anno"

if(args.dataset_name == "2016_r2"):
    args.uint16 = True
    args.scale_p = 1
    start_point = [30, 30, 50]
    if(torch.cuda.device_count() == 1):
        training_dir = ["/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_1"]
        training_masks = ["/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_1_anno"]
        testing_dir = "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_2"  # "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_s/5.35/HR/HR_5.35p"
        testing_mask = "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_2_anno"  # "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_s/5.35/HR/HR_5.35p_anno"
    else:
        training_dir = ["/pub2/aguilarh/DATASETS/Tomasz/HR3_1"]
        training_masks = ["/pub2/aguilarh/DATASETS/Tomasz/HR3_1_anno"]
        testing_dir = "/pub2/aguilarh/DATASETS/Tomasz/HR3_2"  # "/pub2/aguilarh/DATASETS/Tomasz/HR_5.35p"
        testing_mask = "/pub2/aguilarh/DATASETS/Tomasz/HR3_2_anno"  # "/pub2/aguilarh/DATASETS/Tomasz/HR_5.35p_anno"

elif(args.dataset_name == "AFRL"):
    args.uint16 = False
    args.scale_p = 2
    if(args.dataset_version == 0):
        # start_point = [100, 100, 50]
        start_point = [0, 0, 0]
    elif(args.dataset_version == 1):
        start_point = [100, 100, 50]
    else:
        start_point = [200, 200, 200]

    if(torch.cuda.device_count() == 1):
        testing_dir = "/Storage/DATASETS/GLOBUS/Fibers_voids2/recon_20141209_235811_127_1_InSituPyro_curedRT_HC_7_enhcont_cropped"
        testing_mask = None

elif(args.dataset_name == "Tianyu_syn"):
    args.uint16 = False
    args.scale_p = 1
    start_point = [0, 0, 0]
    testing_dir = "/Storage/DATASETS/Synthetic_Fiber_dataset/SynSeq/seq1"
    testing_mask = "/Storage/DATASETS/Synthetic_Fiber_dataset/SynSeq/seg1"

elif(args.dataset_name == "Sangids"):
    args.save_side = True
    args.uint16 = False
    args.scale_p = 2
    start_point = [0, 0, 0]
    args.cleaning_sangids = True
    args.cleaning = True
    if(args.network == 1):
        args.cleaning = False

    if(torch.cuda.device_count() == 2):
        training_dir = ["/pub2/aguilarh/DATASETS/Sangid_s_data_w_results/NewTrainData_Sep9/sV1/data", "/pub2/aguilarh/DATASETS/Sangid_s_data_w_results/NewTrainData_Sep9/sV2/data"]
        training_masks = ["/pub2/aguilarh/DATASETS/Sangid_s_data_w_results/NewTrainData_Sep9/sV1/fibers_uint16_sV1", "/pub2/aguilarh/DATASETS/Sangid_s_data_w_results/NewTrainData_Sep9/sV2/fibers_uint16_sV2"]
        testing_dir = "/pub2/aguilarh/DATASETS/Sangid_s_data_w_results/NewTrainData_Sep9/sV3/data"
        testing_mask = "/pub2/aguilarh/DATASETS/Sangid_s_data_w_results/NewTrainData_Sep9/sV3/fibers_uint16_sV3"
    else:
        training_dir = ["/Storage/DATASETS/Fibers/NewTrainData_Sep9/subV1/data", "/Storage/DATASETS/Fibers/NewTrainData_Sep9/subV2/data"]
        training_masks = ["/Storage/DATASETS/Fibers/NewTrainData_Sep9/subV1/fibers_uint16_sV1", "/Storage/DATASETS/Fibers/NewTrainData_Sep9/subV2/fibers_uint16_sV2"]
        testing_dir = "/Storage/DATASETS/Fibers/NewTrainData_Sep9/subV3/subV3"
        testing_mask = "/Storage/DATASETS/Fibers/NewTrainData_Sep9/subV3/fibers_uint16_sV3"

if(args.mode == "train"):
    args.train_dataset_number = args.dataset_number
# testing_dir = training_dir[0]
# testing_mask = training_masks[0]

'''  #############################################################   TRAIN   ######################################################################################
     #############################################################   TEST    ######################################################################################
'''
parameters = training_parameters(args)
if(args.network < 3):
    if(args.mode == "train"):
        train_semantic_segmentation_net(parameters, training_dir, training_masks)
    elif(args.mode == "train_s"):
        train_instance_segmentation_net(parameters, training_dir, training_masks)
    elif(args.mode == "both"):
        train_semantic_segmentation_net(parameters, training_dir, training_masks)
        train_instance_segmentation_net(parameters, training_dir, training_masks)
        data_volume, result, precision, recall, f1 = test_segmentation(parameters, testing_dir, testing_mask)
        parameters.save_segmentation_results(data_volume, result, precision, recall, f1, dataset_name=testing_dir.split("/")[-1])

    elif(args.mode == 'test'):
        data_volume, result, precision, recall, f1 = test_segmentation(parameters, testing_dir, testing_mask)
        parameters.save_segmentation_results(data_volume, result, precision, recall, f1, dataset_name=testing_dir.split("/")[-1])
    elif(args.mode == "quick"):
        mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1 = quick_seg_inst_test(parameters, testing_dir, testing_mask, start_point=start_point)
        parameters.save_quick_results(mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1)

    else:
        print("-m: train/test/both")
else:
    if(args.mode == "train"):
        train_multitask_loss_net(parameters, training_dir, training_masks)
    elif(args.mode == "quick"):
        if(args.debug):
            mini_V, final_pred, final_fibers, final_clusters, mini_gt, seg_f1, ins_f1 = quick_seg_inst_test(parameters, testing_dir, testing_mask, start_point=start_point)
        else:
            mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1 = quick_seg_inst_test(parameters, testing_dir, testing_mask, start_point=start_point)
            final_clusters = None
        parameters.save_quick_results(mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1, final_clusters=final_clusters)
