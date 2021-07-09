from codes.models.fcn_tomasz.RNet_model import RNet_model as RNet
from codes.models.encoder_decoder.unet_double import UNet_double
from codes.models.encoder_decoder.unet_single import UNet
# from codes.models.DeepLabV3.deeplabv3_3d import DeepLabV3_Single
import codes.utils.loss_functions as loss_fns
import codes.utils.tensors_io as tensors_io
from sklearn.cluster import DBSCAN  # , OPTICS
from collections import defaultdict
import argparse
import torch
import os

'''  #############################################################   PARAMETERS   ######################################################################################
     #############################################################   PARAMETERS   ######################################################################################
'''


class training_parameters:
    def __init__(self, args):
        self.train_dataset_name = args.network_list[args.train_dataset_number]
        self.train_dataset_number = args.train_dataset_number
        self.training_numbers = defaultdict(list)
        self.dataset_name = args.dataset_name
        self.dataset_version = args.dataset_version
        self.network_name = args.network_list[args.network]
        self.network_number = args.network
        # IO Parameters
        self.scale_p = args.scale_p
        self.uint_16 = args.uint16
        self.labels_in_h5 = args.labels_in_h5
        self.cleaning = args.cleaning
        self.cleaning_sangids = args.cleaning_sangids

        # Network parameters
        self.n_classes = args.n_classes
        self.n_embeddings = args.n_embeddings
        self.intput_channels = 1
        self.ndims = args.n_filters

        # Network marks
        self.return_marks = args.return_marks

        # Training Parameters
        self.epochs = 100000
        self.epochs_instance = 10000
        self.batch_size = 1
        self.cube_size = args.window_size
        self.lr = 0.0001
        self.criterion_s = loss_fns.dice_loss
        if(args.loss == 0):
            self.criterion_i = loss_fns.embedded_loss
            self.offset_clustering = False
        else:
            self.criterion_i = loss_fns.coordinate_loss
            self.offset_clustering = True

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
        # self.offset_clustering = False

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
        self.net_weights_dir.append('info_files/' + self.network_string + '_{}_embeddings_inst.pth'.format(self.n_embeddings))
        self.net_train_dict_dir = 'info_files/' + self.network_string + 'train_info.npy'
        # Network
        if(self.network_name == "unet"):
            self.net = UNet(self.intput_channels, self.n_classes, self.ndims)
            self.net_i = UNet(self.intput_channels, self.n_embeddings, self.ndims)
        elif(self.network_name == "rnet"):
            self.net = RNet(self.intput_channels, self.n_classes, self.ndims)
            self.net_i = RNet(self.intput_channels, self.n_embeddings, self.ndims)
        elif(self.network_name == "dlv3_net"):
            # self.net = DeepLabV3_Single(self.intput_channels, self.n_classes)
            # self.net_i = DeepLabV3_Single(self.intput_channels, self.n_embeddings)
            print("pass")
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
            self.eps_param = 1.5
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
            self.eps_param = 2
            self.debug_cluster_unet_double = True
            print("============================================================")
            print("==================     Alert Unet Double Deebug   ===========")
            print("============================================================")
        else:
            self.debug_cluster_unet_double = False

        self.debug_display_cluster = True
        self.save_side = args.save_side
        # Network weights IO
        self.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

        if(args.pre_trained or args.mode == "test" or args.mode == "quick" or args.mode == "batch" or args.mode == "custom"):
            print("Loading pre-trained-weights")
            print("at {}".format(self.net_weights_dir[0]))
            try:
                self.net.load_state_dict(torch.load(self.net_weights_dir[0], map_location=('cuda:' + str(args.device) if torch.cuda.is_available() else "cpu")))
                if(self.net_i is not None):
                    print("Loading pre-trained-weights")
                    print("at {}".format(self.net_weights_dir[1]))
                    self.net_i.load_state_dict(torch.load(self.net_weights_dir[1], map_location=('cuda:' + str(args.device) if torch.cuda.is_available() else "cpu")))
            except:
                print("~~~~~~~~~~~~~~~~~Weights not found~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("~~~~~~~~~~~~~~~~~Weights not found~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("~~~~~~~~~~~~~~~~~Weights not found~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Loading Training Dictionary")
            try:
                self.training_numbers = np.load(self.net_train_dict_dir)
                print("Training Dictionary Loaded")
            except:
                print("Training Dictionary not found")

        self.training_dir = args.training_dir
        self.training_masks = args.training_masks
        self.testing_dir = args.testing_dir
        self.testing_mask = args.testing_mask

        ##################  object parameters  ##################
        self.mpp_curvature = args.mpp_curvature
        self.mpp_iterations = args.mpp_iterations
        self.mpp_min_r = args.mpp_min_r
        self.mpp_max_r = args.mpp_max_r

        self.mpp_min_l = args.mpp_min_l
        self.mpp_max_l = args.mpp_max_l

        self.mpp_min_t = args.mpp_min_t
        self.mpp_max_t = args.mpp_max_t

        self.mpp_min_p = args.mpp_min_p
        self.mpp_max_p = args.mpp_max_p

        self.mpp_T = args.mpp_T
        self.mpp_Alpha = args.mpp_Alpha
        self.mpp_Threshold_Battacharya = args.mpp_Threshold_Battacharya

        self.mpp_T_ov = args.mpp_T_ov

        self.Vo_t = args.Vo_t
        ##################  object parameters  ##################

    def save_segmentation_results(self, data_volume, results, precision, recall, f1, results_directory="results", dataset_name="segmentation"):
        path = results_directory + "/" + self.network_string
        if not os.path.isdir(path):
            os.mkdir(path)
        tensors_io.save_subvolume_instances(data_volume, results, path + "/" + dataset_name + "_seg_results")

        file1 = open(path + "/" + dataset_name + "_seg_results.txt", "a")
        str1 = dataset_name + ": precision: {}, recall: {}, f1: {}".format(precision, recall, f1)
        file1.write(str1)
        file1.close()

    def save_quick_results(self, mini_volume, mini_seg, mini_inst, mini_gt, seg_eval, inst_eval, masks=None, final_clusters=None, results_directory="results", dataset_name="mini"):
        if(self.not_so_big):
            self.dataset_version = 4
        if(self.debug_cluster_unet_double):
            if(self.network_number == 4):
                self.network_string = "METHOD"
            else:
                self.network_string = "METHOD_double"

            if(self.train_dataset_number > 0):
                self.network_string = self.network_string + "_t_dt_" + str(self.train_dataset_number)
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
        tensors_io.save_volume_h5(mini_inst[0, 0, ...].cpu(), name='fibers_only', directory=path + "/h5_files")
        tensors_io.save_subvolume_instances(mini_volume, mini_inst, path + "/" + dataset_name + "_inst")
        tensors_io.save_subvolume(mini_volume, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/original")
        tensors_io.save_subvolume_instances(mini_volume, mini_gt, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/gt")
        tensors_io.save_subvolume_instances(mini_volume * 0, mini_gt, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/gt_only")

        if(self.save_side):
            tensors_io.save_subvolume_instances_side(mini_volume, mini_gt * 0, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/original_side1", top=1)
            tensors_io.save_subvolume_instances_side(mini_volume, mini_gt * 0, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/original_side2", top=2)
            tensors_io.save_subvolume_instances_side(mini_volume, mini_gt, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/gt_side1", top=1)
            tensors_io.save_subvolume_instances_side(mini_volume, mini_gt, results_directory + "/" + self.dataset_name + "/" + "v_" + str(self.dataset_version) + "/gt_side2", top=2)

        tensors_io.save_subvolume_instances(mini_volume, mini_seg.cpu().long().cpu(), path + "/" + dataset_name + "_seg")
        tensors_io.save_subvolume_instances(mini_volume * 0, mini_seg.cpu(), path + "/" + dataset_name + "_seg_only")
        tensors_io.save_subvolume_instances(mini_volume * 0, mini_inst, path + "/" + dataset_name + "_inst_only")
        if(final_clusters is not None):
            tensors_io.save_subvolume_instances(mini_volume, final_clusters, path + "/" + dataset_name + "_clusters")
            tensors_io.save_subvolume_instances(mini_volume * 0, final_clusters, path + "/" + dataset_name + "_clusters_only")
            tensors_io.save_volume_h5((mini_volume[0, 0, ...] * 255).cpu(), name='og_im', directory=path + "/h5_files")
            tensors_io.save_volume_h5(final_clusters[0, 0, ...].cpu(), name='cluster_only', directory=path + "/h5_files")
            tensors_io.save_volume_h5(mini_gt[0, 0, ...].cpu(), name='mask_only', directory=path + "/h5_files")


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


'''  #############################################################   ARGUMENTS   ######################################################################################
     #############################################################   ARGUMENTS   ######################################################################################
'''


def parse_arguments(network_list, datasets):
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

    parser.add_argument("-w_z", "--window_size", type=int, default=64,
                        help="Window Size")

    parser.add_argument("-s_s", "--subsample", type=str, default='.',
                        help="directory")

    parser.add_argument("-n_e", "--n_embeddings", type=int, default=12,
                        help="directory")

    parser.add_argument("-n_c", "--n_classes", type=int, default=2,
                        help="directory")

    parser.add_argument("-n_filters", "--n_filters", type=int, default=64,
                        help="directory")

    args = parser.parse_args()

    if(args.mode == "train"):
        if(args.dataset_number > 0):
            args.train_dataset_number = args.dataset_number
        if(args.train_dataset_number > 0):
            args.dataset_number = args.train_dataset_number

    args.dataset_name = datasets[args.dataset_number]
    args.network_list = network_list
    args.labels_in_h5 = False
    return args


'''  #############################################################   DATASETS   ######################################################################################
     #############################################################   DATASETS   ######################################################################################
'''


def dataset_specific_parameters(args):
    args.cleaning_sangids = False
    args.cleaning = True
    args.save_side = False

    args.training_dir = None
    args.training_masks = None
    args.testing_dir = None
    args.testing_mask = None

    args.return_marks = False
    ##################  object parameters  ##################
    args.Vo_t = 0.45
    args.mpp_iterations = 100000
    args.mpp_min_r = 1
    args.mpp_max_r = 2

    args.mpp_min_l = 5
    args.mpp_max_l = 200

    args.mpp_min_t = 0
    args.mpp_max_t = 180

    args.mpp_min_p = 0
    args.mpp_max_p = 180

    args.mpp_T = 1
    args.mpp_Alpha = 10 ** (-7.0 / float(args.mpp_iterations))
    args.mpp_Threshold_Battacharya = 0.05

    args.mpp_curvature = 2

    args.mpp_T_ov = 0.2

    ##################  object parameters  ##################
    if(args.debug_cluster == 1):
        args.debug = 1

    if(args.dataset_name == "2016_s"):
        args.uint16 = True
        args.scale_p = 1
        args.mpp_min_r = 1.9
        args.mpp_max_r = 2

        if(args.dataset_version == 0):
            # start_point = [0, 0, 40]
            args.start_point = [100, 100, 100]
        elif(args.dataset_version == 1):
            args.start_point = [64, 64, 100]
        else:
            args.start_point = [128, 128, 100]

        args.training_dir = ["sample_volume"]
        args.training_masks = ["sample_labels"]
        args.testing_dir = "sample_volume"
        args.testing_mask = "sample_labels"

    return args
