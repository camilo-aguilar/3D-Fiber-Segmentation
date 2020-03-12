from codes.train_routines import train_semantic_segmentation_net, train_instance_segmentation_net, train_multitask_loss_net
from codes.models.fcn_tomasz.RNet_model import RNet_model as RNet
from codes.models.encoder_decoder.unet_double import UNet_double
from codes.models.encoder_decoder.unet_single import UNet
from codes.models.deeplabv3_plus.deeplabv3 import DeepLabV3_single
from codes.test_routines import test_segmentation, quick_seg_inst_test
import codes.utils.loss_functions as loss_fns
import codes.utils.tensors_io as tensors_io
import argparse
import torch
import os

from sklearn.cluster import DBSCAN, OPTICS


class training_parameters:
    def __init__(self, args):
        networks = ["unet", "rnet", "dlv3_net", "unet_double", "unet_double_multi", "unet_double_multi_learned_center", "unet_double_bandwidth"]
        self.network_name = networks[args.network]
        # IO Parameters
        self.scale_p = 1
        self.uint_16 = True
        self.cleaning = True

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
            self.net = DeepLabV3_single(self.intput_channels, self.n_classes)
            self.net_i = DeepLabV3_single(self.intput_channels, self.n_embeddings)
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
            self.min_samples_param = 15
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

    def save_quick_results(self, mini_volume, mini_seg, mini_inst, mini_gt, seg_f1, inst_f1, masks=None, results_directory="results", dataset_name="mini"):
        path = results_directory + "/" + self.network_string
        if not os.path.isdir(path):
            os.mkdir(path)
        tensors_io.save_subvolume_instances(mini_volume, mini_seg.cpu() + 2 * (mini_gt > 0).long().cpu(), path + "/" + dataset_name + "_seg_" + str(seg_f1))
        tensors_io.save_subvolume_instances(mini_volume, mini_inst, path + "/" + dataset_name + "_inst_" + str(inst_f1))
        tensors_io.save_subvolume_instances(mini_volume, mini_gt, path + "/" + dataset_name + "_gt")


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str, default='test',
                    help="[train/test/test_s/both/quick]")

parser.add_argument("-n", "--network", type=int, default=0,
                    help="[0: unet 1: rnet 2: ensemble]")

parser.add_argument("-d", "--device", type=int, default=0,
                    help="[0: cuda:0, 1: cuda:1]")
parser.add_argument("-l", "--loss", type=int, default=0,
                    help="[0: embedded, 1: offset]")

parser.add_argument("-bg", "--debug", type=bool, default=False,
                    help="[False, True]")
parser.add_argument("-pt", "--pre_trained", type=int, default=0,
                    help="[0: False, 1: True]")

args = parser.parse_args()


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

# testing_dir = training_dir[0]
# testing_mask = training_masks[0]

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
        mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1 = quick_seg_inst_test(parameters, testing_dir, testing_mask)
        parameters.save_quick_results(mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1)

    else:
        print("-m: train/test/both")
else:
    if(args.mode == "train"):
        train_multitask_loss_net(parameters, training_dir, training_masks)
    elif(args.mode == "quick"):
        mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1 = quick_seg_inst_test(parameters, testing_dir, testing_mask)
        parameters.save_quick_results(mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1)




