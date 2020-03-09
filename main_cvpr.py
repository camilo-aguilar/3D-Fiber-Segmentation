from codes.train_routines import train_semantic_segmentation_net, train_instance_segmentation_net, train_multitask_loss_net
from codes.models.fcn_tomasz.RNet_model import RNet_model as RNet
from codes.models.encoder_decoder.unet_double import UNet_double
from codes.models.encoder_decoder.unet_single import UNet
from codes.test_routines import test_segmentation, quick_seg_inst_test
import codes.utils.loss_functions as loss_fns
import codes.utils.tensors_io as tensors_io
import argparse
import torch
import os

from sklearn.cluster import DBSCAN


class training_parameters:
    def __init__(self, network_name="unet"):
        networks = ["unet", "rnet", "unet_double"]
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
        self.epochs = 100
        self.epochs_instance = 1000
        self.batch_size = 20
        self.cube_size = 64
        self.lr = 0.0001
        self.criterion_s = loss_fns.dice_loss
        self.criterion_i = loss_fns.embedded_loss

        # Instance Segmentation Training Parameters
        self.delta_v = 0.2
        self.delta_d = 5

        self.alpha_i = 2
        self.beta_i = 2
        self.gamma_i = 0.0000001

        # Instance Segmentation Inference parameters
        self.eps_param = 0.2
        self.min_samples_param = 10
        self.clustering = DBSCAN(eps=self.eps_param, min_samples=self.min_samples_param).fit_predict

        # Volume split/merge
        self.percent_overlap = 0.2

        # Network weights IO
        self.pre_trained = True
        self.net_weights_dir = None
        self.device = torch.device("cuda:" + str(args.device))

        self.network_string = self.network_name + '_{}_channels__{}_dims__wz_{}'.format(self.n_classes, self.ndims, self.cube_size)
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
        elif(self.network_name == "unet_double"):
            self.net = UNet_double(self.intput_channels, self.n_classes, self.n_embeddings, self.ndims)
        else:
            print("Select a correct network name")
            exit()
        # debug
        self.debug = True

    def save_segmentation_results(self, data_volume, results, precision, recall, f1, results_directory="results", dataset_name="segmentation"):
        path = results_directory + "/" + self.network_string
        if not os.path.isdir(path):
            os.mkdir(path)
        tensors_io.save_subvolume_instances(data_volume, result, path + "/" + dataset_name + "_seg_results")

        file1 = open(path + "/" + dataset_name + "_seg_results.txt", "a")
        str1 = dataset_name + ": precision: {}, recall: {}, f1: {}".format(precision, recall, f1)
        file1.write(str1)
        file1.close()


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str, default='test',
                    help="select name of parameter file")

parser.add_argument("-n", "--network", type=int, default=0,
                    help="select name of network")

parser.add_argument("-d", "--device", type=int, default=0,
                    help="select device")

args = parser.parse_args()


if(torch.cuda.device_count() == 1):
    training_dir = ["/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_1", "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_2"]
    training_masks = ["/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_1_anno", "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_2_anno"]

    testing_dir = "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_s/5.35/HR/HR_5.35p"
    testing_mask = "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_s/5.35/HR/HR_5.35p_anno"
else:
    training_dir = ["/pub2/aguilarh/DATASETS/Tomasz/HR3_1", "/pub2/aguilarh/DATASETS/Tomasz/HR3_2"]
    training_masks = ["/pub2/aguilarh/DATASETS/Tomasz/HR3_1_anno", "/pub2/aguilarh/DATASETS/Tomasz/HR3_2_anno"]

    testing_dir = "/pub2/aguilarh/DATASETS/Tomasz/HR_5.35p"
    testing_mask = "/pub2/aguilarh/DATASETS/Tomasz/HR_5.35p_anno"


parameters = training_parameters(args)
if(args.network < 2):
    if(args.mode == "train"):
        train_semantic_segmentation_net(parameters, training_dir, training_masks)
    elif(args.mode == "train_s"):
        train_instance_segmentation_net(parameters, training_dir, training_masks)
    elif(args.mode == "both"):
        train_semantic_segmentation_net(parameters, training_dir, training_masks)
        data_volume, result, precision, recall, f1 = test_segmentation(parameters, testing_dir, testing_mask)
        parameters.save_segmentation_results(data_volume, result, precision, recall, f1, dataset_name=testing_dir.split("/")[-1])

    elif(args.mode == 'test'):
        data_volume, result, precision, recall, f1 = test_segmentation(parameters, testing_dir, testing_mask)
        parameters.save_segmentation_results(data_volume, result, precision, recall, f1, dataset_name=testing_dir.split("/")[-1])
    elif(args.mode == "quick"):
        data_volume, semantic_seg, instance_seg = quick_seg_inst_test(parameters, testing_dir, testing_mask)
    else:
        print("-m: train/test/both")
else:
    if(args.mode == "train"):
        train_multitask_loss_net(parameters, training_dir, training_masks)
    elif(args.mode == "quick"):
        data_volume, semantic_seg, instance_seg = quick_seg_inst_test(parameters, testing_dir, testing_mask)
