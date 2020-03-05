from codes.models.encoder_decoder.unet_single import UNet
from codes.train_routines import train_semantic_segmentation_net
from codes.test_routines import test_segmentation
import codes.utils.loss_functions as loss_fns
import argparse
import torch
import os


class training_parameters:
    def __init__(self):
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
        self.batch_size = 20
        self.cube_size = 64
        self.lr = 0.0001
        self.criterion = loss_fns.dice_loss

        # Network weights IO
        self.pre_trained = False
        self.net_weights_dir = None
        self.device = None

        # Weights Directories
        self.net_weights_dir = ['info_files/unet_seg__{}_channels__{}_dims__wz_{}.pth'.format(self.n_classes, self.ndims, self.cube_size)]
        self.net_weights_dir.append('info_files/unet_inst__{}_embeddings__{}_dims__wz_{}.pth'.format(self.n_embeddings, self.ndims, self.cube_size))

        # Network
        self.net = UNet(self.intput_channels, self.n_classes, self.ndims)

        # debuf
        self.debug = True


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str, default='unet',
                    help="select name of parameter file")

args = parser.parse_args()


if(torch.cuda.device_count() == 1):
    training_dir = ["/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_1", "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_2"]
    training_masks = ["/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_1_anno", "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_2_anno"]

    testing_dir = "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_s/5.35/HR/HR_5.35p"
    testing_mask = "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_s/5.35/HR/HR_5.35p_anno"
else:
    training_dir = ["/pub2/aguilarh/DATASETS/Tomasz/HR3_1", "/pub2/aguilarh/DATASETS/Tomasz/HR3_2"]
    training_masks = ["/pub2/aguilarh/DATASETS/Tomasz/HR3_1_anno", "/pub2/aguilarh/DATASETS/Tomasz/HR3_2_anno"]

    testing_dir = training_dir[0] # "/pub2/aguilarh/DATASETS/Tomasz/HR_5.35p"
    testing_mask = training_masks[0] # "/pub2/aguilarh/DATASETS/Tomasz/HR_5.35p_anno"


parameters = training_parameters()
print(args.mode)
if(args.mode is "train"):
    train_semantic_segmentation_net(parameters, training_dir, training_masks)
elif(args.mode == 'test'):
    segmentation = test_segmentation(parameters, testing_dir, testing_mask)
elif(args.mode is "both"):
    train_semantic_segmentation_net(parameters, training_dir, training_masks)
    segmentation = test_segmentation(parameters, testing_dir, testing_mask)
else:
    print("-m: train/test/both")
