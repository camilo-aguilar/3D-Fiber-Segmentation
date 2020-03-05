from codes.models.encoder_decoder.unet_single import UNet
from codes.test_routines import test_net
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--network", type=str, default='unet',
                    help="select name of parameter file")

parser.add_argument("-test_dir", "--dir_test", type=str,
                    help="select directory for testing data")

parser.add_argument("-res_dir", "--dir_results", type=str,
                    help="select directory for saving results")

args = parser.parse_args()

testing_dir = "/Storage/DATASETS/Fibers/Tomas/labeled_fibers/2016_r/HR3_1/HR3_1"#parser.dir_test

network_type = args.network

if(args.dir_results is None):
    data_name = testing_dir.split('/')[-1]
    dir_results = './results/' + data_name
else:
    dir_results = parser.dir_results

dir_results = dir_results + '_' + network_type
if not os.path.isdir(dir_results):
    os.mkdir(dir_results)


net = UNet(1, 2)
results = train_semantic_segmentation_net(net, testing_dir, dir_results)