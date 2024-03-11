# from codes.train_routines import train_semantic_segmentation_net, train_instance_segmentation_net, train_multitask_loss_net
from codes.test_routines import quick_seg_inst_test, test_semantic_w_instance
from parameters import training_parameters, parse_arguments, dataset_specific_parameters
# from parameters import parse_arguments, dataset_specific_parameters
# from codes.utils.resample import resample
# from codes.utils.batch_testing import batch_test, read_plots, show_side_results

#############################################################   Networks   ######################################################################################
network_list = ["unet", "rnet", "dlv3_net", "unet_double", "unet_double_multi", "unet_double_multi_learned_center", "unet_double_bandwidth", "unet_double_multi_fixed"]

#############################################################   Datasets   ######################################################################################
# datasets = ["2016_r", "AFRL", "Sangids", "Nuclei", "2016_s"]
datasets = ["2016_s", "2016_r", "Sangids", "AFRL", "voids"]


args = parse_arguments(network_list, datasets)
args = dataset_specific_parameters(args)
parameters = training_parameters(args)
# parameters.not_so_big = False


#############################################################   TEST    ######################################################################################


## DEFINE PATH TO DATASET. See Folder 'sample_volume' for the format to expect
parameters.testing_dir = 'sample_volume'
parameters.testing_mask = None  # 'sample_labels'


parameters.uint_16 = True  # is data in uint16  format ?
parameters.save_side = 1 # save side view?

quick_test = True
if quick_test:
    # quick test: do a small subvolume to see if it works
    args.start_point = [100, 100, 100] # starting coordinates
    mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1, ins_f1_object_wise, Ra = quick_seg_inst_test(parameters, start_point=args.start_point)
else:
    # large test tiles all volume into small cubes, perform inference, and merges fiber by overlapping results
    mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1 = test_semantic_w_instance(parameters, length=128)

final_clusters = None
parameters.save_quick_results(mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1, final_clusters=final_clusters)