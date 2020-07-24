from codes.train_routines import train_semantic_segmentation_net, train_instance_segmentation_net, train_multitask_loss_net
from codes.test_routines import quick_seg_inst_test, test_semantic_w_instance
from parameters import training_parameters, parse_arguments, dataset_specific_parameters
from codes.utils.resample import resample
from codes.utils.batch_testing import batch_test, read_plots, show_side_results

#############################################################   Networks   ######################################################################################
network_list = ["unet", "rnet", "dlv3_net", "unet_double", "unet_double_multi", "unet_double_multi_learned_center", "unet_double_bandwidth", "unet_double_multi_fixed"]

#############################################################   Datasets   ######################################################################################
# datasets = ["2016_r", "AFRL", "Sangids", "Nuclei", "2016_s"]
datasets = ["2016_s", "2016_r", "Sangids", "AFRL", "voids"]


args = parse_arguments(network_list, datasets)
args = dataset_specific_parameters(args)
parameters = training_parameters(args)
parameters.not_so_big = False

#############################################################   TRAIN   ######################################################################################
if(args.network < 3):
    if(args.mode == "train_s"):
        train_semantic_segmentation_net(parameters, args.training_dir, args.training_masks)
    elif(args.mode == "train_i"):
        train_instance_segmentation_net(parameters, args.training_dir, args.training_masks)
    elif(args.mode == "train"):
        train_semantic_segmentation_net(parameters, args.training_dir, args.training_masks)
        train_instance_segmentation_net(parameters, args.training_dir, args.training_masks)
else:
    if(args.mode == "train"):
        train_multitask_loss_net(parameters, args.training_dir, args.training_masks)

print(args.subsample)
####################################
if(args.subsample != '.'):
    resample(parameters, resample_masks=2, factor=2, directory=args.subsample)
    exit()
####################################

#############################################################   TEST    ######################################################################################
if(args.mode == "quick"):
    if(args.debug):
        mini_V, final_pred, final_fibers, final_clusters, mini_gt, seg_f1, ins_f1, ins_f1_object_wise, Ra = quick_seg_inst_test(parameters, start_point=args.start_point)
    else:
        mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1, ins_f1_object_wise, Ra = quick_seg_inst_test(parameters, start_point=args.start_point)
        final_clusters = None
    parameters.save_quick_results(mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1, final_clusters=final_clusters)
elif(args.mode == "test"):
    parameters.save_side = 1
    parameters.not_so_big = True
    # parameters.testing_mask = None
    # AFRL data length = 320
    # AFRL data length = 128
    mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1 = test_semantic_w_instance(parameters, length=128)
    final_clusters = None
    parameters.save_quick_results(mini_V, final_pred, final_fibers, mini_gt, seg_f1, ins_f1, final_clusters=final_clusters)
elif(args.mode == "resample_v"):
    resample(parameters, resample_masks=0, factor=2)
elif(args.mode == "resample_m"):
    resample(parameters, resample_masks=1, factor=2)
elif(args.mode == "resample_r"):
    resample(parameters, resample_masks=2, factor=2)

elif(args.mode == "batch"):
    batch_test(parameters, args)
elif(args.mode == "plots"):
    read_plots(parameters.dataset_name)

elif(args.mode == "custom"):
    network_architectures = [-2]

    datasets_options = [2]
    datasets_versions = [0, 1, 2]

    for n in network_architectures:
        for dt in datasets_options:
            for dt_v in datasets_versions:
                if(n == -1):
                    # debug cluster multi
                    args.debug_cluster = 1
                    args.network = 4
                elif(n == -2):
                    # debug cluster double
                    args.debug_cluster = 1
                    args.network = 0
                    args.loss = 1
                elif(n == 0.5):
                    # debug cluster offset
                    args.loss = 1
                else:
                    args.network = n

                args.dataset_number = dt
                args.dataset_version = dt_v
                args.dataset_name = datasets[args.dataset_number]
                args = dataset_specific_parameters(args)
                parameters = training_parameters(args)
                batch_test(parameters, args)

elif(args.mode == "compare"):
    show_side_results(parameters.dataset_name, parameters.dataset_version)

else:
    print("Option {} not understood".format(args.mode))
