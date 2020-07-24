import codes.utils.tensors_io as tensors_io
import os


def resample(params_t, resample_masks=0, factor=2, directory='.'):
    if(directory != '.'):
        path_of_interest = directory + '/subsampled'
        resample_masks = 2
        masks = tensors_io.load_volume(directory, scale=2).unsqueeze(0)
        tensors_io.save_subvolume(masks, path_of_interest)
        exit()
    else:
        params_t.cleaning = False
        params_t.cleaning_sangids = False
        params_t.scale_p = factor
        if(resample_masks == 0):
            path_of_interest = "results/" + params_t.dataset_name + "/" + "v_" + str(params_t.dataset_version) + "/or_subsampled"
        elif(resample_masks == 1):
            path_of_interest = "results/" + params_t.dataset_name + "/" + "v_" + str(params_t.dataset_version) + "/gt_subsampled"
        elif(resample_masks == 2):
            if(params_t.debug_cluster_unet_double):
                params_t.network_string = "METHOD"
            path_of_interest = "results/" + params_t.dataset_name + "/" + "v_" + str(params_t.dataset_version) + "/" + params_t.network_string + "/mini_inst_only_subsampled"
            params_t.testing_mask = "results/" + params_t.dataset_name + "/" + "v_" + str(params_t.dataset_version) + "/" + params_t.network_string + "/mini_inst_only"
        else:
            path_of_interest = ''

        data_volume, masks, V_or = tensors_io.load_data_and_masks(params_t)
        path_temp = ""
        for local_path in path_of_interest.split("/"):
            path_temp = path_temp + local_path
            if not os.path.isdir(path_temp):
                os.mkdir(path_temp)
            path_temp = path_temp + "/"

    if(resample_masks == 0):
        tensors_io.save_subvolume(data_volume, path_of_interest)
    else:
        tensors_io.save_subvolume_instances(data_volume * 0, masks, path_of_interest)



def recrop(params_t, resample_masks=0, factor=1, directory='.'):
    if(directory != '.'):
        path_of_interest = directory + '/subsampled'
        resample_masks = 2
        masks = tensors_io.load_volume(directory, scale=2).unsqueeze(0)
        tensors_io.save_subvolume(masks, path_of_interest)
        exit()
    else:
        params_t.cleaning = False
        params_t.cleaning_sangids = False
        params_t.scale_p = factor
        if(resample_masks == 0):
            path_of_interest = "results/" + params_t.dataset_name + "/" + "v_" + str(params_t.dataset_version) + "/or_subsampled"
        elif(resample_masks == 1):
            path_of_interest = "results/" + params_t.dataset_name + "/" + "v_" + str(params_t.dataset_version) + "/gt_subsampled"
        elif(resample_masks == 2):
            if(params_t.debug_cluster_unet_double):
                params_t.network_string = "METHOD"
            path_of_interest = "results/" + params_t.dataset_name + "/" + "v_" + str(params_t.dataset_version) + "/" + params_t.network_string + "/mini_inst_only_subsampled"
            params_t.testing_mask = "results/" + params_t.dataset_name + "/" + "v_" + str(params_t.dataset_version) + "/" + params_t.network_string + "/mini_inst_only"
        else:
            path_of_interest = ''

        data_volume, masks, V_or = tensors_io.load_data_and_masks(params_t)
        path_temp = ""
        for local_path in path_of_interest.split("/"):
            path_temp = path_temp + local_path
            if not os.path.isdir(path_temp):
                os.mkdir(path_temp)
            path_temp = path_temp + "/"

    if(resample_masks == 0):
        tensors_io.save_subvolume(data_volume, path_of_interest)
    else:
        tensors_io.save_subvolume_instances(data_volume * 0, masks, path_of_interest)
