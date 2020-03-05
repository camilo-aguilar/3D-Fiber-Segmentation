import codes.utils.evaluation as evaluation
import codes.utils.tensors_io as tensors_io
from sklearn.cluster import DBSCAN
import torch.optim as optim
import numpy as np
import torch

# import testing_routines


##################################################################################################################################
#                                                      TRAIN Network                                                             #
##################################################################################################################################
def train_semantic_segmentation_net(t_params, data_path_list, mask_path_list):
    print("Starting Training...")
    epochs = t_params.epochs
    batch_size = t_params.batch_size
    scale_p = t_params.scale_p
    uint_16 = t_params.uint_16
    pre_trained = t_params.pre_trained
    net_weights_dir = t_params.net_weights_dir
    cleaning = t_params.cleaning
    device = t_params.device
    n_classes = t_params.n_classes
    cube_size = t_params.cube_size
    Patches_per_Epoch = t_params.batch_size

    net = t_params.net
    net_weights_dir = t_params.net_weights_dir
    criterion = t_params.criterion

    if(device is None):
        device = torch.device("cuda:0")
    if(pre_trained is True):
        net.load_state_dict(torch.load(net_weights_dir[0]))

    data_volume_list = []
    # Load Data Volume
    if(uint_16 is False):
        for data_path in data_path_list:
            data_volume_list.append((tensors_io.load_volume(data_path, scale=scale_p)).unsqueeze(0))
    else:
        for data_path in data_path_list:
            data_volume_list.append((tensors_io.load_fibers_uint16(data_path, scale=scale_p)).unsqueeze(0))

    if(t_params.cleaning is True):
        for i in range(len(data_volume_list)):
            data_volume_list[i] = tensors_io.normalize_dataset(data_volume_list[i])
    # Load Masks
    masks_list = []
    for mask_path in mask_path_list:
        masks_list.append(tensors_io.load_volume_uint16(mask_path, scale=scale_p).long().unsqueeze(0))

    # Optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=t_params.lr)
    num_datasets = len(data_path_list)
    # Send the model to GPU
    net = net.to(device)
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        epoch_loss = 0
        emb_total_loss = 0
        seg_total_loss = 0
        for i in range(Patches_per_Epoch):
            data_volume = data_volume_list[i % num_datasets]
            masks = masks_list[i % num_datasets]
            (mini_V, mini_M) = tensors_io.random_crop_3D_image_batched(data_volume, masks, cube_size)

            mini_V = mini_V.to(device)
            mini_M = mini_M.to(device)

            true_masks = (mini_M > 0).long()
            true_masks_binary = (true_masks > 0).long()

            if(true_masks.max() == 0):
                continue

            segmentation_output = net(mini_V)

            s_loss = criterion(true_masks_binary, segmentation_output)
            epoch_loss += s_loss.item()
            seg_total_loss += s_loss.item()

            optimizer.zero_grad()
            s_loss.backward()
            optimizer.step()

        if(epoch % 30 == 0):
            print('Dict Saved')
            torch.save(net.state_dict(), net_weights_dir[0])
            _, final_pred = segmentation_output.max(1)
            evaluation.evaluate_segmentation(final_pred, mini_M)
            if(t_params.debug is True):
                tensors_io.save_subvolume_instances(mini_V, final_pred, "results/debug_seg_training")

        print("loss: " + str(epoch_loss / i))

    # save dictionary
    torch.save(net.state_dict(), net_weights_dir[0])
    evaluation.evaluate_segmentation(final_pred, mini_M)
    print("FINISHED TRAINING")


def train_r_net_embedded(net, net_s, data_path, mask_path, data_list2, mask_list2, n_classes=2, n_embedded=16, cube_size=32, epochs=10, batch_size=1, Patches_per_Epoch=30, scale_p=2, pre_trained=False, device=None, net_weights_dir=None, cleaning=False):
    print("Starting Training...")
    GPU_YES = torch.cuda.is_available()
    if(device is None):
        device = torch.device("cuda:0" if GPU_YES else "cpu")

    if(net_weights_dir is None):
        net_weights_dir = ['info_files/r_net.pth', 'info_files/r_net_e.pth']

    if(pre_trained == True):
        net.load_state_dict(torch.load(net_weights_dir[1]))
        net_s.load_state_dict(torch.load(net_weights_dir[0]))
    else:
        print("loading pre-trained weights from segmentation")
        pretrained_dict = torch.load(net_weights_dir[0])
        model_dict = net.state_dict()
        net_s.load_state_dict(pretrained_dict)

        # 1. filter out unnecessary keys
        for k, v in pretrained_dict.items():
            if(k == "conv.weight" or k == "conv.bias" or k=="outc.conv.weight" or k == "outc.conv.bias"):
                continue
            pretrained_dict = {k: v}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        net.load_state_dict(model_dict)

    # Load Data Volume
    if(cleaning is True):
        data_volume = tensors_io.load_volume(data_path, scale=scale_p)
        data_volume[0, ...] = tensors_io.clean_noise(data_volume[0, ...], data_path)

        # Load Data Volume2
        temp_volume = tensors_io.load_volume(data_list2[0], scale=scale_p)
        temp_volume[0, ...] = tensors_io.clean_noise(temp_volume[0, ...], data_list2[0])
    else:
        data_volume = tensors_io.load_fibers_uint16(data_path, scale=scale_p)
        temp_volume = tensors_io.load_fibers_uint16(data_list2[0], scale=scale_p)

    data_volume = data_volume.unsqueeze(0)
    (ch2, rows2, cols2, depth2) = temp_volume.shape
    num_datasets = len(data_list2)
    data_volume2 = torch.zeros(num_datasets, ch2, rows2, cols2, depth2)
    data_volume2[0, ...] = temp_volume

    masks = tensors_io.load_volume_uint16(mask_path, scale=scale_p).long().unsqueeze(0)   
    masks2 = torch.zeros(num_datasets, ch2, rows2, cols2, depth2, dtype=torch.long)
    masks2[0, ...] = tensors_io.load_volume_uint16(mask_list2[0], scale=scale_p).long()

    for counter in range(1, num_datasets):
        if(cleaning is True):
            temp_volume = tensors_io.load_volume(data_list2[counter], scale=scale_p)
            temp_volume[0, ...] = tensors_io.clean_noise(temp_volume[0, ...], data_list2[counter])
        else:
            temp_volume = tensors_io.load_fibers_uint16(data_list2[counter], scale=scale_p)

        data_volume2[counter, ...] = temp_volume
        masks2[counter, ...] = tensors_io.load_volume_uint16(mask_list2[counter], scale=scale_p).long()
    [_, channels, rows, cols, slices] = data_volume.size()

    # Optimizer and loss function
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr = 0.001
    # Send the model to CPU or GPU
    net = net.to(device)
    for epoch in range(epochs):
        if((epoch + 1) % 300 == 0):
            lr = lr / 10
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        epoch_loss = 0
        emb_total_loss = 0
        seg_total_loss = 0
        for i in range(Patches_per_Epoch):
            if(i % (num_datasets + 1) == 0):
                (mini_V, mini_M) = tensors_io.random_crop_3D_image_batched(data_volume, masks, cube_size)
            else:
                number = i % num_datasets
                vol = data_volume2[number, ...].unsqueeze(0)
                msk = masks2[number, ...].unsqueeze(0)
                (mini_V, mini_M) = tensors_io.random_crop_3D_image_batched(vol, msk, cube_size)

            mini_V = mini_V.to(device)
            true_masks = mini_M.to(device)

            # Evaluate Net
            embedding_output = net(mini_V)
            embedding_output = embedding_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, n_embedded)

            true_masks = true_masks.contiguous().view(-1)
            loss = embedded_loss(embedding_output, true_masks)
            if(loss is None):
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            emb_total_loss += loss.item()

        if((epoch) % 10 == 0 and epoch > -1):
            torch.save(net.state_dict(), net_weights_dir[1])
            tensors_io.save_data_3D(embedding_output, true_masks, epoch)
            print('Dict Saved')
        print("loss: " + str(epoch_loss / i) + ", e_loss: " + str(emb_total_loss / i) + ", s_loss: " + str(seg_total_loss / i))
    # save dictionary
    torch.save(net.state_dict(), net_weights_dir[1])
    print("FINISHED TRAINING")


def train_r_net_embedded_direction(net, net_s, data_path, mask_path, data_list2, mask_list2, n_classes=2, n_embedded=16, cube_size=32, epochs=10, batch_size=1, Patches_per_Epoch=30, scale_p=2, pre_trained=False, device=None, net_weights_dir=None, cleaning=True):
    print("Starting Training...")
    GPU_YES = torch.cuda.is_available()
    device = torch.device("cuda:0" if GPU_YES else "cpu")
    if(GPU_YES):
        print("~~~~Using GPU~~~~")
    else:
        print("~~~~Using CPU~~~~")

    print("Loading Data...", end='.')

    if(net_weights_dir is None):
        net_weights_dir = ['info_files/r_net.pth']

    if(pre_trained==True):
        net.load_state_dict(torch.load(net_weights_dir[0]))

    # Load Data Volume

    if(cleaning is True):
        data_volume = tensors_io.load_volume(data_path, scale=scale_p)
        data_volume[0, ...] = tensors_io.clean_noise(data_volume[0, ...], data_path)
    else:
        data_volume = tensors_io.load_fibers_uint16(data_path, scale=scale_p)
        
    mask_path = 'MORE_TRAINING/direction_training/UPDATED_TRAINING_LABELSdirections.npy'
    mask_list2 = []
    for i in range(1, 5):
        mask_list2.append('MORE_TRAINING/direction_training/fibers_uint16_sV' + str(i) + 'directions.npy')
    print("Starting Training direction net...")
    GPU_YES = torch.cuda.is_available()
    if(device is None):
        device = torch.device("cuda:0" if GPU_YES else "cpu")
    
    print("Loading Data...", end='.')

    if(pre_trained):
        print("Loading pre-trained-weights")
        net.load_state_dict(torch.load('info_files/r_net_d.pth'))
    else:
        print("No pretrained_dict")


    # Load Data Volume
    data_volume = tensors_io.load_volume(data_path, scale=scale_p)
    data_volume[0, ...] = tensors_io.clean_noise(data_volume[0, ...], data_path)
    data_volume = data_volume.unsqueeze(0)

    # Load Data Volume2
    temp_volume = tensors_io.load_volume(data_list2[0], scale=scale_p)
    temp_volume[0, ...] = tensors_io.clean_noise(temp_volume[0, ...], data_list2[0])
    
    masks = np.load(mask_path)
    masks = torch.from_numpy(masks)
    masks = masks[:, ::scale_p, ::scale_p, ::scale_p].unsqueeze(0).float()

    (ch2, rows2, cols2, depth2) = temp_volume.shape
    num_datasets = len(data_list2)

    data_volume2 = torch.zeros(num_datasets, ch2, rows2, cols2, depth2)
    masks2 = torch.zeros(num_datasets, 3, rows2, cols2, depth2, dtype=torch.float)
    data_volume2[0, ...] = temp_volume
    m2 = torch.from_numpy(np.load(mask_list2[0]))
    m2 = m2[..., 0:-1]
    masks2[0, ...] = m2[:, ::scale_p, ::scale_p, ::scale_p].clone()
    for counter in range(1, num_datasets):
        temp_volume = tensors_io.load_volume(data_list2[counter], scale=scale_p)
        temp_volume[0, ...] = tensors_io.clean_noise(temp_volume[0, ...], data_list2[counter])
        data_volume2[counter, ...] = temp_volume

        m2 = torch.from_numpy(np.load(mask_list2[counter]))
        m2 = m2[..., 0:-1].float()
        masks2[counter, ...] = m2[:, ::scale_p, ::scale_p, ::scale_p].clone()
    # Optimizer and loss function
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)


    optimizer = optim.Adam(net.parameters(), lr=0.0001)


    # Send the model to CPU or GPU
    net = net.to(device)
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        epoch_loss = 0
        emb_total_loss = 0
        seg_total_loss = 0
        for i in range(Patches_per_Epoch):

            if(i % (num_datasets + 1) == 0):
                (mini_V, mini_M) = tensors_io.random_crop_3D_image_batched(data_volume, masks, cube_size)
            else:
                number = i % num_datasets
                vol = data_volume2[number, ...].unsqueeze(0)
                msk = masks2[number, ...].unsqueeze(0)
                (mini_V, mini_M) = tensors_io.random_crop_3D_image_batched(vol, msk, cube_size)
            
            mini_V = mini_V.to(device)
            mini_directions = mini_M.to(device)

            # Evaluate Net
            output = net(mini_V)

            direction_output = output
            direction_output = direction_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)
            mini_directions = mini_directions.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

            # embedding_output = output[1]
            # embedding_output = embedding_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, n_embedded)
            #true_masks = true_masks.contiguous().view(-1)

            loss = direction_loss(direction_output, mini_directions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            emb_total_loss += loss.item()

        if((epoch) % 10 == 0 and epoch > 50):
            torch.save(net.state_dict(), "info_files/r_net_d.pth")
            #save_data(embedding_output, true_masks, epoch)
            print('Dict Saved')
        print("loss: " + str(epoch_loss / i) + ", e_loss: " + str(emb_total_loss / i) + ", s_loss: " + str(seg_total_loss / i))
    # save dictionary
    torch.save(net.state_dict(), "info_files/r_net_d.pth")
    print("FINISHED TRAINING")


def train_r_net_embedded_offset(net, net_s, data_path, mask_path, data_list2, mask_list2, n_classes=2, n_embedded=16, cube_size=32, epochs=10, batch_size=1, Patches_per_Epoch=30, scale_p=2, pre_trained=False, device=None):
    print("Starting Offset Training...")
    GPU_YES = torch.cuda.is_available()
    if(device is None):
        device = torch.device("cuda:0" if GPU_YES else "cpu")
    
    print("Loading Data...", end='.')

    if(pre_trained):
        print("Loading pre-trained-weights")
        net.load_state_dict(torch.load('info_files/r_net_offset.pth'))



    # Load Data Volume
    data_volume = tensors_io.load_volume(data_path, scale=scale_p)
    data_volume[0, ...] = tensors_io.clean_noise(data_volume[0, ...], data_path)
    data_volume = data_volume.unsqueeze(0)

    # Load Data Volume2
    temp_volume = tensors_io.load_volume(data_list2[0], scale=scale_p)
    temp_volume[0, ...] = tensors_io.clean_noise(temp_volume[0, ...], data_list2[0])
    
    masks = tensors_io.load_volume_uint16(mask_path, scale=scale_p).long().unsqueeze(0)
   

    (ch2, rows2, cols2, depth2) = temp_volume.shape
    num_datasets = len(data_list2)

    data_volume2 = torch.zeros(num_datasets, ch2, rows2, cols2, depth2)
    masks2 = torch.zeros(num_datasets, ch2, rows2, cols2, depth2, dtype=torch.long)
    data_volume2[0, ...] = temp_volume
    masks2[0, ...] = tensors_io.load_volume_uint16(mask_list2[0], scale=scale_p).long()
    for counter in range(1, num_datasets):
        temp_volume = tensors_io.load_volume(data_list2[counter], scale=scale_p)
        temp_volume[0, ...] = tensors_io.clean_noise(temp_volume[0, ...], data_list2[counter])
        data_volume2[counter, ...] = temp_volume

        masks2[counter, ...] = tensors_io.load_volume_uint16(mask_list2[counter], scale=scale_p).long()




    # Load Masks
     # masks2 = tensors_io.load_volume_uint16(mask_list2, scale=scale_p).long().unsqueeze(0)
    
    # Adapt Masks to semantic segmentation
    #segmentation = get_only_segmentation(net_s, data_volume, n_classes, n_embedded, cube_size)
    # masks = refine_watershed(masks, segmentation=segmentation)

    # tensors_io.save_subvolume_instances(data_volume, masks, 'refined_seg')
    [_, channels, rows, cols, slices] = data_volume.size()

    # Optimizer and loss function
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.0005)


    # Send the model to CPU or GPU
    net = net.to(device)
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        epoch_loss = 0
        emb_total_loss = 0
        seg_total_loss = 0
        for i in range(Patches_per_Epoch):
            if(i % (num_datasets + 1) == 0):
                (mini_V, mini_M) = tensors_io.random_crop_3D_image_batched(data_volume, masks, cube_size)
            else:
                number = i % num_datasets
                vol = data_volume2[number, ...].unsqueeze(0)
                msk = masks2[number, ...].unsqueeze(0)
                (mini_V, mini_M) = tensors_io.random_crop_3D_image_batched(vol, msk, cube_size)

            mini_V = mini_V.to(device)
            true_masks = mini_M.to(device)

            # Evaluate Net
            embedding_output = net(mini_V)

            loss = embedded_geometric_loss_coords(embedding_output, true_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            emb_total_loss += loss.item()

        if((epoch) % 10 == 0 and epoch > -1):
            torch.save(net.state_dict(), "info_files/r_net_offset.pth")
            # save_data(embedding_output, true_masks, epoch)
            print('Dict Saved')
        print("loss: " + str(epoch_loss / i) + ", e_loss: " + str(emb_total_loss / i) + ", s_loss: " + str(seg_total_loss / i))
    # save dictionary
    torch.save(net.state_dict(), "info_files/r_net_offset.pth")
    print("FINISHED TRAINING")