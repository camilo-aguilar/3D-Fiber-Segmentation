from .unet_parts import up, down, outconv, inconv
from sklearn.cluster import DBSCAN
import torch.nn as nn
import torch


class UNet_double(nn.Module):
    def __init__(self, n_channels, n_classes_d1, n_classes_d2, num_dims=64):
        super(UNet_double, self).__init__()
        # Encoder
        self.inc = inconv(n_channels, num_dims)
        self.down1 = down(num_dims * 1, num_dims * 2)
        self.down2 = down(num_dims * 2, num_dims * 4)
        self.down3 = down(num_dims * 4, num_dims * 8)
        self.down4 = down(num_dims * 8, num_dims * 8)

        # Decoder
        self.up1 = up(num_dims * 16, num_dims * 4)
        self.up2 = up(num_dims * 8, num_dims * 2)
        self.up3 = up(num_dims * 4, num_dims * 1)
        self.up4 = up(num_dims * 2, num_dims * 1)
        self.out_d1 = outconv(num_dims, n_classes_d1)
        self.out_d2 = outconv(num_dims, n_classes_d2)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder1
        x_d1 = self.up1(x5, x4)
        x_d1 = self.up2(x_d1, x3)
        x_d1 = self.up3(x_d1, x2)
        x_d1 = self.up4(x_d1, x1)
        x_d1 = self.out_d1(x_d1)

        # Decoder2
        x_d2 = self.up1(x5, x4)
        x_d2 = self.up2(x_d2, x3)
        x_d2 = self.up3(x_d2, x2)
        x_d2 = self.up4(x_d2, x1)
        x_d2 = self.out_d2(x_d2)

        return [x_d1, x_d2]

    def forward_inference_offset(self, x, final_pred, eps_param=0.4, min_samples_param=10, gt=None):
        device = x.device
        cube_size = x.shape[-1]

        output = self(x)

        magnitudes = output[0]
        offset_vectors = output[1]
        # tensors_io.save_volume_h5(magnitudes[0, 0, ...].cpu().numpy(), "mags", "mags")
        # exit()

        final_pred = (magnitudes > 0.8).long()
        temp = offset_vectors.float() * final_pred.float()
        temp = temp[0, ...].cpu().numpy()
        # np.save("temp_save.npy", temp)
        # exit()

        offset_vectors = offset_vectors.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

        object_indexes = (final_pred == 1).long().view(-1).nonzero()

        if(len(object_indexes) == 0):
            return(None)
        object_pixels = torch.gather(offset_vectors, 0, object_indexes.repeat(1, 3))
        a = torch.norm(object_pixels, dim=1, p=2)

        # object_pixels = object_pixels / torch.norm(object_pixels, p=2, dim=1).unsqueeze(1)

        # Get coordinates of objects
        coordinates = (final_pred[0, 0, ...] == 1).nonzero().float()

        object_pixels = coordinates - object_pixels
        # Numpy

        X = object_pixels.detach().cpu().numpy()

        # Cluster
        labels = DBSCAN(eps=eps_param, min_samples=min_samples_param).fit_predict(X)

        space_labels = torch.zeros_like(final_pred.view(-1))
        space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(device) + 2

        '''
        space_labels2 = torch.zeros_like(final_pred[0, 0, ...])
        space_labels2[object_pixels.long().split(1, dim=1)] = 1
        space_labels2 = space_labels2.view(cube_size, cube_size, cube_size).cpu().numpy()
        tensors_io.save_volume_h5(space_labels2, "offset_vectors", "offset_vectors")
        exit()
        '''

        space_labels = space_labels.view(cube_size, cube_size, cube_size)

        centers, fiber_ids, end_points, fiber_list = get_fiber_properties(space_labels)

        return space_labels, fiber_list

    def forward_inference_debug(self, x, final_pred, eps_param=0.4, min_samples_param=10, gt=None):
        # mathilde
        device = x.device
        cube_size = x.shape[-1]

        output = self(x)

        magnitudes = output[0]
        embedding_output = output[1]

        final_pred = (magnitudes > 0.5).long()
        temp = embedding_output.float() * final_pred.float()
        temp = temp[0, ...].cpu().detach().numpy()
        # np.save("temp_save.npy", temp)
        # exit()

        embedding_output = embedding_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

        object_indexes = (final_pred == 1).long().view(-1).nonzero()

        if(len(object_indexes) == 0):
            return(None)
        object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, 3))

        # object_pixels = object_pixels / torch.norm(object_pixels, p=2, dim=1).unsqueeze(1)

        # Get coordinates of objects
        coordinates = (final_pred[0, 0, ...] == 1).nonzero().float()

        object_pixels = coordinates - object_pixels
        # Numpy

        X = object_pixels.detach().cpu().numpy()

        # Cluster
        labels = DBSCAN(eps=1, min_samples=min_samples_param).fit_predict(X)

        space_labels = torch.zeros_like(final_pred.long().view(-1))
        space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(device) + 2

        space_labels = space_labels.view(cube_size, cube_size, cube_size)

        space_labels2 = torch.zeros_like(gt[0, 0, ...])
        space_labels3 = torch.zeros_like(gt[0, 0, ...])
        for l in torch.unique(gt):
            if(l == 0):
                continue
            coordinates = (gt[0, 0, ...].long() == l).nonzero().float()
            object_indexes = (gt.long() == l).long().view(-1).nonzero()
            object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, 3))
            object_pixels = coordinates - object_pixels
            space_labels2[object_pixels.long().split(1, dim=1)] = l

        for l in torch.unique(space_labels):
            if(l == 0):
                continue
            coordinates = (space_labels.long() == l).nonzero().float()
            object_indexes = (space_labels.long() == l).long().view(-1).nonzero()
            object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, 3))
            object_pixels = coordinates - object_pixels
            space_labels3[object_pixels.long().split(1, dim=1)] = l
        
        space_labels2 = space_labels2.view(cube_size, cube_size, cube_size).cpu().numpy()
        space_labels3 = space_labels3.view(cube_size, cube_size, cube_size).cpu().numpy()
        tensors_io.save_volume_h5(space_labels2, "offset_vectors", "offset_vectors")
        tensors_io.save_volume_h5(space_labels3, "offset_vectors_inference", "offset_vectors_inference")
        tensors_io.save_volume_h5(gt[0,0,...].cpu().detach().numpy(), "offset_gt", "offset_gt")
        exit()

       

        centers, fiber_ids, end_points, fiber_list = get_fiber_properties(space_labels)

        return space_labels, fiber_list
