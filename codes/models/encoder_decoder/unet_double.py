from .unet_parts import up, down, outconv, inconv
import torch.nn as nn
import torch


class UNet_double(nn.Module):
    def __init__(self, n_channels, n_classes_d1, n_classes_d2, num_dims=64):
        super(UNet_double, self).__init__()
        self.n_classes = n_classes_d1
        self.n_embeddings = n_classes_d2
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

    def forward_inference(self, x, params_t, mask=None):
        outputs = self(x)

        if(params_t.network_name == 'unet_double_multi'):
            outputs[0] = outputs[0][:, 0:2, ...]
            outputs[1] = outputs[1][:, 0:3, ...]

        # Segmentation
        masks_probs = nn.functional.softmax(outputs[0], dim=1).float()
        _, final_pred = masks_probs.max(1)

        # Check only segmented pixels
        object_indexes = (final_pred > 0).long().view(-1).nonzero()
        if(len(object_indexes) == 0):
            return(None)

        # Embedding
        embedding_output = outputs[1].permute(0, 2, 3, 4, 1).contiguous().view(-1, params_t.n_embeddings)
        object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, params_t.n_embeddings))

        if(params_t.debug is False):
            # Magnitude Exploring
            magnitudes = torch.norm(object_pixels, dim=1)
            magnitudes = magnitudes / magnitudes.max()
            space_mags = torch.zeros_like(final_pred.view(-1)).float()
            space_mags[object_indexes] = 1 - magnitudes.unsqueeze(1) ** (1.0 / 6.0)
            # import codes.utils.tensors_io as tensors_io
            # space_mags = space_mags.view(x.shape)
            # tensors_io.save_subvolume(space_mags, "test_magnitudes")

        # Transform from embeddings to coordinates if necessary
        if(params_t.offset_clustering):
            object_pixels = transform_embedding_to_coordinates(object_pixels.cpu(), final_pred.cpu())
        labels = clustering_algorithm(magnitudes, object_pixels).cpu().numpy()

        # Vectorize and make it numpy
        # X = object_pixels.detach().cpu().numpy()
        # labels = params_t.clustering(X)

        # Convert back to space domain
        space_labels = torch.zeros_like(final_pred.view(-1))
        space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(params_t.device)# + 2

        # Save the clusters (if they are in 3D)
        if(params_t.debug):
            space_clusters = torch.zeros_like(final_pred[0, ...])
            t_mask = mask.clone()
            t_mask = t_mask[0, 0, ...].to(params_t.device)
            space_clusters[object_pixels.long().split(1, dim=1)] = torch.from_numpy(labels).unsqueeze(1).to(params_t.device) + 2
            space_clusters[object_pixels.long().split(1, dim=1)] = t_mask[object_pixels.long().split(1, dim=1)]
            import codes.utils.tensors_io as tensors_io
            space_mags = space_mags.view(x.shape)
            tensors_io.save_subvolume_instances(space_mags, space_clusters.unsqueeze(0).unsqueeze(0), "test_magnitudes_clusters")
            return final_pred, space_clusters.unsqueeze(0).unsqueeze(0)

        space_labels = space_labels.view(x.shape)
        return final_pred, space_labels


def transform_embedding_to_coordinates(object_pixels, final_pred):
    coordinates = (final_pred[0, ...] == 1).nonzero()
    object_pixels = coordinates - object_pixels
    return object_pixels


def clustering_algorithm(magnitudes, object_pixels):
    # magnitudes is [N_foreground] object_pixels [N_foreground, 3]
    labels = torch.zeros(magnitudes.shape, dtype=torch.long)
    _, indexes = torch.sort(magnitudes)
    threshold = 4
    label = 1
    while(True):
        # get coordinate
        center_coord = object_pixels[indexes[0], :]

        # Same dimension as indexes
        distances = torch.norm(object_pixels[indexes, :] - center_coord, dim=1)

        # find nearby pixels to cluster center
        nearby_pixels = distances < threshold
        if(nearby_pixels.nonzero().shape[0] == 1):
            break
        # Get indexes according to distances
        absolute_indexes = indexes[nearby_pixels]
        labels[absolute_indexes] = label

        # change index array according to distances
        indexes = indexes[distances >= threshold]
        label = label + 1
    return labels
