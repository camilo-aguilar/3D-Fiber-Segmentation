from .unet_parts import up, down, outconv, inconv
from sklearn.cluster import DBSCAN
import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_dims=64):
        super(UNet, self).__init__()
        self.n_embeddings = n_classes
        self.inc = inconv(n_channels, num_dims)
        self.down1 = down(num_dims * 1, num_dims * 2)
        self.down2 = down(num_dims * 2, num_dims * 4)
        self.down3 = down(num_dims * 4, num_dims * 8)
        self.down4 = down(num_dims * 8, num_dims * 8)
        self.up1 = up(num_dims * 16, num_dims * 4)
        self.up2 = up(num_dims * 8, num_dims * 2)
        self.up3 = up(num_dims * 4, num_dims * 1)
        self.up4 = up(num_dims * 2, num_dims * 1)
        self.outc = outconv(num_dims, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def forward_inference(self, x, final_pred, params_t, mask=None):
        embedding_output = self(x).permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_embeddings)

        # Check only segmented pixels
        object_indexes = (final_pred > 0).long().view(-1).nonzero()
        if(len(object_indexes) == 0):
            return(None)
        object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, self.n_embeddings))

        # Transform from embeddings to coordinates if necessary
        # labels = clustering_algorithm(object_pixels, final_pred).cpu().numpy()
        if(params_t.offset_clustering):
            coordinates = (final_pred[0, ...] == 1).nonzero()
            object_pixels = coordinates - object_pixels

        # Vectorize and make it numpy
        X = object_pixels.detach().cpu().numpy()
        labels = params_t.clustering(X)

        # Convert back to space domain
        space_labels = torch.zeros_like(final_pred.view(-1))
        space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(params_t.device) + 2
        space_labels = space_labels.view(x.shape)

                # Save the clusters (if they are in 3D)
        if(params_t.debug):
            space_clusters = torch.zeros_like(final_pred[0, ...])
            t_mask = mask.clone()
            t_mask = t_mask[0, 0, ...].to(params_t.device)
            space_clusters[object_pixels.long().split(1, dim=1)] = torch.from_numpy(labels).unsqueeze(1).to(params_t.device) + 2
            space_clusters[object_pixels.long().split(1, dim=1)] = t_mask[object_pixels.long().split(1, dim=1)]
            return space_clusters

        return space_labels
