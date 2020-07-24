from .unet_parts import up, down, outconv, inconv
from codes.utils.clustering import clustering_algorithm
from sklearn.cluster import DBSCAN
import torch.nn as nn
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import DBSCAN 
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
        #############################################DEBUG
        # final_pred = mask.cuda()
        # Check only segmented pixels
        object_indexes = (final_pred > 0).long().view(-1).nonzero()
        if(len(object_indexes) == 0):
            if(params_t.debug_cluster_unet_double):
                return None, None
            return(None)
        object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, self.n_embeddings))

        # Transform from embeddings to coordinates if necessary
        # labels = clustering_algorithm(object_pixels, final_pred).cpu().numpy()
        # if(params_t.offset_clustering):
        #    coordinates = (final_pred[0, ...] == 1).nonzero()
        #    object_pixels = coordinates - object_pixels

                    # Transform from embeddings to coordinates if necessary
        
        if(params_t.debug_cluster_unet_double):
            labels, marks = clustering_algorithm(object_pixels, final_pred[0, ...], mask, ((x[0, 0, ...] * params_t.std_d) + params_t.mu_d).cpu(), params_t)
            labels = labels.cpu().numpy()
            space_labels = torch.zeros_like(final_pred.view(-1))
            space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(params_t.device) + 2
            space_labels = space_labels.view(x.shape)
            if(params_t.return_marks):
                return marks.unsqueeze(0).unsqueeze(0).to(space_labels.device), space_labels
            return space_labels, marks.unsqueeze(0).unsqueeze(0).to(space_labels.device)

        else:
            if(params_t.offset_clustering):
                object_pixels = object_pixels.cpu()
                final_pred = final_pred.cpu()
                object_pixels = transform_embedding_to_coordinates(object_pixels, final_pred)
                final_pred = final_pred.to(params_t.device)
                object_pixels = object_pixels.to(params_t.device)
            # Vectorize and make it numpy
            X = object_pixels.detach().cpu().numpy()
            labels = params_t.clustering(X)
            labels_eps1 = DBSCAN(eps=params_t.eps_param / 10.0, min_samples=params_t.min_samples_param).fit_predict(X)
            labels_eps2 = DBSCAN(eps=params_t.eps_param * 4, min_samples=params_t.min_samples_param).fit_predict(X)
            # labels = labels_eps2
        # Convert back to space domain
        space_labels = torch.zeros_like(final_pred.view(-1))
        space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(params_t.device) + 2
        space_labels = space_labels.view(x.shape)

        # Save the clusters (if they are in 3D)
        if(params_t.offset_clustering):
            if(params_t.debug):
                space_clusters = torch.zeros_like(final_pred[0, 0, ...])
                if(mask is not None):
                    t_mask = mask.clone()
                    t_mask = t_mask[0, 0, ...].to(params_t.device)
                    object_pixels = torch.clamp(object_pixels, 0, params_t.cube_size - 1)
                    space_clusters[object_pixels.long().split(1, dim=1)] = t_mask[object_pixels.long().split(1, dim=1)]
                    # space_clusters[object_pixels.long().split(1, dim=1)] = torch.from_numpy(labels).unsqueeze(1).to(params_t.device) + 2
                else:
                    space_clusters[object_pixels.long().split(1, dim=1)] = torch.from_numpy(labels).unsqueeze(1).to(params_t.device) + 2

                final_pred = final_pred.unsqueeze(0)
                return space_labels, space_clusters.unsqueeze(0).unsqueeze(0)
        # Save the clusters (if they are in 3D)
        if(params_t.debug):
            '''
            t_mask = mask.clone().view(-1)
            idx_tuple = object_indexes.split(1, dim=1)
            labeled_pixels1 = t_mask[idx_tuple].squeeze(1)

            print("TSNE 2D")
            import matplotlib.pyplot as plt
            Y = TSNE(n_components=2).fit_transform(X)
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(Y[:, 0], Y[:, 1], 5, labeled_pixels1, cmap='tab20b')
            plt.savefig("embedded_gt.png")
            plt.close(fig)

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(Y[:, 0], Y[:, 1], 5, labels, cmap='tab20b')
            plt.savefig("embedded_inference.png")
            plt.close(fig)

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(Y[:, 0], Y[:, 1], 5, labels_eps1, cmap='tab20b')
            plt.savefig("embedded_inference_eps1.png")
            plt.close(fig)

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(Y[:, 0], Y[:, 1], 5, labels_eps2, cmap='tab20b')
            plt.savefig("embedded_inference_eps2.png")
            plt.close(fig)



            print("TSNE 3D")
            X_embedded = TSNE(n_components=3).fit_transform(X)
            min_x = X_embedded.min()
            max_x = X_embedded.max()
            X_embedded = (X_embedded - min_x) / (max_x - min_x) * (x.shape[-1] - 1)
            X_embedded = torch.from_numpy(X_embedded)
            '''
            space_clusters = torch.zeros_like(final_pred[0, 0, ...])
            space_clusters[object_pixels.long().split(1, dim=1)] = torch.from_numpy(labels).unsqueeze(1).to(params_t.device) + 2
            return space_labels, space_clusters.unsqueeze(0).unsqueeze(0)

        return space_labels


def transform_embedding_to_coordinates(object_pixels, final_pred):
    coordinates = (final_pred[0, 0, ...] == 1).nonzero().float()
    object_pixels = coordinates - object_pixels
    return object_pixels
