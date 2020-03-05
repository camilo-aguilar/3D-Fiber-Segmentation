import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from torch.autograd import Function
from sklearn.cluster import MeanShift


class RNet_model(nn.Module):
    def __init__(self, n_channels, n_classes, num_dims=16):
        super(RNet_model, self).__init__()
        self.rb1 = ResidualBlock(n_channels, num_dims)
        self.rb2 = ResidualBlock(num_dims, 2 * num_dims)
        self.rb3 = ResidualBlock(2 * num_dims, 4 * num_dims)
        self.rb4 = ResidualBlock(4 * num_dims, 8 * num_dims)
        self.rb5 = ResidualBlock(8 * num_dims, 16 * num_dims)
        self.conv = nn.Conv3d(16 * num_dims, n_classes, kernel_size=1, stride=1)

        self.n_embeddings = n_classes

    def forward(self, x):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.conv(x)
        return x

    def forward_inference_fast(self, x, final_pred, num_fibers=0, eps_param=0.4, min_samples_param=10):
        GPU_YES = torch.cuda.is_available()
        device = x.device
        cube_size = final_pred.shape[-1]

        embedding_output = self(x)

        embedding_output = embedding_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_embeddings)

        object_indexes = (final_pred > 0).long().view(-1).nonzero()
        if(len(object_indexes) == 0):
            return None
        object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, self.n_embeddings))

        # Numpy
        X = object_pixels.detach().cpu().numpy()
        labels = DBSCAN(eps=eps_param, min_samples=min_samples_param).fit_predict(X)  # db_scan_clusering(object_pixels.detach().cpu().numpy())

        space_labels = torch.zeros_like(final_pred.view(-1))
        space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(device) + 2

        space_labels = space_labels.view(cube_size, cube_size, cube_size)

        centers, fiber_ids, end_points, fiber_list = get_fiber_properties(space_labels)

        return (space_labels, fiber_list)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)


    def forward(self, x):
        residual = self.res_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


def embedded_loss(outputs, labels):
    delta_v = 0.3
    delta_d = 4

    alpha = 2
    beta = 5
    gamma = 0.000001
    GPU_YES = torch.cuda.is_available()
    device = outputs.device

    N_embedded = outputs.shape[1]
    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, N_embedded))

    labels = torch.unique(labeled_pixels, sorted=True)
    N_objects = len(labels)

    mu_vector = torch.zeros(N_objects, N_embedded).to(device)

    # Find mu vector
    Lv = 0  # Lv_same_object_proximity
    Lr = 0  # Ld_minimal_distance_clusters
    Ld = 0  # Lr_cluster_center_regularizer
    for c in range(N_objects):
        fiber_id = labels[c]
        idx_c = (labeled_pixels == fiber_id).nonzero()

        # xi vector
        x_i = torch.gather(object_pixels, 0, idx_c.repeat(1, N_embedded))
        # get mu
        mu = x_i.mean(0)
        mu_vector[c, :] = mu

        # get Lv
        Nc = len(idx_c)

        lv_term = torch.norm(mu_vector[c, :] - x_i, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = lv_term**2
        lv_term = torch.sum(lv_term, dim=0)
        Lv += (lv_term / Nc)
        Lr += torch.norm(mu, 2)

    for ca in range(N_objects):
        for cb in range(ca + 1, N_objects):
            ld_term = delta_d - torch.norm(mu_vector[ca, :] - mu_vector[cb, :], 2)
            ld_term = torch.clamp(ld_term, 0, 10000000)
            ld_term = ld_term ** 2
            Ld += ld_term

    N_objects = float(N_objects)
    Lv /= N_objects
    Lr /= N_objects
    Ld /= (N_objects * (N_objects - 1))

    loss = alpha * Lv + beta * Ld + gamma * Lr
    return loss


import matplotlib.pyplot as plt
import pylab
def save_data(embeddings, labels, iteration=None, detected_labels=None):
    N_embedded = embeddings.shape[1]
    # num_dims = N_embedded
    # num_display_dims = 2
    # tsne_lr = 20.0

    # Get only the non-zero indexes
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(embeddings, 0, idx_array.repeat(1, N_embedded))

    from tsnecuda import TSNE
    # from tsne import tsne
    X = object_pixels.cpu().detach().numpy()
    Y = TSNE(n_components=2, perplexity=40, learning_rate=50).fit_transform(X)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(Y[:, 0], Y[:, 1], 5, labeled_pixels, cmap='tab20b')
    if iteration is None:
        iteration = 0
    plt.savefig("low_dim_embeeding/embedded_%d.png" % iteration)
    plt.close(fig)


    if detected_labels is not None:
        detected_labels_pixels = detected_labels[idx_tuple].squeeze(1)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        pylab.scatter(Y[:, 0], Y[:, 1], 5, detected_labels_pixels, cmap='tab20b')
        pylab.savefig("low_dim_embeeding/embedded_%d.png" % (iteration + 1))
        pylab.close(fig)

def get_fiber_properties(space_labels):
    end_points = []
    fiber_ids = []
    centers = {}
    fiber_list = {}
    # id center1, center2, center3, L, R, Ty, Tz, error
    for fiber_id in torch.unique(space_labels):
        if(fiber_id == 0 or fiber_id == 1):
            continue
        idx = (space_labels == fiber_id).nonzero().float()
        # idx is shape [N, 3]

        center = idx.mean(0)
        rs0 = torch.norm(idx - center, p=2, dim=1)

        # Find farthest distance from center
        end_point0_idx = (rs0 == rs0.max()).nonzero()
        
        # Find points close to EP1
        end_point0_idx = end_point0_idx[0, 0]
        idx_split = idx.split(1, dim=1)

        end_point0 = torch.tensor([idx_split[0][end_point0_idx], idx_split[1][end_point0_idx], idx_split[2][end_point0_idx]], device=idx.device)
        
        # Find closes points from end point 0
        rs1 = torch.norm(idx - end_point0, p=2, dim=1)
        end_point1_idx = (rs1 < 3).nonzero()
        end_point1_idx = end_point1_idx[:, 0]
        end_point1 = torch.tensor([idx_split[i][end_point1_idx][:, 0].mean() for i in range(3)])
         #, idx_split[1][end_point1_idx], idx_split[2][end_point1_idx]], device=idx.device)
     
        # Find farthest point from end point 1
        rs2 = torch.norm(idx - end_point0, p=2, dim=1)
        # Find farthest point from end point 1
        end_point2_idx = (rs2 > rs2.max() - 3).nonzero()
        end_point2_idx = end_point2_idx[:, 0]
        end_point2 = torch.tensor([idx_split[i][end_point2_idx][:, 0].mean() for i in range(3)])


        '''
        close_to_end_point1 = (rs2 < 3).nonzero().long()
        close_to_end_point1 = close_to_end_point1[:, 0]
        end_points_image[idx_split[0][close_to_end_point1].long(), idx_split[1][close_to_end_point1].long(), idx_split[2][close_to_end_point1].long()] = fiber_id
        close_to_end_point2 = (rs2 > rs2.max() - 3).nonzero().long()
        close_to_end_point2 = close_to_end_point2[:, 0]
        end_points_image[idx_split[0][close_to_end_point2].long(), idx_split[1][close_to_end_point2].long(), idx_split[2][close_to_end_point2].long()] = fiber_id
        '''
        end_points.append(end_point1.cpu().numpy())
        end_points.append(end_point2.cpu().numpy())

        fiber_ids.append(fiber_id.cpu().item())
        fiber_ids.append(fiber_id.cpu().item())

        c_np = center.cpu().numpy()
        centers[fiber_id.cpu().item()] = c_np

        length = torch.norm(end_point1 - end_point2, p=2).cpu().item()
        direction = (end_point1 - end_point2)
        direction = direction / torch.norm(direction, p=2)

        '''
        if(math.isnan(direction)):
            idx = (space_labels == fiber_id).nonzero().split(1, dim=1)
            space_labels[idx] = 1
        '''
       #  rr = fit_t.r2(direction.unsqueeze(1), idx, center.unsqueeze(1))
        R = 1.5 # rr[1]
        # G = fit_t.G(direction.unsqueeze(1), idx)

        direction = direction.cpu().numpy()
        fiber_list[fiber_id.cpu().item()] = [fiber_id.cpu().item(), c_np[0], c_np[1], c_np[2], R, length, direction[0], direction[1], direction[2]]

    return centers, fiber_ids, end_points, fiber_list
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    Net = RNet_model(1, 2).to(device)
    X = torch.ones(1, 1, 32, 32, 32).to(device)
    Y = Net(X)

    criterion = Embedded_Loss()
    print(X.device)
    # print(Y.shape)




