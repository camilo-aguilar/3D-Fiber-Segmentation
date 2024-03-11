
import numpy as np
import torch
import random
try:
    from .geometry import rotation_matrix_from_axis_and_angle, projection_matrix, rotation_matrix_from_axis_and_vector
    from .tensors_io import save_subvolume
except:
    from codes.utils.geometry import rotation_matrix_from_axis_and_angle, projection_matrix, rotation_matrix_from_axis_and_vector
    from codes.utils.tensors_io import save_subvolume

class Fiber:
    def __init__(self, tube, label=1):
        self.N = 0
        self.list_tubes = []
        self.label = label
        self.Vo = 0
        self.center = np.array(tube.C)
        self.AddTube(tube)

    def AddTube(self, tube):
        self.list_tubes.append(tube)
        self.N = self.N + 1
        self.center = (self.center + np.array(tube.C)) / 2

    def RemoveTube(self, tube):
        print("Still to construct")

    def CalculateEnergy(self, V_raw, V_objects, list_of_objects, paramaters=None):
        energy = 0
        for tube in self.list_tubes:
            energy += tube.get_energy(V_raw, V_objects, list_of_objects, paramaters)
        return (energy)

    def draw(self, V, V_connection):
        for tube in self.list_tubes:
            coordinates = tube.get_coordinates_render(V.shape)
            V[coordinates] = self.label

            if(tube.is_end_tube_f or tube.is_end_tube_b):
                end_points_b, end_points_f = tube.get_coordinates_connection(V.shape)
                if(tube.is_end_tube_f):
                    V_connection[end_points_f] = self.label
                if(tube.is_end_tube_b):
                    V_connection[end_points_b] = -self.label


class Tube:
    def __init__(self, C=None, R=None, H=None, w=None, label=1, marks=None, parameters=None):
        if(marks is not None):
            self.C = marks[0].cpu().numpy()
            self.H = int(min(max(marks[2], parameters.mpp_min_l), parameters.mpp_max_l))  # max(marks[2], 4)
            self.R = float(min(max(marks[1].cpu().numpy(), parameters.mpp_min_r), parameters.mpp_max_r)) # max(marks[1], 1.5)#
            self.w = marks[3]
        else:
            self.C = C
            self.H = H
            self.R = float(R)
            self.w = w

        self.label = label
        self.Area = 0.0
        self.data_energy = 0.0
        self.prior_energy = 0.0
        self.Energy = 0

        self.is_end_tube_f = True
        self.is_end_tube_b = True

        # self.Ty = 0
        # self.Tz = 0

    def get_coordinates_render(self, ndims, R_hat=None):
            if(R_hat is None):
                R_hat = self.R
            R_hat = float(R_hat) / 2.0
            R_hat = max(R_hat, 0.5)
            resolution = self.H * 10
            l_fit = float(self.H)

            # Get the transformation matrix
            w = self.w[:, 0]
            Txy = np.arctan2(w[1], w[0])

            Tz = np.arccos(np.dot(w, np.array([0, 0, 1])))
            M = np.dot(rotation_matrix_from_axis_and_angle(np.array([0, 0, 1]), Txy),
                       rotation_matrix_from_axis_and_angle(np.array([0, 1, 0]), Tz))

            # M = rotation_matrix_from_axis_and_vector(np.array([0, 0, 1]), self.w)
            #M = np.dot(rotation_matrix_from_axis_and_vector(np.array([0, 0, 1]), self.w.numpy()),
            #           rotation_matrix_from_axis_and_vector(np.array([0, 1, 0]), self.w.numpy()))
            #M = projection_matrix(self.w).numpy()

            # Plot the cylinder surface
            delta = np.linspace(-np.pi, np.pi, resolution)
            z = np.linspace(-l_fit / 2.0, l_fit / 2.0, resolution)

            Delta, Z = np.meshgrid(delta, z)
            X = R_hat * np.cos(np.reshape(Delta, (1, -1)))
            Y = R_hat * np.sin(np.reshape(Delta, (1, -1)))

            Z = np.reshape(Z, (1, -1))

            cylinder = np.concatenate((X, Y, Z), axis=0)
            coordinates = np.round(np.matmul(M, cylinder)).astype(np.int) + np.expand_dims(self.C + 1, 1)
            
            #coordinates = np.clip(coordinates, 0, ndims[0] - 1)

            idx_i = (coordinates[0, :] >= 0) & (coordinates[0, :] < ndims[0])
            idx_j = (coordinates[1, :] >= 0) & (coordinates[1, :] < ndims[1])
            idx_k = (coordinates[2, :] >= 0) & (coordinates[2, :] < ndims[2])
            idx = idx_i & idx_j & idx_k

            coordinates = (coordinates[0, idx], coordinates[1, idx], coordinates[2, idx])
            return coordinates

    def get_coordinates_connection(self, ndims):
        l_fit = self.H

        # Get the transformation matrix
        #M = np.dot(rotation_matrix_from_axis_and_angle(np.array([0, 1, 0]), self.Tz),
        #           rotation_matrix_from_axis_and_angle(np.array([1, 0, 0]), self.Ty))
        M = skew_matrix(self.w)

        end_points_coords = np.array([[0, 0], [0, 0], [-l_fit, l_fit]])
        end_points_coords = np.round(np.matmul(M, end_points_coords)).astype(np.int)

        ball_coords = np.where(ball(self.R) > 0)
        coordinates_f = tuple([np.clip(ball_coords[i] + end_points_coords[i, 0] + self.C[i] - np.around(self.R), 0, ndims[0] - 1) for i in range(3)])

        coordinates_b = tuple([np.clip(ball_coords[i] + end_points_coords[i, 1] + self.C[i] - np.round(self.R), 0, ndims[0] - 1) for i in range(3)])

        return (coordinates_f, coordinates_b)

    def get_prior_and_data_energy(self, V_objects, V_seg, list_of_objects, parameters=None, V_connection=None):
        R = self.R
        Area = 0.0
        overlap = 0
        data_energy = 0.0
        other_area = 10000
        while(R >= 0):
            coordinates = self.get_coordinates_render(V_objects.shape, R_hat=R)
            Area = Area + len(coordinates[0])
            overlap_pixels = V_objects[coordinates]

            data_energy_pixels = V_seg[coordinates].sum()

            overlap_fibers = torch.unique(overlap_pixels)
            for f_ov in overlap_fibers:
                if(f_ov == 0 or f_ov == 1):
                    continue
                for el in list_of_objects:
                    if(f_ov.item() == el.label):
                        # other_area = min(other_area, el.Area)
                        el.prior_energy += (overlap_pixels == el.label).float().sum().item()
                else:
                    continue
            data_energy += data_energy_pixels.float().sum().item()
            overlap += (overlap_pixels > 0).float().sum().item()
            R = R - 1

        # min_area = float(min(Area, other_area))
        # area_overlap = overlap / float(min_area)
        prior_energy = overlap

        # if(parameters is not None):
        #    T_ov = parameters.mpp_T_ov
        # else:
        #    T_ov = 1
        # if(area_overlap > T_ov):
        #   prior_energy = 100

        self.Area = float(Area)
        self.prior_energy = prior_energy
        self.data_energy = data_energy
        return prior_energy, data_energy

    def get_prior_energy(self, V_objects, list_of_objects, parameters=None, V_connection=None):
        R = self.R
        Area = 0.0
        overlap = 0
        other_area = 10000
        while(R >= 0):
            coordinates = self.get_coordinates_render(V_objects.shape, R_hat=R)
            Area = Area + len(coordinates[0])
            overlap_pixels = V_objects[coordinates]
            overlap_fibers = torch.unique(overlap_pixels)
            for f_ov in overlap_fibers:
                if(f_ov == 0 or f_ov == 1):
                    continue
                for el in list_of_objects:
                    if(f_ov.item() == el.label):
                        # other_area = min(other_area, el.Area)
                        el.prior_energy += (overlap_pixels == el.label).float().sum().item()
                else:
                    continue

            overlap += (overlap_pixels > 0).float().sum().item()
            R = R - 1

        # min_area = float(min(Area, other_area))
        # area_overlap = overlap / float(min_area)
        prior_energy = overlap

        # if(parameters is not None):
        #    T_ov = parameters.mpp_T_ov
        # else:
        #    T_ov = 1
        # if(area_overlap > T_ov):
        #   prior_energy = 100

        self.Area = float(Area)
        self.prior_energy = prior_energy
        return prior_energy

    def get_prior_and_overlap_ids(self, V_objects, list_of_objects):
        R = self.R
        overlap_ids = set()
        Area = 0.0
        overlap = 0
        other_area = 10000
        while(R >= 0):
            coordinates = self.get_coordinates_render(V_objects.shape, R_hat=R)
            Area = Area + len(coordinates[0])
            overlap_pixels = V_objects[coordinates]
            overlap_fibers = torch.unique(overlap_pixels)
            for f_ov in overlap_fibers:
                if(f_ov == 0 or f_ov == 1):
                    continue
                for el in list_of_objects:
                    if(f_ov.item() == el.label):
                        other_area = min(other_area, el.Area)
                        overlap_ids.add(f_ov.item())
                    else:
                        continue

            overlap += (overlap_pixels > 0).float().sum().item()
            R = R - 1

        if(Area == 0):
            return 22222, 0
        min_area = float(min(Area, other_area))
        area_overlap = overlap / float(min_area)
        prior_energy = area_overlap

        self.Area = float(Area)
        self.prior_energy = prior_energy
        return prior_energy, overlap_ids

    def get_data_energy(self, V_raw, parameters=None):
        R = self.R + 5
        # Get outer coordinates
        outer_values = None
        while(R >= self.R + 4):
            coordinates = self.get_coordinates_render(V_raw.shape, R_hat=R)
            if(outer_values is None):
                outer_values = V_raw[coordinates].float()
            else:
                outer_values = torch.cat((outer_values, V_raw[coordinates].float()))
            R -= 0.25

        # Get inner coordinates
        R = 0
        # Get outer coordinates
        inner_values = None
        while(R <= self.R):
            coordinates = self.get_coordinates_render(V_raw.shape, R_hat=R)
            if(inner_values is None):
                inner_values = V_raw[coordinates].float()
            else:
                inner_values = torch.cat((inner_values, V_raw[coordinates].float()))
            R += 0.25
        u_in = inner_values.mean() + 0.00001
        o2_in = inner_values.std() + 0.00001
  
        u_out = outer_values.mean() + 0.00001
        o2_out = outer_values.std() + 0.00001
        Battacharya = 0.25 * np.log(0.25 * ((o2_out / o2_in) + (o2_in / o2_out) + 2)) + 0.25 * ((u_out - u_in)**2) / (o2_out + o2_in)

        self.Battacharya = Battacharya
        self.u_in = u_in
        self.u_out = u_out
        self.o2_in = o2_in
        self.o2_out = o2_out

        if(Battacharya < parameters.mpp_Threshold_Battacharya):
            data_energy = 1 - Battacharya / parameters.mpp_Threshold_Battacharya
        else:
            data_energy = torch.exp(-(Battacharya - parameters.mpp_Threshold_Battacharya) / (3.0 * parameters.mpp_Threshold_Battacharya)) - 1
        if(u_in < u_out):
            data_energy = torch.tensor(1111)
        self.data_energy = data_energy
        return data_energy

    def get_data_energy_counting(self, V_raw, parameters=None):
        coords = self.get_coordinates_render(V_raw.shape, self.R)
        white_pixels = V_raw[coords].sum()
        self.Area = float(len(coords[0]))
        if(self.Area == 0):
            return 1000000
        data_energy = float(white_pixels)
        self.data_energy = data_energy
        return data_energy

    def get_energy(self, V_raw, V_objects, list_of_objects, paramaters=None):
        # data_energy = self.get_data_energy_counting(V_raw, paramaters)
        # prior_energy = self.get_prior_energy(V_objects, list_of_objects, paramaters)
        prior_energy, data_energy = self.get_prior_and_data_energy(V_objects, V_raw, list_of_objects, paramaters)
        object_energy = data_energy + prior_energy
        self.Energy = object_energy
        return object_energy

    def draw(self, V):
        R = self.R
        Area = 0
        while(R >= 0):
            coordinates = self.get_coordinates_render(V.shape, R_hat=R)
            V[coordinates] = self.label
            R = R - 0.5
            Area = Area + len(coordinates[0])
        self.Area = float(Area)
        return V

    def draw_debug(self, V_raw):
        R = self.R + 5
        # Get outer coordinates
        while(R > self.R + 4):
            coordinates = self.get_coordinates_render(V_raw.shape, R_hat=R)
            V_raw[coordinates] = self.label + 1
            R -= 0.25

        # Get inner coordinates
        R = self.R - 1
        # Get outer coordinates
        while(R < self.R):
            coordinates = self.get_coordinates_render(V_raw.shape, R_hat=R)
            V_raw[coordinates] = self.label + 2
            R += 0.25

        coordinates = self.get_coordinates_render(V_raw.shape, R_hat=self.R)
        V_raw[coordinates] = self.label
        return V_raw

    def delete(self, V):
        R = self.R
        while(R >= 0):
            coordinates = self.get_coordinates_render(V.shape, R_hat=R)
            V[coordinates] = 0
            R = R - 1
        return V

    def draw_connections(self, V):
        end_points_b, end_points_f = self.get_coordinates_connection(V.shape)
        V[end_points_f] = self.label
        V[end_points_b] = -self.label


    def print_properties(self):
        print("Label : {}".format(self.label))
        print("Center : {}, {}, {}.    R: {}.  H: {}. Ty: {}. Tz {}. Vo: {}. Vp: {}.".format(self.C[0], self.C[1], self.C[2], self.R, self.H, int(np.degrees(self.Ty)), int(np.degrees(self.Tz)), self.data_energy, self.prior_energy))
        # print("uin {} uout {} o_in {} o_out {} . Battacharya {}".format(self.u_in, self.u_out, self.o2_in, self.o2_out, self.Battacharya))


    # def fit(self, volume, proposed_coords):
