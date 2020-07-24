'''
Code added to repository by Camilo Aguilar
Code created by : Xingjie Pan
Source: https://pypi.org/project/cylinder_fitting/

'''
import numpy as np
import torch
from scipy.optimize import minimize
from skimage.morphology import ball
import torch.optim as optim


def get_ref_angle(w):
    Txy = np.arctan2(w[1], w[0]) * 180.0 / np.pi
    Tz = np.arccos(np.dot(w, np.array([0, 0, 1]))) * 180.0 / np.pi
    return Txy, Tz

def G_torch(w, Xs):
    '''Calculate the G function given a cylinder direction w and a
    list of data points Xs to be fitted.'''
    n = len(Xs)
    P = projection_matrix(w)
    Ys = torch.mm(P, Xs.t())
    Ys = Ys.to(Xs.device)
    A = calc_A(Ys)
    A = A.to(Xs.device)
    A_hat = calc_A_hat(A, skew_matrix(w))
    A_hat = A_hat.to(Xs.device)
    u = sum(torch.mm(Y.t(), Y) for Y in Ys.split(1, dim=1)) / n
    v = torch.mm(A_hat, sum(torch.mm(Y.t(), Y) * Y for Y in Ys.split(1, dim=1))) / torch.trace(torch.mm(A_hat, A))

    return sum((torch.mm(Y.t(), Y) - u - 2 * torch.mm(Y.t(), v)) ** 2 for Y in Ys.split(1, dim=1))


def get_angle_refined(w_hat, center, points):
    def F(w_hat, points):
        return torch.norm(u - torch.mm(torch.mm(u, w_hat), w_hat.t()), dim=1).mean()

    u = points - center
    opt = optim.SGD([w_hat], lr=0.1)

    for e in range(5):
        loss = G_torch(w_hat, points)
        opt.zero_grad()
        loss.backward()
        opt.step()
    w_hat = w_hat[:, 0]
    return w_hat.detach().cpu()

def normalize(v):
    '''Normalize a vector based on its 2 norm.'''
    if 0 == np.linalg.norm(v):
        return v
    return v / np.linalg.norm(v)


def rotation_matrix_from_axis_and_angle(u, theta):
    '''Calculate a rotation matrix from an axis and an angle.'''
    x = u[0]
    y = u[1]
    z = u[2]
    s = np.sin(theta)
    c = np.cos(theta)

    return np.array([[c + x**2 * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y**2 * (1 - c), y * z * (1 - c) - x * s ],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z**2 * (1 - c) ]])


def rotation_matrix_from_axis_and_vector(u, w):
    '''Calculate a rotation matrix from an axis and an angle.'''
    x = u[0]
    y = u[1]
    z = u[2]
    s = np.linalg.norm(np.cross(w[:, 0], u))
    c = np.dot(u, w[:, 0])

    return np.array([[c + x**2 * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y**2 * (1 - c), y * z * (1 - c) - x * s ],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z**2 * (1 - c) ]])


def point_line_distance(p, l_p, Txy, Tz):
    '''Calculate the distance between a point and a line defined
    by a point and a direction vector.
    '''
    l_v = direction(Tz * np.pi / 180.0, Txy * np.pi / 180.0)
    l_v = normalize(l_v)
    u = p - l_p
    return np.linalg.norm(u - np.dot(u, l_v) * l_v)


def point_line_distance_torch(p_array, l_p, l_v):
    '''Calculate the distance between a point and a line defined
    by a point and a direction vector.
    '''
    u = p_array - l_p
    l_v = l_v.to(u.device)
    return torch.norm(u - torch.mm(torch.mm(u, l_v), l_v.t()), dim=1)


def direction(theta, phi):
    '''Return the direction vector of a cylinder defined
    by the spherical coordinates theta and phi.
    '''
    return np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta),
                     np.cos(theta)])


def projection_matrix(w):
    I_m = torch.eye(3).to(w.device)
    mult = torch.mm(w, w.t())
    return I_m - mult

def skew_matrix(w):
    '''Return the skew matrix of a direction w.'''
    return torch.tensor([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], device=w.device)

def calc_A(Ys):
    '''Return the matrix A from a list of Y vectors.'''
    return sum( [ torch.mm(Y, Y.t())
            for Y in Ys.split(1, dim=1)])

def calc_A_hat(A, S):
    '''Return the A_hat matrix of A given the skew matrix S'''
    return torch.mm(S, torch.mm(A, S.t()))

def preprocess_data(Xs_raw):
    '''Translate the center of mass (COM) of the data to the origin.
    Return the prossed data and the shift of the COM'''
    n = len(Xs_raw)
    Xs_raw_mean = sum(X for X in Xs_raw) / n

    return [X - Xs_raw_mean for X in Xs_raw], Xs_raw_mean

def G(w, Xs):
    '''Calculate the G function given a cylinder direction w and a
    list of data points Xs to be fitted.'''
    n = len(Xs)
    P = projection_matrix(w)
    Ys = [np.dot(P, X) for X in Xs]
    A = calc_A(Ys)
    A_hat = calc_A_hat(A, skew_matrix(w))

    
    u = sum(np.dot(Y, Y) for Y in Ys) / n
    v = np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(np.dot(A_hat, A))

    return sum((np.dot(Y, Y) - u - 2 * np.dot(Y, v)) ** 2 for Y in Ys)

def C(w, Xs):
    '''Calculate the cylinder center given the cylinder direction and 
    a list of data points.
    '''
    n = len(Xs)
    P = projection_matrix(w)
    Ys = [np.dot(P, X) for X in Xs]
    A = calc_A(Ys)
    A_hat = calc_A_hat(A, skew_matrix(w))

    return np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(np.dot(A_hat, A))

def r_numpy(w, Xs):
    '''Calculate the radius given the cylinder direction and a list
    of data points.
    '''
    n = len(Xs)
    P = projection_matrix(w)
    c = C(w, Xs)

    return np.sqrt(sum(np.dot(c - X, np.dot(P, c - X)) for X in Xs) / n)

def r(w, Xs, c):
    '''Calculate the radius given the cylinder direction and a list
    of data points.
    '''
    n = len(Xs)
    P = projection_matrix(w).to(Xs.device)
    Xs_off = Xs - c
    Xs_off = Xs_off.t().split(1, dim=1)
    return torch.sqrt(sum(torch.mm(X.t(),torch.mm(P, X)) for X in Xs_off) / n)

def H(c, Xs):
    '''Calculate the height given the cylinder center and a list
    of data points.
    '''
    distances = [np.sqrt(np.dot(X - c, X - c)) for X in Xs]
    return 1.5 * np.mean(distances)

def fit(data, guess_angles=None):
    '''Fit a list of data points to a cylinder surface. The algorithm implemented
    here is from David Eberly's paper "Fitting 3D Data with a Cylinder" from 
    https://www.geometrictools.com/Documentation/CylinderFitting.pdf

    Arguments:
        data - A list of 3D data points to be fitted.
        guess_angles[0] - Guess of the theta angle of the axis direction
        guess_angles[1] - Guess of the phi angle of the axis direction
    
    Return:
        Direction of the cylinder axis
        A point on the cylinder axis
        Radius of the cylinder
        Fitting error (G function)
    '''
    Xs, t = preprocess_data(data)  

    # Set the start points

    start_points = [(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)]
    if guess_angles:
        start_points = guess_angles

    # Fit the cylinder from different start points 

    best_fit = None
    best_score = float('inf')

    for sp in start_points:
        fitted = minimize(lambda x : G(direction(x[0], x[1]), Xs),
                    sp, method='Powell', tol=1e-6)

        if fitted.fun < best_score:
            best_score = fitted.fun
            best_fit = fitted

    w = direction(best_fit.x[0], best_fit.x[1])

    center = C(w, Xs)
    rad = r(w, Xs)

    Txy = np.arctan2(w[1], w[0]) * 180.0 / np.pi
    if(Txy < 0):
        Txy = 180 + Txy
    Tz = np.arccos(np.dot(w, np.array([0, 0, 1]))) * 180.0 / np.pi
    L = int(H(center, Xs).item())

    # exit()
    return center + t, rad, L, Txy, Tz, best_fit.fun 


def get_spatial_properties(coordinates, offset_coordinates, parameters):
    coordinates_split = coordinates.split(1, dim=1)

    # Get center
    center = coordinates.float().mean(0)

    # Get distances from center
    rs0 = torch.norm(coordinates - center, p=2, dim=1)

    # Find farthest distance from center
    end_point0_idx = (rs0 == rs0.max()).nonzero()
    end_point0_idx = end_point0_idx[0, 0]

    # Get endpoint 0
    end_point0 = torch.tensor([coordinates_split[0][end_point0_idx], coordinates_split[1][end_point0_idx], coordinates_split[2][end_point0_idx]], device=coordinates.device)

    # Find closes points from end point 0
    rs1 = torch.norm(coordinates - end_point0, p=2, dim=1)
    end_point1_idx = (rs1 < 3).nonzero()
    # end_point1_idx = end_point1_idx[:, 0]

    # Get formally end point1
    end_point1 = torch.tensor([coordinates_split[i][end_point1_idx][:, 0].mean() for i in range(3)])

    # Find farthest point from end point 1
    rs2 = torch.norm(coordinates - end_point0, p=2, dim=1)
    # Find farthest point from end point 1
    end_point2_idx = (rs2 > rs2.max() - 3).nonzero()
    # end_point2_idx = end_point2_idx[:, 0]

    # Get formally end point2
    end_point2 = torch.tensor([coordinates_split[i][end_point2_idx][:, 0].mean() for i in range(3)])

    direction = (end_point1 - end_point2)
    direction = direction / torch.norm(direction, p=2)

    # direction = direction.unsqueeze(1).to(center.device)
    # direction.requires_grad = True
    # direction = get_angle_refined(direction, center, coordinates)

    w = direction.unsqueeze(1)

    Length = torch.norm((end_point1 - end_point2))
    Length = Length.clamp(parameters.mpp_min_l, parameters.mpp_max_l)
    Length = Length.cpu().int().numpy().item()

    r = point_line_distance_torch(coordinates, center, w).mean().cpu()
    r = r.clamp(parameters.mpp_min_r, parameters.mpp_max_r)
    offset_center = offset_coordinates.float().mean(0)
    return offset_center, r, Length, w, 0


def get_angle_w(coordinates):
    coordinates_split = coordinates.split(1, dim=1)

    # Get center
    center = coordinates.float().mean(0)

    # Get distances from center
    rs0 = torch.norm(coordinates - center, p=2, dim=1)

    # Find farthest distance from center
    end_point0_idx = (rs0 == rs0.max()).nonzero()
    end_point0_idx = end_point0_idx[0, 0]

    # Get endpoint 0
    end_point0 = torch.tensor([coordinates_split[0][end_point0_idx], coordinates_split[1][end_point0_idx], coordinates_split[2][end_point0_idx]], device=coordinates.device)

    # Find closes points from end point 0
    rs1 = torch.norm(coordinates - end_point0, p=2, dim=1)
    end_point1_idx = (rs1 < 3).nonzero()
    # end_point1_idx = end_point1_idx[:, 0]

    # Get formally end point1
    end_point1 = torch.tensor([coordinates_split[i][end_point1_idx][:, 0].mean() for i in range(3)])

    # Find farthest point from end point 1
    rs2 = torch.norm(coordinates - end_point0, p=2, dim=1)
    # Find farthest point from end point 1
    end_point2_idx = (rs2 > rs2.max() - 3).nonzero()
    # end_point2_idx = end_point2_idx[:, 0]

    # Get formally end point2
    end_point2 = torch.tensor([coordinates_split[i][end_point2_idx][:, 0].mean() for i in range(3)])

    w = (end_point1 - end_point2)
    w = w / torch.norm(w, p=2)
    return w
