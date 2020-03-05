import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
import torch
import h5py
import os
import matplotlib.pyplot as plt


#################### #################### Save Images #################### #################### ####################
def save_subvolume(V, path, scale=1, start=1, end=None):
    print('Saving Volumw at ' + path)
    if not os.path.isdir(path):
        os.mkdir(path)

    trans = transforms.ToPILImage()
    if(scale > 1):
        V = F.interpolate(V, scale_factor=scale, mode='nearest')
    if(len(V.size()) > 4):
        V = V[0, :, :, :, :]
    device = torch.device("cpu")
    V = V.to(device)
    size = V.size()

    if(end is None):
        end = size[-1]

    for i in range(start, end):
        img = V[:, :, :, i - start]
        img = trans(img)
        if(i  > 999):
            img.save(path + "/subV_" + str(i) + ".tif")
        elif(i  > 99):
            img.save(path + "/subV_0" + str(i) + ".tif")
        elif(i  > 9):
            img.save(path + "/subV_00" + str(i) + ".tif")
        else:
            img.save(path + "/subV_000" + str(i ) + ".tif")


def save_subvolume_instances(V, M, path,  start=0, end=None):
    print('Saving Instances at ' + path)
    if not os.path.isdir(path):
        os.mkdir(path)

    trans = transforms.ToPILImage()
    # V is initially as channels x rows x cols x slices
    # Make it slices x rows x cols x channels
    # V = V.permute(3, 1, 2, 0)
    if(len(V.size()) > 4):
        V = V[0, :, :, :, :]

    if(len(M.size()) > 4):
        M = M[0, :, :, :, :]

    device = torch.device("cpu")
    V = V.to(device)
    M = M.to(device)
    num_classes = M.max().int().item()
    colors_r = torch.rand(num_classes)
    colors_g = torch.rand(num_classes)
    colors_b = torch.rand(num_classes)
    
    if(end is None):
        end = V.shape[-1]
    for i in range(start, end):
        img = V[:, :, :, i - start]
        mask = M[:, :, :, i - start]

        overlay = torch.cat([img, img, img], 0)
        for c in mask.unique():
            if(c == 0):
                continue
            indxs = (mask[0, :, :] == c).nonzero()
            if(c == 1):
                for idx in indxs:
                    overlay[0, idx[0], idx[1]] = 1
                    overlay[1, idx[0], idx[1]] = 1
                    overlay[2, idx[0], idx[1]] = 1
            else:
                for idx in indxs:
                    overlay[0, idx[0], idx[1]] = colors_r[c - 1]
                    overlay[1, idx[0], idx[1]] = colors_g[c - 1]
                    overlay[2, idx[0], idx[1]] = colors_b[c - 1]

        # overlay[0, :, :] += 2 * mask[0, :, :].clamp(0, 1)
        # overlay[1, :, :] += 2 * (mask[0, :, :] - 1).clamp(0, 1)
        # overlay = overlay.clamp(min_v, max_v)
        img = trans(overlay)

        if(i > 999):
            img.save(path + "/subV_" + str(i) + ".tif")
        elif(i > 99):
            img.save(path + "/subV_0" + str(i) + ".tif")
        elif(i > 9):
            img.save(path + "/subV_00" + str(i) + ".tif")
        else:
            img.save(path + "/subV_000" + str(i) + ".tif")


def save_subvolume_color(V, M, path, num_classes=3, scale=1, start=1, end=None):
    if not os.path.isdir(path):
        os.mkdir(path)

    # resize_vol = transforms.Resize([450, 450], interpolation=2)
    # resize_mask = transforms.Resize([450,450], interpolation=Image.NEAREST)
    if(scale > 1):
        V = F.interpolate(V, scale_factor=scale, mode='trilinear', align_corners=True)
        M = F.interpolate(M, scale_factor=scale, mode='nearest')

    trans = transforms.ToPILImage()
    # V is initially as channels x rows x cols x slices
    # Make it slices x rows x cols x channels
    # V = V.permute(3, 1, 2, 0)
    if(len(V.size()) > 4):
        V = V[0, :, :, :, :]

    if(len(M.size()) > 4):
        M = M[0, :, :, :, :]

    device = torch.device("cpu")
    V = V.to(device)
    M = M.to(device)
    size = V.size()
    colors_r = [  0, 0.5, 1,   0,  0.5,  1,   0, 0.5, 1,   1]
    colors_g = [1 , 0.5,   0,  1, 1,  1, 0.5, 0.5,   0,   1]
    colors_b = [  0, 1, 1,  0.5,    0,   0,   0,   0,   0, 1]
    #    colors_r = [  0  ,  0,   0,  0,    0,  0,     0.5, 0.5,  0.5,    0.5]
    #    colors_g = [  0  ,  0, 0.5,  1,    1,  1,     0.5,   1,    1,      1]
    #    colors_b = [  0.5,  1, 0.5,  0,  0.5,  1,     0.5,   0,  0.5,      1]
    if(end is None):
        end = size[-1] - 1

    for i in range(start, end + 1):
        img = V[:, :, :, i - start]
        mask = M[:, :, :, i - start]
        # img = resize_vol(img)
        # mask = resize_mask(img)
        overlay = torch.cat([img, img, img], 0)
        for c in range(1, num_classes + 1):
            indxs = (mask[0, :, :] == c).nonzero()
            for idx in indxs:
                overlay[0, idx[0], idx[1]] = colors_r[c-1]
                overlay[1, idx[0], idx[1]] = colors_g[c-1]
                overlay[2, idx[0], idx[1]] = colors_b[c-1]

        # overlay[0, :, :] += 2 * mask[0, :, :].clamp(0, 1)
        # overlay[1, :, :] += 2 * (mask[0, :, :] - 1).clamp(0, 1)
        # overlay = overlay.clamp(min_v, max_v)
        img = trans(overlay)

        if(i > 999):
            img.save(path + "/subV_" + str(i) + ".tif")
        elif(i > 99):
            img.save(path + "/subV_0" + str(i) + ".tif")
        elif(i > 9):
            img.save(path + "/subV_00" + str(i) + ".tif")
        else:
            img.save(path + "/subV_000" + str(i) + ".tif")


def load_full_volume(path, start=0, end=None, scale=2):
    list_of_names = sorted(os.listdir(path))
    num_Z = len(list_of_names)
    # Update End Value
    if(end is None):
        end = num_Z
    print("Reading image indexes: " + str(start) + ":" + str(end) + "/" + str(num_Z))
    # Read and crop first image
    im = Image.open(path + '/' + list_of_names[0])
    im = torch.from_numpy(np.array(im, np.int16, copy=False)).unsqueeze(0)
    im = im[:, 200:-311, 280:-231]
    im = im[:, ::scale, ::scale]
    size = im.size()
    V = torch.zeros(size[0], size[1], size[2], (end - start + 1) // scale)
    countZ = 0

    # Read and crop all images
    for i in range(start, end + 1, scale):
        name = list_of_names[i]
        if name[-1] == 'f':
            im = Image.open(path + '/' + name)
            im = np.asarray(im, np.uint16)
            im = im[200:-311, 280:-231]
            im = im.astype(np.float)
            im = im / 2**16
            # im = im.astype(np.float)
            im = torch.from_numpy(im).unsqueeze(0)
            im = im[:, ::scale, ::scale]
            V[:, :, :, countZ] = im
            countZ += 1
    return V

def load_full_volume_crop(path, sx, sy, sz, window_size):
    list_of_names = sorted(os.listdir(path))
    num_Z = len(list_of_names)

    start = sz
    # Update End Value
    end = sz + window_size
    print("Reading image indexes: " + str(start) + ":" + str(end) + "/" + str(num_Z))
    # Read and crop first image
    im = Image.open(path + '/' + list_of_names[0])
    im = torch.from_numpy(np.array(im, np.int16, copy=False)).unsqueeze(0)
    im = im[:, 200:-311, 280:-231]
    size = im.size()
    V = torch.zeros(size[0], window_size, window_size, window_size)
    countZ = 0

    # Read and crop all images
    for i in range(window_size):
        name = list_of_names[i + sz]
        if name[-1] == 'f':
            im = Image.open(path + '/' + name)
            im = np.asarray(im, np.uint16)
            im = im[200:-311, 280:-231]
            im = im.astype(np.float)
            im = im / 2**16
            im = im[sx:sx + window_size, sy:sy + window_size]
            im = torch.from_numpy(im).unsqueeze(0)
            V[:, :, :, countZ] = im
            countZ += 1
    return V

def load_volume_uint16(path, start=0, end=None, scale=2):
    print('Loading Instances from ' + path)
    list_of_names = sorted(os.listdir(path))
    num_Z = len(list_of_names)
    # Update End Value
    if(end is None):
        number = num_Z
        end = num_Z
    else:
        number = (end - start + 1)
    # Read and crop first image
    im = Image.open(path + '/' + list_of_names[0])
    im = torch.from_numpy(np.array(im, np.int16, copy=False)).unsqueeze(0)
    im = im[:, ::scale, ::scale]
    size = im.size()
    V = torch.zeros(size[0], size[1], size[2], number // scale)
    countZ = 0
    # Read and crop all images
    for i in range(start, end , scale):
        name = list_of_names[i]
        if name[-1] == 'f':
            im = Image.open(path + '/' + name)
            im = np.asarray(im, np.uint16)
            im = im.astype(np.float)
            # im = im / 2**16
            # im = im.astype(np.float)
            im = torch.from_numpy(im).unsqueeze(0)
            im = im[:, ::scale, ::scale]
            V[:, :, :, countZ] = im
            countZ += 1
    return V


def load_fibers_uint16(path, start=0, end=None, scale=2):
    print('Loading Instances from ' + path + " at scale:" + str(scale))
    list_of_names = sorted(os.listdir(path))
    num_Z = len(list_of_names)
    # Update End Value
    if(end is None):
        number = num_Z
        end = num_Z
    else:
        number = (end - start + 1)
    # Read and crop first image
    im = Image.open(path + '/' + list_of_names[0])
    im = torch.from_numpy(np.array(im, np.int16, copy=False)).unsqueeze(0)
    im = im[:, ::scale, ::scale]
    size = im.size()
    V = torch.zeros(size[0], size[1], size[2], number // scale)
    countZ = 0
    # Read and crop all images
    for i in range(start, end , scale):
        name = list_of_names[i]
        if name[-1] == 'f':
            im = Image.open(path + '/' + name)
            im = np.asarray(im, np.uint16)
            im = im.astype(np.float)
            im = im / 2.0**16
            # im = im.astype(np.float)
            im = torch.from_numpy(im).unsqueeze(0)
            im = im[:, ::scale, ::scale]
            V[:, :, :, countZ] = im
            countZ += 1
    return V


def load_volume(path, scale=2):
    print('Loading Volume from ' + path)
    make_tensor = transforms.ToTensor()
    list_of_names = sorted(os.listdir(path))
    num_Z = len(list_of_names)
    im = Image.open(path + '/' + list_of_names[0])
    im = make_tensor(im)
    im = im[:, ::scale, ::scale]
    size = im.size()
    V = torch.zeros(size[0], size[1], size[2], num_Z // scale)
    countZ = 0
    for i in range(0, len(list_of_names) - 1, scale):
        name = list_of_names[i]
        if name[-1] == 'f':
            im = Image.open(path + '/' + name)
            im = make_tensor(im)
            im = im[:, ::scale, ::scale]
            V[:, :, :, countZ] = im
            countZ += 1
    return V

#################### #################### H5 Files #################### #################### ####################

def save_volume_h5(V, name='Volume', dataset_name='Volume', directory='./h5_files'):
    if not os.path.isdir(directory):
        os.mkdir(directory)

    with h5py.File(directory + "/" + name + '.h5', 'w') as f:
        dset = f.create_dataset(dataset_name, data=V, maxshape=(V.shape[0], V.shape[1], None))

    create_xml_file(V.shape, directory, name, dataset_name)


def append_volume_h5(V, name='Volume', dataset_name='Volume', directory='./h5_files'):
   
    with h5py.File(directory + "/" + name + ".h5", 'a') as f:
        V_rows, V_cols, V_depth  = V.shape
        old_rows, old_cols, old_depth = f[dataset_name].shape

        if(V_rows != old_rows or V_cols != old_cols):
            print("Dataset Append Error: Dimensions must match")
            return

        new_depth = old_depth + V_depth

        f[dataset_name].resize(new_depth, axis=2)
        f[dataset_name][:, :, -V_depth:] = V

    new_dims = [old_rows, old_cols, new_depth]
    create_xml_file(new_dims, directory, name, dataset_name)


def create_xml_file(volume_shape, directory, name, dataset_name):
    Nx, Ny, Nz = volume_shape
    xml_filename = directory + "/" + name + ".xmf"
    f = open(xml_filename, 'w')
    # Header for xml file
    f.write('''<?xml version="1.0" ?>
            <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
            <Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">
            <Domain>
            ''')

    #  Naming datasets
    dataSetName1 = name +'.h5:/' + dataset_name

    f.write('''
                <Grid Name="Box Test" GridType="Uniform"> #
                <Topology TopologyType="3DCORECTMesh" Dimensions="%d %d %d"/>
                <Geometry GeometryType="ORIGIN_DXDYDZ">
                <DataItem Name="Origin" DataType="Float" Dimensions="3" Format="XML">0.0 0.0 0.0</DataItem>
                <DataItem Name="Spacing" DataType="Float" Dimensions="3" Format="XML">1.0 1.0 1.0</DataItem>
                </Geometry>
                ''' % (Nx, Ny, Nz))

    f.write('''\n
                <Attribute Name="S" AttributeType="Scalar" Center="Node">
                <DataItem Format="HDF" Dimensions="%d %d %d" NumberType="UInt" Precision="2"
                >%s
                </DataItem>
                </Attribute>
                ''' % (Nx, Ny, Nz, dataSetName1))

    # End the xmf file
    f.write('''
       </Grid>
    </Domain>
    </Xdmf>
    ''')

    f.close()


def read_volume_h5(name='Volume', dataset_name='Volume', directory='./h5_files'):
    filename = directory + "/" + name + ".h5"
    f = h5py.File(filename, 'r')
    a_group_key = list(f.keys())[0]
    data = f[a_group_key][()]
    return data


def save_images_of_h5_only(h5_volume_dir, output_path, volume_h5_name='Volume', volume_voids_h5="outV", start=0, end=None, scale=2):
    #if(end <= start)
    #Vv = read_volume_h5(volume_voids_h5, volume_voids_h5, h5_volume_dir)
    #Vv = torch.from_numpy(Vv[:, :, start:end].astype(np.long)).unsqueeze(0)

    Vf = read_volume_h5(volume_h5_name, volume_h5_name, h5_volume_dir)
    Vv = read_volume_h5(volume_voids_h5, volume_voids_h5, h5_volume_dir)

    print(Vf.shape)
    print(Vv.shape)
    Vf[np.where(Vv == 2)] = 1
    
    if(end is None):
        end = Vf.shape[-1]

    Vf = torch.from_numpy(Vf[:, :, start:end].astype(np.long)).unsqueeze(0)

    save_subvolume_instances(torch.zeros(Vf.shape), Vf.long(), output_path)#, start=start, end=end)

def save_images_of_h5(h5_volume_dir, data_volume_path, output_path, volume_h5_name='Volume', volume_voids_h5="outV", start=0, end=None, scale=2):
    #if(end <= start)
    #Vv = read_volume_h5(volume_voids_h5, volume_voids_h5, h5_volume_dir)
    #Vv = torch.from_numpy(Vv[:, :, start:end].astype(np.long)).unsqueeze(0)

    Vf = read_volume_h5(volume_h5_name, volume_h5_name, h5_volume_dir)
    print(Vf.shape)
    #Vf = read_volume_h5('outV', "voids", h5_volume_dir)
    if(end is None):
        end = Vf.shape[-1]

    Vf = torch.from_numpy(Vf[:, :, start:end].astype(np.long)).unsqueeze(0)

    #Vf[np.where(Vv == 1)] = 1

    data_volume = load_full_volume(data_volume_path, start * scale, end * scale - 1, scale=scale)
    print("Results Shape: {} ".format(Vf.shape))
    print("Volume Shape: {} ".format(data_volume.shape))

    #data_volume = data_volume[:, :, 500, :].unsqueeze(3)
    #Vf = Vf[:, :, 500, :].unsqueeze(3)
    #print(data_volume.shape)
    # exit()

    save_subvolume_instances(data_volume, Vf.long() * 0, output_path + "_originals")
    save_subvolume_instances(data_volume, Vf.long(), output_path)#, start=start, end=end)

def save_images_of_h5_side(h5_volume_dir, data_volume_path, output_path, volume_h5_name='Volume', volume_voids_h5="outV", start=0, end=None, scale=2):
    #if(end <= start)
    #Vv = read_volume_h5(volume_voids_h5, volume_voids_h5, h5_volume_dir)
    #Vv = torch.from_numpy(Vv[:, :, start:end].astype(np.long)).unsqueeze(0)

    if(end is None):
        end = start + 5
    Vf = read_volume_h5(volume_h5_name, volume_h5_name, h5_volume_dir)
    
    Vf = torch.from_numpy(Vf.astype(np.long)).unsqueeze(0)
    slices = Vf.shape[-1]
    Vf = torch.transpose(Vf, 3, 2)
    Vf = Vf[..., start:end]

    #Vf[np.where(Vv == 1)] = 1

    data_volume = load_full_volume(data_volume_path, 0, (slices * scale) - 1, scale=scale)
    data_volume[:, ::scale, ::scale, ::scale]

    data_volume = torch.transpose(data_volume, 3, 2)
    data_volume = data_volume[..., start:end]

    print(Vf.shape)
    print(data_volume.shape)
    #print(data_volume.shape)
    # exit()

    save_subvolume_instances(data_volume, Vf.long(), output_path)#, start=start, end=end)
    save_subvolume_instances(data_volume, Vf.long() * 0, output_path + "_originals")


#################### #################### Cropping Utils #################### #################### ####################

def random_crop_3D_image_batched(img, mask, crop_size):
    if(type(crop_size) not in (tuple, list)):
        crop_size = [crop_size] * (len(img.shape) - 2)
    else:
        assert len(crop_size) == (len(img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    if crop_size[0] < img.shape[2]:
        lb_x = np.random.randint(0, img.shape[2] - crop_size[0])
    elif crop_size[0] == img.shape[2]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < img.shape[3]:
        lb_y = np.random.randint(0, img.shape[3] - crop_size[1])
    elif crop_size[1] == img.shape[3]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    if crop_size[2] < img.shape[4]:
        lb_z = np.random.randint(0, img.shape[4] - crop_size[2])
    elif crop_size[2] == img.shape[4]:
        lb_z = 0
    else:
        raise ValueError("crop_size[2] must be smaller or equal to the images z dimension")
    V = img[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]
    Mask = mask[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]
    return (V, Mask)

def random_crop_3D_image_batched2(img, mask, directions, crop_size):
    if(type(crop_size) not in (tuple, list)):
        crop_size = [crop_size] * (len(img.shape) - 2)
    else:
        assert len(crop_size) == (len(img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    if crop_size[0] < img.shape[2]:
        lb_x = np.random.randint(0, img.shape[2] - crop_size[0])
    elif crop_size[0] == img.shape[2]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < img.shape[3]:
        lb_y = np.random.randint(0, img.shape[3] - crop_size[1])
    elif crop_size[1] == img.shape[3]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    if crop_size[2] < img.shape[4]:
        lb_z = np.random.randint(0, img.shape[4] - crop_size[2])
    elif crop_size[2] == img.shape[4]:
        lb_z = 0
    else:
        raise ValueError("crop_size[2] must be smaller or equal to the images z dimension")
    V = img[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]
    Mask = mask[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]
    Dirs = directions[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]
    return (V, Mask, Dirs)

def full_crop_3D_image_batched(img, mask, lb_x, lb_y, lb_z, crop_size):
    V = img[:, :, lb_x:lb_x + crop_size, lb_y:lb_y + crop_size, lb_z:lb_z + crop_size]
    Mask = mask[:, :, lb_x:lb_x + crop_size, lb_y:lb_y + crop_size, lb_z:lb_z + crop_size]
    return (V, Mask)


#################### #################### Filters #################### #################### ####################

def create_histogram_ref(reference):
    reference = reference.numpy()
    tmpl_values, tmpl_counts = np.unique(reference.ravel(), return_counts=True)
    tmpl_quantiles = np.cumsum(tmpl_counts).astype(np.float) / (reference.size)
    print("Creating Reference...")
    reference_hist = [tmpl_values, tmpl_quantiles]
    with open('info_files/histogram_reference.pickle', 'wb') as f:
        pickle.dump(reference_hist, f)

    return (reference_hist)


def normalize_dataset(vol):
    '''
        Normalize dataset to zero mean and unit variance
    '''
    mu = vol.mean()
    std = vol.std()
    vol = (vol - mu) / std
    return vol

def normalize_dataset_w_info(vol):
    '''
        Normalize dataset to zero mean and unit variance
    '''
    mu = vol.mean()
    std = vol.std()
    vol = (vol - mu) / std
    return (vol, mu, std)


def clean_noise(vol, data_path):
    '''
        vol must be tensor of shape [height, width, depth]
    '''
    vol = vol.numpy()
    clean_vol = np.zeros(vol.shape)

    try:
        tmpl_values, tmpl_quantiles =  pickle.load( open('info_files/histogram_reference.pickle', "rb" ) )
    except:
        # Keep preset values
        print("WARNING: DID NOT FIND REFERENCE HISTOGRAM...RESULTS MAY LOOK UGLY")
        reference =  load_full_volume(data_path, 0 ,5)
        tmpl_values, tmpl_quantiles = create_histogram_ref(reference[0, ...])    
    
    print("Adapting Sample for Neural Net")
    for i in range(vol.shape[-1]):
        source = vol[..., i]

        src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)

        src_quantiles = np.cumsum(src_counts).astype(np.float) / (source.size)
        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        matched = interp_a_values[src_unique_indices].reshape(source.shape)
        clean_vol[..., i] = matched
    clean_vol = torch.tensor(clean_vol)
    return clean_vol

def cylinder_filter(data_volume_shape, center=None, radius=None):
    [rows, cols, slices] = data_volume_shape
    grid = np.mgrid[[slice(i) for i in [rows, cols]]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum((grid)**2, 0))
    res = (phi > 0).astype(np.float)
    res = np.repeat(res[:, :, np.newaxis], slices , axis=2)
    return res
    
################################################## Embedded Files ####################################    
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

    unique_labels = torch.unique(labeled_pixels)
    N_objects = len(unique_labels)
    mu_vector = torch.zeros(N_objects, N_embedded)

    for c in range(N_objects):
        fiber_id = unique_labels[c]
        idx_c = (labeled_pixels == fiber_id).nonzero()

        # xi vector
        x_i = torch.gather(embeddings, 0, idx_c.repeat(1, N_embedded))

        # get mu
        mu = x_i.mean(0)
        mu_vector[c, :] = mu
    resta = mu_vector[c, :] - x_i.cpu()
    lv_term = torch.norm(resta, 2) - 0

    lv_temp2 = torch.norm(resta, 2, dim=1)
    ff = 0
    for k in range(x_i.shape[0]):
        ff += torch.norm(resta[k, :], 2)

    mu_vector = mu_vector.detach().cpu().numpy()
    # Y = tsne(object_pixels[::4, :].detach().numpy(), num_display_dims, num_dims, tsne_lr)
    # Y = pca(object_pixels.detach().numpy(), no_dims=2).real
    from tsnecuda import TSNE
    # from tsne import tsne
    X = object_pixels.cpu().detach().numpy()
    Y = TSNE(n_components=2, perplexity=20, learning_rate=50).fit_transform(X)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(Y[:, 0], Y[:, 1], 5, labeled_pixels, cmap='tab20b')
    if iteration is None:
        iteration = 0
    plt.savefig("low_dim_embeeding/embedded_%d.png" % iteration)
    plt.close(fig)

    fig = pylab.figure()
    ax1 = fig.add_subplot(221)
    ax1.scatter(X[:, 0], X[:, 1], 5, labeled_pixels, cmap='tab20b')
    ax1.scatter(mu_vector[:, 0], mu_vector[:, 1], 40, unique_labels, cmap='tab20b', marker="x")
    ax1 = fig.add_subplot(222)
    ax1.scatter(X[:, 0], X[:, 2], 5, labeled_pixels, cmap='tab20b')
    ax1.scatter(mu_vector[:, 0], mu_vector[:, 2], 40, unique_labels, cmap='tab20b', marker="x")
    ax1 = fig.add_subplot(223)
    ax1.scatter(X[:, 0], X[:, 3], 5, labeled_pixels, cmap='tab20b')
    ax1.scatter(mu_vector[:, 0], mu_vector[:, 3], 40, unique_labels, cmap='tab20b', marker="x")
    ax1 = fig.add_subplot(224)
    ax1.scatter(X[:, 0], X[:, 4], 5, labeled_pixels, cmap='tab20b')
    ax1.scatter(mu_vector[:, 0], mu_vector[:, 4], 40, unique_labels, cmap='tab20b', marker="x")
    plt.savefig("2_embedding/embedded_%d.png" % iteration)
    plt.close(fig)

    if detected_labels is not None:
        pylab.scatter(Y[:, 0], Y[:, 1], 5, detected_labels, cmap='tab20b')
        pylab.savefig("low_dim_embeeding/embedded_%d.png" % (iteration + 1000))
        pylab.close(fig)


def create_tiling(V, cube_size, percent_overlap):
    print("Starting Tiling")
    (rows, cols, depth) = V.shape
    T_volume = np.zeros(V.shape)
    overlap = int((1 - percent_overlap) * cube_size)
    st = 0 + cube_size 
    starting_points_x = []
    starting_points_y = []
    starting_points_z = []
    while(st + cube_size < rows):
        starting_points_x.append(st)
        st = st + overlap
    starting_points_x.append(rows - cube_size)

    st = 0 + cube_size 
    while(st + cube_size < cols):
        starting_points_y.append(st)
        st = st + overlap
    starting_points_y.append(cols - cube_size)

    st = 0 + cube_size 
    while(st + cube_size < depth):
        starting_points_z.append(st)
        st = st + overlap
    starting_points_z.append(depth - cube_size)

    print("Here")
    counter = 1
    for lb_z in starting_points_z:
        T_volume[0:rows, 0:cols, lb_z] = counter
        if(lb_z + cube_size < depth):
            T_volume[0:rows, 0:cols, lb_z + cube_size] = counter
        counter = counter + 1

    for lb_y in starting_points_y:
        T_volume[0:rows, lb_y, 0:depth] = counter
        if(lb_y + cube_size < cols):
            T_volume[0:rows, lb_y + cube_size, 0:depth] = counter
        counter = counter + 1

    for lb_x in starting_points_x:
        T_volume[lb_x, 0:cols, 0:depth] = counter
        if(lb_x + cube_size < rows):
            T_volume[lb_x + cube_size, 0:cols, 0:depth] = counter
        counter = counter + 1

    save_volume_h5(T_volume, "tiling_for_ws_" + str(cube_size) + "ovlp_" + str(percent_overlap), "tiling_for_ws_" + str(cube_size) + "ovlp_" + str(percent_overlap))


def refine_connected(labels):
    device = labels.device
    labels = labels.cpu().numpy()
    labels_t = np.zeros(labels.shape)
    num_labels = labels.max().astype(np.int)
    counter = 2
    for c in range(2, num_labels + 1):
        im = labels == c
        temp_labels, temp_nums = measure.label(im, return_num=True)


        for points in range(1, temp_nums + 1):
            idx_c = np.where(temp_labels == points)
            print(len(idx_c[0]))
            if(len(idx_c[0]) > 10):
                labels_t[idx_c] = counter
                counter = counter + 1

    labels = torch.from_numpy(labels_t).long().to(device)
    return (labels)

def crop_random_volume(data_path, coords=[100, 100, 100], window_size=512, output_path=None):
    if(output_path is None):
        output_path = 'volume_at_' + str(coords[0]) + '_' + str(coords[1]) + '_' + str(coords[2])
    if(type(coords) == int):
        sx = coords
        sy = coords
        sz = coords
    else:
        sx = coords[0]
        sy = coords[1]
        sz = coords[2]

    V = load_full_volume_crop(data_path, sx, sy, sz, window_size)
    save_subvolume(V, output_path)

if __name__ == '__main__':
    print("Starting")
    from scipy import misc
    # a = 
    # V = read_volume_h5('final_fibers_single_unet', 'final_fibers_single_unet')
    #V = torch.from_numpy(V).unsqueeze(0)

    V = load_volume_uint16('im_uint_600_results_transformed', scale=2)
    print(V.shape)
    save_subvolume_instances((V * 0).float(), V.long(), 'instances_for_display_mpp')
    exit()
    #from skimage import measure
    #read_dictionary("from_meanpill/dict.txt")
    '''
    from scipy import misc
    # a = misc.imresize(a, 2, interp='nearest')
    
    data_path = 'TRAINING_DATA'
    # mask_path = 'TRAINING_LABELS'
    mask_path = "updated_fibers/UPDATED_TRAINING_LABELS"
    masks = load_volume_uint16(mask_path, scale=2)
    masks = masks / masks.max()

    save_subvolume(masks, "test_writting_uint16", scale=1, start=1, end=None)
    
    '''

    '''
    V = load_volume("process_all_instances", scale=1)
    print(V.shape)
    V = V.transpose(2, 3)
    save_subvolume(V, "process_all_instances_t2")
    '''
    '''
    data_path = '/Storage/DATASETS/Fibers/Tiff_files_tomo_data'
    crop_random_volume(data_path, coords=[1300, 1300, 600], window_size=512, output_path=None)
    '''


    V = read_volume_h5('volume_fiber_voids_labeled_voids', 'volume_fiber_voids_labeled_voids', 'h5_statistics')
    print(V.shape)
    V = V[200:400, 200:400, 200:400]
    V = save_volume_h5(V, "Volume_1", "Volume1", "./")

    '''
    V = V[0:100, 0:100, 0:100]
    print("Getting Fiber Properties")
    list_2 = fit_all_torch(torch.from_numpy(V).unsqueeze(0).unsqueeze(0).cuda())

    centers, fiber_ids, end_points, fiber_list= get_fiber_properties(torch.from_numpy(V).cuda())

    f = open("dict_my_way.txt", "w")
    for k in fiber_list.keys():
        el = fiber_list[k]
        w_fit = np.array([el[6], el[7], el[8]])
        Txy = np.arctan2(w_fit[1], w_fit[0]) * 180 / np.pi
        if(Txy < 0):
            Txy = 180 + Txy
        Tz = np.arccos(np.dot(w_fit, np.array([0, 0, 1])) / np.linalg.norm(w_fit, 2)) * 180 / np.pi

        f.write("{},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f},{:.3f},{:.3f}\n".format(el[0],el[1],el[2],el[3],el[4],el[5],Txy, Tz))
    f.close()

    print("Getting Fiber Properties 2 ")
    list_1 = fit_all_fibers_parallel(V)


    

    f = open("dict_other_way.txt", "w")
    for k in range(len(list_1)):
        el = list_1[k]
        el3 = list_2[k]
        el2 = fiber_list[el[0]]
        w_fit = np.array([el2[6], el2[7], el2[8]])
        Txy = np.arctan2(w_fit[1], w_fit[0]) * 180 / np.pi
        if(Txy < 0):
            Txy = 180 + Txy
        Tz = np.arccos(np.dot(w_fit, np.array([0, 0, 1])) / np.linalg.norm(w_fit, 2)) * 180 / np.pi

        f.write("{},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f},  {:.0f},  {:.0f}\n".format(el2[0],el2[1],el2[2],el2[3],el2[4],el2[5],Txy, Tz))
        f.write("{},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f},  {:.0f},  {:.0f},     {:.3f}\n".format(el[0],el[1],el[2],el[3],el[4],el[5],el[6],el[7],el[8]))
        f.write("{},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f},  {:.0f},  {:.0f},     {:.3f}\n\n".format(el3[0],el3[1],el3[2],el3[3],el3[4],el3[5],el3[6],el3[7],el3[8]))
    f.close()
    '''


    #save_images_of_h5('from_meanpill', data_path, 'converted_from_h5', volume_h5_name='final_fibers', start=0, end=200)

    '''
    '''
    '''
    print(a[0].mean())
    print(a[1].mean())
    print(a[2].mean())
    V = torch.from_numpy(V.astype(np.int16)).unsqueeze(0)
    save_subvolume(V, "upsampled")
    print(V[0, 25,196, 2])
    '''
    #data_path = "/pub2/aguilarh/DATASETS/Tiff_files_tomo_data"
    #massive_fitting('h5_files',  volume_h5_name='final_fibers', start=0, end=100)
    #save_images_of_h5('h5_files', data_path, 'converted_from_h5_3', volume_h5_name='final_fibers0')#, start=0, end=20)
    #save_images_of_h5_voids(data_path, 'converted_from_h5', 'h5_files', 'final_fibers', "../../NN_TAPEOUT/h5_files", volume_voids_h5="outV", start=0, end=50)
    #debug_merge_volume_z()

    #save_images_of_h5('from_meanpill', data_path, 'converted_from_h5', volume_h5_name='final_fibers', start=0, end=200)
    #displaying_statistics('statistics', 'final_fibers', 'voids')

    #save_images_of_h5_only('h5_files', "comer", volume_h5_name='simple_fibers_angle_30_p2', volume_voids_h5="simple_voids")

    # create_tiling(np.zeros([256, 256, 256]), 64, 0.2)
    '''
    V = read_volume_h5('simple_fibers_merged','simple_fibers_merged','h5_files')
    print(V.shape)
    Z = np.zeros(V.shape)
    Z[np.where(V == 336)] = 2
    # 
    Z = torch.from_numpy(Z)
    Z = refine_connected(Z)
    Z = Z.numpy()
    # print(A == Z)

    print("Here")

    temp_labels, temp_nums = measure.label(Z, return_num=True)
    print(temp_nums)
    '''

