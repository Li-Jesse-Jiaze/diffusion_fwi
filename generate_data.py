import os
import sys
from pathlib import Path
import numpy as np
import h5py
import copy
import subprocess

from skimage.transform import resize
from matplotlib import pyplot as plt
os.environ['PATH'] += os.pathsep + '/usr/local/cuda/bin'

data_dir = Path("./extra_data")
bscan_dir = data_dir / "Bscan"
labels_dir = data_dir / "Labels"
data_dir.mkdir(exist_ok=True)
bscan_dir.mkdir(exist_ok=True)
labels_dir.mkdir(exist_ok=True)

def write_file(fil):
    file_path = data_dir / fil
    with file_path.open('w') as f:
        # Define the domain of the problem
        # 2D with 50.02x26 dimensions.
        # z dimension equals with spatial step i.e. 2D geometry
        f.write('#domain: 50.02 26 0.02 \n')

        # Choosing the spatial step which is 2 cm
        f.write('#dx_dy_dz: 0.02 0.02 0.02 \n')

        # Simulation will run for 15000 iterations
        f.write('#time_window: 15000 \n')
        f.write('#time_step_stability_factor: 0.99 \n')

        # In every realisation of the function the model will choose a random shape for the pulse
        wave = {0: 'gaussiandot', 1: 'ricker', 2: 'gaussiandotnorm', 3: 'gaussiandotdot'}
        # Randomly choose the index of the pulse
        ind = np.random.randint(4)

        # Randomly choose the central frequency of the pulse from 60 to 100 MHz
        cf = 60+40*np.random.rand()

        # Define the waveform with the randomly selected shape and central frequency
        f.write('#waveform: {} 1 {}e6 my_pulse \n'.format( wave[ind], cf ));

        # Place your transmitters at y = 22.8 meters.
        # If you use one transmiter it will be one-shot configuration
        f.write('#hertzian_dipole: z 2 22.8 0 my_pulse \n');
        # f.write('#hertzian_dipole: z 48 22.8 0 my_pulse \n');
        # f.write('#hertzian_dipole: z 25 22.8 0 my_pulse \n');

        # Place your receivers from 101*0.02 with step 10*0.02 all the way to 2400*0.02
        for i in range(101,2400,10):
            f.write('#rx: {} 22.8 0 \n'.format(i*0.02) );

        # We use 60 cells for the PML to ensure no artifacts from the boundaries
        f.write('#pml_cells: 60 60 0 60 60 0 \n');
        # You can play with the PML parameters and reduce the number of PML cells
        # f.write('#pml_cfs: constant reverse 0.015 0 constant forward 1 5 quadratic forward 0 None \n');



        ###### First layer #########################
        # Choose the fractal dimension of the first layer
        d = 0.5 + 3*np.random.rand()

        # Choose how assymetrical the fractal distribution will be
        dx = 0.5 + 1*np.random.rand()

        # Choose the lower permittivity of the formation randomly from 1.5 to 9.5 (similar to planetary setups)
        r1 = 1.5 + 8*np.random.rand()

        # Choose the upper permittivity of the formation randomly from r1 to 10
        r2 = r1 + (10-r1)*np.random.rand()

        # z1 and z2 and the lower and upper y-axis of the formation. The first formation extents from y = 0-22
        # Subsequent layers will be written on tom of the first layer.
        z1 = 0
        z2 = 22

        # Choose the discretisation bins of the first formation
        bin = 3 + 37*np.random.rand()

        # pro_1 is the ID of the soil for the first formation
        pro = "pro_{}".format(1)

        # geo_1 is the ID of the fractal box for the first layer
        geo = "geo_{}".format(1)

        # Define the soil properties for the first layer
        f.write('#soil_peplinski: 0.5 0.5 2 2.66 {} {} {} \n'.format(r1, r2, pro))
        f.write('#fractal_box: 0 {} 0 50.02 {} 0.02 {} {} 1 1 {} {} {} \n'.format(z1, z2, d, dx, int(np.round(bin)), pro, geo))

        # Select the fractal dimension of the surface of the first formation
        sd = 0.9 + 2*np.random.rand()

        # Select the minimum and maximum amplitude of the first layer.
        # The first layer is chosen to be relatively flat i.e. have a topography variation of 20 cm
        t1 = 0.2
        f.write('#add_surface_roughness: 0 {} 0 50.02 {} 0.02 {} 1 1 {} {} {} \n'.format(z2, z2, sd, z2 - t1, z2 + t1, geo))
        #########################


        for i in range(2,200000):
            # This for-lop will generate layers one after the other until it reaches the limits of the model

            # Choose the fractal dimension of the layer
            d = 0.5 + 3*np.random.rand()

            # Choose the assymetry factor for the layer
            dx = 0.5 + 1*np.random.rand()

            # Choose the minimum permittivity
            r1 = 1.5 + 8*np.random.rand()

            # Choose the maximum permittivity
            r2 = r1 + (10-r1)*np.random.rand()

            # Choose the upper depth of the layer. The lower depth is always zero i.e. the next layer is written on the previous one
            # The upper depth is updated based on the previous one i.e. iteration by iteration the depth for every layer decreases.
            z2 = z2 - 10*np.random.rand()

            # Make the depth a multiple of the spatial step i.e. 0.02.
            z2 = 0.02 * np.round(z2/0.02)
            z1 = 0

            # Choose the number of discritised bins
            bin = 3 + 37*np.random.rand()

            # Choose the maximum topographic variation of the topography of hte layer. Randomly from 0.2 - 1.7 i.e. maximum 3.4 m
            t1 = 0.2 + 1.5*np.random.rand()

            # Make t1 a multiple of 0.02
            t1 = np.round(t1/0.02)*0.02

            # If the upper depth of the layer is below 1 meter then stop making more layers. Notice that top surface is at 22 meters.
            if z2 - t1 < 1:
                break

            # pro_{i} is the ID of the ith soil parameters
            pro = "pro_{}".format(i)

            # geo_{i} is the ID of the ith fractal box
            geo = "geo_{}".format(i)

            # Define the ith layer
            f.write('#soil_peplinski: 0.5 0.5 2 2.66 {} {} {} \n'.format(r1, r2, pro))
            f.write('#fractal_box: 0 {} 0 50.02 {} 0.02 {} {} 1 1 {} {} {} \n'.format(z1, z2, d, dx, int(np.round(bin)), pro, geo))

            # Choose the fractal dimension of the topography of the ith layer
            sd = 0.9 + 2*np.random.rand()

            # Build the topographu for the ith layer
            f.write('#add_surface_roughness: 0 {} 0 50.02 {} 0.02 {} 1 1 {} {} {} \n'.format(z2, z2, sd, z2 - t1, z2 + t1, geo))

        ##############

        # In this saves the model parameters from x,y,z = (0,0,0) to x,y,z = (50, 22, 0.02) i.e. it saves the permittivity of the model exluding the free space
        # We need this file in order to get the ground truth for training ML
        f.write("#geometry_objects_write: 0 0 0 50 22 0.02 file_geo \n")


def get_output_data(filename, rxnumber, rxcomponent):
    file_path = data_dir / filename
    with h5py.File(file_path, 'r') as f:
        nrx = f.attrs['nrx']
        dt = f.attrs['dt']
        if nrx == 0:
            raise ValueError(f'No receivers found in {filename}')
        path = f'/rxs/rx{rxnumber}/'
        if rxcomponent not in f[path]:
            raise ValueError(f'{rxcomponent} not found in {filename}')
        return np.array(f[path + rxcomponent]), dt

def make_bscan(fil, plot=False):
    # The function opens .out file, process the BScan (gain) and saves it in a numpy array
    # plot: When True it plots the BScan after saturated its values for visual inspection
    # fil: The name of the .out file


    # Number of receivers
    # This is for the case study examined in this script. If you decide to change the numnber of receivers in the write_file, then you need to ...
    # change the number of receivers here as well.
    n_receivers = np.shape(np.arange(101,2400,10))[0]

    Bscan=[]
    for i in range(0,n_receivers):
        # Go through all the receivers from the fil
        [fi, t]=get_output_data(fil, i+1, "Ez")

        # Apply gain. You can change this, or you can remove it completely
        gain = np.arange(0,np.shape(fi)[0],40)**3

        Bscan.append(fi[0:np.shape(fi)[0]:40]*gain)
        # plt.plot(fi[0:np.shape(fi)[0]:40])
        # plt.show()

    Bs = np.array(Bscan)
    image = Bs.T

    # Resize your BScan to 230x230 dimensions.
    B = resize(image, (230, 230))
    B2 = copy.copy(B)
    if plot:
        B[B>np.max(B)*0.05] = np.max(B)*0.05
        B[B<np.min(B)*0.05] = np.min(B)*0.05
        plt.imshow(B,aspect='auto',cmap= 'bone')
        plt.show()

    # B2 is a numpy array with the resized (230x230) and processed BScan
    return B2

def make_ground_truth(file_name, plot=False):
    file_path = data_dir / file_name
    # This function opens the file_name to extract the ground truth for every model


    # Id is a 2D array with the IDs of the materials. The IDs of the materials are saved in the file
    # file_geo_materials.txt
    f = h5py.File(file_path, "r")
    n=np.array(f['data'])
    Idf = np.array(n[:,:,0])
    Id = np.zeros(np.shape(Idf))


    # Open the materials file, and isolate the permittivity associated with every ID
    f = open(data_dir / "file_geo_materials.txt")
    mat = []
    for c, x in enumerate(f):
        paok = x.split(" ")
        mat.append(float(paok[1]))
        # Replace the IDs with their permittivitie associated with these IDs
        Id[Idf == c] = float(paok[1])


    # Reduce the discretisation undersample by 8 and 4 the x and y axis respectively
    km = np.shape(Id)
    Id2 = Id[0:km[0]:8, 0:km[1]:4]

    Id = np.flipud(Id.T)
    Id2 = np.flipud(Id2.T)







    # min_max_scaler = p.MinMaxScaler()
    # normalizedData = min_max_scaler.fit_transform(Id2)
    # Id2 = mm + normalizedData*(MM-mm)


    if plot:
        mm = np.min(np.array(mat[2:]))
        MM = np.max(np.array(mat[2:]))
        plt.imshow(Id,vmin=mm, vmax=MM)
        plt.colorbar()
        plt.show()

        plt.imshow(Id2,vmin=mm, vmax=MM)
        plt.colorbar()
        plt.show()


    # B2 is a numpy array with the resized ground truth
    return Id2

def lookupNearest(x0, y0, x, y, data):
    # Function for reshaping the ground truth based on nearest point
    xi = np.abs(x-x0).argmin()
    yi = np.abs(y-y0).argmin()
    return data[xi,yi]

# The dimensions of the ground truth 275x313
x = np.linspace(0, 1, 275)
y = np.linspace(0, 1, 313)

# Resizing it to be square i.e. 224x224
x_new = np.linspace(0, 1, 224)
y_new = np.linspace(0, 1, 224)

for i in range(15000):
    input_file = data_dir / "test_test.in"
    output_file = data_dir / "test_test.out"
    label_file = labels_dir / f"Model_{i}.npy"
    bscan_file = bscan_dir / f"Bscan_{i}.npy"
    
    if label_file.exists() and bscan_file.exists():
        continue
    
    write_file("test_test.in")
    
    subprocess.run([sys.executable, "-m", "gprMax", str(input_file), "-gpu"], check=True)
    
    B2 = make_bscan("test_test.out", plot=False)
    np.save(bscan_file, B2)
    
    temporary_gd = make_ground_truth("file_geo.h5", plot=False)
    Id2 = np.zeros((224,224))
    # Interpolate temporary_gd to resize the ground truth to a 224x224
    for ax, q in enumerate(x_new):
        for ay, w in enumerate(y_new):
            Id2[ax, ay] = lookupNearest(q, w, x, y, temporary_gd)
    np.save(label_file, Id2)