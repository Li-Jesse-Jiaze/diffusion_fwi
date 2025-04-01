import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import h5py
from scipy.ndimage import zoom, sobel
from skimage.transform import resize
import matplotlib.pyplot as plt


def get_output_data(filename, rxnumber, rxcomponent):
    """Gets B-scan output data from a model."""
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    dt = f.attrs['dt']

    if nrx == 0:
        raise ValueError(f'No receivers found in {filename}')

    path = f'/rxs/rx{rxnumber}/'
    available_outputs = list(f[path].keys())

    if rxcomponent not in available_outputs:
        raise ValueError(f'{rxcomponent} output requested, but available outputs are {", ".join(available_outputs)}')
    
    output_data = np.array(f[path + '/' + rxcomponent])
    f.close()
    return output_data, dt


def make_bscan(file_path, plot=False):
    n_receivers = len(np.arange(101, 2400, 10))
    Bscan = []
    
    for i in range(n_receivers):
        fi, _ = get_output_data(file_path, i + 1, "Ez")
        gain = np.arange(0, fi.shape[0], 40) ** 3
        Bscan.append(fi[0:fi.shape[0]:40] * gain)

    Bscan_array = np.array(Bscan).T
    B = resize(Bscan_array, (230, 230))
    
    if plot:
        B[B > np.max(B) * 0.05] = np.max(B) * 0.05
        B[B < np.min(B) * 0.05] = np.min(B) * 0.05
        plt.imshow(B, aspect='auto', cmap='bone')
        plt.show()
    
    return B


def make_ground_truth(file_path):
    f = h5py.File(file_path, "r")
    data = np.array(f['data'])
    Id_raw = np.array(data[:, :, 0])
    Id = np.zeros_like(Id_raw)
    
    script_dir = Path(__file__).resolve().parent
    materials_file = script_dir / "tmp/file_from_geo_materials.txt"
    
    with materials_file.open() as f:
        materials = [float(line.split()[1]) for line in f]
        for idx, value in enumerate(materials):
            Id[Id_raw == idx] = value
    
    Id_resized = Id[::8, ::4]
    return np.flipud(Id_resized.T)


def FDTD(eps_array, num_materials=20):
    script_dir = Path(__file__).resolve().parent
    tmp_dir = script_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    usr_test_in_path = (script_dir / "tmp/usr_test.in").resolve()
    usr_test_out_path = tmp_dir / "usr_test.out"
    
    materials, data = np.unique(eps_array, return_inverse=True)
    
    if len(materials) > num_materials:
        indices = np.linspace(0, len(materials) - 1, num_materials, dtype=int)
        rep = materials[indices]
        mapping = np.array([np.argmin(np.abs(val - rep)) for val in materials])
        data = mapping[data]
        materials = rep
    
    materials_file = tmp_dir / "materials.txt"
    with materials_file.open("w") as f:
        for i, eps in enumerate(materials):
            loss = eps * 1e-4
            name = f"|m_{i:02d}|"
            line = f"#material: {eps} {loss:.8f} 1 0 {name}"
            f.write(line + "\n")
    
    zoomed_data = zoom(np.flipud(data).T, (2500/224, 1100/224), order=0, mode='nearest')
    zoomed_data = zoomed_data[:, :, np.newaxis]
    
    hdf5_file = tmp_dir / "data.hdf5"
    with h5py.File(hdf5_file, "w") as f:
        f.create_dataset("data", data=zoomed_data.astype(np.int16))
        f.attrs['dx_dy_dz'] = (0.02, 0.02, 0.02)
    
    os.environ['PATH'] += os.pathsep + '/usr/local/cuda/bin'
    command = [sys.executable, "-m", "gprMax", str(usr_test_in_path), "-gpu"]
    subprocess.run(command, check=True, text=True, capture_output=True)
    
    B2 = make_bscan(str(usr_test_out_path), plot=False)
    return B2


def compute_gradient(image):
    grad_x = sobel(image, axis=0)
    grad_y = sobel(image, axis=1)
    return np.hypot(grad_x, grad_y)


def normalized_cross_correlation(img1, img2):
    img1 = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
    img2 = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
    return np.mean(img1 * img2)


def gradient_ncc(bscan1, bscan2):
    grad1 = compute_gradient(bscan1)
    grad2 = compute_gradient(bscan2)
    return normalized_cross_correlation(grad1, grad2).item()
