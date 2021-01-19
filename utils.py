import numpy as np
from glob import glob as glob
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2


IM_SIZE = np.array((192, 168))

# Data processing helpers

def filter_files(fnames, cond, size=5):
    fnames_base = [os.path.basename(i)[:-4] for i in fnames]
    acc_files = []
    for i, f in enumerate(fnames_base):
        az = int(f[12:16])
        elev = int(f[17:20])
        if cond(az, elev):
#             print(az, elev)
            acc_files.append(fnames[i])
#     samps = np.random.choice(acc_files, size, replace=False)
    return acc_files

def fread(f):
    return plt.imread(f).flatten().T

def random_sample_cond(train_cond, test_cond):

    all_fnames = glob("*/*/*_P00A*.pgm")

    train_fnames = filter_files(all_fnames, train_cond)
    test_fnames = filter_files(all_fnames, test_cond)

    num_train = len(train_fnames)
    num_test = len(test_fnames)

    A = np.zeros((np.prod(IM_SIZE), num_train))
    y = np.zeros((np.prod(IM_SIZE), num_test))

    train_gt = np.zeros(num_train).astype(int)
    test_gt = np.zeros(num_test).astype(int)

    for i, f in enumerate(train_fnames):
        A[:,i] = fread(f)
        train_gt[i] = int(os.path.basename(f)[5:7])-1

    for i, f in enumerate(test_fnames):
        y[:,i] = fread(f)
        test_gt[i] = int(os.path.basename(f)[5:7])-1

    return A, y, train_gt, test_gt, (train_fnames, test_fnames)

def random_sample():
    all_fnames = glob("*/*/*_P00A*.pgm")
    half = len(all_fnames)//2
    np.random.shuffle(all_fnames)

    train_fnames = all_fnames[:half]
    test_fnames = all_fnames[half:]

    A = np.zeros((np.prod(IM_SIZE), len(train_fnames)))
    y = np.zeros((np.prod(IM_SIZE), len(test_fnames)))

    train_gt = np.zeros(len(train_fnames)).astype(int)
    test_gt = np.zeros(len(test_fnames)).astype(int)

    for i, f in enumerate(train_fnames):
        A[:,i] = fread(f)
        train_gt[i] = int(os.path.basename(f)[5:7])-1

    for i, f in enumerate(test_fnames):
        y[:,i] = fread(f)
        test_gt[i] = int(os.path.basename(f)[5:7])-1

    return A, y, train_gt, test_gt, (train_fnames, test_fnames)

#

def delta_i(x, i, gt):
    return np.where(gt==i, x, 0)

class CAE(nn.Module):
    def __init__(self, N):
        super().__init__()
        convs = []
        deconvs = []

        strides = [1, 2]
        kerns = [3,4]
        convs.append(nn.Conv2d(1, N[0], kernel_size=3, stride=1, padding=1))
        for i in range(len(N)-1):
            s = strides[(i+1)%2]
            convs.append(nn.PReLU())
            convs.append(nn.Conv2d(N[i], N[i+1], kernel_size=3, stride=s, padding=1))

        for i in range(len(N)-1, 0, -1):
            s = strides[(i+1)%2]
            k = kerns[(i+1)%2]
#             s = strides[i%2]
#             k = kerns[i%2]
            deconvs.append(nn.ConvTranspose2d(N[i], N[i-1], kernel_size=k, stride=s, padding=1))
            deconvs.append(nn.PReLU())
        deconvs.append(nn.ConvTranspose2d(N[0],1, kernel_size=4, stride=2, padding=1))

        self.encoder = nn.Sequential(*convs)
        self.decoder = nn.Sequential(*deconvs)

    def forward(self, x):
        low_dim = self.encoder(x)
        recon = self.decoder(low_dim)
        return low_dim, recon

def np_to_torch(x):
    x_ims = x.reshape((*IM_SIZE, -1))
    x_ims_resize = np.zeros((192, 176, x_ims.shape[-1]))
    for i in range(x_ims.shape[-1]):
        x_ims_resize[...,i] = cv2.resize(x_ims[...,i], (176, 192))
    return torch.from_numpy(x_ims_resize.transpose(2, 0, 1)[:, None, ...]).float()
