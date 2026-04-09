# import numpy as np
# from numpy.linalg import norm

# b = np.load('/home/dy/DDESeg-main/feabank1.npy')
# b_mean = b.mean(axis=1)

# def cosine(x, y, eps=1e-12):
#     return np.dot(x, y) / (norm(x) * norm(y) + eps)

# sim_b = np.zeros((71, 71), dtype=np.float32)
# for i in range(71):
#     for j in range(71):
#         sim_b[i, j] = cosine(b_mean[i], b_mean[j])

# # 去掉对角线，看非对角项
# mask = ~np.eye(71, dtype=bool)
# off_diag = sim_b[mask]

# print("b self-sim off-diag mean:", off_diag.mean())
# print("b self-sim off-diag max :", off_diag.max())
# print("b self-sim off-diag min :", off_diag.min())

# a = np.load('/mnt/sdc/dy/data/Re_AVS/1s_k5_feabank.npy')
# a_mean = a.mean(axis=1)

# sim_a = np.zeros((71, 71), dtype=np.float32)
# for i in range(71):
#     for j in range(71):
#         sim_a[i, j] = cosine(a_mean[i], a_mean[j])

# mask = ~np.eye(71, dtype=bool)
# off_diag = sim_a[mask]

# print("a self-sim off-diag mean:", off_diag.mean())
# print("a self-sim off-diag max :", off_diag.max())
# print("a self-sim off-diag min :", off_diag.min())

# c = np.load('/home/dy/DDESeg-main/feabank2.npy')
# c_mean = c.mean(axis=1)

# sim_c = np.zeros((71, 71), dtype=np.float32)
# for i in range(71):
#     for j in range(71):
#         sim_c[i, j] = cosine(c_mean[i], c_mean[j])

# mask = ~np.eye(71, dtype=bool)
# off_diag = sim_c[mask]

# print("c self-sim off-diag mean:", off_diag.mean())
# print("c self-sim off-diag max :", off_diag.max())
# print("c self-sim off-diag min :", off_diag.min())

# '''
#     (py310) dy@user-V2:/mnt/sdc/dy$ /home/dy/miniconda3/envs/py310/bin/python /mnt/sdc/dy/check_mask.py
#     b self-sim off-diag mean: 0.93352515
#     b self-sim off-diag max : 0.99718815
#     b self-sim off-diag min : 0.0
#     a self-sim off-diag mean: 0.8643193
#     a self-sim off-diag max : 0.975559
#     a self-sim off-diag min : 0.7362786
#     c self-sim off-diag mean: 0.9219438
#     c self-sim off-diag max : 0.99478525
#     c self-sim off-diag min : 0.0
# '''

import numpy as np
from numpy.linalg import norm

def cosine(x, y, eps=1e-12):
    return np.dot(x, y) / (norm(x) * norm(y) + eps)

def check_bank(path, name):
    bank = np.load(path)
    mean = bank.mean(axis=1)

    norms = np.linalg.norm(mean, axis=1)
    print(f"\n{name}")
    print("row norms min/max:", norms.min(), norms.max())
    print("class 0 norm:", norms[0])

    sim = np.zeros((71, 71), dtype=np.float32)
    for i in range(71):
        for j in range(71):
            sim[i, j] = cosine(mean[i], mean[j])

    mask_all = ~np.eye(71, dtype=bool)
    print("all off-diag min:", sim[mask_all].min())

    sim_fg = sim[1:, 1:]
    mask_fg = ~np.eye(70, dtype=bool)
    print("exclude bg off-diag min:", sim_fg[mask_fg].min())
    print("exclude bg off-diag mean:", sim_fg[mask_fg].mean())

check_bank('/home/dy/DDESeg-main/feabank1.npy', 'b')
check_bank('/home/dy/DDESeg-main/feabank2.npy', 'c')
check_bank('/mnt/sdc/dy/data/Re_AVS/1s_k5_feabank.npy', 'paper')

# (py310) dy@user-V2:/mnt/sdc/dy$ /home/dy/miniconda3/envs/py310/bin/python /mnt/sdc/dy/check_mask.py

# b
# row norms min/max: 0.0 4.9887376
# class 0 norm: 0.0
# all off-diag min: 0.0
# exclude bg off-diag min: 0.82885605
# exclude bg off-diag mean: 0.96058387

# c
# row norms min/max: 0.0 5.047799
# class 0 norm: 0.0
# all off-diag min: 0.0
# exclude bg off-diag min: 0.80810696
# exclude bg off-diag mean: 0.94866675

# paper
# row norms min/max: 3.284671 3.849205
# class 0 norm: 3.6078417
# all off-diag min: 0.7362786
# exclude bg off-diag min: 0.7362786
# exclude bg off-diag mean: 0.86467344