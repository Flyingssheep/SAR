# coding:utf-8
import os
import subprocess
import torch
import torch.nn.functional as F
import random
from PIL import Image
from torchvision import utils
from scipy.integrate import quad
from scipy.fftpack import dct, idct
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.cluster import KMeans
import math


def save_and_resize_image(image_tensor, file_path):
    utils.save_image(image_tensor, file_path)
    img = Image.open(file_path)
    img = img.resize((224, 224), Image.NEAREST)
    img.save(file_path, format='BMP')  # 保存为位图格式


def concatenate_images(image_paths):
    images = [[Image.open(img) for img in imgrol] for imgrol in image_paths]
    new_img = Image.new('RGB', (224*len(images[0]), 224*len(images)))
    for y, imgrol in enumerate(images):
        for x, img in enumerate(imgrol):
            new_img.paste(img, (224*x, 224*y))
    return new_img

""" """
def save_images(images, folder_name, plustring):
    img_floder_name = os.path.join(folder_name, "adversarial_samples")
    if not os.path.exists(img_floder_name):
        os.makedirs(img_floder_name)

    paths = []
    for i, imgrol in enumerate(images):
        paths.append([])
        for j, img in enumerate(imgrol):
            paths[i].append(os.path.join(img_floder_name, f"{i}-{j}OF_{plustring}.bmp"))
            save_and_resize_image(img, paths[i][j])

    # Concatenate and save combined image in BMP format
    combined_image = concatenate_images(paths)
    combined_image.save(os.path.join(img_floder_name, f"COMB_{plustring}.bmp"), format='BMP')
    """ """
    for i, pthrol in enumerate(paths):
        for j, pth in enumerate(pthrol):
            os.remove(pth)


def open_image(image_path):
    if os.name == 'nt':  # Windows
        os.startfile(image_path)
    elif os.name == 'posix':  # macOS and Linux
        subprocess.run(['open', image_path], check=True)


def progress_bar(imgi, query, iter, total, ADB, l2, label_origin=0, label_after=0, bar_length=10):
    # percent = 100 * (progress / float(total))
    #bar_fill = int(bar_length * query / total)
    #bar = '█' * bar_fill + '-' * (bar_length - bar_fill)
    # sys.stdout.write(f'\r[{bar}] Q{percent:.1f}% R{Rnow:.3f}')
    sys.stdout.write(f'\rImg{imgi} Query{query :.0f} \tIter{iter :.0f} \tADB={ADB:.6f}({l2:.6f}) \tLAB={label_origin:.0f}->{label_after:.0f}')
    sys.stdout.flush()

def to_01(InTensor):
    return (InTensor - torch.min(InTensor)) / (torch.max(InTensor) - torch.min(InTensor))

def top_percent_to_1(tensor, a=0.5):
    threshold = torch.quantile(tensor, a)
    binary_tensor = (tensor >= threshold)
    return binary_tensor

def bot_percent_to_1(tensor, a=0.5):
    threshold = torch.quantile(tensor, a)
    binary_tensor = (tensor <= threshold)
    return binary_tensor

def normal_vector_approximation(x0, boundary_x, dim_reduc_factor, q_max, sigma, model, tar_img, x0_label, tar_label):
    def generate_balanced_nvs(K, c, w, h):
        indices = torch.rand(K, c, w, h).argsort(dim=0).cuda()
        result = torch.ones(K, c, w, h, dtype=torch.float32).cuda()
        result[indices < K // 2] = -1
        return result
    def generate_balanced_nvs_final(K, c, w, h, codebook_size=2048, device="cuda"):
        N = c * w * h
        dev = device
        neg = K // 2
        pos = K - neg
        base = torch.cat([
            -torch.ones(neg, device=dev, dtype=torch.int8),
            torch.ones(pos, device=dev, dtype=torch.int8)
        ], dim=0)  # (K,)
        # --- Codebook cache (per K, codebook_size, device) to avoid regenerating every call ---
        cache_key = (K, codebook_size, str(dev))
        if not hasattr(generate_balanced_nvs_final, "_codebook_cache"):
            generate_balanced_nvs_final._codebook_cache = {}
        cache = generate_balanced_nvs_final._codebook_cache
        if cache_key not in cache:
            # Build codebook: (K, codebook_size), each column is a random permutation of 'base'
            # Every column is balanced by construction.
            codebook = torch.empty((K, codebook_size), device=dev, dtype=torch.int8)
            for j in range(codebook_size):
                codebook[:, j] = base[torch.randperm(K, device=dev)]
            cache[cache_key] = codebook
        else:
            codebook = cache[cache_key]
        # --- Assign each pixel a random balanced pattern from the codebook ---
        idx = torch.randint(0, codebook_size, (N,), device=dev, dtype=torch.int64)  # (N,)
        flat = codebook[:, idx]  # (K, N) int8, each column is balanced
        # reshape to (K,c,w,h) and cast to float32 for downstream ops
        result = flat.view(K, c, w, h).to(torch.float32)
        return result
    random_noises = None
    boundary_v = boundary_x - x0
    boundary_v_l2 = torch.norm(boundary_v).item()
    if dim_reduc_factor < 1.0:
        raise Exception(
            "The dimension reduction factor should be greater than 1 for reduced dimension, and should be 1 for Full dimensional image space.")
    if dim_reduc_factor > 1.0:
        fill_size = int(x0.shape[-1] / dim_reduc_factor)
        random_noises = torch.zeros(q_max, int(x0.shape[-3]), int(x0.shape[-2]), int(x0.shape[-1]),
                                    dtype=torch.float32).cuda()
        for i in range(q_max):
            random_noises[i][:, 0:fill_size, 0:fill_size] = torch.randn(x0.shape[0], x0.shape[1], fill_size,
                                                                        fill_size)
            random_noises[i] = torch.from_numpy(
                idct(idct(random_noises[i].cpu().numpy(), axis=2, norm='ortho'), axis=1, norm='ortho'))
    else:
        random_noises = torch.randn(q_max, x0.shape[1], x0.shape[2], x0.shape[3], dtype=torch.float32).cuda()
        #random_noises = torch.randint(0, 2, [q_max, x0.shape[1], x0.shape[2], x0.shape[3]], dtype=torch.float32).cuda() * 2 - 1
        #random_noises = torch.rand(q_max, x0.shape[1], x0.shape[2], x0.shape[3], dtype=torch.float32).cuda() * 2 - 1
        #random_noises = generate_balanced_nvs(q_max, x0.shape[1], x0.shape[2], x0.shape[3])
    labels_out = []
    pixnum = torch.numel(x0)
    for i in range(q_max):
        k_to_one = (math.sqrt(pixnum) / (dim_reduc_factor * torch.norm(random_noises[i]).item()))
        random_noises[i] *= (k_to_one * sigma)
        noise_l2 = torch.norm(boundary_v + random_noises[i])
        #if noise_l2 < boundary_v_l2:
        #    random_noises[i] *= -1
        labels_out.append(model.predict_label(x0 + boundary_v + random_noises[i]))

    z = []  # sign of grad_tmp
    for i, predict_label in enumerate(labels_out):
        if tar_img == None:
            if predict_label == x0_label:
                z.append(-1)
                random_noises[i] *= -1
            else:
                z.append(1)
        if tar_img != None:
            if predict_label != tar_label:
                z.append(-1)
                random_noises[i] *= -1
            else:
                z.append(1)
    normal_v = sum(random_noises)
    mean_z = sum(z) / q_max
    return normal_v, mean_z

def tangent_vector_approximation(x0, boundary_x, dim_reduc_factor, q_max, sigma, model, tar_img, x0_label, tar_label):
    random_noises = None
    boundary_v = boundary_x - x0
    boundary_v_l2 = torch.norm(boundary_v).item()
    if dim_reduc_factor < 1.0:
        raise Exception(
            "The dimension reduction factor should be greater than 1 for reduced dimension, and should be 1 for Full dimensional image space.")
    if dim_reduc_factor > 1.0:
        fill_size = int(x0.shape[-1] / dim_reduc_factor)
        random_noises = torch.zeros(q_max, int(x0.shape[-3]), int(x0.shape[-2]), int(x0.shape[-1]),
                                    dtype=torch.float32).cuda()
        for i in range(q_max):
            random_noises[i][:, 0:fill_size, 0:fill_size] = torch.randn(x0.shape[0], x0.shape[1], fill_size,
                                                                        fill_size)
            random_noises[i] = torch.from_numpy(
                idct(idct(random_noises[i].cpu().numpy(), axis=2, norm='ortho'), axis=1, norm='ortho'))
    else:
        #random_noises = torch.randn(q_max, x0.shape[1], x0.shape[2], x0.shape[3], dtype=torch.float32).cuda()
        #random_noises = torch.randint(0, 2, [q_max, x0.shape[1], x0.shape[2], x0.shape[3]], dtype=torch.float32).cuda() * 2 - 1
        random_noises = torch.rand(q_max, x0.shape[1], x0.shape[2], x0.shape[3], dtype=torch.float32).cuda() * 2 - 1
        #random_noises = generate_balanced_nvs(q_max, x0.shape[1], x0.shape[2], x0.shape[3])
    labels_out = []
    pixnum = torch.numel(x0)
    for i in range(q_max):
        k_to_one = (math.sqrt(pixnum) / (dim_reduc_factor * torch.norm(random_noises[i]).item()))
        random_noises[i] *= (k_to_one * sigma)
        MCS_v = boundary_v + random_noises[i]
        kk = (boundary_v_l2/torch.norm(MCS_v).item())
        tangent_v = MCS_v * kk
        random_noises[i] = tangent_v - boundary_v
        labels_out.append(model.predict_label(x0 + tangent_v))

    z = []  # sign of grad_tmp
    for i, predict_label in enumerate(labels_out):
        if tar_img == None:
            if predict_label == x0_label:
                z.append(-1)
                random_noises[i] *= -1
            else:
                z.append(1)
        if tar_img != None:
            if predict_label != tar_label:
                z.append(-1)
                random_noises[i] *= -1
            else:
                z.append(1)
    normal_v = sum(random_noises)
    mean_z = sum(z) / q_max
    return normal_v, mean_z, random_noises

def tangent_vector_approximation222(x0, boundary_x, dim_reduc_factor, q_max, sigma, model, tar_img, x0_label, tar_label):
    def generate_fixed_norm_noises(v, c, num):
        orig_shape = v.shape
        v = v.flatten()
        v_norm2 = torch.dot(v, v)
        noises = []
        for _ in range(num):
            # ---- Step 1: random noise orthogonal to v ----
            raw = torch.rand_like(v)
            proj = torch.dot(raw, v) / v_norm2 * v
            n_perp = raw - proj
            # Normalize orthogonal direction
            n_perp = n_perp / (n_perp.norm() + 1e-12)
            # ---- Step 2: parallel component along v ----
            n_parallel = -(c ** 2) / (2 * v_norm2) * v
            # ---- Step 3: compute remaining magnitude for orthogonal part ----
            ortho_mag = torch.sqrt(c ** 2 - n_parallel.norm() ** 2)
            # ---- Combine components ----
            n = ortho_mag * n_perp + n_parallel
            # restore original shape
            noises.append(n.view(orig_shape))
        return noises
    random_noises = None
    boundary_v = boundary_x - x0
    boundary_v_l2 = torch.norm(boundary_v).item()
    if dim_reduc_factor < 1.0:
        raise Exception(
            "The dimension reduction factor should be greater than 1 for reduced dimension, and should be 1 for Full dimensional image space.")
    if dim_reduc_factor > 1.0:
        fill_size = int(x0.shape[-1] / dim_reduc_factor)
        random_noises = torch.zeros(q_max, int(x0.shape[-3]), int(x0.shape[-2]), int(x0.shape[-1]),
                                    dtype=torch.float32).cuda()
        for i in range(q_max):
            random_noises[i][:, 0:fill_size, 0:fill_size] = torch.randn(x0.shape[0], x0.shape[1], fill_size,
                                                                        fill_size)
            random_noises[i] = torch.from_numpy(
                idct(idct(random_noises[i].cpu().numpy(), axis=2, norm='ortho'), axis=1, norm='ortho'))
    else:
        #random_noises = torch.randn(q_max, x0.shape[1], x0.shape[2], x0.shape[3], dtype=torch.float32).cuda()
        #random_noises = torch.randint(0, 2, [q_max, x0.shape[1], x0.shape[2], x0.shape[3]], dtype=torch.float32).cuda() * 2 - 1
        #random_noises = torch.rand(q_max, x0.shape[1], x0.shape[2], x0.shape[3], dtype=torch.float32).cuda() * 2 - 1
        #random_noises = generate_balanced_nvs(q_max, x0.shape[1], x0.shape[2], x0.shape[3])
        random_noises = generate_fixed_norm_noises(boundary_v, sigma, q_max)
    labels_out = []
    pixnum = torch.numel(x0)
    for i in range(q_max):
        labels_out.append(model.predict_label(x0 + boundary_v + random_noises[i]))

    z = []  # sign of grad_tmp
    for i, predict_label in enumerate(labels_out):
        if tar_img == None:
            if predict_label == x0_label:
                z.append(-1)
                random_noises[i] *= -1
            else:
                z.append(1)
        if tar_img != None:
            if predict_label != tar_label:
                z.append(-1)
                random_noises[i] *= -1
            else:
                z.append(1)
    normal_v = sum(random_noises)
    mean_z = sum(z) / q_max
    return normal_v, mean_z, random_noises

def RlineQ(Rline, radius_line, budget):
    start = 0
    for t in range(len(Rline) - 1):
        for q in range(start, min(Rline[t + 1][0], budget)):
            radius_line[q] = radius_line[q] + Rline[t][1]
            start = Rline[t + 1][0]
    return

def draw_distribution(x, name="pic"):
    # 绘制直方图
    plt.figure(figsize=(10, 8))
    counts, bins, patches = plt.hist(x, bins=50, color='skyblue', edgecolor='black', weights=np.ones(len(x)) / len(x))
    # 转换为百分比显示
    #for i in range(len(patches)):
    #    plt.text(bins[i] + (bins[1] - bins[0]) / 2, counts[i] + 0.001, f"{counts[i] * 100:.1f}%", ha='center')
    plt.xlim(0, 1)
    plt.xticks(np.arange(0.1, 1.1, 0.1), fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=30)
    # 设置 y 轴为百分比
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    # 添加标题和标签
    #plt.title("Histogram with Percentage", fontsize=14)
    #plt.xlabel("Value", fontsize=12)
    #plt.ylabel("Frequency (Percentage)", fontsize=12)
    # 显示图形
    plt.tight_layout()
    plt.show()


def gaussian_kernel(window_size: int, sigma: float, channels: int):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_1d = g[:, None]
    kernel_2d = kernel_1d @ kernel_1d.t()  # (W, W)
    kernel_2d = kernel_2d.expand(channels, 1, window_size, window_size)
    return kernel_2d
def ssim(img1, img2, data_range=1.0, sigma=1.5, size_average=True):
    """
    img1, img2: torch.Tensor, (N, C, H, W) or (C, H, W)
    value within [0,1]
    """

    # to (N, C, H, W)
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    N, C, H, W = img1.shape

    # -------- auto window_size --------
    win = min(11, H, W)

    if win % 2 == 0:
        win -= 1
    win = max(win, 3)

    window_size = win
    # --------------------------------------

    kernel = gaussian_kernel(window_size, sigma, C).to(img1.device)
    padding = window_size // 2

    # mean
    mu1 = F.conv2d(img1, kernel, padding=padding, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=padding, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # variance
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=padding, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=padding, groups=C) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, kernel, padding=padding, groups=C) - mu1_mu2

    # SSIM
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.flatten(2,3).mean(-1).mean(-1).item()


##################################################################################
# 给定参数 cifar10/Imagenet

a_hat = 0.03133292769944518
b_hat = 3.0659694403842903
c_hat = 0.16755646970211466
d_hat = 0.13403850261898806


# 定义函数 y
def func_y(r, a, b, c, d):
    return a / ((r + d) ** b) + c


# 为了使用 fsolve，需要定义一个差函数
def find_midK_of_k1k2(k1, k2):
    Sk1k2, error = quad(func_y, k1, k2, args=(a_hat, b_hat, c_hat, d_hat))
    low, high = k1, k2
    mid = (low + high) / 2
    Sk1mid = None
    while high - low > 1.0 / 10000:
        mid = (low + high) / 2
        Sk1mid, error = quad(func_y, k1, mid, args=(a_hat, b_hat, c_hat, d_hat))
        if Sk1mid < Sk1k2 / 2:
            low = mid
        else:
            high = mid
    return mid


def next_binary_rref(r1, r2, max_r, mod):
    if mod == 0:
        return (r1 + r2) / 2.0
    if mod == 1:
        k1, k2 = r1 / max_r, r2 / max_r
        kmid = find_midK_of_k1k2(1-k2, 1-k1)
        median = (1-kmid) * max_r
        return median

def cosine_similarity(tensor1, tensor2):
    dot_product = torch.dot(tensor1.flatten(), tensor2.flatten())
    norm1, norm2 = torch.norm(tensor1), torch.norm(tensor2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

#--------------------------------------------------------------------------------

def spectral_bipartition(cm):
    # 1. 归一化混淆矩阵
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized_cm = cm / row_sums
    # 2. 计算类别间的相似度
    similarity = normalized_cm @ normalized_cm.T
    # 3. 构造图的拉普拉斯矩阵
    # 邻接矩阵 A 使用负的相似度作为权重（使得相似的类别有较强连接）
    A = similarity
    D = np.diag(A.sum(axis=1))  # 度矩阵
    L = D - A  # 非标准化拉普拉斯矩阵
    # 4. 谱分解
    # 求第二小的特征值对应的特征向量（即 Fiedler 向量）
    eigenvalues, eigenvectors = eigh(L, subset_by_index=[1, 1])
    fiedler_vector = eigenvectors[:, 0]
    # 5. 使用 Fiedler 向量进行二分
    # 聚类方式1：简单地根据 Fiedler 向量的正负号分成两堆
    # labels = (fiedler_vector > 0).astype(int)
    # 聚类方式2：用KMeans在 Fiedler 向量上一维数据上划分成两类
    kmeans = KMeans(n_clusters=2, n_init=10)
    labels = kmeans.fit_predict(fiedler_vector.reshape(-1, 1))
    return labels


def hierarchical_partition(cm):
    def recursive_partition(subset, cm):
        # 如果堆内只有一个类，停止分堆
        if len(subset) <= 1:
            return [subset]
        # 提取当前子集的子矩阵
        sub_cm = cm[np.ix_(subset, subset)]
        # 使用谱分割将 subset 划分为两堆
        labels = spectral_bipartition(sub_cm)
        # 根据分堆结果分离出两个新子集
        subset_0 = [subset[i] for i in range(len(subset)) if labels[i] == 0]
        subset_1 = [subset[i] for i in range(len(subset)) if labels[i] == 1]
        # 初始化结果
        result = []
        # 只有当子集大小超过1时才进入递归分堆
        if len(subset_0) > 1:
            result.append(recursive_partition(subset_0, cm))
        else:
            result.append(subset_0)
        if len(subset_1) > 1:
            result.append(recursive_partition(subset_1, cm))
        else:
            result.append(subset_1)
        return result

    # 初始化整个类别集
    n_classes = cm.shape[0]
    initial_subset = list(range(n_classes))
    # 递归分堆，结果存储为多层嵌套列表
    final_structure = recursive_partition(initial_subset, cm)
    return final_structure


def print_tree(tree, prefix=''):
    if isinstance(tree, list):
        for i, branch in enumerate(tree):
            is_last = (i == len(tree) - 1)
            if isinstance(branch, list):
                print(f"{prefix}{'└── ' if is_last else '├── '}Node")
                print_tree(branch, prefix + ('    ' if is_last else '│   '))
            else:
                print(f"{prefix}{'└── ' if is_last else '├── '}Class {branch}")
    else:
        print(f"{prefix}└── Class {tree}")

