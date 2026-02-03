# coding:utf-8
import DataTools
import torch
import torch.nn.functional as F
import math
import numpy as np
import copy
import random
import matplotlib.cm as cm
from torchvision.transforms import Compose, RandomResizedCrop, Pad, RandomCrop
from datetime import datetime
import sys
import matplotlib.pyplot as plt

class V2D:
    def __init__(self, size_channel, size_x, size_y, initSign, label):
        self.size_channel = size_channel
        self.size_x = size_x
        self.size_y = size_y
        self.pixnum = size_x * size_y * size_channel
        self.initV = torch.ones(self.size_channel, self.size_y, self.size_x, dtype=torch.float32).cuda()
        self.sens = torch.ones(self.size_channel, self.size_y, self.size_x, dtype=torch.float32).cuda() #
        self.ADBmax = 1.0  #
        self.ADBmin = 0.0  #
        self.adv_v = self.initV.clone()
        self.theta = 0
        self.adversarial_image = self.adv_v
        self.label_attack = label
        self.RealL2 = torch.norm(self.adv_v, p=2)
        self.RealLinf = torch.max(abs(self.adv_v))

    def real_R(self, imageXcuda):
        self.adversarial_image = torch.clamp(imageXcuda + self.adv_v, 0.0, 1.0)
        adv_v_real = self.adversarial_image - imageXcuda
        self.RealL2 = torch.norm(adv_v_real, p=2).item()
        self.RealLinf = torch.max(abs(adv_v_real)).item()

class Attacker:
    def __init__(self, args, model, original_image, imgi, label, ATK_target_xi, ATK_target_yi, InitATKer=None, iter_n=0):
        self.InitATKer = InitATKer
        self.chosen_v = -1
        self.args = args
        self.model = model
        self.x0_cpu = original_image
        self.x0 = original_image.cuda()
        self.Img_i = imgi
        self.label_origin = label
        self.ATK_target_xi = ATK_target_xi
        self.ATK_target_yi = ATK_target_yi
        self.C, self.W, self.H = (
            original_image.shape[1], original_image.shape[2], original_image.shape[3])
        self.Img_pixes = self.C * self.H * self.W
        self.old_best_adv = V2D(self.C, self.H, self.W, self.args.initSign, label)

        self.aim_l2 = args.epsilon
        self.aim_ADB = self.aim_l2 / torch.norm(self.old_best_adv.adv_v, p=2).item()

        self.blocks = []
        self.chosen_block_i = 0
        self.chosen_theta = -1
        self.query = 0
        self.iter_n = iter_n
        self.success = -1
        self.ACCquery = 0
        self.ACCiter_n = 0
        self.File_string = "none"
        self.Img_result = []
        self.heatmaps = []

        self.Noises = InitATKer.Noises
        self.total_Kbridge = []

        self.L2_line = [[0, self.x0_cpu.numel() ** 0.5]]
        self.Linf_line = [[0, 1.0]]
        if InitATKer is not None:
            self.old_best_adv.initV = InitATKer.old_best_adv.adv_v.clone()#(InitATKer.Img_result[2] - InitATKer.Img_result[1])[0].cuda()##############
            self.old_best_adv.sens = torch.ones(self.C, self.W, self.H, dtype=torch.float32).cuda()
            self.old_best_adv.adv_v = InitATKer.old_best_adv.adv_v.clone()
            self.success = copy.deepcopy(InitATKer.success)

    def if_atk_success(self, F_x_add_adv):
        if self.args.targeted == 0:
            if F_x_add_adv != self.label_origin.item():
                return True
            else:
                return False
        if self.args.targeted == 1:
            if F_x_add_adv == self.ATK_target_yi:
                return True
            else:
                return False

    def show_message(self):
        sys.stdout.write(
            f'\rImg{self.Img_i} Query{self.query :.0f}'
            f'\tIter{self.iter_n :.0f}'
            f'\tl2=({self.old_best_adv.RealL2:.4f})'
            f'\tLAB={self.label_origin.item():.0f}->{self.old_best_adv.label_attack.item():.0f}')
        sys.stdout.flush()
        return

    def attack(self):
        VARs = []
        for noises in self.Noises:
            VARs.append(DataTools.to_01(torch.var(noises, dim=0, keepdim=False)))
        #Important = 1-torch.mean(torch.cat(VARs, dim=0),dim=0,keepdim=False)
        if self.InitATKer.old_best_adv.adv_v.dim() == 4:
            Important = torch.mean(abs(self.InitATKer.old_best_adv.adv_v[0]),dim=0)#
            Important = Important.unsqueeze(0)
            #Important = Important.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
        else:
            print("SAR dim error")
        pooling_factor_r = 16
        if Important.shape[-1] < 224:
            pooling_factor_r = 4
        if Important.shape[-1] < 32:
            pooling_factor_r = 7
        pooled = F.avg_pool2d(Important, kernel_size=pooling_factor_r, stride=pooling_factor_r)
        Important = pooled[0].repeat_interleave(pooling_factor_r, dim=0).repeat_interleave(pooling_factor_r, dim=1)

        i = 0
        N = 10000
        L = math.ceil(math.log2(N + 1))
        binary_tree = [(2 * k + 1) / (2 ** (i + 1))
                       for i in range(L)
                       for k in range(2 ** i)]
        self.old_best_adv.adv_v = self.InitATKer.old_best_adv.adv_v.clone()
        while self.query < self.args.budget2:
            persent = binary_tree[i]
            i = i + 1
            adv_temp = self.old_best_adv.adv_v.clone()
            chosen = DataTools.bot_percent_to_1(Important, persent)
            mask = 1.0 - chosen.to(dtype=torch.float32) * 0.1
            adv_temp *= mask
            kk = (torch.norm(self.old_best_adv.adv_v) / torch.norm(adv_temp))
            adv_temp *= kk
            if self.is_adversarial(self.x0 + adv_temp) == 1:
                newadv_x, newadv_vl2, query_bin = self.bin_search_fast(torch.cat([self.x0.clone(), self.x0 + adv_temp], dim=0), 1e-4)
                if newadv_vl2[1] < torch.norm(self.old_best_adv.adv_v):
                    self.old_best_adv.adversarial_image,self.old_best_adv.RealL2,self.old_best_adv.adv_v = \
                        (newadv_x.clone(), newadv_vl2[1].clone().item(), newadv_x[1] - self.x0)

                    self.old_best_adv.real_R(self.x0)
                    self.L2_line.append([self.query+self.InitATKer.query, self.old_best_adv.RealL2])
                    self.Linf_line.append([self.query+self.InitATKer.query, self.old_best_adv.RealLinf])
                    if self.query >= self.args.budget2:
                        break
                    if self.success == -1 and self.old_best_adv.RealL2 <= self.args.epsilon:
                        self.success = 1
                        self.ACCquery, self.ACCiter_n = self.query, self.iter_n
                    if self.args.early == 1 and self.success == 1:
                        break
        self.old_best_adv.real_R(self.x0)
        print(f'\t VSdist={self.old_best_adv.RealL2:.4f} \t persent={persent:.4f} \t que={self.query}')
        self.old_best_adv.label_attack = self.model.predict_label(self.old_best_adv.adversarial_image.cuda())
        """"""
        self.File_string = (str(self.args.dataset) + ",Img" + str(self.Img_i) + ",I-Q[" + str(self.iter_n) + "-" + str(
            self.args.budget2) + "],Label[" + str(self.label_origin.item()) + "-" + str(self.old_best_adv.label_attack.item()) + "],l2{:.4f}".format(
            self.old_best_adv.RealL2) + ",T" + str(datetime.now().strftime("%H-%M-%S"))
                            )
        sens_array = (self.old_best_adv.adv_v / (self.old_best_adv.initV+1e-12))[0].cpu()
        heat_array = (sens_array-sens_array.min()) / (sens_array.max()-sens_array.min())
        #heat_array = heat_array / torch.quantile(heat_array, 0.97)
        cmap = cm.get_cmap('coolwarm')
        advimg = 0.5 * (1 + (self.old_best_adv.ADBmax / self.old_best_adv.RealLinf) * self.old_best_adv.adv_v)
        self.Img_result = [self.x0,
                           DataTools.to_01(self.InitATKer.old_best_adv.adv_v),
                           advimg,
                           torch.tensor(cmap(heat_array.mean(dim=0))[:, :, :3]).permute(2, 0, 1).float() * 0.5 + self.x0_cpu * 0.5,
                           self.old_best_adv.adversarial_image]
        """
        x_flat = sens_array.flatten()
        x_sorted, _ = torch.sort(x_flat)
        x_sorted = x_sorted.detach().cpu().numpy()
        plt.figure()
        plt.plot(x_sorted)
        plt.xlabel("Sorted Index")
        plt.ylabel("Value")
        plt.title("Sorted Tensor Values")
        plt.grid(True)
        plt.show()"""
        return


    def is_adversarial(self, image_in):
        image = copy.deepcopy(image_in)
        if self.args.RandResizePad == 1:
            transform = Compose([
                RandomResizedCrop(size=224, scale=(0.8, 1.2)), 
                Pad(padding=10, fill=0), 
                RandomCrop(size=224) 
            ])
            image = transform(image)
        predict_label = self.model.predict_label(torch.clamp(image,0.0,1.0)).cpu().item()
        self.query += 1
        if self.InitATKer.tar_img == None:
            is_adv = predict_label != self.InitATKer.x0_label
        else:
            is_adv = predict_label == self.InitATKer.tar_label
        if is_adv:
            return 1
        else:
            return -1

    def bin_search_fast(self, adv_x, tol):
        out_adv_x = adv_x.clone()
        num_calls = 1
        l2 = torch.norm((out_adv_x - self.x0).view(2, -1), dim=1)
        while l2[1] - l2[0] > tol:
            num_calls += 1
            adv_mid = (out_adv_x[1] + out_adv_x[0]) / 2
            if self.is_adversarial(adv_mid) == 1:
                out_adv_x[1] = adv_mid
            else:
                out_adv_x[0] = adv_mid
            l2 = torch.norm((out_adv_x - self.x0).view(2, -1), dim=1)

        return out_adv_x, l2, num_calls
