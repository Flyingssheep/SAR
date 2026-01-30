import copy
import torch
from scipy.fftpack import idct
import math
import sys
import DataTools
import csv

class V(object):
    def __init__(self):
        self.ADBmax = 0
        self.RealL2 = 0
        self.RealLinf = 0
        self.adv_v = None

class Attacker():
    def __init__(self, args, model, im_orig, imgi, attack_method='HSJA',
                 tar_img=None, tar_label=None, dim_reduc_factor=4,
                 iteration=100, initial_query=30, tol=1e-4, sigma=3e-4):
        self.model = model
        self.x0 = im_orig.cuda()
        self.x0_label = model.predict_label(self.x0).cpu().item()
        self.out_label = self.x0_label
        self.tar_img = tar_img
        self.tar_label = tar_label
        if tar_img != None:
            self.tar_img = torch.unsqueeze(self.tar_img, dim=0)
        self.dim_reduc_factor = dim_reduc_factor
        if im_orig.shape[3] < 224:
            self.dim_reduc_factor = 1
        self.Max_iter = iteration
        self.Init_query = initial_query

        self.tol = tol
        self.sigma = sigma
        self.attack_method = attack_method
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.args = args
        self.Img_i = imgi
        self.old_best_adv = V()
        self.File_string = "HSJA"
        self.Img_result = []
        self.heatmaps = self.Img_result
        self.L2_line = [[0, self.x0.numel() ** 0.5]]
        self.Linf_line = [[0, 1.0]]
        self.success = -1
        self.query = 0
        self.ACCquery = 0
        self.ACCiter_n = 1

        self.Noises = []
        self.total_Kbridge = []

        self.d = self.x0.numel()  # 总维度

    def generate_tensor_with_fixed_similarity(self, v, cos_sim):
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        v_unit = v / v_norm
        random_tensor = torch.randn_like(v, dtype=torch.float32).cuda()
        dot_product = torch.sum(random_tensor * v_unit, dim=-1, keepdim=True)
        projection = dot_product * v_unit
        perpendicular_vector = random_tensor - projection
        perpendicular_unit = perpendicular_vector / torch.norm(perpendicular_vector, dim=-1, keepdim=True)
        v2 = cos_sim * v_norm * v_unit + (1 - cos_sim ** 2) ** 0.5 * v_norm * perpendicular_unit
        return v2

    def normal_vector_approximation(self, boundary_x, q_max):
        x0 = self.x0
        random_noises = None
        boundary_v = boundary_x - x0
        boundary_v_l2 = torch.norm(boundary_v).item()
        if self.dim_reduc_factor < 1.0:
            raise Exception(
                "The dimension reduction factor should be greater than 1 for reduced dimension, and should be 1 for Full dimensional image space.")
        if self.dim_reduc_factor > 1.0:
            fill_size = int(x0.shape[-1] / self.dim_reduc_factor)
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
        labels_out = []
        pixnum = torch.numel(x0)
        for i in range(q_max):
            k_to_one = (math.sqrt(pixnum) / (self.dim_reduc_factor * torch.norm(random_noises[i]).item()))
            random_noises[i] *= (k_to_one * self.sigma)
            noise_l2 = torch.norm(boundary_v + random_noises[i])
            #if noise_l2 < boundary_v_l2:
            #    random_noises[i] *= -1
            labels_out.append(self.model.predict_label(x0 + boundary_v + random_noises[i]))
        self.query += q_max

        z = []  # sign of grad_tmp
        for i, predict_label in enumerate(labels_out):
            if self.tar_img == None:
                if predict_label == self.x0_label:
                    z.append(-1)
                    random_noises[i] *= -1
                else:
                    z.append(1)
            if self.tar_img != None:
                if predict_label != self.tar_label:
                    z.append(-1)
                    random_noises[i] *= -1
                else:
                    z.append(1)
        normal_v = sum(random_noises)
        mean_z = sum(z) / q_max
        return normal_v, mean_z

    def geometric_progression_step_size(self, x_t, v_t, init_step_size):
        step_size = init_step_size
        max_trials = 20
        for _ in range(max_trials):
            x_candidate = x_t + step_size * v_t
            x_candidate = torch.clamp(x_candidate, 0.0, 1.0)
            if self.is_adversarial(x_candidate) == 1:
                return step_size, x_candidate
            step_size = step_size / 2
        return step_size, x_t + step_size * v_t

    def attack(self):
        if self.tar_img is not None:
            x_adv, query_random = self.tar_img.clone(), 0
        else:
            x_adv, query_random = self.find_init_adv_x(self.x0)

        boundary_points, l2_distances, query_bin = self.bin_search_fast(
            torch.cat((self.x0.clone(), x_adv.clone()), dim=0), self.tol)
        x_t = boundary_points[1]

        sys.stdout.write(
            f'\rImg{self.Img_i} query{self.query :.0f} \tIter{0 :.0f} \treal_d={l2_distances[1]:.6f}')
        sys.stdout.flush()

        for t in range(self.Max_iter):
            current_dist = torch.norm(x_t - self.x0).item()
            q_norm_v = int(self.Init_query * (t + 1) ** 0.5)
            delta_t = self.d ** (-1) * current_dist  # HSJA δ
            normal_v, mean_z = self.normal_vector_approximation(x_t, q_norm_v)

            # ξ_t = ||x_t - x*|| / sqrt(t+1)
            init_step_size = current_dist / math.sqrt(t + 1)
            step_size, x_candidate = self.geometric_progression_step_size(x_t, normal_v, init_step_size)

            search_points = torch.cat((self.x0.clone(), x_candidate.unsqueeze(0)), dim=0)
            boundary_points_new, l2_new, query_proj = self.bin_search_fast(search_points, self.tol)
            x_t_next = boundary_points_new[1]

            if self.tar_img is not None:
                target_proj_points = torch.cat((self.tar_img.clone(), x_t_next.unsqueeze(0)), dim=0)
            else:
                target_proj_points = torch.cat((self.x0.clone(), x_t_next.unsqueeze(0)), dim=0)

            final_boundary, final_l2, query_final = self.bin_search_fast(target_proj_points, self.tol)
            x_t = final_boundary[1]

            current_dist = torch.norm(x_t - self.x0).item()
            current_linf = torch.max(torch.abs(x_t - self.x0)).item()

            sys.stdout.write(
                f'\rImg{self.Img_i} Query{self.query:5d} Iter{t + 1:3d} '
                f'dist={current_dist:.6f} step={step_size:.4f} meanZ={mean_z:.2}')
            sys.stdout.flush()

            self.old_best_adv.RealL2 = current_dist
            self.old_best_adv.RealLinf = current_linf
            self.old_best_adv.ADBmax = current_dist
            self.old_best_adv.adv_v = x_t - self.x0

            self.L2_line.append([self.query, current_dist])
            self.Linf_line.append([self.query, current_linf])

            if self.query >= self.args.budget:
                break
            if self.success == -1 and current_dist <= self.args.epsilon:
                self.success = 1
                self.ACCquery, self.ACCiter_n = self.query, t
            if self.args.early == 1 and self.success == 1:
                break

        self.out_label = self.model.predict_label(torch.clamp(x_t, 0.0, 1.0)).cpu().item()
        adv_v = x_t - self.x0
        advimg = 0.5 * (1.0 + adv_v / self.old_best_adv.RealLinf)
        self.Img_result = [advimg, self.x0, self.x0 + adv_v]
        self.heatmaps = copy.deepcopy(self.Img_result)

        if self.tar_img is not None:
            self.heatmaps[0] = self.tar_img.clone()
            self.heatmaps[1] = 0.5 * (1.0 + self.tar_img)
        self.File_string = (f"Img{self.Img_i},I-Q[{self.ACCiter_n}-{self.query}], "
                            f"ADB{self.old_best_adv.ADBmax:.4f},"
                            f"LB[{self.x0_label}-{self.out_label}]")
        print(f'\nFinal: queries={self.query}, distance={current_dist:.6f}, '
              f'success={self.success}')
        return []

    def is_adversarial(self, image):
        predict_label = self.model.predict_label(torch.clamp(image, 0.0, 1.0)).cpu().item()
        self.query += 1
        if self.tar_img is None:
            is_adv = predict_label != self.x0_label
        else:
            is_adv = predict_label == self.tar_label
        return 1 if is_adv else -1

    def find_init_adv_x(self, image):
        num_calls = 1
        step = 0.02
        perturbed = image
        while self.is_adversarial(perturbed) == -1:
            if self.args.budget / 10 <= num_calls <= self.args.budget / 10 + 2:
                perturbed = torch.ones(image.shape).to(self.device)
                if self.is_adversarial(perturbed) == -1:
                    perturbed = torch.zeros(image.shape).to(self.device)
            else:
                pert = torch.randn(image.shape).cuda()
                perturbed = torch.clamp(image + num_calls * step * pert, 0.0, 1.0)
                num_calls += 1

            sys.stdout.write(f'\rImg{self.Img_i} query{self.query:.0f} '
                             f'real_d={torch.norm(perturbed):.6f} ')
            sys.stdout.flush()
        return perturbed, num_calls

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