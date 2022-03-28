# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
# This code is the implementation of the DiGS loss functions
# It was partly based on SIREN implementation and architecture but with several significant modifications.
# for the original SIREN version see: https://github.com/vsitzmann/siren

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils

def eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type='abs'):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal 
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad
    
    if eikonal_type == 'abs':
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()
    
    return eikonal_term

def latent_rg_loss(latent_reg, device):
    # compute the VAE latent representation regularization loss
    if latent_reg is not None:
        reg_loss = latent_reg.mean()
    else:
        reg_loss = torch.tensor([0.0], device=device)

    return reg_loss

class DiGSLoss(nn.Module):
    def __init__(self, weights=[3e3, 1e2, 1e2, 5e1, 1e2], loss_type='siren', div_decay='none',
                 div_type='l2', div_clamp=50):
        super().__init__()
        self.weights = weights #sdf, intern, normal, eikonal, div
        self.loss_type = loss_type
        self.div_decay = div_decay
        self.div_type = div_type
        self.div_clamp = div_clamp
        self.use_curvs = True if 'curv' in self.loss_type else False
        self.use_div = True if 'div' in self.loss_type else False

    def forward(self, output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt=None, nonmnfld_dist=None, curvatures=None,
                network_weights=None, nonmnfld_div_gt=None, mnfld_div_gt=None):
        dims = mnfld_points.shape[-1]
        device = mnfld_points.device

        #########################################
        # Compute required terms
        #########################################

        non_manifold_pred = output_pred["nonmanifold_pnts_pred"]
        manifold_pred = output_pred["manifold_pnts_pred"]
        latent_reg = output_pred["latent_reg"]
        latent = output_pred["latent"]

        div_loss = torch.tensor([0.0], device=mnfld_points.device)
        curv_term = torch.tensor([0.0], device=mnfld_points.device)
        latent_reg_term = torch.tensor([0.0], device=mnfld_points.device)

        # compute gradients for div (divergence), curl and curv (curvature)
        if manifold_pred is not None:
            mnfld_grad = utils.gradient(mnfld_points, manifold_pred)
        else:
            mnfld_grad = None

        nonmnfld_grad = utils.gradient(nonmnfld_points, non_manifold_pred)

        # curv term
        if self.use_curvs:
            if curvatures is None:
                raise Warning(" loss type requires curvatuers but none were provided.")
            mnfld_dx = utils.gradient(mnfld_points, mnfld_grad[:, :, 0])
            mnfld_dy = utils.gradient(mnfld_points, mnfld_grad[:, :, 1])
            if dims == 3:
                mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
                mnfld_divergence = mnfld_dx[:, :, 0] + mnfld_dy[:, :, 1] + mnfld_dz[:, :, 2]
            else:
                mnfld_divergence = mnfld_dx[:, :, 0] + mnfld_dy[:, :, 1]

            if self.div_type == 'l2':
                gt_mean_curvature = torch.square(torch.sum(curvatures, dim=-1))
                curv_term = torch.clamp((torch.square(mnfld_divergence) - gt_mean_curvature), 0.1, self.div_clamp)
            elif self.div_type == 'l1':
                gt_mean_curvature = torch.abs(torch.sum(curvatures, dim=-1))
                curv_term = torch.clamp((torch.abs(mnfld_divergence) - gt_mean_curvature), 0.1, self.div_clamp)
                # curv_term = (torch.abs(mnfld_divergence) - gt_mean_curvature)

        # div_term
        if self.use_div:
            nonmnfld_dx = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 0])
            nonmnfld_dy = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 1])
            if dims == 3:
                nonmnfld_dz = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 2])
                nonmnfld_divergence = nonmnfld_dx[:, :, 0] + nonmnfld_dy[:, :, 1] + nonmnfld_dz[:, :, 2]
            else:
                nonmnfld_divergence = nonmnfld_dx[:, :, 0] + nonmnfld_dy[:, :, 1]
            nonmnfld_divergence[nonmnfld_divergence.isnan()] = 0

            if self.div_type == 'l2':
                nonmnfld_divergence_term = torch.clamp(torch.square(nonmnfld_divergence), 0.1, self.div_clamp)
            elif self.div_type == 'l1':
                nonmnfld_divergence_term = torch.clamp(torch.abs(nonmnfld_divergence), 0.1, self.div_clamp)
            elif self.div_type == 'gt_l2':
                nonmnfld_divergence_term = torch.square(nonmnfld_divergence - nonmnfld_div_gt) + \
                                           torch.square(mnfld_divergence - mnfld_div_gt)
            elif self.div_type == 'gt_l1':
                nonmnfld_divergence_term = torch.abs(nonmnfld_divergence.abs() - nonmnfld_div_gt.abs()) + \
                                           torch.abs(mnfld_divergence.abs() - mnfld_div_gt.abs())
            else:
                raise Warning("unsupported divergence type. only suuports l1 and l2")

            div_loss = nonmnfld_divergence_term.mean() #+ mnfld_divergence_term.mean()

        # eikonal term
        eikonal_term = eikonal_loss(nonmnfld_grad, mnfld_grad=mnfld_grad, eikonal_type='abs')

        # latent regulariation for multiple shape learning
        latent_reg_term = latent_rg_loss(latent_reg, device)
        
        # normal term
        if mnfld_n_gt is not None:
            if 'igr' in self.loss_type:
                normal_term = ((mnfld_grad - mnfld_n_gt).abs()).norm(2, dim=1).mean()
            else:
                normal_term = (1 - torch.abs(torch.nn.functional.cosine_similarity(mnfld_grad, mnfld_n_gt, dim=-1))).mean()

        # signed distance function term
        sdf_term = torch.abs(manifold_pred).mean()

        # inter term
        inter_term = torch.exp(-1e2 * torch.abs(non_manifold_pred)).mean()
        # inter_term = torch.exp(-1e0 * torch.abs(non_manifold_pred)).mean()

        #########################################
        # Losses
        #########################################

        # losses used in the paper
        if self.loss_type == 'siren': # SIREN loss
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + \
                   self.weights[2]*normal_term + self.weights[3]*eikonal_term
        elif self.loss_type == 'siren_wo_n': # SIREN loss without normal constraint
            self.weights[2] = 0
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + self.weights[3]*eikonal_term
        elif self.loss_type == 'igr': # IGR loss
            self.weights[1] = 0
            loss = self.weights[0]*sdf_term + self.weights[2]*normal_term + self.weights[3]*eikonal_term
        elif self.loss_type == 'igr_wo_n': # IGR without normals loss
            self.weights[1] = 0
            self.weights[2] = 0
            loss = self.weights[0]*sdf_term + self.weights[3]*eikonal_term
        elif self.loss_type == 'siren_w_div': # SIREN loss with divergence term
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + \
                   self.weights[2]*normal_term + self.weights[3]*eikonal_term + \
                   self.weights[4] * div_loss
        elif self.loss_type == 'siren_wo_n_w_div':  # SIREN loss without normals and with divergence constraint
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + self.weights[3]*eikonal_term + \
                   self.weights[4] * div_loss
        elif self.loss_type == 'siren_w_curv':
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + \
                   self.weights[2]*normal_term + self.weights[3]*eikonal_term + self.weights[4]*curv_term.mean()
        elif self.loss_type == 'siren_w_div_w_curv':
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + \
                   self.weights[2]*normal_term + self.weights[3]*eikonal_term + self.weights[4]*curv_term.mean() \
                   + self.weights[4]*div_loss
        elif self.loss_type == 'siren_wo_n_w_div_w_curv':
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + self.weights[3]*eikonal_term + \
                   self.weights[4] * (div_loss + curv_term.mean())
        else:
            raise Warning("unrecognized loss type")
        
        # If multiple surface reconstruction, then latent and latent_reg are defined so reg_term need to be used
        if latent is not None and latent_reg is not None:
            loss += self.weights[5] * latent_reg_term
        
        return {"loss": loss, 'sdf_term': sdf_term, 'inter_term': inter_term, 'latent_reg_term': latent_reg_term,
                'eikonal_term': eikonal_term, 'normals_loss': normal_term, 'div_loss': div_loss, 
                'curv_loss': curv_term.mean()}, mnfld_grad

    def update_div_weight(self, current_iteration, n_iterations, params=None):
        # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
        # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.75, 1] of the training process, the weight should 
        #   be [1e2,1e2,0.0,0.0]. Between these points, the weights change as per the div_decay parameter, e.g. linearly, quintic, step etc.
        #   Thus the weight stays at 1e2 from 0-0.5, decay from 1e2 to 0.0 from 0.5-0.75, and then stays at 0.0 from 0.75-1.

        if not hasattr(self, 'decay_params_list'):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            self.decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]],[0, *params[1:-1][::2], 1]))

        curr = current_iteration / n_iterations
        we, e = min([tup for tup in self.decay_params_list if tup[1]>= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in self.decay_params_list if tup[1]<= curr], key=lambda tup: tup[1])

        # Divergence term anealing functions
        if self.div_decay == 'linear': # linearly decrease weight from iter s to iter e
            if current_iteration < s*n_iterations:
                self.weights[4] = w0
            elif  current_iteration >= s*n_iterations and current_iteration < e*n_iterations:
                self.weights[4] = w0 + (we - w0) * (current_iteration/n_iterations - s) / (e - s)
            else:
                self.weights[4] = we
        elif self.div_decay == 'quintic': # linearly decrease weight from iter s to iter e
            if current_iteration < s*n_iterations:
                self.weights[4] = w0
            elif  current_iteration >= s*n_iterations and current_iteration < e*n_iterations:
                self.weights[4] = w0 + (we - w0) * (1 - (1 -(current_iteration/n_iterations - s) / (e - s))**5)
            else:
                self.weights[4] = we
        elif self.div_decay == 'step': # change weight at s
            if current_iteration < s*n_iterations:
                self.weights[4] = w0
            else:
                self.weights[4] = we
        elif self.div_decay == 'none':
            pass
        else:
            raise Warning("unsupported div decay value")