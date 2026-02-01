# Not Zscore-norm, Not MLLD, Not remove C type, Loss Rank

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from fast_soft_sort.pytorch_ops import soft_rank

def weighted_spearman(logits_student, logits_teacher, N0):
    sr = soft_rank(logits_student)
    tr = soft_rank(logits_teacher)

    weights = 1.0 / (tr + N0).pow(2)
    w_sum = weights.sum()

    x_bar = (weights * sr).sum() / w_sum
    y_bar = (weights * tr).sum() / w_sum

    cov = (weights * (sr - x_bar) * (tr - y_bar)).sum()
    var_x = (weights * (sr - x_bar).pow(2)).sum()
    var_y = (weights * (tr - y_bar).pow(2)).sum()

    return cov / torch.sqrt(var_x * var_y + 1e-8)

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

class KCD_0001(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KCD_0001, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT_1
        self.wspm_loss_weight = cfg.KD.LOSS.WSPM_WEIGHT
        self.z_score = cfg.KD.Z_SCORE_0
        self.N0 = cfg.KD.WSM

    def z_score_norm(self, logits):
        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True) + 1e-6
        result = (logits - mean) / std
        return result

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        if self.z_score:
            logits_student = self.z_score_norm(logits_student)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
            if self.z_score:
                logits_teacher = self.z_score_norm(logits_teacher)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student,
            logits_teacher,
            self.temperature,
        ) + self.kd_loss_weight * kd_loss(
            logits_student,
            logits_teacher,
            3.0,
        ) + self.kd_loss_weight * kd_loss(
            logits_student,
            logits_teacher,
            5.0,
        ) + self.kd_loss_weight * kd_loss(
            logits_student,
            logits_teacher,
            2.0,
        ) + self.kd_loss_weight * kd_loss(
            logits_student,
            logits_teacher,
            6.0,
        )

        loss_wspm = self.wspm_loss_weight * (1 - weighted_spearman(logits_student, logits_teacher, self.N0))

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_wspm": loss_wspm
        }
        return logits_student, losses_dict