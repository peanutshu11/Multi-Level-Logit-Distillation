# Zscore-norm, MLLD, Not remove C type, Loss Rank

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

def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


class MLLD_KCD_1101(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(MLLD_KCD_1101, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.wspm_loss_weight = cfg.KD.LOSS.WSPM_WEIGHT
        self.z_score = cfg.KD.Z_SCORE
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

        loss_cc = self.kd_loss_weight * (
            cc_loss(
                logits_student,
                logits_teacher,
                self.temperature,
                reduce=True
            ) + cc_loss(
                logits_student,
                logits_teacher,
                3.0,
                reduce=True
            ) + cc_loss(
                logits_student,
                logits_teacher,
                5.0,
                reduce=True
            ) + cc_loss(
                logits_student,
                logits_teacher,
                2.0,
                reduce=True
            ) + cc_loss(
                logits_student,
                logits_teacher,
                6.0,
                reduce=True
            )
        )

        loss_bc = self.kd_loss_weight * (
            bc_loss(
                logits_student,
                logits_teacher,
                self.temperature,
                reduce=True
            ) + bc_loss(
                logits_student,
                logits_teacher,
                3.0,
                reduce=True
            ) + bc_loss(
                logits_student,
                logits_teacher,
                5.0,
                reduce=True
            ) + bc_loss(
                logits_student,
                logits_teacher,
                2.0,
                reduce=True
            ) + bc_loss(
                logits_student,
                logits_teacher,
                6.0,
                reduce=True
            )
        )

        loss_wspm = self.wspm_loss_weight * (1 - weighted_spearman(logits_student, logits_teacher, self.N0))

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_cc": loss_cc,
            "loss_bc": loss_bc,
            "loss_wspm": loss_wspm
        }
        return logits_student, losses_dict