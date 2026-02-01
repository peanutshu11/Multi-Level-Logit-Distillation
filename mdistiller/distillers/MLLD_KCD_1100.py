# Zscore-norm, MLLD, Not remove C type, Not Loss Rank

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

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
        self.z_score = cfg.KD.Z_SCORE

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

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_cc": loss_cc,
            "loss_bc": loss_bc,
        }
        return logits_student, losses_dict