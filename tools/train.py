import os
import argparse
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict


def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    train_loader, val_loader, num_data, num_classes, train_set = get_dataset_strong(cfg)

    # vanilla
    # if cfg.DISTILLER.TYPE == "NONE":
    #     if cfg.DATASET.TYPE == "imagenet":
    #         model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
    #     else:
    #         model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
    #             num_classes=num_classes
    #         )
    #     distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)

    # distillation
    print(log_msg("Loading teacher model", "INFO"))
    net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]

    # load teacher
    assert (
        pretrain_model_path is not None
    ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
    model_teacher = net(num_classes=num_classes)
    model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])

    # load student
    model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)

    # load distiller
    distiller = distiller_dict[cfg.DISTILLER.TYPE](
        model_student, model_teacher, cfg
    )
    distiller = torch.nn.DataParallel(distiller.cuda())

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg, train_set
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts)
