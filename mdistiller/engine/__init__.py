from .trainer import BaseTrainer, CRDTrainer, AugTrainer, KCDTrainer, KCDTrainer_old

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "ours": AugTrainer,
    "kcd": KCDTrainer,
    "kcd_old": KCDTrainer_old
}
