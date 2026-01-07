from .trainer import BaseTrainer, CRDTrainer, AugTrainer, KCDTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "ours": AugTrainer,
    "kcd": KCDTrainer
}
