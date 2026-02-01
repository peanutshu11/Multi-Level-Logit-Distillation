from ._base import Vanilla
from .KD import KD
from .KD_ours import KD_ours
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .Sonly import Sonly
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .KD_z_score import KD_z_scores
from .KD_wsm import KD_WSpearman
from .KD_zwsm import KD_ZWSpearman
from .KD_wRKKD import wRKKD
from .MLLD_KCD import MLLD_KCD
from .MLLD_KCD_old import MLLD_KCD_old
from .MLLD_KCD_1101 import MLLD_KCD_1101
from .MLLD_KCD_1100 import MLLD_KCD_1100
from .MLLD_KCD_0101 import MLLD_KCD_0101
from .MLLD_KCD_0100 import MLLD_KCD_0100
from .KCD_0001 import KCD_0001
from .KCD_0000 import KCD_0000
from .KCD_1001 import KCD_1001
from .KCD_1000 import KCD_1000

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "KD_ours": KD_ours,
    "KD_z_scores": KD_z_scores,
    "KD_WSpearman": KD_WSpearman,
    "MLLD_KCD": MLLD_KCD,
    "MLLD_KCD_old": MLLD_KCD_old,
    "MLLD_KCD_1101": MLLD_KCD_1101,
    "MLLD_KCD_1100": MLLD_KCD_1100,
    "MLLD_KCD_0101": MLLD_KCD_0101,
    "MLLD_KCD_0100": MLLD_KCD_0100,
    "KCD_0001": KCD_0001,
    "KCD_0000": KCD_0000,
    "KCD_1001": KCD_1001,
    "KCD_1000": KCD_1000,
    "KD_ZWSpearman": KD_ZWSpearman,
    "wRKKD": wRKKD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "Sonly": Sonly,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
}
