import os

KAGGLE_ROOT = r"D:\New research fog work\kaggle_fog\Competition Dataset"
DAPHNET_ROOT = r"D:\Daphnet dataset\dataset"
DAPHNET_AXES = (("tr_y", -1.0), ("tr_z", 1.0), ("tr_x", 1.0))

PREP = "prepared_fi"
OUT = "results_fi"

FS = 64
HOP = 16
WIN_LONG = 256
WIN_SHORT = 128
G = 9.80665

LOC_BAND = (0.5, 3.0)
FRZ_BAND = (3.0, 8.0)
ARREST_PAST = (40, 16)

PRE_S = 4.0
POST_S = 3.0
FA_BUDGETS = [5.0, 10.0, 15.0]
MAIN_BUDGET = 10.0
SPEC_FLOOR = 0.80

N_FOLDS = 5
N_SHIFTS = 1000
SEED = 42

DEV_DATASETS = ["tdcsfog", "defog", "daphnet"]


def hop_s():
    return HOP / float(FS)


def ensure(path):
    os.makedirs(path, exist_ok=True)
    return path
