# ==============================================================
# KAMP B-7
# Peak Magnitude Regression
#
# 분류가 아닌 연속값 회귀:
# target = max(15분, 30분, 45분, 60분)
#
# 목적:
# - 다음 1시간의 최대 측정값을 예측
# - 회귀 prediction을 peak warning score로 사용
# - empirical peak = actual_max >= 176
#
# 모델:
# - ExtraTreesRegressor
# - LightGBM L1
# - LightGBM Huber
# - LightGBM Quantile
# - CatBoost MAE
# - CatBoost Quantile
#
# 선택:
# - DEV folds 1~3만 사용
# - AUDIT folds 4~6
# - Validation 878h는 마지막 참고 평가
#
# 운영점:
# - FPR <= 3%
# - FPR <= 5%
#
# 자동 저장:
# - model × fold 즉시 Drive 저장
# - 중단 후 동일 셀 재실행 -> 자동 재개
# ==============================================================

import os
import io
import re
import sys
import json
import time
import math
import shutil
import zipfile
import subprocess
import importlib.util
import warnings

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error
)

warnings.filterwarnings("ignore")

# ==============================================================
# 0. 설정
# ==============================================================

T0 = time.time()

SEED = 42
np.random.seed(SEED)

PEAK_THRESHOLD = 176.0

DEV_FOLDS = [1, 2, 3]
AUDIT_FOLDS = [4, 5, 6]

B7_VERSION = "B7_v1"

# ==============================================================

# Only I/O wiring is replaced; all feature and selected model definitions below
# are derived from the SHA-verified original B7 source.
ROOT = Path(os.environ["KAMP_SUBMISSION_ROOT"])
DATA = Path(os.environ["KAMP_RAW_ZIP"])
CK_ROOT = Path(os.environ["KAMP_STAGE_WORK"]) / "ck"
OUT = Path(os.environ["KAMP_STAGE_WORK"]) / "out"
CK_ROOT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST_FILE = CK_ROOT / "manifest.json"
manifest = {"status": "submission_selected_only", "run_id": "submission"}
RUN_ID = "submission"

OOF_DIR = CK_ROOT / "oof"
VAL_DIR = CK_ROOT / "validation"
for folder in (OOF_DIR, VAL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# 3. ML packages
# ==============================================================

for package in [
    "lightgbm",
    "catboost"
]:

    if importlib.util.find_spec(
        package
    ) is None:

        print(
            package,
            "설치 중..."
        )

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            package
        ])

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

print(
    "② ML 라이브러리 준비 완료"
)

# ==============================================================
# 4. Utility
# ==============================================================

def safe_name(
    x
):

    return re.sub(
        r"[^A-Za-z0-9_\-\.]+",
        "_",
        str(x)
    )


def save_csv(
    df,
    filename
):

    df.to_csv(
        OUT / filename,
        index=False,
        encoding="utf-8-sig"
    )

    df.to_csv(
        CK_ROOT / filename,
        index=False,
        encoding="utf-8-sig"
    )


def save_json(
    obj,
    filename
):

    for folder in [
        OUT,
        CK_ROOT
    ]:

        with open(
            folder / filename,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                obj,
                f,
                ensure_ascii=False,
                indent=2
            )


def read_zip_csv(
    zip_path
):

    with zipfile.ZipFile(
        zip_path
    ) as z:

        names = [
            n
            for n in z.namelist()
            if n.lower().endswith(
                ".csv"
            )
        ]

        if len(
            names
        ) != 1:

            raise ValueError(
                names
            )

        return pd.read_csv(
            io.BytesIO(
                z.read(
                    names[0]
                )
            ),
            encoding="utf-8-sig"
        )


def save_npz_atomic(
    path,
    **kwargs
):

    path = Path(
        path
    )

    temp = (
        path.parent
        / (
            path.stem
            + ".tmp.npz"
        )
    )

    np.savez_compressed(
        temp,
        **kwargs
    )

    os.replace(
        temp,
        path
    )


def load_npz(
    path
):

    try:

        data = np.load(
            path,
            allow_pickle=False
        )

        return {
            k:
                data[k]
            for k in data.files
        }

    except Exception:

        return None


# ==============================================================
# 5. Classification metrics from regression score
# ==============================================================

def counts(
    y_binary,
    alert
):

    y = np.asarray(
        y_binary,
        dtype=bool
    )

    a = np.asarray(
        alert,
        dtype=bool
    )

    tp = int(
        np.sum(
            y & a
        )
    )

    fp = int(
        np.sum(
            ~y & a
        )
    )

    fn = int(
        np.sum(
            y & ~a
        )
    )

    tn = int(
        np.sum(
            ~y & ~a
        )
    )

    precision = (
        tp / (
            tp + fp
        )
        if (
            tp + fp
        ) > 0
        else 0
    )

    recall = (
        tp / (
            tp + fn
        )
        if (
            tp + fn
        ) > 0
        else 0
    )

    fpr = (
        fp / (
            fp + tn
        )
        if (
            fp + tn
        ) > 0
        else 0
    )

    beta2 = 4

    denom = (
        beta2
        * precision
        + recall
    )

    f2 = (
        (
            1
            + beta2
        )
        * precision
        * recall
        / denom
        if denom > 0
        else 0
    )

    return {

        "TP":
            tp,

        "FP":
            fp,

        "FN":
            fn,

        "TN":
            tn,

        "Precision":
            precision,

        "Recall":
            recall,

        "FPR":
            fpr,

        "F2":
            f2
    }


def choose_threshold(
    y_binary,
    score,
    max_fpr
):

    y = np.asarray(
        y_binary,
        dtype=int
    )

    score = np.asarray(
        score,
        dtype=float
    )

    candidates = np.unique(
        score
    )

    candidates = np.sort(
        candidates
    )[::-1]

    candidates = np.r_[

        np.nextafter(
            score.max(),
            np.inf
        ),

        candidates
    ]

    best = None

    for threshold in candidates:

        m = counts(
            y,
            score >= threshold
        )

        if (
            m[
                "FPR"
            ]
            <= max_fpr
            + 1e-12
        ):

            candidate = {

                "Threshold":
                    float(
                        threshold
                    ),

                **m
            }

            if best is None:

                best = candidate

            else:

                key = (
                    candidate[
                        "Recall"
                    ],
                    -candidate[
                        "FP"
                    ],
                    candidate[
                        "Precision"
                    ],
                    candidate[
                        "F2"
                    ],
                    candidate[
                        "Threshold"
                    ]
                )

                best_key = (
                    best[
                        "Recall"
                    ],
                    -best[
                        "FP"
                    ],
                    best[
                        "Precision"
                    ],
                    best[
                        "F2"
                    ],
                    best[
                        "Threshold"
                    ]
                )

                if key > best_key:

                    best = candidate

    return best


# ==============================================================
# 6. 원본
# ==============================================================

raw = read_zip_csv(
    DATA
)

raw[
    "시간"
] = pd.to_numeric(
    raw[
        "시간"
    ],
    errors="coerce"
)

invalid = raw.loc[
    ~raw[
        "시간"
    ].between(
        0,
        23
    )
].copy()

clean = raw.loc[
    raw[
        "시간"
    ].between(
        0,
        23
    )
].copy()

clean[
    "timestamp"
] = (
    pd.to_datetime(
        clean[
            "날짜"
        ]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
        .str.zfill(
            8
        ),
        format="%Y%m%d"
    )
    +
    pd.to_timedelta(
        clean[
            "시간"
        ],
        unit="h"
    )
)

hourly = (
    clean
    .set_index(
        "timestamp"
    )
    .sort_index()
    .asfreq(
        "h"
    )
)

numeric_cols = [
    "평균",
    "생산량",
    "15분",
    "30분",
    "45분",
    "60분",
    "기온",
    "풍속",
    "습도",
    "강수량"
]

for col in numeric_cols:

    if col in hourly.columns:

        hourly[
            col
        ] = pd.to_numeric(
            hourly[
                col
            ],
            errors="coerce"
        )

target_mean = hourly[
    "평균"
]

production = hourly[
    "생산량"
]

hour_max = (
    hourly[
        [
            "15분",
            "30분",
            "45분",
            "60분"
        ]
    ]
    .max(
        axis=1,
        skipna=False
    )
)

idx = hourly.index

print(
    "③ 원본:",
    raw.shape,
    "| invalid:",
    len(
        invalid
    )
)

# ==============================================================
# 7. 기존 5856 index
# ==============================================================

X17 = pd.DataFrame(
    index=idx
)

X17[
    "hour_sin"
] = np.sin(
    2 * np.pi
    * idx.hour
    / 24
)

X17[
    "hour_cos"
] = np.cos(
    2 * np.pi
    * idx.hour
    / 24
)

X17[
    "weekday"
] = idx.dayofweek

X17[
    "month"
] = idx.month

X17[
    "is_weekend"
] = (
    idx.dayofweek >= 5
).astype(int)

for lag in [
    1,
    2,
    3,
    24,
    168
]:

    X17[
        f"mean_lag{lag}"
    ] = target_mean.shift(
        lag
    )

X17[
    "mean_rolling3"
] = (
    target_mean
    .shift(1)
    .rolling(
        3
    )
    .mean()
)

X17[
    "mean_rolling24"
] = (
    target_mean
    .shift(1)
    .rolling(
        24
    )
    .mean()
)

X17[
    "production_lag1"
] = production.shift(
    1
)

X17[
    "production_lag24"
] = production.shift(
    24
)

X17[
    "max_lag1"
] = hour_max.shift(
    1
)

X17[
    "max_lag24"
] = hour_max.shift(
    24
)

X17[
    "max_rolling3"
] = (
    hour_max
    .shift(1)
    .rolling(
        3
    )
    .mean()
)

base_mask = (
    X17.notna().all(
        axis=1
    )
    &
    target_mean.notna()
    &
    hour_max.notna()
)

base_index = idx[
    base_mask
]

if len(
    base_index
) != 5856:

    raise ValueError(
        len(
            base_index
        )
    )

cut1 = int(
    5856
    * 0.70
)

cut2 = int(
    5856
    * 0.85
)

# ==============================================================
# 8. Zero-production streak
# ==============================================================

zero_run = np.zeros(
    len(
        production
    ),
    dtype=int
)

for i, value in enumerate(
    production.to_numpy()
):

    if (
        np.isfinite(
            value
        )
        and value == 0
    ):

        zero_run[
            i
        ] = (
            1
            +
            (
                zero_run[
                    i - 1
                ]
                if i > 0
                else 0
            )
        )

zero_streak = (
    pd.Series(
        zero_run,
        index=idx
    )
    .shift(
        1
    )
)

# ==============================================================
# 9. Feature pool
# ==============================================================

F = pd.DataFrame(
    index=idx
)

# Calendar

F[
    "hour"
] = idx.hour

F[
    "hour_sin"
] = np.sin(
    2 * np.pi
    * idx.hour
    / 24
)

F[
    "hour_cos"
] = np.cos(
    2 * np.pi
    * idx.hour
    / 24
)

F[
    "weekday"
] = idx.dayofweek

F[
    "dow_sin"
] = np.sin(
    2 * np.pi
    * idx.dayofweek
    / 7
)

F[
    "dow_cos"
] = np.cos(
    2 * np.pi
    * idx.dayofweek
    / 7
)

F[
    "month"
] = idx.month

F[
    "is_weekend"
] = (
    idx.dayofweek >= 5
).astype(int)

# mean history

for lag in [
    1,
    2,
    3,
    6,
    12,
    24,
    48,
    72,
    168
]:

    F[
        f"mean_lag{lag}"
    ] = target_mean.shift(
        lag
    )

for window in [
    3,
    6,
    24
]:

    F[
        f"mean_roll{window}"
    ] = (
        target_mean
        .shift(1)
        .rolling(
            window
        )
        .mean()
    )

# max history

for lag in [
    1,
    2,
    3,
    6,
    12,
    24,
    48,
    72,
    168
]:

    F[
        f"max_lag{lag}"
    ] = hour_max.shift(
        lag
    )

for window in [
    3,
    6,
    24
]:

    F[
        f"max_rollmean{window}"
    ] = (
        hour_max
        .shift(1)
        .rolling(
            window
        )
        .mean()
    )

    F[
        f"max_rollmax{window}"
    ] = (
        hour_max
        .shift(1)
        .rolling(
            window
        )
        .max()
    )

# production

for lag in [
    1,
    2,
    3,
    6,
    24,
    48,
    168
]:

    F[
        f"prod_lag{lag}"
    ] = production.shift(
        lag
    )

for window in [
    3,
    6,
    24
]:

    F[
        f"prod_rollmean{window}"
    ] = (
        production
        .shift(1)
        .rolling(
            window
        )
        .mean()
    )

F[
    "mean_change1"
] = (
    target_mean.shift(
        1
    )
    - target_mean.shift(
        2
    )
)

F[
    "max_change1"
] = (
    hour_max.shift(
        1
    )
    - hour_max.shift(
        2
    )
)

F[
    "prod_change1"
] = (
    production.shift(
        1
    )
    - production.shift(
        2
    )
)

F[
    "zero_streak_lag1"
] = zero_streak

F[
    "zero_streak_log1p"
] = np.log1p(
    zero_streak.clip(
        lower=0
    )
)

F[
    "zero_hours_last24"
] = (
    production
    .shift(1)
    .eq(0)
    .rolling(
        24
    )
    .sum()
)

F[
    "zero_hours_last72"
] = (
    production
    .shift(1)
    .eq(0)
    .rolling(
        72
    )
    .sum()
)

historical_peak = (
    hour_max
    .ge(
        PEAK_THRESHOLD
    )
    .astype(float)
)

historical_peak.loc[
    hour_max.isna()
] = np.nan

F[
    "prior_peak_lag1"
] = historical_peak.shift(
    1
)

for window in [
    3,
    6,
    24
]:

    F[
        f"prior_peak_count{window}"
    ] = (
        historical_peak
        .shift(1)
        .rolling(
            window
        )
        .sum()
    )

# ==============================================================
# 10. Feature sets
# ==============================================================

CORE = [
    "hour",
    "hour_sin",
    "hour_cos",
    "weekday",
    "month",
    "is_weekend",
    "mean_lag1",
    "mean_lag2",
    "mean_lag3",
    "mean_lag24",
    "mean_lag168",
    "mean_roll3",
    "mean_roll24",
    "prod_lag1",
    "prod_lag24",
    "max_lag1",
    "max_lag24",
    "max_rollmean3"
]

CORE_PLUS = CORE + [
    "dow_sin",
    "dow_cos",
    "mean_lag6",
    "mean_lag12",
    "mean_lag48",
    "max_lag2",
    "max_lag3",
    "max_lag6",
    "max_lag12",
    "max_lag168",
    "max_rollmax3",
    "prod_lag2",
    "prod_lag3",
    "mean_change1",
    "max_change1",
    "prod_change1",
    "zero_streak_lag1"
]

HISTORY = CORE_PLUS + [
    "mean_lag72",
    "mean_roll6",
    "max_lag48",
    "max_lag72",
    "max_rollmean6",
    "max_rollmean24",
    "max_rollmax6",
    "max_rollmax24",
    "prod_lag6",
    "prod_lag48",
    "prod_lag168",
    "prod_rollmean3",
    "prod_rollmean6",
    "prod_rollmean24",
    "zero_streak_log1p",
    "zero_hours_last24",
    "zero_hours_last72"
]

FEATURE_SETS = {

    "CORE":
        list(
            dict.fromkeys(
                CORE
            )
        ),

    "CORE_PLUS":
        list(
            dict.fromkeys(
                CORE_PLUS
            )
        ),

    "HISTORY":
        list(
            dict.fromkeys(
                HISTORY
            )
        )
}

# ==============================================================
# 11. Dataset
# ==============================================================

Xall = F.loc[
    base_index
].copy()

target_max_all = (
    hour_max.loc[
        base_index
    ]
)

peak_all = (
    target_max_all
    >= PEAK_THRESHOLD
).astype(int)

Xtr = Xall.iloc[
    :cut1
]

yreg_tr = (
    target_max_all.iloc[
        :cut1
    ]
)

ypeak_tr = (
    peak_all.iloc[
        :cut1
    ]
)

Xva = Xall.iloc[
    cut1:cut2
]

yreg_va = (
    target_max_all.iloc[
        cut1:cut2
    ]
)

ypeak_va = (
    peak_all.iloc[
        cut1:cut2
    ]
)

if len(
    Xtr
) != 4099:

    raise ValueError(
        "train !=4099"
    )

if len(
    Xva
) != 878:

    raise ValueError(
        "val !=878"
    )

if int(
    ypeak_va.sum()
) != 156:

    raise ValueError(
        "peak !=156"
    )

print(
    "④ Dataset:",
    len(
        Xtr
    ),
    len(
        Xva
    ),
    "| val peak:",
    int(
        ypeak_va.sum()
    )
)

# ==============================================================
# 12. CV
# ==============================================================

folds = list(

    TimeSeriesSplit(
        n_splits=6,
        test_size=336
    ).split(
        Xtr
    )
)

coverage = []

for fold_no, (
    train_idx,
    eval_idx
) in enumerate(
    folds,
    start=1
):

    coverage.append({

        "fold":
            fold_no,

        "train_n":
            len(
                train_idx
            ),

        "eval_n":
            len(
                eval_idx
            ),

        "eval_peak":
            int(
                ypeak_tr.iloc[
                    eval_idx
                ].sum()
            )
    })

coverage = pd.DataFrame(
    coverage
)

save_csv(
    coverage,
    "B7_cv_coverage.csv"
)

# ==============================================================
# 13. Candidate models
# ==============================================================

CANDIDATES = []

# --------------------------------------------------------------
# ExtraTrees
# --------------------------------------------------------------

for feature_set in [
    "CORE",
    "CORE_PLUS",
    "HISTORY"
]:

    for leaf in [
        1,
        2,
        4
    ]:

        for max_features in [
            "sqrt",
            0.7
        ]:

            CANDIDATES.append({

                "family":
                    "ET",

                "name":
                    (
                        f"ETR_{feature_set}"
                        f"_leaf{leaf}"
                        f"_mf{str(max_features).replace('.', 'p')}"
                    ),

                "feature_set":
                    feature_set,

                "leaf":
                    leaf,

                "max_features":
                    max_features
            })

# --------------------------------------------------------------
# LightGBM L1
# --------------------------------------------------------------

for feature_set in [
    "CORE",
    "CORE_PLUS",
    "HISTORY"
]:

    for leaves in [
        7,
        15,
        31
    ]:

        CANDIDATES.append({

            "family":
                "LGB_L1",

            "name":
                (
                    f"LGB_L1_{feature_set}"
                    f"_leaf{leaves}"
                ),

            "feature_set":
                feature_set,

            "leaves":
                leaves
        })

# --------------------------------------------------------------
# LightGBM Huber
# --------------------------------------------------------------

for feature_set in [
    "CORE",
    "CORE_PLUS"
]:

    for leaves in [
        15,
        31
    ]:

        CANDIDATES.append({

            "family":
                "LGB_HUBER",

            "name":
                (
                    f"LGB_HUBER_{feature_set}"
                    f"_leaf{leaves}"
                ),

            "feature_set":
                feature_set,

            "leaves":
                leaves
        })

# --------------------------------------------------------------
# LightGBM Quantile
# --------------------------------------------------------------

for feature_set in [
    "CORE",
    "CORE_PLUS"
]:

    for alpha in [
        0.75,
        0.85,
        0.95
    ]:

        CANDIDATES.append({

            "family":
                "LGB_Q",

            "name":
                (
                    f"LGB_Q{int(alpha*100)}"
                    f"_{feature_set}"
                ),

            "feature_set":
                feature_set,

            "alpha":
                alpha
        })

# --------------------------------------------------------------
# CatBoost
# --------------------------------------------------------------

for feature_set in [
    "CORE",
    "CORE_PLUS"
]:

    for depth in [
        4,
        6
    ]:

        CANDIDATES.append({

            "family":
                "CAT_MAE",

            "name":
                (
                    f"CAT_MAE_{feature_set}"
                    f"_d{depth}"
                ),

            "feature_set":
                feature_set,

            "depth":
                depth
        })

        CANDIDATES.append({

            "family":
                "CAT_Q80",

            "name":
                (
                    f"CAT_Q80_{feature_set}"
                    f"_d{depth}"
                ),

            "feature_set":
                feature_set,

            "depth":
                depth
        })

print(
    "⑤ 회귀 후보:",
    len(
        CANDIDATES
    )
)

# ==============================================================
# 14. Model factory
# ==============================================================

def make_model(
    spec
):

    family = spec[
        "family"
    ]

    if family == "ET":

        return ExtraTreesRegressor(

            n_estimators=600,

            min_samples_leaf=
                spec[
                    "leaf"
                ],

            max_features=
                spec[
                    "max_features"
                ],

            random_state=SEED,

            n_jobs=-1
        )

    if family == "LGB_L1":

        return LGBMRegressor(

            objective="regression_l1",

            n_estimators=600,

            learning_rate=0.025,

            num_leaves=
                spec[
                    "leaves"
                ],

            min_child_samples=30,

            subsample=0.9,

            colsample_bytree=0.9,

            reg_lambda=1.0,

            random_state=SEED,

            n_jobs=-1,

            verbosity=-1
        )

    if family == "LGB_HUBER":

        return LGBMRegressor(

            objective="huber",

            n_estimators=600,

            learning_rate=0.025,

            num_leaves=
                spec[
                    "leaves"
                ],

            min_child_samples=30,

            subsample=0.9,

            colsample_bytree=0.9,

            reg_lambda=1.0,

            random_state=SEED,

            n_jobs=-1,

            verbosity=-1
        )

    if family == "LGB_Q":

        return LGBMRegressor(

            objective="quantile",

            alpha=
                spec[
                    "alpha"
                ],

            n_estimators=600,

            learning_rate=0.025,

            num_leaves=15,

            min_child_samples=30,

            subsample=0.9,

            colsample_bytree=0.9,

            reg_lambda=1.0,

            random_state=SEED,

            n_jobs=-1,

            verbosity=-1
        )

    if family == "CAT_MAE":

        return CatBoostRegressor(

            iterations=700,

            depth=
                spec[
                    "depth"
                ],

            learning_rate=0.025,

            loss_function="MAE",

            random_seed=SEED,

            verbose=False,

            allow_writing_files=False,

            thread_count=-1
        )

    if family == "CAT_Q80":

        return CatBoostRegressor(

            iterations=700,

            depth=
                spec[
                    "depth"
                ],

            learning_rate=0.025,

            loss_function="Quantile:alpha=0.8",

            random_seed=SEED,

            verbose=False,

            allow_writing_files=False,

            thread_count=-1
        )

    raise ValueError(
        spec
    )

# ==============================================================
# 15. OOF checkpoint
# ==============================================================

def checkpoint_path(
    model_name,
    fold_no
):

    folder = (
        OOF_DIR
        / safe_name(
            model_name
        )
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        folder
        / f"fold_{fold_no}.npz"
    )


def checkpoint_valid(
    path,
    eval_idx
):

    if not Path(
        path
    ).exists():

        return False

    loaded = load_npz(
        path
    )

    if loaded is None:

        return False

    if (
        "eval_idx"
        not in loaded
        or "pred"
        not in loaded
    ):

        return False

    if not np.array_equal(
        loaded[
            "eval_idx"
        ].astype(int),
        np.asarray(
            eval_idx,
            dtype=int
        )
    ):

        return False

    if not np.isfinite(
        loaded[
            "pred"
        ]
    ).all():

        return False

    return True


# ==============================================================
