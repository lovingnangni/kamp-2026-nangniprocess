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
# 1. Drive
# ==============================================================

from google.colab import drive

if not Path(
    "/content/drive/MyDrive"
).exists():

    drive.mount(
        "/content/drive"
    )

ROOT = Path(
    "/content/drive/MyDrive/KAMP_2026"
)

DATA = (
    ROOT
    / "KAMP_5_data.zip"
)

if not DATA.exists():

    raise FileNotFoundError(
        f"데이터 없음: {DATA}"
    )

# ==============================================================
# 2. B7 영구 체크포인트
# ==============================================================

CK_ROOT = (
    ROOT
    / "KAMP_B7_RESUME_CHECKPOINT"
)

CK_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_FILE = (
    CK_ROOT
    / "manifest.json"
)

if MANIFEST_FILE.exists():

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        manifest = json.load(f)

    if (
        manifest.get(
            "version"
        )
        != B7_VERSION
    ):

        raise RuntimeError(
            """
기존 B7 체크포인트가 다른 버전입니다.

KAMP_B7_RESUME_CHECKPOINT 폴더를 삭제한 뒤
다시 실행하세요.
"""
        )

    RUN_ID = (
        manifest[
            "run_id"
        ]
    )

    print(
        "♻️ 기존 B7 체크포인트 재개"
    )

else:

    RUN_ID = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    manifest = {

        "version":
            B7_VERSION,

        "run_id":
            RUN_ID,

        "status":
            "running",

        "created_at":
            datetime.now().isoformat()
    }

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        "🆕 새 B7 실행"
    )

OUT = Path(
    "/content/KAMP_B7_results"
)

if OUT.exists():

    shutil.rmtree(
        OUT
    )

OUT.mkdir(
    parents=True,
    exist_ok=True
)

OOF_DIR = (
    CK_ROOT
    / "oof"
)

OOF_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VAL_DIR = (
    CK_ROOT
    / "validation"
)

VAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(
    "① Drive:",
    ROOT
)

print(
    "Checkpoint:",
    CK_ROOT
)

# ==============================================================
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
# 16. 모든 model × fold 실행
# ==============================================================

TOTAL_TASKS = (
    len(
        CANDIDATES
    )
    * 6
)

completed = 0

for spec in CANDIDATES:

    for fold_no, (
        train_idx,
        eval_idx
    ) in enumerate(
        folds,
        start=1
    ):

        if checkpoint_valid(
            checkpoint_path(
                spec[
                    "name"
                ],
                fold_no
            ),
            eval_idx
        ):

            completed += 1

print(
    "기존 저장:",
    completed,
    "/",
    TOTAL_TASKS
)

durations = []

for model_no, spec in enumerate(
    CANDIDATES,
    start=1
):

    print()
    print(
        f"⑥ Model "
        f"{model_no}/"
        f"{len(CANDIDATES)} "
        f"{spec['name']}"
    )

    features = FEATURE_SETS[
        spec[
            "feature_set"
        ]
    ]

    for fold_no, (
        train_idx,
        eval_idx
    ) in enumerate(
        folds,
        start=1
    ):

        path = checkpoint_path(
            spec[
                "name"
            ],
            fold_no
        )

        if checkpoint_valid(
            path,
            eval_idx
        ):

            loaded = load_npz(
                path
            )

            pred = loaded[
                "pred"
            ]

            yy = yreg_tr.iloc[
                eval_idx
            ]

            print(
                f"   fold {fold_no}: "
                f"SKIP | "
                f"MAE="
                f"{mean_absolute_error(yy,pred):.3f}"
            )

            continue

        started = time.time()

        X_train = (
            Xtr.iloc[
                train_idx
            ][
                features
            ]
        )

        X_eval = (
            Xtr.iloc[
                eval_idx
            ][
                features
            ]
        )

        y_train = (
            yreg_tr.iloc[
                train_idx
            ]
        )

        imputer = SimpleImputer(
            strategy="median"
        )

        X_train_i = (
            imputer.fit_transform(
                X_train
            )
        )

        X_eval_i = (
            imputer.transform(
                X_eval
            )
        )

        model = make_model(
            spec
        )

        model.fit(
            X_train_i,
            y_train
        )

        pred = model.predict(
            X_eval_i
        )

        save_npz_atomic(
            path,
            eval_idx=np.asarray(
                eval_idx,
                dtype=int
            ),
            pred=np.asarray(
                pred,
                dtype=float
            )
        )

        elapsed = (
            time.time()
            - started
        )

        durations.append(
            elapsed
        )

        completed += 1

        mae = mean_absolute_error(
            yreg_tr.iloc[
                eval_idx
            ],
            pred
        )

        remaining = (
            TOTAL_TASKS
            - completed
        )

        eta = (
            np.mean(
                durations
            )
            * remaining
            / 60
            if durations
            else np.nan
        )

        print(
            f"   fold {fold_no}: "
            f"MAE={mae:.3f} | "
            f"{elapsed:.1f}s | "
            f"{completed}/{TOTAL_TASKS} | "
            f"ETA {eta:.1f}m"
        )

# ==============================================================
# 17. OOF 통합
# ==============================================================

oof_parts = []

for spec in CANDIDATES:

    for fold_no, (
        train_idx,
        eval_idx
    ) in enumerate(
        folds,
        start=1
    ):

        loaded = load_npz(
            checkpoint_path(
                spec[
                    "name"
                ],
                fold_no
            )
        )

        oof_parts.append(
            pd.DataFrame({

                "model":
                    spec[
                        "name"
                    ],

                "family":
                    spec[
                        "family"
                    ],

                "feature_set":
                    spec[
                        "feature_set"
                    ],

                "fold":
                    fold_no,

                "timestamp":
                    Xtr.index[
                        eval_idx
                    ],

                "actual_max":
                    yreg_tr.iloc[
                        eval_idx
                    ].to_numpy(),

                "actual_peak":
                    ypeak_tr.iloc[
                        eval_idx
                    ].to_numpy(
                        dtype=int
                    ),

                "predicted_max":
                    loaded[
                        "pred"
                    ]
            })
        )

oof = pd.concat(
    oof_parts,
    ignore_index=True
)

save_csv(
    oof,
    "B7_oof_predictions.csv"
)

# ==============================================================
# 18. DEV / AUDIT ranking
# ==============================================================

score_rows = []

for model_name, sub in (
    oof.groupby(
        "model"
    )
):

    dev = sub.loc[
        sub[
            "fold"
        ].isin(
            DEV_FOLDS
        )
    ].copy()

    audit = sub.loc[
        sub[
            "fold"
        ].isin(
            AUDIT_FOLDS
        )
    ].copy()

    dev_y = dev[
        "actual_peak"
    ].to_numpy(
        dtype=int
    )

    dev_score = dev[
        "predicted_max"
    ].to_numpy(
        dtype=float
    )

    dev_actual_max = dev[
        "actual_max"
    ].to_numpy(
        dtype=float
    )

    audit_y = audit[
        "actual_peak"
    ].to_numpy(
        dtype=int
    )

    audit_score = audit[
        "predicted_max"
    ].to_numpy(
        dtype=float
    )

    fpr3 = choose_threshold(
        dev_y,
        dev_score,
        0.03
    )

    fpr5 = choose_threshold(
        dev_y,
        dev_score,
        0.05
    )

    audit3 = counts(
        audit_y,
        audit_score
        >= fpr3[
            "Threshold"
        ]
    )

    audit5 = counts(
        audit_y,
        audit_score
        >= fpr5[
            "Threshold"
        ]
    )

    dev_ap = float(
        average_precision_score(
            dev_y,
            dev_score
        )
    )

    audit_ap = float(
        average_precision_score(
            audit_y,
            audit_score
        )
    )

    dev_mae = float(
        mean_absolute_error(
            dev_actual_max,
            dev_score
        )
    )

    dev_rmse = float(
        np.sqrt(
            mean_squared_error(
                dev_actual_max,
                dev_score
            )
        )
    )

    score_rows.append({

        "model":
            model_name,

        "family":
            sub[
                "family"
            ].iloc[0],

        "feature_set":
            sub[
                "feature_set"
            ].iloc[0],

        "DEV_AP":
            dev_ap,

        "AUDIT_AP":
            audit_ap,

        "DEV_MAE":
            dev_mae,

        "DEV_RMSE":
            dev_rmse,

        "FPR3_threshold":
            fpr3[
                "Threshold"
            ],

        "FPR3_DEV_TP":
            fpr3[
                "TP"
            ],

        "FPR3_DEV_FP":
            fpr3[
                "FP"
            ],

        "FPR3_DEV_FN":
            fpr3[
                "FN"
            ],

        "FPR3_DEV_Recall":
            fpr3[
                "Recall"
            ],

        "FPR3_AUDIT_TP":
            audit3[
                "TP"
            ],

        "FPR3_AUDIT_FP":
            audit3[
                "FP"
            ],

        "FPR3_AUDIT_FN":
            audit3[
                "FN"
            ],

        "FPR3_AUDIT_Recall":
            audit3[
                "Recall"
            ],

        "FPR5_threshold":
            fpr5[
                "Threshold"
            ],

        "FPR5_DEV_TP":
            fpr5[
                "TP"
            ],

        "FPR5_DEV_FP":
            fpr5[
                "FP"
            ],

        "FPR5_DEV_FN":
            fpr5[
                "FN"
            ],

        "FPR5_DEV_Recall":
            fpr5[
                "Recall"
            ],

        "FPR5_AUDIT_TP":
            audit5[
                "TP"
            ],

        "FPR5_AUDIT_FP":
            audit5[
                "FP"
            ],

        "FPR5_AUDIT_FN":
            audit5[
                "FN"
            ],

        "FPR5_AUDIT_Recall":
            audit5[
                "Recall"
            ]
    })

scores = pd.DataFrame(
    score_rows
)

# ==============================================================
# 19. FPR3 / FPR5 모델 선택
#
# AUDIT은 정렬에 절대 사용 안 함
# ==============================================================

fpr3_ranked = (
    scores
    .sort_values(
        [
            "FPR3_DEV_Recall",
            "FPR3_DEV_FP",
            "DEV_AP",
            "DEV_MAE"
        ],
        ascending=[
            False,
            True,
            False,
            True
        ]
    )
    .reset_index(
        drop=True
    )
)

fpr5_ranked = (
    scores
    .sort_values(
        [
            "FPR5_DEV_Recall",
            "FPR5_DEV_FP",
            "DEV_AP",
            "DEV_MAE"
        ],
        ascending=[
            False,
            True,
            False,
            True
        ]
    )
    .reset_index(
        drop=True
    )
)

FPR3_MODEL = (
    fpr3_ranked.iloc[
        0
    ][
        "model"
    ]
)

FPR5_MODEL = (
    fpr5_ranked.iloc[
        0
    ][
        "model"
    ]
)

FPR3_THRESHOLD = float(
    fpr3_ranked.iloc[
        0
    ][
        "FPR3_threshold"
    ]
)

FPR5_THRESHOLD = float(
    fpr5_ranked.iloc[
        0
    ][
        "FPR5_threshold"
    ]
)

save_csv(
    scores,
    "B7_all_model_scores.csv"
)

save_csv(
    fpr3_ranked,
    "B7_FPR3_ranking.csv"
)

save_csv(
    fpr5_ranked,
    "B7_FPR5_ranking.csv"
)

print()
print(
    "⑦ FPR3 선택:",
    FPR3_MODEL,
    "| threshold:",
    FPR3_THRESHOLD
)

print(
    "⑧ FPR5 선택:",
    FPR5_MODEL,
    "| threshold:",
    FPR5_THRESHOLD
)

# ==============================================================
# 20. Selected model specs
# ==============================================================

SPEC_MAP = {
    spec[
        "name"
    ]:
        spec
    for spec in CANDIDATES
}

selected_models = list(
    dict.fromkeys([
        FPR3_MODEL,
        FPR5_MODEL
    ])
)

# ==============================================================
# 21. Validation full fit
# ==============================================================

validation_pred_map = {}

for model_name in selected_models:

    path = (
        VAL_DIR
        / (
            safe_name(
                model_name
            )
            + ".npz"
        )
    )

    loaded = (
        load_npz(
            path
        )
        if path.exists()
        else None
    )

    if (
        loaded is not None
        and "pred"
        in loaded
        and len(
            loaded[
                "pred"
            ]
        ) == 878
    ):

        validation_pred_map[
            model_name
        ] = loaded[
            "pred"
        ]

        print(
            "⑨ Validation SKIP:",
            model_name
        )

        continue

    print(
        "⑨ Validation FIT:",
        model_name
    )

    spec = SPEC_MAP[
        model_name
    ]

    features = FEATURE_SETS[
        spec[
            "feature_set"
        ]
    ]

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_i = (
        imputer.fit_transform(
            Xtr[
                features
            ]
        )
    )

    X_val_i = (
        imputer.transform(
            Xva[
                features
            ]
        )
    )

    model = make_model(
        spec
    )

    model.fit(
        X_train_i,
        yreg_tr
    )

    pred = model.predict(
        X_val_i
    )

    save_npz_atomic(
        path,
        pred=np.asarray(
            pred,
            dtype=float
        )
    )

    validation_pred_map[
        model_name
    ] = pred

# ==============================================================
# 22. Validation metrics
# ==============================================================

validation_rows = []

for mode, model_name, threshold in [

    (
        "B7_FPR3",
        FPR3_MODEL,
        FPR3_THRESHOLD
    ),

    (
        "B7_FPR5",
        FPR5_MODEL,
        FPR5_THRESHOLD
    )
]:

    pred = (
        validation_pred_map[
            model_name
        ]
    )

    metric = counts(
        ypeak_va.to_numpy(),
        pred >= threshold
    )

    ap = float(
        average_precision_score(
            ypeak_va,
            pred
        )
    )

    mae = float(
        mean_absolute_error(
            yreg_va,
            pred
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                yreg_va,
                pred
            )
        )
    )

    validation_rows.append({

        "mode":
            mode,

        "model":
            model_name,

        "threshold":
            threshold,

        **metric,

        "AP":
            ap,

        "MaxValue_MAE":
            mae,

        "MaxValue_RMSE":
            rmse
    })

validation_metrics = pd.DataFrame(
    validation_rows
)

save_csv(
    validation_metrics,
    "B7_validation_metrics.csv"
)

# ==============================================================
# 23. Validation row-level
# ==============================================================

validation = pd.DataFrame({

    "timestamp":
        Xva.index,

    "actual_max":
        yreg_va.to_numpy(
            dtype=float
        ),

    "actual_peak":
        ypeak_va.to_numpy(
            dtype=int
        ),

    "production_lag1":
        Xva[
            "prod_lag1"
        ].to_numpy()
})

for model_name in selected_models:

    validation[
        (
            "pred_"
            + safe_name(
                model_name
            )
        )
    ] = (
        validation_pred_map[
            model_name
        ]
    )

validation[
    "b7_fpr3_alert"
] = (
    validation_pred_map[
        FPR3_MODEL
    ]
    >= FPR3_THRESHOLD
).astype(int)

validation[
    "b7_fpr5_alert"
] = (
    validation_pred_map[
        FPR5_MODEL
    ]
    >= FPR5_THRESHOLD
).astype(int)

save_csv(
    validation,
    "B7_validation_predictions.csv"
)

# ==============================================================
# 24. Peak severity recall
# ==============================================================

severity_rows = []

severity_groups = [

    (
        "176-185",
        176,
        185
    ),

    (
        "186-200",
        186,
        200
    ),

    (
        "201+",
        201,
        np.inf
    )
]

for mode, alert_col in [

    (
        "FPR3",
        "b7_fpr3_alert"
    ),

    (
        "FPR5",
        "b7_fpr5_alert"
    )
]:

    for group, low, high in (
        severity_groups
    ):

        mask = (
            validation[
                "actual_max"
            ] >= low
        )

        if np.isfinite(
            high
        ):

            mask &= (
                validation[
                    "actual_max"
                ] <= high
            )

        sub = validation.loc[
            mask
        ]

        n = len(
            sub
        )

        hit = int(
            sub[
                alert_col
            ].sum()
        )

        severity_rows.append({

            "mode":
                mode,

            "severity":
                group,

            "N":
                n,

            "Detected":
                hit,

            "Recall":
                (
                    hit / n
                    if n > 0
                    else np.nan
                )
        })

severity = pd.DataFrame(
    severity_rows
)

save_csv(
    severity,
    "B7_peak_severity_recall.csv"
)

# ==============================================================
# 25. B2 missed 11
# ==============================================================

B2_MISSED = pd.to_datetime([

    "2021-07-02 14:00:00",
    "2021-07-06 14:00:00",
    "2021-07-06 16:00:00",
    "2021-07-09 14:00:00",
    "2021-07-12 11:00:00",
    "2021-07-12 14:00:00",
    "2021-07-26 14:00:00",
    "2021-07-26 18:00:00",
    "2021-07-26 19:00:00",
    "2021-07-28 16:00:00",
    "2021-07-30 16:00:00"
])

miss_compare = (
    validation.loc[
        validation[
            "timestamp"
        ].isin(
            B2_MISSED
        )
    ]
    .copy()
)

miss_compare[
    "margin_over_176"
] = (
    miss_compare[
        "actual_max"
    ]
    - 176
)

save_csv(
    miss_compare,
    "B7_B2_missed_comparison.csv"
)

# ==============================================================
# 26. Reference
# ==============================================================

reference = pd.DataFrame([

    {
        "model":
            "B2",

        "TP":
            145,

        "FP":
            21,

        "FN":
            11,

        "Precision":
            145 / 166,

        "Recall":
            145 / 156
    },

    {
        "model":
            "B3.1",

        "TP":
            150,

        "FP":
            39,

        "FN":
            6,

        "Precision":
            150 / 189,

        "Recall":
            150 / 156
    },

    {
        "model":
            "B5_FPR5",

        "TP":
            146,

        "FP":
            18,

        "FN":
            10,

        "Precision":
            146 / 164,

        "Recall":
            146 / 156
    },

    {
        "model":
            "B6_FPR5",

        "TP":
            151,

        "FP":
            47,

        "FN":
            5,

        "Precision":
            151 / 198,

        "Recall":
            151 / 156
    }
])

save_csv(
    reference,
    "B7_reference.csv"
)

# ==============================================================
# 27. Metadata
# ==============================================================

metadata = {

    "experiment":
        "KAMP_B7_peak_magnitude_regression",

    "version":
        B7_VERSION,

    "run_id":
        RUN_ID,

    "target":
        "hourly max of 15/30/45/60 minute measurements",

    "empirical_peak_threshold":
        PEAK_THRESHOLD,

    "candidate_count":
        len(
            CANDIDATES
        ),

    "dev_folds":
        DEV_FOLDS,

    "audit_folds":
        AUDIT_FOLDS,

    "FPR3_model":
        FPR3_MODEL,

    "FPR3_threshold":
        FPR3_THRESHOLD,

    "FPR5_model":
        FPR5_MODEL,

    "FPR5_threshold":
        FPR5_THRESHOLD,

    "validation_used_for_selection":
        False,

    "validation_previously_viewed":
        True,

    "future_information_used":
        False,

    "peak_is_empirical_not_contract_peak":
        True,

    "units_verified":
        False
}

save_json(
    metadata,
    "B7_metadata.json"
)

# ==============================================================
# 28. Source
# ==============================================================

try:

    source = (
        get_ipython()
        .history_manager
        .input_hist_raw[
            -1
        ]
    )

    if (
        "KAMP B-7"
        in source
        and len(
            source
        ) > 2000
    ):

        (
            OUT
            / "B7_source.py"
        ).write_text(
            source,
            encoding="utf-8"
        )

        (
            CK_ROOT
            / "B7_source.py"
        ).write_text(
            source,
            encoding="utf-8"
        )

except Exception:

    pass

# ==============================================================
# 29. ZIP
# ==============================================================

archive = Path(
    shutil.make_archive(
        str(
            OUT
        ),
        "zip",
        root_dir=OUT
    )
)

destination = (
    ROOT
    / (
        f"KAMP_B7_results_"
        f"{RUN_ID}.zip"
    )
)

shutil.copy2(
    archive,
    destination
)

manifest[
    "status"
] = "complete"

manifest[
    "completed_at"
] = datetime.now().isoformat()

manifest[
    "final_zip"
] = str(
    destination
)

with open(
    MANIFEST_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        ensure_ascii=False,
        indent=2
    )

# ==============================================================
# 30. ChatGPT 출력
# ==============================================================

print()
print(
    "========== ChatGPT에 보낼 결과 시작 =========="
)

print()
print("[기본 정보]")

print(
    "회귀 후보:",
    len(
        CANDIDATES
    )
)

print(
    "Train:",
    len(
        Xtr
    ),
    "| Validation:",
    len(
        Xva
    ),
    "| Validation peak:",
    int(
        ypeak_va.sum()
    )
)

print()
print(
    "[FPR3 DEV 상위 15]"
)

cols3 = [

    "model",
    "family",
    "feature_set",

    "DEV_AP",
    "AUDIT_AP",

    "DEV_MAE",

    "FPR3_DEV_TP",
    "FPR3_DEV_FP",
    "FPR3_DEV_FN",
    "FPR3_DEV_Recall",

    "FPR3_AUDIT_TP",
    "FPR3_AUDIT_FP",
    "FPR3_AUDIT_FN",
    "FPR3_AUDIT_Recall",

    "FPR3_threshold"
]

print(
    fpr3_ranked
    .head(
        15
    )[
        cols3
    ]
    .round(
        4
    )
    .to_string(
        index=False
    )
)

print()
print(
    "[FPR5 DEV 상위 15]"
)

cols5 = [

    "model",
    "family",
    "feature_set",

    "DEV_AP",
    "AUDIT_AP",

    "DEV_MAE",

    "FPR5_DEV_TP",
    "FPR5_DEV_FP",
    "FPR5_DEV_FN",
    "FPR5_DEV_Recall",

    "FPR5_AUDIT_TP",
    "FPR5_AUDIT_FP",
    "FPR5_AUDIT_FN",
    "FPR5_AUDIT_Recall",

    "FPR5_threshold"
]

print(
    fpr5_ranked
    .head(
        15
    )[
        cols5
    ]
    .round(
        4
    )
    .to_string(
        index=False
    )
)

print()
print("[선택]")

print(
    "FPR3:",
    FPR3_MODEL,
    "| threshold:",
    FPR3_THRESHOLD
)

print(
    "FPR5:",
    FPR5_MODEL,
    "| threshold:",
    FPR5_THRESHOLD
)

print()
print(
    "[Reference]"
)

print(
    reference
    .round(
        4
    )
    .to_string(
        index=False
    )
)

print()
print(
    "[B7 동일 878시간 / 동일 156 peak]"
)

print(
    validation_metrics
    .sort_values(
        [
            "FN",
            "FP"
        ]
    )
    .round(
        4
    )
    .to_string(
        index=False
    )
)

print()
print(
    "[Peak severity recall]"
)

print(
    severity
    .round(
        4
    )
    .to_string(
        index=False
    )
)

print()
print(
    "[B2 기존 FN 11개 중 B7 구조]"
)

print(
    "FPR3:",
    int(
        miss_compare[
            "b7_fpr3_alert"
        ].sum()
    ),
    "/11"
)

print(
    "FPR5:",
    int(
        miss_compare[
            "b7_fpr5_alert"
        ].sum()
    ),
    "/11"
)

print()
print(
    "[B2 missed 상세]"
)

show_cols = [

    "timestamp",
    "actual_max",
    "margin_over_176",
    "production_lag1",
    "b7_fpr3_alert",
    "b7_fpr5_alert"
]

for model in selected_models:

    col = (
        "pred_"
        + safe_name(
            model
        )
    )

    if col in (
        miss_compare.columns
    ):

        show_cols.append(
            col
        )

print(
    miss_compare[
        show_cols
    ]
    .round(
        3
    )
    .to_string(
        index=False
    )
)

print()
print("[판정 원칙]")

print(
    "- 모델/threshold 선택은 DEV folds 1~3에서만"
)

print(
    "- AUDIT folds 4~6은 선택에 사용하지 않음"
)

print(
    "- Validation 878시간은 선택에 사용하지 않음"
)

print(
    "- 회귀 target은 다음 1시간 actual_max"
)

print(
    "- classification과 다른 문제정의이므로 "
    "상호보완성이 있는지 확인하는 실험"
)

print()
print("[주의]")

print(
    "- 176은 empirical peak threshold"
)

print(
    "- 단위와 전력회사 계약피크 여부는 확인되지 않음"
)

print(
    "- 현재/미래 t 값은 feature로 사용하지 않음"
)

print()
print(
    "실행 시간(분):",
    round(
        (
            time.time()
            - T0
        )
        / 60,
        2
    )
)

print(
    "Drive ZIP:",
    destination
)

print(
    "Checkpoint:",
    CK_ROOT
)

print(
    "========== ChatGPT에 보낼 결과 끝 =========="
)