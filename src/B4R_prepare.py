# ==============================================================
# KAMP B-4R
# Broad Peak Warning Search
# ★ 자동 저장 + 런타임 종료 후 자동 재개 버전
#
# 중요
# - peak threshold = 176 고정
# - validation peak = 156 assert
# - 68개 모델 × 6 folds
# - fold 하나 끝날 때마다 Google Drive 즉시 저장
# - Colab 종료 시 같은 셀 다시 실행 -> 저장된 fold 자동 skip
# - 최종 ensemble + operating point 비교
#
# Selection:
#   DEV   = folds 1~3
#   AUDIT = folds 4~6
#
# Operating points:
#   F2_MAX
#   FPR3
#   FPR5
#   REC95
# ==============================================================

import os
import io
import sys
import json
import time
import shutil
import zipfile
import subprocess
import importlib.util
import warnings
import itertools
import re

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ==============================================================
# 0. 기본 설정
# ==============================================================

T0 = time.time()

SEED = 42
np.random.seed(SEED)

EXPERIMENT_VERSION = "B4R_v1"

PEAK_THRESHOLD = 176.0

DEV_FOLDS = [1, 2, 3]
AUDIT_FOLDS = [4, 5, 6]

# ==============================================================

# Only I/O wiring is replaced; all feature and selected model definitions below
# are derived from the SHA-verified original B4R source.
ROOT = Path(os.environ["KAMP_SUBMISSION_ROOT"])
DATA = Path(os.environ["KAMP_RAW_ZIP"])
CK_ROOT = Path(os.environ["KAMP_STAGE_WORK"]) / "ck"
OUT = Path(os.environ["KAMP_STAGE_WORK"]) / "out"
CK_ROOT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST_FILE = CK_ROOT / "manifest.json"
manifest = {"status": "submission_selected_only", "run_id": "submission"}
RUN_ID = "submission"

OOF_DIR = CK_ROOT / "oof_fold_checkpoints"
FIXED_DIR = CK_ROOT / "validation_fixed"
EXPANDING_DIR = CK_ROOT / "validation_expanding"
SUMMARY_DIR = CK_ROOT / "summaries"
for folder in (OOF_DIR, FIXED_DIR, EXPANDING_DIR, SUMMARY_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# 4. 라이브러리
# ==============================================================

required_packages = [
    "lightgbm",
    "catboost"
]

for package in required_packages:

    if importlib.util.find_spec(
        package
    ) is None:

        print(
            f"{package} 설치 중..."
        )

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            package
        ])

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

print(
    "② ML 라이브러리 준비 완료"
)

# ==============================================================
# 5. Utility
# ==============================================================

def safe_name(text):

    return re.sub(
        r"[^A-Za-z0-9_\-\.]+",
        "_",
        str(text)
    )


def save_json_atomic(
    obj,
    path
):

    path = Path(path)

    temp = path.with_suffix(
        ".tmp"
    )

    with open(
        temp,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(
        temp,
        path
    )


def update_manifest(
    **kwargs
):

    global manifest

    manifest.update(
        kwargs
    )

    manifest[
        "last_update"
    ] = datetime.now().isoformat()

    save_json_atomic(
        manifest,
        MANIFEST_FILE
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

        if len(names) != 1:

            raise ValueError(
                f"CSV가 정확히 1개가 아닙니다: {names}"
            )

        return pd.read_csv(
            io.BytesIO(
                z.read(
                    names[0]
                )
            ),
            encoding="utf-8-sig"
        )


def save_df(
    df,
    filename
):

    local_path = (
        OUT
        / filename
    )

    drive_path = (
        SUMMARY_DIR
        / filename
    )

    df.to_csv(
        local_path,
        index=False,
        encoding="utf-8-sig"
    )

    df.to_csv(
        drive_path,
        index=False,
        encoding="utf-8-sig"
    )


def save_npz_atomic(
    path,
    **arrays
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp = path.parent / (
        path.stem
        + ".tmp.npz"
    )

    np.savez_compressed(
        temp,
        **arrays
    )

    os.replace(
        temp,
        path
    )


def load_npz_safe(
    path
):

    try:

        data = np.load(
            path,
            allow_pickle=False
        )

        return {
            key:
                data[key]
            for key in data.files
        }

    except Exception:

        return None


# ==============================================================
# 6. Metrics
# ==============================================================

def cls_metrics(
    y_true,
    prob,
    threshold
):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    prob = np.asarray(
        prob,
        dtype=float
    )

    pred = (
        prob
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            pred,
            labels=[0, 1]
        ).ravel()
    )

    precision = precision_score(
        y_true,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        pred,
        zero_division=0
    )

    beta = 2.0

    denominator = (
        beta**2
        * precision
        + recall
    )

    if denominator > 0:

        f2 = (
            (
                1
                + beta**2
            )
            * precision
            * recall
            / denominator
        )

    else:

        f2 = 0.0

    negative_n = (
        tn + fp
    )

    if negative_n > 0:

        fpr = (
            fp
            / negative_n
        )

    else:

        fpr = 0.0

    if len(
        np.unique(
            y_true
        )
    ) > 1:

        ap = float(
            average_precision_score(
                y_true,
                prob
            )
        )

    else:

        ap = np.nan

    return {

        "N":
            int(
                len(
                    y_true
                )
            ),

        "Positive_N":
            int(
                y_true.sum()
            ),

        "TP":
            int(tp),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TN":
            int(tn),

        "Precision":
            float(
                precision
            ),

        "Recall":
            float(
                recall
            ),

        "FPR":
            float(
                fpr
            ),

        "F1":
            float(
                f1
            ),

        "F2":
            float(
                f2
            ),

        "AP":
            float(
                ap
            ),

        "Threshold":
            float(
                threshold
            )
    }


def threshold_table(
    y,
    prob
):

    y = np.asarray(
        y,
        dtype=int
    )

    prob = np.asarray(
        prob,
        dtype=float
    )

    thresholds = np.unique(
        np.concatenate([
            [0.0],
            np.unique(
                prob
            ),
            [1.0]
        ])
    )

    rows = []

    for threshold in thresholds:

        rows.append(
            cls_metrics(
                y,
                prob,
                threshold
            )
        )

    return pd.DataFrame(
        rows
    )


def choose_operating_points(
    y,
    prob
):

    table = threshold_table(
        y,
        prob
    )

    # F2
    f2_row = (
        table
        .sort_values(
            [
                "F2",
                "Recall",
                "Precision",
                "Threshold"
            ],
            ascending=[
                False,
                False,
                False,
                False
            ]
        )
        .iloc[0]
    )

    def best_fpr(
        maximum_fpr
    ):

        eligible = table.loc[
            table[
                "FPR"
            ]
            <= maximum_fpr
        ].copy()

        if not len(
            eligible
        ):
            return f2_row

        return (
            eligible
            .sort_values(
                [
                    "Recall",
                    "Precision",
                    "F2",
                    "Threshold"
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False
                ]
            )
            .iloc[0]
        )

    fpr3_row = best_fpr(
        0.03
    )

    fpr5_row = best_fpr(
        0.05
    )

    rec95 = table.loc[
        table[
            "Recall"
        ] >= 0.95
    ].copy()

    if len(rec95):

        rec95_row = (
            rec95
            .sort_values(
                [
                    "Precision",
                    "FPR",
                    "F2",
                    "Threshold"
                ],
                ascending=[
                    False,
                    True,
                    False,
                    False
                ]
            )
            .iloc[0]
        )

    else:

        rec95_row = (
            table
            .sort_values(
                [
                    "Recall",
                    "Precision",
                    "FPR"
                ],
                ascending=[
                    False,
                    False,
                    True
                ]
            )
            .iloc[0]
        )

    output = {

        "F2_MAX":
            float(
                f2_row[
                    "Threshold"
                ]
            ),

        "FPR3":
            float(
                fpr3_row[
                    "Threshold"
                ]
            ),

        "FPR5":
            float(
                fpr5_row[
                    "Threshold"
                ]
            ),

        "REC95":
            float(
                rec95_row[
                    "Threshold"
                ]
            )
    }

    return (
        output,
        table
    )

# ==============================================================
# 7. 원본 데이터
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

    + pd.to_timedelta(
        clean[
            "시간"
        ],
        unit="h"
    )
)

if clean[
    "timestamp"
].duplicated().any():

    raise ValueError(
        "timestamp 중복"
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

        hourly[col] = (
            pd.to_numeric(
                hourly[col],
                errors="coerce"
            )
        )

target_mean = (
    hourly[
        "평균"
    ]
)

production = (
    hourly[
        "생산량"
    ]
)

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

print()
print(
    "③ 원본:",
    raw.shape,
    "| 오류 시간:",
    len(invalid),
    "| peak:",
    PEAK_THRESHOLD
)

# ==============================================================
# 8. 기존 5856 timestamp 재현
# ==============================================================

X17 = pd.DataFrame(
    index=idx
)

X17[
    "hour_sin"
] = np.sin(
    2
    * np.pi
    * idx.hour
    / 24
)

X17[
    "hour_cos"
] = np.cos(
    2
    * np.pi
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
    idx.dayofweek
    >= 5
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
    ] = (
        target_mean.shift(
            lag
        )
    )

X17[
    "mean_rolling3"
] = (
    target_mean
    .shift(1)
    .rolling(3)
    .mean()
)

X17[
    "mean_rolling24"
] = (
    target_mean
    .shift(1)
    .rolling(24)
    .mean()
)

X17[
    "production_lag1"
] = (
    production.shift(
        1
    )
)

X17[
    "production_lag24"
] = (
    production.shift(
        24
    )
)

X17[
    "max_lag1"
] = (
    hour_max.shift(
        1
    )
)

X17[
    "max_lag24"
] = (
    hour_max.shift(
        24
    )
)

X17[
    "max_rolling3"
] = (
    hour_max
    .shift(1)
    .rolling(3)
    .mean()
)

base_mask = (

    X17.notna().all(
        axis=1
    )

    & target_mean.notna()

    & hour_max.notna()
)

base_index = (
    idx[
        base_mask
    ]
)

if len(
    base_index
) != 5856:

    raise ValueError(
        f"5856 불일치: {len(base_index)}"
    )

cut1 = int(
    len(base_index)
    * 0.70
)

cut2 = int(
    len(base_index)
    * 0.85
)

# ==============================================================
# 9. zero streak
# ==============================================================

zero_run = np.zeros(
    len(production),
    dtype=int
)

for k, value in enumerate(
    production.to_numpy()
):

    if (
        np.isfinite(
            value
        )
        and value == 0
    ):

        zero_run[k] = (
            1
            +
            (
                zero_run[
                    k - 1
                ]
                if k > 0
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
# 10. Feature pool
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
    2
    * np.pi
    * idx.hour
    / 24
)

F[
    "hour_cos"
] = np.cos(
    2
    * np.pi
    * idx.hour
    / 24
)

F[
    "weekday"
] = idx.dayofweek

F[
    "dow_sin"
] = np.sin(
    2
    * np.pi
    * idx.dayofweek
    / 7
)

F[
    "dow_cos"
] = np.cos(
    2
    * np.pi
    * idx.dayofweek
    / 7
)

F[
    "month"
] = idx.month

F[
    "is_weekend"
] = (
    idx.dayofweek
    >= 5
).astype(int)

# Mean history
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

# Max history
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

# Production history
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

# Changes
F[
    "mean_change1"
] = (
    target_mean.shift(1)
    - target_mean.shift(2)
)

F[
    "max_change1"
] = (
    hour_max.shift(1)
    - hour_max.shift(2)
)

F[
    "prod_change1"
] = (
    production.shift(1)
    - production.shift(2)
)

# Zero state
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
    .rolling(24)
    .sum()
)

F[
    "zero_hours_last72"
] = (
    production
    .shift(1)
    .eq(0)
    .rolling(72)
    .sum()
)

# Historical peak
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
] = (
    historical_peak.shift(
        1
    )
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

# Lagged weather
for weather in [
    "기온",
    "풍속",
    "습도",
    "강수량"
]:

    if weather in hourly.columns:

        F[
            f"{weather}_lag1"
        ] = (
            hourly[
                weather
            ]
            .shift(1)
        )

# ==============================================================
# 11. Feature sets
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

print(
    "④ Feature 수:",
    {
        k: len(v)
        for k, v
        in FEATURE_SETS.items()
    }
)

# ==============================================================
# 12. Dataset
# ==============================================================

Xall = F.loc[
    base_index
].copy()

maxall = hour_max.loc[
    base_index
]

yall = (
    maxall
    .ge(
        PEAK_THRESHOLD
    )
    .astype(int)
)

Xtr = Xall.iloc[
    :cut1
]

ytr = yall.iloc[
    :cut1
]

maxtr = maxall.iloc[
    :cut1
]

Xva = Xall.iloc[
    cut1:cut2
]

yva = yall.iloc[
    cut1:cut2
]

maxva = maxall.iloc[
    cut1:cut2
]

if len(
    Xtr
) != 4099:
    raise ValueError(
        "train != 4099"
    )

if len(
    Xva
) != 878:
    raise ValueError(
        "validation != 878"
    )

if int(
    yva.sum()
) != 156:

    raise ValueError(
        f"validation peak != 156: {int(yva.sum())}"
    )

print(
    "⑤ B2와 동일한 156개 peak 재현 성공"
)

# ==============================================================
# 13. CV folds
# ==============================================================

folds = list(
    TimeSeriesSplit(
        n_splits=6,
        test_size=336
    ).split(
        Xtr
    )
)

coverage_rows = []

for fold_no, (
    train_idx,
    eval_idx
) in enumerate(
    folds,
    start=1
):

    coverage_rows.append({

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

        "train_peak":
            int(
                ytr.iloc[
                    train_idx
                ].sum()
            ),

        "eval_peak":
            int(
                ytr.iloc[
                    eval_idx
                ].sum()
            )
    })

coverage = pd.DataFrame(
    coverage_rows
)

save_df(
    coverage,
    "B4R_cv_coverage.csv"
)

print()
print(
    "⑥ CV coverage"
)

print(
    coverage.to_string(
        index=False
    )
)

# ==============================================================
# 14. 68개 후보 생성
# ==============================================================

CANDIDATES = []

# LightGBM 36
for feature_set in [
    "CORE",
    "CORE_PLUS",
    "HISTORY"
]:

    for num_leaves in [
        7,
        15,
        31
    ]:

        for min_child in [
            20,
            40
        ]:

            for pos_weight in [
                1.0,
                1.5
            ]:

                CANDIDATES.append({

                    "family":
                        "LGB",

                    "name":
                        (
                            f"LGB_{feature_set}"
                            f"_leaf{num_leaves}"
                            f"_mc{min_child}"
                            f"_w{str(pos_weight).replace('.', 'p')}"
                        ),

                    "feature_set":
                        feature_set,

                    "num_leaves":
                        num_leaves,

                    "min_child":
                        min_child,

                    "pos_weight":
                        pos_weight
                })

# XGBoost 12
for feature_set in [
    "CORE",
    "CORE_PLUS"
]:

    for depth in [
        2,
        3,
        4
    ]:

        for pos_weight in [
            1.0,
            1.5
        ]:

            CANDIDATES.append({

                "family":
                    "XGB",

                "name":
                    (
                        f"XGB_{feature_set}"
                        f"_d{depth}"
                        f"_w{str(pos_weight).replace('.', 'p')}"
                    ),

                "feature_set":
                    feature_set,

                "depth":
                    depth,

                "pos_weight":
                    pos_weight
            })

# CatBoost 8
for feature_set in [
    "CORE",
    "CORE_PLUS"
]:

    for depth in [
        4,
        6
    ]:

        for pos_weight in [
            1.0,
            1.5
        ]:

            CANDIDATES.append({

                "family":
                    "CAT",

                "name":
                    (
                        f"CAT_{feature_set}"
                        f"_d{depth}"
                        f"_w{str(pos_weight).replace('.', 'p')}"
                    ),

                "feature_set":
                    feature_set,

                "depth":
                    depth,

                "pos_weight":
                    pos_weight
            })

# ExtraTrees 12
for feature_set in [
    "CORE",
    "CORE_PLUS"
]:

    for leaf in [
        1,
        2,
        4
    ]:

        for class_weight in [
            None,
            "balanced"
        ]:

            weight_name = (
                "none"
                if class_weight is None
                else "bal"
            )

            CANDIDATES.append({

                "family":
                    "ET",

                "name":
                    (
                        f"ET_{feature_set}"
                        f"_leaf{leaf}"
                        f"_{weight_name}"
                    ),

                "feature_set":
                    feature_set,

                "leaf":
                    leaf,

                "class_weight":
                    class_weight
            })

print()
print(
    "⑦ 전체 후보:",
    len(
        CANDIDATES
    )
)

if len(
    CANDIDATES
) != 68:

    raise ValueError(
        f"후보 수가 68이 아님: {len(CANDIDATES)}"
    )

# Candidate manifest
candidate_manifest = pd.DataFrame([
    {
        "name":
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

        "config":
            json.dumps(
                spec,
                ensure_ascii=False
            )
    }
    for spec in CANDIDATES
])

save_df(
    candidate_manifest,
    "B4R_candidate_manifest.csv"
)

# ==============================================================
# 15. Model factory
# ==============================================================

def make_model(
    spec
):

    family = spec[
        "family"
    ]

    if family == "LGB":

        return LGBMClassifier(

            objective="binary",

            n_estimators=550,

            learning_rate=0.025,

            num_leaves=
                spec[
                    "num_leaves"
                ],

            min_child_samples=
                spec[
                    "min_child"
                ],

            subsample=0.90,

            colsample_bytree=0.90,

            reg_lambda=1.0,

            scale_pos_weight=
                spec[
                    "pos_weight"
                ],

            random_state=SEED,

            n_jobs=-1,

            verbosity=-1
        )

    if family == "XGB":

        return XGBClassifier(

            n_estimators=550,

            learning_rate=0.025,

            max_depth=
                spec[
                    "depth"
                ],

            min_child_weight=3,

            subsample=0.90,

            colsample_bytree=0.90,

            reg_lambda=1.0,

            scale_pos_weight=
                spec[
                    "pos_weight"
                ],

            objective="binary:logistic",

            eval_metric="logloss",

            random_state=SEED,

            n_jobs=-1
        )

    if family == "CAT":

        return CatBoostClassifier(

            iterations=700,

            depth=
                spec[
                    "depth"
                ],

            learning_rate=0.025,

            loss_function="Logloss",

            eval_metric="AUC",

            l2_leaf_reg=5,

            class_weights=[
                1.0,
                spec[
                    "pos_weight"
                ]
            ],

            random_seed=SEED,

            verbose=False,

            allow_writing_files=False,

            thread_count=-1
        )

    if family == "ET":

        return ExtraTreesClassifier(

            n_estimators=500,

            min_samples_leaf=
                spec[
                    "leaf"
                ],

            max_features="sqrt",

            class_weight=
                spec[
                    "class_weight"
                ],

            random_state=SEED,

            n_jobs=-1
        )

    raise ValueError(
        spec
    )

# ==============================================================
# 16. Fold checkpoint helpers
# ==============================================================

def fold_checkpoint_path(
    model_name,
    fold_no
):

    model_dir = (
        OOF_DIR
        / safe_name(
            model_name
        )
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        model_dir
        / f"fold_{fold_no}.npz"
    )


def valid_fold_checkpoint(
    path,
    eval_idx
):

    if not Path(path).exists():
        return False

    loaded = load_npz_safe(
        path
    )

    if loaded is None:
        return False

    if (
        "eval_idx"
        not in loaded
        or "prob"
        not in loaded
    ):
        return False

    saved_idx = (
        loaded[
            "eval_idx"
        ].astype(int)
    )

    expected_idx = np.asarray(
        eval_idx,
        dtype=int
    )

    if not np.array_equal(
        saved_idx,
        expected_idx
    ):
        return False

    prob = loaded[
        "prob"
    ]

    if (
        len(prob)
        != len(
            expected_idx
        )
    ):
        return False

    if not np.isfinite(
        prob
    ).all():
        return False

    return True


# ==============================================================
