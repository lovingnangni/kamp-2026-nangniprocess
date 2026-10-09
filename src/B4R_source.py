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
# 1. Google Drive
# ==============================================================

from google.colab import drive

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")

ROOT = Path(
    "/content/drive/MyDrive/KAMP_2026"
)

DATA = (
    ROOT
    / "KAMP_5_data.zip"
)

if not DATA.exists():
    raise FileNotFoundError(
        f"원본 데이터 없음: {DATA}"
    )

# ==============================================================
# 2. 영구 체크포인트
#
# ★ 여기가 런타임이 꺼져도 살아남는 Drive 영역
# ==============================================================

CK_ROOT = (
    ROOT
    / "KAMP_B4R_RESUME_CHECKPOINT"
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
            "experiment_version"
        )
        != EXPERIMENT_VERSION
    ):
        raise RuntimeError(
            """
기존 B4R 체크포인트가 다른 코드 버전입니다.

아래 폴더를 삭제한 뒤 다시 실행해야 합니다.

KAMP_B4R_RESUME_CHECKPOINT
"""
        )

    RUN_ID = manifest[
        "run_id"
    ]

    print(
        "♻️ 기존 B4R 체크포인트 발견"
    )

    print(
        "RUN_ID:",
        RUN_ID
    )

else:

    RUN_ID = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    manifest = {

        "experiment_version":
            EXPERIMENT_VERSION,

        "run_id":
            RUN_ID,

        "status":
            "running",

        "created_at":
            datetime.now().isoformat(),

        "last_update":
            datetime.now().isoformat(),

        "completed_fold_tasks":
            0
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
        "🆕 새 B4R 실행 생성"
    )

    print(
        "RUN_ID:",
        RUN_ID
    )

if (
    manifest.get("status")
    == "complete"
):

    final_zip = manifest.get(
        "final_zip"
    )

    if (
        final_zip
        and Path(
            final_zip
        ).exists()
    ):

        print()
        print(
            "✅ 이 B4R 실행은 이미 완료되어 있습니다."
        )

        print(
            "결과 ZIP:",
            final_zip
        )

        raise SystemExit

# ==============================================================
# 3. 런타임 출력 폴더
# ==============================================================

OUT = Path(
    "/content/KAMP_B4R_results"
)

if OUT.exists():
    shutil.rmtree(OUT)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

OOF_DIR = (
    CK_ROOT
    / "oof_fold_checkpoints"
)

OOF_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIXED_DIR = (
    CK_ROOT
    / "validation_fixed"
)

FIXED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EXPANDING_DIR = (
    CK_ROOT
    / "validation_expanding"
)

EXPANDING_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_DIR = (
    CK_ROOT
    / "summaries"
)

SUMMARY_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print()
print(
    "① Google Drive 연결 완료"
)

print(
    "ROOT:",
    ROOT
)

print(
    "체크포인트:",
    CK_ROOT
)

# ==============================================================
# 4. 라이브러리
# ==============================================================

required_packages = [
    "lightgbm",
    "xgboost",
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
from xgboost import XGBClassifier
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
# 17. Model × fold search
#
# ★ 매 fold 종료 즉시 Drive 저장
# ==============================================================

TOTAL_FOLD_TASKS = (
    len(
        CANDIDATES
    )
    * 6
)

completed_before = 0

for spec in CANDIDATES:

    for fold_no, (
        train_idx,
        eval_idx
    ) in enumerate(
        folds,
        start=1
    ):

        path = fold_checkpoint_path(
            spec[
                "name"
            ],
            fold_no
        )

        if valid_fold_checkpoint(
            path,
            eval_idx
        ):
            completed_before += 1

print()
print(
    f"♻️ 이미 저장된 fold: "
    f"{completed_before}/{TOTAL_FOLD_TASKS}"
)

new_durations = []

completed = completed_before

for model_no, spec in enumerate(
    CANDIDATES,
    start=1
):

    model_name = spec[
        "name"
    ]

    print()
    print(
        f"⑧ Model "
        f"{model_no}/{len(CANDIDATES)}: "
        f"{model_name}"
    )

    for fold_no, (
        train_idx,
        eval_idx
    ) in enumerate(
        folds,
        start=1
    ):

        ck_path = fold_checkpoint_path(
            model_name,
            fold_no
        )

        if valid_fold_checkpoint(
            ck_path,
            eval_idx
        ):

            loaded = load_npz_safe(
                ck_path
            )

            yy = (
                ytr.iloc[
                    eval_idx
                ]
                .to_numpy(
                    dtype=int
                )
            )

            prob = loaded[
                "prob"
            ]

            if (
                yy.sum() > 0
                and yy.sum() < len(yy)
            ):

                ap = (
                    average_precision_score(
                        yy,
                        prob
                    )
                )

                print(
                    f"   fold {fold_no}: "
                    f"SKIP(saved) | "
                    f"AP={ap:.3f}"
                )

            else:

                print(
                    f"   fold {fold_no}: "
                    f"SKIP(saved)"
                )

            continue

        started = time.time()

        features = FEATURE_SETS[
            spec[
                "feature_set"
            ]
        ]

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
            ytr.iloc[
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

        prob = (
            model.predict_proba(
                X_eval_i
            )[
                :,
                1
            ]
        )

        yy = (
            ytr.iloc[
                eval_idx
            ]
            .to_numpy(
                dtype=int
            )
        )

        save_npz_atomic(
            ck_path,

            eval_idx=np.asarray(
                eval_idx,
                dtype=int
            ),

            prob=np.asarray(
                prob,
                dtype=float
            )
        )

        elapsed = (
            time.time()
            - started
        )

        new_durations.append(
            elapsed
        )

        completed += 1

        update_manifest(
            completed_fold_tasks=
                completed
        )

        if (
            yy.sum() > 0
            and yy.sum() < len(yy)
        ):

            ap = float(
                average_precision_score(
                    yy,
                    prob
                )
            )

            ap_text = (
                f"{ap:.3f}"
            )

        else:

            ap_text = "NA"

        if len(
            new_durations
        ) > 0:

            avg_sec = np.mean(
                new_durations
            )

            remaining = (
                TOTAL_FOLD_TASKS
                - completed
            )

            eta_min = (
                remaining
                * avg_sec
                / 60
            )

        else:

            eta_min = np.nan

        print(
            f"   fold {fold_no}: "
            f"AP={ap_text} | "
            f"{elapsed:.1f}초 | "
            f"전체 {completed}/"
            f"{TOTAL_FOLD_TASKS} | "
            f"남은 예상 {eta_min:.1f}분"
        )

# ==============================================================
# 18. 모든 OOF checkpoint 합치기
# ==============================================================

print()
print(
    "⑨ 모든 fold 계산 완료"
)

oof_parts = []

for spec in CANDIDATES:

    model_name = (
        spec[
            "name"
        ]
    )

    for fold_no, (
        train_idx,
        eval_idx
    ) in enumerate(
        folds,
        start=1
    ):

        path = fold_checkpoint_path(
            model_name,
            fold_no
        )

        if not valid_fold_checkpoint(
            path,
            eval_idx
        ):

            raise RuntimeError(
                f"누락 checkpoint: "
                f"{model_name}, fold {fold_no}"
            )

        loaded = load_npz_safe(
            path
        )

        oof_parts.append(
            pd.DataFrame({

                "model":
                    model_name,

                "family":
                    spec[
                        "family"
                    ],

                "fold":
                    fold_no,

                "timestamp":
                    Xtr.index[
                        eval_idx
                    ],

                "actual_peak":
                    ytr.iloc[
                        eval_idx
                    ].to_numpy(
                        dtype=int
                    ),

                "actual_max":
                    maxtr.iloc[
                        eval_idx
                    ].to_numpy(
                        dtype=float
                    ),

                "prob":
                    loaded[
                        "prob"
                    ]
            })
        )

oof = pd.concat(
    oof_parts,
    ignore_index=True
)

save_df(
    oof,
    "B4R_all_oof_predictions.csv"
)

# ==============================================================
# 19. Single model DEV/AUDIT ranking
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
    ]

    audit = sub.loc[
        sub[
            "fold"
        ].isin(
            AUDIT_FOLDS
        )
    ]

    dev_ap = float(
        average_precision_score(
            dev[
                "actual_peak"
            ],
            dev[
                "prob"
            ]
        )
    )

    audit_ap = float(
        average_precision_score(
            audit[
                "actual_peak"
            ],
            audit[
                "prob"
            ]
        )
    )

    fold_aps = {}

    for fold_no in range(
        1,
        7
    ):

        part = sub.loc[
            sub[
                "fold"
            ] == fold_no
        ]

        yy = part[
            "actual_peak"
        ].to_numpy()

        if (
            yy.sum() > 0
            and yy.sum() < len(yy)
        ):

            fold_aps[
                fold_no
            ] = float(
                average_precision_score(
                    yy,
                    part[
                        "prob"
                    ]
                )
            )

        else:

            fold_aps[
                fold_no
            ] = np.nan

    dev_fold_values = [
        fold_aps[
            f
        ]
        for f in DEV_FOLDS
        if np.isfinite(
            fold_aps[
                f
            ]
        )
    ]

    dev_mean_fold_ap = (
        float(
            np.mean(
                dev_fold_values
            )
        )
        if len(
            dev_fold_values
        )
        else np.nan
    )

    spec = next(
        x
        for x in CANDIDATES
        if x[
            "name"
        ] == model_name
    )

    score_rows.append({

        "model":
            model_name,

        "family":
            spec[
                "family"
            ],

        "feature_set":
            spec[
                "feature_set"
            ],

        "DEV_AP":
            dev_ap,

        "DEV_MeanFold_AP":
            dev_mean_fold_ap,

        "AUDIT_AP":
            audit_ap,

        "Fold1_AP":
            fold_aps[1],

        "Fold2_AP":
            fold_aps[2],

        "Fold3_AP":
            fold_aps[3],

        "Fold4_AP":
            fold_aps[4],

        "Fold5_AP":
            fold_aps[5],

        "Fold6_AP":
            fold_aps[6]
    })

scores = (
    pd.DataFrame(
        score_rows
    )
    .sort_values(
        [
            "DEV_AP",
            "DEV_MeanFold_AP"
        ],
        ascending=[
            False,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)

save_df(
    scores,
    "B4R_single_model_scores.csv"
)

print()
print(
    "⑩ DEV 상위 단일 모델"
)

print(
    scores
    .head(15)
    [
        [
            "model",
            "family",
            "feature_set",
            "DEV_AP",
            "AUDIT_AP"
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)

# ==============================================================
# 20. Top 8 ensemble search
# ==============================================================

TOP_MODELS = (
    scores
    .head(8)[
        "model"
    ]
    .tolist()
)

base_model = (
    TOP_MODELS[
        0
    ]
)

base_oof = (
    oof.loc[
        oof[
            "model"
        ] == base_model
    ]
    .sort_values(
        [
            "fold",
            "timestamp"
        ]
    )
    .reset_index(
        drop=True
    )
)

prob_map = {}

for model_name in TOP_MODELS:

    temp = (
        oof.loc[
            oof[
                "model"
            ] == model_name
        ]
        .sort_values(
            [
                "fold",
                "timestamp"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if not np.array_equal(
        temp[
            "timestamp"
        ].to_numpy(),
        base_oof[
            "timestamp"
        ].to_numpy()
    ):

        raise RuntimeError(
            "OOF timestamp mismatch"
        )

    prob_map[
        model_name
    ] = temp[
        "prob"
    ].to_numpy(
        dtype=float
    )

ensemble_candidates = []

# Singles
for model_name in TOP_MODELS:

    ensemble_candidates.append({

        "name":
            model_name,

        "components":
            [
                model_name
            ],

        "weights":
            [
                1.0
            ],

        "prob":
            prob_map[
                model_name
            ]
    })

# Pair blends
for left, right in itertools.combinations(
    TOP_MODELS,
    2
):

    for w_left in [
        0.20,
        0.35,
        0.50,
        0.65,
        0.80
    ]:

        ensemble_candidates.append({

            "name":
                (
                    f"BLEND__"
                    f"{left}__"
                    f"{right}__"
                    f"{w_left:.2f}"
                ),

            "components":
                [
                    left,
                    right
                ],

            "weights":
                [
                    w_left,
                    1.0 - w_left
                ],

            "prob":
                (
                    w_left
                    * prob_map[
                        left
                    ]
                    +
                    (
                        1.0
                        - w_left
                    )
                    * prob_map[
                        right
                    ]
                )
        })

# Top 3 equal
if len(
    TOP_MODELS
) >= 3:

    top3 = TOP_MODELS[
        :3
    ]

    ensemble_candidates.append({

        "name":
            "BLEND_TOP3_EQUAL",

        "components":
            top3,

        "weights":
            [
                1/3,
                1/3,
                1/3
            ],

        "prob":
            np.mean(
                np.column_stack(
                    [
                        prob_map[
                            x
                        ]
                        for x in top3
                    ]
                ),
                axis=1
            )
    })

# Top 5 equal
if len(
    TOP_MODELS
) >= 5:

    top5 = TOP_MODELS[
        :5
    ]

    ensemble_candidates.append({

        "name":
            "BLEND_TOP5_EQUAL",

        "components":
            top5,

        "weights":
            [
                0.2
            ] * 5,

        "prob":
            np.mean(
                np.column_stack(
                    [
                        prob_map[
                            x
                        ]
                        for x in top5
                    ]
                ),
                axis=1
            )
    })

dev_mask = (
    base_oof[
        "fold"
    ].isin(
        DEV_FOLDS
    )
).to_numpy()

audit_mask = (
    base_oof[
        "fold"
    ].isin(
        AUDIT_FOLDS
    )
).to_numpy()

ensemble_rows = []

for item in ensemble_candidates:

    dev_ap = float(
        average_precision_score(
            base_oof.loc[
                dev_mask,
                "actual_peak"
            ],
            item[
                "prob"
            ][
                dev_mask
            ]
        )
    )

    audit_ap = float(
        average_precision_score(
            base_oof.loc[
                audit_mask,
                "actual_peak"
            ],
            item[
                "prob"
            ][
                audit_mask
            ]
        )
    )

    ensemble_rows.append({

        "model":
            item[
                "name"
            ],

        "DEV_AP":
            dev_ap,

        "AUDIT_AP":
            audit_ap,

        "components":
            "|".join(
                item[
                    "components"
                ]
            ),

        "weights":
            "|".join(
                [
                    f"{w:.4f}"
                    for w
                    in item[
                        "weights"
                    ]
                ]
            )
    })

ensemble_scores = (
    pd.DataFrame(
        ensemble_rows
    )
    .sort_values(
        [
            "DEV_AP"
        ],
        ascending=[
            False
        ]
    )
    .reset_index(
        drop=True
    )
)

save_df(
    ensemble_scores,
    "B4R_ensemble_scores.csv"
)

SELECTED_NAME = str(
    ensemble_scores.iloc[
        0
    ][
        "model"
    ]
)

selected_item = next(
    item
    for item
    in ensemble_candidates
    if item[
        "name"
    ] == SELECTED_NAME
)

selected_components = (
    selected_item[
        "components"
    ]
)

selected_weights = (
    selected_item[
        "weights"
    ]
)

selected_oof_prob = (
    selected_item[
        "prob"
    ]
)

print()
print(
    "⑪ DEV 최종 선택:"
)

print(
    SELECTED_NAME
)

print(
    "components:",
    selected_components
)

print(
    "weights:",
    selected_weights
)

selection_manifest = {

    "selected_name":
        SELECTED_NAME,

    "components":
        selected_components,

    "weights":
        selected_weights,

    "top_models":
        TOP_MODELS
}

save_json_atomic(
    selection_manifest,
    CK_ROOT
    / "selected_model.json"
)

# ==============================================================
# 21. DEV에서 threshold 결정
# ==============================================================

dev_y = (
    base_oof.loc[
        dev_mask,
        "actual_peak"
    ]
    .to_numpy(
        dtype=int
    )
)

dev_prob = (
    selected_oof_prob[
        dev_mask
    ]
)

(
    operating_points,
    threshold_grid_df
) = choose_operating_points(
    dev_y,
    dev_prob
)

save_df(
    threshold_grid_df,
    "B4R_threshold_grid.csv"
)

save_json_atomic(
    operating_points,
    CK_ROOT
    / "operating_points.json"
)

print()
print(
    "⑫ Operating points:"
)

print(
    operating_points
)

# ==============================================================
# 22. DEV / AUDIT metrics
# ==============================================================

cv_metric_rows = []

for scope, mask in [
    (
        "DEV",
        dev_mask
    ),
    (
        "AUDIT",
        audit_mask
    )
]:

    yy = (
        base_oof.loc[
            mask,
            "actual_peak"
        ]
        .to_numpy(
            dtype=int
        )
    )

    prob = (
        selected_oof_prob[
            mask
        ]
    )

    for mode, threshold in (
        operating_points.items()
    ):

        mm = cls_metrics(
            yy,
            prob,
            threshold
        )

        cv_metric_rows.append({

            "scope":
                scope,

            "mode":
                mode,

            **mm
        })

cv_metrics = pd.DataFrame(
    cv_metric_rows
)

save_df(
    cv_metrics,
    "B4R_selected_cv_metrics.csv"
)

# ==============================================================
# 23. Specs map
# ==============================================================

SPEC_MAP = {
    spec[
        "name"
    ]:
        spec
    for spec in CANDIDATES
}

# ==============================================================
# 24. Full fit helper
# ==============================================================

def fit_component(
    component_name,
    train_times,
    eval_times
):

    spec = SPEC_MAP[
        component_name
    ]

    features = FEATURE_SETS[
        spec[
            "feature_set"
        ]
    ]

    X_train = Xall.loc[
        train_times,
        features
    ]

    X_eval = Xall.loc[
        eval_times,
        features
    ]

    y_train = yall.loc[
        train_times
    ]

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

    return (
        model.predict_proba(
            X_eval_i
        )[
            :,
            1
        ]
    )

# ==============================================================
# 25. Fixed validation
#
# ★ component별 즉시 Drive 저장
# ==============================================================

fixed_component_probs = {}

for component in (
    selected_components
):

    path = (
        FIXED_DIR
        / (
            safe_name(
                component
            )
            + ".npz"
        )
    )

    loaded = (
        load_npz_safe(
            path
        )
        if path.exists()
        else None
    )

    valid = (
        loaded is not None
        and "prob" in loaded
        and len(
            loaded[
                "prob"
            ]
        ) == len(
            Xva
        )
        and np.isfinite(
            loaded[
                "prob"
            ]
        ).all()
    )

    if valid:

        print(
            "⑬ FIXED SKIP(saved):",
            component
        )

        fixed_component_probs[
            component
        ] = (
            loaded[
                "prob"
            ]
        )

        continue

    print(
        "⑬ FIXED fit:",
        component
    )

    prob = fit_component(
        component,
        Xtr.index,
        Xva.index
    )

    save_npz_atomic(
        path,
        prob=np.asarray(
            prob,
            dtype=float
        )
    )

    fixed_component_probs[
        component
    ] = prob

fixed_prob = np.zeros(
    len(
        Xva
    ),
    dtype=float
)

for component, weight in zip(
    selected_components,
    selected_weights
):

    fixed_prob += (
        weight
        * fixed_component_probs[
            component
        ]
    )

# ==============================================================
# 26. Weekly expanding
#
# ★ block × component별 자동 저장
# ==============================================================

validation_start = (
    Xva.index[
        0
    ]
)

block_id = np.asarray(
    (
        Xva.index
        - validation_start
    ).days // 7,
    dtype=int
)

unique_blocks = sorted(
    np.unique(
        block_id
    )
)

expanding_prob = np.full(
    len(
        Xva
    ),
    np.nan
)

for block_seq, block in enumerate(
    unique_blocks,
    start=1
):

    block_mask = (
        block_id
        == block
    )

    block_times = (
        Xva.index[
            block_mask
        ]
    )

    block_start = (
        block_times[
            0
        ]
    )

    historical_times = (
        Xall.index[
            Xall.index
            < block_start
        ]
    )

    block_total_prob = np.zeros(
        len(
            block_times
        ),
        dtype=float
    )

    for component, weight in zip(
        selected_components,
        selected_weights
    ):

        component_dir = (
            EXPANDING_DIR
            / f"block_{int(block)}"
        )

        component_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        path = (
            component_dir
            / (
                safe_name(
                    component
                )
                + ".npz"
            )
        )

        loaded = (
            load_npz_safe(
                path
            )
            if path.exists()
            else None
        )

        valid = (
            loaded is not None
            and "prob" in loaded
            and len(
                loaded[
                    "prob"
                ]
            ) == len(
                block_times
            )
            and np.isfinite(
                loaded[
                    "prob"
                ]
            ).all()
        )

        if valid:

            probability = (
                loaded[
                    "prob"
                ]
            )

            print(
                f"⑭ block "
                f"{block_seq}/"
                f"{len(unique_blocks)} "
                f"{component}: "
                f"SKIP(saved)"
            )

        else:

            print(
                f"⑭ block "
                f"{block_seq}/"
                f"{len(unique_blocks)} "
                f"{component}: fit"
            )

            probability = fit_component(
                component,
                historical_times,
                block_times
            )

            save_npz_atomic(
                path,
                prob=np.asarray(
                    probability,
                    dtype=float
                )
            )

        block_total_prob += (
            weight
            * probability
        )

    expanding_prob[
        block_mask
    ] = (
        block_total_prob
    )

if not np.isfinite(
    expanding_prob
).all():

    raise RuntimeError(
        "Expanding probability 누락"
    )

# ==============================================================
# 27. Validation metrics
# ==============================================================

validation = pd.DataFrame({

    "timestamp":
        Xva.index,

    "actual_peak":
        yva.to_numpy(
            dtype=int
        ),

    "actual_max":
        maxva.to_numpy(
            dtype=float
        ),

    "fixed_prob":
        fixed_prob,

    "expanding_prob":
        expanding_prob,

    "production_lag1":
        Xva[
            "prod_lag1"
        ].to_numpy(),

    "hour":
        Xva.index.hour
})

validation_rows = []

for strategy, prob in [
    (
        "FIXED",
        fixed_prob
    ),
    (
        "EXPANDING",
        expanding_prob
    )
]:

    for mode, threshold in (
        operating_points.items()
    ):

        mm = cls_metrics(
            yva,
            prob,
            threshold
        )

        validation_rows.append({

            "strategy":
                strategy,

            "mode":
                mode,

            **mm
        })

        validation[
            (
                f"{strategy.lower()}_"
                f"{mode.lower()}_alert"
            )
        ] = (
            prob
            >= threshold
        ).astype(int)

validation_metrics = (
    pd.DataFrame(
        validation_rows
    )
)

save_df(
    validation,
    "B4R_validation_predictions.csv"
)

save_df(
    validation_metrics,
    "B4R_validation_metrics.csv"
)

# ==============================================================
# 28. References
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

        "TN":
            701,

        "Precision":
            145 / 166,

        "Recall":
            145 / 156,

        "AP":
            0.939
    },

    {
        "model":
            "B3.1_FIXED_BALANCED",

        "TP":
            150,

        "FP":
            39,

        "FN":
            6,

        "TN":
            683,

        "Precision":
            150 / 189,

        "Recall":
            150 / 156,

        "AP":
            0.950
    }
])

save_df(
    reference,
    "B4R_reference.csv"
)

# ==============================================================
# 29. B2 missed 11
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

if len(
    miss_compare
) != 11:

    raise RuntimeError(
        f"B2 missed 11개 중 "
        f"{len(miss_compare)}개만 찾음"
    )

miss_compare[
    "margin_over_176"
] = (
    miss_compare[
        "actual_max"
    ]
    - 176.0
)

save_df(
    miss_compare,
    "B4R_B2_missed_comparison.csv"
)

# ==============================================================
# 30. FN / FP 상세
# ==============================================================

error_rows = []

for strategy in [
    "fixed",
    "expanding"
]:

    for mode in [
        "f2_max",
        "fpr3",
        "fpr5",
        "rec95"
    ]:

        alert_col = (
            f"{strategy}_"
            f"{mode}_alert"
        )

        missed_mask = (
            (
                validation[
                    "actual_peak"
                ] == 1
            )
            &
            (
                validation[
                    alert_col
                ] == 0
            )
        )

        fp_mask = (
            (
                validation[
                    "actual_peak"
                ] == 0
            )
            &
            (
                validation[
                    alert_col
                ] == 1
            )
        )

        missed_df = (
            validation.loc[
                missed_mask
            ]
            .copy()
        )

        false_df = (
            validation.loc[
                fp_mask
            ]
            .copy()
        )

        save_df(
            missed_df,
            (
                f"B4R_{strategy}_"
                f"{mode}_missed.csv"
            )
        )

        save_df(
            false_df,
            (
                f"B4R_{strategy}_"
                f"{mode}_false.csv"
            )
        )

        error_rows.append({

            "strategy":
                strategy.upper(),

            "mode":
                mode.upper(),

            "FN":
                len(
                    missed_df
                ),

            "FP":
                len(
                    false_df
                )
        })

error_summary = pd.DataFrame(
    error_rows
)

save_df(
    error_summary,
    "B4R_error_summary.csv"
)

# ==============================================================
# 31. Final metadata
# ==============================================================

final_metadata = {

    "experiment":
        "KAMP_B4R_broad_peak_search_resume",

    "experiment_version":
        EXPERIMENT_VERSION,

    "run_id":
        RUN_ID,

    "peak_threshold":
        PEAK_THRESHOLD,

    "validation_peak_count":
        int(
            yva.sum()
        ),

    "candidate_count":
        len(
            CANDIDATES
        ),

    "total_fold_tasks":
        TOTAL_FOLD_TASKS,

    "dev_folds":
        DEV_FOLDS,

    "audit_folds":
        AUDIT_FOLDS,

    "selected_name":
        SELECTED_NAME,

    "selected_components":
        selected_components,

    "selected_weights":
        selected_weights,

    "operating_points":
        operating_points,

    "validation_used_for_selection":
        False,

    "validation_previously_viewed":
        True,

    "future_information_used":
        False,

    "peak_is_empirical_not_contract_peak":
        True,

    "measurement_units_verified":
        False,

    "resume_checkpoint":
        str(
            CK_ROOT
        )
}

save_json_atomic(
    final_metadata,
    OUT
    / "B4R_metadata.json"
)

save_json_atomic(
    final_metadata,
    SUMMARY_DIR
    / "B4R_metadata.json"
)

# ==============================================================
# 32. Source 저장
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
        "KAMP B-4R"
        in source
        and len(source) > 2000
    ):

        (
            OUT
            / "B4R_source.py"
        ).write_text(
            source,
            encoding="utf-8"
        )

except Exception:

    pass

# ==============================================================
# 33. requirements
# ==============================================================

try:

    from importlib.metadata import (
        version
    )

    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "catboost"
    ]

    text = "\n".join([
        f"{p}=={version(p)}"
        for p in packages
    ])

    (
        OUT
        / "B4R_requirements.txt"
    ).write_text(
        text + "\n",
        encoding="utf-8"
    )

except Exception:

    pass

# ==============================================================
# 34. 최종 ZIP
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
        f"KAMP_B4R_results_"
        f"{RUN_ID}.zip"
    )
)

shutil.copy2(
    archive,
    destination
)

update_manifest(

    status=
        "complete",

    completed_at=
        datetime.now().isoformat(),

    completed_fold_tasks=
        TOTAL_FOLD_TASKS,

    final_zip=
        str(
            destination
        )
)

# ==============================================================
# 35. ChatGPT 결과 출력
# ==============================================================

print()
print(
    "========== ChatGPT에 보낼 결과 시작 =========="
)

print()
print("[기본 정보]")

print(
    "전체:",
    len(
        Xall
    ),
    "| 학습:",
    len(
        Xtr
    ),
    "| 검증:",
    len(
        Xva
    ),
    "| 검증 peak:",
    int(
        yva.sum()
    ),
    "| peak 기준:",
    PEAK_THRESHOLD
)

print(
    "후보:",
    len(
        CANDIDATES
    ),
    "| fold 작업:",
    TOTAL_FOLD_TASKS
)

print()
print("[단일 모델 DEV 상위 15]")

print(
    scores
    .head(15)
    [
        [
            "model",
            "family",
            "feature_set",
            "DEV_AP",
            "AUDIT_AP"
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)

print()
print("[Ensemble 상위 15]")

print(
    ensemble_scores
    .head(15)
    .round(4)
    .to_string(
        index=False
    )
)

print()
print("[최종 DEV 선택]")

print(
    "모델:",
    SELECTED_NAME
)

print(
    "components:",
    selected_components
)

print(
    "weights:",
    selected_weights
)

print()
print("[Operating points]")

print(
    operating_points
)

print()
print("[DEV / AUDIT]")

print(
    cv_metrics
    .round(4)
    .to_string(
        index=False
    )
)

print()
print("[B2 / B3.1 Reference]")

print(
    reference
    .round(4)
    .to_string(
        index=False
    )
)

print()
print(
    "[B4R 동일 878시간 / 동일 156 peak]"
)

print(
    validation_metrics
    .sort_values(
        [
            "FN",
            "FP"
        ]
    )
    .round(4)
    .to_string(
        index=False
    )
)

print()
print("[FN / FP 요약]")

print(
    error_summary
    .sort_values(
        [
            "FN",
            "FP"
        ]
    )
    .to_string(
        index=False
    )
)

print()
print(
    "[B2 missed 11개 중 구조한 개수]"
)

for strategy in [
    "fixed",
    "expanding"
]:

    for mode in [
        "f2_max",
        "fpr3",
        "fpr5",
        "rec95"
    ]:

        col = (
            f"{strategy}_"
            f"{mode}_alert"
        )

        saved = int(
            miss_compare[
                col
            ].sum()
        )

        print(
            strategy.upper(),
            "+",
            mode.upper(),
            ":",
            f"{saved}/11"
        )

print()
print(
    "[B2 missed 11 상세]"
)

show_cols = [
    "timestamp",
    "actual_max",
    "margin_over_176",
    "production_lag1",
    "fixed_prob",
    "expanding_prob"
]

for col in (
    miss_compare.columns
):

    if (
        col.endswith(
            "_alert"
        )
        and col not in show_cols
    ):

        show_cols.append(
            col
        )

print(
    miss_compare[
        show_cols
    ]
    .round(4)
    .to_string(
        index=False
    )
)

print()
print("[체크포인트]")

print(
    "중간 저장 폴더:",
    CK_ROOT
)

print(
    "런타임 종료 시 같은 코드를 다시 실행하면 자동 재개"
)

print(
    "완료된 model × fold는 다시 학습하지 않음"
)

print()
print("[주의]")

print(
    "- empirical peak = 176 고정"
)

print(
    "- validation peak 156개 확인"
)

print(
    "- 모델/ensemble/threshold 선택 = DEV folds 1~3"
)

print(
    "- AUDIT folds 4~6은 이번 선택 코드에서 사용하지 않음"
)

print(
    "- 다만 이전 실험에서 이미 관찰한 기간이므로 완전한 untouched test는 아님"
)

print(
    "- 878시간 validation 역시 이전에 본 기간이며 선택에는 사용하지 않음"
)

print(
    "- 176은 전력회사 계약피크가 아닌 empirical threshold"
)

print()
print(
    "총 실행 시간(분):",
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
    "최종 ZIP:",
    destination
)

print(
    "체크포인트:",
    CK_ROOT
)

print(
    "========== ChatGPT에 보낼 결과 끝 =========="
)