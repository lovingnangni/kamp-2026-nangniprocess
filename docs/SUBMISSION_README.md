# KAMP 소스코드 및 예측결과 제출본 V2

## 제출 구성

- run_all.py, RUN_ALL.ipynb, src/: 원본부터 고정된 선택 모델을 학습하고 결과를 만드는 Python 소스코드
- requirements.txt: 실행에 사용한 라이브러리 고정 버전
- data/KAMP_5_data.zip: 원본 학습 데이터. 원본 CSV 해시는 매니페스트에 기록
- predictions/test_predictions.csv: KAMP-NOTE에서 새로 학습한 선택 모델의 Later 879시간 예측 결과. 같은 시각의 실제 정답, 피크 정답, 예측 오차는 없음
- PACKAGE_MANIFEST.json: ZIP 내부 파일 해시와 예측결과 출처

## 실행

ZIP을 푼 폴더에서 다음 두 명령을 순서대로 실행합니다.

    python -m pip install -r requirements.txt
    python run_all.py --data data/KAMP_5_data.zip --output outputs

노트북을 선호하면 RUN_ALL.ipynb의 실행 코드 셀 하나를 사용할 수 있습니다.
선택 모델 전체를 원본부터 새로 학습하고 예측합니다. 과거 저장 예측이나 Drive
체크포인트가 필요하지 않습니다. KAMP-NOTE User defined CPU(Python 3.11.9)에서
원본부터 결과까지 실행하는 데 약 4.09분이 소요되었고, Colab에서는 약 12.88분이
소요되었습니다. 실행 시간은 환경에 따라 달라집니다.

## 예측결과의 의미

predictions/test_predictions.csv는 KAMP-NOTE 실행 ID
20261005_154319_925393_UTC의
outputs/RUN_20261005_154319_925393_UTC/operator_predictive_only.csv를
바이트 그대로 복사한 제출용 예측 파일입니다. 예측 시각은
2021-08-09 09:00부터 2021-09-14 23:00까지입니다. A12의 시간별 평균값 예측,
평균값 오차 구간, B5 위험점수, B7 시간 내 최대값 예측, B8·B11 경고 및
NORMAL/WATCH/WARNING 검토 단계가 포함됩니다.
이 기간은 시간상 Later 구간이지만 프로젝트 전체의 완전 미열람 홀드아웃은
아닙니다. 원본 ZIP에는 평가에 사용한 관측값도 포함되어 있으나 예측 결과
파일에는 같은 시각의 실제값을 넣지 않았습니다.

실행 후 생성되는 outputs/RUN_<UTC>/manifest.json에서 실행 상태
PASS_SUBMISSION_RAW_TO_RESULTS와 입력 원본 CSV 해시를 확인할 수 있습니다.
원본 ZIP을 다시 압축하면 ZIP 바이트 해시는 달라져도 내부 CSV 바이트는
동일할 수 있습니다.

## 범위와 한계

본 코드는 이미 선택된 A11/A12, B5/B7/B8/B11만 새로 학습·추론하며
전체 후보 탐색이나 새 임계값 선택은 재실행하지 않습니다. B11 Later 임계값은
과거 Validation 결과로 계산합니다. 176은 경험적 피크 경계이며 계약전력
기준은 확인되지 않았습니다. 공정 제약 엔진의 13→9는 가상 사례이고
실제 전기요금 절감 실적이 아닙니다. 실제 설비·재고·제품 전환 입력
27개 항목이 없어 실제 일정 후보와 자동제어는 각각 0건입니다.

src/B4R_source.py, src/B7_source.py의 SHA256은 실행 시작 시 확인합니다.
과거 예측과의 29항목 대조 및 전체 후보 재선택 근거는 별도 검증 자료에
보관됩니다.
