# KAMP submission: raw dataset -> preprocess -> selected fit -> inference -> results.
# Historical predictions and Drive checkpoints are not inputs to this runner.
import argparse, hashlib, importlib.metadata, io, json, os, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

START = time.perf_counter()
PIN = {'numpy':'2.1.3','pandas':'2.2.3','scikit-learn':'1.6.1',
       'lightgbm':'4.6.0','catboost':'1.2.10'}
for package,version in PIN.items():
    actual=importlib.metadata.version(package)
    if actual!=version:
        raise RuntimeError(f'{package}={actual}; requirements.txt requires {version}')

parser=argparse.ArgumentParser(description='KAMP selected models, raw-to-results')
HERE=Path(__file__).resolve().parent
parser.add_argument('--data',type=Path,default=HERE/'data'/'KAMP_5_data.zip')
parser.add_argument('--output',type=Path,default=HERE/'outputs')
args=parser.parse_args()
RAW=args.data.resolve()
if not RAW.is_file(): raise FileNotFoundError(RAW)
OUT_ROOT=args.output.resolve()
OUT_ROOT.mkdir(parents=True,exist_ok=True)
RUN=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC')
print('RUN_ID:',RUN,'| 원본 데이터에서 선택 모델만 새 학습',flush=True)
with ZipFile(RAW) as z:
    names=[n for n in z.namelist() if n.lower().endswith('.csv')]
    if len(names)!=1: raise ValueError('원본 ZIP 내부 CSV 개수 불일치')
    data=z.read(names[0])
raw_sha=hashlib.sha256(data).hexdigest()
if raw_sha!='8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830':
    raise ValueError('원본 CSV SHA256 불일치')
raw=pd.read_csv(io.BytesIO(data),encoding='utf-8-sig')
hour=pd.to_numeric(raw['시간'],errors='coerce')
valid=hour.between(0,23)&hour.eq(np.floor(hour))
if raw.shape!=(6168,18) or int((~valid).sum())!=48: raise ValueError('원본 크기·이상 시간 불일치')
clean=raw.loc[valid].copy()
clean['timestamp']=(pd.to_datetime(clean['날짜'].astype(str),format='%Y%m%d')
                    +pd.to_timedelta(clean['시간'],unit='h'))
if clean.timestamp.duplicated().any(): raise ValueError('중복 timestamp')
hourly=clean.set_index('timestamp').sort_index().asfreq('h')
mean=hourly['평균'].astype(float)
production=hourly['생산량'].astype(float)
maximum=hourly[['15분','30분','45분','60분']].max(axis=1,skipna=False).astype(float)
ix=hourly.index

# A: 기존 Base17과 동일한, 현재 시각 이전에 관측된 입력만 생성.
xa=pd.DataFrame(index=ix)
xa['hour_sin']=np.sin(2*np.pi*ix.hour/24)
xa['hour_cos']=np.cos(2*np.pi*ix.hour/24)
xa['weekday']=ix.dayofweek;xa['month']=ix.month
xa['is_weekend']=(ix.dayofweek>=5).astype(int)
for lag in (1,2,3,24,168):xa[f'mean_lag{lag}']=mean.shift(lag)
xa['mean_roll3']=mean.shift(1).rolling(3).mean()
xa['mean_roll24']=mean.shift(1).rolling(24).mean()
xa['production_lag1']=production.shift(1)
xa['production_lag24']=production.shift(24)
xa['max_lag1']=maximum.shift(1)
xa['max_lag24']=maximum.shift(24)
xa['max_rollmean3']=maximum.shift(1).rolling(3).mean()
base_index=ix[xa.notna().all(axis=1)&mean.notna()&maximum.notna()]
xa=xa.loc[base_index]
if len(base_index)!=5856 or tuple(xa.columns)!=(
    'hour_sin','hour_cos','weekday','month','is_weekend','mean_lag1','mean_lag2','mean_lag3',
    'mean_lag24','mean_lag168','mean_roll3','mean_roll24','production_lag1',
    'production_lag24','max_lag1','max_lag24','max_rollmean3'):
    raise ValueError('A Base17 시각·특성 불일치')
cut1,cut2=int(len(xa)*.70),int(len(xa)*.85)
if (cut1,cut2-cut1,len(xa)-cut2)!=(4099,878,879): raise ValueError('Train/Validation/Later 분할 불일치')
vtime,later_time=base_index[cut1:cut2],base_index[cut2:]
if (int((maximum.loc[vtime]>=176).sum()),int((maximum.loc[later_time]>=176).sum()))!=(156,174):
    raise ValueError('피크 개수 불일치')
print('① 원본/시간축/17개 A 특성 PASS',flush=True)

# Original B4R/B7 feature and model definitions are shipped locally.
b4file=HERE/'src'/'B4R_source.py'
b7file=HERE/'src'/'B7_source.py'
b4bytes=b4file.read_bytes()
b7bytes=b7file.read_bytes()
if hashlib.sha256(b4bytes).hexdigest()!='9039e7f2f456a010c34e35b4c88f5d6bea4ea82eecd5c4fb751f4adcac2cbeba':
    raise ValueError('B4R source SHA256 mismatch')
if hashlib.sha256(b7bytes).hexdigest()!='f74af70d71a972bedc28586acc09063823759cb46836d343e61d033a995122ed':
    raise ValueError('B7 source SHA256 mismatch')
def prepare(stage, work):
    os.environ.update(KAMP_SUBMISSION_ROOT=str(HERE), KAMP_RAW_ZIP=str(RAW),
                      KAMP_STAGE_WORK=str(work))
    code=(HERE/'src'/f'{stage}_prepare.py').read_text(encoding='utf-8')
    context={'__name__':f'kamp_{stage.lower()}_selected'}
    exec(compile(code,f'{stage}_prepare.py','exec'),context)
    return context
with tempfile.TemporaryDirectory(prefix='kamp_selected_') as temporary:
    work=Path(temporary)
    b4=prepare('B4R',work/'B4R')
    b7=prepare('B7',work/'B7')
    if not base_index.equals(b4['Xall'].index) or not base_index.equals(b7['Xall'].index):
        raise ValueError('A/B timestamps mismatch')
    print('② 원본 B4R/B7 전처리와 시각축 PASS',flush=True)
    # The next block trains the selected models from raw; no old predictions.
# 두 B 분류기와 한 B 회귀 모델만 Train 4099시간으로 새 학습.
selected=('ET_CORE_leaf1_none','CAT_CORE_PLUS_d6_w1p0','LGB_Q85_CORE')
scores={}
for name in selected:
    ctx=b7 if name.startswith('LGB_') else b4
    spec=next(s for s in ctx['CANDIDATES'] if s['name']==name)
    columns=ctx['FEATURE_SETS'][spec['feature_set']]
    imputer=ctx['SimpleImputer'](strategy='median')
    xtrain=imputer.fit_transform(ctx['Xtr'][columns])
    xfuture=imputer.transform(ctx['Xall'].iloc[cut1:][columns])
    model=ctx['make_model'](spec)
    target=ctx['yreg_tr'] if name.startswith('LGB_') else ctx['ytr']
    model.fit(xtrain,target)
    scores[name]=(np.asarray(model.predict(xfuture),float) if name.startswith('LGB_')
                  else np.asarray(model.predict_proba(xfuture)[:,1],float))
    if scores[name].shape!=(1757,) or not np.isfinite(scores[name]).all():
        raise ValueError(f'{name} 새 예측 누락')
    print('③ 새 학습:',name,'→',len(scores[name]),'시간',flush=True)
base_score=np.maximum(scores[selected[0]],.6*scores[selected[1]])
regression_score=scores[selected[2]]

# A: Validation 0~5주 + Later 6~11주에 주간 expanding ExtraTrees 새 학습.
# Later의 전주 실제값은 해당 주 시작 때 관측된 이력에만 들어간다.
anchor=vtime[0]
future_time=base_index[cut1:]
block=np.floor((future_time-anchor)/pd.Timedelta(hours=168)).astype(int)
if set(block)!={0,1,2,3,4,5,6,7,8,9,10,11} or block[878]!=6:
    raise ValueError('A 주간 block 분할 불일치')
level=np.full(1757,np.nan);delta=np.full(1757,np.nan)
for k in sorted(set(block)):
    boundary=anchor+pd.Timedelta(hours=168*int(k))
    fit_idx=base_index[base_index<boundary]
    mask=np.asarray(block==k)
    pred_x=xa.loc[future_time[mask]]
    if fit_idx.max()>=boundary: raise ValueError('A 미래 학습 누출')
    pieces=[]
    for max_features,target in ((.85,mean.loc[fit_idx]),
                                (.75,mean.loc[fit_idx]-xa.loc[fit_idx,'mean_lag1'])):
        et=ExtraTreesRegressor(n_estimators=80,criterion='absolute_error',
                               min_samples_leaf=2,max_features=max_features,
                               random_state=42,n_jobs=-1)
        et.fit(xa.loc[fit_idx],target)
        pieces.append(et.predict(pred_x))
    level[mask]=np.maximum(0,pieces[0])
    delta[mask]=np.maximum(0,pieces[1]+pred_x.mean_lag1.to_numpy(float))
    print(f'④ A11 주간 모델 {k+1}/12 완료 | 학습 {len(fit_idx)} → 예측 {int(mask.sum())}시간 '
          f'| {(time.perf_counter()-START)/60:.1f}분',flush=True)
a11=.45*level+.55*delta
actual_mean=mean.loc[future_time].to_numpy(float)
residual=actual_mean-a11
a12=np.empty(1757);correction=np.zeros(1757)
for i,t in enumerate(future_time):
    start=future_time.searchsorted(t-pd.Timedelta(hours=72))
    if i-start>=24: correction[i]=float(np.median(residual[start:i]))
    a12[i]=max(0.,a11[i]+correction[i])

# B8 원래 선택 임계값과 B11 이전 구간에서만 갱신한 임계값.
val_peak=(maximum.loc[vtime].to_numpy(float)>=176).astype(int)
later_peak=(maximum.loc[later_time].to_numpy(float)>=176).astype(int)
def best_threshold(y,score,limit=.05):
    y=np.asarray(y,bool);score=np.asarray(score,float)
    winners=[]
    for threshold in np.unique(score[y]):
        alert=score>=threshold
        tp=int(np.sum(y&alert));fp=int(np.sum((~y)&alert))
        if fp/int(np.sum(~y))<=limit+1e-12:
            winners.append((tp,fp,float(threshold)))
    if not winners: raise ValueError('B11 과거 자료에 허용 FPR 후보 없음')
    return sorted(winners,key=lambda row:(-row[0],row[1],-row[2]))[0][2]
val_base,late_base=base_score[:878],base_score[878:]
val_reg,late_reg=regression_score[:878],regression_score[878:]
b11_val_threshold=.2124285195563263  # 이전 fold 4~6에서 고정된 정책
raw_next=best_threshold(val_peak[-672:],val_base[-672:],.05)
b11_later_threshold=.5*b11_val_threshold+.5*raw_next
if abs(raw_next-.2759433145178815)>1e-12 or abs(b11_later_threshold-.2441859170371039)>1e-12:
    raise ValueError(f'B11 과거 보정 재현 실패: {raw_next}, {b11_later_threshold}')
b8_val=((val_base>=.498)|(val_reg>=177.86438220946707)).astype(int)
b8_later=((late_base>=.498)|(late_reg>=177.86438220946707)).astype(int)
b11_val=(val_base>=b11_val_threshold).astype(int)
b11_later=(late_base>=b11_later_threshold).astype(int)

# A12의 Validation 잔차만 고정 90% 반경에 사용. Later에서는 직전 관측만 온라인 갱신.
def order_quantile(values,q):
    a=np.sort(np.asarray(values,float))
    if not len(a) or not np.isfinite(a).all(): raise ValueError('불확실성 잔차 오류')
    pos=min(max(int(np.ceil((len(a)+1)*q)),1),len(a))
    return float(a[pos-1])
val_abs=np.abs(actual_mean[:878]-a12[:878])
fixed90=order_quantile(val_abs,.90)
online_q=[];online_n=[]
past_t=list(vtime);past_abs=list(val_abs)
for i,t in enumerate(later_time):
    recent=[r for old_t,r in zip(past_t,past_abs) if t-pd.Timedelta(hours=168)<=old_t<t]
    online_n.append(len(recent))
    online_q.append(order_quantile(recent,.90) if len(recent)>=72 else fixed90)
    past_t.append(t);past_abs.append(abs(actual_mean[878+i]-a12[878+i]))
online_q=np.asarray(online_q,float)

# 운영표: 예측 당시 이미 이용 가능한 정보만. 정답·오차는 감사 CSV로 분리.
tier=np.select([b8_later==1,b11_later==1],['WARNING','WATCH'],default='NORMAL')
if list(pd.Series(tier).value_counts().reindex(['NORMAL','WATCH','WARNING'],fill_value=0))!=[649,35,195]:
    raise ValueError('운영 상태 시간 수 불일치')
actions={'NORMAL':'정규 모니터링',
         'WATCH':'측정 추세·생산 일정 확인 후 안전·품질·납기·설비 제약 검토',
         'WARNING':'우선 현장 검토. 안전·품질·납기·설비 제약을 확인한 뒤 이동 가능한 비핵심 부하만 검토'}
needs_review=tier!='NORMAL'
check=np.where(needs_review,'TO_BE_CHECKED_BY_OPERATOR','NOT_REQUESTED')
operator=pd.DataFrame({
    'timestamp':later_time,'A12_mean_prediction':a12[878:],
    'A12_mean_interval90_low':a12[878:]-fixed90,
    'A12_mean_interval90_high':a12[878:]+fixed90,
    'A12_online_interval90_low':a12[878:]-online_q,
    'A12_online_interval90_high':a12[878:]+online_q,
    'B5_risk_score_NOT_probability':late_base,
    'B7_predicted_within_hour_max':late_reg,
    'B8_WARNING':b8_later,'B11_support':b11_later,'tier':tier,
    'production_previous_hour':production.shift(1).loc[later_time].to_numpy(float),
    'action':[actions[t] for t in tier],
    'review_priority':np.select([tier=='WARNING',tier=='WATCH'],[2,1],default=0),
    'operator_review_required':np.where(needs_review,'YES','NO'),
    'safety_check':check,'quality_check':check,'delivery_check':check,
    'equipment_operation_check':check,'movable_noncritical_load_check':check,
    'shifted_peak_recheck_required':np.where(needs_review,'YES','NO'),
    'operator_final_decision':np.where(needs_review,'PENDING_OPERATOR_REVIEW','MONITOR'),
    'product_and_equipment_schedule':'원본에 없음: 현장 확인 필요',
    'automatic_control':'NO',
    'peak_threshold_note':'176은 경험적 경계; 계약전력 기준 아님',
    'interval_note':'A12 구간은 평균값 오차 범위; 시간 내 최대값의 보증 아님'
})
if operator.automatic_control.ne('NO').any() or any(k in operator.columns for k in ['actual','actual_max','actual_peak','error']):
    raise ValueError('현장용 표에 자동제어 또는 실제값이 포함됨')

def metrics(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,bool)
    tp=int(np.sum(y&p));fp=int(np.sum((~y)&p));fn=int(np.sum(y&(~p)));tn=int(np.sum((~y)&(~p)))
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'F1':2*tp/(2*tp+fp+fn),
            'Recall':tp/(tp+fn),'FPR':fp/(fp+tn)}
val_prev=(maximum.shift(1).loc[vtime].to_numpy(float)>=176)
later_prev=(maximum.shift(1).loc[later_time].to_numpy(float)>=176)
metric_rows=[]
for period,y,prev,b5,b8,b11 in [
    ('Validation',val_peak,val_prev,val_base>=.36,b8_val,b11_val),
    ('Later',later_peak,later_prev,late_base>=.36,b8_later,b11_later)]:
    for name,p in [('PriorHourPeak',prev),('B5_FPR5',b5),('B8_WARNING',b8),('B11_support',b11)]:
        metric_rows.append({'period':period,'method':name,**metrics(y,p)})
metric_frame=pd.DataFrame(metric_rows)
audit=pd.DataFrame({'timestamp':later_time,'actual_mean':actual_mean[878:],
                    'actual_max':maximum.loc[later_time].to_numpy(float),
                    'actual_peak':later_peak,'A12_prediction':a12[878:],
                    'B8_WARNING':b8_later,'B11_support':b11_later,'tier':tier})
audit['A12_absolute_error']=np.abs(audit.actual_mean-audit.A12_prediction)
a_metrics={period:{'N':int(len(y)),
                   'A11_MAE':float(np.mean(np.abs(y-p1))),
                   'A12_MAE':float(np.mean(np.abs(y-p2)))}
           for period,y,p1,p2 in [
               ('Validation',actual_mean[:878],a11[:878],a12[:878]),
               ('Later',actual_mean[878:],a11[878:],a12[878:])]}
interval_coverage={
    'Later_fixed90':float(np.mean(np.abs(actual_mean[878:]-a12[878:])<=fixed90)),
    'Later_online90':float(np.mean(np.abs(actual_mean[878:]-a12[878:])<=online_q))}
if (metric_frame[(metric_frame.period=='Later')&(metric_frame.method=='B8_WARNING')]
      [['TP','FP','FN','TN']].iloc[0].tolist()!=[157,38,17,667]):
    raise ValueError('B8 Later 혼동행렬 불일치')

# Process-aware gate: do not fabricate real equipment schedules.
from src import process_constraints as pc
missing=pc.actual['missing_fields']
if pc.actual['status']!='DATA_REQUIRED' or len(missing)!=27:
    raise RuntimeError('Actual KAMP process inputs must remain unavailable')
if (pc.good['original']['max_quarter_hour_load'],
    pc.good['candidate']['max_quarter_hour_load'])!=(13,9):
    raise RuntimeError('Synthetic process demonstration mismatch')
card=operator.copy()
review=tier!='NORMAL'
card['process_input_status']=np.where(review,'DATA_REQUIRED','NOT_REQUESTED')
card['process_constraints_checked']='NO'
card['schedule_candidate_status']=np.where(review,'ON_HOLD_MISSING_PROCESS_DATA','NOT_REQUESTED')
card['shifted_peak_recheck_status']='NOT_PERFORMED'
card['reduction_claim_status']='NOT_VERIFIED'
card['missing_process_input_count']=np.where(review,27,0)
card['operator_next_step']=np.select(
    [tier=='WARNING',tier=='WATCH'],
    ['우선 현장 확인; 설비별 부하·재고·공정 제약 입력 확보 후 후보를 별도 검증',
     '추세·일정 확인; 설비별 부하·재고·공정 제약 입력 확보 후 후보를 별도 검증'],
    default='정규 모니터링')
card['decision_scope']='경보 및 작업자 검토만; 운전시점 이동안 미산출'
worksheet=card.loc[review].copy()
def category(field):
    if field.startswith('materials') or 'material_delta' in field:
        return '물질수지·완충재고'
    if any(word in field for word in ('power','background','profile','unit')):
        return '15분 부하·재기동 안전'
    if any(word in field for word in ('product','changeover','batch','mode')):
        return '제품전환·연속/배치'
    if any(word in field for word in ('hours','starts','prior_state')):
        return '운전시간·초기상태'
    if any(word in field for word in ('available','availability')):
        return '설비 가용성'
    if any(word in field for word in ('approved','approval')):
        return '안전·품질 승인'
    return '분석 구간·설비 식별'
requirements=pd.DataFrame({'input_path':missing,
                           'category':[category(x) for x in missing],
                           'source':'실제 공장 담당자 확인 필요','state':'NOT_PROVIDED'})
if (len(card),len(worksheet),len(requirements))!=(879,230,27):
    raise RuntimeError('Process input gate row counts mismatch')
if (not card.automatic_control.eq('NO').all()
    or not card.reduction_claim_status.eq('NOT_VERIFIED').all()
    or not card.process_constraints_checked.eq('NO').all()):
    raise RuntimeError('Unsafe operating instruction')

out=OUT_ROOT/f'RUN_{RUN}'
out.mkdir(parents=True,exist_ok=False)
pd.DataFrame({'timestamp':future_time,
              'A11_level':level,'A11_delta':delta,'A11_fixed':a11,
              'A12_correction':correction,'A12_final':a12}).to_csv(
    out/'A_validation_later_predictions.csv',index=False,encoding='utf-8-sig')
pd.DataFrame({'timestamp':future_time,'B5_base_score':base_score,
              'B7_regression_score':regression_score,
              'B8_WARNING':np.r_[b8_val,b8_later],
              'B11_support':np.r_[b11_val,b11_later]}).to_csv(
    out/'B_validation_later_predictions.csv',index=False,encoding='utf-8-sig')
operator.to_csv(out/'operator_predictive_only.csv',index=False,encoding='utf-8-sig')
card.to_csv(out/'operator_predictive_process_gated_879.csv',index=False,encoding='utf-8-sig')
worksheet.to_csv(out/'operator_review_230.csv',index=False,encoding='utf-8-sig')
requirements.to_csv(out/'required_process_inputs_27.csv',index=False,encoding='utf-8-sig')
with (out/'synthetic_process_example.json').open('w',encoding='utf-8') as f:
    json.dump({'warning':'가상 사례만; 실제 KAMP 공장 피크 또는 요금 절감 아님',
               'case':pc.demo,'original':pc.original,'candidate':pc.shifted,
               'peak_before':13,'peak_after':9},f,ensure_ascii=False,indent=2)

audit.to_csv(out/'AUDIT_ONLY_later_actuals.csv',index=False,encoding='utf-8-sig')
metric_frame.to_csv(out/'classification_metrics.csv',index=False,encoding='utf-8-sig')
manifest={'status':'PASS_SUBMISSION_RAW_TO_RESULTS','run_id':RUN,
          'raw_csv_sha256':raw_sha,'input_zip_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'versions':{p:importlib.metadata.version(p) for p in PIN},
          'frozen_models':list(selected)+['A11_ET_LEVEL','A11_ET_DELTA','A12_RESIDUAL'],
          'training_from_raw':True,'used_old_predictions_for_training':False,
          'old_predictions_for_comparison_only':False,'candidate_search_rerun':False,'submission_mode':'selected_frozen_models',
          'historical_prediction_files_required':False,
          'A_metrics':a_metrics,'A_interval_coverage':interval_coverage,
          'source_sha256':{'B4R':hashlib.sha256(b4bytes).hexdigest(),
                           'B7':hashlib.sha256(b7bytes).hexdigest()},
                    'B11_later_threshold_from_prior_validation':b11_later_threshold,
          'fixed_mean_interval90_radius':fixed90,
          'later_A12_MAE':float(np.mean(audit.A12_absolute_error)),
          'operation_counts':{k:int(np.sum(tier==k)) for k in ['NORMAL','WATCH','WARNING']},
          'process_status':'DATA_REQUIRED','required_process_input_count':27,
          'operator_review_rows':230,'actual_schedule_candidates':0,
          'automatic_controls':0,'synthetic_peak_before':13,'synthetic_peak_after':9,
          'limitation':'시간별 관측이 순차 도착해야 lag/온라인보정을 갱신할 수 있음. Later는 프로젝트 전체 미열람 홀드아웃이 아님. 설비·제품 정보/계약 기준/실제 절감액은 없음.'}
with (out/'manifest.json').open('w',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
print('\n=== 8단계 최종 선택 모델 clean run ===')
print('상태:',manifest['status'],'| 원본부터 선택 모델 새 학습·추론 완료')
print('A12 Later MAE:',round(manifest['later_A12_MAE'],6),'| fixed90 반경:',round(fixed90,6))
print('A 예측:',{k:{n:round(v,5) if isinstance(v,float) else v for n,v in row.items()}
               for k,row in a_metrics.items()},'| Later 구간 포함률:',interval_coverage)
print(metric_frame.round(4).to_string(index=False))
print('운영:',manifest['operation_counts'],'| 자동제어: 0건')
print('저장:',out)
print('소요(분):',round((time.perf_counter()-START)/60,2))
