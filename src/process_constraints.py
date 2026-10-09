# Original process predicate and synthetic cases, without Drive I/O.
import copy, math
def missing_fields(case):
    paths=['horizon_hours','fixed_background_15m','power_unit',
           'background_excludes_equipment','profile_source_verified',
           'materials','equipment.id',
           'equipment.mode','equipment.power_profile_15m',
           'equipment.startup_extra_first_15m','equipment.safe_power_15m_min',
           'equipment.safe_power_15m_max',
           'equipment.max_power_ramp_per_15m','equipment.material_delta_by_product',
           'equipment.allowed_products','equipment.changeover_off_hours',
           'equipment.min_on_hours','equipment.min_off_hours',
           'equipment.batch_hours','equipment.required_on_hours',
           'equipment.max_starts','equipment.availability_hourly',
           'equipment.prior_state.on','equipment.prior_state.duration_hours',
           'equipment.prior_state.last_product','equipment.safety_approved',
           'equipment.quality_approved']
    missing=[]
    for path in paths:
        value=case
        for part in path.split('.'):
            value=value.get(part) if isinstance(value,dict) else None
        if path=='equipment.prior_state.last_product' and value is None:
            prior=(case.get('equipment') or {}).get('prior_state')
            if isinstance(prior,dict) and prior.get('on') is False:
                continue  # 이전에 설비가 꺼져 있었다면 직전 제품이 없어도 된다.
        if path=='equipment.batch_hours' and value is None:
            if (case.get('equipment') or {}).get('mode')=='continuous':
                continue
        if value is None or value==[] or value=={}: missing.append(path)
    materials=case.get('materials')
    if isinstance(materials,dict):
        for key,material in materials.items():
            for field in ['initial','capacity','inflow_hourly','external_use_hourly','min_final']:
                if not isinstance(material,dict) or material.get(field) is None:
                    missing.append(f'materials.{key}.{field}')
    return missing

def evaluate(case,schedule):
    missing=missing_fields(case)
    if missing: return {'status':'DATA_REQUIRED','missing_fields':missing,'violations':[]}
    h=case['horizon_hours'];e=case['equipment'];materials=case['materials']
    if not isinstance(h,int) or h<=0 or not isinstance(schedule,list) or len(schedule)!=h:
        return {'status':'INVALID_INPUT','violations':['horizon_or_schedule_length']}
    if (len(case['fixed_background_15m'])!=4*h or len(e['power_profile_15m'])!=4
        or len(e['availability_hourly'])!=h or e['mode'] not in ('batch','continuous')
        or any(len(m['inflow_hourly'])!=h or len(m['external_use_hourly'])!=h
               for m in materials.values())):
        return {'status':'INVALID_INPUT','violations':['profile_length_or_mode']}
    if case['background_excludes_equipment'] is not True or case['profile_source_verified'] is not True:
        return {'status':'BLOCKED','violations':['unverified_load_decomposition']}
    def nonnegative(v):
        return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0
    parameters=[e[k] for k in ('startup_extra_first_15m','safe_power_15m_min',
                               'safe_power_15m_max',
                               'max_power_ramp_per_15m','changeover_off_hours',
                               'min_on_hours','min_off_hours','required_on_hours','max_starts')]
    if (not all(nonnegative(v) for v in parameters)
        or e['safe_power_15m_min']>e['safe_power_15m_max']
        or not isinstance(e['required_on_hours'],int) or e['required_on_hours']>h
        or not isinstance(e['prior_state']['on'],bool)
        or not isinstance(e['prior_state']['duration_hours'],int)
        or e['prior_state']['duration_hours']<0
        or not all(isinstance(v,bool) for v in e['availability_hourly'])):
        return {'status':'INVALID_INPUT','violations':['process_parameter_values']}
    for m in materials.values():
        if (not all(nonnegative(m[k]) for k in ('initial','capacity','min_final'))
            or m['initial']>m['capacity'] or m['min_final']>m['capacity']
            or not all(nonnegative(v) for v in m['inflow_hourly']+m['external_use_hourly'])):
            return {'status':'INVALID_INPUT','violations':['material_parameter_values']}
    if not all(nonnegative(v) for v in case['fixed_background_15m']):
        return {'status':'INVALID_INPUT','violations':['background_values']}
    if not all(nonnegative(v) for v in e['power_profile_15m']):
        return {'status':'INVALID_INPUT','violations':['equipment_power_values']}
    if any(p is not None and p not in e['allowed_products'] for p in schedule):
        return {'status':'INVALID_INPUT','violations':['unknown_product']}
    if any(p not in e['material_delta_by_product'] for p in e['allowed_products']):
        return {'status':'DATA_REQUIRED','missing_fields':['material_delta_by_product'],
                'violations':[]}
    if any(k not in materials for p in e['allowed_products']
           for k in e['material_delta_by_product'][p]):
        return {'status':'DATA_REQUIRED','missing_fields':['materials_for_process'],
                'violations':[]}
    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v)
           for p in e['allowed_products'] for v in e['material_delta_by_product'][p].values()):
        return {'status':'INVALID_INPUT','violations':['material_rate_values']}
    if e['mode']=='batch' and (not isinstance(e['batch_hours'],int) or e['batch_hours']<=0):
        return {'status':'INVALID_INPUT','violations':['batch_hours']}
    if not e['safety_approved'] or not e['quality_approved']:
        return {'status':'BLOCKED','violations':['safety_or_quality_approval']}

    violations=[];inventory={k:float(m['initial']) for k,m in materials.items()}
    trajectory={k:[] for k in materials}
    prev=bool(e['prior_state']['on']);run=int(e['prior_state']['duration_hours'])
    last=e['prior_state']['last_product'];starts=0;on_hours=0
    equipment_prev=float(e['power_profile_15m'][-1]) if prev else 0.0
    loads=[];fixed=case['fixed_background_15m']
    for t,product in enumerate(schedule):
        on=product is not None
        starting=on and not prev
        if on and not e['availability_hourly'][t]:
            violations.append(f'hour{t}:equipment_unavailable')
        if starting:
            if run<int(e['min_off_hours']):
                violations.append(f'hour{t}:minimum_off_time')
            if last is not None and product!=last and run<int(e['changeover_off_hours']):
                violations.append(f'hour{t}:product_changeover')
            starts+=1;run=1
        elif on:
            if last is not None and product!=last and int(e['changeover_off_hours'])>0:
                violations.append(f'hour{t}:product_changeover')
            run+=1
        elif prev:
            if run<int(e['min_on_hours']):
                violations.append(f'hour{t}:minimum_on_time')
            if e['mode']=='batch' and run!=int(e['batch_hours']):
                violations.append(f'hour{t}:incomplete_batch')
            run=1
        else:
            run+=1

        if on: last=product;on_hours+=1
        for q in range(4):
            eq=(float(e['power_profile_15m'][q]) if on else 0.0)
            if starting and q==0: eq+=float(e['startup_extra_first_15m'])
            if on and eq<float(e['safe_power_15m_min'])-1e-10:
                violations.append(f'hour{t}_quarter{q+1}:unsafe_equipment_power')
            if eq>float(e['safe_power_15m_max'])+1e-10:
                violations.append(f'hour{t}_quarter{q+1}:unsafe_equipment_power')
            if abs(eq-equipment_prev)>float(e['max_power_ramp_per_15m'])+1e-10:
                violations.append(f'hour{t}_quarter{q+1}:power_ramp')
            equipment_prev=eq
            loads.append(float(fixed[4*t+q])+eq)
        for key,m in materials.items():
            change=e['material_delta_by_product'][product].get(key,0.0) if on else 0.0
            inventory[key]+=float(m['inflow_hourly'][t])-float(m['external_use_hourly'][t])+float(change)
            trajectory[key].append(inventory[key])
            if inventory[key]<-1e-10 or inventory[key]>float(m['capacity'])+1e-10:
                violations.append(f'hour{t}:inventory_{key}_out_of_range')
        prev=on
    if prev:
        if run<int(e['min_on_hours']): violations.append('horizon:minimum_on_time')
        if e['mode']=='batch' and run!=int(e['batch_hours']):
            violations.append('horizon:incomplete_batch')
    if starts>int(e['max_starts']): violations.append('horizon:max_starts')
    if on_hours!=int(e['required_on_hours']): violations.append('horizon:required_on_hours')
    for key,m in materials.items():
        if inventory[key]<float(m['min_final'])-1e-10:
            violations.append(f'horizon:{key}_final_inventory')
    return {'status':'FEASIBLE' if not violations else 'BLOCKED',
            'violations':sorted(set(violations)),
            'quarter_hour_load':loads,'max_quarter_hour_load':max(loads),
            'total_quarter_hour_load':sum(loads),'inventory_trace':trajectory,
            'starts':starts,'on_hours':on_hours}

def compare(case,original,candidate):
    first=evaluate(case,original);second=evaluate(case,candidate)
    result={'original':first,'candidate':second,'decision':'NO_RECOMMENDATION'}
    if first['status']=='FEASIBLE' and second['status']=='FEASIBLE':
        result['peak_change']=second['max_quarter_hour_load']-first['max_quarter_hour_load']
        result['load_sum_change']=second['total_quarter_hour_load']-first['total_quarter_hour_load']
        if result['peak_change']<-1e-10:
            result['decision']='DEMO_PEAK_REDUCTION_ONLY'
    return result

# 실제 제공 데이터에는 설비별 부하·제품·재고·물질수지·안전 한계가 없다.
actual_template={'horizon_hours':None,'fixed_background_15m':None,'power_unit':None,
                 'background_excludes_equipment':None,'profile_source_verified':None,
                 'materials':None,
                 'equipment':{k:None for k in ['id','mode','power_profile_15m',
                 'startup_extra_first_15m','safe_power_15m_min','safe_power_15m_max',
                 'max_power_ramp_per_15m','material_delta_by_product','allowed_products',
                 'changeover_off_hours','min_on_hours','min_off_hours','batch_hours',
                 'required_on_hours','max_starts','availability_hourly','prior_state',
                 'safety_approved','quality_approved']}}
actual=evaluate(actual_template,[None]*4)
assert actual['status']=='DATA_REQUIRED' and actual['missing_fields']

# 아래 숫자는 모두 가상 예시이며 KAMP 공장·전력 단위·절감액이 아니다.
demo={'horizon_hours':4,'fixed_background_15m':[3.0]*4+[8.0]*4+[2.0]*4+[4.0]*4,
      'power_unit':'synthetic_units','background_excludes_equipment':True,
      'profile_source_verified':True,
      'materials':{
        'RAW':{'initial':10,'capacity':20,'inflow_hourly':[0]*4,
               'external_use_hourly':[0]*4,'min_final':0},
        'FG':{'initial':2,'capacity':10,'inflow_hourly':[0]*4,
              'external_use_hourly':[1]*4,'min_final':2}},
      'equipment':{'id':'SYNTHETIC_E1','mode':'batch','power_profile_15m':[5.0]*4,
        'startup_extra_first_15m':2.0,'safe_power_15m_min':4.0,
        'safe_power_15m_max':10.0,
        'max_power_ramp_per_15m':10.0,
        'material_delta_by_product':{'A':{'RAW':-2.0,'FG':2.0}},
        'allowed_products':['A'],'changeover_off_hours':1,
        'min_on_hours':2,'min_off_hours':1,'batch_hours':2,'required_on_hours':2,
        'max_starts':1,'availability_hourly':[True]*4,
        'prior_state':{'on':False,'duration_hours':3,'last_product':None},
        'safety_approved':True,'quality_approved':True}}
original=['A','A',None,None];shifted=[None,None,'A','A']
good=compare(demo,original,shifted)
assert (good['original']['status']=='FEASIBLE' and good['candidate']['status']=='FEASIBLE'
        and good['original']['max_quarter_hour_load']==13
        and good['candidate']['max_quarter_hour_load']==9
        and good['decision']=='DEMO_PEAK_REDUCTION_ONLY')
short=evaluate(demo,[None,'A',None,None])
assert short['status']=='BLOCKED' and any('minimum_on_time' in v for v in short['violations'])
minimum_off=copy.deepcopy(demo);minimum_off['equipment']['prior_state']['duration_hours']=0
off_result=evaluate(minimum_off,original)
assert off_result['status']=='BLOCKED' and any('minimum_off_time' in v for v in off_result['violations'])
starved=copy.deepcopy(demo);starved['materials']['FG']['initial']=0
starved_result=evaluate(starved,shifted)
assert starved_result['status']=='BLOCKED' and any('inventory_FG' in v for v in starved_result['violations'])
switch=copy.deepcopy(demo);switch['equipment'].update({
    'mode':'continuous','min_on_hours':1,'batch_hours':1,'allowed_products':['A','B'],
    'material_delta_by_product':{'A':{'RAW':-2,'FG':2},'B':{'RAW':-2,'FG':2}}})
switched=evaluate(switch,['A','B',None,None])
assert switched['status']=='BLOCKED' and any('product_changeover' in v for v in switched['violations'])
unavailable=copy.deepcopy(demo);unavailable['equipment']['availability_hourly'][2]=False
blocked=evaluate(unavailable,shifted)
assert blocked['status']=='BLOCKED' and any('equipment_unavailable' in v for v in blocked['violations'])
unverified=copy.deepcopy(demo);unverified['background_excludes_equipment']=False
unverified_result=evaluate(unverified,shifted)
assert unverified_result['status']=='BLOCKED' and 'unverified_load_decomposition' in unverified_result['violations']
unsafe_start=copy.deepcopy(demo);unsafe_start['equipment']['startup_extra_first_15m']=8
unsafe_result=evaluate(unsafe_start,shifted)
assert unsafe_result['status']=='BLOCKED' and any('unsafe_equipment_power' in v for v in unsafe_result['violations'])
unapproved=copy.deepcopy(demo);unapproved['equipment']['quality_approved']=False
approval_result=evaluate(unapproved,shifted)
assert approval_result['status']=='BLOCKED' and 'safety_or_quality_approval' in approval_result['violations']
rebound=copy.deepcopy(demo);rebound['fixed_background_15m']=[3.0]*4+[1.0]*4+[20.0]*4+[4.0]*4
rebound_result=compare(rebound,original,shifted)
assert rebound_result['decision']=='NO_RECOMMENDATION' and rebound_result['peak_change']>0

