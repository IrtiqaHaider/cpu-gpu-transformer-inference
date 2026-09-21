from __future__ import annotations
import copy
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from offload_research.cost_model import Workload, fit_model, CostModel
from offload_research.memory_model import (MIB, components, features, fit_memory,
    MemoryModel, MemoryGuard, predict_memory_frame, FEATURE_NAMES)
from offload_research.planner import PlacementPlanner, choose, validate_candidates, CANDIDATES
from offload_research.step2 import read_members, score_selection, policy_stats

CONFIG={"vocab_size":50257,"n_positions":1024,"n_embd":768,"n_layer":12,"n_head":12,"n_inner":None}
META={"model_config":CONFIG,"model":{"model_id":"openai-community/gpt2","revision":"a"*40}}

def frame():
    records=[]
    for b in (1,2,4):
        for s in (32,128,512):
            for k in CANDIDATES:
                w=Workload(b,s,32,k)
                c=components(w,CONFIG)
                overhead=features(w,CONFIG)@np.array([8,2,0.5,1,1])
                pre=(13-k)*b*s/8+1
                dec=(13-k)*b+3
                records.append({**w.__dict__,"case_index":len(records),
                    "gpu_peak_allocated_bytes":c['payload_floor_bytes']+int(overhead*MIB),
                    "prefill_ms_median":pre,"ttft_ms_median":pre+1,
                    "decode_tpot_ms_median":dec,"generation_ms_median":pre+1+31*dec})
    return pd.DataFrame(records)

@pytest.fixture
def models():
    d=frame()
    mem=MemoryModel(fit_memory(d,META))
    lat={v:CostModel(fit_model(d,META,v)) for v in ("compute","compute_transfer")}
    return mem,lat

@pytest.mark.parametrize('k,parameters,buffers',[(0,0,0),(3,242595840,3145728),
    (6,327650304,6291456),(9,412704768,9437184),(12,497759232,12582912)])
def test_exact_resident_counts(k,parameters,buffers):
    c=components(Workload(1,128,32,k),CONFIG)
    assert c['gpu_parameter_bytes']==parameters
    assert c['gpu_buffer_bytes']==buffers
    assert c['gpu_parameter_bytes']+c['cpu_parameter_bytes']==497759232

@pytest.mark.parametrize('b,s,n,k',[(1,128,32,6),(4,512,32,12),(3,64,16,3),(1,32,64,0)])
def test_cache_accounting(b,s,n,k):
    c=components(Workload(b,s,n,k),CONFIG)
    assert c['final_cached_positions']==s+n-1
    assert c['final_kv_gpu_bytes']==2*k*b*(s+n-1)*768*4

def test_cpu_memory_zero(models):
    mem,_=models
    p=mem.predict(batch_size=4,sequence_length=512,new_tokens=32,gpu_layers=0)
    assert p['predicted_peak_mib']==0
    assert MemoryGuard().apply(p['predicted_peak_mib'],0)==0

def test_memory_at_least_floor(models):
    mem,_=models
    p=predict_memory_frame(mem.model,frame())
    assert (p.predicted_peak_mib>=p.payload_floor_bytes/MIB).all()

def test_nonnegative_memory_fit(models):
    mem,_=models
    assert np.isfinite(mem.model['coefficients_mib']).all()
    assert min(mem.model['coefficients_mib'])>=0

def test_memory_serialization(tmp_path,models):
    mem,_=models
    mem.save(tmp_path/'m.json')
    loaded=MemoryModel.load(tmp_path/'m.json')
    assert loaded.model==mem.model

@pytest.mark.parametrize('updates',[{'n_embd':1024},{'n_head':16},{'n_inner':2048},{'vocab_size':10}])
def test_reject_other_architecture(updates):
    with pytest.raises(ValueError):components(Workload(1,32,32,6),{**CONFIG,**updates})

@pytest.mark.parametrize('bad',[{'relative':-1},{'extra_mib':-1},{'relative':float('nan')},{'extra_mib':float('inf')}])
def test_guard_reject_invalid(bad):
    with pytest.raises(ValueError):MemoryGuard(**bad)

def test_fixed_guard_formula():
    assert MemoryGuard().apply(400,6)==436

@pytest.mark.parametrize('params',[{'batch_size':8},{'sequence_length':16},{'new_tokens':16},{'gpu_layers':5},{'layout':'suffix'}])
def test_memory_support(models,params):
    mem,_=models
    kw=dict(batch_size=1,sequence_length=128,new_tokens=32,gpu_layers=6)
    with pytest.raises(ValueError):mem.predict(**{**kw,**params})

def test_memory_extrapolation_is_explicit(models):
    mem,_=models
    with pytest.warns(UserWarning,match='extrapolation'):
        p=mem.predict(batch_size=3,sequence_length=64,new_tokens=16,gpu_layers=6,allow_extrapolation=True)
    assert p['extrapolation'] is True and p['measured'] is False

@pytest.mark.parametrize('candidates',[[],[0,0],[5],[True],[3.0],[-1],[13]])
def test_candidate_validation(candidates):
    with pytest.raises(ValueError):validate_candidates(candidates)

def test_default_candidates():
    assert validate_candidates(reversed(CANDIDATES))==CANDIDATES

def prediction_table():
    return pd.DataFrame({'gpu_layers':[0,3,6], 'guarded_peak_mib':[0.,300.,400.],
                         'predicted_generation_ms_compute':[30.,20.,25.],
                         'predicted_generation_ms_compute_transfer':[30.,20.,25.]})

def test_model_can_differ_from_max_gpu_without_actuals():
    table=prediction_table()
    assert choose(table,450,'max_gpu')['gpu_layers']==6
    assert choose(table,450,'compute')['gpu_layers']==3

def test_shared_budget_filter():
    table=prediction_table()
    for policy in ('max_gpu','compute','compute_transfer'):
        assert choose(table,350,policy)['gpu_layers']==3

def test_zero_budget_cpu_fallback():
    assert choose(prediction_table(),0,'compute')['gpu_layers']==0

def test_no_feasible_returns_none():
    assert choose(prediction_table().iloc[1:],0,'compute') is None

@pytest.mark.parametrize('budget',[float('nan'),float('inf'),-1,True])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):choose(prediction_table(),budget,'max_gpu')

def test_budget_boundary_inclusive():
    assert choose(prediction_table(),400,'max_gpu')['gpu_layers']==6

def test_actual_columns_cannot_change_selection():
    table=prediction_table()
    orig=choose(table,500,'compute')
    table['actual_generation_ms']=[1,1e9,1]
    table['gpu_peak_allocated_bytes']=[1e9,1e9,0]
    assert choose(table,500,'compute')['gpu_layers']==orig['gpu_layers']

def test_predictor_tie_break():
    table=prediction_table()
    table['predicted_generation_ms_compute']=1
    assert choose(table,500,'compute')['gpu_layers']==0

def test_planner_schema(models):
    mem,lat=models
    planner=PlacementPlanner(mem,lat['compute'],lat['compute_transfer'])
    p=planner.recommend(batch_size=3,sequence_length=64,new_tokens=32,gpu_budget_mib=400)
    assert p['status']=='predicted_feasible'
    assert p['measured'] is False
    assert p['selected']['guarded_peak_mib']<=400

def test_checkpoint_mismatch(models):
    mem,lat=models
    m=copy.deepcopy(mem.model)
    m['model_source']['revision']='b'*40
    with pytest.raises(ValueError,match='different'):
        PlacementPlanner(MemoryModel(m),lat['compute'],lat['compute_transfer'])

def test_input_hash_mismatch(models):
    mem,lat=models
    mem.model['calibration_input_sha256']='a'
    lat['compute'].model['calibration_input_sha256']='b'
    with pytest.raises(ValueError,match='calibration'):
        PlacementPlanner(mem,lat['compute'],lat['compute_transfer'])

def test_memory_training_only_scaling():
    d=frame()
    train=d[d.batch_size==1]
    m=fit_memory(train,META)
    x=np.stack([features(Workload(**r),CONFIG) for r in train[['batch_size','sequence_length','new_tokens','gpu_layers','layout']].to_dict('records')])
    np.testing.assert_allclose(m['feature_scales'],np.maximum(x.max(0),1))
    assert m['domain']['batch_size_range']==[1,1]

def actual_table():
    return pd.DataFrame({'gpu_layers':[0,3,6], 'gpu_peak_allocated_bytes':np.array([0,290,410])*MIB,
                         'generation_ms_median':[100.,50.,25.]})

def test_infeasible_choice_not_given_good_regret():
    r=score_selection({'gpu_layers':6},actual_table(),400)
    assert r['status']=='budget_violation'
    assert r['regret_pct'] is None
    assert r['budget_excess_mib']==10
    assert r['oracle_gpu_layers']==3

def test_feasible_regret():
    r=score_selection({'gpu_layers':0},actual_table(),400)
    assert r['regret_pct']==100
    assert not r['matches_oracle']

def test_oracle_same_candidate_set():
    r=score_selection({'gpu_layers':3},actual_table(),400)
    assert r['regret_pct']==0 and r['matches_oracle']

def test_abstention_explicit():
    r=score_selection(None,actual_table(),400)
    assert r['status']=='abstain' and r['regret_pct'] is None

def test_excluded_candidate_not_oracle():
    with pytest.raises(ValueError,match='not measured'):
        score_selection({'gpu_layers':12},actual_table(),400)

def test_missing_zip_member(tmp_path):
    p=tmp_path/'a.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('a.json','{}')
    with pytest.raises(ValueError,match='Missing'):
        read_members(p,['b.json'])

def test_zip_does_not_extract_unrequested_members(tmp_path):
    p=tmp_path/'a.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('a.json','{}')
        z.writestr('../evil.py','raise RuntimeError()')
    assert read_members(p,['a.json'])=={'a.json':b'{}'}
    assert not (tmp_path.parent/'evil.py').exists()

@pytest.mark.parametrize('key,value',[('coefficients_mib',[float('nan')]*5),('feature_scales',[0]*5),
                                      ('coefficients_mib',[-1]*5),('feature_names',['bad'])])
def test_invalid_saved_memory_rejected(models,key,value):
    m=copy.deepcopy(models[0].model)
    m[key]=value
    with pytest.raises(ValueError):MemoryModel(m)

def test_policy_summary_does_not_drop_failure():
    rows=[{**score_selection({'gpu_layers':3},actual_table(),400),'policy':'compute'},
          {**score_selection({'gpu_layers':6},actual_table(),400),'policy':'compute'}]
    r=policy_stats(pd.DataFrame(rows))['compute']
    assert r['decisions']==2 and r['budget_violations']==1 and r['feasible']==1
