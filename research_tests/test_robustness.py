"""Offline Step 4 tests. Synthetic fixtures are NOT empirical GPU results."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import pytest
from offload_research.fresh import make_protocol as parent_protocol
from offload_research import robustness as r
from offload_research import collect_robustness as c


def test_design_counts():
    p=r.make_protocol(parent_protocol())
    assert len(p['cases'])==13
    assert sum(x['kind']=='primary' for x in p['cases'])==10
    assert len(p['cases'])*len(p['blocks'])*p['measured_repetitions_per_case_per_block']==260
    assert p['threads_by_block']==[2,1,1,2]
    assert p['expected_primary_policy_decisions']==96


def test_orders_reversed_within_threads():
    p=r.make_protocol(parent_protocol())
    assert p['blocks'][0]==p['blocks'][3][::-1]
    assert p['blocks'][1]==p['blocks'][2][::-1]
    assert all(len(set(x))==13 for x in p['blocks'])
    assert r.make_protocol(parent_protocol())['blocks']==p['blocks']


def test_parent_not_mutated():
    p=parent_protocol();original=copy.deepcopy(p);r.make_protocol(p)
    assert p==original


def test_frozen_baselines_not_tuned():
    p=r.make_protocol(parent_protocol())
    assert p['memory_guard']=={'relative':.05,'extra_mib':16.0}
    assert p['candidates']==[0,3,6,9,12]
    assert p['budgets_mib']==[256,320,384,448,512,640,768,1024]
    assert p['new_session_required'] and not p['physical_memory_cap_enforced']
    assert 'post' in p['future_test_status'].lower()


def runtime(threads=2,boot='new'):
    return dict(cuda_available=True,gpu_name='Tesla T4',torch_threads=threads,
                torch_interop_threads=1,boot_id=boot)


@pytest.mark.parametrize('threads',[1,2])
def test_runtime_threads(threads):c.require_runtime(runtime(threads),threads)


@pytest.mark.parametrize('field,value',[('cuda_available',False),('gpu_name','A100'),
    ('torch_threads',4),('torch_interop_threads',2),('boot_id',None)])
def test_runtime_rejects_invalid(field,value):
    cur=runtime();cur[field]=value
    with pytest.raises(ValueError):c.require_runtime(cur,2)


def test_same_vm_rejected():
    with pytest.raises(ValueError,match='same VM'):c.different_vm(runtime(),runtime())
    c.different_vm(runtime(boot='new'),runtime(boot='old'))


@pytest.mark.parametrize('field,value',[('boot_id',None),('boot_id','')])
def test_unknown_vm_rejected(field,value):
    cur=runtime();cur[field]=value
    with pytest.raises(ValueError,match='Missing'):c.different_vm(cur,runtime())


def test_counter_snapshot_is_read_only_data():
    s=c.cpu_snapshot()
    assert s['process_user_seconds']>=0 and isinstance(s['accessible_cgroups'],dict)
    assert 'not only measured' in s['scope']


@pytest.mark.parametrize('path',['../outside','/outside','a/../../outside','a\\evil'])
def test_unsafe_zip_rejected(tmp_path,path):
    z=tmp_path/'bad.zip'
    with zipfile.ZipFile(z,'w') as f:f.writestr(path,b'x')
    with pytest.raises(ValueError,match='Unsafe'):r.safe_extract(z,tmp_path/'dest')
    assert not (tmp_path/'dest').exists()


def test_archive_duplicate_rejected(tmp_path):
    z=tmp_path/'bad.zip'
    with zipfile.ZipFile(z,'w') as f:
        f.writestr('same',b'a')
        with pytest.warns(UserWarning):f.writestr('same',b'b')
    with pytest.raises(ValueError,match='Duplicate'):r.safe_extract(z,tmp_path/'dest')


def test_safe_zip(tmp_path):
    z=tmp_path/'ok.zip'
    with zipfile.ZipFile(z,'w') as f:f.writestr('a/b.txt',b'data')
    r.safe_extract(z,tmp_path/'dest')
    assert (tmp_path/'dest/a/b.txt').read_bytes()==b'data'


def test_wrong_input_rejected(tmp_path):
    z=tmp_path/'wrong.zip';z.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='unchanged'):r.prepare(z,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def synthetic_tables():
    casebase=dict(workload_id='B3_S64_N32',kind='primary',batch_size=3,sequence_length=64,
                  new_tokens=32,layout='prefix')
    preds=[];obs=[]
    for k,m,t in zip([0,3,6,9,12],[0,260,350,440,530],[100,75,50,30,20]):
        base=dict(casebase,case_id=f'B3_S64_N32_G{k}',gpu_layers=k)
        row=dict(base,predicted_peak_mib=m,guarded_peak_mib=0 if k==0 else m*1.05+16)
        observed=dict(base,gpu_peak_mib=m,generation_ms_median=t)
        for metric in r.METRICS:
            observed[metric+'_median']=float(t)
            for variant in r.VARIANTS:row[f'predicted_{metric}_{variant}']=float(t)
        preds.append(row);obs.append(observed)
    return pd.DataFrame(preds),pd.DataFrame(obs)


def test_sensitivity_does_not_mutate_input():
    pred,obs=synthetic_tables();old=pred.copy(deep=True)
    scores=r.margin_sensitivity(pred,obs,[540],['max_gpu'])
    pd.testing.assert_frame_equal(pred,old)
    assert len(scores)==4 and (scores.scope.str.contains('exploratory')).all()
    assert scores.loc[scores.guard=='point_only','selected_gpu_layers'].iloc[0]==12
    original=scores[scores.guard=='original_5_percent_plus_16_MiB'].iloc[0]
    assert original.selected_gpu_layers==9 and original.regret_pct==pytest.approx(50)


def test_score_does_not_reselect_frozen_choice():
    pred,obs=synthetic_tables()
    d=pd.DataFrame([dict(workload_id='B3_S64_N32',budget_mib=600,policy='compute',
                        selected_gpu_layers=3,selection_status='predicted_feasible')])
    errors,metrics,scores,memory=r.score_observations(pred,d,obs)
    assert scores.iloc[0].selected_gpu_layers==3
    assert scores.iloc[0].regret_pct==pytest.approx(275)
    assert memory['gpu_positive_cases']==4 and memory['median_ape_pct']==0
    assert (metrics.median_ape_pct==0).all()


def test_scoring_budget_violation_not_rewarded():
    pred,obs=synthetic_tables()
    d=pd.DataFrame([dict(workload_id='B3_S64_N32',budget_mib=400,policy='max_gpu',
                        selected_gpu_layers=12,selection_status='predicted_feasible')])
    _,_,scores,_=r.score_observations(pred,d,obs)
    assert scores.iloc[0].observed_budget_violation and pd.isna(scores.iloc[0].regret_pct)


def fixture_trials():
    case=dict(case_id='B1_S64_N32_G6',workload_id='B1_S64_N32',kind='primary',
              batch_size=1,sequence_length=64,new_tokens=32,gpu_layers=6,layout='prefix')
    rows=[]
    for trial in range(5):
        rows.append(dict(case,block=0,trial=trial,prefill_ms=10.,ttft_ms=10.,decode_ms=31.,
            decode_tpot_ms=1.,generation_ms=41.,final_cached_positions=95,
            final_kv_gpu_bytes=6*2*95*768*4,final_kv_cpu_bytes=6*2*95*768*4,
            decode_step_ms_json=json.dumps([1.]*31),prefill_input_tokens_per_s=6400.,
            decode_generated_tokens_per_s=1000.,generation_generated_tokens_per_s=32000/41))
    return case,pd.DataFrame(rows)


def test_valid_raw_accounting():
    case,t=fixture_trials();r.validate_trial_table(case,t,0,5)


@pytest.mark.parametrize('column,value',[('trial',9),('block',1),('case_id','wrong'),
    ('batch_size',3),('gpu_layers',9),('final_cached_positions',96),('final_kv_cpu_bytes',0),
    ('final_kv_gpu_bytes',0),('decode_step_ms_json','[1]'),
    ('generation_generated_tokens_per_s',1),('prefill_input_tokens_per_s',0)])
def test_corrupt_raw_rejected(column,value):
    case,t=fixture_trials();t.loc[0,column]=value
    with pytest.raises(ValueError):r.validate_trial_table(case,t,0,5)


@pytest.mark.parametrize('values',[[],[0],[-1],[float('nan')],[float('inf')]])
def test_bad_distribution_rejected(values):
    with pytest.raises(ValueError):r.distribution(values)


def test_slow_trials_kept():
    d=r.distribution([1,1,1,1,10])
    assert d['slow_trials_over_1_5x_median']==1
    assert d['generation_max_over_median']==10


def test_complete_analysis_with_explicit_synthetic_fixtures(tmp_path,monkeypatch):
    """Exercise analysis plumbing, NOT a GPU performance experiment."""
    p=r.make_protocol(parent_protocol())
    out=tmp_path/'SYNTHETIC_ONLY';out.mkdir()
    r.write_json(out/'frozen/protocol.json',p)
    sealed=dict(status='verified',seal_sha256='SYNTHETIC')
    monkeypatch.setattr(r,'verify',lambda _:sealed)
    monkeypatch.setattr(c,'load_reference_gate',lambda *args:{'synthetic':True})
    monkeypatch.setattr(c,'check_saved_runtime',lambda *args:None)
    r.write_json(out/'runtime_preflight.json',dict(runtime={'boot_id':'synthetic_new'},parent_boot_id='synthetic_old'))
    for threads in (1,2):r.write_json(out/f'reference_threads_{threads}.json',{'synthetic':True})
    preds=[];parentobs=[]
    for case in p['cases']:
        k=case['gpu_layers'];scale=float(13-k)
        pred=dict(case,predicted_peak_mib=k*40.,guarded_peak_mib=0 if k==0 else k*42.+16)
        for metric,timing in zip(r.METRICS,[10.,10.,1.,41.]):
            for v in r.VARIANTS:pred[f'predicted_{metric}_{v}']=timing*scale
        preds.append(pred)
        parentobs.append(dict(case,gpu_peak_mib=k*40.,generation_ms_median=41.*scale))
        for block,threads in enumerate(p['threads_by_block']):
            timing_scale=scale*threads/2
            rows=[]
            positions=case['sequence_length']+31
            kv=2*case['batch_size']*positions*768*4
            for trial in range(5):
                rows.append(dict(case,block=block,trial=trial,prefill_ms=10*timing_scale,
                    ttft_ms=10*timing_scale,decode_ms=31*timing_scale,decode_tpot_ms=timing_scale,
                    generation_ms=41*timing_scale,final_cached_positions=positions,
                    final_kv_cpu_bytes=(12-k)*kv,final_kv_gpu_bytes=k*kv,
                    decode_step_ms_json=json.dumps([timing_scale]*31),
                    prefill_input_tokens_per_s=case['batch_size']*case['sequence_length']*1000/(10*timing_scale),
                    decode_generated_tokens_per_s=case['batch_size']*1000/timing_scale,
                    generation_generated_tokens_per_s=case['batch_size']*32000/(41*timing_scale)))
            folder=out/f'measurement/block_{block}'/case['case_id'];folder.mkdir(parents=True)
            t=pd.DataFrame(rows);t.to_csv(folder/'trials.csv',index=False)
            s=dict(case,block=block,status='ok',seal_sha256='SYNTHETIC',gpu_peak_allocated_bytes=k*40*2**20)
            s.update({m+'_median':float(t[m].median()) for m in r.METRICS})
            r.write_json(folder/'summary.json',s)
    for block,threads in enumerate(p['threads_by_block']):
        r.write_json(out/f'measurement/block_{block}/runtime.json',dict(runtime={'synthetic':True},
            seal_sha256='SYNTHETIC',reference_sha256=r.sha((out/f'reference_threads_{threads}.json').read_bytes())))
        r.write_json(out/f'measurement/block_{block}/completed.json',dict(status='completed',successful_cases=13,seal_sha256='SYNTHETIC'))
    pred=pd.DataFrame(preds);pred.to_csv(out/'frozen/predictions.csv',index=False)
    (out/'parent/analysis').mkdir(parents=True)
    pd.DataFrame(parentobs).to_csv(out/'parent/analysis/observed_cases.csv',index=False)
    choices=[]
    for wid,g in pred[pred.kind=='primary'].groupby('workload_id'):
        for budget in p['budgets_mib']:
            for policy in p['policies']:
                picked=r.choose(g,budget,policy)
                choices.append(dict(workload_id=wid,budget_mib=budget,policy=policy,
                    selected_gpu_layers=int(picked['gpu_layers']),selection_status='predicted_feasible'))
    pd.DataFrame(choices).to_csv(out/'frozen/decisions.csv',index=False)
    result=r.analyze(out)
    assert result['complete'] and result['measured_generation_trials']==260
    comparison=pd.read_csv(out/'analysis/thread_comparison.csv')
    assert np.allclose(comparison.generation_ratio_one_vs_two,.5)
    assert len(pd.read_csv(out/'analysis/policy_scores.csv'))==96
    assert len(pd.read_csv(out/'analysis/observed_cases.csv'))==26
    # A corrupt record must fail instead of silently completing analysis.
    sample=out/'measurement/block_0'/p['cases'][0]['case_id']/'trials.csv'
    t=pd.read_csv(sample);t.loc[0,'final_cached_positions']+=1;t.to_csv(sample,index=False)
    with pytest.raises(ValueError,match='Cache'):r.analyze(out)
    assert r.read_json(out/'analysis/coverage.json')['complete'] is False
    assert not (out/'analysis/summary.json').exists()


def test_partial_export_not_claimed_complete(tmp_path,monkeypatch):
    output=tmp_path/'partial';output.mkdir()
    (output/'inputs').mkdir();(output/'inputs/step3_export.zip').write_bytes(b'fixture')
    (output/'parent').mkdir();(output/'parent/not_exported.txt').write_text('working copy')
    monkeypatch.setattr(r,'verify',lambda _:dict(status='verified'))
    target=r.export(output)
    with zipfile.ZipFile(target) as z:
        status=json.loads(z.read('export_status.json'))
        assert status['complete_analysis'] is False
        assert not any(n.startswith('parent/') for n in z.namelist())
        assert 'inputs/step3_export.zip' in z.namelist()


def test_partial_analysis_reports_missing_cases(tmp_path,monkeypatch):
    out=tmp_path/'partial';out.mkdir()
    r.write_json(out/'frozen/protocol.json',r.make_protocol(parent_protocol()))
    monkeypatch.setattr(r,'verify',lambda _:dict(status='verified',seal_sha256='SYNTHETIC'))
    summary=r.analyze(out)
    assert summary['complete'] is False and len(summary['failures'])==52
    assert not (out/'analysis/summary.json').exists()
