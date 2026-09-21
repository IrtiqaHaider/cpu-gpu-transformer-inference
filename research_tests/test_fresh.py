"""Offline protocol, integrity, scoring, and collector tests; no pretrained downloads."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
import pytest
from offload_research import fresh
from offload_research.fresh import (make_protocol, evaluate_policy_decisions, aggregate_case,
    environment_differences, require_t4, prepare, verify, read_json, write_json, sha)
from offload_research.collect_fresh import (compare_teacher_forced, collect_case,
    make_tokens, load_reference_gate, session_fingerprint)


def observations():
    return pd.DataFrame([dict(workload_id='test',gpu_layers=k,gpu_peak_mib=m,
                             generation_ms_median=t)
                         for k,m,t in zip((0,3,6,9,12),(0,260,350,440,530),(100,75,50,30,20))])


def decision(k=6,budget=400):
    return pd.DataFrame([dict(workload_id='test',budget_mib=budget,policy='compute',
                             selected_gpu_layers=k,selection_status='predicted_feasible')])


def test_protocol_counts():
    p=make_protocol()
    assert len(p['cases'])==23
    assert sum(c['kind']=='primary' for c in p['cases'])==20
    assert len(p['blocks'])==2 and all(len(x)==23 for x in p['blocks'])
    assert len(p['cases'])*len(p['blocks'])*p['measured_repetitions_per_case_per_block']==230


def test_reversed_order():
    p=make_protocol()
    assert p['blocks'][0]==list(reversed(p['blocks'][1]))
    assert len(set(p['blocks'][0]))==23


def test_order_reproducible():
    assert make_protocol()['blocks']==make_protocol()['blocks']


def test_primary_shapes_new():
    historical={(b,s,32) for b in (1,2,4) for s in (32,128,512)}
    for c in make_protocol()['cases']:
        if c['kind']=='primary':
            assert (c['batch_size'],c['sequence_length'],c['new_tokens']) not in historical


def test_controls_explicitly_separate():
    p=make_protocol();controls=[c for c in p['cases'] if c['kind']=='historical_control']
    assert {c['gpu_layers'] for c in controls}=={0,6,12}
    assert all((c['batch_size'],c['sequence_length'],c['new_tokens'])==(1,128,32) for c in controls)


def test_frozen_guard_and_budgets_unchanged():
    p=make_protocol()
    assert p['memory_guard']=={'relative':.05,'extra_mib':16.0}
    assert p['budgets_mib']==[256,320,384,448,512,640,768,1024]
    assert p['primary_workloads']*len(p['budgets_mib'])*len(p['policies'])==96


def test_choices_not_reselected_using_observed_data():
    scored=evaluate_policy_decisions(decision(3,600),observations()).iloc[0]
    assert scored.selected_gpu_layers==3 and scored.oracle_gpu_layers==12
    assert scored.regret_pct==pytest.approx(275)


def test_violation_not_rewarded():
    scored=evaluate_policy_decisions(decision(12,400),observations()).iloc[0]
    assert scored.observed_budget_violation
    assert scored.regret_pct is None or pd.isna(scored.regret_pct)
    assert scored.outcome=='budget_violation'


def test_correct_oracle():
    scored=evaluate_policy_decisions(decision(6,400),observations()).iloc[0]
    assert scored.oracle_gpu_layers==6 and scored.regret_pct==0


def test_abstention_recorded():
    scored=evaluate_policy_decisions(decision(None,400),observations()).iloc[0]
    assert scored.outcome=='abstained' and not scored.observed_budget_violation


def test_boundary_budget_inclusive():
    scored=evaluate_policy_decisions(decision(6,350),observations()).iloc[0]
    assert scored.outcome=='feasible'


@pytest.mark.parametrize('removed',[0,1,2,3,4])
def test_missing_candidate_rejected(removed):
    with pytest.raises(ValueError,match='incomplete'):
        evaluate_policy_decisions(decision(),observations().drop(index=removed))


def test_duplicate_candidate_rejected():
    with pytest.raises(ValueError):
        evaluate_policy_decisions(decision(),pd.concat([observations(),observations().iloc[[0]]]))


def test_nonfinite_latency_rejected():
    o=observations();o.loc[0,'generation_ms_median']=np.nan
    with pytest.raises(ValueError):evaluate_policy_decisions(decision(),o)


def valid_runtime():
    return dict(cuda_available=True,gpu_name='Tesla T4',torch_threads=2,torch_interop_threads=1)


def test_t4_gate_passes():require_t4(valid_runtime())


@pytest.mark.parametrize('gpu',['NVIDIA L4','NVIDIA A100','RTX 4090','NotT4'])
def test_wrong_gpu_rejected(gpu):
    r=valid_runtime();r['gpu_name']=gpu
    with pytest.raises(ValueError):require_t4(r)


@pytest.mark.parametrize('key,value',[('cuda_available',False),('torch_threads',1),('torch_interop_threads',2)])
def test_incorrect_hardware_settings_rejected(key,value):
    r=valid_runtime();r[key]=value
    with pytest.raises(ValueError):require_t4(r)


def test_environment_changes_not_hidden():
    a=valid_runtime();b=copy.deepcopy(a);b['packages']={'torch':'other'}
    assert any(x['field']=='packages.torch' for x in environment_differences(b,a))


def test_load_change_not_new_session():
    a={'boot_id':'a','load_average':[0,0,0]};b={**a,'load_average':[1,2,3]}
    assert session_fingerprint(a)==session_fingerprint(b)
    assert session_fingerprint(a)!=session_fingerprint({**b,'boot_id':'b'})


def raw_trials():
    return pd.DataFrame([dict(block=b,trial=t,prefill_ms=x,ttft_ms=x,generation_ms=x+9,
                              decode_ms=9,decode_tpot_ms=3)
                         for b,values in enumerate(([1,2,100],[3,4,5])) for t,x in enumerate(values)])


def test_aggregate_pools_raw_trials_not_medians():
    c=dict(case_id='test',new_tokens=4,batch_size=1)
    r=aggregate_case(c,[{'gpu_peak_allocated_bytes':10},{'gpu_peak_allocated_bytes':20}],raw_trials(),6)
    assert r['generation_ms_median']==pytest.approx(12.5)
    assert r['gpu_peak_mib']==20/2**20
    assert r['generation_ms_max']==109 # Retain slow trial.
    assert r['pooled_generation_tokens_s']==pytest.approx(6*4*1000/raw_trials().generation_ms.sum())


def test_bad_accounting_rejected():
    x=raw_trials();x.loc[0,'generation_ms']=100
    with pytest.raises(ValueError,match='accounting'):
        aggregate_case(dict(case_id='t',new_tokens=4,batch_size=1),[{'gpu_peak_allocated_bytes':0}],x,6)


def test_bad_trial_count_rejected():
    with pytest.raises(ValueError,match='trial count'):
        aggregate_case(dict(case_id='t',new_tokens=4,batch_size=1),[{'gpu_peak_allocated_bytes':0}],raw_trials(),10)


def test_duplicate_trial_rejected():
    x=raw_trials();x.loc[1,['block','trial']]=[0,0]
    with pytest.raises(ValueError,match='Duplicate'):
        aggregate_case(dict(case_id='t',new_tokens=4,batch_size=1),[{'gpu_peak_allocated_bytes':0}],x,6)


def test_identical_tokens_across_runs():
    import torch
    assert torch.equal(make_tokens(3,64,123,100),make_tokens(3,64,123,100))


def test_checkpoint_rejects_wrong_zip(tmp_path):
    p=tmp_path/'wrong.zip';p.write_bytes(b'not the checkpoint')
    with pytest.raises(ValueError,match='Wrong Step 2'):
        prepare(p,tmp_path/'output')


@pytest.fixture
def minimal_sealed(tmp_path,monkeypatch):
    root=tmp_path/'code';(root/'offload_research').mkdir(parents=True)
    file=root/'offload_research/test.py';file.write_text('x = 1\n')
    monkeypatch.setattr(fresh,'SOURCE_ROOT',root)
    out=tmp_path/'output';f=out/'frozen';f.mkdir(parents=True);(out/'inputs').mkdir()
    (out/'inputs/step2_export.zip').write_bytes(b'input')
    (f/'protocol.json').write_text('{}')
    (f/'predictions.csv').write_text('prediction\n1\n')
    write_json(f/'seal.json',dict(file_sha256={'protocol.json':sha(b'{}'),'predictions.csv':sha(b'prediction\n1\n')},
        executable_source_sha256={'offload_research/test.py':sha(file.read_bytes())},input_sha256=sha(b'input')))
    return out,root


def test_verify_good_seal(minimal_sealed):
    out,_=minimal_sealed;assert verify(out)['status']=='verified'


@pytest.mark.parametrize('file',['protocol.json','predictions.csv'])
def test_mutation_detected(minimal_sealed,file):
    out,_=minimal_sealed;(out/'frozen'/file).write_text('edited')
    with pytest.raises(ValueError,match='Frozen file changed'):verify(out)


def test_source_mutation_detected(minimal_sealed):
    out,root=minimal_sealed;(root/'offload_research/test.py').write_text('edited')
    with pytest.raises(ValueError,match='Executable source changed'):verify(out)


def test_input_mutation_detected(minimal_sealed):
    out,_=minimal_sealed;(out/'inputs/step2_export.zip').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Input checkpoint changed'):verify(out)


def test_export_preserves_partial_and_failed_integrity(minimal_sealed):
    out,_=minimal_sealed;(out/'frozen/predictions.csv').write_text('changed')
    archive=fresh.export(out)
    with zipfile.ZipFile(archive) as z:
        report=json.loads(z.read('export_status.json'))
        assert report['integrity']['status']=='failed' and report['complete_analysis'] is False


def test_incomplete_reference_rejected(tmp_path):
    write_json(tmp_path/'reference_validation.json',dict(status='passed',tiny_random=False,
        seal_sha256='s',cuda_available=True,gpu_name='Tesla T4',checks=[]))
    with pytest.raises(ValueError,match='Incomplete'):load_reference_gate(tmp_path,{'seal_sha256':'s'})


def test_tiny_reference_rejected(tmp_path):
    write_json(tmp_path/'reference_validation.json',dict(status='passed',tiny_random=True))
    with pytest.raises(ValueError,match='real successful'):load_reference_gate(tmp_path,{'seal_sha256':'s'})


def test_teacher_forced_helper_cpu():
    import torch
    from hetero.model import GPT2,GPT2Spec
    from hetero.engine import OffloadEngine
    torch.set_num_threads(1)
    e=OffloadEngine(GPT2(GPT2Spec(vocab_size=97,n_positions=32,n_embd=24,n_layer=4,n_head=4)))
    prompt=make_tokens(2,7,123,97);cont=make_tokens(2,3,777,97)
    expected=[]
    for i in range(4):
        ids=torch.cat([prompt,cont[:,:i]],dim=1)
        logits,_=e.forward(ids,use_cache=False)
        expected.append(logits.clone())
    result=compare_teacher_forced(e,prompt,cont,expected,0)
    assert result['passed'] and result['teacher_forced_decode_steps']==3


def test_cpu_tiny_collect_case_writes_accounted_trials(tmp_path):
    import torch
    from hetero.model import GPT2,GPT2Spec
    from hetero.engine import OffloadEngine
    from hetero.benchmark import reference_outputs
    torch.set_num_threads(1)
    e=OffloadEngine(GPT2(GPT2Spec(vocab_size=97,n_positions=32,n_embd=24,n_layer=4,n_head=4)))
    ids=make_tokens(2,8,123,97);refs=reference_outputs(e,ids)
    p=make_protocol();p['warmup_per_case']=0;p['measured_repetitions_per_case_per_block']=2
    c=dict(case_id='tiny_software_test',batch_size=2,sequence_length=7,new_tokens=4,gpu_layers=0,layout='prefix')
    r=collect_case(e,c,p,tmp_path/'case',0,{'seal_sha256':'test'},refs,ids)
    assert r['status']=='ok' and r['gpu_peak_allocated_bytes']==0
    raw=pd.read_csv(tmp_path/'case/trials.csv')
    assert len(raw)==2
    assert all(len(json.loads(x))==3 for x in raw.decode_step_ms_json)
    assert np.allclose(raw.generation_ms,raw.ttft_ms+raw.decode_ms)
    assert (tmp_path/'case/trials.partial.jsonl').exists()
