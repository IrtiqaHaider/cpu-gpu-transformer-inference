from __future__ import annotations
import json
import sys
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import pytest

from offload_research.cost_model import (CostModel, Workload, features, fit_model,
    fit_nonnegative, predict_nonnegative, predict_frame, save_model)
from offload_research.fit import group_folds, load_main, errors

CONFIG = {"n_layer": 12, "n_embd": 768, "n_positions": 1024}
META = {"model_config": CONFIG, "model": {"model_id": "test-only", "revision": "a" * 40}}


def synthetic():
    rows = []
    for b in [1, 2, 4]:
        for s in [32, 128, 512]:
            for k in [0, 3, 6, 9, 12]:
                w = Workload(b, s, 32, k)
                pre = features(w, CONFIG, "prefill", "compute")
                dec = features(w, CONFIG, "decode", "compute")
                beta = np.array([1, .1, .01, .001, .0001, .00001, 1, .1])
                rows.append({**w.__dict__, "prefill_ms_median": float(pre @ beta),
                    "ttft_ms_median": float(pre @ beta + .1),
                    "decode_tpot_ms_median": float(dec @ beta),
                    "generation_ms_median": float(pre @ beta + .1 + 31 * (dec @ beta))})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("k,count", [(0,0), (3,2), (6,2), (9,2), (12,0)])
def test_prefix_boundary_and_payload(k, count):
    v = features(Workload(4,512,32,k), CONFIG, "prefill", "compute_transfer")
    assert v[-2] == count
    assert v[-1] == count * 4 * 512 * 768 * 4 / 2**20


def test_decode_payload_and_mean_context():
    v = features(Workload(4,128,32,6), CONFIG, "decode", "compute_transfer")
    assert v[-1] == 2 * 4 * 768 * 4 / 2**20
    assert v[4] == 6 * 4 * (128 + 16)


def test_endpoint_moves_only_at_cpu_only_case():
    cpu = features(Workload(2,128,32,0), CONFIG, "prefill", "compute")
    mixed = features(Workload(2,128,32,3), CONFIG, "prefill", "compute")
    np.testing.assert_array_equal(cpu[-2:], [2,0])
    np.testing.assert_array_equal(mixed[-2:], [0,2])


@pytest.mark.parametrize("kwargs", [
    {"batch_size":0}, {"batch_size":1.5}, {"batch_size":True},
    {"sequence_length":0}, {"new_tokens":1}, {"gpu_layers":13},
    {"gpu_layers":-1}, {"layout":"suffix"}, {"sequence_length":1024},
])
def test_invalid_workload_rejected(kwargs):
    values = dict(batch_size=1, sequence_length=32, new_tokens=32, gpu_layers=6)
    values.update(kwargs)
    with pytest.raises(ValueError):
        features(Workload(**values), CONFIG, "prefill", "compute")


def test_context_last_selected_token_not_processed():
    features(Workload(1,1023,2,6), CONFIG, "decode", "compute")


@pytest.mark.parametrize("x,y", [
    ([[1,2]],[0]), ([[1,2]],[-1]), ([[1,np.nan]],[1]),
    ([[1,-2]],[1]), ([[1,2]],[float('inf')]),
])
def test_bad_fit_inputs_rejected(x,y):
    with pytest.raises(ValueError):
        fit_nonnegative(np.asarray(x),np.asarray(y))


def test_target_unit_invariance_and_positive_coefficients():
    x = np.array([[1,1],[1,2],[1,3],[1,4]],float)
    y = np.array([2,3,4,5],float)
    a = fit_nonnegative(x,y)
    b = fit_nonnegative(x,1000*y)
    assert min(a["coefficients_ms"]) >= 0
    np.testing.assert_allclose(predict_nonnegative(x,b),1000*predict_nonnegative(x,a),rtol=1e-9)


def test_scaler_fits_only_supplied_training_rows():
    x = np.array([[1,2],[1,3],[1,4]],float)
    fitted = fit_nonnegative(x,np.array([2,3,4],float))
    assert fitted["feature_scales"] == [1,4]
    predict_nonnegative(np.array([[1,10000]],float),fitted)
    assert fitted["feature_scales"] == [1,4]


def test_nine_group_folds_and_no_workload_leakage():
    data = synthetic()
    seen = []
    for name,tr,te in group_folds(data):
        assert len(tr)==40 and len(te)==5
        train_groups = set(data.iloc[tr][["batch_size","sequence_length","new_tokens"]].itertuples(index=False,name=None))
        test_groups = set(data.iloc[te][["batch_size","sequence_length","new_tokens"]].itertuples(index=False,name=None))
        assert train_groups.isdisjoint(test_groups)
        seen.extend(te)
    assert sorted(seen) == list(range(45))


def test_serialization_and_inference_accounting(tmp_path):
    data = synthetic()
    fitted = fit_model(data,META,"compute_transfer")
    path=tmp_path/'model.json'
    save_model(fitted,path)
    before=predict_frame(fitted,data)
    loaded=CostModel.load(path)
    p=loaded.predict(batch_size=1,sequence_length=32,new_tokens=32,gpu_layers=0)
    assert p['predicted_generation_ms']==pytest.approx(p['predicted_ttft_ms']+31*p['predicted_decode_tpot_ms'])
    assert p['predicted_prefill_ms']==pytest.approx(before.predicted_prefill_ms.iloc[0])
    assert p['measured'] is False


def test_unseen_output_lengths_require_explicit_extrapolation():
    model=CostModel(fit_model(synthetic(),META,"compute"))
    with pytest.raises(ValueError,match="Outside calibration"):
        model.predict(batch_size=1,sequence_length=32,new_tokens=64,gpu_layers=6)
    with pytest.warns(UserWarning,match="Unvalidated extrapolation"):
        p=model.predict(batch_size=1,sequence_length=32,new_tokens=64,gpu_layers=6,allow_extrapolation=True)
    assert p['extrapolation'] is True


def test_synthetic_nonnegative_model_can_fit():
    data=synthetic()
    model=fit_model(data,META,"compute")
    p=predict_frame(model,data)
    assert errors(data.prefill_ms_median,p.predicted_prefill_ms)['median_ape_pct'] < 1


def test_no_torch_import():
    import subprocess
    subprocess.run([sys.executable, "-c",
        "import offload_research.fit; import sys; assert \"torch\" not in sys.modules"], check=True)


def test_archive_with_no_main_is_rejected(tmp_path):
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('layouts/summary.csv','not a main run')
    with pytest.raises(ValueError,match='main/summary'):
        load_main(p)


def test_ambiguous_main_archives_are_rejected(tmp_path):
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('one/main/summary.csv','')
        z.writestr('two/main/summary.csv','')
    with pytest.raises(ValueError,match='exactly one'):
        load_main(p)


def build_synthetic_archive(path, mutation=None):
    from offload_research.fit import METRICS
    rows, trials = [], []
    for case, row in enumerate(synthetic().to_dict('records')):
        local=[]
        for t,mult in enumerate([.9,1.0,1.1]):
            b,s,n=row['batch_size'],row['sequence_length'],row['new_tokens']
            pre=row['prefill_ms_median']*mult
            ttft=row['ttft_ms_median']*mult
            dec=row['decode_tpot_ms_median']*(n-1)*mult
            total=ttft+dec
            r={k:row[k] for k in ['layout','gpu_layers','batch_size','sequence_length','new_tokens']}
            r.update(case_index=case,trial=t,prefill_ms=pre,ttft_ms=ttft,
                generation_ms=total,decode_ms=dec,decode_tpot_ms=dec/(n-1),
                prefill_input_tokens_per_s=b*s*1000/pre,
                decode_generated_tokens_per_s=b*(n-1)*1000/dec,
                generation_generated_tokens_per_s=b*n*1000/total,
                final_cached_positions=s+n-1,
                decode_step_ms_json=json.dumps([dec/(n-1)]*(n-1)))
            local.append(r)
            trials.append(r)
        summary={k:row[k] for k in ['layout','gpu_layers','batch_size','sequence_length','new_tokens']}
        summary.update(case_index=case,status='ok')
        for key in METRICS:
            vals=[r[key] for r in local]
            for name,fn in [('median',np.median),('min',np.min),('max',np.max),('p95',lambda x:np.percentile(x,95))]:
                summary[key+'_'+name]=fn(vals)
        rows.append(summary)
    meta={'model_config':CONFIG,'model':{'model_id':'openai-community/gpt2','revision':'a'*40},
          'dtype':'float32 on both devices','attention':'explicit eager causal attention',
          'torch_threads':2,'arguments':{'repeats':3,'tiny_random':False}}
    frames={'summary':pd.DataFrame(rows),'trials':pd.DataFrame(trials),'metadata':meta}
    if mutation:mutation(frames)
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('main/summary.csv',frames['summary'].to_csv(index=False))
        z.writestr('main/trials.csv',frames['trials'].to_csv(index=False))
        z.writestr('main/metadata.json',json.dumps(frames['metadata']))
    return path


def test_archive_audit_preserves_all_cases_and_trials(tmp_path):
    data,meta,manifest=load_main(build_synthetic_archive(tmp_path/'ok.zip'))
    assert len(data)==45
    assert manifest['raw_trials_checked']==135
    assert manifest['workload_groups']==9
    assert manifest['all_trial_rows_retained_in_medians'] is True


@pytest.mark.parametrize('mutation,match',[
    (lambda f: f['summary'].__setitem__('prefill_ms_median',0),'Summary mismatch'),
    (lambda f: f['metadata'].__setitem__('torch_threads',1),'two-thread'),
    (lambda f: f['metadata'].__setitem__('dtype','float16'),'FP32'),
    (lambda f: f['metadata']['arguments'].__setitem__('tiny_random',True),'Random-model'),
    (lambda f: f['summary'].__setitem__('status','cuda_oom'),'Failed cases'),
    (lambda f: f['trials'].__setitem__('trial',0),'Duplicate trial'),
    (lambda f: f.__setitem__('trials',f['trials'].iloc[1:].copy()),'Incomplete trial'),
])
def test_malformed_run_rejected(tmp_path,mutation,match):
    with pytest.raises(ValueError,match=match):
        load_main(build_synthetic_archive(tmp_path/'bad.zip',mutation))
