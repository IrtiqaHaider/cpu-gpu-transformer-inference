"""Offline packaging checks. These are not additional hardware measurements."""
from pathlib import Path
import hashlib, json, zipfile
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]

def test_main_study_trial_count():
    paths=['baseline/main/trials.csv','baseline/layouts/trials.csv','step3/analysis/all_trials.csv','step4/analysis/all_trials.csv']
    assert [len(pd.read_csv(ROOT/'data'/p)) for p in paths]==[450,120,230,260]

@pytest.mark.parametrize('name',['compute','compute_transfer','memory'])
def test_frozen_models_unchanged(name):
    assert (ROOT/f'models/{name}.json').read_bytes()==(ROOT/f'data/step4/frozen/models/{name}.json').read_bytes()

def test_original_core_source_unchanged():
    seal=json.loads((ROOT/'data/step4/frozen/seal.json').read_text())
    for name,digest in seal['executable_source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest

def test_step4_archive_identity():
    audit=json.loads((ROOT/'release_checks/step4_audit.json').read_text())
    assert hashlib.sha256((ROOT/'data/archives/step4_export.zip').read_bytes()).hexdigest()==audit['archive_sha256']

@pytest.mark.parametrize('threads',[1,2])
def test_reference_checks_preserved(threads):
    gate=json.loads((ROOT/f'data/step4/reference_threads_{threads}.json').read_text())
    assert gate['status']=='passed' and len(gate['checks'])==13
    assert all(g['passed'] for g in gate['checks'])

def test_memory_guard_unchanged():
    p=pd.read_csv(ROOT/'data/step4/frozen/predictions.csv')
    gpu=p[p.gpu_layers>0]
    assert ((gpu.guarded_peak_mib - (gpu.predicted_peak_mib*1.05+16)).abs()<1e-8).all()

def test_policy_equality_is_reported_not_hidden():
    s=pd.read_csv(ROOT/'data/step4/analysis/policy_scores.csv')
    assert s.groupby(['cpu_threads','workload_id','budget_mib']).selected_gpu_layers.nunique().eq(1).all()

def test_no_weights_distributed():
    assert not any(p.is_file() and p.suffix in ['.pt','.pth','.safetensors','.bin'] for p in ROOT.rglob('*') if 'work' not in p.parts)

def test_readme_links_resolve():
    import re
    for path in re.findall(r'\]\(([^)]+)\)',(ROOT/'README.md').read_text()):
        if '://' not in path and not path.startswith('#'):
            assert (ROOT/path.split('#')[0]).exists(),path
