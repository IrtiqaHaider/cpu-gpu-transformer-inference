#!/usr/bin/env python3
"""Reproduce archived analyses without model downloads or GPU inference.

The raw archives are never mutated. Working extracts go into a NEW directory.
This executes the repository's reviewed, hash-checked code, not code imported
from the archive. Input archives are bounded and traversal/symlink-checked.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import platform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from offload_research import robustness, fresh, step2, fit


def read_json(p: Path):
    return json.loads(p.read_text())


def compare_csv_tree(expected: Path, actual: Path) -> int:
    checked = 0
    for p in expected.glob('*.csv'):
        pd.testing.assert_frame_equal(pd.read_csv(p), pd.read_csv(actual/p.name),
                                      check_exact=False, rtol=1e-8, atol=1e-7)
        checked += 1
    return checked


def run(out: Path, refit: bool = False) -> dict:
    out = out.resolve()
    if out.exists():
        raise ValueError('Output already exists. Choose a NEW --out path; never overwrite evidence.')
    out.mkdir(parents=True)
    archived = ROOT/'data/archives/step4_export.zip'
    expected_sha = read_json(ROOT/'release_checks/step4_audit.json')['archive_sha256']
    if hashlib.sha256(archived.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('Published Step 4 archive does not match the audited input hash.')
    s4 = out/'step4'
    robustness.safe_extract(archived, s4)
    robustness.safe_extract(s4/'inputs/step3_export.zip', s4/'parent')
    saved4 = read_json(s4/'analysis/summary.json')
    saved3 = read_json(s4/'parent/analysis/summary.json')
    parent_audit = robustness.audit_parent(s4/'parent')
    actual4 = robustness.analyze(s4)
    if actual4 != saved4 or parent_audit['summary'] != saved3:
        raise ValueError('Recomputed summary differs from the archived summary.')
    csvs = compare_csv_tree(ROOT/'data/step4/analysis', s4/'analysis')
    csvs += compare_csv_tree(ROOT/'data/step3/analysis', s4/'parent/analysis')
    for p in (s4/'frozen/models').glob('*.json'):
        if p.read_bytes() != (ROOT/'models'/p.name).read_bytes():
            raise ValueError('Published model differs from frozen model: '+p.name)
    # Retrospective re-fitting is optional and never alters frozen/public models.
    if refit:
        with zipfile.ZipFile(s4/'parent/inputs/step2_export.zip') as z:
            baseline = out/'baseline_export.zip'; baseline.write_bytes(z.read('inputs/baseline_export.zip'))
            step1 = out/'step1_export.zip'; step1.write_bytes(z.read('inputs/step1_export.zip'))
        fit.run(baseline, out/'refit_step1')
        step2.run(step1, baseline, out/'refit_step2')
        for name in ('compute','compute_transfer','memory'):
            a = read_json(out/f'refit_step2/models/{name}.json')
            b = read_json(ROOT/f'models/{name}.json')
            # Check coefficients and predictions via model arrays; metadata paths can differ.
            if name == 'memory':
                for k in ('coefficients_mib','feature_scales'):
                    np.testing.assert_allclose(a[k], b[k], rtol=1e-7, atol=1e-7)
            else:
                for target in a['targets']:
                    for k in ('coefficients_ms','feature_scales'):
                        np.testing.assert_allclose(a['targets'][target][k],b['targets'][target][k],rtol=1e-7,atol=1e-7)
    result = dict(status='passed', archived_sha256=expected_sha, csv_tables_reproduced=csvs,
                  step3_trials=saved3['primary_generation_trials']+saved3['control_generation_trials'],
                  step4_trials=actual4['measured_generation_trials'], frozen_models_unchanged=True,
                  historical_refit_checked=refit, python=platform.python_version(),
                  new_gpu_inference=False,
                  note='Analysis reproduction, not a replication of GPU measurements. Raw input archives unchanged.')
    (out/'REPRODUCTION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT/'work/reproduction')
    p.add_argument('--refit', action='store_true', help='Also refit historical predictors in separate output folders.')
    args=p.parse_args()
    try:
        print(json.dumps(run(args.out,args.refit),indent=2))
    except (OSError,ValueError,AssertionError,KeyError,zipfile.BadZipFile) as exc:
        raise SystemExit('Reproduction failed; retain partial working output. '+str(exc)) from exc
