"""Step 3: freeze prediction-only decisions BEFORE fresh T4 measurement.

No fitting occurs here. Existing engine and Step 1/2 source files are unmodified.
A local hash seal detects accidental edits, not malicious rewriting or a public
preregistration. Read README_STEP3.md for the scope and failure protocol.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import zipfile

import numpy as np
import pandas as pd
from .cost_model import CostModel, KEYS
from .memory_model import MemoryModel, MemoryGuard
from .planner import PlacementPlanner, CANDIDATES, POLICIES, BUDGETS_MIB, choose
from .step2 import load_memory_data, read_members

EXPECTED_STEP2_SHA256 = '83d4e5f5d358ef7506676bb8a3f02c3511be086d1a78d84c944f075779ead1ce'
VARIANTS = ('compute', 'compute_transfer')
METRICS = ('prefill_ms', 'ttft_ms', 'decode_tpot_ms', 'generation_ms')
SOURCE_ROOT = Path(__file__).resolve().parent.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def workload_id(b, s, n=32):
    return f'B{b}_S{s}_N{n}'


def make_protocol() -> dict:
    cases = []
    for b in (1, 3):
        for s in (64, 256):
            for k in CANDIDATES:
                wid = workload_id(b, s)
                cases.append(dict(case_id=f'{wid}_G{k}', workload_id=wid,
                                  kind='primary', batch_size=b, sequence_length=s,
                                  new_tokens=32, gpu_layers=k, layout='prefix'))
    for k in (0, 6, 12):
        wid = workload_id(1, 128)
        cases.append(dict(case_id=f'{wid}_G{k}', workload_id=wid,
                          kind='historical_control', batch_size=1, sequence_length=128,
                          new_tokens=32, gpu_layers=k, layout='prefix'))
    order = [c['case_id'] for c in cases]
    random.Random(20260920).shuffle(order)
    return dict(
        version='step3-v0.3.0', created_utc=now(),
        question='Do the frozen latency/memory predictors and policies generalize to new workload shapes?',
        model_id='openai-community/gpt2',
        revision='607a30d783dfa663caf39e06633721c8d4cfcd7e',
        dtype='float32', attention='explicit eager causal attention',
        layout='prefix', cpu_threads=2, interop_threads=1,
        gpu_required='T4', candidates=list(CANDIDATES), policies=list(POLICIES),
        budgets_mib=list(BUDGETS_MIB), memory_guard={'relative':0.05, 'extra_mib':16.0},
        cases=cases, blocks=[order, list(reversed(order))], warmup_per_case=5,
        measured_repetitions_per_case_per_block=5, input_seed=123,
        primary_workloads=4, primary_candidates=20, historical_controls=3,
        expected_case_block_records=46, expected_generation_trials=230,
        expected_primary_policy_decisions=96,
        selection='Predictions and decisions are written before validation or fresh timing. No refitting.',
        objective='median complete-generation latency',
        memory_target='maximum peak PyTorch allocated bytes over timed prefill and generation across both blocks',
        oracle='Best measured feasible candidate among the same five prefix placements, no guard on actual memory.',
        controls='Historical B1/S128/N32 at G0/G6/G12: diagnostic only; NEVER recalibrate using these measurements.',
        block_scope='Two order-reversed blocks in one session, NOT two independent Colab sessions.',
        unchanged_output_length='32; output-length extrapolation deliberately deferred.',
        synthetic_budget=True, physical_memory_cap_enforced=False,
        failures='Retain all trials. OOM/runtime failures and missing cases prevent complete oracle scoring. No silent retry.',
        numerical_gate='Four primary prompt shapes, each of five placements, teacher-forced 3-token continuation versus HF.',
        environment='Require T4; record CPU/software differences against calibration and between processes. No adaptation.',
        public_preregistration=False,
        future_test_status='New timing measurements, but workload choices were informed by historical development.',
    )


def load_planner(folder: Path) -> PlacementPlanner:
    return PlacementPlanner(MemoryModel.load(folder/'memory.json'),
                            CostModel.load(folder/'compute.json'),
                            CostModel.load(folder/'compute_transfer.json'), MemoryGuard())


def prediction_tables(planner: PlacementPlanner, protocol: dict):
    rows, decisions = [], []
    for case in protocol['cases']:
        args = {k:case[k] for k in ('batch_size','sequence_length','new_tokens')}
        table = planner.candidates(**args, candidates=(case['gpu_layers'],))
        row = {**case, **table.iloc[0].to_dict()}
        for variant in VARIANTS:
            p = planner.latency[variant].predict(**args, gpu_layers=case['gpu_layers'])
            for metric in METRICS:
                row[f'predicted_{metric}_{variant}'] = p[f'predicted_{metric}']
        rows.append(row)
    prediction = pd.DataFrame(rows)
    for wid, table in prediction[prediction.kind == 'primary'].groupby('workload_id', sort=True):
        for budget in protocol['budgets_mib']:
            for policy in protocol['policies']:
                picked = choose(table, budget, policy)
                decisions.append(dict(workload_id=wid, budget_mib=budget, policy=policy,
                    selected_gpu_layers=None if picked is None else int(picked['gpu_layers']),
                    selection_status='abstained' if picked is None else 'predicted_feasible'))
    return prediction, pd.DataFrame(decisions)


def prepare(step2: Path, output: Path) -> dict:
    """Bounded archive reads only. Never import or execute code from user ZIP."""
    require(step2.is_file(), 'Step 2 checkpoint not found.')
    require(sha(step2.read_bytes()) == EXPECTED_STEP2_SHA256,
            'Wrong Step 2 export: upload step2_20260919T194826078232Z_export.zip unchanged.')
    frozen = output/'frozen'
    require(not frozen.exists(), 'Freeze already exists. Verify it; do not regenerate after observing results.')
    names = ['input_manifest.json','source_metadata.json','calibration_with_memory.csv',
             'test_output.txt','summary.json','inputs/baseline_export.zip']
    original = ('__init__.py','cost_model.py','fit.py','memory_model.py','planner.py','step2.py')
    names += ['models/'+name+'.json' for name in ('memory',)+VARIANTS]
    names += ['code_snapshot/'+name for name in original]
    blobs = read_members(step2, names)
    manifest = json.loads(blobs['input_manifest.json'])
    require(sha(blobs['inputs/baseline_export.zip']) == manifest['baseline']['input_sha256'],
            'Baseline identity does not match Step 2 manifest.')
    for name in original:
        require(blobs['code_snapshot/'+name] == (SOURCE_ROOT/'offload_research'/name).read_bytes(),
                'Step 2 source mismatch: '+name)
    output.mkdir(parents=True, exist_ok=True)
    inputs = output/'inputs'; inputs.mkdir(exist_ok=True)
    shutil.copyfile(step2, inputs/'step2_export.zip')
    (inputs/'baseline_export.zip').write_bytes(blobs['inputs/baseline_export.zip'])
    data, metadata, audit = load_memory_data(inputs/'baseline_export.zip')
    require(metadata == json.loads(blobs['source_metadata.json']), 'Calibration metadata mismatch.')
    saved = pd.read_csv(io.BytesIO(blobs['calibration_with_memory.csv']))
    pd.testing.assert_frame_equal(saved.sort_values(KEYS).reset_index(drop=True),
        data.sort_values(KEYS).reset_index(drop=True), check_exact=False, rtol=1e-8, atol=1e-8)
    frozen.mkdir(); (frozen/'models').mkdir()
    for name in ('memory',)+VARIANTS:
        (frozen/'models'/f'{name}.json').write_bytes(blobs[f'models/{name}.json'])
    protocol = make_protocol()
    historical = {tuple(int(r[k]) for k in ('batch_size','sequence_length','new_tokens'))
                  for r in data.to_dict('records')}
    require(all((c['batch_size'],c['sequence_length'],c['new_tokens']) not in historical
                for c in protocol['cases'] if c['kind']=='primary'),
            'A primary workload already appears in calibration.')
    planner = load_planner(frozen/'models')
    require(planner.memory.model['model_source'] == {'model_id':protocol['model_id'], 'revision':protocol['revision']},
            'Model revision mismatch.')
    prediction, decisions = prediction_tables(planner, protocol)
    prediction.to_csv(frozen/'predictions.csv', index=False)
    decisions.to_csv(frozen/'decisions.csv', index=False)
    write_json(frozen/'protocol.json', protocol)
    write_json(frozen/'calibration_metadata.json', metadata)
    data.to_csv(frozen/'historical_calibration.csv', index=False)
    write_json(frozen/'step2_audit.json', dict(input_sha256=sha(step2.read_bytes()),
        baseline_sha256=audit['input_sha256'], configurations=len(data),
        original_trials_audited=audit['raw_trials_checked'],
        uploaded_test_log=blobs['test_output.txt'].decode(),
        source_verified=True, refitting=False, predictions_are_measurements=False))
    source_hashes = {}
    for package in ('offload_research','hetero'):
        for path in sorted((SOURCE_ROOT/package).glob('*.py')):
            relative = path.relative_to(SOURCE_ROOT).as_posix()
            dest = frozen/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(path.read_bytes());source_hashes[relative] = sha(path.read_bytes())
    hashes = {p.relative_to(frozen).as_posix():sha(p.read_bytes())
              for p in sorted(frozen.rglob('*')) if p.is_file()}
    seal = dict(schema='step3-seal-v1', created_utc=now(), file_sha256=hashes,
                executable_source_sha256=source_hashes, input_sha256=sha(step2.read_bytes()),
                limitation='Local integrity seal, not a third-party timestamp or public preregistration.')
    write_json(frozen/'seal.json', seal)
    return verify(output)


def verify(output: Path) -> dict:
    frozen=output/'frozen';seal=read_json(frozen/'seal.json')
    for name,digest in seal['file_sha256'].items():
        target = (frozen/name).resolve()
        require(target.is_relative_to(frozen.resolve()), 'Unsafe frozen path.')
        require(target.is_file() and sha(target.read_bytes())==digest, 'Frozen file changed: '+name)
    for name,digest in seal['executable_source_sha256'].items():
        target = (SOURCE_ROOT/name).resolve()
        require(target.is_relative_to(SOURCE_ROOT.resolve()), 'Unsafe source path.')
        require(target.is_file() and sha(target.read_bytes())==digest, 'Executable source changed: '+name)
    require(sha((output/'inputs'/'step2_export.zip').read_bytes())==seal['input_sha256'], 'Input checkpoint changed.')
    return dict(status='verified', seal_sha256=sha((frozen/'seal.json').read_bytes()),
                protocol_sha256=seal['file_sha256']['protocol.json'],
                new_gpu_inference=False)


def environment_differences(current: dict, calibration: dict) -> list[dict]:
    fields = ['gpu_name','cpu_count_logical','torch_threads','torch_interop_threads','cuda_runtime']
    changes = [dict(field=k, calibration=calibration.get(k), current=current.get(k))
               for k in fields if current.get(k) != calibration.get(k)]
    for key in ('torch','numpy','pandas','safetensors','huggingface-hub','transformers'):
        a=calibration.get('packages',{}).get(key);b=current.get('packages',{}).get(key)
        if a!=b:changes.append(dict(field='packages.'+key,calibration=a,current=b))
    def cpu_line(meta, label):
        return next((line.split(':',1)[1].strip() for line in meta.get('lscpu','').splitlines()
                     if line.startswith(label+':')),None)
    for label in ('Model name','Core(s) per socket','Thread(s) per core'):
        a,b=cpu_line(calibration,label),cpu_line(current,label)
        if a!=b:changes.append(dict(field='CPU '+label,calibration=a,current=b))
    return changes


def require_t4(current: dict):
    require(current.get('cuda_available') is True, 'A CUDA GPU is required for Step 3 measurement.')
    require('T4' in str(current.get('gpu_name','')).split(),
            'This protocol targets a T4. Do not substitute another accelerator silently.')
    require(current.get('torch_threads')==2 and current.get('torch_interop_threads')==1,
            'CPU thread settings must be 2 intra-op and 1 inter-op.')


def evaluate_policy_decisions(decisions: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Score already frozen choices; never select using observed test values."""
    results=[]
    for d in decisions.to_dict('records'):
        table=observations[observations.workload_id==d['workload_id']]
        require(set(table.gpu_layers.astype(int))==set(CANDIDATES) and len(table)==5,
                'Cannot compute candidate-set oracle from incomplete or duplicate measurements.')
        require(np.isfinite(table[['gpu_peak_mib','generation_ms_median']].to_numpy(float)).all(),
                'Invalid measurements.')
        feasible=table[table.gpu_peak_mib<=d['budget_mib']]
        oracle=None if feasible.empty else feasible.sort_values(['generation_ms_median','gpu_layers']).iloc[0]
        row={**d,'oracle_gpu_layers':None if oracle is None else int(oracle.gpu_layers),
             'observed_budget_violation':False, 'regret_pct':None,'measured_generation_ms':None,
             'measured_peak_mib':None}
        if pd.isna(d['selected_gpu_layers']):row['outcome']='abstained'
        else:
            selected=table[table.gpu_layers==int(d['selected_gpu_layers'])]
            require(len(selected)==1,'Frozen selection is absent from measurements.')
            s=selected.iloc[0];violation=bool(s.gpu_peak_mib>d['budget_mib'])
            row.update(measured_generation_ms=float(s.generation_ms_median),measured_peak_mib=float(s.gpu_peak_mib),
                       observed_budget_violation=violation,outcome='budget_violation' if violation else 'feasible')
            if not violation and oracle is not None:
                row['regret_pct']=float((s.generation_ms_median/oracle.generation_ms_median-1)*100)
        results.append(row)
    return pd.DataFrame(results)


def aggregate_case(case: dict, per_block: list[dict], trials: pd.DataFrame, expected_trials: int) -> dict:
    require(len(trials)==expected_trials, 'Wrong trial count for '+case['case_id'])
    require(not trials.duplicated(['block','trial']).any(), 'Duplicate trials.')
    out={**case,'trials':len(trials),
         'gpu_peak_mib':max(float(s['gpu_peak_allocated_bytes']) for s in per_block)/2**20}
    for metric in METRICS:
        values=trials[metric].to_numpy(float)
        require(np.isfinite(values).all() and (values>0).all(), 'Invalid trial latency.')
        out[metric+'_median']=float(np.median(values))
        out[metric+'_min']=float(values.min());out[metric+'_max']=float(values.max())
    require(np.allclose(trials.generation_ms,trials.ttft_ms+trials.decode_ms,rtol=1e-7,atol=1e-7),
            'Generation accounting mismatch.')
    require(np.allclose(trials.decode_ms/(case['new_tokens']-1),trials.decode_tpot_ms,rtol=1e-7,atol=1e-7),
            'Decode accounting mismatch.')
    out['pooled_generation_tokens_s']=float(len(trials)*case['batch_size']*case['new_tokens']*1000/trials.generation_ms.sum())
    return out


def analyze(output: Path) -> dict:
    verified=verify(output);p=read_json(output/'frozen/protocol.json')
    rows=[];failures=[]
    all_trials=[]
    for case in p['cases']:
        records=[];raw=[]
        for block in range(2):
            folder=output/'measurement'/f'block_{block}'/case['case_id']
            path=folder/'summary.json'
            if not path.exists():failures.append({'block':block,'case_id':case['case_id'],'status':'missing'});continue
            s=read_json(path)
            if s.get('status')!='ok':failures.append({'block':block,'case_id':case['case_id'],'status':s.get('status')});continue
            require(s['seal_sha256']==verified['seal_sha256'],'Measurement used a different freeze.')
            for key in ('case_id','gpu_layers','batch_size','sequence_length','new_tokens','layout'):
                require(s[key]==case[key], 'Measured case identity differs: '+key)
            t=pd.read_csv(folder/'trials.csv')
            require(list(t.trial)==list(range(p['measured_repetitions_per_case_per_block'])),'Missing trial indices.')
            require((t.block==block).all() and (t.case_id==case['case_id']).all(),'Raw trial identity mismatch.')
            for metric in METRICS:
                require(np.isclose(float(s[metric+'_median']),float(t[metric].median()),rtol=1e-7,atol=1e-7),
                        'Recorded block summary disagrees with raw trials.')
            for key in ('gpu_layers','batch_size','sequence_length','new_tokens','layout'):
                require((t[key]==case[key]).all(), 'Raw trial workload mismatch: '+key)
            require((t.final_cached_positions==case['sequence_length']+case['new_tokens']-1).all(),
                    'Final cache length mismatch.')
            for value,total in zip(t.decode_step_ms_json,t.decode_ms):
                steps=json.loads(value)
                require(len(steps)==case['new_tokens']-1 and np.isfinite(steps).all() and min(steps)>0,
                        'Decode step sequence is incomplete or invalid.')
                require(sum(steps)<=total+1e-5,'Decode step times exceed total decode time.')
            records.append(s);raw.append(t)
        if len(records)==2:
            merged=pd.concat(raw,ignore_index=True);all_trials.append(merged)
            rows.append(aggregate_case(case,records,merged,2*p['measured_repetitions_per_case_per_block']))
    analysis=output/'analysis';analysis.mkdir(exist_ok=True)
    coverage=dict(expected_cases=len(p['cases']),complete_cases=len(rows),failures=failures,
                  complete=not failures,scope='One session, two reverse-order blocks; no refitting.')
    write_json(analysis/'coverage.json',coverage)
    if failures:
        (analysis/'report.md').write_text('# Step 3 incomplete\n\nSome cases failed or are missing. Export all files. '
            'No complete policy/oracle result is claimed. Do not delete failed trials or retry until a favorable result.\n')
        return coverage
    # Complete scores additionally require the genuine reference gate and consistent
    # runtime records. Partial exports remain readable without either gate.
    from .collect_fresh import load_reference_gate, session_fingerprint
    reference=load_reference_gate(output,verified)
    preflight=read_json(output/'runtime_preflight.json')
    require(preflight['seal_sha256']==verified['seal_sha256'],'Preflight seal mismatch.')
    for block in range(2):
        runtime=read_json(output/'measurement'/f'block_{block}'/'runtime.json')
        require(runtime['seal_sha256']==verified['seal_sha256'],'Runtime seal mismatch.')
        require(runtime['reference_sha256']==sha((output/'reference_validation.json').read_bytes()),
                'Reference validation changed after collection.')
        require_t4(runtime['runtime'])
        require(session_fingerprint(runtime['runtime'])==preflight['session_fingerprint'],
                'Measurement blocks belong to different runtime configurations.')
        require((output/'measurement'/f'block_{block}'/'completed.json').exists(),
                'Measurement block was not finalized.')
    require(session_fingerprint(reference['runtime'])==preflight['session_fingerprint'],
            'Reference validation belongs to another runtime configuration.')
    observed=pd.DataFrame(rows);observed.to_csv(analysis/'observed_cases.csv',index=False)
    pd.concat(all_trials,ignore_index=True).to_csv(analysis/'all_trials.csv',index=False)
    pred=pd.read_csv(output/'frozen/predictions.csv')
    merged=pred.merge(observed, on=['case_id','workload_id','kind','batch_size','sequence_length','new_tokens','gpu_layers','layout'],validate='one_to_one')
    primary=merged[merged.kind=='primary'].copy()
    metrics=[]
    for v in VARIANTS:
        for m in METRICS:
            error=(primary[f'predicted_{m}_{v}']/primary[m+'_median']-1).abs()*100
            primary[f'{m}_ape_pct_{v}']=error
            metrics.append(dict(variant=v,target=m,median_ape_pct=float(error.median()),worst_ape_pct=float(error.max())))
    gpu=primary[primary.gpu_layers>0]
    memory_error=(gpu.predicted_peak_mib/gpu.gpu_peak_mib-1).abs()*100
    primary.to_csv(analysis/'primary_prediction_errors.csv',index=False)
    pd.DataFrame(metrics).to_csv(analysis/'prediction_metrics.csv',index=False)
    decisions=pd.read_csv(output/'frozen/decisions.csv')
    scored=evaluate_policy_decisions(decisions,observed[observed.kind=='primary'])
    scored.to_csv(analysis/'policy_scores.csv',index=False)
    policy_metrics=[]
    for policy,g in scored.groupby('policy',sort=False):
        regrets=g.regret_pct.dropna()
        policy_metrics.append(dict(policy=policy,decisions=len(g),budget_violations=int(g.observed_budget_violation.sum()),
            abstentions=int((g.outcome=='abstained').sum()),
            median_feasible_regret_pct=None if regrets.empty else float(regrets.median()),
            worst_feasible_regret_pct=None if regrets.empty else float(regrets.max())))
    pd.DataFrame(policy_metrics).to_csv(analysis/'policy_metrics.csv',index=False)
    history=pd.read_csv(output/'frozen/historical_calibration.csv')
    controls=observed[observed.kind=='historical_control'].merge(history[KEYS+['generation_ms_median']],on=KEYS,suffixes=('','_historical'),validate='one_to_one')
    controls['generation_ratio_vs_historical']=controls.generation_ms_median/controls.generation_ms_median_historical
    controls.to_csv(analysis/'historical_controls.csv',index=False)
    runtimes=[read_json(output/'measurement'/f'block_{b}'/'runtime.json') for b in range(2)]
    summary=dict(coverage, primary_cases=len(primary), primary_generation_trials=int(primary.trials.sum()),
        control_generation_trials=int(controls.trials.sum()),
        memory_median_ape_pct_gpu_only=float(memory_error.median()),memory_worst_ape_pct_gpu_only=float(memory_error.max()),
        prediction_metrics=metrics,policy_metrics=policy_metrics,
        runtime_differences=[r['differences_from_calibration'] for r in runtimes],
        timing_distributions_retained=True, source_seal=verified,
        limitations=['Only four new prompt/batch workload shapes at output length 32.',
                     'Two blocks are not independent sessions; ten trials per setting do not establish stable p95.',
                     'Historical controls diagnose environment changes but never rescale frozen predictions.',
                     'Peak allocated-memory budgets are synthetic, not enforced physical device limits.',
                     'Best measured feasible reference is restricted to the five supported prefix candidates.',
                     'New measurements are now test data; any subsequent model tuning must use another future test.'])
    write_json(analysis/'summary.json',summary)
    text=['# Step 3: fresh-workload evaluation','',f'Completed {len(primary)} primary configurations and 3 historical controls. '
          f'{int(observed.trials.sum())} measured generations across two blocks.','',
          '## Frozen latency predictions','', '| Predictor | Target | Median error | Worst error |','|---|---|---:|---:|']
    for r in metrics:text.append(f"| {r['variant']} | {r['target']} | {r['median_ape_pct']:.2f}% | {r['worst_ape_pct']:.2f}% |")
    text+=['','## Policy decisions','', '| Policy | Decisions | Violations | Worst feasible regret |','|---|---:|---:|---:|']
    for r in policy_metrics:text.append(f"| {r['policy']} | {r['decisions']} | {r['budget_violations']} | {r['worst_feasible_regret_pct']}% |")
    text+=['','## Interpretation limits','']+['- '+x for x in summary['limitations']]
    text+=['','Inspect historical_controls.csv and each block runtime.json before attributing error to the predictor. '
           'Do not retune the margin, drop slow trials, or claim a policy improvement unless the measured comparison supports it.','']
    (analysis/'report.md').write_text('\n'.join(text),encoding='utf-8')
    return summary


def export(output: Path) -> Path:
    require(output.is_dir(),'Output directory is missing.')
    # Integrity is reported even for a partial run. Do not suppress failures.
    try:integrity=verify(output)
    except Exception as exc:integrity={'status':'failed','error':str(exc)}
    write_json(output/'export_status.json',dict(exported_utc=now(),integrity=integrity,
               complete_analysis=(output/'analysis/summary.json').exists() and
                   (output/'analysis/coverage.json').exists() and
                   read_json(output/'analysis/coverage.json').get('complete') is True and
                   integrity.get('status')=='verified'))
    target=output.parent/(output.name+'_export.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(output.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(output).as_posix())
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--step2',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    for name in ('verify','analyze','export'):
        p=sub.add_parser(name);p.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='prepare':result=prepare(a.step2,a.out)
    elif a.command=='verify':result=verify(a.out)
    elif a.command=='analyze':result=analyze(a.out)
    else:result=str(export(a.out))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
