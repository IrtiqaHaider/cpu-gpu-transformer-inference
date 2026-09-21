"""Step 4: frozen-model replication and CPU-thread sensitivity.

This is a post-Step-3 robustness study, not another blind workload test.
Nothing here refits the latency or memory models or changes the default guard.
"""
from __future__ import annotations
import argparse
import copy
import io
import json
from pathlib import Path, PurePosixPath
import random
import shutil
import stat
import zipfile

import numpy as np
import pandas as pd
from . import fresh
from .fresh import (now, sha, read_json, write_json, require, METRICS, VARIANTS,
                    evaluate_policy_decisions, aggregate_case, load_planner)
from .planner import choose

EXPECTED_STEP3_SHA256 = 'a24306bcc49770085a586b6b717c3490ce272fd227a4b40e53d14a00dc0b4181'
ROOT = Path(__file__).resolve().parent.parent
GUARDS = (('point_only', 0.0, 0.0), ('16_MiB', 0.0, 16.0),
          ('5_percent', 0.05, 0.0), ('original_5_percent_plus_16_MiB', 0.05, 16.0))


def safe_extract(archive: Path, destination: Path) -> None:
    """Bounded extraction; no executable code is imported from an input archive."""
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        require(len(infos) <= 10000, 'Archive has too many entries.')
        require(sum(i.file_size for i in infos) <= 256 * 2**20, 'Archive is too large.')
        seen = set()
        for i in infos:
            p = PurePosixPath(i.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename,
                    'Unsafe archive path.')
            require(i.filename not in seen, 'Duplicate archive member.')
            require(not stat.S_ISLNK(i.external_attr >> 16), 'Archive symlinks are not supported.')
            seen.add(i.filename)
        destination.mkdir(parents=True, exist_ok=False)
        z.extractall(destination)


def make_protocol(parent: dict) -> dict:
    cases = [copy.deepcopy(c) for c in parent['cases']
             if c['batch_size'] == 3 or c['kind'] == 'historical_control']
    require(len(cases) == 13, 'Expected ten batch-three settings and three controls.')
    order_a = [c['case_id'] for c in cases]
    order_b = order_a.copy()
    random.Random(20260921).shuffle(order_a)
    random.Random(20260922).shuffle(order_b)
    p = copy.deepcopy(parent)
    p.update(version='step4-v0.4.0', created_utc=now(), cases=cases,
             question='Do selected frozen-model results replicate in a new T4 VM, and how sensitive are they to CPU thread count?',
             cpu_threads=None, threads_by_block=[2, 1, 1, 2],
             blocks=[order_a, order_b, list(reversed(order_b)), list(reversed(order_a))],
             expected_case_block_records=52, expected_generation_trials=260,
             expected_policy_decisions_per_thread_setting=48,
             expected_reference_checks_per_thread_setting=13,
             expected_primary_policy_decisions=96,
             primary_workloads=2, primary_candidates=10, historical_controls=3,
             block_scope='Four ABBA thread-order blocks in ONE new VM; not four independent sessions.',
             selection='Cases selected after Step 3 to investigate batch-three prediction error and a guard-induced boundary penalty. Not a blind test.',
             controls='B1/S128/N32 at G0/G6/G12 retained; no rescaling or refitting.',
             numerical_gate='Each of 13 cases at one AND two CPU threads, HF last-position logits + 3 teacher-forced cached steps.',
             future_test_status='Prospective measurements on post-hoc selected replication workloads.',
             thread_shift='One-thread results challenge an unchanged two-thread-calibrated model; report separately.',
             objective='Median complete-generation latency within each CPU-thread setting.',
             memory_target='Maximum peak PyTorch allocation over both blocks within each thread setting.',
             environment='Require T4 and a different boot ID from Step 3. Keep software/CPU differences visible.',
             new_session_required=True,
             robustness_claim='A different VM boot ID documents runtime separation, not statistical independence.',
             telemetry_scope='Before/after each whole case, including placement, checks and warmups. Not per-trial causal attribution.',
             margin_sensitivity='Exploratory re-scoring only. Original guard remains 5% + 16 MiB for primary results.')
    return p


def audit_parent(parent: Path) -> dict:
    verified = fresh.verify(parent)
    saved = read_json(parent/'analysis/summary.json')
    p = read_json(parent/'frozen/protocol.json')
    predictions, decisions = fresh.prediction_tables(load_planner(parent/'frozen/models'), p)
    for name, actual in [('predictions', predictions), ('decisions', decisions)]:
        pd.testing.assert_frame_equal(pd.read_csv(parent/f'frozen/{name}.csv'), actual,
                                      check_exact=False, rtol=1e-8, atol=1e-8)
    recomputed = fresh.analyze(parent)
    require(recomputed == saved, 'Step 3 analysis did not reproduce exactly.')
    require(saved.get('complete') is True and saved['primary_generation_trials'] == 200,
            'Step 3 is incomplete.')
    return dict(integrity=verified, predictions_reproduced=True, decisions_reproduced=True,
                analysis_reproduced=True, new_gpu_inference=False, summary=saved,
                uploaded_test_log=(parent/'test_output.txt').read_text())


def margin_sensitivity(predictions: pd.DataFrame, observations: pd.DataFrame,
                       budgets: list, policies: list) -> pd.DataFrame:
    """Post-hoc counterfactuals from frozen point predictions; never refit a guard."""
    primary = predictions[predictions.kind == 'primary']
    records = []
    for label, relative, extra in GUARDS:
        choices = []
        for wid, group in primary.groupby('workload_id', sort=True):
            table = group.copy()
            table['guarded_peak_mib'] = np.where(table.gpu_layers == 0, 0.0,
                                                table.predicted_peak_mib * (1 + relative) + extra)
            for budget in budgets:
                for policy in policies:
                    picked = choose(table, budget, policy)
                    choices.append(dict(workload_id=wid, budget_mib=budget, policy=policy,
                        selected_gpu_layers=None if picked is None else int(picked['gpu_layers']),
                        selection_status='abstained' if picked is None else 'predicted_feasible'))
        scored = evaluate_policy_decisions(pd.DataFrame(choices), observations[observations.kind == 'primary'])
        scored['guard'] = label
        scored['scope'] = 'exploratory counterfactual; not deployed or tuned'
        records.append(scored)
    return pd.concat(records, ignore_index=True)


def prepare(step3: Path, output: Path) -> dict:
    require(step3.is_file() and sha(step3.read_bytes()) == EXPECTED_STEP3_SHA256,
            'Upload the unchanged step3_20260919T202327166768Z_export.zip.')
    require(not output.exists(), 'Output already exists. Do not overwrite a run.')
    output.mkdir(parents=True)
    (output/'inputs').mkdir()
    shutil.copyfile(step3, output/'inputs/step3_export.zip')
    parent = output/'parent'
    safe_extract(step3, parent)
    audit = audit_parent(parent)
    write_json(output/'step3_audit.json', audit)
    frozen = output/'frozen'; frozen.mkdir()
    protocol = make_protocol(read_json(parent/'frozen/protocol.json'))
    write_json(frozen/'protocol.json', protocol)
    cases = {c['case_id'] for c in protocol['cases']}
    predictions = pd.read_csv(parent/'frozen/predictions.csv')
    predictions[predictions.case_id.isin(cases)].to_csv(frozen/'predictions.csv', index=False)
    decisions = pd.read_csv(parent/'frozen/decisions.csv')
    workloads = {c['workload_id'] for c in protocol['cases'] if c['kind'] == 'primary'}
    decisions[decisions.workload_id.isin(workloads)].to_csv(frozen/'decisions.csv', index=False)
    # Copy model bytes; do not fit models and do not silently choose a different margin.
    shutil.copytree(parent/'frozen/models', frozen/'models')
    source_hashes = {}
    for package in ('hetero', 'offload_research'):
        for path in sorted((ROOT/package).glob('*.py')):
            rel = path.relative_to(ROOT).as_posix()
            target = frozen/'code_snapshot'/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            source_hashes[rel] = sha(path.read_bytes())
    hashes = {f.relative_to(frozen).as_posix():sha(f.read_bytes())
              for f in frozen.rglob('*') if f.is_file()}
    write_json(frozen/'seal.json', dict(schema='step4-seal-v1', created_utc=now(),
        input_sha256=EXPECTED_STEP3_SHA256,
        parent_seal_sha256=audit['integrity']['seal_sha256'],
        file_sha256=hashes, executable_source_sha256=source_hashes,
        scope='Local integrity seal, not public preregistration or a trusted external timestamp.'))
    exploratory = output/'step3_exploratory'; exploratory.mkdir()
    scores = margin_sensitivity(pd.read_csv(parent/'frozen/predictions.csv'),
        pd.read_csv(parent/'analysis/observed_cases.csv'), protocol['budgets_mib'], protocol['policies'])
    scores.to_csv(exploratory/'margin_sensitivity.csv', index=False)
    scores.groupby(['guard','policy'], sort=False).agg(
        decisions=('outcome','size'), violations=('observed_budget_violation','sum'),
        median_regret_pct=('regret_pct','median'), worst_regret_pct=('regret_pct','max')
    ).reset_index().to_csv(exploratory/'margin_summary.csv', index=False)
    return dict(verify(output), expected_generation_trials=260, refitting=False,
                note='Parent results audited. No new GPU work has run.')


def verify(output: Path) -> dict:
    frozen = output/'frozen'; seal = read_json(frozen/'seal.json')
    require(sha((output/'inputs/step3_export.zip').read_bytes()) == seal['input_sha256'],
            'Parent archive changed.')
    parent = fresh.verify(output/'parent')
    require(parent['seal_sha256'] == seal['parent_seal_sha256'], 'Parent freeze changed.')
    for base, mapping in ((frozen, seal['file_sha256']), (ROOT, seal['executable_source_sha256'])):
        for name, digest in mapping.items():
            target = (base/name).resolve()
            require(target.is_relative_to(base.resolve()), 'Unsafe seal path.')
            require(target.is_file() and sha(target.read_bytes()) == digest, 'Sealed file changed: '+name)
    return dict(status='verified', seal_sha256=sha((frozen/'seal.json').read_bytes()),
                parent_seal_sha256=parent['seal_sha256'], new_gpu_inference=False)


def validate_trial_table(case: dict, table: pd.DataFrame, block: int, repeats: int) -> None:
    require(list(table.trial) == list(range(repeats)), 'Missing or duplicate trials.')
    require((table.block == block).all(), 'Wrong block number.')
    for key in ('case_id','workload_id','kind','batch_size','sequence_length','new_tokens','gpu_layers','layout'):
        require((table[key] == case[key]).all(), 'Trial identity mismatch: '+key)
    require((table.final_cached_positions == case['sequence_length'] + case['new_tokens'] - 1).all(),
            'Cache length mismatch.')
    positions = case['sequence_length'] + case['new_tokens'] - 1
    per_layer_bytes = 2 * case['batch_size'] * positions * 768 * 4
    require((table.final_kv_gpu_bytes == case['gpu_layers'] * per_layer_bytes).all(), 'GPU KV bytes mismatch.')
    require((table.final_kv_cpu_bytes == (12-case['gpu_layers']) * per_layer_bytes).all(), 'CPU KV bytes mismatch.')
    for row in table.to_dict('records'):
        steps = json.loads(row['decode_step_ms_json'])
        require(len(steps) == case['new_tokens']-1 and np.isfinite(steps).all() and min(steps)>0,
                'Invalid cached step sequence.')
        require(sum(steps) <= row['decode_ms'] + 1e-5, 'Step timing exceeds decode total.')
        for metric, expected in (
            ('prefill_input_tokens_per_s', case['batch_size']*case['sequence_length']*1000/row['prefill_ms']),
            ('decode_generated_tokens_per_s', case['batch_size']*(case['new_tokens']-1)*1000/row['decode_ms']),
            ('generation_generated_tokens_per_s', case['batch_size']*case['new_tokens']*1000/row['generation_ms'])):
            require(np.isclose(row[metric], expected, rtol=1e-7, atol=1e-7), 'Throughput accounting mismatch.')


def distribution(values) -> dict:
    x = np.asarray(values, dtype=float)
    require(len(x)>0 and np.isfinite(x).all() and (x>0).all(), 'Invalid timings.')
    median = float(np.median(x))
    return dict(generation_iqr_over_median=float((np.quantile(x,.75)-np.quantile(x,.25))/median),
                generation_max_over_median=float(x.max()/median),
                slow_trials_over_1_5x_median=int((x>1.5*median).sum()))


def score_observations(predictions: pd.DataFrame, decisions: pd.DataFrame,
                       observed: pd.DataFrame):
    primary = observed[observed.kind == 'primary']
    keys = ['case_id','workload_id','kind','batch_size','sequence_length','new_tokens','gpu_layers','layout']
    joined = predictions.merge(primary, on=keys, validate='one_to_one')
    metrics = []
    for variant in VARIANTS:
        for metric in METRICS:
            error = (joined[f'predicted_{metric}_{variant}']/joined[metric+'_median']-1).abs()*100
            joined[f'{metric}_ape_pct_{variant}'] = error
            metrics.append(dict(variant=variant,target=metric,median_ape_pct=float(error.median()),
                                worst_ape_pct=float(error.max())))
    scores = evaluate_policy_decisions(decisions, primary)
    gpu = joined[joined.gpu_layers>0]
    memory = (gpu.predicted_peak_mib/gpu.gpu_peak_mib-1).abs()*100
    return joined, pd.DataFrame(metrics), scores, dict(gpu_positive_cases=len(gpu),
        median_ape_pct=float(memory.median()),worst_ape_pct=float(memory.max()))


def analyze(output: Path) -> dict:
    from .collect_robustness import check_saved_runtime, load_reference_gate
    seal = verify(output); p = read_json(output/'frozen/protocol.json')
    analysis=output/'analysis';analysis.mkdir(exist_ok=True)
    write_json(analysis/'coverage.json', dict(complete=False, status='analysis_started'))
    if (analysis/'summary.json').exists():
        (analysis/'summary.json').unlink()  # Derived report only; raw measurements are never deleted.
    (analysis/'report.md').write_text('# Step 4 analysis in progress\n\nNo current complete result yet.\n')
    rows=[]; all_trials=[]; block_rows=[]; failures=[]
    for threads in (1,2):
        blocks = [i for i,t in enumerate(p['threads_by_block']) if t == threads]
        for case in p['cases']:
            summaries=[]; trial_parts=[]
            for block in blocks:
                folder = output/'measurement'/f'block_{block}'/case['case_id']
                if not (folder/'summary.json').exists():
                    failures.append(dict(block=block,case_id=case['case_id'],status='missing'));continue
                s = read_json(folder/'summary.json')
                if s.get('status') != 'ok':
                    failures.append(dict(block=block,case_id=case['case_id'],status=s.get('status')));continue
                require(s['seal_sha256'] == seal['seal_sha256'], 'Wrong measurement freeze.')
                for key in ('case_id','workload_id','kind','batch_size','sequence_length','new_tokens','gpu_layers','layout'):
                    require(s[key] == case[key], 'Summary identity mismatch.')
                t = pd.read_csv(folder/'trials.csv')
                validate_trial_table(case,t,block,p['measured_repetitions_per_case_per_block'])
                for metric in METRICS:
                    require(np.isclose(s[metric+'_median'],t[metric].median(),rtol=1e-7,atol=1e-7),
                            'Summary does not match raw trial timings.')
                summaries.append(s);trial_parts.append(t)
                block_rows.append(dict(case_id=case['case_id'],block=block,cpu_threads=threads,
                    gpu_peak_mib=s['gpu_peak_allocated_bytes']/2**20,
                    **{m+'_median':float(t[m].median()) for m in METRICS},
                    **distribution(t.generation_ms)))
            if len(summaries)==2:
                raw=pd.concat(trial_parts,ignore_index=True)
                row=aggregate_case(case,summaries,raw,2*p['measured_repetitions_per_case_per_block'])
                row.update(cpu_threads=threads,**distribution(raw.generation_ms))
                rows.append(row);raw['cpu_threads']=threads;all_trials.append(raw)
    analysis=output/'analysis';analysis.mkdir(exist_ok=True)
    coverage=dict(complete=False,complete_case_thread_settings=len(rows),
                  expected_case_thread_settings=26,failures=failures)
    write_json(analysis/'coverage.json',coverage)
    if failures:
        (analysis/'report.md').write_text('# Step 4 incomplete\n\nPreserve all failures. No complete policy result is claimed.\n')
        return coverage
    # Do not accept timing tables without same-session, successful numerical gates.
    for block, threads in enumerate(p['threads_by_block']):
        gate=load_reference_gate(output,threads,seal)
        runtime=read_json(output/f'measurement/block_{block}/runtime.json')
        require(runtime['seal_sha256']==seal['seal_sha256'], 'Runtime seal mismatch.')
        require(runtime['reference_sha256']==sha((output/f'reference_threads_{threads}.json').read_bytes()),
                'Reference gate changed after collection.')
        check_saved_runtime(output,runtime['runtime'],threads)
        completed=read_json(output/f'measurement/block_{block}/completed.json')
        require(completed['successful_cases']==13 and completed['status']=='completed' and
                completed['seal_sha256']==seal['seal_sha256'], 'Incomplete block.')
    observed=pd.DataFrame(rows);observed.to_csv(analysis/'observed_cases.csv',index=False)
    pd.concat(all_trials,ignore_index=True).to_csv(analysis/'all_trials.csv',index=False)
    pd.DataFrame(block_rows).to_csv(analysis/'block_timing.csv',index=False)
    pred=pd.read_csv(output/'frozen/predictions.csv')
    decisions=pd.read_csv(output/'frozen/decisions.csv')
    metrics=[];policy_metrics=[];memory_metrics=[];scored=[]
    for threads in (1,2):
        obs=observed[observed.cpu_threads==threads].copy()
        errors,m,scores,mem=score_observations(pred,decisions,obs)
        errors.to_csv(analysis/f'prediction_errors_threads_{threads}.csv',index=False)
        scores['cpu_threads']=threads;scored.append(scores)
        m['cpu_threads']=threads;metrics.append(m)
        memory_metrics.append(dict(cpu_threads=threads,**mem))
        for policy,group in scores.groupby('policy',sort=False):
            regrets=group.regret_pct.dropna()
            policy_metrics.append(dict(cpu_threads=threads,policy=policy,decisions=len(group),
                budget_violations=int(group.observed_budget_violation.sum()),
                median_feasible_regret_pct=float(regrets.median()) if len(regrets) else None,
                worst_feasible_regret_pct=float(regrets.max()) if len(regrets) else None))
    pd.concat(metrics,ignore_index=True).to_csv(analysis/'prediction_metrics.csv',index=False)
    pd.concat(scored,ignore_index=True).to_csv(analysis/'policy_scores.csv',index=False)
    pd.DataFrame(policy_metrics).to_csv(analysis/'policy_metrics.csv',index=False)
    pd.DataFrame(memory_metrics).to_csv(analysis/'memory_metrics.csv',index=False)
    t1=observed[observed.cpu_threads==1];t2=observed[observed.cpu_threads==2]
    comparison=t1.merge(t2,on='case_id',suffixes=('_threads1','_threads2'),validate='one_to_one')
    comparison['generation_ratio_one_vs_two']=comparison.generation_ms_median_threads1/comparison.generation_ms_median_threads2
    comparison.to_csv(analysis/'thread_comparison.csv',index=False)
    parent=pd.read_csv(output/'parent/analysis/observed_cases.csv')
    replication=t2.merge(parent,on='case_id',suffixes=('_step4','_step3'),validate='one_to_one')
    replication['generation_ratio_step4_vs_step3']=replication.generation_ms_median_step4/replication.generation_ms_median_step3
    replication.to_csv(analysis/'session_replication.csv',index=False)
    preflight=read_json(output/'runtime_preflight.json')
    coverage['complete']=True
    summary=dict(coverage,measured_generation_trials=sum(len(t) for t in all_trials),
        thread_design='ABBA (2,1,1,2); two blocks per thread setting in one new VM',
        memory_metrics=memory_metrics,policy_metrics=policy_metrics,
        source_seal=seal,new_vm_boot_id=preflight['runtime']['boot_id'],
        parent_vm_boot_id=preflight['parent_boot_id'],different_vm=True,
        refitting=False,prediction_metrics=pd.concat(metrics).to_dict('records'),
        calibration_scope='Models remain two-thread calibrated. One-thread errors are domain-shift diagnostics.',
        limitations=['Post-hoc selected replication subset, not a second blind workload evaluation.',
            'Only one additional VM; no universal latency or tail guarantee.',
            'Budget constraint is peak allocated tensor memory, not a hard physical VRAM limit.',
            'No claim that CPU thread count alone causes all timing changes.',
            'All three policies share the same unchanged 5% + 16 MiB guard.',
            'Whole-case telemetry includes warmup, placement and checks; it is not per-trial attribution.'])
    write_json(analysis/'summary.json',summary)
    text=['# Step 4: replication and CPU-thread sensitivity','',
          '260 measured generations. Frozen model and guard unchanged.','',
          '## Prediction errors (new session only)','',
          '| Threads | Predictor | Target | Median error | Worst error |','|---:|---|---|---:|---:|']
    for r in summary['prediction_metrics']:
        if r['target']=='generation_ms':
            text.append(f"| {r['cpu_threads']} | {r['variant']} | generation | {r['median_ape_pct']:.2f}% | {r['worst_ape_pct']:.2f}% |")
    text += ['','## Policy results','','| Threads | Policy | Violations | Worst feasible regret |','|---:|---|---:|---:|']
    for r in policy_metrics:
        text.append(f"| {r['cpu_threads']} | {r['policy']} | {r['budget_violations']} | {r['worst_feasible_regret_pct']}% |")
    text += ['','## Interpretation','',
        'Read thread_comparison.csv and session_replication.csv separately. A ratio >1 means the numerator condition was slower.',
        'Do not pool old and new sessions as exchangeable trials, or turn these results into an optimization win without support.','']
    text += ['- '+x for x in summary['limitations']]
    (analysis/'report.md').write_text('\n'.join(text)+'\n')
    write_json(analysis/'coverage.json',coverage)
    return summary


def export(output: Path) -> Path:
    require(output.is_dir(), 'No output directory to export.')
    try: status=verify(output)
    except Exception as exc: status=dict(status='failed',error=str(exc))
    coverage=read_json(output/'analysis/coverage.json') if (output/'analysis/coverage.json').exists() else {}
    complete=(coverage.get('complete') is True and (output/'analysis/summary.json').exists()
              and status.get('status')=='verified')
    write_json(output/'export_status.json',dict(exported_utc=now(),integrity=status,complete_analysis=complete,
        note='parent/ working extraction is omitted; inputs/step3_export.zip preserves it byte-for-byte.'))
    target=output.parent/(output.name+'_export.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob('*')):
            relative=path.relative_to(output)
            if path.is_file() and relative.parts[0]!='parent':z.write(path,relative.as_posix())
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','analyze','export'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--step3',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        require(args.step3 is not None,'Provide --step3');result=prepare(args.step3,args.out)
    elif args.command=='verify':result=verify(args.out)
    elif args.command=='analyze':result=analyze(args.out)
    else:result=str(export(args.out))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
