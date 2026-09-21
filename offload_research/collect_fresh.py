"""T4-only prospective collector using the original, unchanged timed functions.

Run preflight, reference, then collect --block 0 and --block 1. All modes verify
the pre-measurement seal. The reference process exits before memory measurement.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from hetero.benchmark import (configure_cpu, metadata, timed_prefill, timed_generation,
                             reference_outputs, check_placement)
from hetero.model import GPT2, GPT2Spec
from hetero.engine import OffloadEngine
from .fresh import (verify, read_json, write_json, now, sha, require,
                    environment_differences, require_t4, METRICS)


def session_boot_id():
    path=Path('/proc/sys/kernel/random/boot_id')
    return path.read_text().strip() if path.is_file() else None


def runtime_snapshot(protocol, verified):
    fake=SimpleNamespace(model=SimpleNamespace(source={'model_id':protocol['model_id'],
                        'revision':protocol['revision']}, cfg=GPT2Spec()))
    current=metadata(fake,dict(phase='step3',seal_sha256=verified['seal_sha256']))
    current['boot_id']=session_boot_id()
    current['created_utc']=now()
    current['load_average']=list(os.getloadavg()) if hasattr(os,'getloadavg') else None
    return current


def session_fingerprint(runtime):
    # A mismatch requires a separate session, not an unnoticed resume of block 2.
    fields={k:runtime.get(k) for k in ('boot_id','gpu_name','cpu_count_logical','cuda_runtime','packages')}
    return sha(json.dumps(fields,sort_keys=True).encode())


def setup(output):
    verified=verify(output)
    protocol=read_json(output/'frozen/protocol.json')
    configure_cpu(protocol['cpu_threads'],protocol['input_seed'])
    current=runtime_snapshot(protocol,verified)
    require_t4(current)
    baseline=read_json(output/'frozen/calibration_metadata.json')
    return protocol,verified,current,environment_differences(current,baseline)


def preflight(output):
    p,v,current,changes=setup(output)
    payload=dict(status='ok',seal_sha256=v['seal_sha256'],runtime=current,
                 session_fingerprint=session_fingerprint(current),
                 differences_from_calibration=changes,
                 note='CPU/software differences are retained as domain shift; they do not trigger calibration.')
    target=output/'runtime_preflight.json'
    if target.exists():
        old=read_json(target)
        require(old['session_fingerprint']==payload['session_fingerprint'],
                'The runtime changed. Preserve/export the partial run instead of mixing sessions.')
        return old
    write_json(target,payload)
    print('T4 gate passed. Calibration environment differences:',len(changes),flush=True)
    for d in changes:print(d,flush=True)
    return payload


def verify_session(output,current,seal):
    checked=read_json(output/'runtime_preflight.json')
    require(checked['seal_sha256']==seal['seal_sha256'],'Preflight used a different freeze.')
    require(checked['session_fingerprint']==session_fingerprint(current),
            'Session or package configuration changed since preflight. Do not mix measurements.')


def make_tokens(b,s,seed,vocab):
    return torch.randint(vocab,(b,s),generator=torch.Generator().manual_seed(seed+b*1009+s))


@torch.inference_mode()
def compare_teacher_forced(engine, prompt, continuation, expected, gpu_layers):
    engine.set_placement(gpu_layers,'prefix')
    actual,cache=engine.forward(prompt)
    errors=[];agreements=[]
    def compare(a,b):
        a=a.cpu()
        torch.testing.assert_close(a,b,atol=5e-4 if gpu_layers==0 else 1e-3,rtol=1e-3)
        errors.append(float((a-b).abs().max().item()))
        agreements.append(float((a.argmax(-1)==b.argmax(-1)).float().mean().item()))
    compare(actual,expected[0])
    for index in range(continuation.shape[1]):
        actual,cache=engine.forward(continuation[:,index:index+1],cache)
        compare(actual,expected[index+1])
    return dict(gpu_layers=gpu_layers,prefill_max_abs_error=errors[0],
                decode_max_abs_error=max(errors[1:]),argmax_agreements=agreements,
                teacher_forced_decode_steps=continuation.shape[1],passed=True)


@torch.inference_mode()
def reference(output):
    p,v,current,changes=setup(output);verify_session(output,current,v)
    target=output/'reference_validation.json'
    require(not target.exists(),'Reference validation already exists; do not overwrite it.')
    # Only this numerical-validation process imports the reference library.
    from transformers import GPT2LMHeadModel
    import transformers
    require(transformers.__version__=='4.48.3','Reference gate expects transformers==4.48.3.')
    reference_model=GPT2LMHeadModel.from_pretrained(p['model_id'],revision=p['revision'],
        use_safetensors=True,attn_implementation='eager',torch_dtype=torch.float32).cpu().eval()
    fixtures=[]
    for b in (1,3):
        for s in (64,256):
            ids=make_tokens(b,s,p['input_seed'],50257)
            continuation=make_tokens(b,3,p['input_seed']+50000+s,50257)
            r=reference_model(ids,use_cache=True)
            expected=[r.logits[:,-1:,:].clone()];cache=r.past_key_values
            del r
            for i in range(3):
                r=reference_model(continuation[:,i:i+1],past_key_values=cache,use_cache=True)
                expected.append(r.logits[:,-1:,:].clone());cache=r.past_key_values
                del r
            del cache
            fixtures.append((b,s,ids,continuation,expected))
    del reference_model;gc.collect()
    model=GPT2.from_pretrained(p['model_id'],p['revision'])
    require(model.source['revision']==p['revision'],'Loaded model revision changed.')
    engine=OffloadEngine(model);checks=[]
    payload=dict(created_utc=now(),seal_sha256=v['seal_sha256'],model=model.source,
                 tiny_random=False,cuda_available=torch.cuda.is_available(),gpu_name=current['gpu_name'],
                 reference_transformers=transformers.__version__,runtime=current,checks=checks,
                 scope='Last-position prefill logits plus 3 teacher-forced cached steps for each primary prompt shape. Not every generated token.')
    try:
        for b,s,ids,continuation,expected in fixtures:
            for k in p['candidates']:
                entry=compare_teacher_forced(engine,ids,continuation,expected,k)
                entry.update(batch_size=b,sequence_length=s)
                checks.append(entry)
                print(f'Reference check B{b} S{s} G{k}: passed',flush=True)
        payload['status']='passed'
    except Exception as exc:
        payload.update(status='failed',error=str(exc))
        write_json(target,payload);raise
    write_json(target,payload)
    return payload


def load_reference_gate(output,verified):
    gate=read_json(output/'reference_validation.json')
    require(gate.get('status')=='passed' and gate.get('tiny_random') is False,
            'A real successful pretrained reference validation is required.')
    require(gate.get('seal_sha256')==verified['seal_sha256'],'Reference gate used another freeze.')
    require(gate.get('cuda_available') is True and 'T4' in str(gate.get('gpu_name','')).split(),
            'Reference gate did not use the required GPU.')
    expected={(b,s,k) for b in (1,3) for s in (64,256) for k in (0,3,6,9,12)}
    checks=gate.get('checks',[])
    require(len(checks)==20 and {(c['batch_size'],c['sequence_length'],c['gpu_layers']) for c in checks}==expected,
            'Incomplete reference validation.')
    require(all(c.get('passed') is True and c.get('teacher_forced_decode_steps')==3 for c in checks),
            'Failed or incomplete reference comparison.')
    return gate


def collect_case(engine, case, protocol, folder, block, seal, references, check_ids):
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'started.json',dict(case,block=block,created_utc=now(),seal_sha256=seal['seal_sha256']))
    values=[]
    try:
        gc.collect();engine.set_placement(case['gpu_layers'],case['layout'])
        correctness=check_placement(engine,check_ids,references)
        ids=make_tokens(case['batch_size'],case['sequence_length'],protocol['input_seed'],engine.model.cfg.vocab_size)
        if engine.uses_cuda:
            engine.synchronize();torch.cuda.empty_cache()
        for _ in range(protocol['warmup_per_case']):
            timed_prefill(engine,ids);timed_generation(engine,ids,case['new_tokens'])
        gc.collect()
        if engine.uses_cuda:
            engine.synchronize();baseline=torch.cuda.memory_allocated(0);torch.cuda.reset_peak_memory_stats(0)
        else:baseline=0
        for trial in range(protocol['measured_repetitions_per_case_per_block']):
            result={**timed_prefill(engine,ids),**timed_generation(engine,ids,case['new_tokens'])}
            result['decode_step_ms_json']=json.dumps(result.pop('decode_step_ms'))
            row={**case,'block':block,'trial':trial,**result};values.append(row)
            # Outside all timed intervals; retain trials even if Colab is interrupted.
            with (folder/'trials.partial.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n');f.flush()
        memory={'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(0) if engine.uses_cuda else 0,
                'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(0) if engine.uses_cuda else 0,
                'gpu_baseline_allocated_bytes':baseline,**engine.resident_bytes()}
        summary=dict(case,block=block,status='ok',created_utc=now(),seal_sha256=seal['seal_sha256'],
                     **memory,**correctness)
        for metric in METRICS:
            summary[metric+'_median']=float(np.median([row[metric] for row in values]))
        pd.DataFrame(values).to_csv(folder/'trials.csv',index=False)
        write_json(folder/'summary.json',summary)
        return summary
    except BaseException as exc:
        status='cuda_oom' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        summary=dict(case,block=block,status=status,error=str(exc),created_utc=now(),
                     completed_trials=len(values),seal_sha256=seal['seal_sha256'])
        write_json(folder/'summary.json',summary)
        (folder/'failure.txt').write_text(traceback.format_exc())
        if status!='cuda_oom':raise
        return summary


def collect(output,block):
    p,v,current,changes=setup(output);verify_session(output,current,v)
    require(block in (0,1),'Block must be 0 or 1.')
    gate=load_reference_gate(output,v)
    require(session_fingerprint(gate['runtime'])==session_fingerprint(current),
            'Reference validation and measurements belong to different sessions.')
    if block==1:
        require((output/'measurement/block_0/completed.json').exists(),'Complete block 0 before block 1.')
    destination=output/'measurement'/f'block_{block}'
    require(not destination.exists(),'This block already started. Keep/export it; no silent rerun or partial-case replacement.')
    destination.mkdir(parents=True)
    write_json(destination/'runtime.json',dict(runtime=current,differences_from_calibration=changes,
              seal_sha256=v['seal_sha256'],reference_sha256=sha((output/'reference_validation.json').read_bytes())))
    model=GPT2.from_pretrained(p['model_id'],p['revision'])
    require(model.source['revision']==p['revision'],'Pinned revision mismatch.')
    require(asdict(model.cfg)==read_json(output/'frozen/calibration_metadata.json')['model_config'],'Model config mismatch.')
    engine=OffloadEngine(model)
    check_ids=make_tokens(2,8,p['input_seed'],model.cfg.vocab_size)
    references=reference_outputs(engine,check_ids)
    indexed={c['case_id']:c for c in p['cases']}
    completed=[]
    for index,case_id in enumerate(p['blocks'][block]):
        verify(output)
        print(f'Block {block+1}/2 — case {index+1}/23: {case_id}',flush=True)
        result=collect_case(engine,indexed[case_id],p,destination/case_id,block,v,references,check_ids)
        completed.append(result)
        pd.DataFrame(completed).to_csv(destination/'summary.csv',index=False)
        print('  '+result['status'],flush=True)
    verify(output)
    write_json(destination/'completed.json',dict(status='completed',created_utc=now(),cases=len(completed),
              successful_cases=sum(r['status']=='ok' for r in completed),seal_sha256=v['seal_sha256']))
    return dict(block=block,cases=len(completed),successful_cases=sum(r['status']=='ok' for r in completed))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preflight','reference','collect'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--block',type=int,choices=(0,1))
    args=parser.parse_args()
    if args.command=='preflight':result=preflight(args.out)
    elif args.command=='reference':result=reference(args.out)
    else:
        require(args.block is not None,'Specify --block 0 or 1.')
        result=collect(args.out,args.block)
    if args.command=='reference':
        print(json.dumps({'status':result['status'],'checks':len(result['checks']),'output':str(args.out/'reference_validation.json')},indent=2))
    elif args.command=='preflight':
        print(json.dumps({'status':result['status'],'gpu':result['runtime']['gpu_name'],'environment_differences':result['differences_from_calibration']},indent=2))
    else:
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
