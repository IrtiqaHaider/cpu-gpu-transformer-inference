"""T4 collection for Step 4, with unchanged model and timed functions."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import resource

import pandas as pd
import torch
from hetero.benchmark import configure_cpu, reference_outputs
from hetero.engine import OffloadEngine
from hetero.model import GPT2
from .collect_fresh import (runtime_snapshot, session_fingerprint, make_tokens,
                           compare_teacher_forced, collect_case)
from .fresh import read_json, write_json, now, sha, require, environment_differences
from .robustness import verify


def require_runtime(current: dict, threads: int) -> None:
    require(current.get('cuda_available') is True and 'T4' in str(current.get('gpu_name','')).split(),
            'Step 4 requires a T4 GPU. No silent accelerator substitution.')
    require(threads in (1,2) and current.get('torch_threads')==threads and
            current.get('torch_interop_threads')==1, 'Unexpected CPU thread settings.')
    require(bool(current.get('boot_id')), 'Cannot establish VM identity: boot ID unavailable.')


def different_vm(current: dict, parent: dict) -> None:
    require(bool(parent.get('boot_id')) and bool(current.get('boot_id')), 'Missing VM identity.')
    require(current['boot_id'] != parent['boot_id'],
            'This is the same VM as Step 3. Export any partial files, disconnect/delete the old Colab runtime, '
            'then use a new T4 VM. Restarting only the Python kernel is not enough.')


def setup(output: Path, threads: int):
    sealed=verify(output);p=read_json(output/'frozen/protocol.json')
    configure_cpu(threads,p['input_seed'])
    runtime=runtime_snapshot(p,sealed)
    runtime['arguments']['phase']='step4'
    require_runtime(runtime,threads)
    parent=read_json(output/'parent/runtime_preflight.json')['runtime']
    different_vm(runtime,parent)
    return p,sealed,runtime,parent


def cpu_snapshot() -> dict:
    """Read accessible counters only; never change cgroup limits or affinity."""
    def text(path):
        try:return Path(path).read_text().strip()
        except (OSError,UnicodeError):return None
    usage=resource.getrusage(resource.RUSAGE_SELF)
    raw=text('/proc/self/cgroup'); cgroups={}
    candidates=[Path('/sys/fs/cgroup')]
    if raw:
        for line in raw.splitlines():
            pieces=line.split(':',2)
            if len(pieces)==3 and pieces[:2]==['0','']:
                target=(Path('/sys/fs/cgroup')/pieces[2].lstrip('/')).resolve()
                if target.is_relative_to(Path('/sys/fs/cgroup').resolve()):candidates.append(target)
    for directory in dict.fromkeys(candidates):
        values={n:text(directory/n) for n in ('cpu.max','cpu.stat','cpu.pressure')}
        if any(v is not None for v in values.values()):cgroups[str(directory)]=values
    return dict(created_utc=now(),load_average=list(os.getloadavg()) if hasattr(os,'getloadavg') else None,
        process_user_seconds=usage.ru_utime,process_system_seconds=usage.ru_stime,
        voluntary_context_switches=usage.ru_nvcsw,involuntary_context_switches=usage.ru_nivcsw,
        process_cgroup_membership=raw,accessible_cgroups=cgroups,
        scope='Whole case, not only measured intervals. Cgroup counters can cover other processes; '
              'root or missing counters cannot exclude ancestor/host limits.')


def preflight(output: Path):
    p,sealed,current,parent=setup(output,2)
    target=output/'runtime_preflight.json'
    payload=dict(status='ok',created_utc=now(),runtime=current,parent_boot_id=parent['boot_id'],
        seal_sha256=sealed['seal_sha256'],session_fingerprint=session_fingerprint(current),
        differences_from_step3=environment_differences(current,parent),
        cpu_diagnostics=cpu_snapshot())
    if target.exists():
        old=read_json(target)
        require(old['session_fingerprint']==payload['session_fingerprint'],
                'VM/packages changed within Step 4. Preserve partial results; do not mix sessions.')
        require(old['seal_sha256']==sealed['seal_sha256'],'Preflight freeze differs.')
        return old
    write_json(target,payload)
    return payload


def check_saved_runtime(output: Path, current: dict, threads: int):
    require_runtime(current,threads)
    checked=read_json(output/'runtime_preflight.json')
    require(checked['seal_sha256']==verify(output)['seal_sha256'], 'Preflight used another freeze.')
    require(session_fingerprint(current)==checked['session_fingerprint'],
            'Runtime or packages changed since preflight. Export; do not resume in another VM.')
    require(current['boot_id']!=checked['parent_boot_id'], 'Replication uses the Step 3 VM.')


@torch.inference_mode()
def reference(output: Path, threads: int):
    p,sealed,current,parent=setup(output,threads)
    check_saved_runtime(output,current,threads)
    target=output/f'reference_threads_{threads}.json'
    require(not target.exists(),'Reference already attempted. Do not overwrite a failed gate.')
    payload=dict(status='started',created_utc=now(),seal_sha256=sealed['seal_sha256'],
        cpu_threads=threads,runtime=current,tiny_random=False,checks=[],
        scope='Last-position prompt logits and 3 teacher-forced cache steps; not all generated tokens.')
    write_json(target,payload)
    try:
        from transformers import GPT2LMHeadModel
        import transformers
        require(transformers.__version__=='4.48.3','Use the pinned Transformers 4.48.3 reference.')
        ref=GPT2LMHeadModel.from_pretrained(p['model_id'],revision=p['revision'],
            use_safetensors=True,attn_implementation='eager',torch_dtype=torch.float32).cpu().eval()
        fixtures={}
        for b,s in sorted({(c['batch_size'],c['sequence_length']) for c in p['cases']}):
            ids=make_tokens(b,s,p['input_seed'],50257)
            continuation=make_tokens(b,3,p['input_seed']+50000+s,50257)
            r=ref(ids,use_cache=True);cache=r.past_key_values
            expected=[r.logits[:,-1:,:].clone()];del r
            for i in range(3):
                r=ref(continuation[:,i:i+1],past_key_values=cache,use_cache=True)
                expected.append(r.logits[:,-1:,:].clone());cache=r.past_key_values;del r
            del cache
            fixtures[(b,s)]=(ids,continuation,expected)
        del ref;gc.collect()
        model=GPT2.from_pretrained(p['model_id'],p['revision'])
        require(model.source['revision']==p['revision'],'Checkpoint revision changed.')
        engine=OffloadEngine(model)
        for case in p['cases']:
            ids,continuation,expected=fixtures[(case['batch_size'],case['sequence_length'])]
            check=compare_teacher_forced(engine,ids,continuation,expected,case['gpu_layers'])
            check.update(case_id=case['case_id'],batch_size=case['batch_size'],sequence_length=case['sequence_length'])
            payload['checks'].append(check)
            print(f"Reference T{threads} {case['case_id']}: passed",flush=True)
        payload.update(status='passed',model=model.source,reference_transformers=transformers.__version__)
    except BaseException as exc:
        payload.update(status='failed',error=str(exc));write_json(target,payload);raise
    write_json(target,payload)
    return dict(status='passed',cpu_threads=threads,checks=len(payload['checks']))


def load_reference_gate(output: Path, threads: int, seal: dict):
    gate=read_json(output/f'reference_threads_{threads}.json')
    p=read_json(output/'frozen/protocol.json')
    expected={c['case_id'] for c in p['cases']}
    require(gate.get('status')=='passed' and gate.get('tiny_random') is False,
            'A successful real pretrained reference gate is required.')
    require(gate.get('cpu_threads')==threads and gate.get('seal_sha256')==seal['seal_sha256'],
            'Reference used another thread setting or freeze.')
    require(gate.get('model')=={'model_id':p['model_id'],'revision':p['revision']},
            'Reference model identity mismatch.')
    require(len(gate['checks'])==13 and {c['case_id'] for c in gate['checks']}==expected,
            'Incomplete reference cases.')
    require(all(c.get('passed') is True and c.get('teacher_forced_decode_steps')==3 for c in gate['checks']),
            'Numerical comparison failed.')
    check_saved_runtime(output,gate['runtime'],threads)
    return gate


def collect(output: Path, block: int):
    require(block in (0,1,2,3),'Block must be 0,1,2,3.')
    p=read_json(output/'frozen/protocol.json');threads=p['threads_by_block'][block]
    p,sealed,current,parent=setup(output,threads)
    check_saved_runtime(output,current,threads)
    load_reference_gate(output,threads,sealed)
    if block>0:
        previous=read_json(output/f'measurement/block_{block-1}/completed.json')
        require(previous.get('successful_cases')==13,'Previous block incomplete; preserve results and stop.')
    destination=output/f'measurement/block_{block}'
    require(not destination.exists(),'This block already started. No silent retry or replacement.')
    destination.mkdir(parents=True)
    write_json(destination/'runtime.json',dict(runtime=current,cpu_threads=threads,
        differences_from_step3=environment_differences(current,parent),seal_sha256=sealed['seal_sha256'],
        reference_sha256=sha((output/f'reference_threads_{threads}.json').read_bytes())))
    model=GPT2.from_pretrained(p['model_id'],p['revision'])
    require(model.source['revision']==p['revision'],'Pinned checkpoint differs.')
    expected=read_json(output/'parent/frozen/calibration_metadata.json')['model_config']
    require(asdict(model.cfg)==expected,'Model configuration differs.')
    engine=OffloadEngine(model)
    check_ids=make_tokens(2,8,p['input_seed'],model.cfg.vocab_size)
    references=reference_outputs(engine,check_ids)
    indexed={c['case_id']:c for c in p['cases']};completed=[]
    for index,case_id in enumerate(p['blocks'][block]):
        verify(output)
        telemetry=destination/(case_id+'_telemetry.json')
        before=cpu_snapshot()
        print(f'Block {block+1}/4, threads={threads}, case {index+1}/13: {case_id}',flush=True)
        try:
            s=collect_case(engine,indexed[case_id],p,destination/case_id,block,sealed,references,check_ids)
        finally:
            write_json(telemetry,dict(before=before,after=cpu_snapshot()))
        completed.append(s)
        pd.DataFrame(completed).to_csv(destination/'summary.csv',index=False)
        if s.get('status')!='ok':
            require(False,'Case failed or ran out of memory. Export this run; do not continue to a favorable retry.')
    verify(output)
    write_json(destination/'completed.json',dict(status='completed',created_utc=now(),cases=13,
        successful_cases=sum(s['status']=='ok' for s in completed),cpu_threads=threads,
        seal_sha256=sealed['seal_sha256']))
    return dict(block=block,cpu_threads=threads,successful_cases=len(completed))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preflight','reference','collect'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--threads',type=int,choices=(1,2))
    parser.add_argument('--block',type=int,choices=(0,1,2,3))
    a=parser.parse_args()
    if a.command=='preflight':
        result=preflight(a.out)
        result={k:result[k] for k in ('status','parent_boot_id','differences_from_step3')}
    elif a.command=='reference':
        require(a.threads is not None,'Specify --threads 1 or 2.');result=reference(a.out,a.threads)
    else:
        require(a.block is not None,'Specify --block.');result=collect(a.out,a.block)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
