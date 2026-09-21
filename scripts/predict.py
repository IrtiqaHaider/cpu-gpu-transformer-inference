#!/usr/bin/env python3
"""Prediction demonstration only; no GPU work, measured-speed claim or refitting."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from offload_research.fresh import load_planner
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=int,default=3)
    p.add_argument('--prompt',type=int,default=256)
    p.add_argument('--output',type=int,default=32)
    p.add_argument('--budget-mib',type=float,default=640)
    p.add_argument('--policy',choices=['max_gpu','compute','compute_transfer'],default='compute_transfer')
    args=p.parse_args()
    result=load_planner(ROOT/'models').recommend(batch_size=args.batch,sequence_length=args.prompt,
        new_tokens=args.output,gpu_budget_mib=args.budget_mib,policy=args.policy)
    print(json.dumps(result,indent=2))
