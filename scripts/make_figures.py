#!/usr/bin/env python3
"""Rebuild four figures from measured CSVs. Does not fit or benchmark anything."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main(show=False):
    figures=ROOT/'docs/figures';figures.mkdir(parents=True,exist_ok=True)
    baseline=pd.read_csv(ROOT/'data/baseline/main/summary.csv')
    selected=baseline[(baseline.batch_size==1)&(baseline.sequence_length==128)].sort_values('gpu_layers')
    fig,ax=plt.subplots(figsize=(7.5,4.4))
    ax.plot(selected.gpu_peak_allocated_bytes/2**20,selected.generation_ms_median,marker='o')
    for _,r in selected.iterrows():
        ax.annotate(f'{int(r.gpu_layers)} GPU blocks',(r.gpu_peak_allocated_bytes/2**20,r.generation_ms_median),xytext=(5,8),textcoords='offset points',fontsize=9)
    ax.set(xlabel='Peak PyTorch GPU allocation (MiB)',ylabel='Median generation latency (ms)',title='Original baseline: memory savings cost latency\nBatch 1 · prompt 128 · output 32 · FP32',xlim=(-35,640))
    ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(figures/'baseline_tradeoff.png',dpi=180);plt.close(fig)
    observations=pd.read_csv(ROOT/'data/step3/analysis/observed_cases.csv')
    predictions=pd.read_csv(ROOT/'data/step3/frozen/predictions.csv')
    e=observations.merge(predictions[['case_id','predicted_generation_ms_compute_transfer']],on='case_id',validate='one_to_one')
    e=e[e.kind=='primary']
    fig,ax=plt.subplots(figsize=(6.6,4.6))
    for k,g in e.groupby('gpu_layers'):ax.scatter(g.generation_ms_median,g.predicted_generation_ms_compute_transfer,label=f'{k} GPU blocks')
    limit=max(e.generation_ms_median.max(),e.predicted_generation_ms_compute_transfer.max())*1.07
    ax.plot([0,limit],[0,limit],linestyle='--',label='Exact prediction')
    ax.set(xlabel='Measured median generation latency (ms)',ylabel='Frozen predicted generation latency (ms)',title='Fresh workload evaluation: predictions are imperfect',xlim=(0,limit),ylim=(0,limit))
    ax.legend(fontsize=8);ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(figures/'frozen_predictions.png',dpi=180);plt.close(fig)
    s=pd.read_csv(ROOT/'data/step4/analysis/observed_cases.csv');p=pd.read_csv(ROOT/'data/step4/frozen/predictions.csv')
    c=s[(s.batch_size==3)&(s.sequence_length==256)&(s.cpu_threads==2)].merge(p[['case_id','predicted_peak_mib','guarded_peak_mib']],on='case_id',validate='one_to_one').sort_values('gpu_layers')
    fig,ax=plt.subplots(figsize=(7.5,4.3))
    for column,label,marker in [('gpu_peak_mib','Observed peak allocation','o'),('predicted_peak_mib','Frozen point estimate','s'),('guarded_peak_mib','Estimate + 5% + 16 MiB','^')]:ax.plot(c.gpu_layers,c[column],marker=marker,label=label)
    ax.axhline(640,linestyle='--',label='640 MiB experimental budget')
    ax.set(xlabel='GPU transformer blocks',ylabel='GPU allocation (MiB)',title='The margin changes the decision at the budget boundary',xticks=[0,3,6,9,12])
    ax.legend(fontsize=8);ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(figures/'budget_boundary.png',dpi=180);plt.close(fig)
    t=pd.read_csv(ROOT/'data/step4/analysis/thread_comparison.csv')
    labels=[f'B{int(r.batch_size_threads1)} S{int(r.sequence_length_threads1)} G{int(r.gpu_layers_threads1)}' for _,r in t.iterrows()]
    fig,ax=plt.subplots(figsize=(8.3,5.2))
    ax.plot(t.generation_ratio_one_vs_two,np.arange(len(t)),marker='o',linestyle='none',label='Ratio of median generation times')
    ax.axvline(1,linestyle='--',label='Equal median latency')
    ax.set(yticks=np.arange(len(t)),yticklabels=labels,xlabel='One-thread latency / two-thread latency\nBelow 1: one thread faster · above 1: two threads faster',title='Step 4: no universal median-latency winner',xlim=(.88,1.11))
    ax.invert_yaxis();ax.grid(True,axis='x',alpha=.25);ax.legend(loc='lower right',fontsize=8);fig.tight_layout();fig.savefig(figures/'thread_sensitivity.png',dpi=180)
    if show:plt.show()
    plt.close(fig)
    print('Saved four figures derived from the supplied measurements.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--show',action='store_true')
    main(parser.parse_args().show)
