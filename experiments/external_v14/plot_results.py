"""Descriptive figures only; formal decisions are in the prespecified statistics tables."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'MAIN');args=ap.parse_args();OUT=args.output
df=pd.read_csv(OUT/'rows.csv.gz')
colors={'oracle-test':'#b51f2b','oracle-split':'#d47a0a','oracle-exante':'#151c28','Random':'#6d7c9c','Uncertainty':'#2868ae','CoreSet':'#29846b'}
names={'cifar10_cat_dog':'CIFAR-10 cat/dog | MobileNetV3-Small','agnews_business_scitech':'AG News Business/Sci-Tech | MiniLM-L6'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(12.8,4.2))
for ax,(ds,title) in zip(axes,names.items()):
    for m,c in colors.items():
        x=df[(df.dataset==ds)&(df.method==m)].groupby('n_labels').brier_T.agg(['mean','sem'])
        ax.plot(x.index,x['mean'],label=m,color=c,lw=2 if m.startswith('oracle') else 1.3,linestyle='-' if m.startswith('oracle') else '--')
        ax.fill_between(x.index,x['mean']-x['sem'],x['mean']+x['sem'],color=c,alpha=.08,lw=0)
    ax.set(title=title,xlabel='Acquired labels (including initial 68)',ylabel='Held-out test Brier score')
    ax.grid(alpha=.14)
axes[1].legend(frameon=False,fontsize=8,ncol=2)
fig.suptitle('Frozen deep encoders + nonlinear neural head | 12 paired seeds per task',fontsize=12)
fig.tight_layout();fig.savefig(OUT/'learning_curves.png',bbox_inches='tight');fig.savefig(OUT/'learning_curves.pdf',bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12.8,4.2))
for ax,(ds,title) in zip(axes,names.items()):
    x=df[(df.dataset==ds)&(df.step>0)].pivot_table(index=['seed','n_labels'],columns='method',values='brier_T')
    # Fixed Random anchor keeps the decomposition free of post-hoc best-strategy selection.
    comp=pd.DataFrame({'Test reuse':x['oracle-split']-x['oracle-test'],'Outcome-luck contrast':x['oracle-exante']-x['oracle-split'],'Random minus ex-ante':x['Random']-x['oracle-exante']})
    for key,c in zip(comp.columns,['#626b7d','#d47a0a','#2868ae']):
        a=comp[key].groupby('n_labels').agg(['mean','sem'])*1000
        ax.plot(a.index,a['mean'],color=c,label=key,lw=1.8)
        ax.fill_between(a.index,a['mean']-a['sem'],a['mean']+a['sem'],color=c,alpha=.12,lw=0)
    ax.axhline(0,color='#212121',lw=.6);ax.set(title=title,xlabel='Acquired labels',ylabel='Paired Brier difference (x 1,000)');ax.grid(alpha=.14)
axes[1].legend(frameon=False,fontsize=8)
fig.suptitle('Own-trajectory contrasts | shading: seed standard error (descriptive)',fontsize=12)
fig.tight_layout();fig.savefig(OUT/'components_over_curve.png',bbox_inches='tight');fig.savefig(OUT/'components_over_curve.pdf',bbox_inches='tight');plt.close(fig)
