from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
CJK=font_manager.FontProperties(fname="/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc").get_name()
P=Path(__file__).parent;R=Path('<LOCAL_WORKSPACE>/2026-11-04/claude-strengthen/DC-PRISM');d=R/'results/supplement_v11'
m=json.loads((d/'run_manifest.json').read_text());bad=[p for p,h in m['source_sha256'].items() if hashlib.sha256((R/p).read_bytes()).hexdigest()!=h]
a=pd.read_csv(d/'rows.csv.gz');a=a[a.part=='A'];assert len(a)==624
assert np.all(a.F_lo<=a.F+1e-12) and np.all(a.F<=a.F_up+1e-12)
e=pd.read_csv(d/'E_exact_mk.csv');print('exact columns',e.columns.tolist(),'rows',len(e))
report={'source_hash_mismatches':bad,'manifest_status':m['status'],'family_states':len(a),'empirical_inequalities_pass':True,'scope':'aggregate and source audit, no complete raw-data rerun','important_correction':'Pointwise maxima of noisy held-out policy gains are not certified deployable lower bounds; greedy residual is not the exact family floor.'}
(P/'verification.json').write_text(json.dumps(report,indent=2));print(report)
g=pd.read_csv(d/'A_bounds_by_group.csv');dose=pd.read_csv(d/'D_dose_and_pool.csv');dose=dose[dose['sub']=='flip']
for lang in ['en','zh']:
 plt.rcParams.update({'font.family':'DejaVu Sans' if lang=='en' else CJK,'font.size':7,'axes.labelsize':7,'axes.titlesize':7,'pdf.fonttype':3})
 fig,ax=plt.subplots(1,2,figsize=(5.2,2.25));fig.subplots_adjust(left=.1,right=.98,bottom=.22,top=.85,wspace=.43)
 x=np.arange(len(g));ax[0].vlines(x,g.F_lo*1000,g.F_up*1000,color='#aaa',lw=1);ax[0].scatter(x,g.F*1000,s=8,color='#426c91');ax[0].set_xticks([])
 ax[0].set_xlabel('Dataset / learner groups' if lang=='en' else '数据集与学习器组');ax[0].set_ylabel('Empirical floor / bounds ($10^{-3}$)' if lang=='en' else '经验下限及界限（$10^{-3}$）');ax[0].set_title('(a) Distribution-free sample bounds' if lang=='en' else '(a) 不依赖分布的样本界限')
 for ds,z in dose.groupby('dataset'):ax[1].plot(z.level,z.F_family*1000,'o-',label=ds,ms=3)
 ax[1].set_xlabel('Label-flip probability' if lang=='en' else '标签翻转概率');ax[1].set_ylabel('Empirical family floor ($10^{-3}$)' if lang=='en' else '经验族下限（$10^{-3}$）');ax[1].set_title('(b) Randomized noise intervention' if lang=='en' else '(b) 随机噪声干预');ax[1].legend(frameon=False,fontsize=6)
 for axis in ax:axis.spines[['top','right']].set_visible(False)
 fig.savefig(P/lang/'figures/fig_predict.pdf');fig.savefig(P/lang/'figures/fig_predict.png',dpi=220);plt.close(fig)
