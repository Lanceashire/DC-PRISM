"""Audit completed outputs and package reproducible code/aggregate results, excluding data caches."""
import csv,gzip,hashlib,json,shutil,zipfile
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parent;O=R/'MAIN';D=R.parents[1]/'outputs';D.mkdir(exist_ok=True)
assert json.loads((O/'run_manifest.json').read_text())['status']=='COMPLETE'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
F=json.loads((R/'freeze.json').read_text())
assert all(sha(R/n)==h for n,h in F['sources'].items())
assert all(sha(R/n)==h for n,h in F['data_sha256'].items())
P=json.loads((R/'protocol.json').read_text())
audits=[json.loads((O/f'{ds}_{seed}_audit.json').read_text()) for ds in P['datasets'] for seed in P['seeds']]
assert len(audits)==24 and all(x['pairwise_disjoint'] for x in audits)
(O/'split_audits.json').write_text(json.dumps(audits,indent=2))
with gzip.open(O/'rows.csv.gz','rt') as f:
    reader=csv.DictReader(f);columns=reader.fieldnames;count=0;cats={}
    for row in reader:
        count+=1
        for k,v in row.items():
            if k in ['dataset','method']:cats.setdefault(k,set()).add(v)
            else:float(v)
assert count==4464
df=pd.read_csv(O/'rows.csv.gz')
p0=df[df.step==0].groupby(['dataset','seed']).brier_T.agg(['min','max'])
assert ((p0['max']-p0['min']).abs()<1e-15).all()
x=df.pivot_table(index=['dataset','seed','step'],columns='method',values='brier_T')
identity=(x['Random']-x['oracle-test'])-((x['oracle-split']-x['oracle-test'])+(x['oracle-exante']-x['oracle-split'])+(x['Random']-x['oracle-exante']))
assert identity.abs().max()<1e-14
audit={'status':'PASS','frozen_sources_and_data_unchanged':True,'all_24_split_assertions_pass':True,'initial_scores_identical_across_methods':True,'decomposition_max_absolute_error':float(identity.abs().max()),'compressed_results':{'sha256':sha(O/'rows.csv.gz'),'rows':count,'columns':columns,'categorical_values':{k:sorted(v) for k,v in cats.items()},'all_other_fields_numeric':True,'content':'Only per-experiment trajectory/step aggregate Brier, accuracy, label budget and estimated noise. No source examples, features, personal identifiers, or per-record predictions.'}}
(O/'content_integrity_audit.json').write_text(json.dumps(audit,indent=2))
tab=pd.concat([pd.read_csv(O/f).assign(family=n) for f,n in [('primary_components.csv','whole_components'),('primary_early.csv','early'),('primary_shortfalls.csv','shortfalls')]],ignore_index=True)
tab.to_csv(D/'v14_主要统计结果_20261111.csv',index=False)
shutil.copyfile(O/'learning_curves.png',D/'v14_完整学习曲线_20261111.png')
shutil.copyfile(O/'components_over_curve.png',D/'v14_早晚阶段分解_20261111.png')
sources=['prepare.py','gpu_head.py','run.py','stats.py','check_search.py','plot_results.py','build_report.py','replication_freeze.py','package_results.py','protocol.json','freeze.json','assets_manifest.json','environment.json','requirements.txt','README.md','head_benchmark.json','vision_duplicate_audit.json']
results=['rows.csv.gz','run_manifest.json','integrity_check.json','content_integrity_audit.json','split_audits.json','summary.csv','primary_components.csv','primary_early.csv','primary_shortfalls.csv','prediction_check.json','per_trajectory_all.csv','per_trajectory_early.csv','per_trajectory_late.csv','learning_curves.png','learning_curves.pdf','components_over_curve.png','components_over_curve.pdf']
files=[(R/n,'experiments/external_v14/'+n) for n in sources]+[(O/n,'results/external_v14/'+n) for n in results]
for n in ['claude_commit_publication_map.json','compressed_results_content_audit.json','claude_saved_experiment_verification.json','claude_v13_recomputed_statistics.json']:
    files.append((R.parent/n,'provenance/'+n))
pdf=D/'v14_视觉与文本外部验证报告_20261111.pdf';files.append((pdf,'deliverables/external_v14/report.pdf'))
manifest={name:sha(path) for path,name in files}
(D/'v14_交付清单_20261111.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
with zipfile.ZipFile(D/'v14_外部验证_代码与结果_20261111.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for path,name in files:z.write(path,name)
    z.writestr('manifest_sha256.json',json.dumps(manifest,indent=2))
print(json.dumps({'audit':audit,'packaged_files':len(files)},indent=2))
