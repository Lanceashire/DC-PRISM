"""Prepare the UCI Diabetes 130-US hospitals cohort (Strack et al., 2014) as a
non-credit replay cohort in the same array format as the credit NPZ.

Adverse outcome: readmission within 30 days. Exposure: time in hospital (days),
the resource committed by the encounter. Six numeric covariates; the last two
(prior inpatient and emergency visits) are heavy-tailed counts, matching the
auxiliary-tail slots of the replay preprocessing. First encounter per patient.
Usage: python -I scripts/prepare_diabetes.py <diabetic_data.csv> <out.npz>
"""
import csv, hashlib, sys
import numpy as np

src, out = sys.argv[1], sys.argv[2]
AGE = {f'[{10*i}-{10*(i+1)})': 10*i+5 for i in range(10)}
seen, X, y, e = set(), [], [], []
with open(src, newline='') as f:
    for r in csv.DictReader(f):
        if r['patient_nbr'] in seen or r['discharge_disposition_id'] in {'11', '13', '14', '19', '20', '21'}:
            continue                                      # first encounter; exclude death/hospice discharges
        seen.add(r['patient_nbr'])
        X.append([AGE[r['age']], float(r['num_lab_procedures']), float(r['num_procedures']), float(r['num_medications']),
                  float(r['number_inpatient']), float(r['number_emergency'])])
        y.append(1 if r['readmitted'] == '<30' else 0); e.append(float(r['time_in_hospital']))
X, y, e = np.array(X), np.array(y, np.int8), np.array(e)
np.savez(out, Diabetes_X=X, Diabetes_y=y, Diabetes_ead=e)
print(dict(rows=len(y), adverse_rate=round(float(y.mean()), 4), source_sha256=hashlib.sha256(open(src, 'rb').read()).hexdigest()))
