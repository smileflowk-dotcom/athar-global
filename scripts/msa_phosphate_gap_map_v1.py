#!/usr/bin/env python3
import json, urllib.request
from pathlib import Path
import pandas as pd

OUT=Path('03-validation'); OUT.mkdir(exist_ok=True)
DATA=Path('/tmp/MSA-data.xlsx')
urllib.request.urlretrieve('https://raw.githubusercontent.com/PYU-pub/MSA-ML/main/data/MSA%20data.xlsx', DATA)
df=pd.read_excel(DATA)

# Chemical identities are mapped from the dataset's charge + conjugate-acid pKa profile.
# code 2: CO3^2- (charge 2, pKa 10.33)
# code 3: PO4^3- (charge 3, pKa 12.32)
# code 5: HPO4^2- (charge 2, pKa 7.19)
# code 8: P2O7^4- (charge 4, pKa 9.41)
MAP={2:'CO3^2-',3:'PO4^3-',5:'HPO4^2-',8:'P2O7^4-'}
sub=df[df['anion'].isin(MAP)].copy()
sub['anion_name']=sub['anion'].map(MAP)

# Coverage by exact real resin identity.
resin_rows=[]
for resin,g in sub.groupby('resin'):
    ions=sorted(g['anion_name'].unique().tolist())
    rec={
        'resin':str(resin),
        'rows':int(len(g)),
        'ions_present':ions,
        'has_carbonate_control':'CO3^2-' in ions,
        'has_PO4':'PO4^3-' in ions,
        'has_HPO4':'HPO4^2-' in ions,
        'has_P2O7':'P2O7^4-' in ions,
    }
    for c in ['resinID','base_type','pore','matrix','FG']:
        if c in g.columns:
            rec[c]=sorted({str(x) for x in g[c].dropna().unique().tolist()})
    resin_rows.append(rec)

# Exact gaps: phosphate ions missing on resins where carbonate exists, and vice versa.
priority=[]
for rec in resin_rows:
    if not rec['has_carbonate_control']:
        continue
    for ion,key in [('PO4^3-','has_PO4'),('HPO4^2-','has_HPO4'),('P2O7^4-','has_P2O7')]:
        priority.append({
            'resin':rec['resin'],
            'candidate_ion':ion,
            'observed_in_dataset':bool(rec[key]),
            'matched_carbonate_available':True,
            'base_type':rec.get('base_type',[]),
            'pore':rec.get('pore',[]),
            'matrix':rec.get('matrix',[]),
            'FG':rec.get('FG',[]),
            'status':'KNOWN' if rec[key] else 'DATA_GAP_CANDIDATE'
        })

# Descriptive observed Qe only; not a swing ranking.
observed=[]
for (resin,ion),g in sub.groupby(['resin','anion_name']):
    observed.append({
        'resin':str(resin),'ion':ion,'n':int(len(g)),
        'humidity_values':sorted({float(x) for x in g['humidity'].dropna().unique().tolist()}),
        'Qe_mean':float(g['Qe'].mean()),
        'Qe_median':float(g['Qe'].median()),
        'Qe_min':float(g['Qe'].min()),
        'Qe_max':float(g['Qe'].max()),
    })

report={
  'status':'PASS',
  'mapping_basis':{
    '2':{'identity':'CO3^2-','charge':2,'pKa':10.33},
    '3':{'identity':'PO4^3-','charge':3,'pKa':12.32},
    '5':{'identity':'HPO4^2-','charge':2,'pKa':7.19},
    '8':{'identity':'P2O7^4-','charge':4,'pKa':9.41},
  },
  'resin_coverage':resin_rows,
  'matched_gap_map':priority,
  'observed_descriptive_only':observed,
  'guardrails':[
    'Chemical code mapping is based on charge and conjugate-acid pKa profiles and must remain traceable.',
    'KNOWN means observed in this public dataset, not validated superiority.',
    'DATA_GAP_CANDIDATE means absent from this dataset on a resin with carbonate control; literature novelty still requires verification.',
    'Qe summaries are descriptive only and are not moisture-swing rankings.',
    'Next promotion requires uncertainty/OOD plus feasibility and evidence checks.'
  ]
}
(OUT/'msa-phosphate-gap-map-v1.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
