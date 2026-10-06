#!/usr/bin/env python3
import json, urllib.request
from pathlib import Path
import pandas as pd

OUT=Path("03-validation"); OUT.mkdir(exist_ok=True)
DATA=Path("/tmp/MSA-data.xlsx")
urllib.request.urlretrieve("https://raw.githubusercontent.com/PYU-pub/MSA-ML/main/data/MSA%20data.xlsx", DATA)
df=pd.read_excel(DATA)

cols=[c for c in ["anion","anion_charge","anion-pKa","resin","resinID","base_type","pore","matrix","FG","humidity","Qe"] if c in df.columns]
unique=(df[cols].drop_duplicates().sort_values(by=[c for c in ["anion","resin","humidity"] if c in cols]))
by_anion=[]
for anion,g in df.groupby("anion", dropna=False):
    rec={"anion_code":str(anion),"rows":int(len(g))}
    for c in ["anion_charge","anion-pKa","base_type","pore","matrix","FG"]:
        if c in g.columns:
            vals=sorted({str(x) for x in g[c].dropna().unique().tolist()})
            rec[c]=vals[:30]
    for c in ["resin","resinID"]:
        if c in g.columns:
            rec[c]=sorted({str(x) for x in g[c].dropna().unique().tolist()})[:50]
    by_anion.append(rec)

report={
  "columns": list(df.columns),
  "anion_code_profiles": by_anion,
  "note":"This is a codebook discovery aid. Numeric anion codes must not be interpreted as chemical identities without a source mapping or a uniquely identifying property match."
}
(OUT/"msa-anion-codebook-audit-v0.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
