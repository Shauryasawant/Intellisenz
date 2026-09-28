import re
import sys
import pandas as pd

path, device = sys.argv[1], sys.argv[2]
raw = pd.read_csv(path, header=None, names=['device', 'site', 'time', 'value'],
                  quotechar='"', engine='python')
sub = raw[raw['device'] == device]

def fingerprint(v):
    # replace every number with N so only the *shape* of the packet remains
    return re.sub(r'-?\d+(\.\d+)?', 'N', str(v))

fp = sub['value'].map(fingerprint)
for pattern, n in fp.value_counts().head(10).items():
    example = sub.loc[fp == pattern, 'value'].iloc[0]
    print(f"{n:7d}  ({n / len(sub):.1%})  {pattern}")
    print(f"         e.g. {example}\n")
