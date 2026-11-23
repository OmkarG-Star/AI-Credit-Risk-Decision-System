"""Download Lending Club issued-loan data (2007-2017) and save one parquet file.

Source: the public R data package github.com/kuhnrl30/LendingClubData, which
mirrors Lending Club's published quarterly loan files, one .rda file per year.
Run from the repository root:  python scripts/get_data.py
"""

import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pyreadr

OUT = Path("data/lending_club_2007_2017.parquet")

with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/kuhnrl30/LendingClubData.git", tmp], check=True)
    parts = []
    for year in range(2007, 2018):
        frame = next(iter(pyreadr.read_r(f"{tmp}/data/LendingClubData_{year}.rda").values()))
        parts.append(frame)
        print(year, frame.shape)

df = pd.concat(parts, ignore_index=True)
for col in df.columns[df.dtypes == object]:
    df[col] = df[col].astype("string")
OUT.parent.mkdir(exist_ok=True)
df.to_parquet(OUT, index=False)
print("saved", OUT, df.shape)
