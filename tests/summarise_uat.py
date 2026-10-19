"""Table 5.12 from a Google Forms export (CSV) or a plain CSV.
Usage: python summarise_uat.py responses.csv
It uses every column whose answers are all whole numbers 1-5 (your five rating statements, in order)
and prints the mean, median and share agreeing (4 or 5) for each. Writes table_5_12.csv."""
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
cols = [c for c in d.columns if pd.to_numeric(d[c], errors="coerce").between(1, 5).all() and d[c].notna().all()]
print("respondents:", len(d), "| rating columns found:", len(cols))
for c in cols: print("  -", c[:90])
r = d[cols].apply(pd.to_numeric)
out = pd.DataFrame({"statement": [c[:90] for c in cols], "mean": r.mean().round(2).values,
                    "median": r.median().values, "agree_%": ((r >= 4).mean() * 100).round(1).values})
print(out.to_string(index=False)); print("overall mean:", round(r.values.mean(), 2))
out.to_csv("table_5_12.csv", index=False)
