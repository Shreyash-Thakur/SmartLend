"""Structural sanity checks for the paper source (no LaTeX engine needed)."""
import re
import pathlib
from collections import Counter

HERE = pathlib.Path(__file__).parent
s = (HERE / "creddefer_rait2027.tex").read_text(encoding="utf-8")

print("brace balance:", s.count("{") - s.count("}"))
b = Counter(re.findall(r"\\begin\{(\w+\*?)\}", s))
e = Counter(re.findall(r"\\end\{(\w+\*?)\}", s))
print("env mismatch:", {k: (b[k], e.get(k, 0)) for k in b if b[k] != e.get(k, 0)} or "none")

cited = set(c for g in re.findall(r"\\cite\{([^}]+)\}", s) for c in g.split(","))
items = set(re.findall(r"\\bibitem\{([^}]+)\}", s))
print("cites missing bibitem:", cited - items or "none")
print("bibitems never cited:", items - cited or "none")

figs = re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", s)
print("missing figure files:", [f for f in figs if not (HERE / f).exists()] or "none")

labels = set(re.findall(r"\\label\{([^}]+)\}", s))
refs = set(re.findall(r"\\ref\{([^}]+)\}", s))
print("refs missing labels:", refs - labels or "none")
