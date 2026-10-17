"""Expand the paper toward 8-9 pages with repo-grounded content only."""
import pathlib

p = pathlib.Path(__file__).parent / "creddefer_rait2027.tex"
s = p.read_text(encoding="utf-8")

# ---- 1. Expand CBES + capture detail in System Design -----------------------
anchor = "\\emph{explanation layer}, not a predictor.\n"
addition = anchor + """CBES exists because an all-ML pipeline fails
two institutional tests at once: a reviewer cannot argue with a gradient, and
a regulator cannot inspect one. The pillar weights (0.35 credit, 0.30
capacity, 0.20 behaviour, 0.10 liquidity, 0.05 stability) are fixed domain
policy, calibrated to a probability through a logistic link, and every pillar
sub-score is shown to the reviewer alongside the decision, so disagreement
between the two opinions is itself interpretable: a high ML score over a weak
capacity pillar names the conversation the underwriter should have.
"""
assert anchor in s
s = s.replace(anchor, addition, 1)

# ---- 2. Parameters table + claim-rule equation at end of Protocol -----------
anchor = "only with their threshold stated.\n"
addition = anchor + """
Formally, for candidate $B$ and incumbent $A$ evaluated on identical folds
$k = 1,\\dots,5$ with per-fold AUCs $a_k^{B}, a_k^{A}$, the claim rule is
\\begin{equation}
\\underbrace{\\tfrac{1}{5}\\textstyle\\sum_k \\big(a_k^{B}-a_k^{A}\\big)}_{\\text{fold-paired mean delta}} > 0.0036
\\quad\\text{and, for feature claims,}\\quad a_k^{B} > a_k^{A}\\;\\forall k .
\\label{eq:claim}
\\end{equation}
Table~\\ref{tab:params} lists every decision-layer quantity, how it was fitted,
and the artifact recording it; none is fitted on evaluation data.

\\begin{table}[t]
\\centering
\\caption{Decision-layer parameters: value, fitting procedure, evidence.}
\\label{tab:params}
\\begin{tabular}{@{}l l l@{}}
\\toprule
\\textbf{Parameter} & \\textbf{Value} & \\textbf{How set} \\\\
\\midrule
Blend weight $\\alpha$ & 0.10 & cost table on tuning split \\\\
Base threshold $t_{base}$ & 0.65 & expected-cost minimisation \\\\
Approve / reject cutoffs & 0.6425 / 0.6575 & band around $t_{base}$ \\\\
Deferral rate & 22.5\\% & capacity band (20--25\\%) quantile \\\\
Claim floor & $\\pm0.0036$ AUC & observed fold std.\\ dev. \\\\
GBM protocol & 2000 trees, lr 0.05 & early stop 100, inner 10\\% split \\\\
SGLB budget & 1000 iter., $K{=}10$ & fixed (posterior sampling) \\\\
Monotone constraints & 21 signs & domain knowledge, pre-fixed \\\\
Cost matrix (studied) & FN:FP = 7.5:1 & stated LGD vs.\\ margin \\\\
Exploration arm & 3\\% & counterfactual-label preservation \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
assert anchor in s
s = s.replace(anchor, addition, 1)

# ---- 3. Formal audit statistic in the deferral audit subsection -------------
anchor = "statistics versus a random router) as a deployment check for any L2D system.\n"
addition = """statistics versus a random router) as a deployment check for any L2D system.
Concretely, with $n_d$ deferrals and deferred-pile statistic $T$ (we use the
no-deferral engine's accuracy on the deferred cases), draw $T_1,\\dots,T_m$
from $m{=}200$ routers deferring $n_d$ uniformly random cases and report
\\begin{equation}
z \\;=\\; \\frac{T - \\bar{T}_{\\mathrm{rand}}}{\\sigma_{\\mathrm{rand}}} ,
\\label{eq:audit}
\\end{equation}
where a working router should be strongly \\emph{negative} (it removes cases
the engine would get wrong). The production rule scored $z \\approx +28$;
no sign test or calibration analysis is needed to know something is inverted.
"""
assert anchor in s
s = s.replace(anchor, addition, 1)

# ---- 4. SGLB decomposition equations ----------------------------------------
anchor = "from data uncertainty. At the matched rate on the test half\n"
addition = """from data uncertainty: with member probabilities
$p_1,\\dots,p_K$ from one SGLB model's virtual ensembles and
$\\bar{p} = \\tfrac{1}{K}\\sum_k p_k$,
\\begin{equation}
\\underbrace{\\mathcal{H}(\\bar{p})}_{\\text{total}} \\;=\\;
\\underbrace{\\tfrac{1}{K}\\textstyle\\sum_k \\mathcal{H}(p_k)}_{\\text{data (aleatoric)}}
\\;+\\; \\underbrace{\\mathcal{U}_{\\mathrm{know}}}_{\\text{knowledge (epistemic)}} ,
\\label{eq:decomp}
\\end{equation}
where $\\mathcal{H}$ is binary entropy and the residual mutual information
$\\mathcal{U}_{\\mathrm{know}}$ is the deferral signal. At an 8\\% base rate
nearly all predictions sit below $0.5$, making total and data entropy
near-monotone in $\\bar{p}$ itself (correlation 0.94); knowledge uncertainty
is the only component not reducible to the score, which is precisely why it
carries routing information the reject-option rule cannot see (their
correlation is $-0.36$). At the matched rate on the test half
"""
assert anchor in s
s = s.replace(anchor, addition, 1)

# ---- 5. Deployment: connection-run stages + decision mix + ops lessons ------
anchor = "capacity planning.\n"
addition = """capacity planning. The connection run proceeds in four ordered
gates (configured; reachable; scores returned for every row; sane, i.e.\\ the
recomputed AUC on the 400 labelled rows clears 0.60), so each failure mode is
named rather than inferred. On the full population the engine's decision mix
is 167{,}751 instant approvals, 70{,}593 instant rejections and 69{,}167
deferrals (22.5\\%); the deployed demonstration additionally routes reviewer
adjudications (including rejections of deferred cases with written reason
codes) through the same capture path, so the human-in-the-loop story is
exercised end to end rather than mocked.
"""
assert anchor in s
s = s.replace(anchor, addition, 1)

# ---- 6. RBI table ------------------------------------------------------------
anchor = "never writes decisions; the roadmap is to promote the engineered-feature"
table = """Table~\\ref{tab:rbi} summarises the mapping.

\\begin{table}[t]
\\centering
\\caption{RBI expectation $\\to$ CredDefer mechanism $\\to$ evidence.}
\\label{tab:rbi}
\\begin{tabular}{@{}p{1.9cm} p{3.1cm} p{2.4cm}@{}}
\\toprule
\\textbf{Expectation} & \\textbf{Mechanism} & \\textbf{Evidence} \\\\
\\midrule
Explainability & SHAP per decision; CBES pillar breakdown; 21 monotone signs
(directional reasons by construction) & monotonicity report: $-0.0008$ AUC,
0/14{,}400 violations \\\\
Human oversight & capacity-aware deferral (22.5\\%); reviewer capture with
reason codes, confidence, time; override monitoring & deferral audit:
$z{=}{-}99.7$ after repair \\\\
Auditability & engine version + artifact + threshold hash on every decision;
all numbers regenerate from committed scripts & provenance stamps in API,
dashboards, audit report \\\\
Model governance & versioned research/serving separation; relearning gate
closed; 3\\% exploration arm & gate evaluation on every inspection \\\\
\\bottomrule
\\end{tabular}
\\end{table}

We state one caveat openly: TabPFN-2.5
outperforms the deployed serving model on identical features (0.7446
best-case vs.\\ 0.6919) and remains advisory because of its non-commercial
output licence, its GPU footprint, and the governance rule that advisory AI
never writes decisions; the roadmap is to promote the engineered-feature"""
# replace the original caveat lead-in sentence start
old_caveat = """We state one caveat openly: TabPFN-2.5
outperforms the deployed serving model on identical features (0.7446
best-case vs.\\ 0.6919) and remains advisory because of its non-commercial
output licence, its GPU footprint, and the governance rule that advisory AI
never writes decisions; the roadmap is to promote the engineered-feature"""
assert old_caveat in s
s = s.replace(old_caveat, table, 1)

# ---- 7. Discussion section before Limitations --------------------------------
anchor = "\\section{Limitations}"
addition = """\\section{Discussion: What Generalises}
Four practices from this project transfer beyond credit scoring. First, the
\\emph{deferral audit} of Eq.~\\eqref{eq:audit} costs a few lines and no new
data, yet it caught a production router doing the opposite of its job; any
deployed abstention rule deserves it. Second, \\emph{pre-registration with a
symmetric floor} changed the sociology of our own development: because the
rule in Eq.~\\eqref{eq:claim} rejected our tuning gains before it rejected
anyone else's fusion, negative results became reportable rather than
embarrassing. Third, \\emph{licence-aware challenger selection}: benchmark
rank is not deployability, and for regulated decisioning the licence of a
model's \\emph{outputs} is as binding as its accuracy; screening only
permissively licensed challengers kept every measured alternative actionable.
Fourth, \\emph{serve-time contracts}: freezing the feature derivation shared
by training and serving cost us headline accuracy (0.6919 deployed vs.\\
0.7757 offline) but bought byte-identical provenance; we argue the right
response is to promote richer features \\emph{through} the contract, never
around it.

\\section{Limitations}"""
assert anchor in s
s = s.replace(anchor, addition, 1)

p.write_text(s, encoding="utf-8")
print("expanded ok; em dashes:", s.count("---"))
