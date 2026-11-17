"""Remove em dashes from the paper, repairing each sentence grammatically."""
import pathlib

p = pathlib.Path(__file__).parent / "creddefer_rait2027.tex"
s = p.read_text(encoding="utf-8")

R = [
 ("applications, and---unusually for applied\nwork---every claim in the system's development was adjudicated",
  "applications. Unusually for applied\nwork, every comparative modelling claim in its development was adjudicated"),
 ("models---gradient-boosted ensembles, the TabPFN-2.5 and Google TabFM tabular\nfoundation models, or 2026 deep-tabular challengers---ever cleared the floor.",
  "models (gradient-boosted ensembles, the TabPFN-2.5 and Google TabFM tabular\nfoundation models, or 2026 deep-tabular challengers) ever cleared the floor."),
 ("system---rule-based scoring, calibrated blending, capacity-aware deferral,",
  "system, comprising rule-based scoring, calibrated blending, capacity-aware deferral,"),
 ("decision provenance---runs as a reproducible cloud deployment.",
  "decision provenance, runs as a reproducible cloud deployment."),
 ("number---ranking accuracy (AUC)---while",
  "number, ranking accuracy (AUC), while"),
 ("fusion we attempted---symmetry that",
  "fusion we attempted: symmetry that"),
 ("the improved models gained $+0.0016$---under the floor, as was every fusion",
  "the improved models gained $+0.0016$, under the floor, as was every fusion"),
 ("deployed disagreement-based deferral rule can be \\emph{anti}-selective---ours\ndeferred the model's most confident cases---and that this is detectable with",
  "deployed disagreement-based deferral rule can be \\emph{anti}-selective (ours\ndeferred the model's most confident cases), and that this is detectable with"),
 ("constraints on 21 domain-signed features---which make directional\nadverse-action explanations true by construction---cost $-0.0008$ AUC (inside",
  "constraints on 21 domain-signed features, which make directional\nadverse-action explanations true by construction, cost $-0.0008$ AUC (inside"),
 ("The complete system---API, rule",
  "The complete system (API, rule"),
 ("provenance stamps---runs on commodity cloud infrastructure from a single",
  "provenance stamps) runs on commodity cloud infrastructure from a single"),
 ("For gradient-boosted trees---still the strongest\nmodel class on tabular data---Malinin",
  "For gradient-boosted trees, still the strongest\nmodel class on tabular data, Malinin"),
 ("with a standalone AUC of 0.5650---deliberately an",
  "with a standalone AUC of 0.5650: deliberately an"),
 ("delta exceeds $+0.0036$---the observed fold standard deviation---and, for",
  "delta exceeds $+0.0036$ (the observed fold standard deviation) and, for"),
 ("$+0.0016$---real in direction (paired bootstrap CI $[0.0010, 0.0021]$) but",
  "$+0.0016$: real in direction (paired bootstrap CI $[0.0010, 0.0021]$) but"),
 ("ensembles improved the early-stopped default by $+0.001$---under a third of\nthe floor---and the capacity-adaptive learning curve",
  "ensembles improved the early-stopped default by $+0.001$, under a third of\nthe floor, and the capacity-adaptive learning curve"),
 ("reproduces to four decimals)---a safeguard we added after an early\nrow-misalignment incident, and which we recommend as standard practice for\nfoundation-model comparisons.",
  "reproduces to four decimals); this safeguard was added after an early\nrow-misalignment incident and we recommend it as standard practice for\nfoundation-model comparisons."),
 ("ever tested ($\\rho = 0.66$)---diversity obtained by being wrong does not\ncancel errors.",
  "ever tested ($\\rho = 0.66$). Diversity obtained by being wrong does not\ncancel errors."),
 ("$\\sim$18\\,GB of offload scratch per fold)---a\ndeployability result in itself.",
  "$\\sim$18\\,GB of offload scratch per fold), a\ndeployability result in itself."),
 ("signature sat at $z \\approx +28\\sigma$ on a held-out split---confident cases\nsystematically sent to humans.",
  "signature sat at $z \\approx +28\\sigma$ on a held-out split: confident cases\nwere systematically sent to humans."),
 ("Fold-paired cost: $-0.0008$ AUC---inside the noise floor---with zero",
  "Fold-paired cost: $-0.0008$ AUC, inside the noise floor, with zero"),
 ("loses at four of five decision-time cost ratios)---an empirical confirmation",
  "loses at four of five decision-time cost ratios), an empirical confirmation"),
 ("The TabPFN-2.5 second opinion---whose checkpoint and\nin-context set exceed commodity GPU memory---is served",
  "The TabPFN-2.5 second opinion, whose checkpoint and\nin-context set exceed commodity GPU memory, is served"),
 ("Treating the decision pipeline---not the classifier---as the research object",
  "Treating the decision pipeline, rather than the classifier, as the research object"),
]

missed = []
for old, new in R:
    if old in s:
        s = s.replace(old, new)
    else:
        missed.append(old[:60])

p.write_text(s, encoding="utf-8")
print("applied:", len(R) - len(missed), "of", len(R))
for m in missed:
    print("MISSED:", m)
print("remaining em dashes:", s.count("---"))
