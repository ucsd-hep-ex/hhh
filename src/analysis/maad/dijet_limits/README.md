# Paired dijet resonances — traditional vs pairwise-attention pairing

Reproductions of two figures of [arXiv:2206.09997](https://arxiv.org/abs/2206.09997)
(CMS, paired dijet resonances, 13 TeV, 138 fb⁻¹) with this repository's
simulation, comparing two jet-pairing methods on **exactly the same events**:

* **Figure 9** — the average-dijet-mass spectrum in the three α bins of the
  nonresonant search, with the three smooth background parameterisations fitted
  to it, example signals, and pull panels → section 11.
* **Figure 12** — expected 95 % CL upper limits versus resonance mass → sections
  5–9.

The two methods compared:

| method | what it does |
| --- | --- |
| `traditional` | the repository's mass-agnostic χ² baseline, [`src/models/mass_agnostic_baseline.py`](../../../models/mass_agnostic_baseline.py): partition the leading four jets into the two pairs that minimise χ² about the mean dijet mass |
| `pairwise` | the pairwise-attention SPANet assignment, i.e. the `(g1, g2)` jet indices predicted for `s1` and `s2` |

Everything else — event selection, alpha categories, m_avg binning, weights,
nuisance parameters, statistical procedure — is identical between the two.

---

## 1. Inputs found in the repository

### Signal

Scalar-colour-octet pair production, `S → gg`, resolved `ss → 4g` topology
([`event_files/assign_resolved_ss4g.yaml`](/maad-vol/event_files/assign_resolved_ss4g.yaml)).

* Predictions: `/maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_signal/octet_mso_<m>_test.h5`
  (SPANet `logs/octet_uaf_500_1500_pairwise/version_2`, checkpoint `epoch=193`), the 5 % test split
  of `/maad-vol/data/octet_uaf_test_datasets`.
* Masses: **500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500 GeV** (11 points).
* Cross sections, generated and selected counts: `data/octet_uaf/efficiency-t2.csv`
  (σ from 4.5215 pb at 500 GeV down to 0.0023605 pb at 1500 GeV; preselection
  efficiency ≈ 0.9985–0.9999).

### Background

QCD multijet, generated in exclusive pt-hat bins.

* Predictions: `/maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_qcd/qcd_pthatmin_*_pthatmax_*.h5`.
* Cross sections: `data/qcd/xsec_err_<sample>.dat` (value, error).
* Generated / selected counts and preselection efficiency: `data/qcd/efficiency-t2.csv`.
* **Excluded**, as instructed and consistent with
  [`predicted_background_category.py`](../predicted_background_category.py): the two
  inclusive samples `qcd_pthatmin_20.0_pthatmax_-1.0` and
  `qcd_pthatmin_200.0_pthatmax_-1.0`, which overlap the exclusive binning.
  The 20 remaining exclusive bins are used.

The prediction files carry a verbatim copy of `INPUTS/Jets`, verified identical to
the source h5 files, so both pairings read their jets from one file and cannot
diverge.

---

## 2. Inputs the repository does **not** provide

None of these were invented silently. They are configurable, printed at the top
of every run, and copied into `model_config.json` under `unresolved_inputs`.

| input | status | default used |
| --- | --- | --- |
| Integrated luminosity | **not in repository** — no config, script or data file records one | `138 fb⁻¹`, taken from the reference analysis arXiv:2206.09997 (`--luminosity-fb`) |
| Collision energy | **not in repository** | `13 TeV`, from the same reference; consistent with the magnitude of the QCD cross sections (`--sqrt-s-tev`) |
| QCD k-factor | **not in repository** — the `.dat` files hold generator cross sections, no k-factor file exists | `1.0` (`--qcd-k-factor`) |
| Signal k-factor | **not in repository** | `1.0` (`--signal-k-factor`) |
| Per-event generator weights | **not in repository** — the h5 files contain no weight dataset | `generator_weight = 1`, `sum_generator_weights =` number of rows in the prediction file (exact for unweighted generation, which the equal per-bin generated counts indicate) |
| QCD normalization uncertainty | **not in repository** | `20 %` (`--qcd-norm-uncertainty`) |
| Luminosity uncertainty | **not in repository** | `1.6 %` (`--luminosity-uncertainty`) |
| Generator-level acceptance `A` | **not separately defined** — there is no generator-level acceptance distinct from the ≥4-jet ntuple preselection | limits are quoted on **σ × B** using the total `A × ε`; the axis is *never* labelled σ × B × A |
| Signal branching fraction | not stated explicitly; the samples are generated with `S → gg` only | the `xsec_pb` column is taken as σ × B (B = 1) |
| Observed data | **none exist** | only expected limits are produced; no observed curve is drawn (QCD simulation is not data) |
| Mass-1000 generated count | **ambiguous, resolved** — `efficiency-t2.csv` describes the 5 M sample while the predicted test split comes from the 1 M sample | only the efficiency *ratio* enters the weight, and the two agree to 1e-5, so it has no numerical effect |

`filter_efficiency` **is** resolved from the repository: it is the `efficiency`
column of `efficiency-t2.csv` (selected/generated at ntuple production, the ≥4
AK4 jet preselection). The prediction files hold only the *selected* events, so
they represent σ × filter_efficiency.

---

## 3. Selection

Applied identically to both methods, from the same `EventBlock`:

1. **Ntuple preselection** — already applied upstream (≥ 4 AK4 jets); it is the
   `filter_efficiency` in the weight formula.
2. **Di-resonance event category** — the SPANet detection probabilities of `s1`
   and `s2` are passed through `reset_collision_dp` and `dp_to_HiggsNumProb`
   ([`src/analysis/utils.py`](../../utils.py)) to give `[P_0s, P_1s, P_2s]`; the
   event is kept when the argmax is **2** (two reconstructable resonances).
   This is the same code path as `predicted_background_category.py` and
   `signal_mass_2d.py`.
3. **Alpha category** — the repository defines none, so the categories requested
   for this study (the categories of arXiv:2206.09997) are used:
   `0.15 < α < 0.25`, `0.25 < α < 0.35`, `0.35 < α < 0.50`, with
   `α = m_avg / m_4j`.
4. **m_avg range** — `300–1800 GeV` in 15 bins of 100 GeV.

> The category selection is derived from the SPANet detection probabilities and
> is deliberately held **common** to both methods. The traditional χ² baseline
> has no event-category output of its own, and inventing one would break the
> "same events" requirement. The comparison therefore isolates the **jet
> assignment**, with the event selection fixed.

### Observables

For each method, from the four jets *that method assigned*:

```
m_jj_1, m_jj_2  invariant masses of the two assigned pairs (sorted, m_jj_1 >= m_jj_2)
m_avg = (m_jj_1 + m_jj_2) / 2
m_4j            invariant mass of the four assigned jets
alpha = m_avg / m_4j
```

The traditional method always uses the leading four jets; the pairwise-attention
method may assign jets beyond the leading four (it does so in ~30 % of events),
so `m_4j` is built per method from its own four jets.

---

## 4. Normalization

Every simulated event carries

```
weight = sigma_pb * luminosity_fb * 1000
         * filter_efficiency * k_factor
         * generator_weight / sum_generator_weights
```

with `generator_weight = 1` and `sum_generator_weights` = the number of rows in
the prediction file. Summing over a whole sample therefore gives exactly

```
sum(w) = sigma_pb * luminosity_fb * 1000 * filter_efficiency * k_factor
```

which is checked to machine precision (`validation.txt`,
`qcd_yield_equals_xsec_times_lumi`).

For the signal the prediction file holds the 5 % test split. Because that split
is an unbiased subsample of the selected events, normalising to the rows
actually present makes the split fraction cancel and the sample represents the
full σ × filter_efficiency — not 5 % of it.

---

## 5. Statistical model

The repository contains no statistical framework, so **pyhf** (0.7.6) with the
**asymptotic CLs** calculator is used, with the MINUIT optimiser
(`strategy=0`).

* One model per (mass, method). The three alpha categories are three pyhf
  channels **combined in one likelihood**, sharing a single signal strength `mu`.
* Data: **background-only Asimov** (`mu = 0`). No observed limit is computed.
* Test statistic `qtilde`, CLs level 0.05.

Nuisance parameters — identical in name, type and size for both methods:

| parameter | type | value |
| --- | --- | --- |
| `staterror_qcd_<category>` | `staterror` | QCD MC statistical uncertainty, `sqrt(sum w²)` per bin |
| `staterror_sig_<category>` | `staterror` | signal MC statistical uncertainty, `sqrt(sum w²)` per bin |
| `qcd_norm` | `normsys` | configurable QCD normalization uncertainty (default ±20 %), correlated across all bins and categories |
| `lumi` | `lumi` | luminosity uncertainty (default 1.6 %), applied to signal and QCD |

### Bin selection entering the fit

Two filters, both depending only on the resonance mass and the QCD template, so
both methods get the same treatment:

* bins with zero expected QCD yield are removed (they make the likelihood
  singular and carry no information about a background-only Asimov dataset);
* only bins whose centre lies within **±50 %** of the resonance mass are fitted
  (`--mavg-window-frac`, `0` fits the full range). With the full 15-bin range
  each model carries ~90 nuisance parameters, most attached to far-off-peak bins
  with no signal, and MINUIT does not converge reliably; the CLs curve becomes
  non-monotonic and POI-bound dependent. Dropping those bins also *weakens* the
  global constraint on `qcd_norm`, so the choice is conservative.

### Limit extraction

A geometric bracket locates the interval containing the CLs = 0.05 crossing of
all five expected quantiles; a coarse (12 point) and then a refined (10 point)
linear scan of the pyhf asymptotic CLs follows, and the crossing is taken by
linear interpolation — the same rule pyhf's own `upper_limit` grid scan uses.
One hypothesis test yields all five quantiles at once, which is what makes the
scan affordable for these ~50-parameter models. CLs values are made
monotonically non-increasing (running minimum) before interpolation, and NaN —
which pyhf returns deep inside the excluded region, where CL_{s+b} and CL_b both
underflow — is mapped to CLs = 0.

---

## 6. Known limitation: QCD MC statistics

The QCD pt-hat samples have roughly equal event counts but wildly different
cross sections, so the low-pt-hat bins carry enormous per-event weights. After
the di-resonance selection, most of the background yield comes from a small
number of MC events in the low-pt-hat bins, and the per-bin QCD MC statistical
uncertainty reaches tens of percent — orders of magnitude larger than the signal
in the same bin.

**The QCD MC statistical uncertainty is the dominant systematic in this fit**, and
the absolute limits are correspondingly weak (far above the simulated cross
section). The per-sample composition is printed by `run_limits.py` and the
uncertainty band is drawn by `plot_templates.py`. The *relative* comparison of
the two pairing methods, which uses identical events, identical binning and
identical nuisance parameters, is unaffected by this common limitation.

---

## 7. Outputs

Written to `--output-dir` (default `/maad-vol/hhh_analysis/maad_plots/dijet_resonance_mass`):

| file | contents |
| --- | --- |
| `figure12_limits.pdf` / `.png` | two Figure-12-style panels (traditional, pairwise): log-y expected 95 % CL upper limit vs resonance mass, dashed median, green ±1σ band, yellow ±2σ band, dot-dashed theory σ × B, "Simulation" + luminosity + energy label (no CMS branding) |
| `figure12_comparison.pdf` / `.png` | top: the two median expected limits overlaid; bottom: `expected_limit_traditional / expected_limit_pairwise` (> 1 means pairwise attention is more sensitive) |
| `templates_mavg.pdf` / `.png` | diagnostic: the m_avg templates per alpha category and method, with the QCD MC-stat band |
| `limits.csv` | one row per (mass, method): theory σ × B, S and B yields, A × ε, number of fit bins and parameters, and the five expected quantiles in both `mu` and fb |
| `templates.npz` | the binned templates (sum w, sum w², raw counts) for redrawing |
| `model_config.json` | the full pyhf spec for one representative model, all configuration values, the binning and categories, and `unresolved_inputs` |
| `validation.txt` | the validation checks |
| `run_limits.log` | the full console log |

---

## 8. Validation checks

Run automatically and written to `validation.txt`:

1. `qcd_yield_equals_xsec_times_lumi` — every sample's summed weight equals
   `σ × L × filter_efficiency × k` to a relative tolerance of 1e-9.
2. `qcd_total_yield_bookkeeping` — the total is compared with the unfiltered
   `σ × L × k`; their ratio is reported and equals the preselection efficiency.
3. `both_methods_same_input_events` / `same_selection_applied_to_both_methods` —
   both pairings are built inside one loop over one `EventBlock` per file; the
   per-file event fingerprints and selected-event counts are recorded.
4. `no_invalid_or_negative_bins` — no NaN, infinity or negative yield in any
   template bin.
5. `fit_bins_have_positive_background` — reports how many bins were removed per
   category, and why.
6. `no_fabricated_mass_points` — the limit curves contain one point per
   simulated mass and nothing else; nothing is interpolated or extrapolated
   beyond the 11 simulated masses.

---

## 9. Commands

All from `/maad-vol/hhh_analysis`.

```bash
# Full analysis: templates, limits, validation, CSV and model config
python -m src.analysis.maad.dijet_limits.run_limits \
    --output-dir /maad-vol/hhh_analysis/maad_plots/dijet_resonance_mass

# Figures (re-runnable from limits.csv / templates.npz without refitting)
python -m src.analysis.maad.dijet_limits.plot_limits \
    --input-dir /maad-vol/hhh_analysis/maad_plots/dijet_resonance_mass
python -m src.analysis.maad.dijet_limits.plot_templates \
    --input-dir /maad-vol/hhh_analysis/maad_plots/dijet_resonance_mass
```

Useful variations:

```bash
# Different luminosity / energy assumption
python -m src.analysis.maad.dijet_limits.run_limits --luminosity-fb 300 --sqrt-s-tev 14

# Different QCD normalization uncertainty
python -m src.analysis.maad.dijet_limits.run_limits --qcd-norm-uncertainty 0.50

# Fit the full m_avg range instead of a window (slow, less numerically stable)
python -m src.analysis.maad.dijet_limits.run_limits --mavg-window-frac 0

# Statistics-only cross-check (no MC-stat nuisances)
python -m src.analysis.maad.dijet_limits.run_limits --no-mc-stat

# A subset of masses, for a quick check
python -m src.analysis.maad.dijet_limits.run_limits --masses 700 1200
```

Dependencies beyond the existing environment: `pyhf` and `iminuit`
(`pip install pyhf iminuit`).

---

## 11. Figure 9 — the average dijet mass spectrum

`figure9_spectrum.py` reproduces Fig. 9 of arXiv:2206.09997: the m̄_jj spectrum
in the three α bins of the nonresonant search, one figure per pairing method
**and per signal mass hypothesis**.

`--signal-mass` is required and takes a single mass. In the paper each mass
point is a separate search — the background function plus *one* simulated signal
shape is fitted to the data, and the scan is repeated mass by mass — so mass
points are never summed or overlaid here either. Run the script once per mass.

Each of the three panels shows

* the **LO QCD multijet simulation** (green, filled) with its MC-statistical
  uncertainty;
* the **three smooth background parameterisations** fitted to it, all with three
  free parameters and all in terms of `x = m / √s`:

  | name | formula | style |
  | --- | --- | --- |
  | PowExp-3p | `p0 exp(−p1 x) / x^p2` | red dotted |
  | Dijet-3p | `p0 (1 − x)^p1 / x^p2` | red dashed |
  | ModDijet-3p | `p0 (1 − x^(1/3))^p1 / x^p2` | red solid |

* **the one signal mass hypothesis** of that figure (blue solid), drawn at the
  cross section of the simulated sample (`data/octet_uaf/efficiency-t2.csv`),
  matching the paper's "cross sections equal to the expected SUSY cross
  sections";
* a **pull panel** with the pulls of the QCD spectrum with respect to the
  ModDijet-3p fit, the same signal expressed as pulls, and χ²/NDF.

Only the **background** is fitted. The signal is not fitted and carries no free
normalisation in this figure: it is the simulated template at its own σ × B,
drawn for scale. Fitting a signal strength is the job of `run_limits.py`
(section 5).

### Necessary difference from the published figure

The paper plots **collision data** as points and fits the functions to them.
This repository contains no collision data, and QCD simulation is not data, so:

* the three functions are fitted to the **QCD simulation**;
* the pulls use the **QCD MC-statistical uncertainty** (`√Σw²`) in place of the
  statistical uncertainty of the data;
* nothing is labelled "Data" and no observed points are drawn.

Everything else — event selection, di-resonance category, α bins, weights,
luminosity — is exactly as in sections 3 and 4, and identical between the two
pairing methods.

### Binning and fit range

`350–2500 GeV` in 50 GeV bins (`--mavg-min`, `--mavg-max`, `--bin-width`); the
paper fits 0.35–3.0 TeV, the upper end being reduced here because the simulated
QCD sample runs out of events. Empty bins carry no MC-statistical uncertainty
and are excluded from the fit and the pulls. The spectrum is plotted as events
per GeV so the normalisation is independent of the bin width.

### What the fits show

None of the three-parameter functions describes the simulated spectrum well:
χ²/NDF is 1.9–5.8 across the six (method, α bin) combinations. Two causes, both
reported in the run log:

1. **The pt-hat stitching.** The samples are generated in pt-hat bins with
   roughly equal event counts but cross sections spanning nine orders of
   magnitude, so per-event weights reach ~10⁶. Many high-mass bins have an
   effective sample size `n_eff = (Σw)²/Σw²` below 10 — as low as 2–4. The
   apparent structure near 1.5–1.9 TeV in the traditional pairing's
   `0.35 < α < 0.50` panel is exactly this: a handful of very-high-weight
   events from the pt-hat 100–200 GeV sample, **not** a shape difference between
   the pairing methods. The run log prints, per method and α bin, how many
   filled bins have `n_eff < 10` and where they start.
2. **No trigger or analysis-level kinematic selection exists in the
   repository**, so the spectrum extends to much lower masses and much higher
   rates than the published measurement, where the functions were only ever
   required to work above the trigger turn-on.

Above roughly 1.5 TeV the simulated QCD shape should not be treated as a
measurement of the background in either method.

### Outputs

| file | contents |
| --- | --- |
| `figure9_mavg_traditional_m<mass>.pdf` / `.png` | Figure-9-style spectra, traditional χ² pairing, one signal mass |
| `figure9_mavg_pairwise_m<mass>.pdf` / `.png` | Figure-9-style spectra, pairwise-attention pairing, one signal mass |
| `figure9_fits.csv` | fitted `p0, p1, p2`, χ², NDF, χ²/NDF, bins fitted and convergence flag for every (method, α bin, function); background-only, so identical for every mass |
| `figure9_spectrum_m<mass>.npz` | the binned spectra (Σw, Σw², raw counts) for QCD and that signal mass |

### Commands

```bash
cd /maad-vol/hhh_analysis

# One mass point
python -m src.analysis.maad.dijet_limits.figure9_spectrum --signal-mass 1000 \
    --output-dir /maad-vol/hhh_analysis/maad_plots/dijet_resonance_mass

# The full 11-point scan (~2 min each)
for m in 500 600 700 800 900 1000 1100 1200 1300 1400 1500; do
    python -m src.analysis.maad.dijet_limits.figure9_spectrum --signal-mass $m \
        --output-dir /maad-vol/hhh_analysis/maad_plots/dijet_resonance_mass
done

# Different binning or fit range
python -m src.analysis.maad.dijet_limits.figure9_spectrum \
    --signal-mass 900 --mavg-min 400 --mavg-max 2000 --bin-width 100
```

---

## 12. Module map

| file | role |
| --- | --- |
| `config.py` | all paths, samples, binning, categories, systematics and the `UNRESOLVED_INPUTS` record |
| `pairing.py` | loads one prediction file into an `EventBlock` and produces both pairings' observables from it |
| `normalization.py` | cross sections, filter efficiencies, per-event weights |
| `templates.py` | fills the m_avg templates for both methods in one pass and returns the per-sample bookkeeping |
| `stats.py` | pyhf model construction and the asymptotic CLs expected limits |
| `validation.py` | the validation checks |
| `run_limits.py` | Figure 12 driver — writes `limits.csv`, `templates.npz`, `model_config.json`, `validation.txt` |
| `plot_limits.py` | `figure12_limits` and `figure12_comparison` |
| `plot_templates.py` | the template diagnostic figure |
| `background_functions.py` | PowExp-3p, Dijet-3p, ModDijet-3p and the weighted least-squares fitter |
| `figure9_spectrum.py` | Figure 9 — the m̄_jj spectra, fits and pull panels (SPANet pairwise pairing) |
| `run_pairing_observables.py` | runs either pairing (`--pairing chi2\|spanet`) over its inputs, writes one-to-one pairings + observables |
| `figure9_pairing_spectrum.py` | Figure-9-style spectrum for one pairing and one α bin, with the paper's mass-asymmetry selection |

---

## 13. The χ² baseline as a standalone analysis

### Why the old "traditional" panels were withdrawn

`figure9_spectrum.py` reads its events from the SPANet prediction files and
applies the SPANet di-resonance category as the event selection (see §3). That
is defensible for the *comparison* — it isolates the jet assignment with the
event selection held fixed — but it means the `traditional` panels were never a
standalone baseline: a SPANet-derived cut was applied to a χ² pairing. Those
files (`figure9_mavg_traditional_m*.pdf`/`.png`) have been deleted.

### What replaced them

The χ² baseline has no event-category output of its own, and **no cut-off study
of the χ² baseline against QCD exists anywhere in this repository** — its only
entry point, `src/models/mass_agnostic_baseline.main`, requires truth `TARGETS`
to compute efficiency and purity, so it had only ever been run on signal. The
selection is therefore taken from arXiv:2206.09997 instead, starting with the
mass asymmetry:

```
A = |m_jj_1 − m_jj_2| / (m_jj_1 + m_jj_2)

A <  0.1   signal region
A >= 0.1   background region
```

`run_pairing_observables.py --pairing chi2` runs the baseline over `/maad-vol/data/octet_uaf_test_datasets`
and `/maad-vol/data/qcd` (inclusive samples excluded) and writes
`/maad-vol/baseline_predictions/mass_agnostic_chi2/<stem>.h5`, one row per input
event: the pairing (`TARGETS/s1,s2/g1,g2`) plus `OBSERVABLES/{m_jj_1, m_jj_2,
m_avg, m_4j, alpha, asymmetry}`. The observable arithmetic is byte-identical to
`pairing._pair_observables` (verified to 0.0 max difference).

The same producer run with `--pairing spanet` reads the SPANet prediction files
instead and writes `/maad-vol/baseline_predictions/spanet_pairwise/`. The
prediction files carry a verbatim copy of `INPUTS/Jets`, so both modes see the
same events in the same order — both produce 2 988 543 rows over 31 files — and
their outputs are comparable row by row.

`figure9_pairing_spectrum.py` then draws one α range, one pairing and one signal
mass per invocation into `<output-dir>/alpha_<lo>_<hi>/`, so the two pairings
land side by side in the same α directory
(`figure9_chi2_m<mass>.*` and `figure9_spanet_m<mass>.*`):

| component | selection | style |
| --- | --- | --- |
| background — **the only thing fitted** | QCD, `A ≥ 0.1` | green filled |
| false positives | QCD, `A < 0.1` | orange |
| signal | that mass, `A < 0.1` | blue |

### α binning

α = m̄_jj/m₄ⱼ is bounded above by 0.5 (two resonances cannot average more than
half the four-jet mass), so the scan covers the whole physical range: eight
uniform 0.05-wide bins from 0.10 to 0.50, plus the paper's three reference bins
(0.15–0.25, 0.25–0.35, 0.35–0.50) for comparison. One directory per range.

### Fit caching

The three parameterisations are badly conditioned on this spectrum — the density
spans ten orders of magnitude with relative MC-statistical errors up to 100% —
and a single `least_squares` fit costs ~2 minutes. The background is the QCD
`A ≥ 0.1` spectrum, which does not depend on the signal mass, so an 11-mass scan
would refit the same thing 11 times (~3.7 h for the full grid). The fit is
therefore cached per (α bin, pairing) in `background_fit_<pairing>.json`, keyed
on a SHA-256 of the exact `(sumw, sumw2, edges, √s)` it was fitted to — a cache
hit is only possible for a byte-identical spectrum. Measured: 107.8 s cold,
3.1 s warm, identical χ²/NDF. `--refit` forces recomputation.

### Not applied yet

Only the mass-asymmetry requirement. The paper also requires an `HT > 1050 GeV`
(or AK8 `pT > 550 GeV`) trigger, jets with `pT > 80 GeV` and `|η| < 2.5`,
`ΔR₁,₂ < 2.0` within each dijet, and `|Δη| < 1.1` between the two dijets. Two
cautions for whoever adds them:

* the paper's pairing *is* a ΔR-based algorithm (minimise
  `|ΔR₁ − 0.8| + |ΔR₂ − 0.8|`), so adopting it wholesale would replace the
  pairing under test rather than select on it — the ΔR/Δη/A cuts must be applied
  downstream of whichever pairing is being evaluated;
* `A < 0.1` is not neutral between pairing methods: the χ² baseline minimises
  very nearly that quantity by construction, so it is biased in the baseline's
  favour and must not be used as-is for a baseline-vs-SPANet comparison.
