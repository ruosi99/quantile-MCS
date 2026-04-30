# Journal Branch Risk Register

## Purpose
This document records the main implementation and publication risks for the multi-horizon stratified conformal idea before heavy coding begins.

## High-Risk Items

### 1. Multi-horizon is not a local patch
The current code path is single-step end to end. Moving to direct multi-horizon forecasting requires coordinated changes across:
- data window creation
- label shape
- model output shape
- loss computation
- checkpoint compatibility
- inference saving
- calibration wrappers
- metrics and plotting

Risk:
- a partial implementation can silently produce wrong tensor alignment or evaluate only the last horizon while appearing to run successfully

### 2. Current model output conventions are already shape-sensitive
The present code comments and actual tensor behavior are not fully aligned in places, especially around the Informer output and quantile head assumptions.

Risk:
- extending to `(B, N, H, Q)` without formalizing a shape contract will create subtle bugs in training and calibration

### 3. Stratified conformal may suffer from sparse cells
Stratifying jointly by horizon, hour-of-day, and station archetype can create many small cells.

Risk:
- noisy thresholds
- unstable coverage estimates
- inability to support fine-grained claims for rare hours or station types

Required mitigation:
- minimum-sample thresholds
- fallback to coarser buckets or parent thresholds
- explicit reporting of effective sample sizes

### 4. Station archetypes may be fragile
The clustering plan uses mean load, volatility, peak ratio, and graph degree. This is cheap and practical, but it introduces design sensitivity.

Risk:
- clusters may reflect scale only
- cluster assignments may not transfer cleanly across cities
- reviewer questions may focus on arbitrariness of feature choice and cluster count

Required mitigation:
- keep archetype construction simple and reproducible
- report clustering features and cluster sizes clearly
- include ablation on the number of archetypes or granularity

### 5. Long-horizon quality may collapse
The horizons `{1, 3, 6, 12, 24}` are scientifically meaningful, but the current backbone may degrade sharply at longer horizons.

Risk:
- poor point accuracy at `H = 12` and `H = 24`
- intervals become too wide to be practically useful
- calibration appears strong only because intervals inflate

Required mitigation:
- report point metrics and interval metrics together
- monitor sharpness versus coverage tradeoff
- be prepared to narrow the horizon set if the longest horizon is not defensible

### 6. One-sided decision calibration depends on target quantile support
The decision layer needs a target quantile `tau_star = c_u / (c_u + c_o)`.

Risk:
- `tau_star` may not lie exactly on the trained quantile grid
- interpolating between learned quantiles can weaken the clean theoretical story
- training many quantiles just to support a range of cost ratios may increase complexity

Required mitigation:
- define in advance whether `tau_star` must be on-grid, interpolated, or approximated by nearest quantile
- keep the decision study aligned with the quantiles the model actually supports

### 7. The five-city transfer claim may be blocked by data reality
The plan mentions Shenzhen first and then calibration-only transfer to five cities.

Risk:
- the repository may not yet contain the required five-city datasets in a consistent format
- station archetype features may not transfer cleanly
- transfer may become a separate data-engineering project

Required mitigation:
- treat cross-city transfer as a phase-two goal until data availability is confirmed
- do not center the early implementation around transfer claims

### 8. Novelty challenge: "this is Mondrian CP"
Reviewers may argue that the calibration contribution is a standard stratified or Mondrian conformal variant.

Risk:
- novelty is judged incremental if the paper does not show why the EV network setting truly needs this design

Required mitigation:
- emphasize the failure mode of global CQR in this application
- show diagnostics that motivate the stratification choices
- combine forecasting, calibration, and decision evaluation into one coherent paper story

## Medium-Risk Items

### 9. Coverage theorem may be true but not persuasive
The within-stratum split-conformal theorem is useful for rigor, but it is standard.

Risk:
- theorem effort may consume time without materially strengthening acceptance odds

Recommendation:
- keep the theorem short and practical
- invest more effort in diagnostics, robustness, and decision results

### 10. Metric burden may explode
The experiment list includes point metrics, interval metrics, reliability curves, heatmaps, regret, and ablations.

Risk:
- large reporting surface
- duplicated scripts
- slow iteration before the core method stabilizes

Recommendation:
- implement metrics in layers:
  1. point metrics by horizon
  2. coverage and width by horizon
  3. heatmap diagnostics
  4. decision analysis
  5. only then extra scoring rules and transfer

## If Results Are Weak
- collapse 24 hourly buckets into 4 to 6 coarse time buckets
- add minimum-sample fallback from stratum threshold to horizon-wise or global threshold
- reduce the number of archetypes
- narrow the paper claim toward "robust calibration diagnostics and decision-aware calibration"
- avoid claiming methodological novelty from stratification alone

## Recommendation Before Major Coding
Start with a three-stage implementation order:
1. Direct multi-horizon forecasting with global calibration only
2. Horizon-wise calibration and horizon diagnostics
3. Stratified calibration and one-sided decision calibration

This staging reduces the risk of mixing architecture bugs with calibration bugs.
