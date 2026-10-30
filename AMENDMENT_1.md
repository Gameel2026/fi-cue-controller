# Amendment 1: a second controller version (v2) and a new held-out test

This amendment is committed before the v2 controller is applied to any test data.

## Background

The v1 configuration was frozen and tested once on the external datasets (results in `results_fi`). Those results are kept unchanged and remain the primary external test of v1. Post hoc analyses (`s10_posthoc.py`) were then run.

## Reasons for v2

Development data showed three limitations of v1: low coverage of akinetic episodes and start hesitation, false alarms concentrated in turning tasks, and logistic-regression weights on trunk-orientation features that changed sign between development datasets (for example the mean anteroposterior acceleration). The post hoc external analyses were also seen before this amendment. This is stated openly, because it may have influenced the design.

## v2 design

- Orientation-free features only: band powers pooled over the three axes at 1-, 2- and 4-s windows, Freeze Index at each scale, high-band ratio, dominant frequency, spectral entropy, step regularity and jerk.
- Wearer-relative features: power features expressed relative to a causal running median of the same recording (about 60 s), following the adaptive-baseline idea of Pham et al. (2017).
- Short-term dynamics: changes of selected features over the last 1 and 2 s.
- A gradient-boosted tree classifier trained on all development datasets together, with equal weight per dataset. The training target covers each episode and the 2 s before its onset.
- The same causal smoothing and hysteresis controller, with thresholds selected by the same rule as v1 (false-alarm budget and cue specificity of at least 0.80) on out-of-fold predictions.

Settings may be refined during development of v2. Any refinement will be recorded here with its date before the test.

## Test plan for v2

- Primary test: the public and private test recordings of the Kaggle FoG contest (Zenodo copy), excluding the files listed by the data providers as overlapping with training subjects. These recordings have not been used in this project. tDCS-FoG test files use the laboratory thresholds and DeFOG test files use the home thresholds.
- The frozen v1 controllers will be tested on the same recordings for comparison.
- Secondary analysis: v2 will also be applied to FoG-STAR, the Multimodal dataset and the Stanford dataset. Because v1 results on these datasets were already known, this analysis will be reported as secondary and not as an independent test.
- Metrics, chance estimation and confidence intervals follow the plan, with subject-level bootstrap intervals added.
