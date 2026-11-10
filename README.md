# fi-cue-controller
[![DOI](https://zenodo.org/badge/1439634254.svg)](https://doi.org/10.5281/zenodo.23955452)

Measured failure modes of the Freeze Index and an interpretable freezing-of-gait (FoG) cue controller, tested on unseen data with a single lower-back accelerometer.

This repository holds the analysis plan, the amendment, the code, the frozen configurations, the results and the embedded implementation of the study.
The plan was committed before the development analyses. The controllers were frozen before any external data were opened.

## Timeline of commits
| Commit | Content |
|---|---|
| 36ed8be | Development plan and code, committed before the development analyses. |
| b230b9f | Frozen development configuration (`results_fi/frozen/`), committed before any external data were analysed. |
| b66bf06 | Development and external test results (external test run once with the frozen configuration). |
| e49a51a | Post hoc analyses (bootstrap CIs, matched false-cue comparison, posture-free logistic regression). |
| 7875f63 | Amendment 1: v2 controller plan and code, committed before v2 development. |
| 568dc47 | Frozen v2 configuration, committed before any v2 test. |
| f1d7ae5 | v2 and frozen v1 test results on the held-out contest test recordings (run once). |
| bd0d198 | Post hoc strict episode-coverage script (`s22_strict_coverage.py`), committed before it was run. |
| b1e89c2 | Strict episode-coverage results and Benjamini–Hochberg correction (`results_fi/posthoc/strict_coverage*.csv`). |

Later commits add the post hoc checks, the embedded implementation and the archive metadata.

## Documents
- `PLAN.md`: development plan.
- `AMENDMENT_1.md`: Amendment 1 (v2 controller).
- `STEPS_AR.md`, `STEPS_V2_AR.md`: run instructions (Arabic).

## Code and order of execution
| Step | Script |
|---|---|
| Environment check | `s0_check.py` |
| Data preparation | `s1_prepare.py` |
| Features and development | `s2_features.py`, `s3_ladder.py`, `s4_cnn.py`, `s5_report.py` |
| Freeze | `s6_freeze.py` |
| External test | `s7_external_review.py`, `s8_external_test.py`, `s9_figures.py` |
| Post hoc | `s10_posthoc.py` |
| Amendment 1 (v2) | `s11_v2_develop.py`, `s12_kaggle_test_prep.py`, `s13_v2_test.py` |
| Further checks | `s14_sensitivity.py`, `s15_patient_latency.py`, `s16_latency_window.py`, `s17_home_smoothing.py`, `s18_nested_tuning.py`, `s19_precision_gate_ap.py`, `s20_precision_chance_cnn_test.py`, `s21_timing_literature_fi.py` |
| Strict episode coverage (post hoc) | `s22_strict_coverage.py`, `s23_strict_bh.py` |
| Embedded | `mcu/` (C code for a Cortex-M4F, emulator run and agreement with Python) |

Shared settings are in `fi_config.py`. Set the data paths to local copies of the datasets before running.

## Data
The study uses public data: the FoG detection contest data (Salomon et al., Nat. Commun. 2024; Zenodo doi:10.5281/zenodo.12565292), Daphnet, FoG-STAR, a multimodal FoG dataset and a Stanford FoG dataset.
The raw data are not redistributed here. Files in `results_fi/` contain derived results only.

## Licence
Code: MIT. Documents and results: CC BY 4.0.
