# Where does the Freeze Index fail? Development and evaluation plan

This plan is committed before the development analyses are run. It describes the intended work. Development choices may still be refined during the development stage, and each refinement will be recorded in this repository with its date and reason. The configuration will be frozen before any external dataset is opened. Any change after the freeze will be added as a dated amendment that states what changed and why.

## Question

The Freeze Index (FI) is a simple and interpretable FoG detector. This study asks where an FI-based cue controller fails on a single lower-back accelerometer, whether targeted and interpretable additions reduce these failures, and how close they bring the controller to small learned models.

## Data

Development data (open during development):
- tDCS-FoG and DeFOG training recordings of the Kaggle FoG contest (Zenodo copy). Per-episode FoG type (start hesitation, turn, walking) and kinetic or akinetic labels are taken from `events.csv`. DeFOG task labels are taken from `tasks.csv`. DeFOG samples are scored only where `Valid` and `Task` are true.
- Daphnet trunk accelerometer. The Freeze Index was first proposed on related data, so Daphnet is used for development only.

External data (planned; not opened until the configuration is frozen):
- FoG-STAR lower-back sensor (episode severity codes 1–3: shuffling, trembling, akinesia).
- Stanford turning and barrier course dataset (O'Day et al. 2022), lumbar sensor.
- Multimodal FoG dataset (Zhang et al. 2022), waist sensor.

The external set may be adjusted before the freeze if a dataset proves unusable (for example missing labels or sensors). Any such change will be documented.

## Signal processing

Acceleration in g is resampled to 64 Hz. Decisions are made every 0.25 s from causal windows ending at the decision time. The Freeze Index is the ratio of power in 3–8 Hz to power in 0.5–3 Hz, pooled over the three axes, in a 4-s window. Additional interpretable features include band powers, axis-specific FI, a 2-s FI, a motion-arrest index (current locomotor-band power relative to the preceding 4–10 s), dominant frequency, step regularity and trunk orientation.

## Controllers (ladder)

- D0: FI threshold.
- D1: D0 with an energy gate.
- D2: D1 with causal smoothing.
- D3: D2 with hysteresis and a minimum cue duration.
- D4: D3 combined with a motion-arrest rule aimed at akinetic freezing.
- D5: logistic regression on the interpretable features, with the same smoothing and hysteresis.
- D6: small causal convolutional network on raw acceleration, as a learned reference.

## Selection and evaluation

Subject-grouped five-fold cross-validation is planned within each development dataset. Thresholds and controller settings are selected on the training subjects of each fold. The planned rule maximises episode coverage subject to a false-alarm budget (main budget 10 per hour; 5 and 15 per hour reported as sensitivity analyses) and a cue specificity of at least 0.80. Ties are broken by timely activation and then by fewer false alarms.

Event-level metrics: episode coverage, timely activation (cue on at onset, started at most 4 s earlier), false alarms per hour and cue-on time. Chance level is estimated by circular shifts of the cue sequence within each recording. Window-level AUC is reported for context.

Failure analysis: coverage and timely activation by FoG type and by kinetic or akinetic episodes, and false alarms by DeFOG task group.

## Freeze and external test

After development, the selected controller(s) and thresholds will be fixed and recorded with file hashes in this repository. The external datasets will then be analysed with the frozen configuration. Results will be reported whether favourable or not. Analyses added after the freeze will be labelled as post hoc.
