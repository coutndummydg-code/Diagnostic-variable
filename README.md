# Automatic Variable Diagnosis and Selection System

A generic CSV analysis tool designed to identify which input variables provide useful information about one or more target variables without relying on a single statistical method.

The project combines statistical association, non-linear dependence, multivariable modelling, permutation importance, redundancy analysis, interaction screening, synthetic noise references, and stability checks to produce an explainable variable-level recommendation.

> **Version:** 1.0  
> **Status:** Experimental / validation stage  
> **Main script:** `variable_diagnostic.py`

---

## Why this project exists

A single correlation coefficient is rarely enough to decide whether a variable is useful.

A predictor may:

- have a strong linear relationship with the target;
- have a non-linear relationship that Pearson correlation misses;
- be useful only when combined with another variable;
- duplicate information that already exists in another predictor;
- appear important only in some data partitions;
- look important simply because of random variation;
- contain useful association without being causal.

This project therefore treats variable selection as an **evidence-combination problem** rather than a correlation-ranking problem.

The central idea is:

```text
CSV
 |
 v
Data inspection and type inference
 |
 +--> Level 1: individual X -> Y evidence
 |
 +--> Level 2: multivariable predictive evidence
 |
 +--> X <-> X redundancy
 |
 +--> Interaction screening
 |
 +--> Synthetic noise reference
 |
 +--> Stability analysis
 |
 v
Normalized evidence
 |
 v
Scores + decision rules
 |
 v
USE / REDUNDANT / UNSTABLE / REVIEW / POSSIBLE NOISE
```

---

## Main features

### Automatic variable-type detection

The analyzer can recognize or process:

- Boolean values;
- binary values such as `0/1` and `OK/NOK`;
- categorical variables;
- ordinal variables through optional type overrides;
- discrete numerical variables;
- continuous numerical variables;
- percentages;
- date/time values;
- vectors or lists stored inside CSV cells;
- matrices stored inside CSV cells;
- likely identifiers;
- constant columns.

Automatic inference can be overridden when domain knowledge provides a better interpretation.

### Multiple target types

A target `Y` can represent, among other cases:

```text
True / False
OK / NOK
A / B / C / D
0 / 1 / 2 / 3 / 4
continuous measurements
percentages
vectors
matrices
```

Multiple target columns can be analyzed in the same execution by repeating the `-y` argument.

### Two analysis levels

#### Level 1: individual evidence

Each predictor is compared individually with the target using methods compatible with the detected variable types.

The system can use evidence derived from:

- Pearson correlation;
- Spearman correlation;
- chi-square;
- Cramér's V;
- eta squared / correlation ratio;
- Kruskal-Wallis as auxiliary inferential evidence;
- Mutual Information.

#### Level 2: joint evidence

All usable predictors are analyzed together using a tree-based model.

The system then estimates how much each predictor contributes to the combined model through permutation importance.

This helps distinguish:

```text
important by itself
```

from:

```text
still useful after the other variables are known
```

### Redundancy analysis

Predictors are also compared against one another.

A variable can therefore be informative about `Y` but still be classified as `REDUNDANT` when most of its information overlaps with another predictor.

### Interaction screening

The analyzer performs a bounded search for pairwise numerical interactions among the strongest candidates.

This is intended to identify situations where two variables become useful together even if their individual evidence is weaker.

### Synthetic noise reference

The system deliberately disrupts information in the test data to obtain a practical random-reference baseline.

This provides additional evidence for distinguishing weak apparent importance from more convincing predictive contribution.

### Stability analysis

The model analysis is repeated across several train/test partitions.

Variables whose importance changes substantially between partitions can be identified as unstable.

---

## Final diagnostic states

Every predictor/target combination receives one primary diagnosis.

### `USE`

Evidence indicates that the predictor contains useful information about the target and should normally be retained for further analysis.

### `REDUNDANT`

The predictor contains relevant information, but a large part of that information overlaps with another variable.

`REDUNDANT` does **not** mean defective or useless. A redundant sensor may still have operational value as a backup or validation channel.

### `UNSTABLE`

The variable appears useful, but its importance changes too much across repeated data partitions.

This can be a reason to investigate process conditions, batches, machines, shifts, time periods, or other hidden structure.

### `REVIEW`

The available evidence is insufficient, contradictory, intermediate, or affected by data-quality limitations.

This is the conservative fallback state.

### `POSSIBLE NOISE`

Several evidence channels indicate that the variable currently contributes little useful information and does not convincingly exceed the random-reference behavior.

This does **not** prove that the variable can never be useful.

---

## Relevance, confidence, and quality

The system deliberately separates three concepts.

### Relevance

How much information the variable appears to contribute about the target.

The v1.0 relevance score is:

```text
Relevance =
    0.30 * UnivariateEvidence
  + 0.30 * MutualInformation
  + 0.30 * PermutationScore
  + 0.10 * InteractionScore
```

### Confidence

How consistent the available evidence is.

Agreement between evidence channels, data quality, and stability contribute to this score.

### Quality

How suitable the available observations are for the analysis.

Missing data and small effective sample size reduce this score.

A quality veto prevents a strong-looking statistic from automatically producing a strong recommendation when the underlying data are inadequate.

---

## Project structure

A minimal project directory can be organized as:

```text
variable-diagnostic/
|
|-- variable_diagnostic.py
|-- README.md
|-- Variable_Analyzer_Practical_Manual_EN.md
|-- Variable_Analyzer_Technical_Manual_EN.md
|
|-- data/
|   `-- example.csv
|
|-- reports/
|   |-- diagnostic.csv
|   `-- diagnostic.html
|
`-- config/
    `-- types.json
```

Only `variable_diagnostic.py` is required to run the analyzer. The other files document or organize the project.

---

## Requirements

The current implementation is written in Python and uses:

- Python 3;
- NumPy;
- pandas;
- SciPy;
- scikit-learn.

A typical installation is:

```bash
python -m pip install numpy pandas scipy scikit-learn
```

For reproducible deployments, pin tested dependency versions in a `requirements.txt` or equivalent environment definition after validation in the intended environment.

---

## CSV format

A normal input file has one observation per row and one variable per column.

Example:

```csv
Temperature,Pressure,Speed,Humidity,Machine,Status
92.1,3.2,105,41,A,OK
93.4,3.1,103,43,A,OK
108.2,3.8,102,45,B,NOK
106.7,3.7,99,46,B,NOK
```

In this example:

```text
X variables:
Temperature
Pressure
Speed
Humidity
Machine

Y variable:
Status
```

---

## Basic usage

Analyze one target:

```bash
python variable_diagnostic.py process.csv -y Status
```

The default output files are:

```text
diagnostico_variables.csv
diagnostico_variables.html
```

### Custom output names

```bash
python variable_diagnostic.py process.csv \
  -y Status \
  --output diagnostic.csv \
  --html diagnostic.html
```

### Multiple targets

Repeat `-y` for each target:

```bash
python variable_diagnostic.py process.csv \
  -y Status \
  -y DefectType \
  -y QualityScore
```

Each target is evaluated independently against the available predictors.

---

## Type overrides

Automatic type detection is useful, but it cannot know the semantic meaning of every column.

For example:

```text
1, 2, 3, 4, 5
```

could represent a measurement, a category code, or an ordered severity scale.

A JSON file can provide explicit type overrides.

Example `types.json`:

```json
{
  "Severity": "ordinal",
  "ProcessStage": "categorical"
}
```

Run the analyzer with:

```bash
python variable_diagnostic.py process.csv \
  -y Status \
  --types types.json
```

---

## Advanced options

### Noise repetitions

```bash
--noise-reps 8
```

Controls the number of random-reference disruption repetitions.

### Stability repetitions

```bash
--stability-reps 5
```

Controls the number of additional train/test splits used to estimate stability.

### Interaction search size

```bash
--interaction-top 8
```

Limits pairwise interaction screening to the strongest preliminary candidates, reducing computation on wide datasets.

Example:

```bash
python variable_diagnostic.py process.csv \
  -y Status \
  --noise-reps 15 \
  --stability-reps 10 \
  --interaction-top 12
```

Increasing these values can increase execution time substantially.

---

## Output fields

The detailed CSV may contain fields such as:

```text
variable
target
type_x
type_y
univariate
mutual_info
pearson
spearman
cramers_v
eta2
p_min
permutation
interaction
interaction_with
redundancy
redundant_with
stability
noise_baseline
beats_noise
relevance
confidence
quality
decision
flags
```

Not every statistical field applies to every X/Y type combination. An unavailable method should not be interpreted as evidence equal to zero.

---

## Suggested report-reading order

For practical use, inspect the output in this order:

1. `decision`
2. `relevance`
3. `confidence`
4. `quality`
5. `redundancy` and `redundant_with`
6. `interaction` and `interaction_with`
7. `beats_noise`
8. `stability`
9. `flags`
10. individual statistical evidence when deeper investigation is required

---

## Example interpretation

A result might conceptually look like:

```text
Temperature
  Decision:    USE
  Relevance:   high
  Confidence:  high
  Quality:     high

Pressure
  Decision:    REDUNDANT
  Redundant with: Temperature

Sensor_14
  Decision:    REVIEW
  Reason:      inconsistent evidence

Sensor_27
  Decision:    POSSIBLE NOISE
  Reason:      weak individual and joint evidence
```

The analyzer is intended to answer not only **what** the recommendation is, but also **why** it was produced.

---

## Important limitations

### This is not causal inference

A predictor that explains or predicts the target does not automatically cause the target.

Use process knowledge, controlled experiments, or Design of Experiments when causal conclusions are required.

### Version 1.0 thresholds are heuristic

Score weights and decision thresholds are explicit engineering choices for the first implementation.

They should be validated and calibrated against representative datasets before this tool is adopted as a formal production or regulated decision standard.

### Automatic type inference has limits

Semantic meaning cannot always be reconstructed from values alone. Use type overrides when domain knowledge is available.

### Vectors and matrices are summarized

Version 1.0 reduces structured values to descriptive statistics. This can discard ordering, waveform shape, temporal dynamics, and spatial structure.

### Current stability is random-split stability

It is not yet a dedicated time, batch, machine, shift, or product-family stability analysis.

### Correlated predictors can share importance

Permutation importance may underestimate a useful predictor when another highly correlated predictor preserves essentially the same information.

The redundancy analysis helps expose this issue but does not completely solve it.

### Information leakage requires human attention

A variable measured after the target outcome can appear to be an excellent predictor while being useless for prospective prediction.

The software cannot always determine process chronology from the dataset alone.

---

## Recommended validation before production use

The project should be regression-tested against controlled datasets with known behavior.

Recommended scenarios include:

- strong linear dependence;
- monotonic non-linear dependence;
- U-shaped dependence with low Pearson correlation;
- an almost duplicated predictor;
- a purely random predictor;
- a variable useful primarily through interaction;
- severe missingness;
- binary and highly imbalanced targets;
- multiclass targets;
- continuous targets;
- high-cardinality categorical variables;
- deliberate information leakage;
- process drift across batches or time.

Expected outcomes should be specified before changing the scoring or decision engine.

---

## Documentation

Two complementary manuals accompany this project:

### Practical manual

`Variable_Analyzer_Practical_Manual_EN.md`

Written for readers who do not need a statistical background. It explains what the analyzer does, why several methods are combined, and how to interpret the five diagnostic states.

### Technical manual

`Variable_Analyzer_Technical_Manual_EN.md`

Documents the statistical methods, model architecture, normalizations, score definitions, decision rules, assumptions, limitations, and recommended future improvements.

---

## Roadmap

Potential future improvements include:

- grouped cross-validation;
- time-aware stability analysis;
- bootstrap uncertainty intervals;
- False Discovery Rate control;
- conditional permutation importance;
- richer vector and waveform feature extraction;
- additional model families;
- SHAP-based model interpretation;
- information-leakage warnings;
- rare-category handling;
- externally configurable score weights and thresholds;
- domain-specific calibration profiles;
- formal model performance gates before accepting permutation evidence;
- automated regression tests using synthetic benchmark datasets.

---

## Design principle

The project is built around one central rule:

> **Do not decide whether a variable is useful from a single statistic.**

Instead, the system combines independent evidence channels, keeps contradictory evidence visible, separates relevance from confidence and data quality, and uses conservative decision rules when the available data do not justify a stronger conclusion.

Conceptually:

```text
Diagnosis(X, Y)
    = association evidence
    + non-linear dependence
    + multivariable contribution
    + redundancy
    + interaction evidence
    + random-reference evidence
    + stability
    + data-quality controls
```

The objective is not merely to calculate correlation. The objective is to create an **automatic, explainable variable-diagnosis and selection system**.

---

## Disclaimer

This project is an analytical decision-support tool. Its output should be interpreted together with domain expertise and appropriate validation.

A diagnostic category such as `POSSIBLE NOISE`, `USE`, or `REDUNDANT` describes the evidence available in the analyzed dataset and under the implemented methodology. It is not a universal statement about the physical or causal role of the variable.
