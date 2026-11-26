# EDI Admissions & Recruitment Equity Analyser

A Streamlit demonstration app that generates and analyses two linked synthetic
CSV datasets for 300 fictional PhD applicants. It compares a **2024 Pre-Reform**
cohort with a **2026 Post-Structured Rubric** cohort using anonymised group
attributes, rubric scores, shortlist outcomes, and offers.

> **Important:** Every record and result produced by this app is synthetic.
> The demonstration does not represent any institution, actual applicant, or
> real-world effect of a structured rubric. Do not use it to make decisions
> about people or to draw conclusions about real recruitment practices.

## Run the app

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On first launch, the app creates a local `data/` directory containing:

- `applicants_edi.csv` — 300 applicant IDs, cohort labels, and synthetic EDI
  attributes.
- `competency_scores.csv` — three synthetic rubric scores and shortlist/offer
  outcomes, keyed by applicant ID.

The app merges the files one-to-one on `Applicant_ID`. Use **Regenerate
datasets** in the sidebar to replace both CSVs with a new synthetic sample.
The generated files are local outputs and are not needed to install the app.

## Analysis and statistical methodology

- **Subgroup rates:** For each EDI category and cohort, shortlist and offer
  counts are divided by the number of applicants in that group. The offer-rate
  denominator is all applicants, not only shortlisted applicants. The chart
  controls let you select an EDI attribute and outcome.
- **Chi-square tests of independence:** For shortlist and offer outcomes, a
  2-by-2 contingency table compares cohort (pre/post) with outcome
  (no/yes). Tests are shown for all applicants and separately within every
  category of each EDI attribute. The app reports the chi-square statistic and
  p-value. If a comparison lacks both cohorts or both outcome values, it is
  labelled as not calculated. Expected cell counts below five are flagged for
  caution.
- **Mann-Whitney U tests:** A two-sided test compares each of the three ordinal
  rubric-score distributions between cohorts. It tests for a distributional
  difference, not specifically a difference in means.
- **Interpretation:** A p-value below 0.05 is described as statistically
  significant, not as evidence of causation. A non-significant result does not
  establish equivalence. The tests are exploratory, and p-values are not
  adjusted for the many subgroup comparisons. Small subgroup counts and
  multiple testing make apparent differences especially uncertain.
- **CSV report:** The download combines subgroup counts/rates and statistical
  test results, including plain-English interpretations and cell-count notes.

The synthetic generator assigns anonymised categories and produces scores and
outcomes with random variation. Consequently, results can vary after
regeneration, and significance is not guaranteed. The post-reform score model
uses slightly lower random score noise for demonstration; this is a
pre-programmed property of the simulation, not an observed or validated
intervention effect.

## Data privacy safeguards

- The app creates fictional IDs (`A0001`–`A0300`) and does not request names,
  contact details, application text, or other direct identifiers.
- EDI data are generated randomly, stored only in local CSV files, and are not
  uploaded by the app. Do not place real or identifiable applicant data in the
  generated files.
- The app does not connect to a database or external analytics service. Normal
  Streamlit hosting and infrastructure may have their own logging and access
  controls; review those separately before deployment.
- Synthetic data are not a substitute for secure governance of real EDI data.
  Any real-world analysis should follow applicable privacy law, institutional
  policies, access controls, retention limits, and disclosure-risk review.

## Files

```text
app.py
requirements.txt
README.md
data/                     # created on first app launch
```
