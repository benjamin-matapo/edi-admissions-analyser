"""Explore synthetic admissions outcomes across anonymised EDI groups."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats


DATA_DIR = Path(__file__).resolve().parent / "data"
APPLICANTS_FILE = DATA_DIR / "applicants_edi.csv"
SCORES_FILE = DATA_DIR / "competency_scores.csv"

COHORTS = ["2024 Pre-Reform", "2026 Post-Structured Rubric"]
EDI_COLUMNS = {
    "Gender identity": "Gender_Identity",
    "Ethnicity": "Ethnicity",
    "Disability": "Disability",
    "First-generation status": "First_Generation",
}
SCORE_COLUMNS = [
    "Research_Potential",
    "Analytical_Rigour",
    "Interview",
]
OUTCOME_COLUMNS = ["Shortlisted", "Offer"]


def generate_synthetic_datasets(seed: int | None = None) -> None:
    """Create and save two linked, fictional datasets with 300 applicants."""
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cohort = np.repeat(COHORTS, 150)
    rng.shuffle(cohort)
    applicants = pd.DataFrame(
        {
            "Applicant_ID": [f"A{i:04d}" for i in range(1, 301)],
            "Cohort": cohort,
            "Gender_Identity": rng.choice(
                ["Woman", "Man", "Non-binary / another identity", "Prefer not to say"],
                size=300,
                p=[0.48, 0.43, 0.04, 0.05],
            ),
            "Ethnicity": rng.choice(
                ["White", "Asian", "Black", "Mixed / multiple", "Other", "Prefer not to say"],
                size=300,
                p=[0.52, 0.20, 0.10, 0.08, 0.05, 0.05],
            ),
            "Disability": rng.choice(
                ["Disabled", "Not disabled", "Prefer not to say"],
                size=300,
                p=[0.12, 0.80, 0.08],
            ),
            "First_Generation": rng.choice(
                ["First-generation", "Continuing-generation", "Prefer not to say"],
                size=300,
                p=[0.34, 0.59, 0.07],
            ),
        }
    )

    post_reform = applicants["Cohort"].eq(COHORTS[1]).to_numpy()
    latent_potential = rng.normal(loc=0.0, scale=1.0, size=300)
    score_noise = rng.normal(
        loc=0.0,
        scale=np.where(post_reform, 0.72, 0.95)[:, None],
        size=(300, len(SCORE_COLUMNS)),
    )
    score_offsets = np.array([3.35, 3.25, 3.20])
    score_values = np.clip(
        np.rint(score_offsets + latent_potential[:, None] * 0.72 + score_noise),
        1,
        5,
    ).astype(int)
    scores = pd.DataFrame(score_values, columns=SCORE_COLUMNS)
    scores.insert(0, "Applicant_ID", applicants["Applicant_ID"])

    average_score = score_values.mean(axis=1)
    shortlist_probability = 1 / (1 + np.exp(-1.35 * (average_score - 3.35)))
    shortlist_probability += np.where(post_reform, 0.015, 0.0)
    shortlisted = rng.binomial(1, np.clip(shortlist_probability, 0.05, 0.95))
    offer_probability = 1 / (1 + np.exp(-1.25 * (average_score - 3.65)))
    offer_probability = np.where(shortlisted == 1, offer_probability, 0.0)
    offered = rng.binomial(1, np.clip(offer_probability, 0.0, 0.95))
    scores["Shortlisted"] = shortlisted
    scores["Offer"] = offered

    applicants.to_csv(APPLICANTS_FILE, index=False)
    scores.to_csv(SCORES_FILE, index=False)


def load_merged_data() -> pd.DataFrame:
    """Load both generated files and validate their one-to-one relationship."""
    applicants = pd.read_csv(APPLICANTS_FILE)
    scores = pd.read_csv(SCORES_FILE)
    merged = applicants.merge(
        scores,
        on="Applicant_ID",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 300:
        raise ValueError(f"Expected 300 merged applicants; found {len(merged)}.")
    return merged


def subgroup_rates(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate outcome counts and rates for each EDI group and cohort."""
    records: list[dict[str, object]] = []
    for attribute, column in EDI_COLUMNS.items():
        grouped = data.groupby([column, "Cohort"], observed=True, dropna=False)
        for (group, cohort), subset in grouped:
            n = len(subset)
            for outcome in OUTCOME_COLUMNS:
                successes = int(subset[outcome].sum())
                records.append(
                    {
                        "Attribute": attribute,
                        "Group": group,
                        "Cohort": cohort,
                        "Outcome": outcome,
                        "Applicants": n,
                        "Count": successes,
                        "Rate": successes / n if n else np.nan,
                    }
                )
    return pd.DataFrame.from_records(records)


def _plain_english(test_name: str, p_value: float) -> str:
    if p_value < 0.05:
        if test_name == "Chi-square":
            return (
                "The data show a statistically significant association. "
                "This does not show that the rubric caused the difference."
            )
        return (
            "The score distributions differ statistically between cohorts. "
            "This does not show that the rubric caused the difference."
        )
    return (
        "No statistically significant difference was detected. "
        "This is not evidence that the cohorts are equivalent."
    )


def statistical_tests(data: pd.DataFrame) -> pd.DataFrame:
    """Compare cohorts using chi-square tests and two-sided Mann-Whitney U tests."""
    records: list[dict[str, object]] = []

    def add_chi_square(
        label: str,
        subset: pd.DataFrame,
        outcome: str,
        group_description: str,
    ) -> None:
        table = pd.crosstab(subset["Cohort"], subset[outcome]).reindex(
            index=COHORTS, columns=[0, 1], fill_value=0
        )
        if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
            records.append(
                {
                    "Comparison": label,
                    "Test": "Chi-square",
                    "Statistic": np.nan,
                    "P-value": np.nan,
                    "Interpretation": (
                        f"Not calculated: there are too few observations to compare "
                        f"{group_description} for {outcome.lower()}."
                    ),
                    "Note": "Both cohorts and both outcome values are required.",
                }
            )
            return

        result = stats.chi2_contingency(table.to_numpy(), correction=False)
        low_expected = bool((result.expected_freq < 5).any())
        records.append(
            {
                "Comparison": label,
                "Test": "Chi-square",
                "Statistic": float(result.statistic),
                "P-value": float(result.pvalue),
                "Interpretation": _plain_english("Chi-square", float(result.pvalue)),
                "Note": (
                    "Some expected cell counts are below 5; interpret cautiously."
                    if low_expected
                    else "Expected cell counts are all at least 5."
                ),
            }
        )

    for outcome in OUTCOME_COLUMNS:
        add_chi_square(f"Overall: cohort vs. {outcome.lower()}", data, outcome, "all applicants")
        for attribute, column in EDI_COLUMNS.items():
            for group in data[column].dropna().unique():
                subset = data.loc[data[column].eq(group)]
                label = f"{attribute} = {group}: cohort vs. {outcome.lower()}"
                add_chi_square(label, subset, outcome, repr(group))

    for score in SCORE_COLUMNS:
        before = data.loc[data["Cohort"].eq(COHORTS[0]), score]
        after = data.loc[data["Cohort"].eq(COHORTS[1]), score]
        result = stats.mannwhitneyu(before, after, alternative="two-sided")
        records.append(
            {
                "Comparison": f"{score.replace('_', ' ')}: pre vs. post",
                "Test": "Mann-Whitney U",
                "Statistic": float(result.statistic),
                "P-value": float(result.pvalue),
                "Interpretation": _plain_english(
                    "Mann-Whitney U", float(result.pvalue)
                ),
                "Note": (
                    "Two-sided test; compares score distributions, not just their means."
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def make_report(rates: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """Combine subgroup rates and statistical tests into one downloadable CSV."""
    rate_report = rates.rename(
        columns={
            "Attribute": "Comparison",
            "Group": "Group",
            "Outcome": "Metric",
            "Applicants": "N",
            "Count": "Count",
            "Rate": "Rate",
        }
    )
    rate_report["Section"] = "Subgroup outcome rate"
    rate_report["Test"] = ""
    rate_report["Statistic"] = np.nan
    rate_report["P-value"] = np.nan
    rate_report["Interpretation"] = ""
    rate_report["Note"] = ""

    test_report = tests.copy()
    test_report["Section"] = "Statistical test"
    test_report["Metric"] = ""
    test_report["Group"] = ""
    test_report["Cohort"] = ""
    test_report["N"] = np.nan
    test_report["Count"] = np.nan
    test_report["Rate"] = np.nan

    columns = [
        "Section",
        "Comparison",
        "Group",
        "Cohort",
        "Metric",
        "N",
        "Count",
        "Rate",
        "Test",
        "Statistic",
        "P-value",
        "Interpretation",
        "Note",
    ]
    return pd.concat(
        [rate_report[columns], test_report[columns]],
        ignore_index=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="EDI Admissions & Recruitment Equity Analyser",
        page_icon="📊",
        layout="wide",
    )
    st.title("EDI Admissions & Recruitment Equity Analyser")
    st.caption(
        "Explore a fictional admissions dataset. Synthetic results are for "
        "demonstration only and must not be used to assess real people or policies."
    )

    if not APPLICANTS_FILE.exists() or not SCORES_FILE.exists():
        generate_synthetic_datasets()

    with st.sidebar:
        st.header("Synthetic data")
        st.write("The app analyses 300 fictional applicants across two cohorts.")
        if st.button("Regenerate datasets", width="stretch"):
            generate_synthetic_datasets()
            st.success("Both synthetic CSV datasets have been regenerated.")
        st.caption("CSV files are saved in the local `data/` folder.")

    data = load_merged_data()
    rates = subgroup_rates(data)
    tests = statistical_tests(data)

    overall_shortlist = data.groupby("Cohort")["Shortlisted"].mean()
    overall_offer = data.groupby("Cohort")["Offer"].mean()
    metrics = st.columns(4)
    metrics[0].metric("Applicants", f"{len(data):,}")
    metrics[1].metric("Cohorts", str(data["Cohort"].nunique()))
    metrics[2].metric(
        "Pre-reform shortlisted",
        f"{overall_shortlist.get(COHORTS[0], 0):.1%}",
    )
    metrics[3].metric(
        "Post-rubric offers",
        f"{overall_offer.get(COHORTS[1], 0):.1%}",
    )

    st.subheader("Subgroup outcome rates")
    attribute_label = st.selectbox("EDI attribute", list(EDI_COLUMNS))
    outcome = st.radio("Outcome", OUTCOME_COLUMNS, horizontal=True)
    selected_rates = rates.loc[
        rates["Attribute"].eq(attribute_label) & rates["Outcome"].eq(outcome)
    ].copy()
    selected_rates["Rate (%)"] = selected_rates["Rate"] * 100
    figure = px.bar(
        selected_rates,
        x="Group",
        y="Rate (%)",
        color="Cohort",
        barmode="group",
        custom_data=["Applicants", "Count"],
        category_orders={"Cohort": COHORTS},
        labels={"Group": attribute_label, "Rate (%)": f"{outcome} rate"},
        title=f"{outcome} rate by {attribute_label.lower()}",
    )
    figure.update_traces(
        hovertemplate=(
            "%{x}<br>%{fullData.name}: %{y:.1f}%"
            "<br>Applicants: %{customdata[0]}"
            "<br>Outcome count: %{customdata[1]}<extra></extra>"
        )
    )
    figure.update_layout(yaxis_range=[0, 100], legend_title_text="Cohort")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Rates use all applicants in each group as the denominator. "
        "Offer rates are not conditional on being shortlisted."
    )
    st.dataframe(
        selected_rates.drop(columns="Rate").rename(
            columns={"Rate (%)": "Rate (%)"}
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Competency score distributions")
    selected_score = st.selectbox(
        "Rubric score", SCORE_COLUMNS, format_func=lambda value: value.replace("_", " ")
    )
    score_figure = px.histogram(
        data,
        x=selected_score,
        color="Cohort",
        barmode="group",
        category_orders={"Cohort": COHORTS},
        labels={selected_score: "Score (1-5)"},
        title=f"{selected_score.replace('_', ' ')} by cohort",
    )
    st.plotly_chart(score_figure, width="stretch")

    st.subheader("Statistical comparisons")
    st.write(
        "Chi-square tests compare cohort with shortlist/offer outcomes overall "
        "and within each EDI category. Mann-Whitney U tests compare each score "
        "distribution between cohorts. P-values are two-sided and use a 0.05 "
        "reference threshold; results are exploratory and unadjusted for "
        "multiple comparisons."
    )
    display_tests = tests.copy()
    display_tests["P-value"] = display_tests["P-value"].map(
        lambda value: f"{value:.4f}" if pd.notna(value) else "Not calculated"
    )
    display_tests["Statistic"] = display_tests["Statistic"].map(
        lambda value: f"{value:.3f}" if pd.notna(value) else "—"
    )
    st.dataframe(display_tests, hide_index=True, width="stretch")

    report_csv = make_report(rates, tests).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV report",
        data=report_csv,
        file_name="edi_equity_analysis_report.csv",
        mime="text/csv",
        width="stretch",
    )


if __name__ == "__main__":
    main()
