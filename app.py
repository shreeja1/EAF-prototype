from pathlib import Path
import math

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "construction_cost_model.pkl"
FEATURE_ORDER = ["Budget", "Duration", "Employees", "Material_Cost", "Labor_Cost"]


st.set_page_config(
    page_title="AI Engineering Project Cost and Risk Advisor",
    page_icon="🏗",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: #f4f7fb; color: #14243a; }
      .block-container { max-width: 1180px; padding-top: 3.4rem; padding-bottom: 3rem; }
      h1, h2, h3 { color: #14243a; letter-spacing: -0.025em; }
      .eyebrow {
        color: #167d91; font-size: .76rem; font-weight: 800;
        letter-spacing: .13em; text-transform: uppercase; margin-bottom: .45rem;
      }
      .intro { color: #52647a; font-size: 1.04rem; max-width: 790px; }
      .notice {
        background: #fff8e8; border: 1px solid #f0d89e; border-left: 4px solid #d49a27;
        border-radius: 8px; padding: 14px 16px; color: #5d4a20; margin: 1rem 0 1.4rem;
      }
      .section-label {
        font-size: .78rem; font-weight: 800; letter-spacing: .1em;
        color: #000000; text-transform: uppercase; margin: .2rem 0 .7rem;
      }
      [data-testid="stWidgetLabel"],
      [data-testid="stWidgetLabel"] * {
        color: #000000 !important; opacity: 1 !important;
        font-size: 1rem !important; font-weight: 700 !important;
      }
      div[data-testid="stMetric"] {
        background: #fff; border: 1px solid #e3eaf2; border-radius: 12px;
        padding: 18px 20px; box-shadow: 0 2px 8px rgba(22, 42, 67, .035);
      }
      div[data-testid="stMetricLabel"] { color: #63748a; }
      div[data-testid="stMetricValue"] { color: #14243a; }
      button[kind="primary"], div[data-testid="stFormSubmitButton"] button {
        background: #116b7c; border: 1px solid #116b7c; border-radius: 8px;
        min-height: 46px; font-weight: 700; color: #fff !important;
      }
      button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] button:hover {
        background: #0b5868 !important; border-color: #0b5868 !important;
      }
      .risk-card {
        border-radius: 11px; padding: 16px 18px; border: 1px solid;
        margin: .2rem 0 1rem;
      }
      .risk-low { background: #eaf6ef; border-color: #b8dec5; color: #24613a; }
      .risk-medium { background: #fff5df; border-color: #eed397; color: #77530d; }
      .risk-high { background: #fff0ed; border-color: #edc0b8; color: #873b30; }
      .risk-title { font-size: 1.08rem; font-weight: 800; }
      .risk-copy { font-size: .91rem; margin-top: 4px; }
      .footnote { color: #69798e; font-size: .86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_usd(amount: float) -> str:
    return f"${amount:,.2f}"


def classify_risk(variance_percent: float) -> tuple[str, str, str]:
    """Apply illustrative decision thresholds to the model's budget variance."""
    if variance_percent <= 5:
        return (
            "Low",
            "risk-low",
            "The estimated overrun is within the illustrative 5% threshold.",
        )
    if variance_percent <= 15:
        return (
            "Medium",
            "risk-medium",
            "The estimated overrun is above 5% and within the illustrative 15% threshold.",
        )
    return (
        "High",
        "risk-high",
        "The estimated overrun is above the illustrative 15% threshold.",
    )


def get_recommendations(risk_level: str) -> list[str]:
    if risk_level == "Low":
        return [
            "Keep the current cost plan and review committed versus actual costs at least monthly.",
            "Confirm material lead times and supplier pricing before major purchase orders.",
            "Track contingency use and refresh the forecast when scope or schedule changes.",
        ]
    if risk_level == "Medium":
        return [
            "Update the cost-to-complete forecast weekly and investigate the largest cost drivers.",
            "Reconfirm subcontractor quotes, material pricing, and labor availability.",
            "Set an approved contingency reserve and require sign-off for forecast changes.",
        ]
    return [
        "Escalate the estimate for a project-controls review and prepare a documented recovery plan.",
        "Recheck scope, quantities, contract terms, procurement pricing, and schedule assumptions.",
        "Review cash flow and contingency weekly; obtain approval before discretionary commitments.",
    ]


def predict_final_cost(inputs: dict[str, float]) -> float:
    """Load the supplied model and predict from one row in the required feature order."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Model file not found. Upload construction_cost_model.pkl to the project directory."
        )

    model = joblib.load(MODEL_PATH)
    input_frame = pd.DataFrame(
        [[inputs[name] for name in FEATURE_ORDER]],
        columns=FEATURE_ORDER,
    )
    prediction = model.predict(input_frame)
    if len(prediction) != 1:
        raise ValueError("The model did not return exactly one project cost.")

    predicted_cost = float(prediction[0])
    if not math.isfinite(predicted_cost):
        raise ValueError("The model returned a non-finite project cost.")
    return predicted_cost


st.markdown(
    '<div class="eyebrow">Project controls • Cost outlook</div>',
    unsafe_allow_html=True,
)
st.title("AI Engineering Project Cost and Risk Advisor")
st.markdown(
    '<p class="intro">Estimate a project’s final cost with your trained model, compare it with '
    "the approved budget, and review practical cost-control actions.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="notice"><strong>Important:</strong> This model was trained on simulated '
    "construction project data and has not been validated for real-world budgeting. "
    "Risk levels below use illustrative decision thresholds; they are not machine-learning "
    "predictions. Use this tool as a planning aid, not as a substitute for professional "
    "cost review.</div>",
    unsafe_allow_html=True,
)

if MODEL_PATH.is_file():
    st.sidebar.success("Model file found")
else:
    st.sidebar.warning(
        "Model file missing. Upload construction_cost_model.pkl to the project directory."
    )
st.sidebar.caption(
    "Only use a model file from a trusted source. Loading joblib/pickle files can execute code."
)

st.markdown(
    '<div class="section-label">Project inputs</div>',
    unsafe_allow_html=True,
)
with st.form("project_inputs", border=True):
    first_row = st.columns(2)
    with first_row[0]:
        budget = st.number_input(
            "Budget (USD)", min_value=0.0, value=0.0, step=1000.0, format="%.2f"
        )
    with first_row[1]:
        duration = st.number_input(
            "Duration (months)", min_value=0.0, value=0.0, step=1.0, format="%.2f"
        )

    second_row = st.columns(2)
    with second_row[0]:
        employees = st.number_input(
            "Employees", min_value=0, value=0, step=1
        )
    with second_row[1]:
        material_cost = st.number_input(
            "Material_Cost (USD)", min_value=0.0, value=0.0, step=1000.0, format="%.2f"
        )

    labor_cost = st.number_input(
        "Labor_Cost (USD)", min_value=0.0, value=0.0, step=1000.0, format="%.2f"
    )
    st.caption("Enter project values. Budget must be greater than zero; costs cannot be negative.")
    analyze_clicked = st.form_submit_button(
        "Analyze Project", type="primary", width="stretch"
    )

if analyze_clicked:
    st.session_state.pop("analysis_result", None)
    current_inputs = {
        "Budget": float(budget),
        "Duration": float(duration),
        "Employees": float(employees),
        "Material_Cost": float(material_cost),
        "Labor_Cost": float(labor_cost),
    }
    validation_errors = []
    if current_inputs["Budget"] <= 0:
        validation_errors.append("Budget must be greater than zero.")
    if current_inputs["Material_Cost"] < 0 or current_inputs["Labor_Cost"] < 0:
        validation_errors.append("Material and labor costs must be nonnegative.")

    if validation_errors:
        for validation_error in validation_errors:
            st.error(validation_error)
    elif not MODEL_PATH.is_file():
        st.error(
            "Model file not found. Upload construction_cost_model.pkl to the project directory, "
            "then analyze the project again."
        )
    else:
        try:
            predicted_cost = predict_final_cost(current_inputs)
            variance = predicted_cost - current_inputs["Budget"]
            variance_percent = variance / current_inputs["Budget"] * 100
            st.session_state["analysis_result"] = {
                "inputs": current_inputs,
                "predicted_cost": predicted_cost,
                "variance": variance,
                "variance_percent": variance_percent,
            }
        except Exception as error:
            st.error(
                "The model could not be loaded or did not produce a valid estimate. "
                "Confirm the file is a trusted, compatible scikit-learn model."
            )
            with st.expander("Technical details"):
                st.code(f"{type(error).__name__}: {error}")


result = st.session_state.get("analysis_result")
if result:
    analyzed_inputs = result["inputs"]
    predicted_cost = result["predicted_cost"]
    variance = result["variance"]
    variance_percent = result["variance_percent"]
    risk_level, risk_class, risk_copy = classify_risk(variance_percent)

    st.divider()
    st.markdown(
        '<div class="section-label">Analysis results</div>',
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Original budget", format_usd(analyzed_inputs["Budget"]))
    metric_columns[1].metric("Predicted final cost", format_usd(predicted_cost))
    metric_columns[2].metric(
        "Estimated budget variance",
        format_usd(variance),
        delta=f"{variance_percent:+.2f}%",
        delta_color="inverse",
    )

    st.markdown(
        f'<div class="risk-card {risk_class}"><div class="risk-title">'
        f"{risk_level} risk indicator</div><div class=\"risk-copy\">{risk_copy} "
        "This classification is an illustrative decision threshold, not a machine-learning "
        "prediction.</div></div>",
        unsafe_allow_html=True,
    )

    chart_column, summary_column = st.columns([1.2, 0.8], gap="large")
    with chart_column:
        st.subheader("Budget comparison")
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        labels = ["Original budget", "Predicted final cost"]
        values = [analyzed_inputs["Budget"], predicted_cost]
        bars = axis.bar(labels, values, color=["#7f91a8", "#147d8d"], width=0.58)
        axis.set_ylabel("Cost (USD)")
        axis.set_ylim(bottom=min(0, min(values) * 1.12))
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#d5dee8")
        axis.tick_params(axis="x", rotation=0, colors="#52647a")
        axis.tick_params(axis="y", colors="#52647a")
        axis.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"${value:,.0f}")
        )
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.annotate(
                format_usd(value),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 7 if value >= 0 else -17),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
                color="#263a52",
            )
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    with summary_column:
        st.subheader("Project input summary")
        summary = pd.DataFrame(
            [
                ("Budget", format_usd(analyzed_inputs["Budget"])),
                ("Duration", f'{analyzed_inputs["Duration"]:,.2f} months'),
                ("Employees", f'{int(analyzed_inputs["Employees"]):,}'),
                ("Material cost", format_usd(analyzed_inputs["Material_Cost"])),
                ("Labor cost", format_usd(analyzed_inputs["Labor_Cost"])),
            ],
            columns=["Input", "Value"],
        )
        st.dataframe(summary, hide_index=True, width="stretch")
        st.caption(
            f"Variance = predicted cost − original budget: {format_usd(variance)} "
            f"({variance_percent:+.2f}%)."
        )

    st.subheader("Risk-management recommendations")
    for recommendation in get_recommendations(risk_level):
        st.markdown(f"- {recommendation}")

st.divider()
st.markdown(
    '<p class="footnote">Model limitation: trained on simulated construction project data; '
    "not validated for real-world budgeting. Independently review estimates, assumptions, and "
    "project-specific risks before making financial decisions.</p>",
    unsafe_allow_html=True,
)