"""Interactive routing and computed benchmark exploration via the real authenticated API."""

import json
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RouteLab", page_icon="🔀", layout="wide")
st.title("RouteLab · Quality, cost and latency")
st.caption("Learned prompt-only routing · SIMULATED demos and RECORDED benchmarks are separate")
base_url = os.getenv("ROUTELAB_API_URL", "http://127.0.0.1:8001")
token = st.sidebar.text_input(
    "Local API token", value=os.getenv("ROUTELAB_API_TOKEN", ""), type="password"
)


def api(method: str, path: str, body=None):
    try:
        response = httpx.request(
            method,
            base_url + path,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        st.error("API request failed. Check server, token, training prerequisites and server logs.")
        return None


if not token:
    st.info("Start the API and enter its local token to load model predictions and benchmarks.")
    st.stop()

playground, benchmark, experiments, insights = st.tabs(
    [
        "Routing playground",
        "Benchmark",
        "Experiment details",
        "Model insights",
    ]
)
models = api("GET", "/api/v1/models") or []
with playground:
    with st.form("routing_form"):
        prompt = st.text_area("Prompt", "Explain the difference between TCP and UDP.")
        strategy = st.selectbox("Strategy", api("GET", "/api/v1/strategies") or ["ml"])
        allowed = st.multiselect(
            "Available candidates",
            [m["model_id"] for m in models],
            default=[m["model_id"] for m in models],
        )
        a, b, c = st.columns(3)
        cost_limit = a.number_input(
            "Maximum estimated cost USD", min_value=0.0, value=0.01, format="%.6f"
        )
        latency_limit = b.number_input("Maximum estimated latency ms", min_value=0.0, value=5000.0)
        quality_min = c.slider("Minimum predicted quality", 0.0, 1.0, 0.0)
        a, b, c = st.columns(3)
        quality_weight = a.number_input("Quality weight", min_value=0.0, max_value=100.0, value=1.0)
        cost_weight = b.number_input("Cost weight", min_value=0.0, max_value=100.0, value=0.15)
        latency_weight = c.number_input("Latency weight", min_value=0.0, max_value=100.0, value=0.1)
        submitted = st.form_submit_button("Route request", type="primary")
    if submitted:
        result = api(
            "POST",
            "/api/v1/route",
            {
                "prompt": prompt,
                "strategy": strategy,
                "candidate_models": allowed,
                "max_cost_usd": cost_limit,
                "max_latency_ms": latency_limit,
                "min_quality": quality_min,
                "quality_weight": quality_weight,
                "cost_weight": cost_weight,
                "latency_weight": latency_weight,
            },
        )
        if result:
            st.session_state["route"] = result
    result = st.session_state.get("route")
    if result:
        st.info(f"Prediction origin: {result['prediction_metadata']['origin']}")
        st.write(result["explanation"])
        if result["selected"]:
            chosen = result["selected"]
            a, b, c, d = st.columns(4)
            a.metric("Selected model", chosen["model_id"])
            b.metric("Predicted quality", f"{chosen['predicted_quality']:.3f}")
            c.metric("Estimated USD", f"{chosen['estimated_cost_usd']:.6f}")
            d.metric("Estimated latency ms", f"{chosen['estimated_latency_ms']:.0f}")
        else:
            st.warning("No feasible model. Hard constraints were kept.")
        predicted = pd.DataFrame(result["candidates"])
        st.dataframe(predicted, hide_index=True)
        st.scatter_chart(predicted, x="estimated_cost_usd", y="predicted_quality", color="model_id")
        st.caption("Pre-inference estimates can differ from final quality, latency and billing.")
        st.json(result["prediction_metadata"])

summaries = api("GET", "/api/v1/experiments/summary") or {}
with benchmark:
    dataset = st.selectbox("Benchmark dataset", list(summaries) or ["No computed reports"])
    summary = summaries.get(dataset)
    folder = Path("reports") / ("synthetic" if dataset == "demo" else "real")
    if summary:
        st.info(
            f"{summary['origin']} · {summary['complete_test_prompts']} complete held-out prompts"
        )
        st.caption(summary["quality_aggregation"])
        comparison = pd.DataFrame(summary["comparison"])
        st.dataframe(
            comparison.drop(columns=["model_distribution"], errors="ignore"), hide_index=True
        )
        if "macro_task_quality" in comparison:
            st.scatter_chart(
                comparison, x="mean_reference_cost_usd", y="macro_task_quality", color="strategy"
            )
        for name in ("cost_quality_frontier.png", "latency_comparison.png", "model_selection.png"):
            if (folder / name).is_file():
                st.image(str(folder / name))
        task_path = folder / "task_performance.csv"
        if task_path.is_file():
            tasks = pd.read_csv(task_path)
            task = st.selectbox("Task filter", ["All", *sorted(tasks.task_type.unique())])
            st.dataframe(
                tasks if task == "All" else tasks.loc[tasks.task_type == task], hide_index=True
            )
        model_path = folder / "observed_model_per_task.csv"
        if model_path.is_file():
            observations = pd.read_csv(model_path)
            selected_model = st.selectbox(
                "Observed model filter", ["All", *sorted(observations.model_id.unique())]
            )
            st.dataframe(
                observations
                if selected_model == "All"
                else observations.loc[observations.model_id == selected_model],
                hide_index=True,
            )
        st.warning(summary["limitations"])
    else:
        st.info("Run train, evaluate and report to generate benchmark visualizations.")
with experiments:
    if summary:
        st.json({key: value for key, value in summary.items() if key != "comparison"})
        metadata_path = Path("artifacts") / dataset / "metadata.json"
        if metadata_path.is_file():
            st.json(json.loads(metadata_path.read_text("utf-8")))
        overview = folder / "dataset_summary.json"
        if overview.is_file():
            st.json(json.loads(overview.read_text("utf-8")))
        st.write(
            "Original test is held out. Duplicate families stay together. "
            "Validation chooses alpha and the strong reference; transforms fit on training. "
            "All strategy metrics use the same successful, complete test cohort."
        )
with insights:
    if summary:
        st.metric("Held-out quality predictor MAE", f"{summary['predictor_quality_mae']:.4f}")
        error_path = folder / "predictor_errors.csv"
        if error_path.is_file():
            errors = pd.read_csv(error_path)
            st.dataframe(
                errors.groupby("model_id").absolute_error.mean().sort_values(),
                column_config={"absolute_error": "Mean quality absolute error"},
            )
        decisions_path = folder / "routing_decisions.csv"
        if decisions_path.is_file():
            decisions = pd.read_csv(decisions_path)
            st.dataframe(decisions.head(50), hide_index=True)
        source = (
            Path(os.getenv("ROUTELAB_DATA_DIR", "data/runtime")) / dataset / "observations.jsonl"
        )
        if source.is_file():
            with source.open(encoding="utf-8") as stream:
                record = json.loads(next(stream))
            with st.expander("Example dataset prompt (untrusted text)"):
                st.text(record["prompt_text"])
        st.caption("Per-observation files are generated locally by evaluate and excluded from Git.")
