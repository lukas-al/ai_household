import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import os
    import dspy
    import mlflow
    import getpass
    import pandas as pd
    from pydantic import BaseModel

    return BaseModel, dspy, getpass, mlflow, mo, os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # DSPy MPC Experiments (Marimo)

    Configure models and household counts below, then run the experiments. Each configuration
    is logged as a separate MLflow run under a single experiment name.
    """
    )
    return


@app.cell
def _(getpass, mo, os):
    def get_tracking_uri(default: str = "http://localhost:5001") -> str:
        uri = os.getenv("MLFLOW_TRACKING_URI", default)
        return uri

    def require_openrouter_api_key() -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            api_key = getpass.getpass("Input API key")
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it before running: export OPENROUTER_API_KEY=..."
            )
        return api_key

    mo.md("Tracking helpers ready.")
    return get_tracking_uri, require_openrouter_api_key


@app.cell
def _(dspy):
    def configure_dspy_lm(
        model_name: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> None:
        llm = dspy.LM(
            model=model_name,
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            max_tokens=max_tokens,
            temperature=temperature,
            cache=False,
        )
        dspy.configure(lm=llm, adapter=dspy.ChatAdapter())

    return (configure_dspy_lm,)


@app.cell
def _(BaseModel, dspy):
    class SpendingDecision(BaseModel):
        amount_spent: float

    class MPCSignature(dspy.Signature):
        """
        Decide how to allocate a one-time windfall between spending and saving.
        """

        windfall_amount: float = dspy.InputField(
            desc="Unexpected one-time windfall amount in GBP (£)"
        )

        decision: SpendingDecision = dspy.OutputField(
            desc="Household allocation decision with total amount spent"
        )

    return (MPCSignature,)


@app.cell
def _(
    MPCSignature,
    configure_dspy_lm,
    dspy,
    get_tracking_uri,
    mlflow,
    mo,
    pd,
    require_openrouter_api_key,
):
    from datetime import datetime

    def run_single_mpc(
        model_name: str,
        n_households: int,
        windfall_amount: float,
        experiment_name: str,
        temperature: float,
        max_tokens: int,
        prompting_technique: str = "zero_shot",
    ) -> pd.DataFrame:
        """
        Run one MPC experiment configuration and log to MLflow.
        Returns the per-household results dataframe.
        """

        # Configure tracking and experiment
        mlflow.set_tracking_uri(get_tracking_uri())
        mlflow.set_experiment(experiment_name)

        # Ensure DSPy autologging is enabled (idempotent)
        mlflow.dspy.autolog()

        api_key = require_openrouter_api_key()
        configure_dspy_lm(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        predictor = dspy.Predict(MPCSignature)

        run_name = f"model={model_name}__n={n_households}__{prompting_technique}"
        with mlflow.start_run(run_name=run_name):
            # Params and tags first
            mlflow.log_param("windfall_amount", windfall_amount)
            mlflow.log_param("num_households", n_households)
            mlflow.log_param("model", model_name)
            mlflow.set_tag("prompting_technique", prompting_technique)

            results = []
            successful = 0

            for i in mo.status.progress_bar(range(n_households), title=f"Running model {model_name}"):
                try:
                    pred = predictor(windfall_amount=windfall_amount)
                    amount_spent = float(pred.decision.amount_spent)
                    results.append(
                        {
                            "household_id": i,
                            "household_type": "baseline",
                            "windfall_amount": windfall_amount,
                            "amount_spent": amount_spent,
                            "error": None,
                        }
                    )
                    successful += 1
                except Exception as e:  # noqa: BLE001
                    results.append(
                        {
                            "household_id": i,
                            "household_type": "baseline",
                            "windfall_amount": windfall_amount,
                            "amount_spent": None,
                            "error": str(e),
                        }
                    )

            results_df = pd.DataFrame(results)
            results_df["mpc"] = results_df["amount_spent"] / results_df["windfall_amount"]

            # Aggregate stats
            mpc_stats = results_df["mpc"].agg(["count", "mean", "std", "min", "max", "median"])

            # Log metrics
            mlflow.log_metric("successful_predictions", successful)
            for metric_name in mpc_stats.index:
                metric_value = mpc_stats[metric_name]
                if pd.notna(metric_value):
                    mlflow.log_metric(metric_name, float(metric_value))

            # Persist raw results as artifact
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            safe_model = model_name.replace("/", "-")
            file_name = f"data/results_{safe_model}_{n_households}_households_{timestamp}.csv"
            results_df.to_csv(file_name, index=False)
            mlflow.log_artifact(file_name, artifact_path="results_data")

        return results_df

    def run_matrix(
        models: list[str],
        household_sizes: list[int],
        windfall_amount: float,
        experiment_name: str,
        temperature: float,
        max_tokens: int,
    ) -> list[pd.DataFrame]:
        outputs: list[pd.DataFrame] = []
        for model in models:
            try: 
                for n in household_sizes:
                    print(
                        f"Running MPC test: model={model}, households={n}, windfall=£{windfall_amount}"  # noqa: T201
                    )
                    df = run_single_mpc(
                        model_name=model,
                        n_households=n,
                        windfall_amount=windfall_amount,
                        experiment_name=experiment_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    df = df.assign(model=model)
                    outputs.append(df)
            except ValueError as e:
                print(f"Model {model} failed for reason \n {e}")
                continue

        return outputs

    return (run_matrix,)


@app.cell
def _():
    MODELS: list[str] = [
        "openrouter/openai/gpt-oss-20b",
        "openrouter/google/gemini-flash-1.5",
        "openrouter/google/gemini-2.0-flash-001",
        "openrouter/google/gemini-2.5-flash-lite",
        "openrouter/meta-llama/llama-3-8b-instruct",
        "openrouter/meta-llama/llama-3.3-70b-instruct",
        "openrouter/moonshotai/kimi-k2:free",
        "openrouter/qwen/qwen-2.5-7b-instruct",
        # "openai/gpt-5-nano",
        # "openai/gpt-4o-mini",
    ]
    HOUSEHOLD_SIZES: list[int] = [150]
    WINDFALL_AMOUNT: float = 500.0
    EXPERIMENT_NAME: str = "DSPy MPC Experiments"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 500
    return (
        EXPERIMENT_NAME,
        HOUSEHOLD_SIZES,
        MAX_TOKENS,
        MODELS,
        TEMPERATURE,
        WINDFALL_AMOUNT,
    )


@app.cell
def _(
    EXPERIMENT_NAME: str,
    HOUSEHOLD_SIZES: list[int],
    MAX_TOKENS: int,
    MODELS: list[str],
    TEMPERATURE: float,
    WINDFALL_AMOUNT: float,
    mlflow,
    run_matrix,
):
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(EXPERIMENT_NAME)

    outputs = run_matrix(
        models=MODELS,
        household_sizes=HOUSEHOLD_SIZES,
        windfall_amount=WINDFALL_AMOUNT,
        experiment_name=EXPERIMENT_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return (outputs,)


@app.cell
def _(
    EXPERIMENT_NAME: str,
    HOUSEHOLD_SIZES: list[int],
    MODELS: list[str],
    WINDFALL_AMOUNT: float,
    mo,
    outputs,
    pd,
):
    all_df = pd.concat(outputs, ignore_index=True)
    summary = (
        all_df.assign(mpc=all_df["amount_spent"] / all_df["windfall_amount"]).groupby(
            ["model", "household_type"]
        )["mpc"].agg(["count", "mean", "std", "min", "max", "median"]).reset_index()
    )
    mo.vstack([
        mo.md(f"### Experiment: `{EXPERIMENT_NAME}`"),
        mo.md(
            f"Models: {', '.join(MODELS)} | Household sizes: {', '.join(map(str, HOUSEHOLD_SIZES))} | Windfall: £{WINDFALL_AMOUNT}"
        ),
        mo.md("#### Summary (by model, household type):"),
        summary,
        mo.md("#### Sample of last run (first 10 rows):"),
        outputs[-1].head(10),
    ])
    return


if __name__ == "__main__":
    app.run()
