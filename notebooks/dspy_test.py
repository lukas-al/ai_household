import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import os
    import getpass
    import dspy
    import pandas as pd
    import pydantic
    import seaborn as sns
    import mlflow
    return dspy, getpass, mlflow, mo, os, pd, pydantic, sns


@app.cell
def _(mo):
    mo.md(
        r"""
    <!-- 
    Start the tracking and experiment server:

    mlflow server \
        --backend-store-uri sqlite:///mlflow_data/mlflow.db \
        --default-artifact-root ./mlflow_data/artifacts \
        --host 0.0.0.0 \
        --port 5001

    -->
    """
    )
    return


@app.cell
def _(mlflow):
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("DSPy Testing")
    mlflow.dspy.autolog()
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
    # Consumption and Saving Responses to Idiosyncratic Shocks - the Marginal Propensity to Consume (DSPy Implementation)

    This notebook replicates the MPC experiment using the DSPy framework instead of the custom experiment runner.

    ## DSPy Approach

    Instead of custom prompt engineering and response parsing, we use:

    - **DSPy Signatures**: Declarative task specifications with typed inputs/outputs
    - **DSPy Predictors**: Automated prompt generation and response parsing
    - **Structured Output**: Type-safe responses without manual Pydantic models

    ## Empirical Benchmarks
    A large and robust body of empirical work has established that the average quarterly MPC for non-durable goods out of transitory windfalls of $500-$1,000 should be in the range of **15% to 25%**.
    """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 1. Setup and Configuration""")
    return


@app.cell
def _(dspy, getpass, os):
    # Configure OpenRouter API key
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    if not OPENROUTER_API_KEY:
        OPENROUTER_API_KEY = getpass.getpass(prompt="Enter your OpenRouter API Key: ")
        os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

    # Configure the LLM
    # model_name = "openai/gpt-4o-mini"
    model_name = "openai/gpt-oss-20b"

    llm = dspy.LM(
        model=model_name,
        api_key=OPENROUTER_API_KEY,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=500,
        temperature=0.7,
        cache=False,
    )

    # Configure DSPy to use this LLM
    dspy.configure(lm=llm, adapter=dspy.ChatAdapter())
    return (model_name,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Define the DSPy Signature

    A **Signature** in DSPy replaces our custom prompt engineering and Pydantic response models.
    It declaratively specifies the task inputs and expected outputs.
    """
    )
    return


@app.cell
def _(dspy, pydantic):
    class SpendingDecision(pydantic.BaseModel):
        amount_spent: float
        # reasoning: str

    class MPCSignature(dspy.Signature):
        """
        You are making financial decisions for a household that has just received an unexpected one-time windfall.
        Please decide how to allocate this windfall between spending and saving.
        """

        windfall_amount: float = dspy.InputField(
            desc="The unexpected one-time windfall amount in GBP (£)"
        )

        decision: SpendingDecision = dspy.OutputField(
            desc="Household allocation decision with total amount spent"
        )

        # reasoning: str = dspy.OutputField(
        #     desc="A very short (single sentence) reason as to why you made the decision you did"
        # )
    return (MPCSignature,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Create the Prediction Module

    We use `dspy.Predict` to replace our custom ExperimentRunner.
    This handles prompt generation, LLM interaction, and response parsing automatically.
    """
    )
    return


@app.cell
def _(MPCSignature, dspy):
    # Create the predictor module
    mpc_predictor = dspy.Predict(MPCSignature)
    return (mpc_predictor,)


@app.cell
def _(mo):
    mo.md(r"""## 4. Run Experiment and Analyze Results""")
    return


@app.cell
def _(mlflow, mo, model_name, mpc_predictor, pd, sns):
    with mlflow.start_run() as run:
        run_name = run.info.run_name

        # Experiment parameters (matching original)
        windfall_amount = 500.0
        n_households = 200

        # Run simulation
        results_list = []
        successful_predictions = 0

        print(f"Running MPC simulation for {n_households} households...")
        print(f"Model: {model_name}")
        print(f"Windfall amount: £{windfall_amount}")
        print(f"MLflow Run Name: {run_name}")
        print("-" * 50)

        for i in mo.status.progress_bar(range(n_households)):
            # Get prediction from DSPy
            prediction = mpc_predictor(windfall_amount=windfall_amount)

            try:
                # Extract outputs (validated by Pydantic via the Signature)    
                results_list.append(
                    {
                        "household_id": i,
                        "household_type": "baseline",
                        "windfall_amount": windfall_amount,
                        "amount_spent": float(prediction.decision.amount_spent),
                        # "reasoning": prediction.decision.reasoning,
                        "error": None,
                    }

                )
                successful_predictions += 1
            except Exception as e:
                results_list.append(
                    {
                        "household_id": i,
                        "household_type": "baseline",
                        "windfall_amount": windfall_amount,
                        "amount_spent": None,
                        # "reasoning": None,
                        "error": e,
                    }
                )
                print("Single call failed for reason:", e)
                continue


        print(f"Simulation complete! Successfully processed {successful_predictions}/{n_households} households")

        # Convert to DataFrame and calculate MPC
        results_df = pd.DataFrame(results_list)
        results_df["mpc"] = results_df["amount_spent"] / results_df["windfall_amount"]
        results_filename = f"data/results_{model_name.replace("/", "-")}_{n_households}_households_{run_name}.csv"
        print(results_filename)

        results_df.to_csv(results_filename, index=False)
        mpc_stats = results_df["mpc"].agg(["count", "mean", "std", "min", "max", "median"])

        # Mlflow
        mlflow.log_param("windfall amount", windfall_amount)
        mlflow.log_param("number of households", n_households)
        mlflow.log_param("model_type", model_name)
        description = "A very basic run with no context pass to the LLM single shot and a basic decision to make. A baseline result."
        mlflow.set_tag("mlflow.note.content", description)
        for val in mpc_stats.index:
            mlflow.log_metric(val, mpc_stats[val])

        mlflow.log_artifact(results_filename, artifact_path="results_data")

    # Display results
    mo.vstack(
        [
            mo.md(f"### Results for Model: `{model_name}`"),
            mo.md(f"**Windfall Amount:** £{windfall_amount:,.2f} | **Successful Predictions:** {successful_predictions}/{n_households}"),
            mo.md("#### MPC Statistics:"),
            mo.md(f"- **Mean MPC:** {mpc_stats['mean']:.3f}"),
            mo.md(f"- **Median MPC:** {mpc_stats['median']:.3f}"),
            mo.md(f"- **Std Dev:** {mpc_stats['std']:.3f}"),
            mo.md(f"- **Range:** [{mpc_stats['min']:.3f}, {mpc_stats['max']:.3f}]"),
            mo.md("#### MPC Distribution:"),
            sns.histplot(data=results_df, x="mpc", bins=20, kde=True),
            mo.md("#### Raw Data (first 10 rows):"),
            results_df.head(10),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Eval against target""")
    return


@app.cell
def _(pd):
    mpc_study_df = pd.read_csv("data/mpc_study_data.csv")
    mpc_study_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
