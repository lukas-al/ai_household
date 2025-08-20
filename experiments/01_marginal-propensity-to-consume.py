import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys
    from openai import OpenAI
    sys.executable
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
    # Consumption and Saving Responses to Idiosyncratic Shocks - the Marginal Propensity to Consume
    The marginal propensity to consume—the fraction of a small, unanticipated, one-time income windfall that a household spends within a given period—is a central concept in modern macroeconomics. It is the primary determinant of the first-round effect of fiscal transfers and a key parameter in the transmission of monetary policy through income channels.

    ## Empirical Benchmarks
    A large and robust body of empirical work, employing diverse methodologies, has established a clear benchmark for the average quarterly MPC. Studies using quasi-experimental evidence from large-scale government transfer programs, such as the 2001 and 2008 U.S. tax rebates, form the bedrock of this consensus. Seminal work by Johnson, Parker, and Souleles (2006) and Parker et al. (2013) found that households spent a significant portion of these rebates on non-durable goods shortly after receipt. This evidence, corroborated by studies of lottery winnings , government shutdowns , and survey-based hypothetical questions, points to an average quarterly MPC for non-durable goods out of transitory windfalls of \$500-\$1,000 in the range of 15% to 25%.  

    More recent evidence from the large-scale Economic Impact Payments (EIPs) distributed during the COVID-19 pandemic has reinforced the significance of the consumption response, often finding even larger short-term effects. Studies using high-frequency, transaction-level bank account data found that consumers spent between 25% and 48% of their stimulus payments within the first two to four weeks of receipt. For example, one study of the first $1,200 EIP in April 2020 estimated a two-week MPC of 46%. While the shorter time horizon of these studies makes direct comparison to the quarterly benchmark complex, they confirm that the consumption response to transfers is both rapid and substantial. 

    ## Summary of Estimates from Natural Experiments and Elicitation Surveys 
    The following table summarizes key empirical estimates of the MPC, providing concrete targets for model validation. It is mainly sourced from [Crawley & Theoloudis (2024)](https://www.federalreserve.gov/econres/feds/files/2024038pap.pdf)

    ---

    """
            ),
            mo.hstack(
                [
                    mo.md("""
            #### **Natural Experiment Studies**

    | Study | MPC | Horizon | $C_{it}$ | Data |
    | :--- | :--- | :--- | :--- | :--- |
    | Baker et al. (2023) | 0.25–0.40 | First weeks | te | SaverLife 2020 |
    | Boehm, Fize, and Jaravel (2023) | 0.23 | 1 month | te | French RCT 2022 |
    | Fagereng, Holm, and Natvik (2021)| 0.35–0.71 | 1st year | te | Norwegian registry data (N) 1993-2015 |
    | Gelman et al. (2023) | ≈1.00 | Permanent shock | te | Financial Aggregator (FA) 2013-16 |
    | Johnson, Parker, and Souleles (2006)| 0.20–0.40 | Three-month period | nde | Consumer Expenditure Survey (CEX) 2001 |
    | Misra and Surico (2014) | 0.43 | 3 months | nde | Consumer Expenditure Survey (CEX) 2001 |
    | | 0.16 | 3 months | te | Consumer Expenditure Survey (CEX) 2008 |
    | Karger and Rajan (2020) | 0.46 | Two weeks | te | Facteus 2020 |
    | Misra, Singh, and Zhang (2022) | 0.29 | A few days | te | Facteus 2020 |
    | Orchard, Ramey, and Wieland (2023)| ≈0.3 | 3 months | te | Consumer Expenditure Survey (CEX) 2008 |
    | Parker et al. (2013) | 0.50–0.90 | 3 months | te | Consumer Expenditure Survey (CEX) 2008 |
    | Parker et al. (2022) | 0.05–0.16 | 3 months | nde | Consumer Expenditure Survey (CEX) 2020-21 |
    | Sahm, Shapiro, and Slemrod (2010)| ≈0.3 | 1 year | te | Michigan Survey of Consumers/CEX 2008 |

    ---
    """),
                    mo.md("""
            #### **Elicitation Survey Studies**

    | Study | MPC | Horizon | $C_{it}$ | Data |
    | :--- | :--- | :--- | :--- | :--- |
    | Bunn et al. (2018) | 0.14 (pos.) 0.64 (neg.)| 1 year | te | Bank of England (BoE) survey 2011-14 |
    | Christelis et al. (2019) | 0.20 (pos.) 0.24 (neg.)| 1 year | te | Dutch survey 2015 |
    | Colarieti, Mei, and Stantcheva (2024)| 0.16 | 1 quarter | te | Authors’ survey 2022-23 |
    | Fuster, Kaplan, and Zafar (2021) | 0.07 (pos.) 0.32 (neg.)| 3 months | te | NY Fed Survey of Consumer Expectations (SCE) 2016-17 |
    | Jappelli and Pistaferri (2014) | 0.48 | Unspecified | te | Italian Survey of Household Income and Wealth (SHIW) 2010 |
    | Jappelli and Pistaferri (2020) | 0.47 | Unspecified | te | Italian Survey of Household Income and Wealth (SHIW) 2016 |

    ---
            """),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    #### **Legend**

    * **neg.**: negative
    * **nde**: non-durable expenditure
    * **pos.**: positive
    * **te**: total expenditure
    * **BoE**: Bank of England
    * **CEX**: Consumer Expenditure Survey
    * **FA**: Financial Aggregator
    * **MPC**: marginal propensity to consume
    * **N**: Norwegian registry data
    * **RCT**: randomized control trial
    * **SCE**: Survey of Consumer Expectations
    * **SHIW**: Italian Survey of Household Income and Wealth
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Summary of estimates from structural methods
    Personally, I find these less convincing the empirical surveys but they remain useful to compare

    | Study | Pass-through (perm.) | Pass-through (trans.) | MPC (perm.) | MPC (trans.) | Variable (Y) | Variable (C) | Data |
    | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
    | Alan, Browning, and Ejrnæs (2018) | .05 to .69 |  |  |  | thy | food | P 1999-2009 |
    | Arellano, Blundell, and Bonhomme (2017)a |  | .2−.4 |  | \-.4 to .2 | thy | nde | P 1999-2009 |
    | Arellano et al. (2024)b |  |  | .33 |  | dhy | nde | P 2005-17 |
    | Blundell, Pistaferri, and Preston (2008) | .64 | .05 |  |  | dhy | nde | P & C 1980-92 |
    | Blundell, Pistaferri, and Saporta-Eksten (2016) | .32 | \-.14 |  |  | mhw | nde | P 1999-2009 |
    |  | .19 | \-.04 |  |  | fhw | nde | P 1999-2009 |
    | Blundell, Pistaferri, and Saporta-Eksten (2018) | .39 | .12 |  |  | mhw | nde | P 1999-2015, and |
    |  | .35 | .13 |  |  | fhw | nde | C & A 2003-15 |
    | Busch and Ludwig (2023)c | .40 | .05 | .38 | .05 | dhy | n/a | P 1977-2012 |
    | Chopra (2023) | .29 r | \-.18 r | .19 r |  | mhw | nde | P 1977-2016 |
    |  | .31 x | \-.26 x | .12 x |  | mhw | nde | P 1977-2016 |
    | Commault (2022) |  | .6 |  | .32 | dhy | nde | P & C 1980-92 |
    | Crawley (2020) | .34 | .24 |  |  | dhy | nde | P & C 1980-92 |
    | Crawley and Kuchler (2023) |  |  | .64 | .64 | dhy | te | D 2003-15 |
    | De Nardi, Fella, and Paz-Pardo (2020) | .54 | .12 |  |  | dhy | nde | P 1968-92, and C 1980-2007 |
    | Ghosh and Theloudis (2023)c | .13 | \-.00 |  |  | dhy | nde | P 1999-2019 |
    | Guvenen and Smith (2014) | .45 |  |  |  | dhy | nde | P & C 1968-92 |
    | Guvenen, Madera, and Ozkan (2023)d | .38 | .11 | .4 | .05 | dhy | nde | external estims |
    | Heathcote, Storesletten, and Violante (2014) | .39 |  |  |  | mhw | nde | P 1968-2007, and C 1980-2006 |
    | Hryshko and Manovskii (2022) | .87 sn | .07 sn |  |  | dhy | nde | P & C 1980-92 |
    |  | .46 dg | .12 dg |  |  | dhy | nde | P & C 1980-92 |
    | Jessen and K¨onig (2023) | .62 |  |  |  | mhw | n/a | P 1970-1997 |
    | Kaplan and Violante (2010) | .78 | .06 |  |  | dhy | nde | P 1980-92, SCF |
    | Low, Meghir, and Pistaferri (2010)e | .56 |  |  |  | mhw | n/a | P 1988-96, SIPP |
    | Madera (2019)d | .50 | .10 |  |  | dhy | te | P 1999-2015 |
    | Theloudis (2021) | .45 | \-.03 |  |  | mhw | nde | P 1999-2011 |
    |  | .27 | \-.05 |  |  | fhw | nde | P 1999-2011 |
    | Wu and Krueger (2021) | .35 | .01 |  |  | mhw | nde | P 1999-2009 |
    |  | .18 | .01 |  |  | fhw | nde | P 1999-2009 |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### **Legend**

    * **dg**: daughters  
    * **dhy**: disposable household income  
    * **fhw**: female hourly wage  
    * **mhw**: male hourly wage  
    * **nde**: non-durable expenditure  
    * **n/a**: not applicable  
    * **r**: recession  
    * **sn**: sons  
    * **te**: total expenditure  
    * **thy**: total household income  
    * **x**: expansion  
    * **A**: American Time Use Survey  
    * **C**: Consumer Expenditure Survey  
    * **D**: Danish registry data  
    * **P**: Panel Study of Income Dynamics  
    * **SCF**: Survey of Consumer Finances  
    * **SIPP**: Survey of Income & Program Participation

    #### **Footnotes**

    * **a**: Results with unobserved household heterogeneity, figures S21 and S24.  
    * **b**: Results with filtering and unobserved household heterogeneity.  
    * **c**: Results for average/medium shock.  
    * **d**: Results at age 40\.  
    * **e**: Pass-through of income following an unemployment shock.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    A synthetic household model should be subjected to the following three-part validation test:

    1. Aggregate MPC: Simulate a small, unanticipated, one-time transfer (e.g., $500) to all households in the model. Calculate the aggregate consumption response for non-durable goods in the quarter the transfer is received. The model's aggregate quarterly MPC should fall within the empirically established 15-25% range.
    2. MPC Distribution: Plot the histogram of individual household MPCs from the simulation. The distribution must be highly dispersed and right-skewed, with a significant mass of households exhibiting an MPC near zero and a smaller but non-trivial mass of households with very high MPCs (e.g., >0.75).
    3. MPC by Liquidity: Group the synthetic households into quintiles based on their ratio of liquid wealth to quarterly income. Calculate the average MPC for each quintile. The model is validated if there is a steep, negative gradient, with the average MPC of the lowest liquidity quintile being substantially higher (e.g., >40%) than the average MPC of the highest liquidity quintile (e.g., <5%).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Implementing the experiment""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Configure environment""")
    return


@app.cell
def _():
    import getpass
    import os

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    if not OPENROUTER_API_KEY:
        # Careful - will echo into the marimo NB
        OPENROUTER_API_KEY = getpass.getpass(prompt="Input the open router API key")
        os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Create custom experiment""")
    return


@app.cell
def _():
    from pydantic import BaseModel, Field
    from src.ai_household import SimpleHousehold

    # Response model for MPC experiment
    class MPCResponse(BaseModel):
        """Response model for marginal propensity to consume decisions."""

        amount_spent: float = Field(
            ...,
            description="The dollar amount the household decides to spend from the windfall",
            ge=0,
        )
        amount_saved: float = Field(
            ...,
            description="The dollar amount the household decides to save from the windfall", 
            ge=0,
        )
        reasoning: str = Field(
            ...,
            description="Brief explanation of the decision-making process"
        )

    return BaseModel, MPCResponse, SimpleHousehold


@app.cell
def _(BaseModel, MPCResponse):
    from src.ai_household import BaseScenario, BaseHousehold

    # MPC scenario implementation  
    class MPCScenario(BaseScenario):
        """Scenario testing marginal propensity to consume with small windfall."""

        def __init__(self, windfall_amount: float = 1000.0):
            self._windfall_amount = windfall_amount

        @property
        def name(self) -> str:
            return f"MPC_Windfall_{int(self._windfall_amount)}"

        @property  
        def response_model(self) -> type[BaseModel]:
            return MPCResponse

        def get_prompt(self, household: BaseHousehold) -> str:
            prompt = f"""
            You are making financial decisions for a household that has just received an unexpected one-time windfall.

            Household details:
            - Annual income: ${household.income:,.0f}
            - Current liquid savings: ${household.liquid_wealth:,.0f}
            - Windfall received: ${self._windfall_amount:,.0f}

            Please decide how to allocate this windfall between spending and saving. Consider your household's financial situation and explain your reasoning briefly.
            """
            return prompt

    return (MPCScenario,)


@app.cell
def _():
    # import instructor

    # client = instructor.from_provider(
    #     "/openai/gpt-oss-20b",
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=OPENROUTER_API_KEY
    # )
    return


@app.cell
def _(MPCScenario, SimpleHousehold):
    from src.ai_household import ExperimentRunner

    # Create households with different wealth levels
    _households = {
        "low_wealth": [
            SimpleHousehold(income=30_000, liquid_wealth=1_000),
            SimpleHousehold(income=35_000, liquid_wealth=2_000),
        ],
        "medium_wealth": [
            SimpleHousehold(income=50_000, liquid_wealth=10_000),
            SimpleHousehold(income=55_000, liquid_wealth=12_000),
        ],
        "high_wealth": [
            SimpleHousehold(income=100_000, liquid_wealth=50_000),
            SimpleHousehold(income=120_000, liquid_wealth=60_000),
        ]
    }

    # Create scenario
    mpc_scenario = MPCScenario(windfall_amount=1000.0)

    # Create experiment runner
    runner = ExperimentRunner(
        populations=_households,
        scenarios=[mpc_scenario]
    )

    return (runner,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Run the experiment""")
    return


@app.cell
def _(runner):
    # Run the experiment
    runner.run()

    # Get results as DataFrame
    results_df = runner.get_results_dataframe()

    return (results_df,)


@app.cell
def _(results_df):

    # Calculate MPC for each household
    results_df['mpc'] = results_df['amount_spent'] / 1000.0  # windfall amount

    # Display basic statistics
    print("MPC Statistics by Household Type:")
    mpc_stats = results_df.groupby('household_type')['mpc'].agg(['mean', 'std', 'min', 'max'])
    print(mpc_stats)

    print(f"\nOverall MPC: {results_df['mpc'].mean():.3f}")

    return


@app.cell
def _(mo, results_df):
    # Display the detailed results
    mo.ui.table(results_df[['household_type', 'hh_income', 'hh_liquid_wealth', 'amount_spent', 'amount_saved', 'mpc', 'reasoning']])
    return


if __name__ == "__main__":
    app.run()
