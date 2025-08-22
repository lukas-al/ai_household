import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import re
    import plotly.express as px
    return mo, np, pd, px, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # create marginal propensity to consume target data

    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 0th order direct target""")
    return


@app.cell(hide_code=True)
def _():
    studies_metadata = {
        "natural_experiments": [
            {
                "study": "Baker et al. (2023)",
                "mpc": "0.25–0.40",
                "horizon": "First weeks",
                "consumption_type": "te",
                "data": "SaverLife 2020",
            },
            {
                "study": "Boehm, Fize, and Jaravel (2023)",
                "mpc": "0.23",
                "horizon": "1 month",
                "consumption_type": "te",
                "data": "French RCT 2022",
            },
            {
                "study": "Fagereng, Holm, and Natvik (2021)",
                "mpc": "0.35–0.71",
                "horizon": "1st year",
                "consumption_type": "te",
                "data": "Norwegian registry data (N) 1993-2015",
            },
            {
                "study": "Gelman et al. (2023)",
                "mpc": "≈1.00",
                "horizon": "Permanent shock",
                "consumption_type": "te",
                "data": "Financial Aggregator (FA) 2013-16",
            },
            {
                "study": "Johnson, Parker, and Souleles (2006)",
                "mpc": "0.20–0.40",
                "horizon": "Three-month period",
                "consumption_type": "nde",
                "data": "Consumer Expenditure Survey (CEX) 2001",
            },
            {
                "study": "Misra and Surico (2014)",
                "mpc": "0.43",
                "horizon": "3 months",
                "consumption_type": "nde",
                "data": "Consumer Expenditure Survey (CEX) 2001",
            },
            {
                "study": "Misra and Surico (2014)",
                "mpc": "0.16",
                "horizon": "3 months",
                "consumption_type": "te",
                "data": "Consumer Expenditure Survey (CEX) 2008",
            },
            {
                "study": "Karger and Rajan (2020)",
                "mpc": "0.46",
                "horizon": "Two weeks",
                "consumption_type": "te",
                "data": "Facteus 2020",
            },
            {
                "study": "Misra, Singh, and Zhang (2022)",
                "mpc": "0.29",
                "horizon": "A few days",
                "consumption_type": "te",
                "data": "Facteus 2020",
            },
            {
                "study": "Orchard, Ramey, and Wieland (2023)",
                "mpc": "≈0.3",
                "horizon": "3 months",
                "consumption_type": "te",
                "data": "Consumer Expenditure Survey (CEX) 2008",
            },
            {
                "study": "Parker et al. (2013)",
                "mpc": "0.50–0.90",
                "horizon": "3 months",
                "consumption_type": "te",
                "data": "Consumer Expenditure Survey (CEX) 2008",
            },
            {
                "study": "Parker et al. (2022)",
                "mpc": "0.05–0.16",
                "horizon": "3 months",
                "consumption_type": "nde",
                "data": "Consumer Expenditure Survey (CEX) 2020-21",
            },
            {
                "study": "Sahm, Shapiro, and Slemrod (2010)",
                "mpc": "≈0.3",
                "horizon": "1 year",
                "consumption_type": "te",
                "data": "Michigan Survey of Consumers/CEX 2008",
            },
        ],
        "elicitation_surveys": [
            {
                "study": "Bunn et al. (208)",
                "mpc": "0.14 (pos.) 0.64 (neg.)",
                "horizon": "1 year",
                "consumption_type": "te",
                "data": "Bank of England (BoE) survey 2011-14",
            },
            {
                "study": "Christelis et al. (2019)",
                "mpc": "0.20 (pos.) 0.24 (neg.)",
                "horizon": "1 year",
                "consumption_type": "te",
                "data": "Dutch survey 2015",
            },
            {
                "study": "Colarieti, Mei, and Stantcheva (2024)",
                "mpc": "0.16",
                "horizon": "1 quarter",
                "consumption_type": "te",
                "data": "Authors’ survey 2022-23",
            },
            {
                "study": "Fuster, Kaplan, and Zafar (2021)",
                "mpc": "0.07 (pos.) 0.32 (neg.)",
                "horizon": "3 months",
                "consumption_type": "te",
                "data": "NY Fed Survey of Consumer Expectations (SCE) 2016-17",
            },
            {
                "study": "Jappelli and Pistaferri (2014)",
                "mpc": "0.48",
                # "horizon": "Unspecified",
                "horizon": "1 year",
                "consumption_type": "te",
                "data": "Italian Survey of Household Income and Wealth (SHIW) 2010",
            },
            {
                "study": "Jappelli and Pistaferri (2020)",
                "mpc": "0.47",
                # "horizon": "Unspecified",
                "horizon": "1 year",
                "consumption_type": "te",
                "data": "Italian Survey of Household Income and Wealth (SHIW) 2016",
            },
        ],
        "structural_methods": [
            {
                "study": "Alan, Browning, and Ejrnæs (2018)",
                "pass_through_perm": ".05 to .69",
                "pass_through_trans": None,
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "thy",
                "variable_c": "food",
                "data": "P 1999-2009",
            },
            {
                "study": "Arellano, Blundell, and Bonhomme (2017)a",
                "pass_through_perm": None,
                "pass_through_trans": ".2−.4",
                "mpc_perm": None,
                "mpc_trans": "-.4 to .2",
                "variable_y": "thy",
                "variable_c": "nde",
                "data": "P 1999-2009",
            },
            {
                "study": "Arellano et al. (2024)b",
                "pass_through_perm": None,
                "pass_through_trans": None,
                "mpc_perm": ".33",
                "mpc_trans": None,
                "variable_y": "dhy",
                "variable_c": "nde",
                "data": "P 2005-17",
            },
            {
                "study": "Blundell, Pistaferri, and Preston (2008)",
                "pass_through_perm": ".64",
                "pass_through_trans": ".05",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "dhy",
                "variable_c": "nde",
                "data": "P & C 1980-92",
            },
            {
                "study": "Blundell, Pistaferri, and Saporta-Eksten (2016)",
                "pass_through_perm": ".32",
                "pass_through_trans": "-.14",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "mhw",
                "variable_c": "nde",
                "data": "P 1999-2009",
            },
            {
                "study": "Blundell, Pistaferri, and Saporta-Eksten (2016)",
                "pass_through_perm": ".19",
                "pass_through_trans": "-.04",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "fhw",
                "variable_c": "nde",
                "data": "P 1999-2009",
            },
            {
                "study": "Blundell, Pistaferri, and Saporta-Eksten (2018)",
                "pass_through_perm": ".39",
                "pass_through_trans": ".12",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "mhw",
                "variable_c": "nde",
                "data": "P 1999-2015, and",

            },
            {
                "study": "Blundell, Pistaferri, and Saporta-Eksten (2018)",
                "pass_through_perm": ".35",
                "pass_through_trans": ".13",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "fhw",
                "variable_c": "nde",
                "data": "C & A 2003-15",
            },
            {
                "study": "Busch and Ludwig (2023)c",
                "pass_through_perm": ".40",
                "pass_through_trans": ".05",
                "mpc_perm": ".38",
                "mpc_trans": ".05",
                "variable_y": "dhy",
                "variable_c": "n/a",
                "data": "P 1977-2012",
            },
            {
                "study": "Chopra (2023)",
                "pass_through_perm": ".29 r",
                "pass_through_trans": "-.18 r",
                "mpc_perm": ".19 r",
                "mpc_trans": None,
                "variable_y": "mhw",
                "variable_c": "nde",
                "data": "P 1977-2016",
            },
            {
                "study": "Chopra (2023)",
                "pass_through_perm": ".31 x",
                "pass_through_trans": "-.26 x",
                "mpc_perm": ".12 x",
                "mpc_trans": None,
                "variable_y": "mhw",
                "variable_c": "nde",
                "data": "P 1977-2016",
            },
            {
                "study": "Commault (2022)",
                "pass_through_perm": None,
                "pass_through_trans": ".6",
                "mpc_perm": None,
                "mpc_trans": ".32",
                "variable_y": "dhy",
                "variable_c": "nde",
                "data": "P & C 1980-92",
            },
            {
                "study": "Crawley (2020)",
                "pass_through_perm": ".34",
                "pass_through_trans": ".24",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "dhy",
                "variable_c": "nde",
                "data": "P & C 1980-92",
            },
            {
                "study": "Crawley and Kuchler (2023)",
                "pass_through_perm": None,
                "pass_through_trans": None,
                "mpc_perm": ".64",
                "mpc_trans": ".64",
                "variable_y": "dhy",
                "variable_c": "te",
                "data": "D 2003-15",
            },
            {
                "study": "De Nardi, Fella, and Paz-Pardo (2020)",
                "pass_through_perm": ".54",
                "pass_through_trans": ".12",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "dhy",
                "variable_c": "nde",
                "data": "P 1968-92, and C 1980-2007",
            },
            {
                "study": "Ghosh and Theloudis (2023)c",
                "pass_through_perm": ".13",
                "pass_through_trans": "-.00",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "dhy",
                "variable_c": "nde",
                "data": "P 1999-2019",
            },
            {
                "study": "Guvenen and Smith (2014)",
                "pass_through_perm": ".45",
                "pass_through_trans": None,
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "dhy",
                "variable_c": "nde",
                "data": "P & C 1968-92",
            },
            {
                "study": "Guvenen, Madera, and Ozkan (2023)d",
                "pass_through_perm": ".38",
                "pass_through_trans": ".11",
                "mpc_perm": ".4",
                "mpc_trans": ".05",
                "variable_y": "dhy",
                "variable_c": "nde",
                "data": "external estims",
            },
            {
                "study": "Heathcote, Storesletten, and Violante (2014)",
                "pass_through_perm": ".39",
                "pass_through_trans": None,
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "mhw",
                "variable_c": "nde",
                "data": "P 1968-2007, and C 1980-2006",
            },
            {
                "study": "Hryshko and Manovskii (2022)",
                "pass_through_perm": ".87 sn",
                "pass_through_trans": ".07 sn",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "dhy",
                "variable_c": "nde",
                "data": "P & C 1980-92",
            },
            {
                "study": "Hryshko and Manovskii (2022)",
                "pass_through_perm": ".46 dg",
                "pass_through_trans": ".12 dg",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "dhy",
                "variable_c": "nde",
                "data": "P & C 1980-92",
            },
            {
                "study": "Jessen and K¨onig (2023)",
                "pass_through_perm": ".62",
                "pass_through_trans": None,
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "mhw",
                "variable_c": "n/a",
                "data": "P 1970-1997",
            },
            {
                "study": "Kaplan and Violante (2010)",
                "pass_through_perm": ".78",
                "pass_through_trans": ".06",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "dhy",
                "variable_c": "nde",
                "data": "P 1980-92, SCF",
            },
            {
                "study": "Low, Meghir, and Pistaferri (2010)e",
                "pass_through_perm": ".56",
                "pass_through_trans": None,
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "mhw",
                "variable_c": "n/a",
                "data": "P 1988-96, SIPP",
            },
            {
                "study": "Madera (2019)d",
                "pass_through_perm": ".50",
                "pass_through_trans": ".10",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "dhy",
                "variable_c": "te",
                "data": "P 1999-2015",
            },
            {
                "study": "Theloudis (2021)",
                "pass_through_perm": ".45",
                "pass_through_trans": "-.03",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "mhw",
                "variable_c": "nde",
                "data": "P 1999-2011",
            },
            {
                "study": "Theloudis (2021)",
                "pass_through_perm": ".27",
                "pass_through_trans": "-.05",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "fhw",
                "variable_c": "nde",
                "data": "P 1999-2011",
            },
            {
                "study": "Wu and Krueger (2021)",
                "pass_through_perm": ".35",
                "pass_through_trans": ".01",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "mhw",
                "variable_c": "nde",
                "data": "P 1999-2009",
            },
            {
                "study": "Wu and Krueger (2021)",
                "pass_through_perm": ".18",
                "pass_through_trans": ".01",
                "mpc_perm": None,
                "mpc_trans": None,
                "variable_y": "fhw",
                "variable_c": "nde",
                "data": "P 1999-2009",
            },
        ],
    }
    return (studies_metadata,)


@app.cell
def _(pd, re, studies_metadata):
    def parse_mpc(mpc_str):
        """Parses an MPC string into lower and upper bounds."""
        if mpc_str is None:
            return None, None
        mpc_str = mpc_str.replace("≈", "").strip()
        if "–" in mpc_str:
            lower, upper = mpc_str.split("–")
            return float(lower), float(upper)
        elif "-" in mpc_str:
            lower, upper = mpc_str.split("-")
            return float(lower), float(upper)
        else:
            value = float(mpc_str)
            return value, value

    def refine_for_plotting(studies_dict):
        """Refines the studies metadata for plotting."""
    
        # Process Natural Experiments
        natural_experiments_data = []
        for study in studies_dict['natural_experiments']:
            lower, upper = parse_mpc(study['mpc'])
            natural_experiments_data.append({
                'study': study['study'],
                'mpc_lower': lower,
                'mpc_upper': upper,
                'horizon': study['horizon'],
                'consumption_type': study['consumption_type'],
                'data': study['data'],
                'type': 'Natural Experiment'
            })
    
        # Process Elicitation Surveys
        elicitation_surveys_data = []
        for study in studies_dict['elicitation_surveys']:
            if '(pos.)' in study['mpc']:
                pos_mpc_str = re.search(r'(\d+\.\d+)\s*\(pos\.\)', study['mpc']).group(1)
                neg_mpc_str = re.search(r'(\d+\.\d+)\s*\(neg\.\)', study['mpc']).group(1)
            
                pos_lower, pos_upper = parse_mpc(pos_mpc_str)
                neg_lower, neg_upper = parse_mpc(neg_mpc_str)

                elicitation_surveys_data.append({
                    'study': study['study'],
                    'mpc_lower': pos_lower,
                    'mpc_upper': pos_upper,
                    'condition': 'positive shock',
                    'horizon': study['horizon'],
                    'consumption_type': study['consumption_type'],
                    'data': study['data'],
                    'type': 'Elicitation Survey'
                })
                elicitation_surveys_data.append({
                    'study': study['study'],
                    'mpc_lower': neg_lower,
                    'mpc_upper': neg_upper,
                    'condition': 'negative shock',
                    'horizon': study['horizon'],
                    'consumption_type': study['consumption_type'],
                    'data': study['data'],
                    'type': 'Elicitation Survey'
                })
            else:
                lower, upper = parse_mpc(study['mpc'])
                elicitation_surveys_data.append({
                    'study': study['study'],
                    'mpc_lower': lower,
                    'mpc_upper': upper,
                    'condition': 'N/A',
                    'horizon': study['horizon'],
                    'consumption_type': study['consumption_type'],
                    'data': study['data'],
                    'type': 'Elicitation Survey'
                })

        # Combine and create DataFrame
        plot_friendly_data = natural_experiments_data
        for item in elicitation_surveys_data:
            if 'condition' not in item:
                item['condition'] = 'N/A'
            plot_friendly_data.append(item)
    
        df = pd.DataFrame(plot_friendly_data)
        df.to_csv("data/mpc_study_.csv", index=False)
    
        return df

    refined_df = refine_for_plotting(studies_metadata)
    print("Refined DataFrame for plotting:")
    refined_df
    return (refined_df,)


@app.cell
def _(np, pd, re):
    def parse_horizon_to_months(horizon_str):
        """
        Converts a string describing a time horizon into a numerical value in months.
        """
        if pd.isna(horizon_str):
            return np.nan
        
        horizon_str = str(horizon_str).lower().strip()
    
        # Extract numbers from the string
        numbers = re.findall(r'\d+\.?\d*', horizon_str)
        num = float(numbers[0]) if numbers else 1

        # Convert based on time unit
        if 'day' in horizon_str:
            return num / 30.0
        elif 'week' in horizon_str:
            return num * (4.345 / 1) # Average weeks in a month
        elif 'month' in horizon_str:
            return num
        elif 'quarter' in horizon_str:
            return num * 3
        elif 'year' in horizon_str:
            return num * 12
        # Handle non-specific terms
        elif 'permanent' in horizon_str or 'unspecified' in horizon_str:
            return np.nan
        else:
            return np.nan

    return (parse_horizon_to_months,)


@app.cell
def _(parse_horizon_to_months, px, refined_df):
    # Apply the parsing function to create a new 'horizon_months' column
    refined_df['horizon_months'] = refined_df['horizon'].apply(parse_horizon_to_months)

    # Drop rows where the horizon could not be parsed into a numerical month value
    df_plot = refined_df.dropna(subset=['horizon_months']).copy()

    # --- 2. Restructure Data for Plotting ---
    # Melt the DataFrame to have separate rows for lower and upper bounds.
    # This makes it easier to plot them as distinct points.
    df_melted = df_plot.melt(
        id_vars=['study', 'horizon_months', 'consumption_type', 'data', 'type', 'condition'],
        value_vars=['mpc_lower', 'mpc_upper'],
        var_name='mpc_bound_type',
        value_name='mpc_value'
    )

    # Make the bound type more readable for the legend
    df_melted['mpc_bound_type'] = df_melted['mpc_bound_type'].str.replace('mpc_', '').str.capitalize()


    # --- 3. Create the Interactive Scatter Plot ---
    fig = px.scatter(
        df_melted,
        x='horizon_months',
        y='mpc_value',
        color='type',  # Color points by study type (Natural Experiment vs. Elicitation Survey)
        symbol='mpc_bound_type', # Use different symbols for lower and upper bounds
        hover_name='study',
        hover_data={
            'study': True,
            'horizon_months': ':.1f', # Format hover data
            'mpc_value': ':.2f',
            'consumption_type': True,
            'data': True,
            'condition': True
        },
        title='Marginal Propensity to Consume (MPC) by Study Horizon',
        labels={
            'horizon_months': 'Horizon (Months)',
            'mpc_value': 'Marginal Propensity to Consume (MPC)',
            'type': 'Study Type',
            'mpc_bound_type': 'MPC Bound'
        }
    )

    # --- 4. Customize and Show the Plot ---
    # Update layout for better readability
    fig.update_layout(
        legend_title_text='Legend',
    )
    fig.update_traces(marker=dict(size=10, opacity=0.7))

    # Display the figure
    fig
    return (df_melted,)


@app.cell
def _(df_melted, px):
    fig_hist = px.histogram(
        df_melted,
        x='mpc_value',
        nbins=20,
        title='Distribution of MPC Estimates from Studies',
        labels={'mpc_value': 'Marginal Propensity to Consume (MPC)'}
    )

    # Display the plot
    fig_hist.show()
    return


if __name__ == "__main__":
    app.run()
