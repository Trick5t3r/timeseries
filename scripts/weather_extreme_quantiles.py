import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def load_and_prepare_data(file_path):
    """Charge et prépare les données pour l'analyse."""
    print("Chargement des données...")
    df = pd.read_csv(file_path)
    
    # Conversion des colonnes Date et Time en datetime
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Extraction et conversion des variables numériques
    numeric_columns = ['Temperature', 'Humidity', 'Wind Speed', 'Pressure']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].str.replace(' °F', '').str.replace(' %', '').str.replace(' mph', '').str.replace(' in', '').astype(float)
    
    # Utiliser DateTime comme index
    df.set_index('DateTime', inplace=True)
    
    # Sélectionner uniquement les colonnes numériques pour le resampling
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Resampling par heure pour avoir des données régulières et interpolation des valeurs manquantes
    df = numeric_df.resample('H').mean().interpolate(method='time')

    df['Temperature'] = (df['Temperature'] -32) * 5/9
    
    return df

import numpy as np
import pandas as pd

def filter_data_by_quantile(data, q):
    
    # Compute the q-th quantile of the specified column
    quantile_value = np.percentile(data, q * 100)
    
    # Filter the data to keep only values greater or equal to the quantile
    filtered_data = data[data >= quantile_value]
    
    return filtered_data


def plot_extreme_quantile(data, q, monthly=True):
    months = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
    Quantile_fr = []
    Quantile_we = []
    
    if monthly:
        fig = make_subplots(rows=3, cols=4, subplot_titles=months, shared_yaxes='all')
        
        for i in range(12):
            temp_data = data[i + 1]['Temperature']
            temp_data = filter_data_by_quantile(temp_data, 0.8)
            
            show_legend = i == 0  # Show legend only for the first month

            hist_data = go.Histogram(
                x=temp_data,
                xbins=dict(start=min(temp_data), end=max(temp_data) + 2, size=1),
                histnorm='probability density',
                name=f'Month {months[i]}' if show_legend else None,  # Hide legend label for other months
                marker=dict(color='#1f77b4'),
                showlegend=show_legend
            )
            fig.add_trace(hist_data, row=(i // 4) + 1, col=(i % 4) + 1)

            # Weibull Fit
            shape_we, loc_we, scale_we = scp.stats.weibull_min.fit(temp_data, floc=0)
            x_we = np.linspace(min(temp_data), max(temp_data), 100)
            pdf_we = scp.stats.weibull_min.pdf(x_we, shape_we, loc_we, scale_we)
            dist_we = go.Scatter(
                x=x_we, y=pdf_we, mode='lines',
                name='Weibull fit' if show_legend else None,
                line=dict(color='green'),
                showlegend=show_legend
            )
            fig.add_trace(dist_we, row=(i // 4) + 1, col=(i % 4) + 1)

            # Fréchet Fit
            loc_fr, scale_fr = scp.stats.gumbel_r.fit(temp_data)
            x_fr = np.linspace(min(temp_data), max(temp_data), 100)
            pdf_fr = scp.stats.gumbel_r.pdf(x_fr, loc_fr, scale_fr)
            dist_fr = go.Scatter(
                x=x_fr, y=pdf_fr, mode='lines',
                name='Fréchet fit' if show_legend else None,
                line=dict(color='red'),
                showlegend=show_legend
            )
            fig.add_trace(dist_fr, row=(i // 4) + 1, col=(i % 4) + 1)

            # Quantile lines
            quantile_we = scp.stats.weibull_min.ppf(q, shape_we, loc_we, scale_we)
            quantile_fr = scp.stats.gumbel_r.ppf(q, loc_fr, scale_fr)

            quantile_we_line = go.Scatter(
                x=[quantile_we, quantile_we], y=[0, 0.6], mode='lines',
                line=dict(color='green', dash='dash'),
                name=f'{q}-quantile Weibull' if show_legend else None,
                showlegend=show_legend
            )
            fig.add_trace(quantile_we_line, row=(i // 4) + 1, col=(i % 4) + 1)

            quantile_fr_line = go.Scatter(
                x=[quantile_fr, quantile_fr], y=[0, 0.6], mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'{q}-quantile Fréchet' if show_legend else None,
                showlegend=show_legend
            )
            fig.add_trace(quantile_fr_line, row=(i // 4) + 1, col=(i % 4) + 1)



        fig.update_layout(title=f'{q}-quantile of Temperature for each month', showlegend=True)
        fig.update_layout(height=1000, width=1500, showlegend=True)
        fig.show()
    
    else:
        # Handle yearly data plotting
        temp_data = data['Temperature']
        temp_data = filter_data_by_quantile(temp_data, 0.8)
        
        # Plot the histogram using plotly
        hist_data = go.Histogram(x=temp_data, 
                                xbins=dict(start=min(temp_data),  # Start the bins at the minimum value in the column
                                           end=max(temp_data) + 2,    # End the bins at the maximum value in the column
                                           size=1                         # Set the bin width to 1
                                           ),
                                marker=dict(color='#1f77b4'),
                                histnorm='probability density', 
                                name='Yearly temperature'
                                )
        fig = go.Figure(data=[hist_data])

        # Weibull Distribution Fit
        shape_we, loc_we, scale_we = scp.stats.weibull_min.fit(temp_data, floc=0)
        x_we = np.linspace(min(temp_data), max(temp_data), 100)
        pdf_we = scp.stats.weibull_min.pdf(x_we, shape_we, loc_we, scale_we)
        dist_we = go.Scatter(x=x_we, y=pdf_we, mode='lines', name=f'Weibull fit', line=dict(color='green'))
        fig.add_trace(dist_we)

        # Fréchet Distribution Fit (same as Gumbel for this case)
        loc_fr, scale_fr = scp.stats.gumbel_r.fit(temp_data)
        x_fr = np.linspace(min(temp_data), max(temp_data), 100)
        pdf_fr = scp.stats.gumbel_r.pdf(x_fr, loc_fr, scale_fr)
        dist_fr = go.Scatter(x=x_fr, y=pdf_fr, mode='lines', name=f'Fréchet fit', line=dict(color='red'))
        fig.add_trace(dist_fr)

        # Estimate extreme quantiles for Weibull and Fréchet
        Quantile_we = scp.stats.weibull_min.ppf(q, shape_we, loc_we, scale_we)
        Quantile_fr = scp.stats.gumbel_r.ppf(q, loc_fr, scale_fr)

        # Plot the extreme quantiles for each distribution
        quantile_we_line = go.Scatter(x=[Quantile_we, Quantile_we], y=[0, 0.25], mode='lines', line=dict(color='green', dash='dash'), name=f'{q}-quantile Weibull')
        fig.add_trace(quantile_we_line)

        quantile_fr_line = go.Scatter(x=[Quantile_fr, Quantile_fr], y=[0, 0.25], mode='lines', line=dict(color='red', dash='dash'), name=f'{q}-quantile Fréchet')
        fig.add_trace(quantile_fr_line)

        fig.update_layout(title=f'{q}-quantile of Temperature (Yearly)', showlegend=True)
        fig.update_layout(height=600, width=900, showlegend=True)
        fig.show()

    return Quantile_fr, Quantile_we

def qqplot(data, threshold=0.8, plot_threshold=39):
    temp_data = data['Temperature']
    temp_data = filter_data_by_quantile(temp_data, threshold)
    
    # Weibull Distribution Fit
    shape_we, loc_we, scale_we = scp.stats.weibull_min.fit(temp_data, floc=0)
    
    # Fréchet Distribution Fit (same as Gumbel for this case)
    loc_fr, scale_fr = scp.stats.gumbel_r.fit(temp_data)

    temp_data = temp_data[temp_data >= 39]
    # Generate theoretical quantiles for Weibull
    quantiles_we = np.linspace(0.4, 0.99, len(temp_data))
    theoretical_we = scp.stats.weibull_min.ppf(quantiles_we, shape_we, loc_we, scale_we)
    sorted_data_we = np.sort(temp_data)

    # Generate theoretical quantiles for Fréchet (Gumbel)
    theoretical_fr = scp.stats.gumbel_r.ppf(quantiles_we, loc_fr, scale_fr)
    sorted_data_fr = np.sort(temp_data)

    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=["QQ Plot - Weibull", "QQ Plot - Fréchet (Gumbel)"])

    # Add Weibull QQ plot
    fig.add_trace(go.Scatter(x=theoretical_we, y=sorted_data_we, mode='markers', name='Data (Weibull)', marker=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=theoretical_we, y=theoretical_we, mode='lines', name='Theoretical Line', line=dict(color='red', dash='dash')), row=1, col=1)

    # Add Fréchet QQ plot
    fig.add_trace(go.Scatter(x=theoretical_fr, y=sorted_data_fr, mode='markers', name='Data (Fréchet)', marker=dict(color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=theoretical_fr, y=theoretical_fr, mode='lines', name='Theoretical Line', line=dict(color='red', dash='dash')), row=1, col=2)

    # Update layout
    fig.update_layout(title="QQ Plots for Weibull and Fréchet Distributions", height=500, width=1000, showlegend=True)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

    fig.show()

    



def proportion_above_threshold(data, Quantile_fr, Quantile_we):
    if type(Quantile_fr) == np.float64:
        # Count the number of rows where Temperature is greater than the threshold
        above_threshold_fr = (data['Temperature'] > Quantile_fr).sum()
        above_threshold_we = (data['Temperature'] > Quantile_we).sum()
        # Total number of rows in the DataFrame
        total_rows = len(data)
        
        # Compute the proportion
        proportion_fr = above_threshold_fr / total_rows
        proportion_we = above_threshold_we / total_rows
        print(above_threshold_fr, above_threshold_we, total_rows)
        return proportion_fr, proportion_we
    
    else: #if a list/dict is given (monthly data)
        Proportion = []
        for i in range(1,13):
            Proportion.append(proportion_above_threshold(data[i], Quantile_fr[i-1], Quantile_we[i-1]))
        return Proportion





def main():
    df = load_and_prepare_data("data/weather_data.csv")
    df['month'] = df.index.month  # Ensure index is datetime
    groups = {month: data for month, data in df.groupby('month')}
    
    # Quantile_fr, Quantile_we = plot_extreme_quantile(groups, 0.99, monthly=True)
    # q_fr, q_we = plot_extreme_quantile(df, 0.99, monthly=False)

    # proportion_above_threshold(df, q_fr, q_we)
    # proportion_above_threshold(groups, Quantile_fr, Quantile_we)

    qqplot(df)

    return 0




if __name__ == '__main__':
    main()