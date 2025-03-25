import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import seaborn as sns


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




def plot_extreme_quantile(data, q, monthly=True):
    months = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
    Quantile_fr = []
    Quantile_we = []
    if monthly:
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10), sharex=True, sharey=True)
    
    
        for i, ax in enumerate(axes.flat):
            # Extract temperature data for the month
            temp_data = data[i+1]['Temperature']
            
        
            
            # Plot the histogram and KDE
            sns.histplot(temp_data, bins=20, ax=ax, stat='density')  # kde=True enables smooth density
            
            # Fit distributions to the data
            # Weibull Distribution Fit
            shape_we, loc_we, scale_we = scp.stats.weibull_min.fit(temp_data, floc=0)
            x_we = np.linspace(min(temp_data), max(temp_data), 100)
            pdf_we = scp.stats.weibull_min.pdf(x_we, shape_we, loc_we, scale_we)
            ax.plot(x_we, pdf_we, color='g', label=f'Weibull fit', alpha=0.7)
            
            # Fréchet Distribution Fit (same as Gumbel for this case, using generalized Gumbel)
            loc_fr, scale_fr = scp.stats.gumbel_r.fit(temp_data)
            x_fr = np.linspace(min(temp_data), max(temp_data), 100)
            pdf_fr = scp.stats.gumbel_r.pdf(x_fr, loc_fr, scale_fr)
            ax.plot(x_fr, pdf_fr, color='r', label=f'Fréchet fit', alpha=0.7)
            
            # Gumbel Distribution Fit
            # loc_gu, scale_gu = scp.stats.gumbel_r.fit(temp_data)
            # x_gu = np.linspace(min(temp_data), max(temp_data), 100)
            # pdf_gu = scp.stats.gumbel_r.pdf(x_gu, loc_gu, scale_gu)
            # ax.plot(x_gu, pdf_gu, color='k', label=f'Gumbel fit', alpha=0.7)

            # Estimate extreme quantiles
            quantile_we = scp.stats.weibull_min.ppf(q, shape_we, loc_we, scale_we)
            quantile_fr = scp.stats.gumbel_r.ppf(q, loc_fr, scale_fr)
            # quantile_gu = stats.gumbel_r.ppf(q, loc_gu, scale_gu)

            Quantile_fr.append(quantile_fr)
            Quantile_we.append(quantile_we)

            # Plot the extreme quantiles for each distribution
            ax.axvline(quantile_we, color='g', linestyle='--', alpha=0.5, label=f'{q}-quantile Weibull')
            ax.axvline(quantile_fr, color='r', linestyle='--', alpha=0.5, label=f'{q}-quantile Fréchet')
            # ax.axvline(quantile_gu, color='k', linestyle='--', alpha=0.5, label=f'{q}-quantile Gumbel')
            
            # print(loc_fr, scale_fr, loc_gu, scale_gu)
            ax.set_title(months[i])

            # Show legend only on the first subplot
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
            else:
                ax.legend().set_visible(False)  # Hide legend for other subplots

        # Add global legend outside the subplots
        fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.15, 0.5), ncol=1)
        plt.suptitle(f'{q}-quantile of Temperature for each month ')
        plt.tight_layout()
    else:
        # Extract temperature data for the month
        temp_data = data['Temperature']
                
        
        # Plot the histogram and KDE
        sns.histplot(temp_data, bins=20, stat='density')  # kde=True enables smooth density
        
        # Fit distributions to the data
        # Weibull Distribution Fit
        shape_we, loc_we, scale_we = scp.stats.weibull_min.fit(temp_data, floc=0)
        x_we = np.linspace(min(temp_data), max(temp_data), 100)
        pdf_we = scp.stats.weibull_min.pdf(x_we, shape_we, loc_we, scale_we)
        plt.plot(x_we, pdf_we, color='g', label=f'Weibull fit', alpha=0.7)
        
        # Fréchet Distribution Fit (same as Gumbel for this case, using generalized Gumbel)
        loc_fr, scale_fr = scp.stats.gumbel_r.fit(temp_data)
        x_fr = np.linspace(min(temp_data), max(temp_data), 100)
        pdf_fr = scp.stats.gumbel_r.pdf(x_fr, loc_fr, scale_fr)
        plt.plot(x_fr, pdf_fr, color='r', label=f'Fréchet fit', alpha=0.7)
        
        # Gumbel Distribution Fit
        # loc_gu, scale_gu = scp.stats.gumbel_r.fit(temp_data)
        # x_gu = np.linspace(min(temp_data), max(temp_data), 100)
        # pdf_gu = scp.stats.gumbel_r.pdf(x_gu, loc_gu, scale_gu)
        # ax.plot(x_gu, pdf_gu, color='k', label=f'Gumbel fit', alpha=0.7)

        # Estimate extreme quantiles
        Quantile_we = scp.stats.weibull_min.ppf(q, shape_we, loc_we, scale_we)
        Quantile_fr = scp.stats.gumbel_r.ppf(q, loc_fr, scale_fr)
        # quantile_gu = stats.gumbel_r.ppf(q, loc_gu, scale_gu)

        # Plot the extreme quantiles for each distribution
        plt.axvline(Quantile_we, color='g', linestyle='--', alpha=0.5, label=f'{q}-quantile Weibull')
        plt.axvline(Quantile_fr, color='r', linestyle='--', alpha=0.5, label=f'{q}-quantile Fréchet')
        # ax.axvline(quantile_gu, color='k', linestyle='--', alpha=0.5, label=f'{q}-quantile Gumbel')
        plt.legend(loc='right', bbox_to_anchor=(1.47, 0.5), ncol=1)
        plt.title(f'{q}-quantile of Temperature yearly')
    
    plt.show()
    return Quantile_fr, Quantile_we





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
    Quantile_fr, Quantile_we = plot_extreme_quantile(groups, 0.99, monthly=True)

    q_fr, q_we = plot_extreme_quantile(df, 0.99, monthly=False)

    proportion_above_threshold(df, q_fr, q_we)
    proportion_above_threshold(groups, Quantile_fr, Quantile_we)

    return 0




if __name__ == '__main__':
    main()