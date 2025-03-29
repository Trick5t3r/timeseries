import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
    
    # Conversion de la température de Fahrenheit en Celsius
    if 'Temperature' in df.columns:
        df['Temperature'] = (df['Temperature'] - 32) * 5/9
    
    # Utiliser DateTime comme index
    df.set_index('DateTime', inplace=True)
    
    # Sélectionner uniquement les colonnes numériques pour le resampling
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Resampling par heure pour avoir des données régulières et interpolation des valeurs manquantes
    df = numeric_df.resample('H').mean().interpolate(method='time')
    
    return df

def plot_decomposition(data, variable, period=24*32):
    """Trace la décomposition de la série temporelle."""
    # Décomposition de la série temporelle
    decomposition = seasonal_decompose(data[variable], period=period)
    
    # Création de la figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
    
    # Données brutes
    ax1.plot(data.index, data[variable], 'b-', label='Données brutes')
    ax1.set_title(f'Données brutes - {variable}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Valeur')
    ax1.grid(True)
    ax1.legend()
    
    # Tendance
    ax2.plot(data.index, decomposition.trend, 'r-', label='Tendance')
    ax2.set_title('Tendance')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Valeur')
    ax2.grid(True)
    ax2.legend()
    
    # Saisonnalité
    ax3.plot(data.index, decomposition.seasonal, 'g-', label='Saisonnalité')
    ax3.set_title('Saisonnalité')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Valeur')
    ax3.grid(True)
    ax3.legend()
    
    # Résidus
    ax4.plot(data.index, decomposition.resid, 'k-', label='Résidus')
    ax4.set_title('Résidus')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Valeur')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('./results/weather_seasonality.png')
    plt.show()

def plot_seasonal_patterns(data, variable, period=24):
    """Trace les patterns saisonniers."""
    # Calculer la moyenne par heure
    hourly_pattern = data.groupby(data.index.hour)[variable].mean()
    
    # Créer la figure
    plt.figure(figsize=(12, 6))
    
    # Tracer le pattern saisonnier
    plt.plot(hourly_pattern.index, hourly_pattern.values, 'b-', marker='o')
    plt.title(f'Pattern saisonnier quotidien - {variable}')
    plt.xlabel('Heure de la journée')
    plt.ylabel('Valeur moyenne')
    plt.grid(True)
    
    # Ajouter des lignes verticales pour le lever et coucher du soleil (approximatif)
    plt.axvline(x=6, color='y', linestyle='--', alpha=0.5, label='Lever du soleil (approximatif)')
    plt.axvline(x=18, color='r', linestyle='--', alpha=0.5, label='Coucher du soleil (approximatif)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/weather_seasonal_patterns.png')
    plt.show()

def plot_trend_analysis(data, variable, window=24):
    """Analyse et trace les tendances."""
    # Calculer la moyenne mobile
    rolling_mean = data[variable].rolling(window=window).mean()
    rolling_std = data[variable].rolling(window=window).std()
    
    # Créer la figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Données brutes et moyenne mobile
    ax1.plot(data.index, data[variable], 'b-', alpha=0.5, label='Données brutes')
    ax1.plot(data.index, rolling_mean, 'r-', label=f'Moyenne mobile ({window}h)')
    ax1.set_title(f'Tendance - {variable}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Valeur')
    ax1.grid(True)
    ax1.legend()
    
    # Écart-type mobile
    ax2.plot(data.index, rolling_std, 'g-', label=f'Écart-type mobile ({window}h)')
    ax2.set_title('Variabilité')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Écart-type')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('./results/weather_trend_analysis.png')
    plt.show()

def analyze_spectral_density(data, variable, sampling_rate=1):
    """Analyse la densité spectrale pour identifier les périodes dominantes."""
    # Calculer la densité spectrale
    f, Pxx = signal.welch(data[variable], fs=sampling_rate, nperseg=1024)
    
    # Créer la figure
    plt.figure(figsize=(12, 6))
    
    # Tracer la densité spectrale
    plt.semilogy(f, Pxx)
    plt.title(f'Densité spectrale - {variable}')
    plt.xlabel('Fréquence (cycles/heure)')
    plt.ylabel('Densité spectrale')
    plt.grid(True)
    
    # Ajouter des lignes verticales pour les périodes importantes
    plt.axvline(x=1/24, color='r', linestyle='--', label='Période journalière')
    plt.axvline(x=1/(24*7), color='g', linestyle='--', label='Période hebdomadaire')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/weather_spectral_density.png')
    plt.show()

def main():
    # Chargement des données
    data = load_and_prepare_data('data/weather_data.csv')
    
    # Variables à analyser
    variables = ['Temperature']#, 'Humidity', 'Wind Speed', 'Pressure']
    
    for variable in variables:
        print(f"\nAnalyse de la saisonnalité et des tendances pour {variable}...")
        
        # Décomposition de la série temporelle
        print(f"\nDécomposition de la série temporelle pour {variable}...")
        plot_decomposition(data, variable)
        
        # Patterns saisonniers
        print(f"\nAnalyse des patterns saisonniers pour {variable}...")
        plot_seasonal_patterns(data, variable)
        
        # Analyse des tendances
        print(f"\nAnalyse des tendances pour {variable}...")
        plot_trend_analysis(data, variable)
        
        # Analyse spectrale
        print(f"\nAnalyse spectrale pour {variable}...")
        analyze_spectral_density(data, variable)
        
        # Statistiques descriptives
        print(f"\nStatistiques descriptives pour {variable}:")
        print(data[variable].describe())
        
        # Calculer la saisonnalité relative
        seasonal_std = data[variable].groupby(data.index.hour).std()
        total_std = data[variable].std()
        seasonality_ratio = seasonal_std.mean() / total_std
        
        print(f"\nRatio de saisonnalité: {seasonality_ratio:.4f}")
        if seasonality_ratio > 0.5:
            print("La saisonnalité est forte")
        elif seasonality_ratio > 0.2:
            print("La saisonnalité est modérée")
        else:
            print("La saisonnalité est faible")

if __name__ == "__main__":
    main()
