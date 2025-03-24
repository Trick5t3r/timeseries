import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Charge et prépare les données pour l'analyse."""
    print("Chargement des données...")
    df = pd.read_csv(file_path)
    
    # Conversion des colonnes Date et Time en datetime
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Extraction et conversion des variables numériques
    df['Temperature'] = df['Temperature'].str.replace(' °F', '').astype(float)
    # Conversion de Fahrenheit en Celsius
    df['Temperature'] = (df['Temperature'] - 32) * 5/9
    df['Wind_Speed'] = df['Wind Speed'].str.replace(' mph', '').astype(float)
    df['Pressure'] = df['Pressure'].str.replace(' in', '').astype(float)
    
    return df

def plot_time_series(data, variable, title):
    """Trace une série temporelle pour une variable donnée."""
    plt.figure(figsize=(15, 6))
    plt.plot(data['DateTime'], data[variable])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(variable)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_distribution(data, variable, title):
    """Trace la distribution d'une variable."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=variable, bins=30)
    plt.title(title)
    plt.xlabel(variable)
    plt.ylabel('Fréquence')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data):
    """Trace la matrice de corrélation entre les variables."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Matrice de corrélation')
    plt.tight_layout()
    plt.show()

def main():
    # Chargement des données
    data = load_and_prepare_data('data/weather_data.csv')
    
    # Affichage des informations générales
    print("\nInformations générales sur le dataset:")
    print("-" * 50)
    print(f"Nombre total d'observations: {len(data)}")
    print(f"Période couverte: du {data['DateTime'].min()} au {data['DateTime'].max()}")
    print("\nVariables disponibles:")
    for col in ['Temperature', 'Wind_Speed', 'Pressure']:
        print(f"\n{col}:")
        print(f"  Moyenne: {data[col].mean():.2f}")
        print(f"  Écart-type: {data[col].std():.2f}")
        print(f"  Minimum: {data[col].min():.2f}")
        print(f"  Maximum: {data[col].max():.2f}")
    
    # Visualisations
    variables = {
        'Temperature': 'Température (°C)',
        'Wind_Speed': 'Vitesse du vent (mph)',
        'Pressure': 'Pression (in)'
    }
    
    for var, label in variables.items():
        # Série temporelle
        plot_time_series(data, var, f'Évolution de la {label}')
        
        # Distribution
        plot_distribution(data, var, f'Distribution de la {label}')
    
    # Matrice de corrélation
    plot_correlation_matrix(data[['Temperature', 'Wind_Speed', 'Pressure']])
    
    # Analyse des tendances journalières
    data['Hour'] = data['DateTime'].dt.hour
    daily_patterns = data[['Hour', 'Temperature', 'Wind_Speed', 'Pressure']].groupby('Hour').mean()
    
    plt.figure(figsize=(12, 6))
    for var in variables:
        plt.plot(daily_patterns.index, daily_patterns[var], label=variables[var])
    plt.title('Patterns journaliers des variables météorologiques')
    plt.xlabel('Heure de la journée')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 