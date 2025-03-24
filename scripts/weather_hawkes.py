import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import poisson
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Charge et prépare les données pour l'analyse de Hawkes."""
    print("Chargement des données...")
    df = pd.read_csv(file_path)
    
    # Conversion des colonnes Date et Time en datetime
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Extraction et conversion des variables numériques
    df['Temperature'] = df['Temperature'].str.replace(' °F', '').astype(float)
    df['Wind_Speed'] = df['Wind Speed'].str.replace(' mph', '').astype(float)
    df['Pressure'] = df['Pressure'].str.replace(' in', '').astype(float)
    
    # Calculer les changements entre les heures
    df['Temperature_Change'] = df['Temperature'].diff()
    df['Wind_Speed_Change'] = df['Wind_Speed'].diff()
    df['Pressure_Change'] = df['Pressure'].diff()
    
    # Supprimer les lignes avec des valeurs manquantes
    df = df.dropna()
    
    return df

def hawkes_intensity(t, events, mu, alpha, beta):
    """Calcule l'intensité du processus de Hawkes à un temps t."""
    intensity = mu
    for event in events:
        if event < t:
            intensity += alpha * np.exp(-beta * (t - event))
    return intensity

def negative_log_likelihood(params, events):
    """Calcule la log-vraisemblance négative du processus de Hawkes."""
    mu, alpha, beta = params
    
    # Vérifier les contraintes
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return float('inf')
    
    T = events[-1]  # Période totale
    n = len(events)  # Nombre d'événements
    
    # Calculer la log-vraisemblance
    log_likelihood = -mu * T
    for i in range(n):
        log_likelihood += np.log(hawkes_intensity(events[i], events[:i], mu, alpha, beta))
    
    return -log_likelihood

def fit_hawkes_model(events):
    """Ajuste le modèle de Hawkes aux données."""
    # Initialisation des paramètres
    initial_params = [np.mean(events), 0.5, 1.0]
    
    # Optimisation
    result = minimize(
        negative_log_likelihood,
        initial_params,
        args=(events,),
        method='Nelder-Mead',
        bounds=[(0, None), (0, None), (0, None)]
    )
    
    return result.x

def plot_hawkes_results(data, params, variable):
    """Trace les résultats de l'analyse de Hawkes."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. Données brutes et intensité
    times = np.arange(len(data))
    events = data[f'{variable}_Change'].values
    
    # Calculer l'intensité pour chaque point
    intensities = [hawkes_intensity(t, events, *params) for t in times]
    
    ax1.plot(times, events, 'b-', label=f'Changements de {variable}', alpha=0.6)
    ax1.plot(times, intensities, 'r-', label='Intensité du processus de Hawkes')
    ax1.set_title(f'Dynamique des changements de {variable} et intensité du processus de Hawkes')
    ax1.set_xlabel('Heures')
    ax1.set_ylabel(f'Changement de {variable}')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Distribution des changements
    sns.histplot(data=data, x=f'{variable}_Change', bins=20, ax=ax2)
    ax2.set_title(f'Distribution des changements de {variable}')
    ax2.set_xlabel(f'Changement de {variable}')
    ax2.set_ylabel('Fréquence')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_hawkes_process(data, variable, threshold_percent=5):
    """Crée une visualisation du processus de Hawkes avec processus ponctuel et intensité."""
    # Calculer les changements en pourcentage
    data[f'{variable}_Change_Percent'] = data[variable].pct_change() * 100
    
    # Identifier les événements (pics)
    events = data[data[f'{variable}_Change_Percent'] > threshold_percent].copy()
    
    # Créer la figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. Processus ponctuel (Nt)
    times = np.arange(len(data))
    cumulative_events = np.zeros(len(data))
    event_times = []
    
    for idx, row in events.iterrows():
        time_idx = data.index.get_loc(idx)
        cumulative_events[time_idx:] += 1
        event_times.append(time_idx)
    
    # Tracer le processus ponctuel
    ax1.step(times, cumulative_events, 'b-', label='Processus ponctuel Nt', where='post')
    ax1.plot(event_times, cumulative_events[event_times], 'ko', label='Événements')
    ax1.set_title(f'Processus ponctuel des pics de {variable}')
    ax1.set_xlabel('Heures')
    ax1.set_ylabel('Nombre cumulé d\'événements')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Intensité conditionnelle (λt)
    # Ajuster le modèle de Hawkes sur les indices des événements
    event_indices = np.array(event_times)
    params = fit_hawkes_model(event_indices)
    
    # Calculer l'intensité pour chaque point
    intensities = np.zeros(len(times))
    for t in times:
        intensities[t] = hawkes_intensity(t, event_indices, *params)
    
    # Tracer l'intensité
    ax2.plot(times, intensities, 'r-', label='Intensité conditionnelle λt')
    ax2.plot(event_times, intensities[event_times], 'ko', label='Événements')
    ax2.set_title(f'Intensité conditionnelle du processus de Hawkes pour {variable}')
    ax2.set_xlabel('Heures')
    ax2.set_ylabel('Intensité')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Afficher les statistiques
    print(f"\nAnalyse du processus de Hawkes pour {variable} (seuil: {threshold_percent}%):")
    print(f"Nombre total d'événements: {len(events)}")
    print(f"Paramètres du modèle:")
    print(f"  μ (taux de base): {params[0]:.2f}")
    print(f"  α (force d'excitation): {params[1]:.2f}")
    print(f"  β (taux de décroissance): {params[2]:.2f}")
    
    # Afficher les moments avec des pics significatifs
    print(f"\nPics de {variable} significatifs:")
    for _, row in events.iterrows():
        print(f"  {row['DateTime']}: {row[f'{variable}_Change_Percent']:.1f}% de changement")

def main():
    # Chargement des données
    data = load_and_prepare_data('data/weather_data.csv')
    
    # Variables à analyser
    variables = ['Temperature', 'Wind_Speed', 'Pressure']
    
    for variable in variables:
        print(f"\nAnalyse de Hawkes pour {variable}...")
        
        # Visualisation du processus de Hawkes
        plot_hawkes_process(data, variable, threshold_percent=5)
        
        # Ajustement du modèle de Hawkes
        events = data[f'{variable}_Change'].values
        params = fit_hawkes_model(events)
        
        # Affichage des résultats
        print(f"\nParamètres du modèle de Hawkes pour {variable}:")
        print(f"μ (taux de base): {params[0]:.2f}")
        print(f"α (force d'excitation): {params[1]:.2f}")
        print(f"β (taux de décroissance): {params[2]:.2f}")
        
        # Calcul de la stabilité du processus
        stability = params[1] / params[2]
        print(f"\nStabilité du processus: {stability:.2f}")
        if stability < 1:
            print("Le processus est stable (α < β)")
        else:
            print("Le processus est instable (α ≥ β)")
        
        # Visualisation des résultats
        plot_hawkes_results(data, params, variable)

if __name__ == "__main__":
    main() 