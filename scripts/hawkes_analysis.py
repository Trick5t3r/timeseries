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
    
    # Conversion de la colonne Year en format datetime
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    
    # Filtrer pour ne garder que les données mondiales
    world_data = df[df['Entity'] == 'World'].copy()
    
    # Calculer les changements d'émigration entre les années
    world_data['Emigration_Change'] = world_data['Total number of emigrants'].diff()
    
    # Supprimer la première ligne (NaN)
    world_data = world_data.dropna()
    
    return world_data

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

def plot_hawkes_results(data, params):
    """Trace les résultats de l'analyse de Hawkes."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. Données brutes et intensité
    years = np.arange(len(data))
    events = data['Emigration_Change'].values
    
    # Calculer l'intensité pour chaque point
    intensities = [hawkes_intensity(t, events, *params) for t in years]
    
    ax1.plot(years, events, 'b-', label='Changements d\'émigration', alpha=0.6)
    ax1.plot(years, intensities, 'r-', label='Intensité du processus de Hawkes')
    ax1.set_title('Dynamique d\'émigration mondiale et intensité du processus de Hawkes')
    ax1.set_xlabel('Années (depuis 1990)')
    ax1.set_ylabel('Nombre d\'émigrants')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Distribution des changements d'émigration
    sns.histplot(data=data, x='Emigration_Change', bins=20, ax=ax2)
    ax2.set_title('Distribution des changements d\'émigration')
    ax2.set_xlabel('Changement d\'émigration')
    ax2.set_ylabel('Fréquence')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_hawkes_process(data, threshold_percent=5):
    """Crée une visualisation du processus de Hawkes avec processus ponctuel et intensité."""
    # Calculer les changements en pourcentage
    data['Change_Percent'] = data['Total number of emigrants'].pct_change() * 100
    
    # Identifier les événements (pics d'émigration)
    events = data[data['Change_Percent'] > threshold_percent].copy()
    
    # Créer la figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. Processus ponctuel (Nt)
    years = np.arange(len(data))
    cumulative_events = np.zeros(len(data))
    event_years = []
    
    for idx, row in events.iterrows():
        year_idx = data.index.get_loc(idx)
        cumulative_events[year_idx:] += 1
        event_years.append(year_idx)
    
    # Tracer le processus ponctuel
    ax1.step(years, cumulative_events, 'b-', label='Processus ponctuel Nt', where='post')
    ax1.plot(event_years, cumulative_events[event_years], 'ko', label='Événements')
    ax1.set_title('Processus ponctuel des pics d\'émigration')
    ax1.set_xlabel('Années (depuis 1990)')
    ax1.set_ylabel('Nombre cumulé d\'événements')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Intensité conditionnelle (λt)
    # Ajuster le modèle de Hawkes sur les indices des événements
    event_indices = np.array(event_years)
    params = fit_hawkes_model(event_indices)
    
    # Calculer l'intensité pour chaque point
    intensities = np.zeros(len(years))
    for t in years:
        intensities[t] = hawkes_intensity(t, event_indices, *params)
    
    # Tracer l'intensité
    ax2.plot(years, intensities, 'r-', label='Intensité conditionnelle λt')
    ax2.plot(event_years, intensities[event_years], 'ko', label='Événements')
    ax2.set_title('Intensité conditionnelle du processus de Hawkes')
    ax2.set_xlabel('Années (depuis 1990)')
    ax2.set_ylabel('Intensité')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Afficher les statistiques
    print(f"\nAnalyse du processus de Hawkes (seuil: {threshold_percent}%):")
    print(f"Nombre total d'événements: {len(events)}")
    print(f"Paramètres du modèle:")
    print(f"  μ (taux de base): {params[0]:.2f}")
    print(f"  α (force d'excitation): {params[1]:.2f}")
    print(f"  β (taux de décroissance): {params[2]:.2f}")
    
    # Afficher les années avec des pics significatifs
    print("\nPics d'émigration significatifs:")
    for _, row in events.iterrows():
        print(f"  {row['Year'].year}: {row['Change_Percent']:.1f}% d'augmentation")

def main():
    # Chargement des données
    data = load_and_prepare_data('data/total-number-of-emigrants.csv')
    
    # Création des visualisations
    print("\nGénération des visualisations...")
    
    # Visualisation du processus de Hawkes
    plot_hawkes_process(data, threshold_percent=5)
    
    # Ajustement du modèle de Hawkes
    print("\nAjustement du modèle de Hawkes...")
    events = data['Emigration_Change'].values
    params = fit_hawkes_model(events)
    
    # Affichage des résultats
    print("\nParamètres du modèle de Hawkes:")
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
    plot_hawkes_results(data, params)

if __name__ == "__main__":
    main() 