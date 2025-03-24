import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Charge et prépare les données pour l'analyse des copules."""
    print("Chargement des données...")
    df = pd.read_csv(file_path)
    
    # Conversion des colonnes Date et Time en datetime
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Extraction et conversion des variables numériques
    df['Temperature'] = df['Temperature'].str.replace(' °F', '').astype(float)
    df['Dew_Point'] = df['Dew Point'].str.replace(' °F', '').astype(float)
    df['Humidity'] = df['Humidity'].str.replace(' %', '').astype(float)
    df['Wind_Speed'] = df['Wind Speed'].str.replace(' mph', '').astype(float)
    df['Pressure'] = df['Pressure'].str.replace(' in', '').astype(float)
    
    # Sélection des variables pour l'analyse
    variables = ['Temperature', 'Dew_Point', 'Humidity', 'Wind_Speed', 'Pressure']
    data = df[variables]
    
    return data

def calculate_copula(x, y):
    """Calcule la copule empirique entre deux séries temporelles."""
    # Conversion en rangs
    u = stats.rankdata(x) / (len(x) + 1)
    v = stats.rankdata(y) / (len(y) + 1)
    
    # Calcul de la copule empirique
    n = len(x)
    copula = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            copula[i, j] = np.sum((u <= u[i]) & (v <= v[j])) / n
    
    return copula

def calculate_dependence_measures(x, y):
    """Calcule différentes mesures de dépendance entre deux séries temporelles."""
    # Kendall's tau
    tau, _ = stats.kendalltau(x, y)
    
    # Spearman's rho
    rho, _ = stats.spearmanr(x, y)
    
    # Pearson correlation
    pearson, _ = stats.pearsonr(x, y)
    
    return {
        'kendall_tau': tau,
        'spearman_rho': rho,
        'pearson': pearson
    }

def plot_copula(copula, var1, var2, dependence_measures, x, y):
    """Trace la copule et les mesures de dépendance."""
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Copule
    plt.subplot(1, 3, 1)
    plt.imshow(copula, cmap='viridis', origin='lower')
    plt.colorbar(label='Valeur de la copule')
    plt.title(f'Copule entre {var1} et {var2}')
    plt.xlabel(var2)
    plt.ylabel(var1)
    
    # 2. Nuage de points
    plt.subplot(1, 3, 2)
    plt.scatter(x, y, alpha=0.5)
    plt.title('Nuage de points')
    plt.xlabel(f'{var2}')
    plt.ylabel(f'{var1}')
    
    # 3. Mesures de dépendance
    plt.subplot(1, 3, 3)
    measures = list(dependence_measures.values())
    labels = list(dependence_measures.keys())
    plt.bar(labels, measures)
    plt.title('Mesures de dépendance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()

def main():
    # Chargement des données
    data = load_and_prepare_data('data/weather_data.csv')
    
    # Calculer toutes les paires possibles
    pairs = list(combinations(data.columns, 2))
    
    # Stocker les résultats
    results = []
    
    print("Analyse des copules entre les différentes variables météorologiques...")
    for var1, var2 in pairs:
        x = data[var1].values
        y = data[var2].values
        
        # Calculer la copule
        copula = calculate_copula(x, y)
        
        # Calculer les mesures de dépendance
        dependence_measures = calculate_dependence_measures(x, y)
        
        # Stocker les résultats
        results.append({
            'var1': var1,
            'var2': var2,
            'copula': copula,
            'dependence_measures': dependence_measures,
            'x': x,
            'y': y
        })
    
    # Trouver la meilleure paire (basée sur la valeur absolue de Kendall's tau)
    best_pair = max(results, key=lambda x: abs(x['dependence_measures']['kendall_tau']))
    
    print("\nMeilleure paire de variables trouvée:")
    print(f"{best_pair['var1']} - {best_pair['var2']}")
    print("\nMesures de dépendance:")
    for measure, value in best_pair['dependence_measures'].items():
        print(f"{measure}: {value:.4f}")
    
    # Tracer la copule pour la meilleure paire
    plot_copula(
        best_pair['copula'],
        best_pair['var1'],
        best_pair['var2'],
        best_pair['dependence_measures'],
        best_pair['x'],
        best_pair['y']
    )
    
    # Afficher les 5 meilleures paires
    print("\nTop 5 des paires de variables les plus corrélées:")
    sorted_results = sorted(results, key=lambda x: abs(x['dependence_measures']['kendall_tau']), reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. {result['var1']} - {result['var2']}")
        print(f"   Kendall's tau: {result['dependence_measures']['kendall_tau']:.4f}")
        print(f"   Spearman's rho: {result['dependence_measures']['spearman_rho']:.4f}")
        print(f"   Pearson correlation: {result['dependence_measures']['pearson']:.4f}\n")

if __name__ == "__main__":
    main() 