import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Liste des entités à ignorer dans l'analyse
IGNORED_ENTITIES_LIST = [
    'Other',  # Catégorie générique
    'Taiwan',  # Données manquantes
    'Saint Martin (French part)',  # Données manquantes
    'Saint Barthelemy',  # Données manquantes
    'Bonaire Sint Eustatius and Saba',  # Données manquantes
    'Curacao',  # Données manquantes
    'Sint Maarten (Dutch part)',  # Données manquantes
    'Vatican',  # Données non représentatives
    'Micronesia (country)',  # Données redondantes
    'Oceania (excluding Australia and New Zealand)',  # Données redondantes
    'Less developed regions, excluding China',  # Données redondantes
    'Less developed regions, excluding least developed countries',  # Données redondantes
    'Developed regions',  # Données redondantes
    'High-income countries',  # Données redondantes
    'Middle-income countries',  # Données redondantes
    'Low-income countries',  # Données redondantes
    'Upper-middle-income countries',  # Données redondantes
    'Lower-middle-income countries',  # Données redondantes
    'Land-locked Developing Countries (LLDC)',  # Données redondantes
    'Small island developing States (SIDS)',  # Données redondantes
    'Least developed countries',  # Données redondantes
    'Europe',  # Catégorie agrégée
    'World'  # Catégorie agrégée
]

def load_and_prepare_data(file_path):
    """Charge et prépare les données pour l'analyse des copules."""
    print("Chargement des données...")
    df = pd.read_csv(file_path)
    
    # Conversion de la colonne Year en format datetime
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    
    # Supprimer les lignes avec des dates manquantes
    df = df.dropna(subset=['Year'])
    
    # Filtrer les entités à ignorer
    df = df[~df['Entity'].isin(IGNORED_ENTITIES_LIST)]
    
    # Pivoter les données pour avoir les années en index et les entités en colonnes
    pivot_df = df.pivot(index='Year', columns='Entity', values='Total number of emigrants')
    
    return pivot_df

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

def plot_copula(copula, entity1, entity2, dependence_measures, x, y):
    """Trace la copule et les mesures de dépendance."""
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Copule
    plt.subplot(1, 3, 1)
    plt.imshow(copula, cmap='viridis', origin='lower')
    plt.colorbar(label='Valeur de la copule')
    plt.title(f'Copule entre {entity1} et {entity2}')
    plt.xlabel(entity2)
    plt.ylabel(entity1)
    
    # 2. Nuage de points
    plt.subplot(1, 3, 2)
    plt.scatter(x, y, alpha=0.5)
    plt.title('Nuage de points')
    plt.xlabel(f'Nombre d\'émigrants - {entity2}')
    plt.ylabel(f'Nombre d\'émigrants - {entity1}')
    
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
    data = load_and_prepare_data('data/total-number-of-emigrants.csv')
    
    # Liste des entités à analyser (régions principales)
    entities = [
        'Asia', 'Africa', 'Northern America',
        'Latin America and the Caribbean', 'Oceania'
    ]
    
    # Filtrer les données pour ne garder que les entités sélectionnées
    data = data[entities]
    
    # Calculer toutes les paires possibles
    pairs = list(combinations(entities, 2))
    
    # Stocker les résultats
    results = []
    
    print("Analyse des copules entre les différentes régions...")
    for entity1, entity2 in pairs:
        x = data[entity1].values
        y = data[entity2].values
        
        # Calculer la copule
        copula = calculate_copula(x, y)
        
        # Calculer les mesures de dépendance
        dependence_measures = calculate_dependence_measures(x, y)
        
        # Stocker les résultats
        results.append({
            'entity1': entity1,
            'entity2': entity2,
            'copula': copula,
            'dependence_measures': dependence_measures,
            'x': x,
            'y': y
        })
    
    # Trouver la meilleure paire (basée sur la valeur absolue de Kendall's tau)
    best_pair = max(results, key=lambda x: abs(x['dependence_measures']['kendall_tau']))
    
    print("\nMeilleure paire de régions trouvée:")
    print(f"{best_pair['entity1']} - {best_pair['entity2']}")
    print("\nMesures de dépendance:")
    for measure, value in best_pair['dependence_measures'].items():
        print(f"{measure}: {value:.4f}")
    
    # Tracer la copule pour la meilleure paire
    plot_copula(
        best_pair['copula'],
        best_pair['entity1'],
        best_pair['entity2'],
        best_pair['dependence_measures'],
        best_pair['x'],
        best_pair['y']
    )
    
    # Afficher les 5 meilleures paires
    print("\nTop 5 des paires de régions les plus corrélées:")
    sorted_results = sorted(results, key=lambda x: abs(x['dependence_measures']['kendall_tau']), reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. {result['entity1']} - {result['entity2']}")
        print(f"   Kendall's tau: {result['dependence_measures']['kendall_tau']:.4f}")
        print(f"   Spearman's rho: {result['dependence_measures']['spearman_rho']:.4f}")
        print(f"   Pearson correlation: {result['dependence_measures']['pearson']:.4f}\n")

if __name__ == "__main__":
    main() 