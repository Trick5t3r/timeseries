import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

# Ignore les warnings pour un affichage plus propre
warnings.filterwarnings("ignore")

def plot_raw_data(data, title=None):
    """Affiche les données brutes avant l'analyse ARIMA."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data.values, marker='o', linestyle='-', linewidth=2, markersize=8)
    if title is None:
        title = 'Évolution du nombre total d\'émigrants dans le monde'
    plt.title(title)
    plt.xlabel('Année')
    plt.ylabel('Nombre d\'émigrants')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ajouter les valeurs sur les points
    for x, y in zip(data.index, data.values):
        plt.annotate(f'{int(y):,}', 
                    (x, y),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.tight_layout()
    plt.show()

def load_and_prepare_data(file_path):
    """Charge et prépare les données pour l'analyse."""
    print("Chargement des données...")
    df = pd.read_csv(file_path)
    
    # Conversion de la colonne Year en format datetime
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    
    # Supprimer les lignes avec des dates manquantes
    df = df.dropna(subset=['Year'])
    
    # Agrégation par année pour le monde entier
    world_data = df[df['Entity'] == 'World'].copy()
    world_data.set_index('Year', inplace=True)
    
    # Utiliser la colonne 'Total number of emigrants' comme série temporelle
    time_series = world_data['Total number of emigrants']
    
    return time_series

def grid_search_arima(data, p_range, d_range, q_range):
    """Effectue une recherche par grille pour trouver les meilleurs paramètres ARIMA."""
    print("Recherche des meilleurs paramètres ARIMA...")
    best_aic = float('inf')
    best_params = None
    best_model = None
    
    for p, d, q in product(p_range, d_range, q_range):
        try:
            model = ARIMA(data, order=(p, d, q))
            model_fit = model.fit()
            
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_params = (p, d, q)
                best_model = model_fit
                
        except:
            continue
    
    print(f"Meilleurs paramètres ARIMA: {best_params}")
    print(f"AIC: {best_aic}")
    return best_model, best_params

def plot_results(data, model_fit, test_data=None):
    """Affiche les résultats et les diagnostics du modèle."""
    # Prédictions sur les données d'entraînement
    forecast = model_fit.get_prediction(start=data.index[0], end=data.index[-1])
    forecast_mean = forecast.predicted_mean
    
    # Prédictions futures
    future_steps = 10  # 10 années
    future_forecast = model_fit.get_forecast(steps=future_steps)
    future_forecast_mean = future_forecast.predicted_mean
    
    # Création des sous-graphiques
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Données et prédictions
    plt.subplot(2, 2, 1)
    plt.plot(data, label='Données d\'entraînement', color='blue')
    if test_data is not None:
        plt.plot(test_data, label='Données de test', color='green')
    plt.plot(forecast_mean.index, forecast_mean, label='Prévisions ARIMA', color='red')
    plt.plot(future_forecast_mean.index, future_forecast_mean, label='Prévisions futures', color='orange')
    plt.title('Nombre total d\'émigrants dans le monde')
    plt.xlabel('Année')
    plt.ylabel('Nombre d\'émigrants')
    plt.legend()
    
    # 2. Résidus
    plt.subplot(2, 2, 2)
    residuals = model_fit.resid
    plt.plot(residuals, label='Résidus')
    plt.title('Résidus du modèle')
    plt.legend()
    
    # 3. Distribution des résidus
    plt.subplot(2, 2, 3)
    sns.histplot(residuals, kde=True)
    plt.title('Distribution des résidus')
    
    # 4. QQ-Plot
    plt.subplot(2, 2, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    
    plt.tight_layout()
    plt.show()

def plot_test_predictions(train_data, test_data, forecast_mean):
    """Affiche un graphique détaillé des prédictions sur l'ensemble de test."""
    plt.figure(figsize=(12, 6))
    
    # Plot des données d'entraînement
    plt.plot(train_data.index, train_data.values, 
             marker='o', color='blue', label='Données d\'entraînement')
    
    # Plot des données de test
    plt.plot(test_data.index, test_data.values, 
             marker='o', color='green', label='Données de test réelles')
    
    # Plot des prédictions
    plt.plot(forecast_mean.index, forecast_mean.values, 
             marker='o', color='red', label='Prévisions ARIMA')
    
    # Ajouter les valeurs sur les points de test et prédictions
    for x, y in zip(test_data.index, test_data.values):
        plt.annotate(f'{int(y):,}', 
                    (x, y),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    color='green')
    
    for x, y in zip(forecast_mean.index, forecast_mean.values):
        plt.annotate(f'{int(y):,}', 
                    (x, y),
                    textcoords="offset points",
                    xytext=(0,-15),
                    ha='center',
                    color='red')
    
    plt.title('Prévisions ARIMA sur l\'ensemble de test')
    plt.xlabel('Année')
    plt.ylabel('Nombre d\'émigrants')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_predictions(actual, predicted):
    """Évalue les performances du modèle avec différentes métriques."""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    print("\nMétriques de performance sur l'ensemble de test:")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE: {mae:,.2f}")
    print(f"R²: {r2:.4f}")
    
    # Calculer l'erreur en pourcentage
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"MAPE: {mape:.2f}%")
    
    # Afficher les valeurs réelles vs prédites
    print("\nComparaison des valeurs réelles vs prédites:")
    for year, actual_val, pred_val in zip(actual.index, actual.values, predicted.values):
        error = abs(actual_val - pred_val)
        error_pct = (error / actual_val) * 100
        print(f"Année {year.year}:")
        print(f"  Réel: {int(actual_val):,}")
        print(f"  Prédit: {int(pred_val):,}")
        print(f"  Erreur absolue: {int(error):,}")
        print(f"  Erreur relative: {error_pct:.2f}%")
        print()

def main():
    # Chargement et préparation des données
    data = load_and_prepare_data('data/total-number-of-emigrants.csv')
    
    # Affichage des données brutes
    print("\nAffichage des données brutes...")
    plot_raw_data(data)
    
    # Division des données en ensemble d'entraînement et de test
    # On utilise les données jusqu'en 2015 pour l'entraînement
    train_data = data[:'2015']
    test_data = data['2015':]
    
    print("\nAffichage des données d'entraînement et de test...")
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, marker='o', label='Données d\'entraînement', color='blue')
    plt.plot(test_data.index, test_data.values, marker='o', label='Données de test', color='green')
    plt.title('Division des données en ensembles d\'entraînement et de test')
    plt.xlabel('Année')
    plt.ylabel('Nombre d\'émigrants')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Définition des paramètres pour la recherche par grille
    p_range = range(0, 4)
    d_range = range(1, 3)
    q_range = range(0, 4)
    
    # Recherche des meilleurs paramètres sur l'ensemble d'entraînement
    best_model, best_params = grid_search_arima(train_data, p_range, d_range, q_range)
    
    # Prédictions sur l'ensemble de test
    forecast = best_model.get_forecast(steps=len(test_data))
    forecast_mean = forecast.predicted_mean
    
    # Affichage des résultats
    plot_results(train_data, best_model, test_data)
    
    # Affichage spécifique des prédictions sur l'ensemble de test
    print("\nAffichage des prédictions sur l'ensemble de test...")
    plot_test_predictions(train_data, test_data, forecast_mean)
    
    # Évaluation des prédictions
    evaluate_predictions(test_data, forecast_mean)

if __name__ == "__main__":
    main() 