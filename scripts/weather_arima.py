import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from arch import arch_model
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

# Ignore les warnings pour un affichage plus propre
warnings.filterwarnings("ignore")

def plot_raw_data(data, title=None):
    """Affiche les données brutes avant l'analyse ARIMA."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data.values, marker='o', linestyle='-', linewidth=2, markersize=4)
    if title is None:
        title = 'Évolution de la température'
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Température (°F)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def load_and_prepare_data(file_path):
    """Charge et prépare les données pour l'analyse."""
    print("Chargement des données...")
    df = pd.read_csv(file_path)
    
    # Conversion des colonnes Date et Time en datetime
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Extraction de la température et conversion en numérique
    df['Temperature'] = df['Temperature'].str.replace(' °F', '').astype(float)
    
    # Utiliser DateTime comme index
    time_series = df.set_index('DateTime')['Temperature']
    
    # Resampling par heure pour avoir des données régulières
    time_series = time_series.resample('H').mean()
    
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
    future_steps = 24  # 24 heures
    future_forecast = model_fit.get_forecast(steps=future_steps)
    future_forecast_mean = future_forecast.predicted_mean
    
    # Création des sous-graphiques
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Données et prédictions
    plt.subplot(2, 2, 1)
    plt.plot(data, label='Données d\'entraînement', color='blue', alpha=0.6)
    if test_data is not None:
        plt.plot(test_data, label='Données de test', color='green', alpha=0.6)
    plt.plot(forecast_mean.index, forecast_mean, label='Prévisions ARIMA', color='red')
    plt.plot(future_forecast_mean.index, future_forecast_mean, label='Prévisions futures', color='orange')
    plt.title('Température et prévisions')
    plt.xlabel('Date')
    plt.ylabel('Température (°F)')
    plt.legend()
    
    # 2. Résidus normalisés
    plt.subplot(2, 2, 2)
    residuals = model_fit.resid
    normalized_residuals = (residuals - residuals.mean()) / residuals.std()
    plt.plot(normalized_residuals, label='Résidus normalisés')
    plt.title('Résidus normalisés du modèle')
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
             marker='o', color='blue', label='Données d\'entraînement', alpha=0.6)
    
    # Plot des données de test
    plt.plot(test_data.index, test_data.values, 
             marker='o', color='green', label='Données de test réelles', alpha=0.6)
    
    # Plot des prédictions
    plt.plot(forecast_mean.index, forecast_mean.values, 
             marker='o', color='red', label='Prévisions ARIMA')
    
    plt.title('Prévisions ARIMA sur l\'ensemble de test')
    plt.xlabel('Date')
    plt.ylabel('Température (°F)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def evaluate_predictions(actual, predicted):
    """Évalue les performances du modèle avec différentes métriques."""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    print("\nMétriques de performance sur l'ensemble de test:")
    print(f"RMSE: {rmse:.2f}°F")
    print(f"MAE: {mae:.2f}°F")
    print(f"R²: {r2:.4f}")
    
    # Calculer l'erreur en pourcentage
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"MAPE: {mape:.2f}%")
    
    # Afficher les valeurs réelles vs prédites
    print("\nComparaison des valeurs réelles vs prédites:")
    for date, actual_val, pred_val in zip(actual.index, actual.values, predicted.values):
        error = abs(actual_val - pred_val)
        error_pct = (error / actual_val) * 100
        print(f"Date {date}:")
        print(f"  Réel: {actual_val:.1f}°F")
        print(f"  Prédit: {pred_val:.1f}°F")
        print(f"  Erreur absolue: {error:.1f}°F")
        print(f"  Erreur relative: {error_pct:.2f}%")
        print()

def fit_arma_garch(data, p=1, q=1, garch_p=1, garch_q=1):
    """Fit an ARMA-GARCH model to the data."""
    print("\nFitting ARMA-GARCH model...")
    
    # Nettoyer les données en supprimant NaN et les valeurs infinies
    clean_data = data.copy()
    clean_data = clean_data.replace([np.inf, -np.inf], np.nan)
    clean_data = clean_data.dropna()
    
    if len(clean_data) == 0:
        raise ValueError("No valid data points after cleaning")
    
    print(f"Using {len(clean_data)} data points after cleaning")
    
    # Ajuster le modèle
    model = arch_model(clean_data, vol='Garch', p=garch_p, q=garch_q, mean='AR', lags=p)
    results = model.fit(disp='off')
    
    print("\nARMA-GARCH Model Summary:")
    print(results.summary())
    
    return results

def plot_arma_garch_results(data, results, test_data=None):
    """Plot results from ARMA-GARCH model."""
    # Nettoyer les données pour le plot
    clean_data = data.copy()
    clean_data = clean_data.replace([np.inf, -np.inf], np.nan)
    clean_data = clean_data.dropna()
    
    if test_data is not None:
        clean_test = test_data.copy()
        clean_test = clean_test.replace([np.inf, -np.inf], np.nan)
        clean_test = clean_test.dropna()
    else:
        clean_test = None
    
    # Obtenir les prévisions (on retire la première ligne qui correspond à la dernière observation d'entraînement)
    horizon = len(clean_test) if clean_test is not None else 24
    forecast = results.forecast(horizon=horizon)
    mean_forecast = forecast.mean.iloc[1:, 0]       # suppression de la première ligne
    variance_forecast = forecast.variance.iloc[1:, 0] # suppression de la première ligne
    
    # Réajuster l'index des prévisions pour correspondre aux données de test
    if clean_test is not None:
        mean_forecast.index = clean_test.index[:len(mean_forecast)]
        variance_forecast.index = clean_test.index[:len(variance_forecast)]
    else:
        # Pour des prévisions futures, créer un nouvel index à partir de la dernière date des données
        last_date = clean_data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(mean_forecast)+1, freq='H')[1:]
        mean_forecast.index = future_dates
        variance_forecast.index = future_dates
    
    # Obtenir les résidus
    residuals = pd.Series(results.resid.squeeze())
    
    # Création de la figure avec sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Prévisions de la moyenne avec intervalles de confiance
    axes[0, 0].plot(clean_data.index, clean_data.values, label='Données d\'entraînement', color='blue', alpha=0.6)
    if clean_test is not None:
        axes[0, 0].plot(clean_test.index, clean_test.values, label='Données de test', color='green', alpha=0.6)
    
    # Plot de la prévision moyenne
    axes[0, 0].plot(mean_forecast.index, mean_forecast.values, label='Prévisions ARMA-GARCH', color='red')
    
    # Plot des intervalles de confiance (±2 écarts-types)
    std_dev = np.sqrt(variance_forecast)
    axes[0, 0].fill_between(mean_forecast.index,
                           mean_forecast - 2*std_dev,
                           mean_forecast + 2*std_dev,
                           color='red', alpha=0.2, label='Intervalle de confiance (±2σ)')
    
    axes[0, 0].set_title('Prévisions de la moyenne avec intervalles de confiance')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Température (°F)')
    axes[0, 0].legend()
    
    # 2. Diagramme en violon des résidus
    sns.violinplot(y=residuals, ax=axes[0, 1], color='skyblue')
    axes[0, 1].set_title("Diagramme de Violon des Résidus")
    axes[0, 1].set_ylabel("Résidus")
    
    # 3. ACF des résidus
    plot_acf(residuals, ax=axes[1, 0])
    axes[1, 0].set_title("Résidus Autocorrélés (ACF)")
    
    # 4. QQ-plot des résidus
    sm.qqplot(residuals, line='s', ax=axes[1, 1])
    axes[1, 1].set_title("QQ-Plot des Résidus")
    
    plt.tight_layout()
    plt.show()

def main():
    # Chargement et préparation des données
    data = load_and_prepare_data('data/weather_data.csv')
    
    # Affichage des données brutes
    print("\nAffichage des données brutes...")
    plot_raw_data(data)
    
    # Division des données en ensemble d'entraînement et de test
    # On utilise les données jusqu'à la dernière semaine pour l'entraînement
    train_data = data[:-24*7]  # Toutes les données sauf la dernière semaine
    test_data = data[-24*7:]   # Dernière semaine
    
    print("\nAffichage des données d'entraînement et de test...")
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, marker='o', label='Données d\'entraînement', color='blue', alpha=0.6)
    plt.plot(test_data.index, test_data.values, marker='o', label='Données de test', color='green', alpha=0.6)
    plt.title('Division des données en ensembles d\'entraînement et de test')
    plt.xlabel('Date')
    plt.ylabel('Température (°F)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Définition des paramètres pour la recherche par grille
    p_range = range(0, 4)
    d_range = range(1, 3)
    q_range = range(0, 4)
    
    # Recherche des meilleurs paramètres sur l'ensemble d'entraînement
    best_model, best_params = grid_search_arima(train_data, p_range, d_range, q_range)
    
    # Prédictions sur l'ensemble de test avec ARIMA
    forecast = best_model.get_forecast(steps=len(test_data))
    forecast_mean = forecast.predicted_mean
    
    # Affichage des résultats ARIMA
    plot_results(train_data, best_model, test_data)
    
    # Affichage spécifique des prédictions sur l'ensemble de test
    print("\nAffichage des prédictions sur l'ensemble de test (ARIMA)...")
    plot_test_predictions(train_data, test_data, forecast_mean)
    
    # Évaluation des prédictions ARIMA
    evaluate_predictions(test_data, forecast_mean)
    
    # Analyse ARMA-GARCH
    print("\n=== Analyse ARMA-GARCH ===")
    arma_garch_results = fit_arma_garch(train_data, p=1, q=1, garch_p=1, garch_q=1)
    plot_arma_garch_results(train_data, arma_garch_results, test_data)
    
    # Évaluation des prédictions ARMA-GARCH
    # Remarque : on retire la première ligne des prévisions qui correspond à la dernière observation d'entraînement
    forecast_ag = arma_garch_results.forecast(horizon=len(test_data))
    mean_forecast_ag = forecast_ag.mean.iloc[1:, 0]
    
    # Réassigner l'index pour que ça corresponde aux données de test
    mean_forecast_ag.index = test_data.index[:len(mean_forecast_ag)]
    
    print("\nÉvaluation des prédictions ARMA-GARCH:")
    evaluate_predictions(test_data[:len(mean_forecast_ag)], mean_forecast_ag)

if __name__ == "__main__":
    main()
