import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import warnings

warnings.filterwarnings('ignore')

################################################################################
# 1) CHARGEMENT ET PRÉPARATION DES DONNÉES
################################################################################

def load_and_prepare_data(file_path):
    """Charge et prépare les données pour l'analyse de la copule d'extrême valeur."""
    print("Chargement des données...")
    df = pd.read_csv(file_path)
    
    # Conversion en datetime
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Filtrage d'une plage de dates (exemple)
    mask = (df['DateTime'] >= '2023-01-01') & (df['DateTime'] <= '2023-05-31')
    df = df[mask]
    
    print(f"Nombre de données après filtrage : {len(df)}")
    
    # Conversion de Fahrenheit en Celsius
    df['Temperature'] = df['Temperature'].str.replace(' °F', '').astype(float)
    df['Temperature'] = (df['Temperature'] - 32) * 5.0/9.0
    
    df['Humidity'] = df['Humidity'].str.replace(' %', '').astype(float)
    
    # Conserver uniquement les colonnes utiles
    data = df[['Temperature', 'Humidity']].dropna()
    
    return data

################################################################################
# 2) FONCTIONS POUR LA COPULE DE GUMBEL
################################################################################

def gumbel_cdf(u, v, theta):
    """
    CDF de la copule de Gumbel.
    C(u,v) = exp(- ( (-ln u)^theta + (-ln v)^theta )^(1/theta ))
    """
    # Éviter les valeurs 0 ou 1 strictes pour le log
    u = np.clip(u, 1e-12, 1-1e-12)
    v = np.clip(v, 1e-12, 1-1e-12)
    inner = ((-np.log(u))**theta + (-np.log(v))**theta)**(1.0/theta)
    return np.exp(-inner)

def gumbel_pdf(u, v, theta):
    """
    PDF (densité) de la copule de Gumbel.
    Formule disponible dans la littérature (Joe, 1997).
    """
    # Clip pour éviter log(0)
    u = np.clip(u, 1e-12, 1-1e-12)
    v = np.clip(v, 1e-12, 1-1e-12)
    
    # (-ln(u))^theta + (-ln(v))^theta
    log_u = -np.log(u)
    log_v = -np.log(v)
    A = (log_u**theta + log_v**theta)**(1/theta)
    
    # Facteur principal
    c = gumbel_cdf(u, v, theta)
    
    # Composants pour la densité
    term1 = (A**(theta) * (theta - 1)) / (u * v)
    term2 = (log_u*log_v)**(theta - 1)
    term3 = (log_u**theta + log_v**theta)**((2.0 - theta)/theta)
    
    # Densité
    pdf_val = c * term1 * term2 / term3
    return pdf_val

def negative_log_likelihood(theta, u, v):
    """
    Log-vraisemblance négative pour la copule de Gumbel.
    On somme -log(pdf).
    """
    pdf_vals = gumbel_pdf(u, v, theta)
    # Éviter les -log(0) -> inf
    pdf_vals = np.clip(pdf_vals, 1e-14, None)
    return -np.sum(np.log(pdf_vals))

def fit_gumbel_copula(u, v, init_theta=2.0):
    """
    Estimation MLE (Maximum de Vraisemblance) du paramètre theta
    pour la copule de Gumbel.
    """
    bounds = [(1.0, 20.0)]  # La Gumbel copula a theta >= 1
    result = optimize.minimize(
        negative_log_likelihood,
        x0=np.array([init_theta]),
        args=(u, v),
        bounds=bounds,
        method='L-BFGS-B'
    )
    theta_hat = result.x[0]
    return theta_hat, result

################################################################################
# 3) FILTRAGE DES EXTREMES & TRANSFORMATION EN PSEUDO-OBSERVATIONS
################################################################################

def extract_extremes(data, variable='Temperature', quantile=0.95):
    """
    Extrait les données supérieures (ou inférieures) à un certain quantile
    pour se concentrer sur les valeurs extrêmes.
    Par défaut, on prend le quantile 0.95 pour la queue supérieure.
    
    Retourne un DataFrame filtré + le masque binaire utilisé.
    """
    threshold = data[variable].quantile(quantile)
    mask = (data[variable] >= threshold)
    # Si on voulait la queue inférieure :  data[variable] <= threshold
    return data[mask], mask

def to_pseudo_observations(x):
    """
    Convertit une série x en pseudo-observations U \in (0,1)
    via rank / (n+1).
    """
    ranks = stats.rankdata(x, method='ordinal')
    return ranks / (len(x) + 1.0)

################################################################################
# 4) SCRIPT PRINCIPAL DE MODÉLISATION PAR COPULE EXTREME
################################################################################

def main():
    # 1) Chargement et préparation du jeu de données
    data = load_and_prepare_data('data/weather_data.csv')
    print("Variables disponibles :", data.columns)
    
    # 2) Exemple : étudier la dépendance extrême entre Temperature et Humidity
    var1, var2 = 'Temperature', 'Humidity'
    
    # 3) Filtrer sur les valeurs extrêmes (e.g. Temperature >= quantile 0.95)
    data_extreme, mask = extract_extremes(data, variable=var1, quantile=0.95)
    print(f"Nombre de points extrêmes (sur {var1}) : {len(data_extreme)}")
    
    # 4) Transformer en pseudo-observations
    u = to_pseudo_observations(data_extreme[var1].values)
    v = to_pseudo_observations(data_extreme[var2].values)
    
    # 5) Estimation MLE de la copule de Gumbel sur ces extrêmes
    theta_hat, opt_result = fit_gumbel_copula(u, v, init_theta=2.0)
    print("\n=== Résultats de l'estimation (copule Gumbel) ===")
    print(f"Paramètre estimé theta = {theta_hat:.4f}")
    print(f"Statut de l'optimisation : {opt_result.message}")
    
    # 6) Visualisation simple : Nuage (u,v) et contours de la copule ajustée
    #    Sur l'espace [0,1]^2 des pseudo-observations
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(u, v, alpha=0.5, label='Pseudo-observations extrêmes')
    
    # Grille pour iso-courbes
    grid = np.linspace(0.01, 0.99, 50)
    Ugrid, Vgrid = np.meshgrid(grid, grid)
    Z = gumbel_pdf(Ugrid.ravel(), Vgrid.ravel(), theta_hat)
    Z = Z.reshape(Ugrid.shape)
    contour = ax.contour(Ugrid, Vgrid, Z, levels=5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    ax.set_xlabel(f'U (pseudo-obs {var1})')
    ax.set_ylabel(f'V (pseudo-obs {var2})')
    ax.set_title("Copule Gumbel sur les extrêmes")
    ax.legend()
    plt.tight_layout()
    plt.savefig('./results/weather_copula_extreme_gumble.png')
    plt.show()

if __name__ == '__main__':
    main()
