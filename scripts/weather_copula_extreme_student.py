import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.special import gamma  # Importation de la fonction gamma
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
# 2) FONCTIONS POUR LA COPULE t DE STUDENT
################################################################################

def t_copula_pdf(u, v, rho, nu):
    """
    Densité de la copule t de Student.
    Pour u,v dans (0,1), on calcule les quantiles t :
      x = t.ppf(u, df=nu) et y = t.ppf(v, df=nu)
    La densité de la copule est donnée par :
      c(u,v) = f_{xy}(x,y) / [f(x) f(y)]
    où f_{xy} est la densité jointe bivariée t et f(x), f(y) les densités marginales.
    """
    # Éviter les problèmes numériques
    u = np.clip(u, 1e-12, 1-1e-12)
    v = np.clip(v, 1e-12, 1-1e-12)
    
    # Calcul des quantiles pour la t-Student
    x = stats.t.ppf(u, df=nu)
    y = stats.t.ppf(v, df=nu)
    
    # Densités marginales t
    fx = stats.t.pdf(x, df=nu)
    fy = stats.t.pdf(y, df=nu)
    
    # Densité jointe bivariée t
    det = 1 - rho**2
    numerator = gamma((nu+2)/2)
    denominator = gamma(nu/2) * np.pi * nu * np.sqrt(det)
    quad_form = (x**2 - 2*rho*x*y + y**2) / (nu * det)
    fxy = (numerator / denominator) * (1 + quad_form)**(-((nu+2)/2))
    
    # Densité de la copule t
    c = fxy / (fx * fy)
    return c

def negative_log_likelihood_t(params, u, v):
    """
    Log-vraisemblance négative pour la copule t.
    params[0] = rho (corrélation), params[1] = nu (degrés de liberté)
    """
    rho, nu = params
    # Calcul de la densité de la copule t pour chaque observation
    pdf_vals = t_copula_pdf(u, v, rho, nu)
    # Éviter log(0)
    pdf_vals = np.clip(pdf_vals, 1e-14, None)
    return -np.sum(np.log(pdf_vals))

def fit_t_copula(u, v, init_params=[-0.5, 5.0]):
    """
    Estimation MLE pour la copule t de Student.
    On estime rho et nu.
    """
    bounds = [(-0.99, 0.99), (2.1, 100)]  # nu > 2 pour assurer une variance finie
    result = optimize.minimize(
        negative_log_likelihood_t,
        x0=np.array(init_params),
        args=(u, v),
        bounds=bounds,
        method='L-BFGS-B'
    )
    rho_hat, nu_hat = result.x
    return (rho_hat, nu_hat), result

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
    # Pour la queue inférieure, utiliser : data[variable] <= threshold
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
    
    # 5) Estimation MLE de la copule t sur ces extrêmes
    (rho_hat, nu_hat), opt_result = fit_t_copula(u, v, init_params=[-0.5, 5.0])
    print("\n=== Résultats de l'estimation (copule t de Student) ===")
    print(f"Paramètre estimé rho = {rho_hat:.4f}")
    print(f"Degrés de liberté estimés nu = {nu_hat:.4f}")
    print(f"Statut de l'optimisation : {opt_result.message}")
    
    # 6) Visualisation simple : Nuage (u,v) et contours de la copule ajustée
    #    Sur l'espace [0,1]^2 des pseudo-observations
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(u, v, alpha=0.5, label='Pseudo-observations extrêmes')
    
    # Grille pour iso-courbes
    grid = np.linspace(0.01, 0.99, 50)
    Ugrid, Vgrid = np.meshgrid(grid, grid)
    Z = t_copula_pdf(Ugrid.ravel(), Vgrid.ravel(), rho_hat, nu_hat)
    Z = Z.reshape(Ugrid.shape)
    contour = ax.contour(Ugrid, Vgrid, Z, levels=5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    ax.set_xlabel(f'U (pseudo-obs {var1})')
    ax.set_ylabel(f'V (pseudo-obs {var2})')
    ax.set_title("Copule t de Student sur les extrêmes")
    ax.legend()
    plt.tight_layout()
    plt.savefig('./results/weather_copula_extreme_t.png')
    plt.show()

if __name__ == '__main__':
    main()
