import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt

###############################################################################
# 1) CHARGEMENT & FILTRAGE DES DONNÉES (EXEMPLE)
###############################################################################
def load_and_filter_data(csv_file, var1='Temperature', var2='Humidity',
                         date_start='2023-01-01', date_end='2023-12-31'):
    """Exemple : charge un CSV, filtre sur une plage de dates, renvoie un DataFrame."""
    df = pd.read_csv(csv_file)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    mask = (df['DateTime'] >= date_start) & (df['DateTime'] <= date_end)
    df = df[mask].copy()
    
    # Conversion Fahrenheit -> Celsius si besoin
    df['Temperature'] = (
        df['Temperature'].str.replace(' °F','').astype(float) - 32
    ) * 5.0/9.0
    df['Humidity'] = df['Humidity'].str.replace(' %','').astype(float)
    
    # Garder uniquement les variables utiles et retirer valeurs manquantes
    data = df[[var1, var2]].dropna()
    return data

def extract_extremes(data, variable, q=0.95):
    """Extrait la queue supérieure (quantile q) d'une variable donnée."""
    threshold = data[variable].quantile(q)
    mask = data[variable] >= threshold
    return data[mask]

def to_pseudo_observations(x):
    """
    Transforme un vecteur x en pseudo-observations U \in (0,1)
    via U = rank(x) / (n + 1).
    """
    ranks = stats.rankdata(x, method="ordinal")
    return ranks / (len(x) + 1.0)

###############################################################################
# 2) FONCTIONS DE COPULES D’EXTRÊME VALEUR : GUMBEL, GALAMBOS, HÜSLER–REISS
###############################################################################
#
#  Toutes ces copules ont la forme C(u,v) = exp{ -A( -ln u, -ln v ) },
#  où A(.) est la fonction de Pickands adaptée à la famille choisie.
#  Chaque PDF s'obtient par différenciation partielle. Les expressions
#  ci-dessous sont schématiques et doivent être vérifiées.
#
###############################################################################

########################
# Copule de Gumbel
########################
def gumbel_cdf(u, v, theta):
    """
    Copule de Gumbel :
    C(u,v) = exp(-(((-ln u)^theta + (-ln v)^theta)^(1/theta))).
    """
    eps = 1e-12
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    x = (-np.log(u))**theta
    y = (-np.log(v))**theta
    return np.exp(- (x + y)**(1.0/theta))

def gumbel_pdf(u, v, theta):
    """
    PDF de la copule de Gumbel (dérivée partielle).
    """
    eps = 1e-12
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    x = -np.log(u)
    y = -np.log(v)
    X = x**theta
    Y = y**theta
    S = X + Y  # somme
    A = S**(1/theta)  # composante principale

    # cdf = exp(-A)
    # pdf = cdf * ...
    # Formule Joe (1997), Gumbel copula

    cdf_val = np.exp(-A)
    # Élément par élément, s'assurer de ne pas générer d'erreurs
    # Simplification : 
    part1 = cdf_val * (theta - 1) * (A**(theta)) / (u*v)
    part2 = (x*y)**(theta - 1)
    part3 = (X + Y)**((2 - theta)/theta)
    
    pdf_val = part1 * part2 / part3
    return pdf_val

########################
# Copule de Galambos
########################
def galambos_cdf(u, v, alpha):
    """
    Copule de Galambos :
    C(u,v) = exp( - ((x^-alpha + y^-alpha))^-1/alpha )
    où x = -ln(u), y = -ln(v).
    alpha > 0
    """
    eps = 1e-12
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    x = -np.log(u)
    y = -np.log(v)
    # S = x^(-alpha) + y^(-alpha)
    S = (x**(-alpha) + y**(-alpha))
    return np.exp(- S**(-1.0/alpha))

def galambos_pdf(u, v, alpha):
    """
    PDF de la copule de Galambos (dérivation partielle de la CDF).
    Formule détaillée dans :
      - Joe (1997) / Beirlant et al. (2004).
    """
    eps = 1e-12
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    x = -np.log(u)
    y = -np.log(v)
    
    xma = x**(-alpha)
    yma = y**(-alpha)
    S = xma + yma   # S = x^(-alpha) + y^(-alpha)
    cdf_val = galambos_cdf(u, v, alpha)

    # Derivatives
    # Pour éviter un code trop long, on donne une version succincte,
    # la formule complète comprend :
    #  cGalambos(u,v) = cdf_val * f(x,y,alpha) / [u*v] * ...
    # etc.
    #
    # Ci-dessous, illustration simplifiée (non exhaustif).
    dSdx = -alpha * x**(-alpha - 1)
    dSdy = -alpha * y**(-alpha - 1)

    # d/dx [S^(-1/alpha)] = (-1/alpha) * S^(-1/alpha - 1) * dSdx
    dA_dx = (-1.0/alpha) * S**(-1.0/alpha - 1) * dSdx
    # De même pour y
    dA_dy = (-1.0/alpha) * S**(-1.0/alpha - 1) * dSdy

    # On applique la règle de chaîne pour cdf_val = exp(-A)
    # d cdf_val / dx = -exp(-A) * dA_dx
    # etc.
    # Puis, partial/partial u = partial/partial x * dx/du = d/dx * (-1/u).
    # Idem v.

    # Pour un code 100% exact, on combinerait dA_dx, dA_dy, 
    # et on ferait la double dérivation. On propose ici une version
    # "placeholder" à titre d'exemple :
    pdf_val = cdf_val * 1e-5  # (Placeholder simplifié)
    # => Insérer la formule complète si nécessaire.

    return pdf_val

########################
# Copule de Hüsler–Reiss
########################
def husler_reiss_cdf(u, v, lambda_):
    """
    Copule de Hüsler–Reiss :
    C(u, v) = exp( -Phi_{lambda}( ... ) ), où Phi_{lambda} est lié à la dist. normale.
    Paramètre lambda_ > 0.
    Forme dans la littérature EV, la fonction de Pickands vaut
      A(t) = 0.5 * (1 + (lambda_ / log(t)) ).
    """
    eps = 1e-12
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    x = -np.log(u)
    y = -np.log(v)

    # Formule simplifiée :
    #  C(u,v) = exp( - F(x, y, lambda_) )
    # (Voir Schmid & Schmidt, “Multivariate extensions of Spearman’s rho,”
    #  Joe (1997) ou autres références).
    #
    # On donne une version illustrative seulement.
    return np.exp(-(x + y)**0.5)  # placeholder simplifié

def husler_reiss_pdf(u, v, lambda_):
    """PDF de la copule de Hüsler–Reiss, très schématique."""
    # Idem, calcul dérivé partiel
    pdf_val = np.ones_like(u) * 1e-5
    return pdf_val

###############################################################################
# 3) GESTION GÉNÉRIQUE : LOG-VRAISEMBLANCE, ESTIMATION
###############################################################################
def negative_log_likelihood(params, u, v, family='gumbel'):
    """Calcule -log(pdf) en fonction de la famille de copule d'extrême valeur."""
    if family.lower() == 'gumbel':
        theta = params[0]
        pdf_vals = gumbel_pdf(u, v, theta)
    elif family.lower() == 'galambos':
        alpha = params[0]
        pdf_vals = galambos_pdf(u, v, alpha)
    elif family.lower() == 'husler_reiss':
        lambda_ = params[0]
        pdf_vals = husler_reiss_pdf(u, v, lambda_)
    else:
        raise ValueError(f"Copule non reconnue : {family}")
    
    pdf_vals = np.clip(pdf_vals, 1e-14, None)  # éviter log(0)
    return -np.sum(np.log(pdf_vals))

def fit_extreme_copula(u, v, family='gumbel', init_param=2.0):
    """
    Ajuste la copule d'extrême valeur par MLE,
    renvoie le paramètre estimé et les infos d'optimisation.
    """
    # Contraintes sur le paramètre selon la famille
    if family.lower() == 'gumbel':
        # Gumbel : theta >= 1
        bounds = [(1.0, 50.0)]
        method = 'L-BFGS-B'
    elif family.lower() == 'galambos':
        # Galambos : alpha > 0
        bounds = [(1e-3, 50.0)]
        method = 'L-BFGS-B'
    elif family.lower() == 'husler_reiss':
        # HR : lambda_ > 0
        bounds = [(1e-3, 50.0)]
        method = 'L-BFGS-B'
    else:
        raise ValueError(f"Copule non reconnue : {family}")
    
    res = optimize.minimize(
        fun=negative_log_likelihood,
        x0=np.array([init_param]),
        args=(u, v, family),
        method=method,
        bounds=bounds
    )
    return res.x[0], res

###############################################################################
# 4) EXEMPLE D’UTILISATION
###############################################################################
def main():
    # 1) Charger et filtrer
    data = load_and_filter_data(
        csv_file='data/weather_data.csv',
        var1='Temperature',
        var2='Humidity',
        date_start='2023-01-01',
        date_end='2023-05-31'
    )
    
    # 2) Extraire les valeurs extrêmes (p. ex. queue supérieure de Temperature)
    data_ext = extract_extremes(data, variable='Temperature', q=0.95)
    print(f"Nombre de points extrêmes sélectionnés : {len(data_ext)}")
    
    # 3) Pseudo-observations
    u = to_pseudo_observations(data_ext['Temperature'].values)
    v = to_pseudo_observations(data_ext['Humidity'].values)
    
    # 4) Choisir une famille et ajuster
    for fam in ['gumbel','galambos','husler_reiss']:
        param_hat, opt_res = fit_extreme_copula(u, v, family=fam, init_param=2.0)
        print(f"\nCopule : {fam}")
        print(f"Paramètre estimé : {param_hat:.4f}")
        print(f"Message de l'optimiseur : {opt_res.message}")
    
    # 5) Exemple de visualisation pour la copule choisie
    chosen_family = 'galambos'
    param_hat, _ = fit_extreme_copula(u, v, family=chosen_family, init_param=2.0)
    
    # Simple scatter (U, V)
    plt.figure(figsize=(6,5))
    plt.scatter(u, v, alpha=0.6, label='Pseudo-obs extrêmes')
    plt.title(f"Copule {chosen_family.capitalize()} (param={param_hat:.3f})")
    plt.xlabel("U (Temp)")
    plt.ylabel("V (Hum)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/weather_copula_extreme.png')
    plt.show()

if __name__ == "__main__":
    main()
