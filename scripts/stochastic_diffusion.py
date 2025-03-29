import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import warnings

# Pour un affichage plus propre, on ignore certains warnings
warnings.filterwarnings("ignore")

# Fréquences fixes en heures :
FIXED_W1 = 2 * np.pi / (365 * 24)  # Fréquence annuelle
FIXED_W2 = 2 * np.pi / 24          # Fréquence journalière

#############################################
# 1. Chargement et préparation des données  #
#############################################
def load_and_prepare_data(file_path):
    """
    Charge le fichier CSV qui contient les colonnes 'Date', 'Time' et 'Temperature'.
      - 'Date' au format "YYYY-MM-DD"
      - 'Time' au format "12:00 AM"
      - 'Temperature' au format "57 °F"
    
    Combine 'Date' et 'Time' en une colonne datetime.
    Extrait la partie numérique de 'Temperature' pour obtenir une colonne 'temp'.
    Crée un index numérique 'time_idx' en heures.
    """
    print("Chargement des données...")
    df = pd.read_csv(file_path)
    
    # Combinaison des colonnes Date et Time en une colonne datetime
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %I:%M %p')
    
    # Tri par datetime et réinitialisation de l'index
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Extraction de la partie numérique de 'Temperature' (exemple : "57 °F" -> 57)
    df['temp'] = df['Temperature'].astype(str).str.extract(r'([\d\.\-]+)')[0].astype(float)
    
    # Création d'un index numérique représentant le nombre d'heures (on suppose que chaque ligne correspond à une heure)
    df['time_idx'] = np.arange(len(df))
    return df

#########################################################
# 2. Modèle OU avec double sinus à fréquences fixes      #
#########################################################
def mu(t, a, b1, phi1, b2, phi2):
    """
    Fonction moyenne avec double périodicité fixe :
    
      mu(t) = a + b1*sin(FIXED_W1*t + phi1) + b2*sin(FIXED_W2*t + phi2)
      
    où FIXED_W1 et FIXED_W2 sont respectivement 2π/(365*24) et 2π/24.
    """
    return a + b1 * np.sin(FIXED_W1 * t + phi1) + b2 * np.sin(FIXED_W2 * t + phi2)

def neg_log_likelihood(params, X, t, dt=1):
    """
    Log-vraisemblance négative pour le modèle OU discrétisé :
    
      dX_t = θ*(mu(t) - X_t)*dt + σ*dW_t
      
    params: [θ, σ, a, b1, φ1, b2, φ2]
    """
    theta, sigma, a, b1, phi1, b2, phi2 = params
    mu_t = mu(t[:-1], a, b1, phi1, b2, phi2)
    drift = theta * (mu_t - X[:-1])
    expected_X = X[:-1] + drift * dt
    var = sigma**2 * dt
    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + ((X[1:] - expected_X)**2) / var)
    return -ll

def fit_ou_parameters(X, t):
    """
    Estime les paramètres du modèle OU à moyenne double sinus avec fréquences fixes.
    Paramètres à estimer : θ, σ, a, b1, φ1, b2, φ2.
    """
    initial_params = [0.1, 1.0, np.mean(X), 10, 0, 1, 0]  # initialisation
    bounds = [(1e-6, None), (1e-6, None),
              (None, None), (None, None),
              (None, None), (None, None),
              (None, None)]
    result = minimize(neg_log_likelihood, initial_params, args=(X, t), method='L-BFGS-B', bounds=bounds)
    return result.x

#########################################################
# 3. Simulation du processus OU par schéma d'Euler-Maruyama #
#########################################################
def simulate_ou(X0, t, theta, sigma, a, b1, phi1, b2, phi2, n_simulations=3, dt=1):
    """
    Simule n_simulations trajectoires du processus OU.
    
    La discrétisation est donnée par :
      X_{t+1} = X_t + θ*(mu(t) - X_t)*dt + σ*sqrt(dt)*Z,
    avec Z ~ N(0,1).
    """
    T = len(t)
    sims = np.zeros((T, n_simulations))
    for i in range(n_simulations):
        X_sim = np.zeros(T)
        X_sim[0] = X0
        for j in range(1, T):
            seasonal_mean = mu(t[j-1], a, b1, phi1, b2, phi2)
            drift = theta * (seasonal_mean - X_sim[j-1])
            X_sim[j] = X_sim[j-1] + drift * dt + sigma * np.sqrt(dt) * np.random.randn()
        sims[:, i] = X_sim
    return sims

##############################################
# 4. Visualisation avec Plotly et sauvegarde  #
##############################################
def plot_with_plotly(dates, real_data, simulations, output_file='ou_simulation.html'):
    """
    Affiche avec Plotly les données réelles et superpose les trajectoires simulées.
    Le graphique est sauvegardé dans un fichier HTML.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=real_data,
        mode='lines+markers',
        name='Données réelles',
        line=dict(color='black', width=2)
    ))
    n_sim = simulations.shape[1]
    for i in range(n_sim):
        fig.add_trace(go.Scatter(
            x=dates,
            y=simulations[:, i],
            mode='lines',
            name=f'Simulation {i+1}',
            opacity=0.6
        ))
    fig.update_layout(
        title='Données réelles vs Trajectoires simulées (OU double sinus)',
        xaxis_title='Date',
        yaxis_title='Température (°F)',
        template='plotly_white'
    )
    fig.write_html(output_file)
    print(f"Plot sauvegardé sous : {output_file}")
    fig.show()

###################
# Script principal#
###################
def main():
    file_path = 'data/weather_data.csv'
    df = load_and_prepare_data(file_path)
    X = df['temp'].values
    t = df['time_idx'].values  # t est en heures
    dates = df['datetime']
    
    # Estimation des paramètres (θ, σ, a, b1, φ1, b2, φ2)
    params = fit_ou_parameters(X, t)
    theta, sigma, a, b1, phi1, b2, phi2 = params
    print("Paramètres estimés :")
    print(f"  θ = {theta:.4f}, σ = {sigma:.4f}")
    print(f"  a = {a:.4f}")
    print(f"  b1 = {b1:.4f}, φ1 = {phi1:.4f}")
    print(f"  b2 = {b2:.4f}, φ2 = {phi2:.4f}")
    
    # Simulation du processus OU (3 trajectoires)
    sims = simulate_ou(X0=X[0], t=t, theta=theta, sigma=sigma, a=a,
                       b1=b1, phi1=phi1, b2=b2, phi2=phi2, n_simulations=3)
    
    # Visualisation et sauvegarde du plot en HTML
    plot_with_plotly(dates, X, sims, output_file='ou_simulation.html')

if __name__ == "__main__":
    main()
