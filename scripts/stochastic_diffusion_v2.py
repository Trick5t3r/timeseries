import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import warnings
import scipy.stats as st
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

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
    
    # Extraction de la partie numérique de 'Temperature' (ex: "57 °F" -> 57)
    df['temp'] = df['Temperature'].astype(str).str.extract(r'([\d\.\-]+)')[0].astype(float)
    
    # Création d'un index numérique représentant le nombre d'heures
    df['time_idx'] = np.arange(len(df))
    return df

#########################################################
# 2. Modèle OU avec double sinus à fréquences fixes      #
#########################################################
def mu(t, a, b1, phi1, b2, phi2):
    """
    Fonction moyenne avec double périodicité fixe :
    
      mu(t) = a + b1*sin(FIXED_W1*t + phi1) + b2*sin(FIXED_W2*t + phi2)
      
    où FIXED_W1 = 2π/(365×24) et FIXED_W2 = 2π/24.
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
# 3. Simulation du processus OU (schéma d'Euler-Maruyama) #
#########################################################
def simulate_ou(X0, t, theta, sigma, a, b1, phi1, b2, phi2, n_simulations=3, dt=0.1):
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
# 4. Visualisation principale avec Plotly    #
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
        line=dict(width=2)
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

#########################################################
# 5. Diagnostic : Calcul et visualisation des résidus    #
#########################################################
def compute_residuals(X, t, theta, a, b1, phi1, b2, phi2, dt=1):
    """
    Calcule les résidus du modèle OU :
      résidu[i] = X[i+1] - (X[i] + θ*(mu(t[i]) - X[i])*dt)
    Retourne un tableau de longueur len(X)-1.
    """
    X_pred = X[:-1] + theta * (mu(t[:-1], a, b1, phi1, b2, phi2) - X[:-1]) * dt
    residuals = X[1:] - X_pred
    return residuals

def plot_qq(residuals, output_file="qq_plot.html"):
    """
    Génère un QQ Plot des résidus en comparant aux quantiles théoriques d'une loi normale.
    """
    (osm, osr), (slope, intercept, r) = st.probplot(residuals, dist="norm")
    line_x = np.linspace(min(osm), max(osm), 100)
    line_y = slope * line_x + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Résidus'))
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Ligne de référence'))
    fig.update_layout(title='QQ Plot des résidus',
                      xaxis_title='Quantiles théoriques',
                      yaxis_title='Quantiles observés',
                      template='plotly_white')
    fig.write_html(output_file)
    print(f"QQ Plot sauvegardé sous : {output_file}")
    fig.show()

def plot_residuals_time(t, residuals, output_file="residuals_time.html"):
    """
    Affiche les résidus en fonction du temps (heures).
    t correspond à l'index numérique original, ici on utilise t[1:] pour les résidus.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[1:], y=residuals, mode='lines+markers', name='Résidus'))
    fig.update_layout(title='Résidus dans le temps',
                      xaxis_title='Temps (heures)',
                      yaxis_title='Résidus',
                      template='plotly_white')
    fig.write_html(output_file)
    print(f"Graphique des résidus dans le temps sauvegardé sous : {output_file}")
    fig.show()

def plot_residuals_histogram(residuals, output_file="residuals_histogram.html"):
    """
    Affiche l'histogramme des résidus avec normalisation en densité.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30, histnorm='probability density', name='Résidus'))
    fig.update_layout(title='Histogramme des résidus',
                      xaxis_title='Résidus',
                      yaxis_title='Densité',
                      template='plotly_white')
    fig.write_html(output_file)
    print(f"Histogramme des résidus sauvegardé sous : {output_file}")
    fig.show()

###################
# Script principal#
###################
def main():
    # Chargement des données depuis 'weather_data.csv'
    file_path = 'data/weather_data.csv'
    df = load_and_prepare_data(file_path)
    X = df['temp'].values
    t = df['time_idx'].values  # t en heures
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
    
    residuals = compute_residuals(X, df['time_idx'].values, theta, a, b1, phi1, b2, phi2)
    
    plot_diagnostics(
    dates=df['datetime'],
    real_data=X,
    simulations=sims,
    t=df['time_idx'].values,
    residuals=residuals
)
    
    print("\n--- Autocorrelation des résidus ---")
    
    # Tracer l'ACF avec Plotly
    lags=40
    acf_vals = sm.tsa.acf(residuals, nlags=lags, fft=True)
    conf = 1.96 / np.sqrt(len(residuals))  # intervalle de confiance 95%
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.arange(lags+1), y=acf_vals, name="ACF"))
    fig.add_shape(type="line", x0=0, x1=lags, y0=conf, y1=conf,
                  line=dict(color="red", dash="dash"))
    fig.add_shape(type="line", x0=0, x1=lags, y0=-conf, y1=-conf,
                  line=dict(color="red", dash="dash"))
    fig.update_layout(title="ACF des résidus",
                      xaxis_title="Lag",
                      showlegend=True,
                      yaxis_title="Autocorrélation")
    fig.write_html('autocorrelation.html')
    print(f"Autocorrelation saved → autocorrelation.html")
    fig.show()


def plot_diagnostics(dates, real_data, simulations, t, residuals, output_file="diagnostics.html"):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Données réelles vs simulations",
            "QQ‑Plot des résidus",
            "Résidus dans le temps",
            "Histogramme des résidus"
        ]
    )

    # 1️⃣ Série réelle + simulations avec légende
    fig.add_trace(go.Scatter(x=dates, y=real_data, mode="lines", name="Données réelles", line=dict(width=2)), row=1, col=1)
    for i in range(simulations.shape[1]):
        fig.add_trace(go.Scatter(x=dates, y=simulations[:, i], mode="lines", opacity=0.5, name=f"Simulation {i+1}"), row=1, col=1)

    # 2️⃣ QQ‑Plot
    osm, osr = st.probplot(residuals, dist="norm")[0]
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Quantiles résidus"), row=1, col=2)
    slope, intercept = np.polyfit(osm, osr, 1)
    fig.add_trace(go.Scatter(x=osm, y=slope * osm + intercept, mode="lines", name="Référence"), row=1, col=2)

    # 3️⃣ Résidus vs temps
    fig.add_trace(go.Scatter(x=dates, y=residuals, mode="markers", name="Résidus temporels", marker=dict(size=1)), row=2, col=1)

    # 4️⃣ Histogramme
    fig.add_trace(go.Histogram(x=residuals, nbinsx=20, histnorm="probability density", name="Histogramme"), row=2, col=2)

    fig.update_layout(
        height=800,
        width=1000,
        showlegend=True,
        title_text="Diagnostics du modèle OU"
    )

    fig.write_html(output_file)
    print(f"Diagnostics saved → {output_file}")
    fig.show()



if __name__ == "__main__":
    main()
