import matplotlib.pyplot as plt
import numpy as np
import os

def plot_correlation(observed, predicted, path="./results/weather_predicted_observed.png"):
    """
    Affiche et sauvegarde un nuage de points entre les valeurs observées et prédites
    avec la corrélation affichée dans le titre.

    Paramètres :
    - observed : liste ou array des valeurs observées
    - predicted : liste ou array des valeurs prédites
    - path : chemin du fichier pour sauvegarder le graphique (format PNG)
    """
    # Conversion en array numpy
    observed = np.array(observed)
    predicted = np.array(predicted)

    # Calcul de la corrélation de Pearson
    correlation = np.corrcoef(observed, predicted)[0, 1]

    # Création du nuage de points
    plt.figure(figsize=(6, 6))
    plt.scatter(observed, predicted, color='purple')
    plt.xlabel("Observé")
    plt.ylabel("Prédit")
    plt.title(f"Corrélation entre Observé et Prédit (Corrélation={correlation:.2f})")
    plt.grid(True)
    plt.tight_layout()

    # Création du dossier s'il n'existe pas
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Sauvegarde de l'image
    plt.savefig(path)

    # Affichage
    plt.show()
