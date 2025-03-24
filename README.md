# Analyse des Séries Temporelles de Température

Ce projet contient un script Python pour analyser les séries temporelles de température en utilisant le processus de Hawkes.

## Structure du Projet

```
.
├── data/
│   └── weather_data.csv    # Données météorologiques
└── scripts/
    └── weather_hawkes.py   # Script d'analyse
```

## Données

Le fichier `weather_data.csv` contient les données météorologiques avec les colonnes suivantes :
- `Date` : Date de la mesure
- `Time` : Heure de la mesure
- `Temperature` : Température en degrés Celsius
- `Wind Speed` : Vitesse du vent en mph
- `Pressure` : Pression atmosphérique en in

## Script d'Analyse

Le script `weather_hawkes.py` implémente une analyse des séries temporelles de température en utilisant le processus de Hawkes. Il permet de :

1. Charger et préparer les données
2. Convertir les températures de Fahrenheit en Celsius
3. Analyser les changements de température
4. Ajuster un modèle de Hawkes aux données
5. Visualiser les résultats avec :
   - Processus ponctuel des pics de température
   - Intensité conditionnelle du processus de Hawkes
   - Distribution des changements de température

## Utilisation

Pour exécuter l'analyse :

```bash
python scripts/weather_hawkes.py
```

Le script générera des visualisations et affichera les statistiques suivantes :
- Nombre total d'événements (pics de température)
- Paramètres du modèle de Hawkes (μ, α, β)
- Stabilité du processus
- Moments des pics significatifs de température
