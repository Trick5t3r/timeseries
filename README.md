## Configuration

### Étape 1 : Cloner le dépôt

Ouvrez votre terminal (ou invite de commandes) et exécutez la commande suivante pour cloner le dépôt :

```bash
git clone https://github.com/Trick5t3r/timeseries.git
```

### Étape 2 : Naviguer dans le répertoire

Accédez au répertoire cloné :

```bash
cd timeseries
```

### Étape 3 : Créer un environnement virtuel

Pour créer un environnement virtuel, exécutez la commande suivante :

```bash
python -m venv .venv
```

### Étape 4 : Activer l'environnement virtuel

- **Sur Windows :**

```bash
.venv\Scripts\activate
```

- **Sur macOS et Linux :**

```bash
source .venv/bin/activate
```

### Étape 5 : Installer les dépendances

Une fois l'environnement virtuel activé, installez les dépendances nécessaires en utilisant le fichier `requirements.txt` :

```bash
pip install -r requirements.txt
```

### Étape 6 : Vérifier l'installation

Pour vérifier que tout est installé correctement, vous pouvez exécuter le script d'analyse :

```bash
python scripts/weather_arima.py
```

