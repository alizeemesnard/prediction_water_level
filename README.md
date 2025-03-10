
## 1. Dataset

L'ensemble de donnees se presente sous la forme d'un fichier .CSV comprenant un total de 2 830 316 lignes et 136 caracteristiques, y compris le label (piezo_groundwater_level_category), fourni par le Hi!Paris Center.

Ce vaste ensemble de donnees est le resultat de la fusion de plusieurs sources d'information, representant chacune des aspects distincts de l'environnement et de l'hydrologie.

Plus precisement, il comprend **4 sous-groupes de donnees**:
    1. **Piezometrie**: donnees des nappes phreatiques prelevees par des sondes meteorologiques.
    2. **Meteo**: temperature moyenne, pression moyenne, etc.
    3. **Hydrometrie**: donnees sur le debit d'eau.
    4. **INSEE**: donnees statistiques sur la population.

Le dictionnaire des variables est present dans le dossier "data".

**Variables importantes** pour predire les niveaux d'eau souterraine:
    
### 1.1. **Piezometrie (Donnees des nappes phreatiques)**:
    
    - `piezo_station_investigation_depth` : Profondeur de la sonde pour mesurer les niveaux d'eau.
        - **Importance** : La profondeur de l'aquifere influence la reaction des niveaux d'eau aux facteurs externes tels que les precipitations et les extractions.
    
### 1.2. **Meteo (Donnees climatiques)**:
    
    - `precipitation` : Precipitations.
        - **Importance** : Contribue directement a la recharge des nappes phreatiques. Plus il pleut, plus les niveaux peuvent augmenter.
    - `temperature` : Temperature moyenne.
        - **Importance** : Affecte les taux d'evaporation et influence l'equilibre entre recharge et pertes.
    - `evapotranspiration` : evaporation combinee a la transpiration des plantes.
        - **Importance** : Reduit la quantite d'eau disponible pour l'infiltration.
    
### 1.3. **Hydrologie (Donnees hydrologiques)**:
    
    - `river_discharge` : Debit des rivieres.
        - **Importance** : Un debit eleve peut indiquer une recharge accrue des nappes connectees.
    - `soil_moisture` : Teneur en eau des sols.
        - **Importance** : Une forte humidite facilite l'infiltration et la recharge des aquiferes.
    
### 1.4. **Prelevement (Donnees d'extraction)**:
    
    - `groundwater_extraction_volume` : Volume d'eau souterraine utilise.
        - **Importance** : Influence directe sur l'epuisement des nappes phreatiques.
    - `agricultural_use_percentage` : Pourcentage d'usage agricole.
        - **Importance** : L'utilisation excessive pour l'agriculture peut reduire significativement les niveaux.
    
### 1.5. **Distance (Donnees geographiques)**:
    
    - `distance_to_river` : Distance jusqu'a la riviere la plus proche.
        - **Importance** : La proximite facilite les interactions avec l'eau de surface et la recharge.
    
### 1.6. **Insee (Donnees economiques)**:
    
    - `population_density` : Densite de population.
        - **Importance** : Une densite elevee est souvent liee a une demande accrue en eau.
    - `urbanization_rate` : Taux d'urbanisation.
        - **Importance** : L'urbanisation reduit l'infiltration et augmente l'utilisation de l'eau souterraine.

## 2. Objectif: prédire le niveau d'eau des nappes phréatriques en France en été

## 3. Pre-processing

Le pré-traitement est effectué dans preprocessing.py et comprend :

- Le nettoyage des données
- Le traitement des valeurs manquantes
- La normalisation et l'encodage des variables

Le fichier qui en résulte est preprocessed_data_all.csv

## 4. Tentatives de modèles de machine learning

Nous avons utilisé un DecisionTree en modèle baseline pour avoir une idée des performances atteignables. 

Notre modèle final est un LSTM. 

- **Type de modèle** : Réseau de neurones LSTM (Long Short-Term Memory)
- **Architecture** :
  - **Première couche LSTM** : 
    - 50 unités (neurones)
    - `return_sequences=True` : permet de transmettre les séquences à la couche suivante
    - Entrée de forme `(30, len(FEATURES))` où `30` est la longueur de la séquence temporelle et `len(FEATURES)` est le nombre de caractéristiques
  - **Dropout** de 20% après la première couche LSTM pour éviter l'overfitting
  - **Deuxième couche LSTM** : 
    - 50 unités (neurones)
    - `return_sequences=False` : cette couche ne renvoie pas les séquences, mais seulement la sortie de la dernière cellule LSTM
  - **Dropout** de 20% après la deuxième couche LSTM pour éviter l'overfitting
  - **Couche Dense (Fully Connected)** : 
    - 25 unités avec fonction d'activation `relu`
  - **Couche de sortie** : 
    - 5 unités avec fonction d'activation `softmax` pour la prédiction multi-classes (5 classes)
  
- **Compilation du modèle** :
  - Optimiseur : `adam`
  - Fonction de perte : `sparse_categorical_crossentropy` (adaptée pour les labels entiers)
  - Métrique : `accuracy`
  
- **Objectif** : Prédiction d'une classe parmi 5 classes possibles à partir de séquences temporelles.


## 5. Performances et analyse des résultats

## 6. Prérequis
Le projet nécessite les bibliothèques suivantes :
- pandas
- numpy
- scikit-learn
- tensorflow

Ces bibliothèques peuvent être installées en utilisant pip :
- pip install pandas numpy scikit-learn tensorflow
