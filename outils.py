import pandas as pd
import os


def save_to_csv(model, x_test, mapping_int_to_string, row_index_test, path):
    """
    Cette fonction génère des prédictions à partir d'un modèle, les mappe selon un dictionnaire d'encodage, 
    puis enregistre les résultats dans un fichier CSV. Elle associe chaque prédiction à un index de ligne spécifique 
    et les sauvegarde dans un fichier à l'emplacement indiqué par le paramètre `path`.

    Étapes principales :
    1. Génération des prédictions :
       - Le modèle (`model`) prédit les valeurs pour le jeu de données de test `x_test`.
       - Les prédictions sont des entiers, qui sont ensuite mappés à des valeurs de chaîne de caractères à l'aide 
         du dictionnaire `mapping_int_to_string`. Ce dictionnaire associe chaque entier prédit à une catégorie sous forme de texte.

    2. Création d'un DataFrame avec les prédictions :
       - Un DataFrame `df_predictions` est créé avec les prédictions mappées, où chaque ligne représente une prédiction 
         et les indices sont définis selon `row_index_test`.
       - La colonne du DataFrame est nommée "piezo_groundwater_level_category", représentant la catégorie prédite par le modèle.

    3. Sauvegarde des résultats dans un fichier CSV :
       - Le DataFrame est ensuite exporté sous forme de fichier CSV à l'emplacement spécifié par le paramètre `path`.
       - La fonction `get_csv_path(path)` est utilisée pour déterminer le chemin exact où le fichier doit être sauvegardé.

    Arguments :
    - model :
        Le modèle entraîné qui sera utilisé pour effectuer des prédictions sur les données de test (`x_test`).
    - x_test : pandas.DataFrame
        Le jeu de données de test contenant les variables pour lesquelles faire des prédictions.
    - mapping_int_to_string : dict
        Un dictionnaire qui mappe les valeurs entières prédites par le modèle à des chaînes de caractères (catégories).
    - row_index_test : pandas.Index
        Les indices des lignes correspondant aux observations dans `x_test`, utilisés comme index dans le DataFrame des prédictions.
    - path : str
        Le chemin où le fichier CSV contenant les prédictions sera sauvegardé.

    Retourne : None
        La fonction ne retourne rien mais effectue une action en enregistrant un fichier CSV.
    """
    predictions = [mapping_int_to_string[pred] for pred in model.predict(x_test)]
    df_predictions = pd.DataFrame(
        predictions, 
        index=row_index_test,  # Set the index
        columns=["piezo_groundwater_level_category"]  # Set the column name
    )
    df_predictions.to_csv(get_csv_path(path))

def get_csv_path(path):
    #if path is a csv return it, otherwist add .csv to the path
    if not path.endswith('.csv'):
        path= path + '.csv'
    # if path is not in output folder, add it
    if not path.startswith('output/'):
        path = 'output/' + path
    #if csv allready exists, add a number to the end of the path, starting from adding nothing
    i = 1
    while os.path.exists(path):
        path = path.replace('.csv',f'_({i}).csv')
        i += 1
    return path