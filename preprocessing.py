import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


def preprocess_data(df):
    """
    Cette fonction effectue plusieurs étapes de prétraitement des données sur un DataFrame. 
    Elle s'assure que les colonnes sont dans un format exploitable, supprime les colonnes inutiles ou problématiques, 
    et normalise les données pour qu'elles soient prêtes à être utilisées. Enfin, elle retourne le DataFrame prétraité.

    Arguments :
    - df : pandas.DataFrame
        Le DataFrame contenant les données brutes à prétraiter.

    Retourne :
    - pandas.DataFrame
        Le DataFrame nettoyé, transformé et prêt à l'emploi.
    """
    #1. Crée une copie du DataFrame d'origine pour éviter toute modification accidentelle des données source:
    df = df.copy()
    #2. Convertit des colonnes spécifiques en type float pour assurer la cohérence des types de données:
    for column_name in ["insee_%_agri", "insee_med_living_level", "insee_%_ind", "insee_%_const"]:
        convert_to_float(df,column_name)
    #3. Identifie et traite les colonnes contenant des dates, en séparant potentiellement la date en différentes composantes (année, mois, jour) grâce à la fonction `split_date_column`:
    split_date_column(df,'piezo_measurement_date')
    #4. Supprime les colonnes non désirées ou inutiles:
    drop_unwanted_columns(df)
    #5. Élimine les caractéristiques fortement corrélées via `remove_high_correlation_features`, pour éviter la redondance:
    remove_high_correlation_features(df)
    #6. Identifie les colonnes contenant des types de données mixtes (par exemple, mélange de chaînes de caractères et d'entiers) et les supprime:  
    mixed_type_columns = [
        col for col in df.columns 
        if df[col].apply(type).nunique() > 1
    ]
    print(f"Columns with mixed types: {mixed_type_columns}")
    df.drop(columns=mixed_type_columns, inplace=True)
    #7. Gère les valeurs manquantes dans les colonnes restantes en appelant `fill_missing_values`
    fill_missing_values(df)
    #8. Normalise les données et applique un encodage pour transformer les caractéristiques catégoriques en un format numérique à l'aide de la fonction `normalize_and_encode_features`.
    df = normalize_and_encode_features(df)

    return df


def convert_to_float(df, column_name):
    """
    Convertit une colonne d'un DataFrame pandas en type float.
    
    Arguments :
    - df : pd.DataFrame
        Le DataFrame contenant la colonne à convertir.
    - column_name : str
        Le nom de la colonne à convertir.
    
    Retourne :
    - pd.DataFrame
        Le DataFrame avec la colonne convertie en float.
    """
    try:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        return df
    except KeyError:
        raise KeyError(f"La colonne '{column_name}' n'existe pas dans le DataFrame.")
    except Exception as e:
        raise Exception(f"Erreur lors de la conversion : {e}")


def normalize_and_encode_features(df):
    """
    Cette fonction effectue la normalisation des colonnes numériques et l'encodage des colonnes catégoriques d'un DataFrame.

    Arguments :
    - df : pandas.DataFrame
        Le DataFrame contenant les données à normaliser et encoder.

    Retourne :
    - pandas.DataFrame
        Le DataFrame transformé, avec des colonnes numériques normalisées et des colonnes catégoriques encodées.
    """
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Initialisation des objets pour la normalisation et l'encodage ordinal
    scaler = MinMaxScaler()
    ordinal_encoder = OrdinalEncoder()

    # Normalisation des colonnes numériques
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encodage des colonnes catégoriques en fonction du nombre de valeurs uniques
    for col in categorical_cols:
        unique_values = df[col].nunique()
        
        if unique_values <= 30:
            # One-Hot Encoding pour moins de 30 catégories uniques
            df = pd.get_dummies(df, columns=[col], drop_first=True)
        else:
            # Encodage ordinal ou naturel pour plus de 30 catégories uniques
            df[col] = ordinal_encoder.fit_transform(df[[col]])
    return df


def split_date_column(df, column_name):
    """
    Cette fonction prend une colonne de dates dans un DataFrame et la divise en trois colonnes distinctes : année, mois et jour.
    Elle transforme également la colonne de dates en format datetime si ce n'est pas déjà le cas, et supprime ensuite la colonne d'origine.

    Arguments :
    - df : pandas.DataFrame
        Le DataFrame contenant la colonne de dates à traiter.
    - column_name : str
        Le nom de la colonne contenant des dates, qui sera divisée en trois nouvelles colonnes (année, mois, jour).

    Retourne :
    - pandas.DataFrame
        Le DataFrame modifié avec trois nouvelles colonnes pour l'année, le mois et le jour, et sans la colonne de dates d'origine.
    """
    df[column_name] = pd.to_datetime(df[column_name])
    df = df.assign(
        **{
            column_name+'_year': df[column_name].dt.year,
            column_name+'_month': df[column_name].dt.month,
            column_name+'_day': df[column_name].dt.day
        }
    )    
    # df.loc[:, [column_name+'_year', column_name+'_month', column_name+'_day']] = df[column_name].str.split('-', expand=True).astype('int64').rename(columns={0: 'year', 1: 'month', 2: 'day'})
    df.drop(columns=[column_name], inplace=True)


def drop_unwanted_columns(df):
    """
    Cette fonction supprime les colonnes jugées inutiles dans un DataFrame. Elle prend une liste de noms de colonnes 
    considérées comme non pertinentes, vérifie si elles existent dans le DataFrame, et les supprime.

    Arguments :
    - df : pandas.DataFrame
        Le DataFrame contenant les colonnes à supprimer.

    Retourne : None
    """

    UselessFeatures = ["piezo_station_commune_name","piezo_station_department_name","piezo_station_bss_code","piezo_station_pe_label","piezo_station_bdlisa_codes","piezo_producer_name","piezo_measure_nature_name","piezo_station_pe_label", "meteo_name", "hydro_status_label", "hydro_method_label", "hydro_method_label"]
    #remove UselessFeatures that are in the dataset
    UselessFeatures = [col for col in UselessFeatures if col in df.columns]
    df.drop(columns=UselessFeatures, inplace = True)


def remove_high_correlation_features(df):
    """
    Cette fonction élimine les colonnes du DataFrame qui sont fortement corrélées entre elles, afin d'éviter la redondance 
    et améliorer l'efficacité des modèles de machine learning. Elle identifie les paires de caractéristiques ayant une 
    corrélation supérieure à un seuil défini, et supprime les colonnes considérées comme redondantes.

    Étapes principales :
    1. Sélection des colonnes numériques :
       - La fonction sélectionne toutes les colonnes numériques (de type `int` et `float`) dans le DataFrame à l'aide de 
         `select_dtypes(include=[np.number])`, ce qui permet de se concentrer uniquement sur les variables continues.

    2. Calcul de la matrice de corrélation :
       - Une matrice de corrélation est calculée pour les colonnes numériques avec `corr()`, puis on prend la valeur absolue 
         des corrélations pour ignorer les signes (corrélations positives et négatives sont traitées de manière équivalente).

    3. Identification des paires de caractéristiques fortement corrélées :
       - La fonction crée une matrice triangulaire supérieure pour ne conserver que les paires uniques de corrélations 
         (en évitant les répétitions et la corrélation d'une colonne avec elle-même).
       - Elle extrait les paires de colonnes ayant une corrélation supérieure au seuil spécifié (0.9 par défaut) et les organise 
         dans un DataFrame pour mieux les visualiser.

    4. Sélection des colonnes redondantes :
       - Si la corrélation entre deux caractéristiques dépasse le seuil (0.9), la fonction considère la deuxième colonne de la paire 
         comme redondante et la marque pour suppression.

    5. Suppression des colonnes redondantes :
       - Les colonnes identifiées comme redondantes sont ensuite supprimées du DataFrame en utilisant la méthode `drop()`, avec 
         `inplace=True` pour modifier directement le DataFrame.

    Arguments :
    - df : pandas.DataFrame
        Le DataFrame contenant les données avec les colonnes à vérifier pour la corrélation.

    Retourne : None
        La fonction modifie le DataFrame directement et ne retourne rien.
    """

    numeric_data = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr().abs() 
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool) 
    high_corr_pairs = corr_matrix.where(upper_triangle).stack().reset_index()
    high_corr_pairs.columns = ['Feature1', 'Feature2', 'Correlation']
    threshold = 0.9
    redundant_features = high_corr_pairs[high_corr_pairs['Correlation'] > threshold]['Feature2'].unique()
    df.drop(columns=redundant_features, inplace=True)


def fill_missing_values(df):
    """
    Cette fonction remplit les valeurs manquantes (NaN) en fonction du type de données des colonnes.
    Elle remplace les valeurs manquantes des colonnes numériques par la médiane et celles des colonnes catégoriques 
    par la valeur la plus fréquente (mode). Si des valeurs manquantes persistent après ces remplissages, elles sont affichées.

    Étapes principales :
    1. Traitement des colonnes numériques :
       - La fonction sélectionne toutes les colonnes numériques (types `float64`, `int64`, et `Int32`).
       - Pour chaque colonne numérique, elle remplace les valeurs manquantes par la médiane de la colonne. 
         La médiane est souvent utilisée pour éviter l'impact des valeurs aberrantes (outliers).

    2. Traitement des colonnes catégoriques :
       - La fonction sélectionne les colonnes de type `object` (celles contenant des chaînes de caractères ou des catégories).
       - Pour chaque colonne catégorique, elle remplace les valeurs manquantes par la valeur la plus fréquente (mode).
         Le mode est choisi pour refléter la catégorie la plus représentée.

    3. Vérification des valeurs manquantes restantes :
       - Après avoir rempli les valeurs manquantes, la fonction vérifie si des valeurs NaN persistent dans le DataFrame.
       - Si des valeurs manquantes sont encore présentes, elle affiche les colonnes concernées.
       - Sinon, elle affiche un message indiquant qu'il n'y a plus de valeurs manquantes dans le DataFrame.

    Arguments :
    df : pandas.DataFrame
        Le DataFrame contenant des valeurs manquantes à remplir.

    Retourne : None
        La fonction modifie le DataFrame directement et ne retourne rien.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'Int32']).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    if df.isna().sum().sum() > 0:
        cols_with_na = df.columns[df.isna().any()].tolist()
        print(f"Columns with missing values: {cols_with_na}")
    else:
        print("No missing values in the dataset")
