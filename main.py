import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def get_clean_cinema_data(cinema_path):
    return (
        pd.read_csv(cinema_path, sep=";", encoding="utf-8", engine="python")
        .drop_duplicates()
        .fillna(
            {
                "population de la commune": 0,
                "écrans": 0,
                "fauteuils": 0,
                "entrées 2021": 0,
                "entrées 2022": 0,
                "label Art et Essai": "non",
            }
        )
        .astype(
            {
                "population de la commune": int,
                "écrans": int,
                "fauteuils": int,
                "entrées 2021": int,
                "entrées 2022": int,
            }
        )
        .assign(
            label_art_et_essai=lambda x: x["label Art et Essai"].str.strip().str.lower()
        )
    )

def prepare_data_for_2018_2021(df):
    df_2018_2021 = df[(df['entrées 2021'] > 0) & (df['entrées 2022'] > 0)]
    df_2018_2021 = df_2018_2021[['écrans', 'fauteuils', 'population de la commune', 'entrées 2021']]
    df_2018_2021 = df_2018_2021.dropna()
    X_2018_2021 = df_2018_2021[['écrans', 'fauteuils', 'population de la commune']]
    y_2018_2021 = df_2018_2021['entrées 2021']
   
    return X_2018_2021, y_2018_2021

def train_and_evaluate_on_2018_2021(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
   
    print(f"R² sur l'ensemble d'entraînement (2018-2021) : {train_r2:.2f}")
    print(f"R² sur l'ensemble de test (2018-2021) : {test_r2:.2f}")
    print(f"Erreur moyenne absolue (MAE) sur l'ensemble de test (2018-2021) : {test_mae:.2f}")
   
    return model

def test_model_on_2022_data(model, df):
    X_2022 = df[["écrans", "fauteuils", "population de la commune"]]
    y_2022 = df["entrées 2022"]
    y_pred_2022 = model.predict(X_2022)
    comparison = pd.DataFrame(
        {"Valeurs réelles (2022)": y_2022, "Prédictions": y_pred_2022}
    )
    print("Comparaison des prédictions avec les valeurs réelles (2022) :")
    print(comparison.head(10))

def recommend_strategy_for_cinema(
    current_ecrans, current_fauteuils, population_commune, model
):
    current_data = pd.DataFrame(
        {
            "écrans": [current_ecrans],
            "fauteuils": [current_fauteuils],
            "population de la commune": [population_commune],
        }
    )
    current_prediction = model.predict(current_data)[0]
    more_ecrans_data = current_data.copy()
    more_ecrans_data["écrans"] += 1
    more_ecrans_prediction = model.predict(more_ecrans_data)[0]
    more_fauteuils_data = current_data.copy()
    more_fauteuils_data["fauteuils"] += 30
    more_fauteuils_prediction = model.predict(more_fauteuils_data)[0]

    return {
        "current_prediction": current_prediction,
        "more_ecrans_prediction": more_ecrans_prediction,
        "more_fauteuils_prediction": more_fauteuils_prediction,
    }

cinema_data = get_clean_cinema_data("data/cinemas.csv")

X_2018_2021, y_2018_2021 = prepare_data_for_2018_2021(cinema_data)

model_2018_2021 = train_and_evaluate_on_2018_2021(X_2018_2021, y_2018_2021)

test_model_on_2022_data(model_2018_2021, cinema_data)

cinema_fictif = {
    "current_ecrans": 2,
    "current_fauteuils": 120,
    "population_commune": 20000,
}

strategy_results = recommend_strategy_for_cinema(
    cinema_fictif["current_ecrans"],
    cinema_fictif["current_fauteuils"],
    cinema_fictif["population_commune"],
    model_2018_2021,
)

print(
    "Projection actuelle des entrées annuelles :",
    strategy_results["current_prediction"],
)
print(
    "Projection avec un écran supplémentaire :",
    strategy_results["more_ecrans_prediction"],
)
print(
    "Projection avec 30 fauteuils supplémentaires :",
    strategy_results["more_fauteuils_prediction"],
)

if (
    strategy_results["more_ecrans_prediction"]
    > strategy_results["more_fauteuils_prediction"]
):
    print(
        "Recommandation : Augmentez le nombre d'écrans. Cela devrait avoir un impact plus significatif."
    )
else:
    print(
        "Recommandation : Augmentez le nombre de fauteuils. Cela devrait avoir un impact plus significatif."
    )
