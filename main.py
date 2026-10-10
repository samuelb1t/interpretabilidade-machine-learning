from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from interpretabilidade import interpretar_arvore

dataset = Path(__file__).parent / "dataset.csv"

features = [
    "age",
    "gender",
    "primary_platform",
    "daily_short_form_hours",
    "avg_clip_duration_sec",
    "swipes_per_minute",
    "scroll_sessions_day",
    "bedtime_scroll_min",
    "total_sleep_hours",
    "multitasking_score",
    "total_screen_time_hours",
]

categoricas = ["gender", "primary_platform"]
numericas = [coluna for coluna in features if coluna not in categoricas]

def preparar_dados():
    dados = pd.read_csv(dataset)

    # Remove dados duplicados
    dados = dados.drop_duplicates().copy()

    # Pega só os dados escolhidos
    X = dados[features].copy()
    y = dados["cognitive_decay_tier"].copy()

    print("\nDistribuição das classes:")
    print(y.value_counts())

    return X, y

def criar_preprocessamento(padronizar=False):
    return ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler() if padronizar else "passthrough", numericas),
            ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
        ]
    )

def treinar_knn(X_train, y_train):
    modelo = Pipeline([
        ("preprocessamento", criar_preprocessamento(padronizar=True)),
        ("classificador", KNeighborsClassifier(n_neighbors=5)),
    ])
    modelo.fit(X_train, y_train)
    return modelo

def treinar_naive_bayes(X_train, y_train):
    modelo = Pipeline([
        ("preprocessamento", criar_preprocessamento()),
        ("classificador", GaussianNB()),
    ])
    modelo.fit(X_train, y_train)
    return modelo

def treinar_arvore_decisao(X_train, y_train):
    modelo = Pipeline([
        ("preprocessamento", criar_preprocessamento()),
        ("classificador", DecisionTreeClassifier(random_state=42)),
    ])
    modelo.fit(X_train, y_train)
    return modelo

def avaliar_modelo(nome, modelo, X_test, y_test):
    # predict aplica o pré-processamento já ajustado no treino, sem reajustá-lo.
    previsoes = modelo.predict(X_test)
    classes = modelo.classes_

    print(f"\n--- {nome} ---")
    print(classification_report(y_test, previsoes, labels=classes, zero_division=0))
    print("Matriz de confusão (linhas = real; colunas = previsto):")
    print(pd.DataFrame(
        confusion_matrix(y_test, previsoes, labels=classes),
        index=classes,
        columns=classes,
    ).to_string())

    return {
        "Modelo": nome,
        "Acurácia": accuracy_score(y_test, previsoes),
        "F1 macro": f1_score(y_test, previsoes, average="macro", zero_division=0),
        "F1 ponderado": f1_score(y_test, previsoes, average="weighted", zero_division=0),
    }

if __name__ == "__main__":
    X, y = preparar_dados()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"\nTreino: {len(X_train)} registros")
    print(f"Teste: {len(X_test)} registros")

    knn = treinar_knn(X_train, y_train)
    naive_bayes = treinar_naive_bayes(X_train, y_train)
    arvore = treinar_arvore_decisao(X_train, y_train)
    print("\nKNN, Naïve Bayes e Árvore de Decisão treinados.")

    resultados = [
        avaliar_modelo("KNN", knn, X_test, y_test),
        avaliar_modelo("Naïve Bayes", naive_bayes, X_test, y_test),
        avaliar_modelo("Árvore de Decisão", arvore, X_test, y_test),
    ]
    print("\nComparação no conjunto de teste:")
    print(pd.DataFrame(resultados).set_index("Modelo").round(4))

    interpretar_arvore(arvore, X_test, y_test)
