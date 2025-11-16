import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# ========= Caminhos =========
# Pasta onde o script está
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho ABSOLUTO para a base dentro da pasta data
DATA_PATH = os.path.join(BASE_DIR, "data", "StudentsPrepared.xlsx")

# Coluna alvo original (multiclasse)
TARGET_COL = "Target"


def carregar_dados(path: str) -> pd.DataFrame:
    """
    Carrega a base Excel de estudantes preparada para o modelo de evasão.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado em: {path}")

    df = pd.read_excel(path)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def separar_features_target(df: pd.DataFrame, target_col: str):
    """
    Separa X (features) e y (alvo) e converte o problema para BINÁRIO:
    - 1 = Desistente (evasão)
    - 0 = Demais (Graduado, Matriculado)
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Coluna alvo '{target_col}' não encontrada na base. "
            f"Verifique o nome da coluna de evasão."
        )

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    print("\nValores únicos em Target (original):")
    print(y_raw.value_counts())

    # Converte para binário: Desistente = 1, outros = 0
    if y_raw.nunique() > 2:
        print(
            "\n[INFO] Convertendo Target para problema binário: "
            "1 = Desistente (evasão), 0 = demais (não evasão)."
        )
        y = (y_raw == "Desistente").astype(int)
    else:
        y = y_raw

    print("\nDistribuição da Target binária (0 = não evasão, 1 = evasão):")
    print(y.value_counts(normalize=True))

    return X, y


def criar_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Cria o pipeline completo: pré-processamento + modelo.

    - Numéricos: imputação (mediana) + StandardScaler
    - Categóricos: imputação (moda) + OneHotEncoder
    - Modelo: RandomForestClassifier
    """
    numeric_features = selector(dtype_include=["int64", "float64"])(X)
    categorical_features = selector(dtype_include=["object", "bool", "category"])(X)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    clf = Pipeline(steps=[("preprocessamento", preprocessor), ("modelo", model)])
    return clf


def avaliar_overfitting_underfitting(modelo: Pipeline, X_train, y_train, X_test, y_test):
    """
    Compara desempenho em treino x teste e imprime classification report,
    matriz de confusão e AUC.
    """
    # Predição
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)

    # Probabilidades (para AUC)
    train_proba = modelo.predict_proba(X_train)[:, 1]
    test_proba = modelo.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)

    print("\n=== Distribuição da classe alvo (y) no conjunto total ===")
    print(pd.Series(np.concatenate([y_train, y_test])).value_counts(normalize=True))

    print("\n=== AUC Treino vs Teste ===")
    print(f"AUC Treino: {train_auc:.3f}")
    print(f"AUC Teste : {test_auc:.3f}")

    diff = train_auc - test_auc
    if diff > 0.05:
        print("\n>> Interpretação: possível OVERFITTING "
              "(desempenho significativamente melhor em treino).")
    elif diff < -0.02:
        print("\n>> Interpretação: possível UNDERFITTING "
              "(modelo não aprendeu bem o padrão).")
    else:
        print("\n>> Interpretação: modelo aparentemente bem balanceado "
              "entre viés e variância.")

    print("\n=== Classification Report (Teste) ===")
    print(classification_report(y_test, y_test_pred))

    print("\n=== Matriz de Confusão (Teste) ===")
    print(confusion_matrix(y_test, y_test_pred))


def main():
    print("Base dir:", BASE_DIR)
    print("Caminho da base:", DATA_PATH)
    print("Carregando base...")
    df = carregar_dados(DATA_PATH)
    print("Dimensão da base:", df.shape)
    print("Colunas:")
    print(list(df.columns))

    X, y = separar_features_target(df, TARGET_COL)

    # Separação treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("\nTamanho treino:", X_train.shape, "| Tamanho teste:", X_test.shape)

    # Cria pipeline completo
    clf = criar_pipeline(X)

    # Validação cruzada (apenas no treino)
    print("\nRodando validação cruzada (AUC, 5-fold, estratificada)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_auc = cross_val_score(
        clf,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    print("AUC em cada fold:", np.round(scores_auc, 3))
    print("Média AUC:", scores_auc.mean().round(3),
          "+/-", scores_auc.std().round(3))

    # Treina modelo final no treino completo
    print("\nTreinando modelo final no conjunto de treino...")
    clf.fit(X_train, y_train)

    # Avaliação final em treino x teste
    avaliar_overfitting_underfitting(clf, X_train, y_train, X_test, y_test)

    # Salva o modelo
    modelo_path = os.path.join(BASE_DIR, "modelo_evasao.pkl")
    print(f"\nSalvando modelo treinado em '{modelo_path}'...")
    joblib.dump(clf, modelo_path)
    print("Modelo salvo com sucesso!")


if __name__ == "__main__":
    main()
