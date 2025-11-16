import os
import joblib
import pandas as pd
import streamlit as st


# Caminhos básicos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "StudentsPrepared.xlsx")
import zipfile

zip_path = os.path.join(BASE_DIR, "modelo_evasao.zip")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_evasao.pkl")

# Se o arquivo ainda não estiver descompactado, descompactamos
if not os.path.exists(MODEL_PATH):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)

TARGET_COL = "Target"


@st.cache_data
def carregar_base(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


@st.cache_resource
def carregar_modelo(path: str):
    modelo = joblib.load(path)
    return modelo


def montar_formulario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monta um formulário com todas as colunas de entrada do modelo.
    Para numéricos usa number_input, para categóricos usa selectbox.
    Retorna um DataFrame com uma única linha, com as mesmas colunas da base.
    """
    if TARGET_COL in df.columns:
        X = df.drop(columns=[TARGET_COL])
    else:
        X = df.copy()

    st.subheader("Preencha os dados do aluno")

    dados = {}

    for col in X.columns:
        serie = X[col]

        if pd.api.types.is_numeric_dtype(serie):
            # valores de referência
            minimo = float(serie.min())
            maximo = float(serie.max())
            mediana = float(serie.median())

            # um step "genérico"
            step = (maximo - minimo) / 100 if maximo != minimo else 1.0

            dados[col] = st.number_input(
                label=col,
                min_value=minimo,
                max_value=maximo,
                value=mediana,
                step=step,
            )
        else:
            opcoes = sorted(serie.dropna().unique().tolist())
            if not opcoes:
                opcoes = [""]
            valor_padrao = opcoes[0]
            dados[col] = st.selectbox(col, opcoes, index=0)

    entrada = pd.DataFrame([dados])
    return entrada


def main():
    st.title("Previsão de Evasão Acadêmica")

    st.write(
        "App desenvolvido para a prova substitutiva da Fase 3 "
        "Pós-graduação em Machine Learning Engineering (FIAP)."
    )

    df = carregar_base(DATA_PATH)
    modelo = carregar_modelo(MODEL_PATH)

    if df.empty:
        st.error("Não foi possível carregar a base de dados.")
        return

    # formulário
    with st.form("form_predicao"):
        entrada = montar_formulario(df)
        enviar = st.form_submit_button("Calcular risco de evasão")

    if enviar:
        try:
            prob = modelo.predict_proba(entrada)[0][1]
            pred = modelo.predict(entrada)[0]

            st.subheader("Resultado da previsão")
            st.write(f"Probabilidade estimada de evasão: {prob:.2%}")
            st.write(f"Classe prevista (0 = não evasão, 1 = evasão): {int(pred)}")

            if prob >= 0.5:
                st.warning("O aluno apresenta risco elevado de evasão.")
            else:
                st.info("O aluno apresenta risco mais baixo de evasão.")
        except Exception as e:
            st.error(f"Ocorreu um erro ao gerar a previsão: {e}")


if __name__ == "__main__":
    main()
