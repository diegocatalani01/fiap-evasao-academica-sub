Predição de Evasão Acadêmica – Prova Substitutiva (Fase 3)

Aluno: Diego Catalani
RM: 359044

Este projeto foi desenvolvido para a prova substitutiva da Fase 3 da pós-graduação em Machine Learning Engineering (FIAP). O objetivo foi construir uma pipeline de machine learning para prever a evasão acadêmica a partir da base StudentsPrepared.xlsx, incluindo preparação dos dados, validação, treinamento do modelo e a criação de uma aplicação em Streamlit.

Base de Dados

A base contém 4.423 estudantes e 28 variáveis relacionadas ao desempenho acadêmico e a características demográficas.
A coluna Target tinha originalmente três categorias (Graduado, Desistente e Matriculado). Como o projeto exige uma classificação binária, ela foi convertida para:

1 = Desistente

0 = Graduado ou Matriculado

Pipeline do Modelo

A pipeline foi construída com ColumnTransformer, separando o pré-processamento de variáveis numéricas e categóricas.

Para as variáveis numéricas:

imputação pela mediana

padronização com StandardScaler

Para as variáveis categóricas:

imputação pela moda

One-Hot Encoding

O modelo usado foi o RandomForestClassifier, por ser robusto, lidar bem com diferentes tipos de dados e exigir poucos ajustes de hiperparâmetros.

Validação e Resultados

A validação foi feita com cross-validation estratificada (5 folds) utilizando AUC como métrica principal.
O resultado médio foi de 0,909, com baixa variação entre as dobras.

No conjunto de teste, o modelo apresentou:

AUC: 0,929

Acurácia: 87%

Recall da classe “evasão”: 69%

A AUC no treino foi 1,00, mostrando algum grau de overfitting esperado para Random Forest, mas o desempenho no teste se manteve elevado e consistente.

Aplicação em Streamlit

Foi criada uma aplicação em Streamlit que carrega o modelo treinado e permite inserir os dados de um aluno para obter a probabilidade estimada de evasão.
O formulário utiliza exatamente as mesmas variáveis da pipeline do modelo.

Para rodar localmente:

pip install -r requirements.txt
streamlit run app.py