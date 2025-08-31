from typing import TypedDict
import cbrkit
import pandas as pd
from watchfiles import run_process

# Variável para o novo arquivo de dataset
DATASET_FILE = "./datasets/adult.csv"


class Case(TypedDict):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    income: str


# 1. Carregar o novo dataset
def load_income_dataset() -> pd.DataFrame:
    """
    Carrega o dataset 'adult.csv' e faz uma limpeza inicial.
    """
    try:
        df = pd.read_csv(
            DATASET_FILE,
            sep=",",
            na_values="?",  # Trata os '?' como valores nulos
            engine="python",
        )
        print("Dataset 'Adult Income' carregado com sucesso.")
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo '{DATASET_FILE}' não encontrado.")
        exit()


# 2. Limpeza e preparação dos dados
def clean_income_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas com dados faltantes e seleciona os atributos mais relevantes.
    """
    # Remove linhas que contêm qualquer valor nulo
    df.dropna(inplace=True)

    # Para simplificar, vamos remover algumas colunas que são redundantes ou menos importantes
    df = df.drop(columns=["fnlwgt", "education-num", "capital-gain", "capital-loss"])

    print(
        "Limpeza de dados concluída. Linhas com dados faltantes e colunas irrelevantes removidas."
    )

    # Pega uma amostra para trabalhar mais rápido inicialmente
    df_sample = df.sample(n=5000, random_state=42)
    print(f"Trabalhando com uma amostra de {len(df_sample)} casos.")

    return df_sample


# 3. Mapear o DataFrame para a estrutura do cbrkit
def map_dataframe_to_casebase(df: pd.DataFrame) -> cbrkit.loaders.pandas:
    """
    Converte o DataFrame para a Casebase do cbrkit usando o wrapper pandas.
    """
    # A forma correta é passar o DataFrame diretamente.
    # Cada linha se torna um caso, e cada coluna um atributo.
    casebase = cbrkit.loaders.pandas(df)

    print("Mapeamento para a base de casos do cbrkit concluído.")
    return casebase


# 4. Construir a função de similaridade global
def build_similarity_function(df: pd.DataFrame) -> cbrkit.sim.attribute_value:
    """
    Cria a função de similaridade global combinando funções locais para cada atributo.
    """
    # Identifica os tipos de atributos
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    # Remove a coluna 'income' da lista, pois ela é a solução, não parte do problema
    categorical_cols.remove("income")

    # Define as funções de similaridade locais
    # Usamos um dicionário para mapear cada atributo à sua função de similaridade
    local_similarities: dict[
        str, cbrkit.sim.numbers.linear | cbrkit.sim.generic.equality
    ] = {
        # Para cada coluna numérica, usamos a similaridade linear
        # O max é o valor máximo na coluna, usado para normalizar a distância
        col: cbrkit.sim.numbers.linear(max=df[col].max())
        for col in numeric_cols
    }

    local_similarities.update(
        {
            # Para cada coluna categórica, usamos a similaridade de igualdade
            col: cbrkit.sim.generic.equality()
            for col in categorical_cols
        }
    )

    # Cria a função de similaridade global
    # `attribute_value` aplica a função local correta para cada atributo do caso
    similarity_func = cbrkit.sim.attribute_value(
        attributes=local_similarities,
        # O agregador `mean` calcula a média das similaridades locais
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    )

    print("Função de similaridade global construída com sucesso.")
    return similarity_func


# 5. Executar a recuperação e o reúso para classificar um novo caso
def perform_retrieval_and_reuse(
    casebase: cbrkit.loaders.pandas, similarity_func: cbrkit.sim.attribute_value
):
    """
    Usa a base de casos e a função de similaridade para classificar um novo caso.
    """
    print(f"\n{'=' * 40} Iniciando Classificação de um Novo Caso {'=' * 40}\n")

    # 1. Construir o recuperador
    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            similarity_func=similarity_func,
        ),
        limit=1,
    )

    # 2. Criar uma consulta (query)
    query_dict = {
        "age": 20,
        "workclass": "Private",
        "education": "Masters",
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "Black",
        "hours-per-week": 1,
        "native-country": "United-States",
        "sex": "Female",
    }
    print("Classificando o seguinte novo caso (consulta):")
    for key, value in query_dict.items():
        print(f"  - {key}: {value}")

    # Isso garante que a estrutura da query seja idêntica à dos casos na casebase.
    query = pd.Series(query_dict)

    # 3. Executar a recuperação
    retrieved_result = cbrkit.retrieval.apply_query(casebase, query, retriever)

    # Verifica se há um resultado
    if not retrieved_result.casebase:
        print("\nNenhum caso similar foi encontrado.")
        return

    # 4. Exibir os resultados
    # Pega o ID do primeiro (e único) caso recuperado
    matched_case_id = next(iter(retrieved_result.casebase.keys()))
    matched_case_data = retrieved_result.casebase[matched_case_id]
    similarity_data = retrieved_result.similarities[matched_case_id]

    predicted_income = matched_case_data["income"]

    # Exibe a similaridade detalhada por atributo e a similaridade global
    print(f"\nSimilaridade detalhada: {similarity_data}")
    print(
        f"\nCaso mais similar encontrado (ID: {matched_case_id}) com similaridade de {similarity_data.value*100:.2f}%"
    )
    print(f"Previsão de Renda para o novo caso: {predicted_income}")


def main():
    print(f"{'=' * 40} Iniciando o Processamento do Dataset de Renda {'=' * 40}\n")

    # Passos 1 e 2: Carregar e limpar os dados
    df = load_income_dataset()
    df_cleaned = clean_income_data(df)

    print(f"\n{'=' * 40} Tipos de dados que serão tratados {'=' * 40}\n")
    print(df_cleaned.info())
    print(f"{'=' * 80}\n")

    # Passo 3: Criar a base de casos
    casebase = map_dataframe_to_casebase(df_cleaned)
    print(f"Número de casos na base: {len(casebase)}")

    # Passo 4: Construir a função de similaridade global
    similarity_func = build_similarity_function(df_cleaned)

    # Passo 5: Executar a recuperação e o reúso para fazer uma classificação
    perform_retrieval_and_reuse(casebase, similarity_func)

    print(f"\n{'=' * 40} Fim do processamento {'=' * 40}\n")


if __name__ == "__main__":
    run_process("./", target=main)
