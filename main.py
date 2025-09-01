from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal, NotRequired, TypedDict, Mapping
import cbrkit
import pandas as pd
from watchfiles import run_process

from helpers.logger_block import logger_block
import random

# Variável para o novo arquivo de dataset
DATASET_FILE = "./datasets/adult.csv"

HEADER = Literal[
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


class Case(TypedDict):
    age: int
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    hours_per_week: int
    native_country: str
    income: NotRequired[str]  # Pode ser '>50K' ou '<=50K'


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
    df = df.drop(columns=["fnlwgt", "education_num", "capital_gain", "capital_loss"])

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
def build_similarity_function(
    df: pd.DataFrame, use_weights: bool
) -> cbrkit.sim.attribute_value:
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

    # Defina os pesos para cada atributo (valores maiores = mais importante)
    # A soma não precisa ser 1.
    typed_weights: Mapping[HEADER, float] = {
        "education": 1.8,  # Nível educacional é o preditor mais forte de renda elevada
        "occupation": 1.6,  # Cargos gerenciais e especializados fortemente associados a alta renda
        "age": 1.4,  # Forte correlação com renda - experiência profissional acumulada
        "hours_per_week": 1.4,  # Horas trabalhadas correlacionam diretamente com potencial de renda
        "workclass": 1.2,  # Tipo de empregador influencia significativamente (auto-empregados, governo)
        "marital_status": 1.0,  # Estado civil tem impacto moderado na classificação de renda
        "relationship": 1.0,  # Posição familiar tem correlação moderada com padrões de renda
        "sex": 0.8,  # Apresenta correlação, mas menos determinante que fatores profissionais
        "race": 0.6,  # Menor poder preditivo independente, mas ainda com alguma influência
        "native_country": 0.6,  # País de origem tem impacto limitado comparado a fatores profissionais
    }

    aggregator = cbrkit.sim.aggregator[HEADER](
        "mean", typed_weights if use_weights else None
    )

    # Cria a função de similaridade global com média ponderada
    similarity_func = cbrkit.sim.attribute_value[HEADER, float](
        attributes=local_similarities,
        aggregator=aggregator,  # type: ignore
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
    # 1. Construir o recuperador
    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            similarity_func=similarity_func,
        ),
        limit=1,
    )

    # 2. Criar uma consulta de exemplo
    query_dict: Case = {
        "age": 20,
        "workclass": "Private",
        "education": "Masters",
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "Black",
        "hours_per_week": 1,
        "native_country": "United-States",
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


def evaluate_single_case(args) -> bool:
    """
    Executa a avaliação para um único caso.
    Retorna True se a previsão for correta, False caso contrário.
    """
    case_id_to_hold_out, full_casebase, similarity_func, k = args

    holdout_case = full_casebase[case_id_to_hold_out]
    correct_solution = holdout_case["income"]
    query = holdout_case.drop("income")

    # Cria a base de casos para o teste (todos, exceto o holdout)
    temp_df = pd.DataFrame(
        [case for idx, case in full_casebase.items() if idx != case_id_to_hold_out]
    )
    test_casebase = cbrkit.loaders.pandas(temp_df)

    # Constrói e aplica o retriever
    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            similarity_func=similarity_func,
        ),
        limit=k,
    )
    retrieved_result = cbrkit.retrieval.apply_query(test_casebase, query, retriever)

    if retrieved_result.casebase:
        votes = [case["income"] for case in retrieved_result.casebase.values()]
        predicted_solution = max(set(votes), key=votes.count)
        return predicted_solution == correct_solution

    return False


# 6. Avaliar o sistema em PARALELO com Leave-One-Out e Votação k-NN
def evaluate_with_leave_one_out(
    full_casebase: cbrkit.loaders.pandas,
    similarity_func: cbrkit.sim.attribute_value,
    sample_size: int = 500,
    k: int = 5,
):
    """
    Avalia o classificador em paralelo usando ProcessPoolExecutor.
    """
    all_case_ids = list(full_casebase.keys())
    case_ids_to_test = all_case_ids[:sample_size]

    tasks = [
        (case_id, full_casebase, similarity_func, k) for case_id in case_ids_to_test
    ]

    correct_predictions = 0
    # Com o import correto, ProcessPoolExecutor e as_completed funcionarão juntos
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_single_case, task) for task in tasks]

        # Esta linha agora usará a função correta de concurrent.futures
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                correct_predictions += 1

            if (i + 1) % 20 == 0:
                print(
                    f"  Casos processados: {i + 1}/{sample_size}. Acurácia atual: {(correct_predictions / (i + 1)) * 100:.2f}%"
                )

    accuracy = (correct_predictions / sample_size) * 100
    print(
        "\n-------------------- Resultado da Avaliação (k-NN, Paralelo) --------------------"
    )
    print(f"Total de casos testados: {sample_size}")
    print(f"Número de vizinhos (k): {k}")
    print(f"Previsões corretas: {correct_predictions}")
    print(f"Acurácia do sistema: {accuracy:.2f}%")


def main():
    sample_size = 500
    k = 10
    use_weights = True

    logger_block(
        "Iniciando o Processamento do Dataset de Renda",
    )

    # Passos 1 e 2: Carregar e limpar os dados
    df = load_income_dataset()
    df_cleaned = clean_income_data(df)

    logger_block(
        "Exibindo informações do DataFrame limpo",
    )
    print(df_cleaned.info())

    # Passo 3: Criar a base de casos
    logger_block(
        "Criando a base de casos a partir do DataFrame",
    )
    casebase = map_dataframe_to_casebase(df_cleaned)
    print(f"Número de casos na base: {len(casebase)}")

    # Passo 4: Construir a função de similaridade global
    logger_block(
        "Construindo a função de similaridade global",
    )
    similarity_func = build_similarity_function(df_cleaned, use_weights=False)

    # Passo 5: Executar a recuperação e o reúso para fazer uma classificação
    logger_block(
        "Executando a recuperação e o reúso",
    )
    perform_retrieval_and_reuse(casebase, similarity_func)

    logger_block(
        f"Iniciando Avaliação Leave-One-Out com k={k} {'com pesos' if use_weights else 'sem pesos'} (amostra de {sample_size} casos)",
    )
    evaluate_with_leave_one_out(casebase, similarity_func, sample_size, k=k)

    logger_block(
        "Fim do processamento",
    )


if __name__ == "__main__":
    run_process("./", target=main)
