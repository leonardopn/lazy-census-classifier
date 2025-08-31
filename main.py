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
        # O arquivo original não tem cabeçalho, então definimos os nomes das colunas manualmente
        column_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]
        df = pd.read_csv(
            DATASET_FILE,
            header=None,
            names=column_names,
            sep=r"\s*,\s*",  # O separador tem espaços variáveis
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


# 3. Mapear o DataFrame para a estrutura do cbrkit (VERSÃO CORRETA)
def map_dataframe_to_casebase(df: pd.DataFrame) -> cbrkit.loaders.pandas:
    """
    Converte o DataFrame para a Casebase do cbrkit usando o wrapper pandas.
    """
    # A forma correta é passar o DataFrame diretamente.
    # Cada linha se torna um caso, e cada coluna um atributo.
    casebase = cbrkit.loaders.pandas(df)

    print("Mapeamento para a base de casos do cbrkit concluído.")
    return casebase


# ===================================== APP ===================================== #


def main():
    print(f"{'=' * 40} Iniciando o Processamento do Dataset de Renda {'=' * 40}\n")

    # Passos 1 e 2: Carregar e limpar os dados
    df = load_income_dataset()
    df_cleaned = clean_income_data(df)

    # Passo 3: Criar a base de casos
    casebase = map_dataframe_to_casebase(df_cleaned)

    # Verificar se a base de casos foi criada corretamente
    print(f"\nNúmero de casos na base: {len(casebase)}")
    if casebase:
        # Pega a chave do primeiro caso e o exibe para verificação
        first_case_key = next(iter(casebase.keys()))
        print("\nExemplo de um caso na base:")
        print(casebase[first_case_key])

    print(f"\n{'=' * 40} Fim do processamento {'=' * 40}\n")


if __name__ == "__main__":
    run_process("./", target=main)
