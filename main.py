from typing import TypedDict
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


# ===================================== APP ===================================== #


def main():
    print(f"{'=' * 40} Iniciando o Processamento do Dataset de Renda {'=' * 40}\n")

    # Passos 1 e 2: Carregar e limpar os dados
    df = load_income_dataset()
    df_cleaned = clean_income_data(df)

    print("\nVisualização das 5 primeiras linhas do dataset limpo:")
    print(df_cleaned.head())

    print(f"\nTipos de dados das colunas:")
    print(df_cleaned.info())

    print(f"\n{'=' * 40} Fim do processamento {'=' * 40}\n")


if __name__ == "__main__":
    run_process("./", target=main)
