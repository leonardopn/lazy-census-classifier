# lazy-census-classifier - Classificador de Renda com Raciocínio Baseado em Casos (CBR)

Este projeto é um classificador de aprendizado de máquina desenvolvido como parte da disciplina de Pós-Graduação em Aprendizado de Máquina. O sistema utiliza a abordagem de Raciocínio Baseado em Casos (CBR), um paradigma de _Lazy Learning_, para prever se a renda anual de um indivíduo excede $50.000 com base em dados demográficos do censo.

O protótipo foi implementado em Python utilizando a biblioteca `cbrkit` e foca na aplicação de conceitos teóricos de CBR, como a construção de funções de similaridade híbridas, ponderação de atributos e otimização do reúso com k-NN.

## 🎯 Objetivo

O objetivo principal do trabalho é construir um sistema CBR funcional que cumpra as etapas de **Recuperação** e **Reúso** do ciclo CBR. A performance do sistema é validada objetivamente através do método de teste **Leave-One-Out Cross-Validation (LOOCV)**, conforme especificado nos requisitos da disciplina.

## ✨ Principais Funcionalidades

O sistema implementa várias técnicas para otimizar a precisão da classificação:

-   **Similaridade Híbrida:** Utiliza diferentes funções de similaridade locais para cada tipo de atributo, combinando `similaridade linear` para dados numéricos (ex: idade) e `similaridade de igualdade` para dados categóricos (ex: escolaridade).
-   **Ponderação de Atributos:** Aplica uma média ponderada para agregar as similaridades locais, permitindo que atributos mais relevantes (como `education` e `occupation`) tenham maior influência no resultado final.
-   **Reúso com k-NN (k-Nearest Neighbors):** Em vez de usar apenas o caso mais similar (1-NN), o sistema recupera os `k` vizinhos mais próximos e realiza uma votação majoritária para determinar a classe final, tornando o classificador mais robusto a ruídos.
-   **Avaliação Paralelizada:** O teste de performance com _Leave-One-Out_ é executado em paralelo utilizando `concurrent.futures`, acelerando significativamente o processo de validação.

## 📊 Dataset

O projeto utiliza o dataset [**"Adult Census Income"**](https://www.kaggle.com/datasets/mosapabdelghany/adult-income-prediction-dataset), obtido do repositório UCI Machine Learning. Ele contém informações demográficas de mais de 30.000 indivíduos e o objetivo é classificar a coluna `income` em uma de duas categorias: `<=50K` ou `>50K`.

O CSV com os dados está localizado na pasta `datasets`.

## 🛠️ Tecnologias Utilizadas

-   **Linguagem:** Python 3.12+
-   **Biblioteca CBR:** `cbrkit`
-   **Manipulação de Dados:** `pandas`
-   **Execução Paralela:** `concurrent.futures`
-   **Desenvolvimento:** `watchfiles` para recarregamento automático

## 🚀 Como Executar

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/leonardopn/lazy-census-classifier
    cd lazy-census-classifier
    ```

2.  **Instale as dependências:**
    _O projeto utiliza [`uv`](https://docs.astral.sh/uv/) para gerenciamento de pacotes. [Clique aqui ](https://docs.astral.sh/uv/) para baixar o uv se for necessário._

    ```bash
    uv run main.py
    ```

## 📈 Resultados e Análise de Performance

Para validar a eficácia do classificador e o impacto das otimizações, foi conduzida uma série de testes utilizando o método _Leave-One-Out_ em uma amostra de 500 casos. As estratégias testadas incluíram a ponderação de atributos e a variação do número de vizinhos (k) no algoritmo k-NN.

### Resumo dos Resultados

A análise dos testes revela uma jornada clara de otimização:

1.  **Impacto do k-NN:** A mudança de um classificador 1-NN para um 5-NN (Teste 1 vs. Teste 3) resultou no ganho de performance mais significativo, com um salto de **19 acertos**, provando que o sistema se torna mais robusto ao considerar múltiplos vizinhos para a votação da classe.
2.  **Eficácia da Ponderação:** A aplicação de pesos nos atributos consistentemente melhorou ou manteve a acurácia do modelo. O maior ganho com esta técnica foi observado no modelo 10-NN (Teste 5 vs. Teste 6), com um aumento de **4 acertos**, o que levou o sistema ao seu pico de performance.
3.  **Ponto Ótimo de Performance:** Os testes indicam que o ponto ideal para o modelo foi alcançado com **k=10 e com pesos**, atingindo **419 acertos** e uma acurácia máxima de **83.80%**.
4.  **Início da Perda (Overfitting):** Ao aumentar o número de vizinhos para k=15, a performance começou a decair. Isso sugere que a "vizinhança" se tornou muito ampla, incluindo casos menos relevantes na votação e diminuindo a capacidade de generalização do modelo.

### Tabela Comparativa de Resultados

| Estratégia do Teste            |                  Nº de Acertos (de 500)                  |                          Acurácia                           |                   Melhora (vs. anterior)                    |
| ------------------------------ | :------------------------------------------------------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| Teste 1 (1-NN, sem pesos)      |                           391                            |                           78.20%                            |                              -                              |
| Teste 2 (1-NN, **com** pesos)  |                           393                            |                           78.60%                            |                           +0.40%                            |
| Teste 3 (5-NN, sem pesos)      |                           410                            |                           82.00%                            | <span style="color:green;font-weight:bold">👑 +3.40%</span> |
| Teste 4 (5-NN, **com** pesos)  |                           410                            |                           82.00%                            |                           +0.00%                            |
| Teste 5 (10-NN, sem pesos)     |                           415                            |                           83.00%                            |                           +1.00%                            |
| Teste 6 (10-NN, **com** pesos) | <span style="color:green;font-weight:bold">👑 419</span> | <span style="color:green;font-weight:bold">👑 83.80%</span> |                           +0.80%                            |
| Teste 7 (15-NN, sem pesos)     |                           415                            |                           83.00%                            |  <span style="color:red;font-weight:bold">👎 -0.80%</span>  |
| Teste 8 (15-NN, **com** pesos) |                           415                            |                           83.00%                            |                            0.00%                            |
