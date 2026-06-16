# 🏥 Medical Cost Predictor

Projeto de Machine Learning para análise e previsão de custos de seguro médico utilizando **Clusterização com K-Means** e **modelos de regressão**.

## 👥 Integrantes

* Mickael Cedraz
* Érica Araujo
* Monyc Luisa
* Pedro Cesar
* Nalbert de Souza
* Tiago Sângil

---

# 📂 Dataset

O projeto utiliza o **Insurance Cost Dataset**, contendo informações de clientes e seus respectivos custos de seguro médico.

Variáveis utilizadas:

* `age` — idade
* `sex` — sexo
* `bmi` — índice de massa corporal
* `children` — quantidade de filhos
* `smoker` — indicador de fumante
* `region` — região do cliente
* `charges` — custo do seguro (variável alvo)

---

# 🔧 Preparação dos Dados

O dataset passou pelos seguintes tratamentos:

### Label Encoding

Variáveis categóricas binárias foram transformadas:

* `female` → 0

* `male` → 1

* `no` → 0

* `yes` → 1

### One Hot Encoding

A variável `region` foi convertida em novas colunas:

* `region_northeast`
* `region_northwest`
* `region_southeast`
* `region_southwest`

### Normalização

Foi aplicado `StandardScaler` para padronizar as variáveis numéricas antes da aplicação dos algoritmos.

---

# 🔎 Clusterização

O método escolhido foi o **K-Means Clustering**.

O objetivo foi encontrar grupos de clientes com perfis semelhantes considerando características como:

* idade
* IMC
* quantidade de filhos
* sexo
* perfil de fumante

A variável `charges` não foi utilizada na criação dos clusters.

---

# 📊 Definição do Número de Clusters

Foram testados valores de K entre 2 e 10 utilizando:

* Método do Cotovelo (Elbow Method) através da inércia (WCSS)
* Silhouette Score

| K  | Inércia (WCSS) | Silhouette |
| -- | -------------- | ---------- |
| 2  | 5336.33        | 0.2238     |
| 3  | 4276.13        | 0.2625     |
| 4  | 3776.98        | 0.2516     |
| 5  | 3376.61        | 0.2285     |
| 6  | 3014.95        | 0.2492     |
| 7  | 2741.45        | 0.2718     |
| 8  | 2487.26        | 0.2893     |
| 9  | 2309.48        | 0.2859     |
| 10 | 2164.56        | 0.2809     |

Apesar do maior Silhouette Score ocorrer em K=8, foi escolhido **K=3** devido à melhor interpretabilidade dos grupos encontrados.

---

# 📌 Perfil dos Clusters

## Cluster 0 — Alto risco

Características:

* Idade média: 38.51 anos
* IMC médio: 30.70
* Filhos médio: 1.11
* Custo médio: US$ 32.050
* Total: 274 pessoas

Distribuição:

* 274 fumantes
* 0 não fumantes

Esse grupo representa clientes fumantes com custos elevados.

---

## Cluster 1 — Baixo custo

Características:

* Idade média: 39.10 anos
* IMC médio: 30.77
* Filhos médio: 1.09
* Custo médio: US$ 8.099
* Total: 516 pessoas

Distribuição:

* 516 não fumantes
* 0 fumantes

---

## Cluster 2 — Baixo custo

Características:

* Idade média: 39.69 anos
* IMC médio: 30.53
* Filhos médio: 1.08
* Custo médio: US$ 8.762
* Total: 547 pessoas

Distribuição:

* 547 não fumantes
* 0 fumantes

---

# 🤖 Modelos de Regressão

Foram testados três modelos:

* Regressão Linear
* KNN Regressor
* Decision Tree Regressor

A variável alvo foi:

`charges`

---

# 📈 Resultados da Regressão Global

| Modelo           | R²     | RMSE     | MAE     |
| ---------------- | ------ | -------- | ------- |
| Regressão Linear | 0.8019 | 6033.25  | 4246.81 |
| KNN Regressor    | 0.0843 | 12972.06 | 8651.49 |
| Decision Tree    | 0.8938 | 4416.77  | 2598.24 |

O melhor modelo global foi o **Decision Tree Regressor**.

---

# 📊 Regressão por Cluster

## Cluster 0

| Modelo        | R²     | RMSE    | MAE     |
| ------------- | ------ | ------- | ------- |
| Linear        | 0.7020 | 5825.13 | 4157.25 |
| KNN           | 0.2594 | 9182.86 | 6657.73 |
| Decision Tree | 0.7836 | 4964.14 | 2553.81 |

---

## Cluster 1

| Modelo        | R²      | RMSE     | MAE     |
| ------------- | ------- | -------- | ------- |
| Linear        | 0.6894  | 6619.69  | 4498.54 |
| KNN           | -0.1189 | 12564.13 | 8863.45 |
| Decision Tree | 0.7716  | 5676.09  | 3276.26 |

---

## Cluster 2

| Modelo        | R²      | RMSE     | MAE     |
| ------------- | ------- | -------- | ------- |
| Linear        | 0.7401  | 6733.49  | 4745.81 |
| KNN           | -0.0072 | 13254.17 | 9318.08 |
| Decision Tree | 0.8203  | 5598.82  | 2958.99 |

---

# ⚖️ Comparação dos Resultados

O modelo global apresentou melhores resultados quando comparado aos modelos treinados separadamente por cluster.

O Decision Tree global obteve:

* R²: 0.8938
* RMSE: 4416.77
* MAE: 2598.24

A divisão dos dados em clusters reduziu a quantidade de exemplos disponíveis para treinamento, causando uma pequena perda de desempenho.

---

# ✅ Conclusão

A clusterização identificou grupos principalmente relacionados ao perfil de fumantes e não fumantes, mostrando que essa variável possui grande influência no custo do seguro.

Apesar dos clusters encontrados apresentarem padrões diferentes de custo, utilizar os clusters como separação para modelos de regressão não trouxe melhora na previsão.

O melhor resultado foi obtido utilizando o **Decision Tree Regressor com todos os dados**, pois o modelo conseguiu capturar as relações não lineares presentes no problema.

O sistema final permite analisar perfis de clientes e realizar previsões de custo de seguro médico.
