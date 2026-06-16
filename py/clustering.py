import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# !! Carregamento dos dados !!
X = pd.read_csv('data/processed.csv')
print(f'Dados carregados para clusterização: {X.shape}')

# !! Aplicação do Elbow Method !!
print('\nCalculando a inérica para diferentes valores K...')
inercias = []
silhuetas = []
k_range = range(2,13)
colunas_perfil = [col for col in X.columns if not col.startswith('charges')]
X_perfil = X[colunas_perfil].copy()


# Calcular para cada K, de 2 a 12
for k in k_range :
    # random_state garante que o resultado seja o mesmo toda vez que rodar
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    kmeans.fit(X_perfil)

    # Inérica = Soma dos Quadrados da Distância
    inercias.append(kmeans.inertia_)
    # Silhouette Score = Quão bem separados estão os grupos
    score = silhouette_score(X_perfil, kmeans.labels_)
    silhuetas.append(score)

    print(f'K={k} calculado')

# !! Gerando o gráfico !!
plt.figure(figsize=(14,6))

# Elbow Method
plt.subplot(1, 2, 1)
plt.plot(k_range, inercias, 'bx-')
plt.xlabel('Número de Clústeres (K)')
plt.ylabel('Inércia')
plt.title('Elbow Method')
plt.grid(True)

# Silhouettes
plt.subplot(1, 2, 2)
plt.plot(k_range, silhuetas, 'ro-')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Score de Silhueta')
plt.title('Análise de Silhueta')
plt.grid(True)
plt.savefig('grafico.png', dpi=300, bbox_inches='tight')
print('\nGráfico salvo como "grafico.png"')

# Imprime na tela os valores consolidados para ajudar na análise técnica
print("\n=======================================================")
print("          TABELA DE ANÁLISE COMPLEMENTAR (K)           ")
print("=======================================================")
print(f"{'K':<5} | {'Inércia (WCSS)':<18} | {'Silhouette Score':<16}")
print("-" * 48)

for k, inercia, silhueta in zip(k_range, inercias, silhuetas):
    print(f"K={k:<3} | {inercia:<18.2f} | {silhueta:.4f}")
print("=======================================================")


# !! Aplicação do K-Means definitivo (K=3) !!
print('\nTreinando o modelo com K=3...')
kmeans_final = KMeans(n_clusters=3, n_init=20, random_state=42)
df_labels = kmeans_final.fit_predict(X_perfil)

# Dataset original p/ cruzar dados
df_original = pd.read_csv('data/medical_insurance.csv')
df_original['cluster'] = df_labels

X_regressao = X.copy()
X_regressao['cluster'] = df_labels
X_regressao.to_csv('data/processed_clusters.csv', index=False)
print('Dataset para regressão salvo em "data/X_clusters.csv" (Contém Regiões + Clusters)')

df_reg = pd.read_csv('data/medical_insurance.csv')

df_reg['cluster'] = df_labels

df_reg.to_csv(
    'data/regression_clusters.csv',
    index=False
)

# !! Interpratação dos Grupos !!
print("\n=======================================================")
print("             PERFIL DOS CLUSTERS ENCONTRADOS           ")
print("=======================================================\n")

perfil_clusters = df_original.groupby('cluster').agg({
    'age': 'mean',
    'bmi': 'mean',
    'children': 'mean',
    'charges': ['mean', 'count'] # Média de custo e quantidade de pessoas no grupo
}).reset_index()

# Renomeando as colunas para ficar legível
perfil_clusters.columns = ['Cluster', 'Idade Média', 'IMC Médio', 'Filhos Média', 'Custo Médio (US$)', 'Total de Pessoas']
print(perfil_clusters.to_string(index=False))

# Vamos ver a distribuição de fumantes por cluster
print("\n--- Distribuição de Fumantes por Cluster ---")
print(pd.crosstab(df_original['cluster'], df_original['smoker']))

# Vamos ver a distribuição de regiões por cluster
print("\n--- Distribuição de Regiões por Cluster ---")
print(pd.crosstab(df_original['cluster'], df_original['region']))