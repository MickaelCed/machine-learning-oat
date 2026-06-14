import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# !! Carregamento dos dados !!
X = pd.read_csv('data/X_ready.csv')
print(f'Dados carregados para clusterização: {X.shape}')

# Reduzindo de 75 colunas para as 5 combinações mais informativas
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X)
print(f"Dados após PCA: {X_pca.shape}")

# !! Aplicação do Elbow Method !!
print('\nCalculando a inérica para diferentes valores K...')
inercia = []
k_range = range(2,13)

# Calcular para cada K, de 2 a 12
for k in k_range :
    # random_state garante que o resultado seja o mesmo toda vez que rodar
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X_pca)
    inercia.append(kmeans.inertia_)

    print(f'K={k} calculated')

# !! Gerando o gráfico !!
plt.figure(figsize=(10,5))
plt.plot(k_range, inercia, 'ro-')
plt.xlabel('Número de Clústeres (K)')
plt.ylabel('Inércia')
plt.title('Elbow Method (Com dados em PCA)')
plt.grid(True)
plt.savefig('grafico_elbow.png', dpi=300, bbox_inches='tight')
print('\nGráfico salvo como "grafico_elbow.png"')

# !! Treinamento Definitivo do K-Means !!
kmeans_final = KMeans(n_clusters=12, n_init='auto', random_state=42)
kmeans_final.fit(X_pca)
X['cluster'] = kmeans_final.labels_

# !! Gerando gráfico de visualização !!
plt.figure(figsize=(10,7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=X['cluster'], palette='tab10', alpha=0.6)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualização dos Clústeres (Eixos PCA)')
plt.grid(True)
plt.savefig('grafico_clusters.png', dpi=300, bbox_inches='tight')
print('Gráfico salvo como "grafico_clusters.png"')

X.to_csv('data/X_with_clusters.csv', index=False)