import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# =========================
# Carregar dados
# =========================

df = pd.read_csv('data/processed_clusters.csv')


# =========================
# 1 - PCA dos clusters
# Mostra como o KMeans separou os dados
# =========================

features = df.drop(
    columns=['cluster', 'charges'],
    errors='ignore'
)


pca = PCA(n_components=2)

pca_result = pca.fit_transform(features)


plt.figure(figsize=(9,6))

sns.scatterplot(
    x=pca_result[:,0],
    y=pca_result[:,1],
    hue=df['cluster'],
    palette='Set1',
    s=80,
    alpha=0.7
)

plt.title(
    'Visualização dos Clusters encontrados pelo KMeans (PCA)'
)

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

plt.grid(True)

plt.savefig(
    'pca_clusters.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()



# =========================
# 2 - Idade x Custo
# Mostra o efeito de smoker
# =========================

plt.figure(figsize=(10,6))


sns.scatterplot(
    data=df,
    x='age',
    y='charges',
    hue='smoker',
    style='cluster',
    palette='Set1',
    alpha=0.7
)


plt.title(
    'Custos por idade separados por fumantes'
)

plt.xlabel('Idade')
plt.ylabel('Charges')


plt.grid(True)

plt.savefig(
    'idade_vs_charges.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()



# =========================
# 3 - Distribuição de custo por cluster
# =========================

plt.figure(figsize=(8,6))


sns.boxplot(
    data=df,
    x='cluster',
    y='charges'
)


plt.title(
    'Distribuição de Charges por Cluster'
)

plt.xlabel('Cluster')
plt.ylabel('Charges')


plt.grid(True)


plt.savefig(
    'charges_por_cluster.png',
    dpi=300,
    bbox_inches='tight'
)


plt.show()



# =========================
# 4 - Perfil médio dos clusters
# =========================

perfil = df.groupby('cluster').agg({
    'age':'mean',
    'bmi':'mean',
    'children':'mean',
    'charges':'mean'
})


print("\nPerfil médio dos clusters:")
print(perfil)