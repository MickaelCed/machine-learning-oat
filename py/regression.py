import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Função auxiliar para calcular todas as métricas de uma vez
def avaliar_modelo(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# 1. CARREGAR DADOS 
df = pd.read_csv('data/processed_clusters.csv') 
df = pd.get_dummies(df, columns=['cluster'], dtype=int, drop_first=True)

X = df.drop(columns=['charges'])
y = df['charges']

# Dicionário com os modelos que vamos testar
modelos = {
    'Regressão Linear': LinearRegression(),
    'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- RESULTADOS REGRESSÃO GLOBAL ---")
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)
    metricas = avaliar_modelo(y_test, preds)
    print(f"\n> {nome}:")
    print(f"  R²: {metricas['R2']:.4f} | RMSE: {metricas['RMSE']:.2f} | MAE: {metricas['MAE']:.2f}")


print("\n--- RESULTADOS REGRESSÃO POR CLUSTER ---")

# Vamos ler o arquivo novamente para garantir que a coluna 'cluster' esteja íntegra (sem get_dummies global)
df_clusters = pd.read_csv('data/processed_clusters.csv')

# Pegar a lista de clusters únicos de forma ordenada [0, 1, 2, 3]
clusters_unicos = sorted(df_clusters['cluster'].unique())

for c in clusters_unicos:
    print(f"\n================ CLUSTER {c} ================")
    
    # 1. Filtrar o dataset para conter apenas os dados do cluster atual
    df_c = df_clusters[df_clusters['cluster'] == c]
    
    # 2. Separar Entrada (X) e Alvo (y)
    # Como estamos DENTRO do cluster específico, removemos a coluna 'cluster' pois ela virou uma constante
    X_c = df_c.drop(columns=['charges', 'cluster'])
    y_c = df_c['charges']
    
    # 3. Divisão de Treino e Teste (80% / 20%) para este cluster específico
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
    
    # 4. Treinar e avaliar cada um dos 3 algoritmos permitidos
    for nome, modelo in modelos.items():
        # Treina o modelo específico do algoritmo para o cluster C
        modelo.fit(X_train_c, y_train_c)
        
        # Faz as previsões nos dados de teste do cluster C
        preds_c = modelo.predict(X_test_c)
        
        # Calcula as métricas locais
        metricas_c = avaliar_modelo(y_test_c, preds_c)
        
        # Exibe os resultados formatados
        print(f"> {nome:16} -> R²: {metricas_c['R2']:.4f} | RMSE: {metricas_c['RMSE']:.2f} | MAE: {metricas_c['MAE']:.2f}")

