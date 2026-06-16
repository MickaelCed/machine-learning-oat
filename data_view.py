import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Carregar a base tratada
caminho_arquivo = "data/processed.csv"

if not os.path.exists(caminho_arquivo):
    print(f"[Erro] O arquivo '{caminho_arquivo}' não foi encontrado.")
    print("Certifique-se de que o script está na mesma pasta do arquivo CSV.")
    exit()

df = pd.read_csv(caminho_arquivo)

# Configuração visual geral dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("Gerando análises visuais... Feche a janela de um gráfico para abrir o próximo.")

# ==========================================
# GRÁFICO 1: O Impacto Avassalador do Fumo (Boxplot)
# ==========================================
plt.figure()
sns.boxplot(x="smoker", y="charges", data=df, palette="Set2")
plt.title("Distribuição de Custos: Não-Fumantes (0) vs Fumantes (1)", fontsize=14, fontweight="bold")
plt.xlabel("Fumante (0 = Não, 1 = Sim)", fontsize=12)
plt.ylabel("Custo (Charges)", fontsize=12)
plt.xticks([0, 1], ["Não-Fumante (0)", "Fumante (1)"])
print("\n[Exibindo Gráfico 1] Análise de impacto do Tabagismo.")
plt.show()

# ==========================================
# GRÁFICO 2: Correlação Linear de Todas as Variáveis (Heatmap)
# ==========================================
plt.figure(figsize=(11, 8))
# Calcula a matriz de correlação de Pearson
matriz_correlacao = df.corr()

# Destaca a linha/coluna do 'charges' para ver quem tem maior valor perto de 1 ou -1
sns.heatmap(matriz_correlacao, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Mapa de Calor: Quais variáveis possuem maior correlação com 'charges'?", fontsize=14, fontweight="bold")
print("[Exibindo Gráfico 2] Matriz de Correlação Geral.")
plt.show()

# ==========================================
# GRÁFICO 3: A Combinação Crítica - Idade, IMC e Fumo (Scatterplot)
# ==========================================
plt.figure()
# Gráfico de dispersão cruzando Idade vs Custo, colorindo por fumo e tamanho pelo IMC
sns.scatterplot(
    x="age", 
    y="charges", 
    hue="smoker", 
    size="bmi", 
    sizes=(20, 200), 
    data=df, 
    palette="vlag", 
    alpha=0.7
)
plt.title("Cruzamento de Impacto: Idade vs Custo (Separado por Fumo e IMC)", fontsize=14, fontweight="bold")
plt.xlabel("Idade (Age)", fontsize=12)
plt.ylabel("Custo (Charges)", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Legenda (Fumo / IMC)")
plt.tight_layout()
print("[Exibindo Gráfico 3] Dispersão de Idade, IMC e Fumo contra o Custo.")
plt.show()

print("\nTodas as visualizações foram exibidas com sucesso!")