import joblib
import pandas as pd
import os


# =========================
# CARREGAR MODELO
# =========================

artefatos = joblib.load('modelo_charges.pkl')

modelo = artefatos['modelo']

kmeans = artefatos['kmeans']

scaler = artefatos['scaler']

colunas = list(artefatos['colunas'])

colunas_cluster = list(artefatos['colunas_cluster'])



# =========================
# FUNÇÕES
# =========================

def limpar():
    os.system('cls' if os.name == 'nt' else 'clear')



def cabecalho():

    print("""
╔════════════════════════════════════════════╗
║          🏥 MEDICAL COST PREDICTOR         ║
║                                            ║
║     KMeans + Decision Tree Regressor       ║
╚════════════════════════════════════════════╝
""")



def prever():

    print("""
╔════════════════════════════════════════════╗
║              DADOS DO CLIENTE              ║
╚════════════════════════════════════════════╝
""")


    age = int(input("Idade: "))
    sex = int(input("Sexo (0=feminino | 1=masculino): "))
    bmi = float(input("BMI: "))
    children = int(input("Número de filhos: "))
    smoker = int(input("Fumante (0=não | 1=sim): "))


    print("""
Região:

[1] Northeast
[2] Northwest
[3] Southeast
[4] Southwest
""")


    region = int(input("Escolha: "))



    dados = {

        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,

        'region_northeast': 0,
        'region_northwest': 0,
        'region_southeast': 0,
        'region_southwest': 0
    }



    regioes = {
        1:'region_northeast',
        2:'region_northwest',
        3:'region_southeast',
        4:'region_southwest'
    }


    if region in regioes:
        dados[regioes[region]] = 1



    entrada = pd.DataFrame([dados])


    print("\nEntrada:")
    print(entrada)



    # =========================
    # KMEANS
    # =========================

    print("\n--- KMEANS ---")

    # 1. Criamos uma cópia para não alterar o DataFrame 'entrada' original (usado na regressão)
    entrada_scaled = entrada.copy()

    # 2. Definimos as colunas que o scaler espera
    cols_to_scale = ['age', 'bmi', 'children']

    # 3. Escalamos APENAS as colunas numéricas necessárias
    entrada_scaled[cols_to_scale] = scaler.transform(entrada[cols_to_scale])



    entrada_scaled = pd.DataFrame(
        entrada_scaled,
        columns=colunas
    )

    # remove regiões antes do KMeans
    entrada_cluster = entrada_scaled.drop(
        columns=[
            'region_northeast',
            'region_northwest',
            'region_southeast',
            'region_southwest'
        ]
    )

    cluster = kmeans.predict(
        entrada_cluster
    )[0]


    print("Cluster:", cluster)



    # =========================
    # REGRESSÃO
    # =========================

    print("\n--- REGRESSÃO ---")


    entrada_regressao = entrada[
        colunas
    ]


    custo = modelo.predict(
        entrada_regressao
    )[0]



    # =========================
    # RESULTADO
    # =========================


    print("""
╔════════════════════════════════════════════╗
║                 RESULTADO                  ║
╚════════════════════════════════════════════╝
""")


    print(
        f"📌 Cluster encontrado: {cluster}"
    )


    print(
        f"💰 Custo estimado: ${custo:,.2f}"
    )





# =========================
# LOOP
# =========================

while True:

    limpar()

    cabecalho()


    print("""
[1] Fazer previsão
[0] Sair
""")


    opcao = input("Escolha: ")



    if opcao == "1":

        try:
            prever()

        except Exception as erro:

            print("\n❌ Erro:")
            print(erro)



        input("\nENTER para continuar...")


    elif opcao == "0":

        break