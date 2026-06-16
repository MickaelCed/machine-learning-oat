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

colunas = artefatos['colunas']
colunas_cluster = artefatos['colunas_cluster']



# =========================
# FUNÇÕES
# =========================

def limpar():
    os.system('cls' if os.name == 'nt' else 'clear')



def cabecalho():

    print("""
╔════════════════════════════════════════════╗
║                                            ║
║          🏥 MEDICAL COST PREDICTOR         ║
║                                            ║
║     Previsão de custo de seguro médico     ║
║                                            ║
║   KMeans + Decision Tree Regressor         ║
║                                            ║
╚════════════════════════════════════════════╝
""")



def prever():

    print("""
╔════════════════════════════════════════════╗
║              DADOS DO CLIENTE              ║
╚════════════════════════════════════════════╝
""")


    age = int(input("Idade: "))

    sex = int(
        input(
            "Sexo (0=feminino | 1=masculino): "
        )
    )

    bmi = float(
        input("BMI: ")
    )

    children = int(
        input("Número de filhos: ")
    )

    smoker = int(
        input(
            "Fumante (0=não | 1=sim): "
        )
    )


    print("""
Região:

[1] Northeast
[2] Northwest
[3] Southeast
[4] Southwest

""")


    region = int(
        input("Escolha: ")
    )


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

        1: 'region_northeast',
        2: 'region_northwest',
        3: 'region_southeast',
        4: 'region_southwest'

    }


    if region in regioes:

        dados[regioes[region]] = 1



    entrada = pd.DataFrame(
        [dados]
    )



    # =========================
    # CLUSTERIZAÇÃO
    # =========================

    entrada_cluster = entrada[
        colunas_cluster
    ]


    entrada_scaled = scaler.transform(
        entrada_cluster
    )


    entrada_scaled = pd.DataFrame(
        entrada_scaled,
        columns=colunas_cluster
    )


    cluster = kmeans.predict(
        entrada_scaled
    )[0]

    print("Entrada para KMeans:")
    print(entrada_scaled)
    
    print("Centro dos clusters:")
    print(kmeans.cluster_centers_)
    
    print("Distâncias:")
    print(kmeans.transform(entrada_scaled))


    # =========================
    # REGRESSÃO
    # =========================

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
        f"📌 Perfil encontrado: Cluster {cluster}"
    )

    print(
        f"💰 Custo estimado: ${custo:,.2f}"
    )



# =========================
# PROGRAMA PRINCIPAL
# =========================


while True:


    limpar()

    cabecalho()


    print("""
[1] Fazer previsão
[0] Sair

""")


    opcao = input(
        "Escolha: "
    )



    if opcao == "1":

        try:

            prever()


        except Exception as erro:

            print(
                "\n❌ Erro ao prever:"
            )

            print(erro)



        continuar = input(
            "\nDeseja fazer outra previsão? (s/n): "
        )


        if continuar.lower() != "s":

            break



    elif opcao == "0":

        break



    else:

        print(
            "\nOpção inválida!"
        )

        input(
            "Pressione ENTER..."
        )



limpar()


print("""
╔════════════════════════════════════════════╗
║                                            ║
║       Obrigado por utilizar o sistema!     ║
║                                            ║
╚════════════════════════════════════════════╝
""")