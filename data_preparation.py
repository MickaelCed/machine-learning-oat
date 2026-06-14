import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Carregar os dados
df = pd.read_csv('data/medical_insurance.csv') 

print('@-- Primeiras linhas do dataset original --@')
print(df.head())
print('\n@-- Informações Gerais --@')
print(df.info())


# !! Tratamento de variáveis !!
df_processed = df.copy()

# Preenchimento de Nulos com 'Unknown'
df_processed['alcohol_freq'] = df_processed['alcohol_freq'].fillna('Unknown')

# VARS. CATEGÓRICAS (One-Hot Encoding)
colunas_categoricas = [
    'sex', 'region', 'alcohol_freq', 'urban_rural', 'education',
    'marital_status', 'employment_status', 'plan_type', 'network_tier',
]  # Sex possui 3 valores (Male, female, other)
df_processed = pd.get_dummies(df_processed, columns=colunas_categoricas, dtype=int)

# VARS. ORDINAIS (Ordinal Encoding)
df_processed['smoker'] = df_processed['smoker'].map(
    {
        'Never': 0,
        'Former': 1,
        'Current': 2,
    }
)

# !! Separação de Entrada e Alvo !!
Y = df_processed['annual_medical_cost']

# Lista de Colunas a serem removidas de X para evitar vazamento de dados...
colunas_p_remover = [
    'person_id',            # Irrelevante 
    'annual_medical_cost',  # Target
    'annual_premium','monthly_premium',  #Valores gerados com base no custo
    'claims_count', 'avg_claim_amount', 'total_claims_paid'  # Histórico financeiro
]
X = df_processed.drop(columns=colunas_p_remover)  

# !! Normalização !!
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)

# Transformando de volta em DF
X_df = pd.DataFrame(X_scaled_array, columns = X.columns)

print('Preparação concluída com sucesso..')
print(f'Formato final dos atributos X: {X_df.shape}')
print(f'Formato final do alvo (y): {Y.shape}')

# Salvar dados
X_df.to_csv('data/X_ready.csv', index=False)
Y.to_csv('data/Y_ready.csv', index=False)