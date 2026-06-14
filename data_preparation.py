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

# Label Encoding
df_processed['sex'] = df_processed['sex'].map({'female': 0, 'male': 1})
df_processed['smoker'] = df_processed['smoker'].map({'no': 0, 'yes': 1})

# One Hot Encoding
df_processed = pd.get_dummies(df_processed, columns=['region'], dtype=int)

# !! Separação de Entrada e Alvo !!
Y = df_processed['charges']
X = df_processed.drop(columns='charges')  

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