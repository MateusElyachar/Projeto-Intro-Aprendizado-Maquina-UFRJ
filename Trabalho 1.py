#==============================================================================
# EXPERIMENTO 01 - Explorar e visualizar o conjunto de df_treino do trabalho 1
#==============================================================================

#------------------------------------------------------------------------------
# Importar a bibliotecas
#------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

#------------------------------------------------------------------------------
# Carregar o conjunto de treinamento do arquivo CSV
#------------------------------------------------------------------------------

df_treino = pd.read_csv('conjunto_de_treinamento.csv')

#------------------------------------------------------------------------------
# excluindo algumas colundas 
#------------------------------------------------------------------------------

colunas_para_remover = [
    'grau_instrucao', 
    'possui_telefone_celular', 
    'qtde_contas_bancarias_especiais'
]

# Removendo as colunas
df_treino = df_treino.drop(columns=colunas_para_remover)

#------------------------------------------------------------------------------
# tratamento de dados
#------------------------------------------------------------------------------

# --- Tratamento para colunas faltantes  ---

print(df_treino.isnull().sum())

# descobrimos vendo a soma das colunas nulas de cada variável que as variaveis profissao_companheiro e grau_instrucao_companheiro 
# tem muitos valores nulos (mais de 50%). Por enquanto irei apenas substituir as colunas faltantes por -1
# fiz alguns testes e vi que isso abaixou um pouco a correlação porem não tanto
# se futuramente existirem alguns problemas irei excluir a coluna

# Para profissao_companheiro (sendo categórica)
df_treino['profissao_companheiro'].fillna(-1, inplace=True)

# Para grau_instrucao_companheiro (sendo categórica)
df_treino['grau_instrucao_companheiro'].fillna(-1, inplace=True) # Usando -1, já que os outros valores são numéricos

#para variavéis categóricas com valores nulos irei preencher com a moda

# Preencher 'profissao' com o valor mais frequente
moda_profissao = df_treino['profissao'].mode()[0] 
df_treino['profissao'].fillna(moda_profissao, inplace=True)

# Preencher 'ocupacao' com o valor mais frequente
moda_ocupacao = df_treino['ocupacao'].mode()[0]
df_treino['ocupacao'].fillna(moda_ocupacao, inplace=True)

# Preencher 'tipo_residencia' com o valor mais frequente
moda_tipo_residencia = df_treino['tipo_residencia'].mode()[0]
df_treino['tipo_residencia'].fillna(moda_tipo_residencia, inplace=True)

#para variavéis numéricas irei preencher com a mediana que não é afetada com outliers como a média

# Preencher 'meses_na_residencia' com a mediana
mediana_meses = df_treino['meses_na_residencia'].median()
df_treino['meses_na_residencia'].fillna(mediana_meses, inplace=True)


# --- Tratamento da coluna 'sexo' como solicitado no dicionário ---

# 1. Preencher os valores nulos (NaN) com 'N'
# A função .fillna() é perfeita para isso. 
df_treino['sexo'] = df_treino['sexo'].fillna('N')

#------------------------------------------------------------------------------
# engenharia de features
#------------------------------------------------------------------------------

#Estou escrevendo essa parte depois de já ter escolhido um bom modelo (random forest) e submetido ao kaggle
#consegui uma boa posição porem quero melhorar ela por isso vou tentar incluir algumas atributos novos

#features relacionadas à renda

#Proporção Renda Extra: proporcao_renda_extra = renda_extra / (renda_total + 1). 
#Uma pessoa que depende muito de renda extra pode ser mais instável.

# 1. Renda Total
# Primeiro, vamos garantir que não há nulos em 'renda_extra' também.
# Preencher com a mediana é uma abordagem segura.
mediana_renda_extra = df_treino['renda_extra'].median()
df_treino['renda_extra'].fillna(mediana_renda_extra, inplace=True)

# Agora podemos criar as novas features de renda com segurança.
# Usamos .loc para evitar o SettingWithCopyWarning
df_treino.loc[:, 'renda_total'] = df_treino['renda_mensal_regular'] + df_treino['renda_extra']

# não teve boa correlação com inadimplente então vamos excluir elá depois  -0.000209

# 2. Proporção da Renda que é Extra
# O +1 no denominador é uma prática comum para evitar divisão por zero se a renda_total for 0.
df_treino.loc[:, 'proporcao_renda_extra'] = df_treino['renda_extra'] / (df_treino['renda_total'] + 1)

# teve correlação com inadimplente de  -0.009177 o que é aceitavel

# 3. Criar a feature 'numero_de_cartoes'
# Primeiro, vamos re-adicionar as colunas de cartão que removemos.
# Remova 'possui_cartao_amex' e 'possui_cartao_visa' da sua lista `colunas_para_remover2`
colunas_cartoes = ['possui_cartao_mastercard', 'possui_cartao_diners', 'possui_cartao_visa', 'possui_cartao_amex']
df_treino.loc[:, 'numero_de_cartoes'] = df_treino[colunas_cartoes].sum(axis=1)

#essa feature teve corelação de -0.012057 o que é aceitavel

print("Novas features criadas: 'renda_total', 'proporcao_renda_extra', 'numero_de_cartoes'")

#features de idade

bins = [18, 30, 45, 60, 110]
labels = ['jovem_adulto', 'adulto', 'meia_idade', 'idoso']
df_treino['faixa_etaria'] = pd.cut(df_treino['idade'], bins=bins, labels=labels, right=False)

#------------------------------------------------------------------------------
# Exibir informações sobre o conjunto de dados de treinamento
#------------------------------------------------------------------------------

print("\n\n Exibir as dimensões do conjunto de dados: \n")

print(df_treino.shape)
print("O conjunto tem",df_treino.shape[0],"amostras com",df_treino.shape[1],"variáveis")


print("\n\n Exibir os tipos das variáveis do conjunto de dados: \n")

print(df_treino.dtypes)

print("\n\n Exibir informações estatísticas sobre os dados: \n")

print(df_treino.describe())


#------------------------------------------------------------------------------
# Explorando o conjunto
#------------------------------------------------------------------------------

# Analizando a variavel alvo


#Se uma classe for muito maior que a outra (ex: 90% de bons pagadores e 10% de inadimplentes), o dataset é desbalanceado.
#Isso é muito importante, pois afeta a escolha do modelo e das métricas de avaliação mais para frente.

# Calcular a contagem e a proporção
print("Contagem de valores para 'inadimplente':")
print(df_treino['inadimplente'].value_counts())

print("\nProporção de valores para 'inadimplente':")
print(df_treino['inadimplente'].value_counts(normalize=True))

# Visualizar com um gráfico de barras
plt.figure(figsize=(8, 6))
sns.countplot(x='inadimplente', data=df_treino)
plt.title('Distribuição da Variável Alvo (Inadimplente)')
plt.xlabel('Classe (0: Bom Pagador, 1: Inadimplente)')
plt.ylabel('Contagem')
plt.show()

#Descobrimos que a distribuição é de 50% para cada classe (0: Bom Pagador, 1: Inadimplente), 10k para cada.
#Portanto a distribuição é balanceada. Caso ideal.
#Não precisamos se preocupar com técnicas complexas para criar amostras sintéticas da classe minoritária ou usar pesos de classe (class_weight) no modelo.
#Métricas de Avaliação: A acurácia se torna uma métrica mais confiável. 
#Em datasets desbalanceados, um modelo pode ter 95% de acurácia simplesmente prevendo a classe dominante o tempo todo. 
#No nosso caso, a acurácia reflete melhor o desempenho real. 
#Mesmo assim, é sempre uma boa prática analisar também a matriz de confusão, precisão, recall e F1-Score.


# Analizando variaeis numéricas

# Lista de colunas numéricas que parecem interessantes
colunas_numericas = ['idade', 'renda_mensal_regular', 'renda_extra', 'valor_patrimonio_pessoal', 'meses_na_residencia']


# 1. Box Plot para 'renda_mensal_regular'
# Para visualizar melhor, vamos limitar a renda para não distorcer o gráfico com outliers
df_renda_limitada = df_treino[df_treino['renda_mensal_regular'] < 10000] # Exemplo de limite
plt.figure(figsize=(10, 7))
sns.boxplot(x='inadimplente', y='renda_mensal_regular', data=df_renda_limitada)
plt.title('Box Plot da Renda Mensal por Classe de Inadimplência')
plt.show()

#Hipótese: A renda_mensal_regular, sozinha, parece não ser um bom diferenciador entre as duas classes. 
#A distribuição de renda é muito similar para ambos os grupos.
# Não descartaremos a variável renda ainda! Ela pode se tornar poderosa quando combinada com outras. 
#Por exemplo, uma pessoa com alta renda e alto patrimônio pode ser um bom pagador, mas uma pessoa com alta renda e nenhum patrimônio pode ser um risco. 
#O poder preditivo dela sozinho é baixo(isso é mostrado também com sua correlação com inadimplente que é -0.000926)


# 2. Matriz de Correlação
# Selecionar apenas colunas numéricas para a correlação
df_numerico = df_treino.select_dtypes(include=['float64', 'int64'])
corr_matrix = df_numerico.corr()

plt.figure(figsize=(24, 21))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Matriz de Correlação das Variáveis Numéricas')
plt.show()

#Nessa matrix foi possível observar que

# Ver a correlação específica com a variável alvo
print("\nCorrelação com a variável 'inadimplente':")
print(corr_matrix['inadimplente'].sort_values(ascending=False))

#A média do módulo das correlações com a variável 'inadimplente' é de aproximadamente 0.02097
#As variàveis  valor_patrimonio_pessoal; nacionalidade; renda_mensal_regular; possui_cartao_amex; possui_cartao_visa; possui_outros_cartoes         
#possuem 10 vezes menos impacto que a média então irei remover elas e futuramente, se precisar melhorar o modelo vou ver se recoloco elas.
#Depois de submeter ao kaggle fiz uma engenharia de features com renda_mensal_regular ; valor_patrimonio_pessoal; possui_carro e as variaves de cartões
#mas ainda sim excluo elas aqui para evitar redundancia
#excluindo a variável possui_carro pois ela tem alta correlação com a qtde_contas_bancarias e menos correlação com a inadimplente
#excluindo a variável local_onde_trabalha pois ela é redundante com a local_onde_reside (correlação de 1)

colunas_para_remover2 = [
    'renda_mensal_regular', 
    'renda_extra',
    'renda_total',
    'nacionalidade',
    'possui_cartao_diners',
    'possui_cartao_visa',
    'possui_cartao_amex',
    'possui_outros_cartoes',
    'local_onde_trabalha'
]
# Removendo as colunas
df_treino = df_treino.drop(columns=colunas_para_remover2)

# 2.1 nova Matriz de Correlação
# Selecionar apenas colunas numéricas para a correlação
df_numerico = df_treino.select_dtypes(include=['float64', 'int64'])
corr_matrix = df_numerico.corr()

plt.figure(figsize=(24, 21))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Matriz de Correlação das Variáveis Numéricas')
plt.show()

# Ver a nova correlação específica com a variável alvo
print("\nCorrelação com a variável 'inadimplente':")
print(corr_matrix['inadimplente'].sort_values(ascending=False))

#------------------------------------------------------------------------------
# Pré-processamento para Modelagem
#------------------------------------------------------------------------------

# Etapa 1: Codificação das Variáveis Categóricas (Transformar Texto em Números)

# Modelos matemáticos não conseguem processar strings como 'presencial', 'internet', 'M', 'F' ou 'Y', 'N'. 
# Precisamos convertê-las em números. A melhor abordagem para variáveis nominais (sem ordem) é o One-Hot Encoding.
# Identificar colunas categóricas que precisam de codificação
# (geralmente do tipo 'object')
print("\nColunas antes da codificação:")
print(df_treino.columns)
print(df_treino.dtypes)
print("Dimensões do df inicial:", df_treino.shape)

# Aplicar One-Hot Encoding
df_processado = pd.get_dummies(df_treino, drop_first=True)
# O argumento drop_first=True remove a primeira categoria de cada feature.
# Isso evita a multicolinearidade perfeita (ex: se sexo_M=0, sabemos que é 'F' ou 'N')

print("\nColunas depois da codificação (prontas para o modelo):")
print(df_processado.columns)
print("Dimensões do df final:", df_processado.shape)

# Etapa 2: Separação dos Dados em Treino e Validação


# 1. Separar as features (X) da variável alvo (y)
X = df_processado.drop(columns=['inadimplente', 'id_solicitante']) 
y = df_processado['inadimplente']

# 2. Dividir em conjuntos de treino e validação
# Usamos 80% para treino e 20% para validação.
# random_state=42 garante que a divisão seja sempre a mesma, para reprodutibilidade.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDimensões dos conjuntos de dados:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)


#------------------------------------------------------------------------------
# Treinamento e Avaliação dos Modelo Baseline 
#------------------------------------------------------------------------------

# 2.1. Escalonar os dados
# Modelos lineares como a Regressão Logística funcionam melhor com dados escalonados.
# É importante treinar o scaler APENAS com os dados de treino (X_train)
# para evitar vazamento de informação do conjunto de validação.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val) # Apenas transforma, não "fita" novamente

# 2.2. Inicializar o modelo de Regressão Logística
log_reg_model = LogisticRegression(random_state=42, n_jobs=-1)

# 2.3. Treinar o modelo com os dados de treino escalonados
print("\nTreinando o modelo de Regressão Logística...")
log_reg_model.fit(X_train_scaled, y_train)
print("Treinamento concluído!")

# 2.4. Fazer previsões nos dados de validação escalonados
y_pred_log_reg = log_reg_model.predict(X_val_scaled)

# 2.5. Avaliar o desempenho do modelo
accuracy_log_reg = accuracy_score(y_val, y_pred_log_reg)
print(f"\nAcurácia do modelo de Regressão Logística: {accuracy_log_reg:.4f}")

print("\nRelatório de Classificação (Regressão Logística):")
print(classification_report(y_val, y_pred_log_reg))

print("\nMatriz de Confusão (Regressão Logística):")
cm_log_reg = confusion_matrix(y_val, y_pred_log_reg)
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - Regressão Logística')
plt.show()
#Resultados(antes de eng de features):

#Acurácia do modelo de Regressão Logística: 0.5815

#Relatório de Classificação (Regressão Logística):
#              precision    recall  f1-score   support

#           0       0.59      0.55      0.57      2027
#           1       0.57      0.61      0.59      1973

#    accuracy                           0.58      4000
#   macro avg       0.58      0.58      0.58      4000
#weighted avg       0.58      0.58      0.58      4000


# Para otimizar a Regressão Logística, precisamos incluir o scaler no processo
#O Pipeline garante que, durante a validação cruzada, o escalonamento (Etapa 1) seja calculado apenas nos dados de treino de cada fold, 
#e depois aplicado ao fold de teste, evitando a "trapaça".

pipeline_log_reg = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(random_state=42, max_iter=1000))])

print("\nAvaliando Regressão Logística com Validação Cruzada...")
scores_log_reg = cross_val_score(pipeline_log_reg, X, y, cv=5, scoring='accuracy')
print(f"Scores de Acurácia (RL): {scores_log_reg}")
print(f"Média da Acurácia (RL): {scores_log_reg.mean():.4f} (+/- {scores_log_reg.std():.4f})")

#Resultados

#Avaliando Regressão Logística com Validação Cruzada...
#Scores de Acurácia (RL): [0.59725 0.58    0.603   0.5895  0.5875 ]
#Média da Acurácia (RL): 0.5914 (+/- 0.0080)

#para melhorar o modelo de RL vamos procurar os melhores hiperparametros com GridSearchCV 

# Definindo a grade de parâmetros para a Regressão Logística
# Note como usamos 'logreg__' para especificar que o parâmetro é da etapa 'logreg' do pipeline
param_grid_log_reg = {
    'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'logreg__penalty': ['l1', 'l2'],
    'logreg__solver': ['liblinear'] # 'liblinear' é um bom solver que funciona com L1 e L2
}

#Agora, passamos o pipeline como estimador e a grade de parâmetros para o GridSearchCV.

# Configurar o Grid Search
# O estimator agora é o nosso pipeline
grid_search_log_reg = GridSearchCV(estimator=pipeline_log_reg, 
                                   param_grid=param_grid_log_reg, 
                                   cv=5,                 # 5 folds é um bom padrão
                                   scoring='accuracy',   # Métrica de avaliação
                                   n_jobs=-1,            # Usar todos os processadores
                                   verbose=1)           # Mostra o progresso

print("\nIniciando GridSearchCV para Regressão Logística...")
grid_search_log_reg.fit(X, y)

# Exibir os melhores resultados
print(f"\nMelhor Acurácia encontrada (CV): {grid_search_log_reg.best_score_:.4f}")
print("Melhores Parâmetros encontrados:")
print(grid_search_log_reg.best_params_)

# O melhor modelo (pipeline treinado) está em:
melhor_modelo_log_reg = grid_search_log_reg.best_estimator_

#Resultados 

#Melhor Acurácia encontrada (CV): 0.5929
#Melhores Parâmetros encontrados:
#{'logreg__C': 0.01, 'logreg__penalty': 'l2', 'logreg__solver': 'liblinear'}


#Melhor Acurácia foi do modelo RF (CV): 0.5936


#------------------------------------------------------------------------------
# Treinamento e Avaliação do modelo Random Forest
#------------------------------------------------------------------------------

# 1.1 Inicializar o modelo
# n_jobs=-1 usa todos os processadores disponíveis para acelerar o treinamento.
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

# 1.2. Treinar o modelo com os dados de treino
print("\nTreinando o modelo RandomForest...")
rf_model.fit(X_train, y_train)
print("Treinamento concluído!")

# 1.3. Fazer previsões nos dados de validação
y_pred = rf_model.predict(X_val)

# 1.4. Avaliar o desempenho do modelo
accuracy = accuracy_score(y_val, y_pred)
print(f"\nAcurácia do modelo: {accuracy:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_val, y_pred))

print("\nMatriz de Confusão:")
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão -RF')
plt.show()

#Resultados do random forest: (antes da submissão pro kegel e engenharia de features)

#Acurácia do modelo: 0.5820
#Relatório de Classificação:
#              precision    recall  f1-score   support

#           0       0.59      0.58      0.59      2027
#           1       0.58      0.58      0.58      1973

#    accuracy                           0.58      4000
#   macro avg       0.58      0.58      0.58      4000
#weighted avg       0.58      0.58      0.58      4000

#Resultados do random forest: (depois da submissão pro kegel e engenharia de features tivemos aumentos marginais com a engenharia de features)

#Acurácia do modelo: 0.5837

#Relatório de Classificação:
 #             precision    recall  f1-score   support

#           0       0.59      0.59      0.59      2027
#           1       0.58      0.58      0.58      1973

#    accuracy                           0.58      4000
#   macro avg       0.58      0.58      0.58      4000
#weighted avg       0.58      0.58      0.58      4000

# Otimização e Experimentação

#3.1 Validação Cruzada (Cross-Validation)
#A avaliação atual foi feita em uma única divisão 80/20. 
#Para ter certeza de que o desempenho não foi sorte, usaremos a validação cruzada. 
#Isso treinará e testará o modelo várias vezes em diferentes "fatias" dos dados e teremos uma média de desempenho mais confiável.

# Usando 5 fatias (cv=5)
print("\nAvaliando RandomForest com Validação Cruzada...")
scores_rf = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(f"Scores de Acurácia em cada fold (RF): {scores_rf}")
print(f"Média da Acurácia (RF): {scores_rf.mean():.4f} (+/- {scores_rf.std():.4f})")

#Resultados
#Avaliando RandomForest com Validação Cruzada...
#Scores de Acurácia (RF): [0.582   0.57175 0.585   0.578   0.5835 ]
#Média da Acurácia (RF): 0.5800 (+/- 0.0048)

#agora vamos utilizar o Grid seach no modelo RF
# Definir a "grade" (Grid) de parâmetros que queremos testar
param_grid = {
    'n_estimators': [50, 100, 200],  # Testa 50, 100 ou 200 árvores
    'max_depth': [5, 10, None],      # Testa profundidade 5, 10 ou sem limite
    'min_samples_leaf': [1, 5, 10]   # Testa o mínimo de amostras em uma folha
}

# Configurar o Grid Search
# Usamos cv=3 para ser mais rápido, mas cv=5 é mais robusto se tiver tempo.
grid_search = GridSearchCV(estimator=rf_model, 
                           param_grid=param_grid, 
                           cv=10, 
                           scoring='accuracy', 
                           n_jobs=-1, # Usa todos os processadores
                           verbose=2) # Mostra o progresso

print("\nIniciando GridSearchCV para RandomForest (pode demorar)...")
grid_search.fit(X, y) # Usamos X e y completos, o GridSearchCV faz a validação cruzada internamente

# Resultados
print("\nMelhor Acurácia encontrada (CV):", grid_search.best_score_)
print("Melhores Parâmetros:", grid_search.best_params_)

# O melhor modelo encontrado já está treinado e pronto para uso:
melhor_modelo_rf = grid_search.best_estimator_

#sem engenharia de features cv = 3
#Melhor Acurácia encontrada (CV): 0.5936001504755245
#Melhores Parâmetros: {'max_depth': None, 'min_samples_leaf': 5, 'n_estimators': 200}

#com engenharia de features cv =3 -> resultados marginalmente melhores

#Melhor Acurácia encontrada (CV): 0.5946998029636471
#Melhores Parâmetros: {'max_depth': None, 'min_samples_leaf': 5, 'n_estimators': 200}

#com engenharia de features cv =10 -> resultados marginalmente melhores

#Melhor Acurácia encontrada (CV): 0.595
#Melhores Parâmetros: {'max_depth': None, 'min_samples_leaf': 10, 'n_estimators': 200}

#Isso mostra que, com os hiperparâmetros corretos, o RandomForest consegue extrair um pouco mais de informação dos dados.   
#agora iremos testar outros modelos para compararmos com o RF e o RegLog(baseline)

#------------------------------------------------------------------------------
# Otimização Avançada do RandomForest com RandomizedSearchCV
#------------------------------------------------------------------------------

# O GridSearchCV é ótimo, mas testa apenas as combinações que definimos.
# O RandomizedSearchCV pode explorar um espaço de parâmetros maior de forma mais eficiente.
# Vamos definir uma grade mais ampla e deixar o RandomizedSearch testar 50 combinações aleatórias.


# 1. Definir o modelo base
rf_para_random_search = RandomForestClassifier(random_state=42)

# 2. Definir a grade de parâmetros (distribuições) para a busca aleatória
# Aumentamos o número de opções para dar mais liberdade à busca.
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'bootstrap': [True, False] # Testa árvores com ou sem amostragem com reposição
}

# 3. Configurar a busca aleatória
# n_iter define quantas combinações aleatórias serão testadas. 50 é um bom número.
random_search = RandomizedSearchCV(estimator=rf_para_random_search, 
                                   param_distributions=param_dist, 
                                   n_iter=50, 
                                   cv=5, # Usar 5 folds para uma validação mais robusta
                                   verbose=2, 
                                   random_state=42, 
                                   n_jobs=-1,
                                   scoring='accuracy')

# 4. Executar a busca
print("\nIniciando RandomizedSearchCV para RandomForest (pode demorar mais que o GridSearch)...")
random_search.fit(X, y)

# 5. Exibir os melhores resultados
print(f"\nMelhor Acurácia encontrada com RandomizedSearch (CV): {random_search.best_score_:.4f}")
print("Melhores Parâmetros encontrados com RandomizedSearch:")
print(random_search.best_params_)

# 6. Guardar o novo melhor modelo
# Este modelo pode ser o seu campeão final se superar o do GridSearchCV
melhor_modelo_rf_random = random_search.best_estimator_


#Melhor Acurácia encontrada com RandomizedSearch (CV): 0.5969
#Melhores Parâmetros encontrados com RandomizedSearch:
#{'n_estimators': 500, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': None, 'bootstrap': True}

#------------------------------------------------------------------------------
# Treinamento e Avaliação do Modelo Gaussian Naive Bayes
#------------------------------------------------------------------------------

# 1. Definir o Pipeline para Naive Bayes
# Etapa 1: Escalonar os dados
# Etapa 2: Treinar o modelo GaussianNB
pipeline_gnb = Pipeline([
    ('scaler', StandardScaler()),
    ('gnb', GaussianNB())
])

# 2. Executar a Validação Cruzada usando o Pipeline
print("\nIniciando Validação Cruzada para Gaussian Naive Bayes (cv=5)...")
scores_gnb = cross_val_score(pipeline_gnb, X, y, cv=5, scoring='accuracy')

# 3. Exibir os resultados
print(f"Scores de Acurácia em cada fold: {scores_gnb}")
print(f"Média da Acurácia (GNB): {np.mean(scores_gnb):.4f}")

# Para ver o Relatório de Classificação e a Matriz de Confusão
# você pode treinar o pipeline e prever no seu conjunto de validação anterior (X_val, y_val)
print("\nTreinando o pipeline GNB para gerar relatório detalhado...")
pipeline_gnb.fit(X_train, y_train)
y_pred_gnb = pipeline_gnb.predict(X_val)

print("\nRelatório de Classificação (Gaussian Naive Bayes):")
print(classification_report(y_val, y_pred_gnb))

print("\nMatriz de Confusão (Gaussian Naive Bayes):")
cm_gnb = confusion_matrix(y_val, y_pred_gnb)
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - Gaussian Naive Bayes')
plt.show()

#Resultados e analises para o Modelo Gaussian Naive Bayes

#Acurácia (CV): 0.5079 - Isso é basicamente um chute aleatório (50%).
#   Matriz de Confusão:
#       Verdadeiro Negativo (VN): 114 - Acertou muito poucos bons pagadores.
#       Falso Positivo (FP): 1913 - Errou absurdamente, classificando quase todos os bons pagadores como inadimplentes.
#       Verdadeiro Positivo (VP): 1900 - Acertou quase todos os inadimplentes.
#       Falso Negativo (FN): 73 - Deixou passar pouquíssimos inadimplentes.
#   Relatório de Classificação:
#       Classe 0 (Bons pagadores): recall de 0.06. Significa que o modelo só conseguiu identificar 6% dos bons pagadores.
#       Classe 1 (Inadimplentes): recall de 0.96. Significa que o modelo identificou 96% dos inadimplentes.
#Conclusão da Análise: 
#   O seu modelo GaussianNB está com um viés extremo. Ele basicamente aprendeu a prever "inadimplente" (1) 
#   para quase todo mundo. Por isso ele acerta quase todos os inadimplentes (recall alto na classe 1) 
#   e erra quase todos os bons pagadores (recall baixo na classe 0). 
#   A acurácia final fica em torno de 50% porque o dataset é balanceado.
#Por Que Isso Aconteceu? A "Ingenuidade" do Naive Bayes
#   O GaussianNB tem duas premissas (suposições) muito fortes que os seus dados violam:
#       Premissa Gaussiana: Ele assume que cada uma das suas 266 colunas (features) segue uma distribuição normal (curva de sino). 
#       nossas features não seguem! A maioria delas são colunas "dummy" (0 ou 1) criadas pelo get_dummies. 
#       Uma variável que só pode ser 0 ou 1 está muito longe de uma distribuição normal. Isso confunde completamente o cálculo de probabilidades do modelo.
#       Premissa da Independência ("Naive"): O modelo assume que todas as features são independentes umas das outras. 
#       Isso também é falso nesse caso. Por exemplo, as colunas estado_onde_reside_SP e local_onde_reside_14 (CEP) são altamente correlacionadas. 
#       A idade (idade) e o tempo de residência (meses_na_residencia) também não são independentes.
#Resumindo: O modelo GaussianNB performou mal porque as suposições fundamentais dele não correspondem à realidade dos seus dados. 
#O código está correto, o problema é a inadequação do algoritmo para este tipo específico de dados pré-processados. (se tivermos tempo podemos re-pré-processar eles)
#Para dados com muitas colunas "dummy", um BernoulliNB ou MultinomialNB poderia ser mais apropriado, mas o GaussianNB é o mais comum de se ensinar primeiro.

#------------------------------------------------------------------------------
# Treinamento e Avaliação do Modelo LinearSVC (SVM Linear)
#------------------------------------------------------------------------------


# Definir o pipeline para o SVM Linear
pipeline_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', LinearSVC(random_state=42, max_iter=2000)) # Aumentar max_iter pode ser necessário
])

# Avaliar com validação cruzada
print("\nIniciando Validação Cruzada para LinearSVC (cv=5)...")
scores_svc = cross_val_score(pipeline_svc, X, y, cv=5, scoring='accuracy')

print(f"Scores de Acurácia em cada fold: {scores_svc}")
print(f"Média da Acurácia (LinearSVC): {np.mean(scores_svc):.4f}")

#Resultados (antes da engenharia de features)
#Scores de Acurácia em cada fold: [0.59675 0.57975 0.60325 0.589   0.588  ]
#Média da Acurácia (LinearSVC): 0.5914

#------------------------------------------------------------------------------
# Otimização do Modelo LinearSVC com GridSearchCV
#------------------------------------------------------------------------------

# 1. Definir o Pipeline para LinearSVC (o mesmo de antes)
pipeline_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', LinearSVC(random_state=42, max_iter=5000)) # max_iter aumentado para garantir convergência
])

# 2. Definir a grade de parâmetros para o LinearSVC
# O principal parâmetro a ser otimizado é o 'C'
param_grid_svc = {
    'svc__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

# 3. Configurar e executar o GridSearchCV
# Usamos o pipeline como estimador
grid_search_svc = GridSearchCV(estimator=pipeline_svc, 
                               param_grid=param_grid_svc, 
                               cv=5, 
                               scoring='accuracy', 
                               n_jobs=-1, 
                               verbose=1)

print("\nIniciando GridSearchCV para LinearSVC...")
grid_search_svc.fit(X, y)

# 4. Exibir os resultados
print(f"\nMelhor Acurácia encontrada para LinearSVC (CV): {grid_search_svc.best_score_:.4f}")
print("Melhores Parâmetros encontrados para LinearSVC:")
print(grid_search_svc.best_params_)

# Guardar o melhor modelo SVC
melhor_modelo_svc = grid_search_svc.best_estimator_
#resultados (antes da engenharia de features)
#Melhor Acurácia (CV): 0.5927 
#Este resultado é quase idêntico ao da Regressão Logística Otimizada (0.5929) e um pouco abaixo do RandomForest Otimizado (0.5936).
#Isso reforça a ideia de que, para este problema, os modelos lineares mais simples, quando bem regularizados, são muito competitivos.
#Melhor Parâmetro Encontrado: {'svc__C': 0.001}
#Assim como na Regressão Logística, o GridSearchCV escolheu o menor valor de C que oferecemos. 
#Isso significa que o modelo se beneficiou da regularização mais forte possível dentro das opções. 
#Ele está ativamente evitando o overfitting e preferindo uma fronteira de decisão mais simples.

#------------------------------------------------------------------------------
# Treinamento e Avaliação do Modelo SVM com Kernel (Não-Linear)
#------------------------------------------------------------------------------

# O SVC também precisa de dados escalonados. Vamos usar os que já criamos.
# X_train_scaled e X_val_scaled

# 1. Inicializar o modelo SVC com kernel 'rbf' (padrão)
# verbose=True para vermos o progresso, pois pode demorar.
svc_model = SVC(random_state=42, verbose=True)

# 2. Treinar o modelo
# ATENÇÃO: Esta etapa pode ser demorada!
print("\nTreinando o modelo SVC com kernel RBF (pode demorar)...")
svc_model.fit(X_train_scaled, y_train)
print("Treinamento concluído!")

# 3. Fazer previsões e avaliar
y_pred_svc = svc_model.predict(X_val_scaled)

accuracy_svc = accuracy_score(y_val, y_pred_svc)
print(f"\nAcurácia do modelo SVC (kernel RBF): {accuracy_svc:.4f}")

print("\nRelatório de Classificação (SVC com kernel RBF):")
print(classification_report(y_val, y_pred_svc))

print("\nMatriz de Confusão (SVC com kernel RBF):")
cm_svc = confusion_matrix(y_val, y_pred_svc)
sns.heatmap(cm_svc, annot=True, fmt='d', cmap='viridis')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - SVC com Kernel RBF')
plt.show()

# resultados e análises(antes da engenharia de features):
    
#Acurácia: 0.5783
#Este resultado é um pouco inferior aos melhores modelos (RandomForest Otimizado com ~0.5936 e Regressão Logística Otimizada com ~0.5929).
#Relatório de Classificação:
#Recall para Inadimplentes (Classe 1): 0.63 - Este é o ponto mais interessante! O SVC teve o maior recall de todos os modelos testados até agora. 
#le conseguiu identificar 63% dos inadimplentes reais.
#Precision para Inadimplentes (Classe 1): 0.56 - Para conseguir esse recall alto, ele teve uma precisão mais baixa, ou seja, gerou mais "alarmes falsos".
#F1-Score para Inadimplentes (Classe 1): 0.60 - O F1-Score, que equilibra precision e recall, é o mais alto até agora para a classe 1.

#------------------------------------------------------------------------------
# Conclusão Final e Escolha do Modelo Campeão
#------------------------------------------------------------------------------

# Após uma série de experimentos, incluindo a criação de novas features e a otimização
# de hiperparâmetros para múltiplos algoritmos, compilamos os resultados finais.
# A métrica principal para comparação é a Acurácia Média obtida na Validação Cruzada (CV),
# pois ela oferece a estimativa mais robusta do desempenho do modelo em dados não vistos.

# Tabela Comparativa de Desempenho dos Modelos

# Modelo	                        Média Acurácia (CV)	     Comentários
# ----------------------------------------------------------------------------------------------------------------------------------
# RandomForest (RandomizedSearch) |    0.5969	             | Melhor performance geral. A busca aleatória em uma grade ampla de
#                                 |                            | parâmetros se mostrou mais eficaz que o GridSearch.
#
# RandomForest (GridSearch)	      |    0.5950	             | Segundo melhor modelo. Mostra que o RF é robusto para este problema,
#                                 |                            | e que a otimização de hiperparâmetros traz ganhos.
#
# Regressão Logística (Otimizada) |    0.5929	             | Melhor modelo linear. Performance muito forte para um modelo simples,
#                                 |                            | indicando que a regularização forte (C=0.01) foi crucial.
#
# LinearSVC (Otimizado)       	  |    0.5927	             | Desempenho praticamente idêntico à Regressão Logística, confirmando
#                                 |                            | a eficácia dos modelos lineares bem regularizados.
#
# SVM com Kernel (Padrão)         |   ~0.5783 (único split)  | Performance inferior e computacionalmente muito caro. A complexidade
#                                 |                            | extra não se traduziu em melhor desempenho nos dados de validação.
#
# Gaussian Naive Bayes	          |   ~0.5079	             | Pior desempenho. Suas premissas (distribuição Gaussiana e independência
#                                 |                            | das features) foram violadas pelos dados, tornando-o inadequado.

# DECISÃO: O modelo RandomForest otimizado pelo RandomizedSearchCV foi escolhido como o modelo final
# para gerar as previsões no conjunto de teste, por apresentar a maior acurácia média na validação cruzada.

#------------------------------------------------------------------------------
# Preparação Final e Submissão para o Kaggle
#------------------------------------------------------------------------------

print("\n--- INICIANDO PREPARAÇÃO DO ARQUIVO DE SUBMISSÃO ---")

# 1. Carregar o conjunto de teste
df_teste = pd.read_csv('conjunto_de_teste.csv')
ids_solicitantes_teste = df_teste['id_solicitante'] # Guardar os IDs para o arquivo final

# 2. APLICAR EXATAMENTE O MESMO PRÉ-PROCESSAMENTO DO TREINO

# 2.1. Remover as mesmas colunas iniciais
# (colunas_para_remover foi definida no início do script)
df_teste = df_teste.drop(columns=colunas_para_remover)

# 2.2. Tratar os valores nulos
# Usamos os valores (moda/mediana) calculados no CONJUNTO DE TREINO para não vazar informação
df_teste['profissao_companheiro'].fillna(-1, inplace=True)
df_teste['grau_instrucao_companheiro'].fillna(-1, inplace=True)
df_teste['profissao'].fillna(moda_profissao, inplace=True)
df_teste['ocupacao'].fillna(moda_ocupacao, inplace=True)
df_teste['tipo_residencia'].fillna(moda_tipo_residencia, inplace=True)
df_teste['meses_na_residencia'].fillna(mediana_meses, inplace=True)
df_teste['sexo'].fillna('N', inplace=True)
df_teste['renda_extra'].fillna(mediana_renda_extra, inplace=True) # Importante para a feature 'renda_total'

# 2.3. Aplicar a Engenharia de Features
print("Aplicando engenharia de features no conjunto de teste...")
df_teste.loc[:, 'renda_total'] = df_teste['renda_mensal_regular'] + df_teste['renda_extra']
df_teste.loc[:, 'proporcao_renda_extra'] = df_teste['renda_extra'] / (df_teste['renda_total'] + 1)
df_teste.loc[:, 'numero_de_cartoes'] = df_teste[colunas_cartoes].sum(axis=1)
df_teste['faixa_etaria'] = pd.cut(df_teste['idade'], bins=bins, labels=labels, right=False)

# 2.4. Remover as colunas que foram removidas no treino
# (colunas_para_remover2 já está definida no seu script)
df_teste = df_teste.drop(columns=colunas_para_remover2)

# 2.5. Aplicar One-Hot Encoding
df_teste_processado = pd.get_dummies(df_teste, drop_first=True)

# 2.6. Alinhar as colunas para garantir consistência
# Garante que o df de teste tenha exatamente as mesmas colunas que o de treino
print("Alinhando colunas do conjunto de teste com o de treino...")
train_cols = X.columns
df_teste_final = df_teste_processado.reindex(columns=train_cols, fill_value=0)

# 3. Selecionar o modelo campeão
# O modelo do RandomizedSearchCV teve a melhor performance na validação cruzada
modelo_final = random_search.best_estimator_
print(f"Modelo campeão selecionado: RandomForest com acurácia CV de {random_search.best_score_:.4f}")

# 4. Fazer as previsões no conjunto de teste
print("Realizando previsões no conjunto de teste...")
previsoes_finais = modelo_final.predict(df_teste_final)

# 5. Criar o arquivo de submissão
submission = pd.DataFrame({
    'id_solicitante': ids_solicitantes_teste,
    'inadimplente': previsoes_finais
})

# 6. Salvar o arquivo em formato CSV
submission.to_csv('submission.csv', index=False)

print("\nArquivo 'submission.csv' criado com sucesso!")

# a submisão final me rendeu um score de 0.5916 e posição 19
# se eu tiver mais tempo depois de acabar o segundo trabalho vou voltar e tentar melhorar o modelo

#-----------------------------------------------------------------------------
# Avaliação Detalhada do Modelo Campeão (RandomForest Otimizado)
#------------------------------------------------------------------------------

# O objeto `random_search.best_estimator_` já contém o melhor modelo treinado com
# todos os dados que foram passados para o .fit() (ou seja, X e y completos).
# Para obter uma matriz de confusão, precisamos de um conjunto de validação separado.
# Vamos usar a divisão train/val que já criamos para fazer previsões e avaliar.

# O melhor modelo já está salvo na variável `melhor_modelo_rf_random`
# Vamos fazer as previsões no conjunto de validação (X_val)

print("\nAvaliando o melhor modelo (RandomForest Otimizado) no conjunto de validação...")
y_pred_best_rf = melhor_modelo_rf_random.predict(X_val)

# Gerar o relatório de classificação e a matriz de confusão
print("\nRelatório de Classificação (RandomForest Otimizado):")
print(classification_report(y_val, y_pred_best_rf))

print("\nMatriz de Confusão (RandomForest Otimizado):")
cm_best_rf = confusion_matrix(y_val, y_pred_best_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_best_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bom Pagador', 'Inadimplente'], 
            yticklabels=['Bom Pagador', 'Inadimplente'])
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - RandomForest Otimizado')

# Salvar a figura para o relatório
plt.savefig('matriz_confusao_rf_otimizado.png', dpi=300, bbox_inches='tight')

plt.show()