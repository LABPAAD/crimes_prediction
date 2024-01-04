
"""models.ipynb

Original file is located at
    https://colab.research.google.com/drive/1P3FZK0fCq35ysIoyEyIi9MaKzRFkxXv-

### Modelagem:

* taxa_crime_k_regiao_i = caracteristica_j_usr_regiao_i

* taxa_crime_k_regiao_i: ideal que seja uma média anual 5 anos

* taxa_crime_k_regiao_i = a*caracteristica_0_usr_regiao_i + b*caracteristica_1_usr_regiao_i + c*caracteristica_2_usr_regiao_i + d*caracteristica_3_usr_regiao_i + e*caracteristica_4_usr_regiao_i + f*caracteristica_5_usr_regiao_i

# Características de usuários na região
- Média valores dos usuários na região

* Características do usuário
 * 0 perfil-protegido: {0,1}
 * 1 qtd_seguidores_ano: número real  (qtd_seguidores/anos_conta)
 * 2 qtd_seguindo_ano: número real
 * 3 qtd_tweets_ano: número real
 * 4 anos_conta: número inteiro
 * 5 perfil-verificado: {0,1}

### Metodologia de avaliação: one-leaves-out

* para cada região i,
    remover a linha i das caracteristicas
    treinar o modelo
    registrar o erro para i (matriz de erros: linha i colunas MAE, RAE, R2)

* Calcular Média de MAE, RAE, R2 para tipo de crime k

"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""### Tratando os dados"""

features = pd.read_csv("POIs.csv")
features = features.drop(columns=["Unnamed: 0"])


targets = pd.read_csv("dataset_soma_ocorr_2022.csv")
targets = targets.drop(columns=["Unnamed: 0"])



"""## Random Forest (POIs)"""

l1 = list(np.arange(1,54))
l2 = [56,57,58,62,63,64,65,66,69,70,73,75,77,78,80,81,85,87,89,90,91,92,96,97,98,99,101,102,103]
areas = l1+l2

features = pd.read_csv("POIs_Primary/POIs.csv")
#features.drop(columns=["Unnamed: 0"], inplace=True)



targets = pd.read_csv("/content/drive/MyDrive/courb2/POIs_Primary/dataset_soma_ocorr_2022.csv")
targets = targets.drop(columns=["Unnamed: 0"])

targets = targets[["total_homicidios", "total_estupro", "total_les_corporal", "total_roubo2", "total_furt"]]


# Definindo a semente aleatória como 42
np.random.seed(42)

# separar as features
X = features.values

#normaliza
scaler = StandardScaler()
X = scaler.fit_transform(X)

# criar um objeto do modelo Random Forest Regressor
model = RandomForestRegressor()

# criar um objeto LeaveOneOut
loo = LeaveOneOut()

importances = pd.DataFrame()
erros = pd.DataFrame()
#matriz de metricas RAE, MAE e R2
metricas = pd.DataFrame(columns=['Ocorrencia','RAE','MAE','R2'])
ocorrencias = []
media_rae = []
media_mae =[]
media_r2 = []

for i in targets.columns:

  list_y_test = []
  list_y_pred = []
  # separa a variável target
  y = targets[i].values

  # percorrer cada amostra do conjunto de dados
  for train_index, test_index in loo.split(X):

      # separar os conjuntos de treinamento e teste
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      # treinar o modelo
      model.fit(X_train, y_train)

      #adiciona a importancia das features para predição da target i
      importances[i] = model.feature_importances_

      # fazer a previsão usando o conjunto de teste
      y_pred = model.predict(X_test)

      list_y_test.append(y_test[0])
      list_y_pred.append(y_pred[0])

  erros[i+"_true"] = list_y_test
  erros[i+"_pred"] = list_y_pred
  # calcular o erro relativo absoluto (RAE) e adiciona na lista
  media_rae.append(mean_absolute_error(list_y_test, list_y_pred) / np.mean(list_y_test))
  # calcular o erro absoluto medio (MAE) e adiciona na lista
  media_mae.append(mean_absolute_error(list_y_test, list_y_pred))
  # calcular R2 e adiciona na lista
  media_r2.append(r2_score(list_y_test,list_y_pred))
  #tipo de ocorrencia
  ocorrencias.append(i)


metricas["Ocorrencia"] = ocorrencias
metricas["RAE"] = media_rae
metricas["MAE"] = media_mae
metricas["R2"] = media_r2

# metricas.to_csv("metricas2022.csv")

importances["feature"] = list(features.columns)

importances.to_csv("/content/drive/MyDrive/courb2/POIs_Primary/importances2022.csv")


"""## SVM (POIs + Caractrísticas de Usuários)"""


# Definindo a semente aleatória como 42
np.random.seed(42)

# separar as features
X = features.values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# criar modelo SVM
model = SVR(kernel='linear')

# normalização min-max
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
# normalização z-score
# scaler_zscore = StandardScaler()
# X_zscore = scaler_zscore.fit_transform(X)

# criar um objeto LeaveOneOut
loo = LeaveOneOut()

#erros
errosSVM = pd.DataFrame()

#matriz de metricas RAE, MAE e R2
metricas = pd.DataFrame(columns=['Ocorrencia','RAE','MAE','R2'])
ocorrencias = []
media_rae = []
media_mae =[]
media_r2 = []



for i in targets.columns:

  list_y_test = []
  list_y_pred = []
  # separa a variável target
  y = targets[i].values


  # percorrer cada amostra do conjunto de dados
  for train_index, test_index in loo.split(X_minmax):

      # separar os conjuntos de treinamento e teste
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      # treinar o modelo
      model.fit(X_train, y_train)

      # fazer a previsão usando o conjunto de teste
      y_pred = model.predict(X_test)

      list_y_test.append(y_test[0])
      list_y_pred.append(y_pred[0])

      #print("index do teste", test_index)
      #print("index do treino", train_index)
  errosSVM[i+"y_true"] = list_y_test
  errosSVM[i+"y_pred"] = list_y_pred
  # # calcular o erro relativo absoluto (RAE) e adiciona na lista
  media_rae.append(mean_absolute_error(list_y_test, list_y_pred) / np.mean(list_y_test))
  # # calcular o erro absoluto medio (MAE) e adiciona na lista
  media_mae.append(mean_absolute_error(list_y_test, list_y_pred))
  # # calcular R2 e adiciona na lista
  media_r2.append(r2_score(list_y_test,list_y_pred))
  # #tipo de ocorrencia
  ocorrencias.append(i)


metricas["Ocorrencia"] = ocorrencias
metricas["RAE"] = media_rae
metricas["MAE"] = media_mae
metricas["R2"] = media_r2


# metricas.to_csv("/content/drive/MyDrive/courb2/POIs_Primary/metricas_SVM_POIs_primary2022.csv")


# errosSVM.to_csv("/content/drive/MyDrive/courb2/POIs_Primary/erros_SVM_POIs_primary2022.csv")

"""## Linear Regression (POIs)"""



# separar as features
X = features_pois.values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# criar um objeto do modelo Random Forest Regressor
model = LinearRegression()


# criar um objeto LeaveOneOut
loo = LeaveOneOut()

#matriz de metricas RAE, MAE e R2
metricas = pd.DataFrame(columns=['Ocorrencia','RAE','MAE','R2'])
ocorrencias = []
media_rae = []
media_mae =[]
media_r2 = []


for i in targets.columns[1:]:
  list_y_test = []
  list_y_pred = []
  # separa a variável target
  y = targets[i].values


  # percorrer cada amostra do conjunto de dados
  for train_index, test_index in loo.split(X):

      # separar os conjuntos de treinamento e teste
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      # treinar o modelo
      model.fit(X_train, y_train)

      #adiciona a importancia das features para predição da target i
      #importances[i] = model.feature_importances_

      # fazer a previsão usando o conjunto de teste
      y_pred = model.predict(X_test)

      list_y_test.append(y_test)
      list_y_pred.append(y_pred)


  # calcular o erro relativo absoluto (RAE) e adiciona na lista
  media_rae.append(mean_absolute_error(list_y_test, list_y_pred) / np.mean(list_y_test))
  # calcular o erro absoluto medio (MAE) e adiciona na lista
  media_mae.append(mean_absolute_error(list_y_test, list_y_pred))
  # calcular R2 e adiciona na lista
  media_r2.append(r2_score(list_y_test,list_y_pred))
  #tipo de ocorrencia
  ocorrencias.append(i)


metricas["Ocorrencia"] = ocorrencias
metricas["RAE"] = media_rae
metricas["MAE"] = media_mae
metricas["R2"] = media_r2

metricas

metricas.to_csv("/content/drive/MyDrive/courb2/CSVs/metricas_LR_82areas_POIs_only_22a23.csv")

"""## GBM"""

# Definindo a semente aleatória como 42
np.random.seed(42)

# separar as features
X = features.values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# criar um objeto do modelo Random Forest Regressor
model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# criar um objeto LeaveOneOut
loo = LeaveOneOut()

#matriz de metricas RAE, MAE e R2
metricas = pd.DataFrame(columns=['Ocorrencia','RAE','MAE','R2'])
ocorrencias = []
media_rae = []
media_mae =[]
media_r2 = []


for i in targets.columns:
  list_y_test = []
  list_y_pred = []
  # separa a variável target
  y = targets[i].values


  # percorrer cada amostra do conjunto de dados
  for train_index, test_index in loo.split(X):

      # separar os conjuntos de treinamento e teste
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      # treinar o modelo
      model.fit(X_train, y_train)

      #adiciona a importancia das features para predição da target i
      #importances[i] = model.feature_importances_

      # fazer a previsão usando o conjunto de teste
      y_pred = model.predict(X_test)

      list_y_test.append(y_test)
      list_y_pred.append(y_pred)


  # calcular o erro relativo absoluto (RAE) e adiciona na lista
  media_rae.append(mean_absolute_error(list_y_test, list_y_pred) / np.mean(list_y_test))
  # calcular o erro absoluto medio (MAE) e adiciona na lista
  media_mae.append(mean_absolute_error(list_y_test, list_y_pred))
  # calcular R2 e adiciona na lista
  media_r2.append(r2_score(list_y_test,list_y_pred))
  #tipo de ocorrencia
  ocorrencias.append(i)


metricas["Ocorrencia"] = ocorrencias
metricas["RAE"] = media_rae
metricas["MAE"] = media_mae
metricas["R2"] = media_r2

metricas.to_csv("/content/drive/MyDrive/courb2/CSVs/metricas_GBM_82areas_POIs_users_22a23.csv")



"""## KNN"""

# Definindo a semente aleatória como 42
np.random.seed(42)

# separar as features
X = features.values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# criar um objeto do modelo Random Forest Regressor
model = KNeighborsRegressor(n_neighbors=5)

# criar um objeto LeaveOneOut
loo = LeaveOneOut()

#matriz de metricas RAE, MAE e R2
metricas = pd.DataFrame(columns=['Ocorrencia','RAE','MAE','R2'])
ocorrencias = []
media_rae = []
media_mae =[]
media_r2 = []


for i in targets.columns:
  list_y_test = []
  list_y_pred = []
  # separa a variável target
  y = targets[i].values


  # percorrer cada amostra do conjunto de dados
  for train_index, test_index in loo.split(X):

      # separar os conjuntos de treinamento e teste
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      # treinar o modelo
      model.fit(X_train, y_train)

      #adiciona a importancia das features para predição da target i
      #importances[i] = model.feature_importances_

      # fazer a previsão usando o conjunto de teste
      y_pred = model.predict(X_test)

      list_y_test.append(y_test)
      list_y_pred.append(y_pred)


  # calcular o erro relativo absoluto (RAE) e adiciona na lista
  media_rae.append(mean_absolute_error(list_y_test, list_y_pred) / np.mean(list_y_test))
  # calcular o erro absoluto medio (MAE) e adiciona na lista
  media_mae.append(mean_absolute_error(list_y_test, list_y_pred))
  # calcular R2 e adiciona na lista
  media_r2.append(r2_score(list_y_test,list_y_pred))
  #tipo de ocorrencia
  ocorrencias.append(i)


metricas["Ocorrencia"] = ocorrencias
metricas["RAE"] = media_rae
metricas["MAE"] = media_mae
metricas["R2"] = media_r2


metricas.to_csv("/content/drive/MyDrive/courb2/CSVs/metricas_KNN_82areas_POIs_users_22a23.csv")
