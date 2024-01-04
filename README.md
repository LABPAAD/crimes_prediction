# Predição de Crimes
Preditor de taxas de crimes anuais em distritos policiais (DP) da cidade de São Paulo em função de pontos de interesse e características de usuários do Twitter na região do DP

> Ferramentas (sklearn) e os modelos

Nossos experimentos foram realizados com as implementações de métodos de regressão da biblioteca *scikit-learn* da linguagem *Python*~[Pedregosa et al., 2011](https://scikit-learn.org/stable/whats_new/v0.24.html).

Os métodos de aprendizado de máquina utilizados são:

- *Floresta Aleatória* (FA)
- *Support Vector Regression* (SVR)
- *Gradient Boosting Regressor* (GBR)
- *K-Nearest Neighbors* (KNN)~[Drucker et al., 1997](https://www.microsoft.com/en-us/research/people/cjdrucker/), [Breiman, 2001](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm).

Para predizer a categoria de crime em uma região alvo $a$, desconsiderando essa região nos valores a serem preditos $y$ e na matriz de atributos $X$ para fins de avaliação do modelo.
Nesse sentido, adotamos a metodologia _leave out one_ que consiste em predizer o total de crimes para uma região utilizando dados das outras regiões.
[Mais detalhes](https://github.com/LABPAAD/crimes_prediction/blob/main/TCC_3___Saul_Rocha.pdf)

## Datasets

1. **Tweets** 
    Os tweets foram coletados com a APIv2 do Twitter utilizando a biblioteca _tweepy_ e resulta nos arquivos CSVs no diretório [tweets coletados](https://github.com/LABPAAD/crimes_prediction/blob/main/Datasets/tweets_coletados). Em [Coleta de Tweets](https://github.com/LABPAAD/crimes_prediction/blob/main/Codes/coleta_tweets.ipynb) contém o código utilizado para gerar os arquivos CSVs.

    [!IMPORTANT] São necessários os arquivos do diretorio "Datasets/mapas" para poder ser feita a coleta.

2. **Pontos de Interesse**

    Os pontos de Interesses de cada DP foram extraídos com a API do [Open Street Maps](https://www.openstreetmap.org/). Mais detalhes sobre essa abordagem pode ser vista em [Extração de POIs](https://github.com/LABPAAD/urban_computing)

3. **Dados de Crimes Oficiais**
    Os dados de crimes oficiais foram coletados da Secretaria de Segurança Pública do Estado de São Paulo disponível em: [SSP](https://www.ssp.sp.gov.br/estatistica)

## Execução dos modelos

1. **Importação de Bibliotecas:**
Importa bibliotecas necessárias, como pandas, numpy, matplotlib, e vários modelos de machine learning do scikit-learn.

2. **Leitura de Dados:**
Lê dois conjuntos de dados: "POIs.csv" e "dataset_soma_ocorr_2022.csv".
Realiza algumas operações de pré-processamento nos dados.

3. **Random Forest:**
Utiliza o modelo RandomForestRegressor para prever as ocorrências criminais usando as características fornecidas em "POIs.csv".
Usa a técnica de validação cruzada Leave-One-Out (LOO) para avaliar o desempenho do modelo.

4. **Support Vector Machine**
Utiliza o modelo Support Vector Machine (SVM) com kernel linear para prever as ocorrências criminais.
Adiciona características de usuários ao conjunto de dados.
Usa LOO para validação cruzada.

5. **Gradient Boosting Regressor:**
Utiliza o modelo Gradient Boosting Regressor para prever as ocorrências criminais usando as características fornecidas em "POIs.csv".
Usa LOO para validação cruzada.

6. **K-Nearest Neighbors:**
Utiliza o modelo K-Nearest Neighbors (KNN) para prever as ocorrências criminais usando as características fornecidas em "POIs.csv".
Usa LOO para validação cruzada.

7. **Salvando Resultados:**
Salva as métricas de desempenho, importâncias de características e erros em arquivos CSV.

8. **Resumo de Métricas:**
Calcula métricas como RAE (Relative Absolute Error), MAE (Mean Absolute Error) e R2 (R-squared) para avaliar o desempenho dos modelos em cada tipo de ocorrência.

9. **Salvando Resultados:**
Salva as métricas em arquivos CSV.

10. Observações
- Certifique-se de ter o ambiente apropriado configurado com as bibliotecas necessárias.
- Certifique-se de que os conjuntos de dados "Datasets/dataset_features_users_POIs.csv" e "dataset_soma_ocorr_2022.csv" estejam disponíveis no mesmo diretório ou no Google Drive conforme esperado.
- Execute o código em um ambiente Python, como Jupyter Notebook ou Google Colab para ter mais flexibilidade nas execuções ou se preferir pode usar o arquivo "models.py".
- Pode também executar apenas o modelo especifico no diretorio Codes/
