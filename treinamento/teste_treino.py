#treino.py
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Lista para armazenar todos os dados e labels
all_data = []
all_labels = []

# Diretório onde os CSVs estão armazenados
data_dir = 'dados_landmarks'

# Ler todos os arquivos CSV e combinar os dados
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_dir, file))
        all_data.append(df.iloc[:, :-1].values)
        all_labels.extend(df.iloc[:, -1].values)

# Concatenar todos os dados em um único array
X = np.concatenate(all_data, axis=0)
y = np.array(all_labels)

# Balanceamento dos dados usando SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Definir a pipeline com o StandardScaler e o SVM
pipeline = make_pipeline(StandardScaler(), SVC(probability=True))

# Definir a grade de parâmetros para busca
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [1, 0.1, 0.01, 0.001],
    'svc__kernel': ['rbf', 'poly', 'sigmoid']
}

# Utilizar GridSearchCV para encontrar os melhores hiperparâmetros
grid_search = GridSearchCV(pipeline, param_grid, refit=True, verbose=2, cv=3)
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado pela busca em grade
best_model = grid_search.best_estimator_

# Avaliação do modelo usando validação cruzada
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print("Pontuações da Validação Cruzada:", cv_scores)
print("Média das Pontuações da Validação Cruzada:", np.mean(cv_scores))

# Salvar o modelo treinado
with open('modelo.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Avaliar o modelo no conjunto de teste
y_pred = best_model.predict(X_test)
print("Acurácia no conjunto de teste:", best_model.score(X_test, y_test))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
