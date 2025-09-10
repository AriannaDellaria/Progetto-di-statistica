"""
Progetto Statistica 
Arianna Dellaria - 0001125416
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
plt.close('all')

#%% fase 1 - Caricare il dataset 
#url = "https://www.kaggle.com/datasets/shivam2503/diamonds"

df = pd.read_csv("diamonds.csv")
#print(df.shape) #dimensione iniziale del dataset


#%% fase 2 - Pre-Proccesing 

#Controllo la presenza di valori NaN e elimino le righe corrispondenti
df = df.dropna()

# Rimuovo le colonne non utili
data = df.drop(columns=['x', 'y', 'z'])

# Imposto gli indici per la rimozione delle righe
start_index = 10000
end_index = 53940

# Rimuovo tutte le righe a partire dalla riga 10000 (impostato nell'indice)
data = data.drop(data.index[start_index:end_index])

#trasformo la colonna della chiarezza in numerica dalla peggiore alla migliore
clarity_map = {
    'I1': 0,
    'SI2': 1,
    'SI1': 2,
    'VS2': 3,
    'VS1': 4,
    'VVS2': 5,
    'VVS1': 6,
    'IF': 7
}
data['clarity'] = data['clarity'].map(clarity_map)

#trasformo la colonna del colore in numerica dal peggiore al migliore
color_map = {
    'J': 0,
    'I': 1,
    'H': 2,
    'G': 3,
    'F': 4,
    'E': 5,
    'D': 6
}
data['color'] = data['color'].map(color_map)


# Definisco una funzione per trasformare la colonna 'cut' in numerica
def trasforma_cut(val):
    if val == 'Premium':
        return 1
    elif val == 'Ideal':
        return 2
    else:
        return 3

# Applico la funzione alla colonna 'cut'
data['cut'] = data['cut'].apply(trasforma_cut)

#Rimuovo gli outlier usando IQR
numeric_cols = ['carat', 'price', 'depth', 'table']

for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    
#dataset aggiornato
print(data)


#%% fase 3 - Exploratory Data Analysis (EDA) 

#non uso la colonna degli indici
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])
    
#vediamo la relazione tra prezzo, carati e cut
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='carat', y='price', hue='cut', palette='Set2', alpha=0.6)
plt.title('Prezzo vs Carati, colorato per Cut')
plt.show()
plt.close()

#distribuzione del prezzo in base alle variabili discrete (cut, color, clarity)
# Lista delle variabili discrete
discrete_vars = ['cut', 'color', 'clarity']
# Imposta la dimensione della figura
plt.figure(figsize=(18, 5))

# istogrammi delle variabili discrete
fig, axes = plt.subplots(1, len(discrete_vars), figsize=(8 * len(discrete_vars), 5))
if len(discrete_vars) == 1:
    axes = [axes]
# Creiamo gli istogrammi
for i, var in enumerate(discrete_vars):
    sns.histplot(data[var], bins=20, kde=False, color='green', ax=axes[i])
    axes[i].set_title(f'Distribution of {var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Frequency')
# Spaziatura tra i subplots
plt.tight_layout()
plt.show()
plt.close()

#boxplot variabili continue
variabili_continue = ['carat', 'price', 'depth', 'table']
fig, axes = plt.subplots(1, len(variabili_continue), figsize=(20, 5))  # 1 riga, n colonne
for i, var in enumerate(variabili_continue):
    sns.boxplot(y=data[var], ax=axes[i], color='lightblue')
    axes[i].set_title(f'Boxplot of {var}')
    axes[i].set_xlabel(var)
plt.tight_layout()
plt.show()
plt.close()

#Istogramma con curva di densità per variabili continue
fig, axes = plt.subplots(4, 1, figsize=(15, 15))
axes = axes.flatten()
for i, var in enumerate(variabili_continue):
    sns.histplot(data[var], kde=True, color='orange', ax=axes[i])
    axes[i].set_title(f'Histogram and Density of {var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.close()

#rappresentiamo la matrice di correlazione per indagare le relazione statistiche
numerical_cols = data.select_dtypes(include=['int64', 'float64'])
corr_matrix = numerical_cols.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title('Matrice di Correlazione')
plt.show()
plt.close()

#%% fase 4 - Splitting

# Definisci variabili di input X e target y
X = data.drop(columns=['cut'])  # input: tutte tranne target 'cut' 
y = data['cut']    # target

X_copy = X.copy() 

# Suddivisione in train (80%) e test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_copy, y, test_size=0.2, random_state=42, stratify=y)

# Suddivisione in train + validation (80%) e test (20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)  
# 0.25 * 0.8 = 0.2 → train 60%, val 20%, test 20%

print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
print()
#%% fase 5 - Addestramento del Modello

# 5. Regressione Logistica
# Standardizzazione dei dati (sia train che validation)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

logreg = LogisticRegression(max_iter=5000) 
logreg.fit(X_train_scaled, y_train)

y_val_pred_logreg = logreg.predict(X_val_scaled)
acc_logreg = accuracy_score(y_val, y_val_pred_logreg)
print(f"Accuratezza Regressione Logistica sul validation set: {acc_logreg:.4f}")
#print("Classification Report Regressione Logistica:\n", classification_report(y_val, y_val_pred_logreg))

# Matrice di confusione Regressione Logistica
conf_mat_logreg = confusion_matrix(y_val, y_val_pred_logreg)
plt.figure(figsize=(6,5))
sns.heatmap(conf_mat_logreg, annot=True, fmt='d', cmap='Greens')
plt.title("Matrice di Confusione Regressione Logistica")
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.show()
plt.close()
#%%
#SVM con kernel lineare
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train)
y_val_pred_svm_linear = svm_linear.predict(X_val_scaled)

print("Accuratezza SVM Lineare:", accuracy_score(y_val, y_val_pred_svm_linear))
#print("\nClassification Report SVM:\n", classification_report(y_val, y_val_pred_svm))

# Matrice di confusione
conf_matrix_svm = confusion_matrix(y_val, y_val_pred_svm_linear)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix_svm, annot=True, cmap='Greens', fmt='d')
plt.title("Confusion Matrix SVM Linear")
plt.show()
plt.close()

#%%
# SVM con kernel polinomiale
svm_poly = SVC(kernel='poly', degree=3, random_state=42)
svm_poly.fit(X_train_scaled, y_train)
y_val_pred_poly = svm_poly.predict(X_val_scaled)

# Accuratezza e matrice di confusione
print("Accuratezza SVM Polinomiale:", accuracy_score(y_val, y_val_pred_poly))

conf_mat_poly = confusion_matrix(y_val, y_val_pred_poly)
plt.figure(figsize=(6,5))
sns.heatmap(conf_mat_poly, annot=True, cmap='Greens', fmt='d')
plt.title("Confusion Matrix SVM Polinomiale")
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.show()
plt.close()

#%%
# SVM con kernel RBF (gaussiano)
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_val_pred_rbf = svm_rbf.predict(X_val_scaled)

# Accuratezza e matrice di confusione
print("Accuratezza SVM RBF:", accuracy_score(y_val, y_val_pred_rbf))
print()
conf_mat_rbf = confusion_matrix(y_val, y_val_pred_rbf)
plt.figure(figsize=(6,5))
sns.heatmap(conf_mat_rbf, annot=True, cmap='Greens', fmt='d')
plt.title("Confusion Matrix SVM RBF")
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.show()
plt.close()

#%% fase 6 - Valutazione della Performance sui modelli

models = {
    "Logistic Regression": logreg,
    "SVM Lineare": svm_linear,
    "SVM RBF": svm_rbf,
    "SVM Polynomial": svm_poly
}

X_test_scaled = scaler.transform(X_test)
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

#%% fase 7 - Hyperparameter Tuning 

from sklearn.model_selection import GridSearchCV

# Standardizza una volta sola
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

#Parametri per Logistic Regression
param_grid_logreg = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

#Parametri per SVM
param_grid_svm = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10]},
    {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]},
    {'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [2, 3, 4]}
]


#GridSearch per Logistic Regression
y_train = np.asarray(y_train)
grid_search_logreg = GridSearchCV(LogisticRegression(max_iter=1000),
                                  param_grid_logreg, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_logreg.fit(X_train_scaled, y_train)

#GridSearch per SVM
y_train = np.asarray(y_train)
grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_svm.fit(X_train_scaled, y_train)

#Confronto delle migliori performance
best_score_logreg = grid_search_logreg.best_score_
best_score_svm = grid_search_svm.best_score_

print("\n=== Migliori Parametri Logistic Regression ===")
print(grid_search_logreg.best_params_)
print(f"Best Cross-Validation Score: {best_score_logreg:.4f}")

print("\n=== Migliori Parametri SVM ===")
print(grid_search_svm.best_params_)
print(f"Best Cross-Validation Score: {best_score_svm:.4f}")

#Determina il modello migliore
#Determina il modello migliore con info kernel
if best_score_logreg > best_score_svm:
    best_model = LogisticRegression(**grid_search_logreg.best_params_, max_iter=10000)
    best_model_name = "Logistic Regression"
    print("\n>>> Logistic Regression è il modello migliore.")
else:
    best_params = grid_search_svm.best_params_
    best_model = SVC(**best_params)
    kernel = best_params['kernel']
    best_model_name = f"SVM con kernel {kernel}"
    if kernel == 'poly':
        best_model_name += f" (degree={best_params.get('degree', 'default')})"
    elif kernel == 'rbf':
        best_model_name += f" (gamma={best_params.get('gamma', 'default')})"
    print(f"\n>>> {best_model_name} è il modello migliore.")

#Allena il modello migliore sull’intero training set
best_model.fit(X_train_scaled, y_train)
y_val_pred = best_model.predict(X_val_scaled)
acc_val = accuracy_score(y_val, y_val_pred)
print(f"Accuracy del modello migliore ({best_model_name}) sul Validation Set: {acc_val:.4f}")


#%% fase 8 - Studio Statistico dei Risultati
import seaborn as sns
import matplotlib.pyplot as plt
import random

# Imposta parametri
k = 10  # numero di iterazioni
z = 1.96  # z-score per intervallo di confidenza al 95%

# Lista per memorizzare le accuracy
accuracy_list = []

# Genera k random_state casuali diversi
rs = random.sample(range(1, 10000), k)

for i in range(k):
    # Risplit randomico
    X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(
        X_copy, y, test_size=0.2, random_state=rs[i], stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled_tmp = scaler.fit_transform(X_train_tmp)
    X_test_scaled_tmp = scaler.transform(X_test_tmp)
    
    # Addestramento del modello migliore
    best_model.fit(X_train_scaled_tmp, y_train_tmp)
    
    # Predizione e calcolo dell'accuracy
    y_test_pred_tmp = best_model.predict(X_test_scaled_tmp)
    acc_tmp = accuracy_score(y_test_tmp, y_test_pred_tmp)
    accuracy_list.append(acc_tmp)

# Analisi Statistica
accuracy_array = np.array(accuracy_list)
mean_acc = np.mean(accuracy_array)
std_acc = np.std(accuracy_array, ddof=1)
conf_int = (
    mean_acc - z * (std_acc / np.sqrt(k)),
    mean_acc + z * (std_acc / np.sqrt(k))
)

print("\n=== Studio Statistico sul Modello Migliore ===")
print(f"Accuracy Media: {mean_acc:.4f}")
print(f"Deviazione Standard: {std_acc:.4f}")
print(f"Intervallo di Confidenza (95%): {conf_int}")

# Istogramma
plt.figure(figsize=(10, 5))
sns.histplot(accuracy_array, bins=10, kde=True, color='skyblue')
plt.title(f"Distribuzione delle Accuracy (Modello Migliore)")
plt.xlabel("Accuracy")
plt.ylabel("Frequenza")
plt.show()
plt.close()

#%% Fase 9 - Regressione Lineare
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats
from scipy.stats import shapiro

print("\n=== Regressione lineare ===")
# Caricamento del dataset
#url = "https://www.kaggle.com/datasets/hussainnasirkhan/multiple-linear-regression-dataset"
data = pd.read_csv("multiple_linear_regression_dataset.csv")
data = data.dropna() #levo i Nan

#rimuovo gli outliers
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Rimuovo outlier 
data = remove_outliers_iqr(data, 'experience')
data = remove_outliers_iqr(data, 'income')

# Preparo i dati per la regressione
X = data[['experience']].values  # variabile indipendente
y = data['income'].values        # variabile dipendente

# 2. Standardizza X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Creo e addestro il modello
lr = LinearRegression()
lr.fit(X, y)

# Predizione
y_pred = lr.predict(X)

# Scatter plot con retta di regressione
plt.figure(figsize=(10,6))
sns.scatterplot(x=data['experience'], y=data['income'], color="blue", s=50)
plt.plot(data['experience'], y_pred, color="red", linewidth=2)
plt.title("Experience vs Income – Regressione lineare")
plt.xlabel("Experience (anni)")
plt.ylabel("Income")
plt.show()

# Stima dei coefficienti
print("Termine noto (intercetta):", lr.intercept_)
print("Pendenza:", lr.coef_[0])

#  Calcolo MSE e R^2
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")
r2 = r2_score(y, y_pred)
print(f"R^2: {r2:.4f}")
if r2 < 0.3:
    print("Attenzione: modello con bassa capacità esplicativa (R^2 basso).")

# Analisi residui
residui = y - y_pred

# Test di Shapiro-Wilk per la normalità dei residui
stat, p_value = shapiro(residui)
print(f"Shapiro-Wilk Test statistic: {stat:.4f}, p-value: {p_value:.4f}")

if p_value > 0.05:
    print("I residui sembrano seguire una distribuzione normale (non rifiutiamo H0).")
else:
    print("I residui NON seguono una distribuzione normale (rifiutiamo H0).")

# Istogramma dei residui
plt.figure(figsize=(8,5))
sns.histplot(residui, kde=True, color="green", bins=15)
plt.title("Distribuzione dei residui")
plt.xlabel("Residui")
plt.show()

# QQ-plot dei residui
plt.figure(figsize=(6,6))
stats.probplot(residui, dist="norm", plot=plt)
plt.title("QQ-plot dei residui")
plt.show()