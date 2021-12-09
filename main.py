import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils import resample
import matplotlib.pyplot as plt

if not os.path.exists('plots'):
    os.mkdir('plots')
if not os.path.exists('plots/prt'):
    os.mkdir('plots/prt')
if not os.path.exists('plots/roc'):
    os.mkdir('plots/roc')
if not os.path.exists('plots/cm'):
    os.mkdir('plots/cm')

# Pega os dados do dataset
df = pd.read_csv("03-11/Portmap.csv",  dtype={'SimillarHTTP': 'str'})

# Ajusta o nome do header
df = df.rename(columns=lambda n: n.strip())

# Limpa dados invalidos
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Pelo fato do dataset ser muito grande, foi feito um resample para um dataset
# menor
df = resample(df, n_samples=1000, stratify=df['Label'])

# Vetoriza a classificação dos dados
lb = LabelBinarizer()
df['Label'] = lb.fit_transform(df['Label'].values)

X = df.copy()
y = df['Label']

# Remove colunas não vetorizaveis
X.drop('Label', axis=1, inplace=True)
X.drop('Flow ID', axis=1, inplace=True)
X.drop('SimillarHTTP', axis=1, inplace=True)
X.drop('Source IP', axis=1, inplace=True)
X.drop('Destination IP', axis=1, inplace=True)
X.drop('Timestamp', axis=1, inplace=True)

# Separa os dados entre as amostras de teste e treino.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=12, stratify=df['Label'])

# Mostra a distribuição
y_train_counter = y_train.value_counts()
y_test_counter = y_test.value_counts()

# Printa a distribuição dos dados
print('=========================Distribuição de Classes (y_train)=========================')
print(y_train.value_counts(normalize=True))
print()
print(y_train_counter),
print('=========================Distribuição de Classes (y_test)=========================')
print(y_test.value_counts(normalize=True))
print()
print(y_test_counter)

# Cria um gráfico contendo a distribuição
plot = plt.bar(["Train(Benign)", "Train(Attack)", "Test(Benign)", "Test(Attack)"], [
    y_train_counter[0], y_train_counter[1], y_test_counter[0], y_test_counter[1]])
plot[0].set_color('b')
plot[2].set_color('b')
plt.savefig('plots/distribution.png')


def grid_search_wrapper(estimator, param_grid, refit_score, name):
    """
    Busca os melhores hiperparametros para o estimador dado.
    """
    scoring = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score),
    }
    skf = StratifiedKFold(n_splits=3)
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring,
                               refit=refit_score, cv=skf, return_train_score=True, n_jobs=1)
    grid_search.fit(X_train.values, y_train.values)
    y_pred = grid_search.predict(X_test.values)
    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    print('Confusion matrix')
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, columns=[
          'pred_neg', 'pred_pos'], index=['neg', 'pos']))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('plots/cm/' + name + '.png')

    return grid_search


def adjusted_classes(y_scores, t):
    """
    Discretiza os dados de acordo com um limiar
    """
    return [1 if y >= t else 0 for y in y_scores]


def plot_roc_curve(fpr, tpr, label, name):
    """
    Gera a curva ROC
    """
    plt.figure(figsize=(8, 8))
    plt.title('ROC Curve(' + name + ')')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, name):
    """
    Gera um gráfico comparando a relação entre a precisão e o recall ao mudar o limiar.
    """
    plt.figure(figsize=(8, 8))
    plt.title(
        "Precision and Recall Scores as a function of the decision threshold(" + name + ')')
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')


def strategy(proba, name, estimator, param_grid):
    """
    Implementação da estratégia 
    """
    # Procura pelos melhores hiperparametros
    grid_search = grid_search_wrapper(
        estimator, param_grid, 'precision_score', name)

    # Gera a probabilidade da amostragem de teste de pertencer a cada classe
    y_scores = proba(grid_search, X_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores)

    # Gerá o gráfico ROC
    print(auc(fpr, tpr))  # AUC of ROC
    plot_roc_curve(fpr, tpr, 'precision_otimized', name)
    plt.savefig('plots/roc/{}.png'.format(name))

    # Imprime o precision, recall, f1-score, accuracy.
    print(classification_report(y_test, grid_search.predict(X_test)))
    p, r, thresholds = precision_recall_curve(y_test, y_scores)

    # Gera o gráfico da relação entre a precision e o recall com o limiar
    plot_precision_recall_vs_threshold(p, r, thresholds, name)
    plt.savefig('plots/prt/{}.png'.format(name))


# Executa a estratégia para uma KNN.
print("Strategy with KNN")
strategy(
    lambda grid_search, X_test: grid_search.predict_proba(X_test)[:, 1],
    'knn',
    KNeighborsClassifier(),
    param_grid={
        'n_neighbors': [1, 2, 3, 4, 5],
    }
)

# Executa a estratégia para uma RandomForest.
print("Strategy with RFC")
strategy(
    lambda grid_search, X_test: grid_search.predict_proba(X_test)[:, 1],
    'rfc',
    RandomForestClassifier(n_jobs=-1),
    param_grid={
        'min_samples_split': [3, 5, 10],
        'n_estimators': [100, 300],
        'max_depth': [3, 5, 15, 25],
        'max_features': [3, 5, 10, 20],
    }
)

# Executa a estratégia para uma SVC.
print("Strategy with SVC")
strategy(
    lambda grid_search, X_test: grid_search.decision_function(X_test),
    'svc',
    SVC(),
    param_grid={
        'kernel': ['linear'],
        'gamma': [0.1, 1, 10, 100],
        'C': [0.1, 1, 10, 100, 1000]
    }
)
