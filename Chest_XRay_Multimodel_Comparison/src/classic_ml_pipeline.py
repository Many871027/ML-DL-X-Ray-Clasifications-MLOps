from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class ClassicMLAlgorithms:
    """Implementa el portafolio de algoritmos clásicos y estadísticos."""
    def __init__(self):
        pass
        
    def train_naive_bayes(self, X_train, y_train):
        print("Entrenando Naive Bayes (Gaussian)...")
        model = GaussianNB()
        model.fit(X_train, y_train)
        return model, {"model": "GaussianNB"}
        
    def train_random_forest(self, X_train, y_train):
        print("Entrenando Random Forest...")
        params = {'n_estimators': 100, 'class_weight': 'balanced'}
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        return model, params
        
    def train_knn(self, X_train, y_train):
        print("Entrenando KNN (GridSearch)...")
        # Mini GridSearch
        param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
        grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"Mejores params KNN: {grid.best_params_}")
        return grid.best_estimator_, grid.best_params_
        
    def train_svm(self, X_train, y_train):
        print("Entrenando SVM (RBF)...")
        params = {'kernel': 'rbf', 'class_weight': 'balanced'}
        model = SVC(**params, random_state=42)
        model.fit(X_train, y_train)
        return model, params
