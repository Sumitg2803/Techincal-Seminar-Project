import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier  # KNN Classifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mealpy.evolutionary_based import GA  # ✅ Changed optimizer to GA
from mealpy import FloatVar
import warnings
warnings.filterwarnings('ignore')

# Load Mushroom dataset
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    col_names = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]

    data = pd.read_csv(url, header=None, names=col_names)
    data = data.replace('?', np.nan)
    data['stalk-root'].fillna(data['stalk-root'].mode()[0], inplace=True)

    X = data.drop('class', axis=1)
    y = data['class'].map({'e': 0, 'p': 1}).astype(int)

    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Feature scaling for KNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# Objective function
def objective_function(solution):
    global X_train, y_train
    n_neighbors = int(solution[0])
    weights = 'uniform' if solution[1] < 0.5 else 'distance'

    try:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        scores = cross_val_score(knn, X_train, y_train, cv=3, scoring='accuracy')
        fitness = -np.mean(scores)  # minimize negative accuracy
    except:
        fitness = 1.0
    return fitness

# GA Optimization
def optimize_knn():
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = load_data()

    print("Starting K-Nearest Neighbors optimization with GA on Mushroom dataset...")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print("-" * 50)

    problem = {
        "bounds": [
            FloatVar(lb=1, ub=30, name="n_neighbors"),
            FloatVar(lb=0, ub=1, name="weights_choice")  # 0=uniform, 1=distance
        ],
        "minmax": "min",
        "obj_func": objective_function
    }

    optimizer = GA.OriginalGA(epoch=5, pop_size=20)  # ✅ GA optimizer
    best_agent = optimizer.solve(problem)

    best_position = best_agent.solution
    best_fitness = best_agent.target.fitness

    print("\nOptimization Results:")
    print("-" * 50)
    print(f"Best fitness (negative accuracy): {best_fitness:.6f}")
    print(f"Best accuracy: {-best_fitness:.6f}")

    best_params = {
        'n_neighbors': int(best_position[0]),
        'weights': 'uniform' if best_position[1] < 0.5 else 'distance'
    }

    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    return best_params

# Evaluate Model
def evaluate_model(best_params):
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION (K-Nearest Neighbors)")
    print("="*50)

    best_knn = KNeighborsClassifier(**best_params)
    best_knn.fit(X_train, y_train)

    y_pred = best_knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {test_accuracy:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Edible', 'Poisonous']))

    print("\n" + "-"*50)
    print("Comparison with default KNN:")
    print("-" * 50)
    default_knn = KNeighborsClassifier()
    default_knn.fit(X_train, y_train)
    default_pred = default_knn.predict(X_test)
    default_accuracy = accuracy_score(y_test, default_pred)

    print(f"Default KNN Accuracy: {default_accuracy:.6f}")
    print(f"Optimized KNN Accuracy: {test_accuracy:.6f}")
    improvement = test_accuracy - default_accuracy
    print(f"Improvement: {improvement:.6f} ({improvement*100:.2f}%)")

# Main
if __name__ == "__main__":
    best_params = optimize_knn()
    evaluate_model(best_params)
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE! (K-Nearest Neighbors + GA)")
    print("="*50)
