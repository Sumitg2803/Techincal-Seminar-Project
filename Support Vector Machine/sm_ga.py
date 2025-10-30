import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mealpy.evolutionary_based import GA  # GA optimizer
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

    # Feature scaling for Support Vector Machine
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# Objective function for GA
def objective_function(solution):
    global X_train, y_train
    C = 10 ** solution[0]
    gamma = 10 ** solution[1]

    try:
        svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=42)  # Support Vector Machine
        scores = cross_val_score(svm, X_train, y_train, cv=3, scoring='accuracy')
        fitness = -np.mean(scores)  # minimize negative accuracy
    except:
        fitness = 1.0
    return fitness

# GA Optimization
def optimize_svm():
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = load_data()

    print("Starting Support Vector Machine optimization with GA on Mushroom dataset...")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print("-" * 50)

    problem = {
        "bounds": [
            FloatVar(lb=-2, ub=3, name="C (log10 scale)"),       # 10^-2 to 10^3
            FloatVar(lb=-4, ub=1, name="gamma (log10 scale)")    # 10^-4 to 10^1
        ],
        "minmax": "min",
        "obj_func": objective_function
    }

    optimizer = GA.OriginalGA(epoch=5, pop_size=20)  # GA optimizer
    best_agent = optimizer.solve(problem)

    best_position = best_agent.solution
    best_fitness = best_agent.target.fitness

    print("\nOptimization Results:")
    print("-" * 50)
    print(f"Best fitness (negative accuracy): {best_fitness:.6f}")
    print(f"Best accuracy: {-best_fitness:.6f}")

    best_params = {
        'C': 10 ** best_position[0],
        'gamma': 10 ** best_position[1],
        'kernel': 'rbf',
        'random_state': 42
    }

    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    return best_params

# Evaluate Model
def evaluate_model(best_params):
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION (Support Vector Machine)")
    print("="*50)

    best_svm = SVC(**best_params)  # Support Vector Machine
    best_svm.fit(X_train, y_train)

    y_pred = best_svm.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {test_accuracy:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Edible', 'Poisonous']))

    print("\n" + "-"*50)
    print("Comparison with default Support Vector Machine:")
    print("-" * 50)
    default_svm = SVC()
    default_svm.fit(X_train, y_train)
    default_pred = default_svm.predict(X_test)
    default_accuracy = accuracy_score(y_test, default_pred)

    print(f"Default SVM Accuracy: {default_accuracy:.6f}")
    print(f"Optimized SVM Accuracy: {test_accuracy:.6f}")
    improvement = test_accuracy - default_accuracy
    print(f"Improvement: {improvement:.6f} ({improvement*100:.2f}%)")

# Main
if __name__ == "__main__":
    best_params = optimize_svm()
    evaluate_model(best_params)
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE! (Support Vector Machine)")
    print("="*50)
