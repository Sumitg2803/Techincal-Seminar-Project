import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression  # Logistic Regression Classifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mealpy.evolutionary_based import GA  # ✅ GA Optimizer
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

    # Feature scaling for Logistic Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# Objective function
def objective_function(solution):
    global X_train, y_train
    C = 10 ** solution[0]  # Regularization strength
    solver_idx = int(solution[1])
    solvers = ['lbfgs', 'liblinear', 'saga', 'newton-cg']
    solver = solvers[solver_idx % len(solvers)]

    try:
        lr = LogisticRegression(C=C, solver=solver, max_iter=500, random_state=42)
        scores = cross_val_score(lr, X_train, y_train, cv=3, scoring='accuracy')
        fitness = -np.mean(scores)  # minimize negative accuracy
    except:
        fitness = 1.0
    return fitness

# GA Optimization
def optimize_logistic_regression():
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = load_data()

    print("Starting Logistic Regression optimization with GA on Mushroom dataset...")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print("-" * 50)

    problem = {
        "bounds": [
            FloatVar(lb=-2, ub=3, name="C (log10 scale)"),   # 10^-2 to 10^3
            FloatVar(lb=0, ub=3, name="solver_choice")       # index 0 to 3
        ],
        "minmax": "min",
        "obj_func": objective_function
    }

    optimizer = GA.OriginalGA(epoch=5, pop_size=20)  # ✅ GA Optimizer
    best_agent = optimizer.solve(problem)

    best_position = best_agent.solution
    best_fitness = best_agent.target.fitness

    print("\nOptimization Results:")
    print("-" * 50)
    print(f"Best fitness (negative accuracy): {best_fitness:.6f}")
    print(f"Best accuracy: {-best_fitness:.6f}")

    solvers = ['lbfgs', 'liblinear', 'saga', 'newton-cg']
    best_params = {
        'C': 10 ** best_position[0],
        'solver': solvers[int(best_position[1]) % len(solvers)],
        'max_iter': 500,
        'random_state': 42
    }

    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    return best_params

# Evaluate Model
def evaluate_model(best_params):
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION (Logistic Regression)")
    print("="*50)

    best_lr = LogisticRegression(**best_params)
    best_lr.fit(X_train, y_train)

    y_pred = best_lr.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {test_accuracy:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Edible', 'Poisonous']))

    print("\n" + "-"*50)
    print("Comparison with default Logistic Regression:")
    print("-" * 50)
    default_lr = LogisticRegression(max_iter=500, random_state=42)
    default_lr.fit(X_train, y_train)
    default_pred = default_lr.predict(X_test)
    default_accuracy = accuracy_score(y_test, default_pred)

    print(f"Default LR Accuracy: {default_accuracy:.6f}")
    print(f"Optimized LR Accuracy: {test_accuracy:.6f}")
    improvement = test_accuracy - default_accuracy
    print(f"Improvement: {improvement:.6f} ({improvement*100:.2f}%)")

# Main
if __name__ == "__main__":
    best_params = optimize_logistic_regression()
    evaluate_model(best_params)
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE! (Logistic Regression + GA)")
    print("="*50)
