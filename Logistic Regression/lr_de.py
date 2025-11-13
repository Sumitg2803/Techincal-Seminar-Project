import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mealpy.evolutionary_based import DE
from mealpy import FloatVar
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Load Mushroom Dataset
# -----------------------------
def load_data(label_noise_rate=0.08, feature_noise_rate=0.05, random_state=42):
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

    # Keep odor but remove some other highly predictive features
    data = data.drop(['gill-size', 'spore-print-color'], axis=1)

    X = data.drop('class', axis=1)
    y = data['class'].map({'e': 0, 'p': 1}).astype(int)

    # Label encode all categorical features
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Split with 30% train and 70% test
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.7, random_state=random_state, stratify=y
    )

    # Inject label noise into training labels (flip labels)
    if label_noise_rate > 0:
        rng = np.random.RandomState(random_state)
        n_flip = max(1, int(label_noise_rate * len(y_train)))
        flip_idx = rng.choice(len(y_train), size=n_flip, replace=False)
        y_train_noisy = y_train.copy()
        y_train_noisy[flip_idx] = 1 - y_train_noisy[flip_idx]
        y_train = y_train_noisy

    # Add feature noise to training data
    if feature_noise_rate > 0:
        rng = np.random.RandomState(random_state + 1)
        noise = rng.normal(0, feature_noise_rate, X_train.shape)
        X_train = X_train + noise

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# -----------------------------
# Objective Function
# -----------------------------
def objective_function(solution):
    global X_train, y_train
    # Map solution vector to LogisticRegression hyperparameters
    C = float(solution[0])                      # inverse regularization strength
    penalty_idx = int(round(solution[1]))       # 0 -> 'l2', 1 -> 'l1'
    max_iter = int(round(solution[2]))
    tol = float(solution[3])

    # clamp / sanitize values
    C = max(1e-4, min(1e4, C))
    penalty_idx = 0 if penalty_idx < 0.5 else 1
    max_iter = max(50, min(2000, max_iter))
    tol = max(1e-8, min(1e-1, tol))

    penalty = ['l2', 'l1'][penalty_idx]
    # choose solver compatible with penalty
    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'

    try:
        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            random_state=42
        )
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        fitness = -np.mean(scores)  # minimize negative accuracy
    except:
        fitness = 1.0
    return fitness

# -----------------------------
# Optimization Function (DE)
# -----------------------------
def optimize_decision_tree():
    global X_train, X_test, y_train, y_test
    # Increased noise rates for more variation
    X_train, X_test, y_train, y_test = load_data(
        label_noise_rate=0.08, 
        feature_noise_rate=0.05, 
        random_state=42
    )

    print("Starting Logistic Regression optimization with DE on Mushroom dataset...")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Train/Test split: 30%/70%")
    print("-" * 50)

    problem = {
        "bounds": [
            FloatVar(lb=1e-4, ub=1e3, name="C"),          # inverse regularization
            FloatVar(lb=0, ub=1, name="penalty_idx"),     # 0 -> l2, 1 -> l1
            FloatVar(lb=50, ub=2000, name="max_iter"),    # iterations
            FloatVar(lb=1e-8, ub=1e-2, name="tol")        # tolerance
        ],
        "minmax": "min",
        "obj_func": objective_function
    }

    optimizer = DE.OriginalDE(epoch=5, pop_size=20)
    best_agent = optimizer.solve(problem)

    best_position = best_agent.solution
    best_fitness = best_agent.target.fitness

    print("\nOptimization Results:")
    print("-" * 50)
    print(f"Best fitness (negative accuracy): {best_fitness:.6f}")
    print(f"Best accuracy (CV estimate): {-best_fitness:.6f}")

    # Convert best_position to readable hyperparameters
    best_C = float(best_position[0])
    best_penalty_idx = int(round(best_position[1]))
    best_max_iter = int(round(best_position[2]))
    best_tol = float(best_position[3])

    best_penalty_idx = 0 if best_penalty_idx < 0.5 else 1
    best_penalty = ['l2', 'l1'][best_penalty_idx]
    best_solver = 'liblinear' if best_penalty == 'l1' else 'lbfgs'

    best_params = {
        'C': best_C,
        'penalty': best_penalty,
        'solver': best_solver,
        'max_iter': best_max_iter,
        'tol': best_tol,
        'random_state': 42
    }

    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    return best_params

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(best_params):
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION (Logistic Regression)")
    print("="*50)

    best_clf = LogisticRegression(**best_params)
    best_clf.fit(X_train, y_train)

    y_pred = best_clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_accuracy:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Edible', 'Poisonous']))

    print("\n" + "-"*50)
    print("Comparison with default Logistic Regression:")
    print("-" * 50)
    default_clf = LogisticRegression(random_state=42, max_iter=200)
    default_clf.fit(X_train, y_train)
    default_pred = default_clf.predict(X_test)
    default_accuracy = accuracy_score(y_test, default_pred)

    print(f"Default LR Accuracy: {default_accuracy:.6f}")
    print(f"Optimized LR Accuracy: {test_accuracy:.6f}")
    improvement = test_accuracy - default_accuracy
    print(f"Improvement: {improvement:.6f} ({improvement*100:.2f}%)")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    best_params = optimize_decision_tree()
    evaluate_model(best_params)
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE! (Logistic Regression + DE)")
    print("="*50)
