import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mealpy.swarm_based import WOA
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
    # Map solution vector to KNN hyperparameters
    n_neighbors = int(round(solution[0]))
    leaf_size = int(round(solution[1]))
    p_val = int(round(solution[2]))    # 1 = manhattan, 2 = euclidean
    weights_idx = int(round(solution[3]))  # 0 -> 'uniform', 1 -> 'distance'

    # clamp values to valid ranges
    n_neighbors = max(1, min(50, n_neighbors))
    leaf_size = max(1, min(100, leaf_size))
    p_val = 1 if p_val < 1.5 else 2
    weights_idx = 0 if weights_idx < 0.5 else 1
    weights = ['uniform', 'distance'][weights_idx]

    try:
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            leaf_size=leaf_size,
            p=p_val,
            weights=weights,
            n_jobs=-1
        )
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        fitness = -np.mean(scores)  # minimize negative accuracy
    except:
        fitness = 1.0
    return fitness

# -----------------------------
# Optimization Function (WOA)
# -----------------------------
def optimize_decision_tree():
    global X_train, X_test, y_train, y_test
    # Increased noise rates for more variation
    X_train, X_test, y_train, y_test = load_data(
        label_noise_rate=0.08, 
        feature_noise_rate=0.05, 
        random_state=42
    )

    print("Starting K-Nearest Neighbors optimization with WOA on Mushroom dataset...")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Train/Test split: 30%/70%")
    print("-" * 50)

    problem = {
        "bounds": [
            FloatVar(lb=1, ub=50, name="n_neighbors"),    # number of neighbors
            FloatVar(lb=1, ub=100, name="leaf_size"),     # leaf size for KD/ Ball tree
            FloatVar(lb=1, ub=2, name="p"),               # 1 or 2 (Manhattan / Euclidean)
            FloatVar(lb=0, ub=1, name="weights_idx")      # 0 -> uniform, 1 -> distance
        ],
        "minmax": "min",
        "obj_func": objective_function
    }

    optimizer = WOA.OriginalWOA(epoch=5, pop_size=20)
    best_agent = optimizer.solve(problem)

    best_position = best_agent.solution
    best_fitness = best_agent.target.fitness

    print("\nOptimization Results:")
    print("-" * 50)
    print(f"Best fitness (negative accuracy): {best_fitness:.6f}")
    print(f"Best accuracy (CV estimate): {-best_fitness:.6f}")

    # Convert best_position to readable hyperparameters
    best_n_neighbors = int(round(best_position[0]))
    best_leaf_size = int(round(best_position[1]))
    best_p = int(round(best_position[2]))
    best_weights_idx = int(round(best_position[3]))
    best_n_neighbors = max(1, min(50, best_n_neighbors))
    best_leaf_size = max(1, min(100, best_leaf_size))
    best_p = 1 if best_p < 1.5 else 2
    best_weights = ['uniform', 'distance'][0 if best_weights_idx < 0.5 else 1]

    best_params = {
        'n_neighbors': best_n_neighbors,
        'leaf_size': best_leaf_size,
        'p': best_p,
        'weights': best_weights,
        'n_jobs': -1
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

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    best_params = optimize_decision_tree()
    evaluate_model(best_params)
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE! (K-Nearest Neighbors + WOA)")
    print("="*50)
