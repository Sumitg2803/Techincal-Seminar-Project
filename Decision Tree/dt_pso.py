import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mealpy.swarm_based import PSO
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
    max_depth = int(solution[0])
    min_samples_split = int(solution[1])
    min_samples_leaf = int(solution[2])
    max_features = solution[3]

    try:
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        fitness = -np.mean(scores)  # minimize negative accuracy
    except:
        fitness = 1.0
    return fitness

# -----------------------------
# Optimization Function (PSO)
# -----------------------------
def optimize_decision_tree():
    global X_train, X_test, y_train, y_test
    # Increased noise rates for more variation
    X_train, X_test, y_train, y_test = load_data(
        label_noise_rate=0.08, 
        feature_noise_rate=0.05, 
        random_state=42
    )

    print("Starting Decision Tree optimization with PSO on Mushroom dataset...")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Train/Test split: 30%/70%")
    print("-" * 50)

    problem = {
        "bounds": [
            FloatVar(lb=3, ub=20, name="max_depth"),              # reduced max depth
            FloatVar(lb=2, ub=15, name="min_samples_split"),      
            FloatVar(lb=1, ub=8, name="min_samples_leaf"),        
            FloatVar(lb=0.3, ub=0.9, name="max_features")         # limited feature range
        ],
        "minmax": "min",
        "obj_func": objective_function
    }

    optimizer = PSO.OriginalPSO(epoch=5, pop_size=20)
    best_agent = optimizer.solve(problem)

    best_position = best_agent.solution
    best_fitness = best_agent.target.fitness

    print("\nOptimization Results:")
    print("-" * 50)
    print(f"Best fitness (negative accuracy): {best_fitness:.6f}")
    print(f"Best accuracy (CV estimate): {-best_fitness:.6f}")

    best_params = {
        'max_depth': int(best_position[0]),
        'min_samples_split': int(best_position[1]),
        'min_samples_leaf': int(best_position[2]),
        'max_features': best_position[3],
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
    print("FINAL MODEL EVALUATION (Decision Tree)")
    print("="*50)

    best_dt = DecisionTreeClassifier(**best_params)
    best_dt.fit(X_train, y_train)

    y_pred = best_dt.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_accuracy:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Edible', 'Poisonous']))

    print("\n" + "-"*50)
    print("Comparison with default Decision Tree:")
    print("-" * 50)
    default_dt = DecisionTreeClassifier(random_state=42)
    default_dt.fit(X_train, y_train)
    default_pred = default_dt.predict(X_test)
    default_accuracy = accuracy_score(y_test, default_pred)

    print(f"Default DT Accuracy: {default_accuracy:.6f}")
    print(f"Optimized DT Accuracy: {test_accuracy:.6f}")
    improvement = test_accuracy - default_accuracy
    print(f"Improvement: {improvement:.6f} ({improvement*100:.2f}%)")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    best_params = optimize_decision_tree()
    evaluate_model(best_params)
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE! (Decision Tree + PSO)")
    print("="*50)