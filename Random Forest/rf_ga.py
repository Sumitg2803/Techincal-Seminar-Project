import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from mealpy.evolutionary_based import GA
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

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.7, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# Objective function
def objective_function(solution):
    global X_train, y_train
    n_estimators = int(solution[0])
    max_depth = int(solution[1]) if solution[1] > 0 else None
    min_samples_split = int(solution[2])
    min_samples_leaf = int(solution[3])
    max_features = solution[4]

    try:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        scores = cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy')
        fitness = -np.mean(scores)  # GA minimizes
    except:
        fitness = 1.0
    return fitness

# GA optimization
def optimize_random_forest():
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = load_data()

    print("Starting Random Forest optimization with GA on Mushroom dataset...")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print("-" * 50)

    problem = {
        "bounds": [
            FloatVar(lb=10, ub=200, name="n_estimators"),
            FloatVar(lb=1, ub=30, name="max_depth"),
            FloatVar(lb=2, ub=20, name="min_samples_split"),
            FloatVar(lb=1, ub=10, name="min_samples_leaf"),
            FloatVar(lb=0.1, ub=1.0, name="max_features")
        ],
        "minmax": "min",
        "obj_func": objective_function
    }

    optimizer = GA.OriginalGA(epoch=5, pop_size=70)
    best_agent = optimizer.solve(problem)

    # Access fitness and solution
    best_position = best_agent.solution
    best_fitness = best_agent.target.fitness

    print("\nOptimization Results:")
    print("-" * 50)
    print(f"Best fitness (negative accuracy): {best_fitness:.6f}")
    print(f"Best accuracy: {-best_fitness:.6f}")

    best_params = {
        'n_estimators': int(best_position[0]),
        'max_depth': int(best_position[1]) if best_position[1] > 0 else None,
        'min_samples_split': int(best_position[2]),
        'min_samples_leaf': int(best_position[3]),
        'max_features': best_position[4],
        'random_state': 42,
        'n_jobs': -1
    }

    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    return best_params

# Evaluate model
def evaluate_model(best_params):
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION")
    print("="*50)

    best_rf = RandomForestClassifier(**best_params)
    best_rf.fit(X_train, y_train)

    y_pred = best_rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {test_accuracy:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Edible', 'Poisonous']))

    print("\n" + "-"*50)
    print("Comparison with default Random Forest:")
    print("-"*50)
    default_rf = RandomForestClassifier(random_state=42)
    default_rf.fit(X_train, y_train)
    default_pred = default_rf.predict(X_test)
    default_accuracy = accuracy_score(y_test, default_pred)

    print(f"Default RF Accuracy: {default_accuracy:.6f}")
    print(f"Optimized RF Accuracy: {test_accuracy:.6f}")
    improvement = test_accuracy - default_accuracy
    print(f"Improvement: {improvement:.6f} ({improvement*100:.2f}%)")

    print("\n" + "-"*50)
    print("Feature Importance (Optimized Model):")
    print("-" * 50)
    feature_names = [
        'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]
    
    feature_importance = list(zip(feature_names, best_rf.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Most Important Features:")
    for i, (name, importance) in enumerate(feature_importance[:10], 1):
        print(f"  {i}. {name}: {importance:.4f}")
    
    print("\nAll Features (sorted by importance):")
    for name, importance in feature_importance:
        print(f"  {name}: {importance:.4f}")

# Main
if __name__ == "__main__":
    best_params = optimize_random_forest()
    evaluate_model(best_params)
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE!")
    print("="*50)
