import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from mealpy import PSO
import warnings
warnings.filterwarnings('ignore')

# Load and prepare the Iris dataset
def load_data():
    """Load and split the Iris dataset"""
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# Define the objective function for optimization
def objective_function(solution):
    """
    Objective function to minimize (we'll minimize negative accuracy)
    
    Parameters in solution:
    - n_estimators: [10, 200] -> int
    - max_depth: [1, 20] -> int  
    - min_samples_split: [2, 20] -> int
    - min_samples_leaf: [1, 10] -> int
    - max_features: [0.1, 1.0] -> float
    """
    global X_train, y_train
    
    # Extract hyperparameters from solution
    n_estimators = int(solution[0])
    max_depth = int(solution[1]) if solution[1] > 0 else None
    min_samples_split = int(solution[2])
    min_samples_leaf = int(solution[3])
    max_features = solution[4]
    
    try:
        # Create Random Forest with current hyperparameters
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        
        # Use cross-validation to evaluate performance
        scores = cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy')
        fitness = -np.mean(scores)  # Negative because we want to minimize
        
    except Exception as e:
        # Return a high penalty for invalid hyperparameter combinations
        fitness = 1.0
    
    return fitness

def optimize_random_forest():
    """Main optimization function using mealpy"""
    
    # Load data
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = load_data()
    
    print("Starting Random Forest optimization with mealpy...")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print("-" * 50)
    
    # Define the problem bounds
    problem = {
        "bounds": [
            FloatVar(lb=10, ub=200, name="n_estimators"),      # Number of trees
            FloatVar(lb=1, ub=20, name="max_depth"),           # Maximum depth  
            FloatVar(lb=2, ub=20, name="min_samples_split"),   # Min samples to split
            FloatVar(lb=1, ub=10, name="min_samples_leaf"),    # Min samples in leaf
            FloatVar(lb=0.1, ub=1.0, name="max_features")     # Max features ratio
        ],
        "minmax": "min",  # We want to minimize the objective function
        "obj_func": objective_function
    }
    
    # Initialize PSO optimizer
    optimizer = PSO.OriginalPSO(epoch=30, pop_size=20)
    
    # Solve the optimization problem
    best_position, best_fitness = optimizer.solve(problem)
    
    print("\nOptimization Results:")
    print("-" * 50)
    print(f"Best fitness (negative accuracy): {best_fitness:.6f}")
    print(f"Best accuracy: {-best_fitness:.6f}")
    
    # Extract best hyperparameters
    best_params = {
        'n_estimators': int(best_position[0]),
        'max_depth': int(best_position[1]) if best_position[1] > 0 else None,
        'min_samples_split': int(best_position[2]),
        'min_samples_leaf': int(best_position[3]),
        'max_features': best_position[4],
        'random_state': 42
    }
    
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_params

def evaluate_model(best_params):
    """Evaluate the optimized model on test data"""
    
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION")
    print("="*50)
    
    # Train model with best parameters
    best_rf = RandomForestClassifier(**best_params)
    best_rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = best_rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {test_accuracy:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Setosa', 'Versicolor', 'Virginica']))
    
    # Compare with default Random Forest
    print("\nComparison with default Random Forest:")
    default_rf = RandomForestClassifier(random_state=42)
    default_rf.fit(X_train, y_train)
    default_pred = default_rf.predict(X_test)
    default_accuracy = accuracy_score(y_test, default_pred)
    
    print(f"Default RF Accuracy: {default_accuracy:.6f}")
    print(f"Optimized RF Accuracy: {test_accuracy:.6f}")
    print(f"Improvement: {test_accuracy - default_accuracy:.6f}")
    
    # Feature importance
    print(f"\nFeature Importance (Optimized Model):")
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    for i, importance in enumerate(best_rf.feature_importances_):
        print(f"  {feature_names[i]}: {importance:.4f}")

if _name_ == "_main_":
    # Run the optimization
    best_params = optimize_random_forest()
    
    # Evaluate the final model
    evaluate_model(best_params)
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE!")
    print("="*50)