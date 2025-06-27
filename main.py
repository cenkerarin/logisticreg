import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initialize the Logistic Regression model

        Parameters:
        learning_rate: float, step size for gradient descent
        max_iterations: int, maximum number of iterations for training
        tolerance: float, convergence tolerance for cost function, to stop the training
        """

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []

    def sigmoid(self, z):
        """
        Sigmoid activation function
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the logistic regression model

        Parameters:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)

            # Compute cost (log-likelihood)
            cost = self.compute_cost(y, predictions)
            self.cost_history.append(cost)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i + 1} iterations")
                break

    def compute_cost(self, y_true, y_pred):
        """
        Compute the logistic regression cost function
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost

    def predict(self, X):
        """
        Make predictions on new data
        """
        linear_pred = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_pred)
        return (predictions >= 0.5).astype(int)

    def predict_proba(self, X):
        """
        Predict class probabilities
        """
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)


def generate_sample_data(n_samples=1000, n_features=2, random_state=42):
    """
    Generate sample data for binary classification
    """
    np.random.seed(random_state)

    # Generate random data
    X = np.random.randn(n_samples, n_features)

    # Create a linear decision boundary
    true_weights = np.array([1.5, -2.0])
    true_bias = 0.5

    # Generate labels based on linear combination + noise
    linear_combination = np.dot(X, true_weights) + true_bias
    probabilities = 1 / (1 + np.exp(-linear_combination))
    y = np.random.binomial(1, probabilities)

    return X, y


def evaluate_model(model, X, y):
    """
    Evaluate the model performance
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Calculate metrics
    accuracy = np.mean(predictions == y)

    # True positives, false positives, etc.
    tp = np.sum((predictions == 1) & (y == 1))
    fp = np.sum((predictions == 1) & (y == 0))
    tn = np.sum((predictions == 0) & (y == 0))
    fn = np.sum((predictions == 0) & (y == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': np.array([[tn, fp], [fn, tp]])
    }


def plot_results(X, y, model, metrics):
    """
    Visualize the results
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Data points and decision boundary
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    ax1.set_title('Data Points and Decision Boundary')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')

    # Create decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)

    contour = ax1.contour(xx, yy, Z, levels=[0.5], colors='red', linestyles='--', linewidths=2)
    ax1.clabel(contour, inline=True, fontsize=10)
    plt.colorbar(scatter, ax=ax1)

    # Plot 2: Cost function over iterations
    ax2.plot(model.cost_history)
    ax2.set_title('Cost Function Over Iterations')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.grid(True)

    # Plot 3: Confusion Matrix
    cm = metrics['confusion_matrix']
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.figure.colorbar(im, ax=ax3)
    ax3.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'],
            title='Confusion Matrix',
            ylabel='True label',
            xlabel='Predicted label')

    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    # Plot 4: Model parameters and metrics
    ax4.axis('off')

    # Display metrics
    metrics_text = f"""Model Performance Metrics:

    Accuracy: {metrics['accuracy']:.4f}
    Precision: {metrics['precision']:.4f}
    Recall: {metrics['recall']:.4f}
    F1-Score: {metrics['f1_score']:.4f}

    Model Parameters:

    Weight 1: {model.weights[0]:.4f}
    Weight 2: {model.weights[1]:.4f}
    Bias: {model.bias:.4f}

    Training Info:

    Learning Rate: {model.learning_rate}
    Total Iterations: {len(model.cost_history)}
    Final Cost: {model.cost_history[-1]:.6f}
    """

    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the logistic regression demo
    """
    # Generate sample data
    X, y = generate_sample_data(n_samples=1000, n_features=2)

    # Split data into training and testing sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Create and train the model
    model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)

    print("\nTraining Set Performance:")
    print(f"   Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"   Precision: {train_metrics['precision']:.4f}")
    print(f"   Recall: {train_metrics['recall']:.4f}")
    print(f"   F1-Score: {train_metrics['f1_score']:.4f}")

    print("\nTest Set Performance:")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall: {test_metrics['recall']:.4f}")
    print(f"   F1-Score: {test_metrics['f1_score']:.4f}")

    plot_results(X_test, y_test, model, test_metrics)

if __name__ == "__main__":
    main()

