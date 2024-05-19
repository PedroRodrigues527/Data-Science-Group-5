import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('datasets/apple_quality_labels.csv')
    X = df.drop('Quality', axis=1).values
    y = df['Quality'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Test different values of k
    k_values = list(range(1, 50))
    cv_scores = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    # Determine the best k
    best_k = k_values[np.argmax(cv_scores)]
    print(f"Best k: {best_k}")

    # Plot the cross-validated accuracy for each k
    plt.plot(k_values, cv_scores)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

    # Train the final model with the best k
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = best_knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("Confusion Matrix:")
    print(cm)

    # Compute ROC-AUC score and plot ROC curve
    if len(np.unique(y)) == 2:  # Check if binary classification
        y_prob = best_knn.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC Score: {roc_auc:.2f}")

        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(false_positive_rate, true_positive_rate, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('RoC Curve')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == "__main__":
    main()
