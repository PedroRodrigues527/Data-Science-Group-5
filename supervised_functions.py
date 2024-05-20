from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt


def corr_quality(df):
    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Assuming 'Quality' is the target variable
    target_correlation = correlation_matrix['Quality'].drop('Quality')  # Remove correlation with itself

    # Sort features by correlation with target variable
    sorted_features = target_correlation.abs().sort_values(ascending=False)

    print("Correlation with target variable:")
    print(sorted_features)


def split_train_test(df):
    best_features = ['Ripeness', 'Juiciness', 'Sweetness', 'Size', 'Crunchiness', 'Acidity', 'Weight']
    worst_festures = ['Crunchiness', 'Acidity', 'Weight']

    X = df[best_features]
    y = df['Quality']

    return train_test_split(X, y, test_size=0.20, random_state=42)


def train_eval_models(X_train, X_test, y_train, y_test):
    models = {

        "XGBoost": XGBClassifier(n_estimators=25, learning_rate=0.001, max_depth=10),
        "SVM": SVC(C=15.0, kernel='rbf', gamma='scale'),
        "Logistic Regression": LogisticRegression(penalty='l2', C=1.0, solver='lbfgs'),
        "Naive Bayes": GaussianNB(var_smoothing=1e-5),
        "Decision Trees": DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=5),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.001, algorithm='SAMME.R'),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {name}: {accuracy}")

        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)

        # Plotting confusion matrix with Seaborn
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()