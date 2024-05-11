"""
Random Forest Classifier Bagging Algorithm
Requirements: sudo apt-get install graphviz
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from plots import plot_confusion_matrix_general
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from scipy.stats import randint

def trainModel(X_train, y_train, withHyperParameters=False):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    if(withHyperParameters):
        param_dist = {
            'n_estimators': randint(50,500),
            'max_depth'   : randint(1,20)
        }
        rand_search = RandomizedSearchCV(rf, param_dist, n_iter=5, cv=5)
        rand_search.fit(X_train, y_train)
        return rand_search
    rf.fit(X_train, y_train)
    return rf

def predictModel(model, X_test):
    return model.predict(X_test)

def evaluateModel(y_test, y_pred):
    return accuracy_score(y_test, y_pred)

def displayMetrics(y_test, y_pred, accuracy):
    print("Accuracy: ", accuracy)
    print("\nConfusion Matrix: ",confusion_matrix(y_test, y_pred))
    plot_confusion_matrix_general(y_test, y_pred, ['1', '2'])

def display_tree(model, feature_names, class_names):
    tree = model.estimators_[0]

    dot_data = export_graphviz(tree, out_file=None, 
                               feature_names=feature_names, 
                               class_names=class_names,
                               filled=True, 
                               rounded=True,
                               max_depth=2,
                               special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data)

    Image(graph.create_png())

    graph.write_png('random_forest_tree/decision_tree.png')

def randomForest(hyperparameters=False):
    apples = pd.read_csv('datasets/apple_quality.csv')

    label_encoder = LabelEncoder()
    apples['Quality'] = label_encoder.fit_transform(apples['Quality'])

    X = apples.drop(columns=['Quality'])
    y = apples['Quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = trainModel(X_train, y_train, hyperparameters)
    if(hyperparameters):
        best_rf = model.best_estimator_
        print("Best Parameters: ", model.best_params_)
        print("best_rf: ", best_rf)
        model = best_rf

    y_pred = predictModel(model, X_test)
    accuracy = evaluateModel(y_test, y_pred)
    displayMetrics(y_test, y_pred, accuracy)
    display_tree(model, X_train.columns, label_encoder.classes_)

if __name__ == "__main__":
    randomForest(True)