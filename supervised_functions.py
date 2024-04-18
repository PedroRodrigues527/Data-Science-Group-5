def corr_quality(df):
    # Calculate correlation matrix
    correlation_matrix = apples.corr()

    # Assuming 'Quality' is the target variable
    target_correlation = correlation_matrix['Quality'].drop('Quality')  # Remove correlation with itself

    # Sort features by correlation with target variable
    sorted_features = target_correlation.abs().sort_values(ascending=False)

    print("Correlation with target variable:")
    print(sorted_features)

def split_train_test(df):
    best_features = ['Ripeness', 'Juiciness', 'Sweetness', 'Size', 'Crunchiness', 'Acidity', 'Weight']
    worst_festures = ['Crunchiness', 'Acidity', 'Weight']

    X = apples[best_features]
    y = apples['Quality']

    return train_test_split(X, y, test_size=0.20, random_state=42)

def train_eval_models(X_train, X_test, y_train, y_test):
    models = {
    "XGBoost": XGBClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(),
    "Gradient Boosting": GradientBoostingClassifier()
    }
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {name}: {accuracy}")
        
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred))
        
        print(f"Confusion Matrix for {name}:")
        print(confusion_matrix(y_test, y_pred))

        dump(model, f"{name}.joblib")