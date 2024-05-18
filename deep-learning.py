import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def pre_process():
    df = pd.read_csv('datasets/apple_quality_labels.csv')
    # Split the data
    X = df.drop('Quality', axis=1).values
    y = df['Quality'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).shuffle(len(X_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    return X_train, X_test, y_train, y_test, train_dataset, test_dataset


def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_dataset, test_dataset):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience=20, restore_best_weights=True)
    history = model.fit(train_dataset, epochs=200, validation_data=test_dataset, callbacks=[early_stopping])
    return history

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    return loss, accuracy

def plot_history(history):
    #Plot accuracy train vs validation
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    #Plot loss train vs validation
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def main():
    X_train, X_test, y_train, y_test, train_dataset, test_dataset = pre_process()
    model = build_model((X_train.shape[1],))
    history = train_model(model, train_dataset, test_dataset)
    loss, accuracy = evaluate_model(model, X_test, y_test)
    print("Accuracy: ", accuracy)
    print("Loss: ", loss)
    plot_history(history)


if __name__ == '__main__':
    main()
