# Lib imports
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from os import path

# Module imports
from ANN import build, train

model_path = './models/q.h5'


def trainNN(X, y):
    # Split training set into training set and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=0.3,
                                                      shuffle=False,
                                                      random_state=42)

    # If model doesn't exist send data to compile NN
    if not path.exists(model_path):
        model = build(X_train)
        model.save(model_path)
        # Print model summary
        model.summary()
        print('MODEL COMPILED AND SAVED!')

    # Send training data with labels to NN
    model, history = train(X_train, y_train, X_val, y_val)

    model.save(model_path)
    print('MODEL TRAINED AND SAVED!')

    # Plot training & validation loss values
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation accuracy values
    plt.figure(1)
    plt.subplot(2, 1, 2)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
