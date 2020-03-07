# Lib imports
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Dense, BatchNormalization,
                                     Dropout)

model_path = './models/q.h5'
best_path = './bestmodels/best.h5'

# Compile CNN function


def build(X_train):
    n = X_train.shape[1]
    input = Input(shape=(n,))
    hidden = Dense(n, activation='relu')(input)
    hidden = Dense(n / 2, activation='relu')(hidden)
    hidden = Dense(n / 4, activation='relu')(hidden)
    output = Dense(1, activation='sigmoid')(hidden)

    # Create model
    classifier = Model(inputs=input, outputs=output)

    # Compile model
    classifier.compile(optimizer='adam', loss='binary_crossentropy',
                       metrics=['accuracy'])

    # Return model
    return classifier


# Train classifier function
def train(X_train, y_train, X_val, y_val):
    # Load model
    classifier = load_model(model_path)

    # Train model
    n_epochs = 300
    b_size = 512
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(best_path, monitor='val_accuracy',
                         mode='max', verbose=1, save_best_only=True)
    callbacks = [es, mc]
    history = classifier.fit(X_train, y_train, batch_size=b_size,
                             epochs=n_epochs, verbose=1,
                             validation_data=(X_val, y_val),
                             callbacks=callbacks).history

    return classifier, history


# Evaluate function
def evaluate(data, labels):
    # Load CNN model
    classifier = load_model(best_path)
    classifier.summary()

    evaluation = classifier.evaluate(data, labels, batch_size=256)

    return evaluation


# Predict winners
def predict(data):
    # Load CNN
    classifier = load_model(best_path)
    classifier.summary()

    predictions = classifier.predict(data, batch_size=1024)

    return predictions
