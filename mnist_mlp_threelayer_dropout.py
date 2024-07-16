# Multilayer Perceptron (MLP) for MNIST dataset with random dropout
# 1.46M parameters with random initialization
# 3 hidden layer: 1024 + 512 + 256 units
# EarlyStopping(patience=5)
# loss: 0.0449 - accuracy: 0.9860 - val_loss: 0.0622 - val_accuracy: 0.9828 - 2s/epoch - 5ms/step
# batch_size=128, epochs=50
# best result obtained at epoch 12

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

def load_data():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]).astype('float32') / 255 
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]).astype('float32') / 255
    Y_train = to_categorical(Y_train)                       # one-hot encoding
    Y_test = to_categorical(Y_test)                         # one-hot encoding
    return (X_train, Y_train), (X_test, Y_test)

def create_model():
    # input & hidden layer 1
    model = Sequential()
    model.add(Dense(units=1024,         # hidden layer with 1024 units
                    input_dim=784,      # image size 28 * 28 = 784
                    kernel_initializer='normal',
                    activation='relu'))
    model.add(Dropout(0.4))             # 40% random dropout

    # hidden layer 2
    model.add(Dense(units=512,          # hidden layer with 512 units
                    kernel_initializer='normal',
                    activation='relu'))
    model.add(Dropout(0.4))             # 40% random dropout

    # hidden layer 3
    model.add(Dense(units=256,          # hidden layer with 256 units
                    kernel_initializer='normal',
                    activation='relu'))
    model.add(Dropout(0.4))             # 40% random dropout

    # output layer
    model.add(Dense(units=10,           # output layer with 10 units (0-9)
                    kernel_initializer='normal',
                    activation='softmax'))
    print(model.summary())  # summary of model
    plot_model(model, to_file='network/mmtd.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',         # gradien descent
                  metrics=['accuracy'])
    return model

def train(model, batch_size, epochs, X_train, Y_train, X_test, Y_test):
    checkpoint = ModelCheckpoint('bestModel/mmtd.keras',
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=True,
                                 verbose=1)
    early_stopping = EarlyStopping(patience=5)
    train_history = model.fit(x=X_train,
                              y=Y_train,
                              validation_data=(X_test, Y_test),
                              epochs=epochs,
                              batch_size=batch_size,
                              callbacks=[early_stopping, checkpoint],
                              verbose=2)
    return train_history

# evaluate model
def evaluate(model, X_test, Y_test):
    scores = model.evaluate(X_test, Y_test, verbose=0)
    return scores

def result_plt(hist):
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.figure(figsize=(9, 6))
    x = np.arange(len(train_loss))

    plt.subplot(1, 2, 1)
    plt.plot(x, train_acc)
    plt.plot(val_acc)
    plt.title("Train History of accuracy")
    plt.ylabel('accuracy')
    plt.xlabel('epoche')
    plt.legend(['train_acc', 'val_acc'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title("Train History of loss")
    plt.ylabel('loss')
    plt.xlabel('epoche')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    acc_ax.plot(train_acc, 'b', label='train_acc')
    acc_ax.plot(val_acc, 'g', label='val_acc')
    loss_ax.plot(train_loss, 'y', label='train_loss')
    loss_ax.plot(val_loss, 'r', label='val_loss')

    loss_ax.legend(loc='lower left')
    acc_ax.legend(loc='upper left')

    plt.show()


def main():
    (X_train, Y_train), (X_test, Y_test) = load_data()
    model = create_model()

    hist = train(model, 128, 50, X_train, Y_train, X_test, Y_test)
    result_plt(hist)

    model.load_weights("bestModel/mmtd.keras")
    print("\nsaved model to disk")
    print("accuracy:", evaluate(model, X_test, Y_test)[1])

    # use model to predict
    index_list = np.random.choice(X_test.shape[0], 10)
    data = X_test[index_list]
    y_preds = model.predict(data)
    print("\npredicts===>>>")
    for i in range(10):
        print('True:' + str(np.argmax(Y_test[index_list[i]])) + 
              ', Predict:' + str(np.argmax(y_preds[i])) +
              ', index:' + str(index_list[i]))

if __name__ == "__main__":
    main()

