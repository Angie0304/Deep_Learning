# advanced_cnn_fashion_mnist.py

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

# =========================
# 1. Cargar y preparar datos
# =========================
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Añadir canal
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return (x_train, y_train), (x_test, y_test)

# =========================
# 2. Data Augmentation
# =========================
def get_augmentation():
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

# =========================
# 3. Bloque residual (ResNet)
# =========================
def residual_block(x, filters, downsample=False):
    shortcut = x

    strides = 2 if downsample else 1

    x = layers.Conv2D(filters, (3,3), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Ajustar dimensiones del shortcut si cambia tamaño
    if downsample:
        shortcut = layers.Conv2D(filters, (1,1), strides=2, padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x

# =========================
# 4. Construir modelo
# =========================
def build_model():
    inputs = layers.Input(shape=(28,28,1))

    aug = get_augmentation()(inputs)

    x = layers.Conv2D(32, (3,3), padding='same')(aug)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Bloques residuales
    x = residual_block(x, 32)
    x = residual_block(x, 64, downsample=True)
    x = residual_block(x, 64)
    x = residual_block(x, 128, downsample=True)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# =========================
# 5. Scheduler (Cosine Decay)
# =========================
def get_optimizer():
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000
    )
    return tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# =========================
# 6. Entrenamiento
# =========================
def train(model, x_train, y_train, x_test, y_test):

    model.compile(
        optimizer=get_optimizer(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    cb = [
        callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
        callbacks.ReduceLROnPlateau(patience=3)
    ]

    history = model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=cb
    )

    return history

# =========================
# 7. Evaluación
# =========================
def evaluate(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test)
    print(f"\nTest accuracy: {acc:.4f}")

# =========================
# 8. Gráficas
# =========================
def plot_history(history):
    plt.figure()

    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training History')

    plt.show()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    model.summary()

    history = train(model, x_train, y_train, x_test, y_test)

    evaluate(model, x_test, y_test)
    plot_history(history)
