import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess the Fashion MNIST dataset
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to include channel dimension (grayscale = 1 channel)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

# Function to create CNN with ReLU activation and MaxPooling
def create_cnn_relu_maxpool():
    model = Sequential(name="CNN_ReLU_MaxPool")
    
    # First convolutional block
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second convolutional block
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    return model

# Function to create CNN with LeakyReLU activation and MaxPooling
def create_cnn_leakyrelu_maxpool():
    model = Sequential(name="CNN_LeakyReLU_MaxPool")
    
    # First convolutional block
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second convolutional block
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    return model

# Function to create CNN with ReLU activation and Strided Convolution
def create_cnn_relu_strided():
    model = Sequential(name="CNN_ReLU_Strided")
    
    # First convolutional block
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))  # Strided conv
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    # Second convolutional block
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))  # Strided conv
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    return model

# Function to create CNN with LeakyReLU activation and Strided Convolution
def create_cnn_leakyrelu_strided():
    model = Sequential(name="CNN_LeakyReLU_Strided")
    
    # First convolutional block
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same'))  # Strided conv
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    # Second convolutional block
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same'))  # Strided conv
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    return model

# Function to count parameters
def count_parameters(model):
    return sum([np.prod(k.get_shape().as_list()) for k in model.trainable_weights])

# Function to train and evaluate each model
def train_and_evaluate(model, optimizer, x_train, y_train, x_test, y_test, epochs=25, batch_size=128):
    if optimizer == 'adam':
        opt = Adam(learning_rate=0.001)
    else:  # SGD with momentum
        opt = SGD(learning_rate=0.01, momentum=0.9)
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{model.name} with {optimizer} optimizer: Test accuracy = {test_acc:.4f}")
    
    return history, test_acc

# Function to plot training history for model comparison
def plot_training_history(histories, names):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot accuracy
    ax1.set_title('Model Accuracy', fontsize=15)
    for i, (history, name) in enumerate(zip(histories, names)):
        ax1.plot(history.history['accuracy'], label=f'{name} (Training)')
        ax1.plot(history.history['val_accuracy'], label=f'{name} (Validation)', linestyle='--')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot loss
    ax2.set_title('Model Loss', fontsize=15)
    for i, (history, name) in enumerate(zip(histories, names)):
        ax2.plot(history.history['loss'], label=f'{name} (Training)')
        ax2.plot(history.history['val_loss'], label=f'{name} (Validation)', linestyle='--')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()

# Function to visualize test accuracy comparison
def plot_accuracy_comparison(names, accuracies):
    plt.figure(figsize=(14, 8))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#d35400']
    
    bars = plt.bar(range(len(names)), accuracies, color=colors[:len(names)])
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Test Accuracy Comparison', fontsize=15)
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f'{acc:.4f}',
            ha='center',
            fontsize=10
        )
    
    plt.ylim(0.85, 0.95)  # Adjust this range based on your results
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300)
    plt.show()

# Main execution function
def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create different model variations
    models = {
        "ReLU+MaxPool+Adam": (create_cnn_relu_maxpool(), "adam"),
        "ReLU+MaxPool+SGD": (create_cnn_relu_maxpool(), "sgd"),
        "LeakyReLU+MaxPool+Adam": (create_cnn_leakyrelu_maxpool(), "adam"),
        "LeakyReLU+MaxPool+SGD": (create_cnn_leakyrelu_maxpool(), "sgd"),
        "ReLU+Strided+Adam": (create_cnn_relu_strided(), "adam"),
        "ReLU+Strided+SGD": (create_cnn_relu_strided(), "sgd"),
        "LeakyReLU+Strided+Adam": (create_cnn_leakyrelu_strided(), "adam"),
        "LeakyReLU+Strided+SGD": (create_cnn_leakyrelu_strided(), "sgd")
    }
    
    # Print parameter counts for each model
    for name, (model, _) in models.items():
        param_count = count_parameters(model)
        print(f"Model: {name}, Parameters: {param_count:,}")
    
    # Train models and collect results
    histories = []
    model_names = []
    test_accuracies = []
    
    for name, (model, optimizer) in models.items():
        print(f"\nTraining {name}...")
        history, test_acc = train_and_evaluate(
            model, optimizer, x_train, y_train, x_test, y_test
        )
        histories.append(history)
        model_names.append(name)
        test_accuracies.append(test_acc)
    
    # Visualize results
    plot_training_history(histories, model_names)
    plot_accuracy_comparison(model_names, test_accuracies)
    
    # Print summary of results
    print("\nTest Accuracy Summary:")
    for name, acc in zip(model_names, test_accuracies):
        print(f"{name}: {acc:.4f}")
    
    print(f"\nBest model: {model_names[np.argmax(test_accuracies)]} with accuracy {max(test_accuracies):.4f}")

if __name__ == "__main__":
    main()