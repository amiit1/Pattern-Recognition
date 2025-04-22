import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Fashion MNIST class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_confusion_matrix(model, x_test, y_test_original):
    """Plot confusion matrix for model evaluation"""
    # Make predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_original, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model.name}')
    plt.tight_layout()
    plt.savefig(f'{model.name}_confusion_matrix.png', dpi=300)
    plt.show()
    
    # Print classification report
    print(f"\nClassification Report - {model.name}:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

def visualize_model_filters(model):
    """Visualize the filters learned by the first convolutional layer"""
    # Get the weights of the first convolutional layer
    filters, biases = model.layers[0].get_weights()
    
    # Normalize filter values to 0-1 range
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    # Plot the filters
    n_filters = filters.shape[3]
    n_rows = n_filters // 8 + (1 if n_filters % 8 != 0 else 0)
    
    plt.figure(figsize=(16, 2 * n_rows))
    for i in range(n_filters):
        plt.subplot(n_rows, 8, i+1)
        plt.imshow(filters[:, :, 0, i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Filter {i+1}')
    
    plt.suptitle(f'First Layer Filters - {model.name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{model.name}_filters.png', dpi=300)
    plt.show()

def visualize_activation_maps(model, x_test, image_index=0):
    """Visualize activation maps for a single image"""
    # Create a model that outputs feature maps after each conv layer
    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get the image and prediction
    input_image = x_test[image_index:image_index+1]
    prediction = model.predict(input_image)
    predicted_class = np.argmax(prediction)
    
    # Get activations
    activations = activation_model.predict(input_image)
    
    # Plot the image
    plt.figure(figsize=(16, 12))
    plt.subplot(121)
    plt.imshow(x_test[image_index].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title(f'Input Image: {class_names[predicted_class]}')
    
    # Plot activations for each conv layer
    for i, activation in enumerate(activations):
        n_features = activation.shape[-1]
        size = activation.shape[1]
        
        # Create a grid to visualize feature maps
        display_grid = np.zeros((size, size * min(8, n_features)))
        
        # Fill the grid with feature maps
        for j in range(min(8, n_features)):
            display_grid[:, j*size:(j+1)*size] = activation[0, :, :, j]
            
        # Scale to 0-1
        max_val = display_grid.max() if display_grid.max() != 0 else 1
        min_val = display_grid.min()
        display_grid = (display_grid - min_val) / (max_val - min_val)
        
        # Plot the grid
        plt.subplot(len(activations) + 1, 2, i*2 + 3)
        plt.imshow(display_grid, cmap='viridis')
        plt.axis('off')
        plt.title(f'Layer {i+1} Activations')
    
    plt.suptitle(f'Activation Maps - {model.name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{model.name}_activation_maps.png', dpi=300)
    plt.show()

def compare_parameters_and_performance(model_names, param_counts, accuracies):
    """Compare model parameters vs performance"""
    plt.figure(figsize=(14, 7))
    
    # Sort by parameter count
    sorted_indices = np.argsort(param_counts)
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_params = [param_counts[i] for i in sorted_indices]
    sorted_accs = [accuracies[i] for i in sorted_indices]
    
    # Create plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Parameter counts on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Parameter Count', color=color, fontsize=12)
    bars = ax1.bar(range(len(sorted_names)), sorted_params, color=color, alpha=0.7)
    
    # Add parameter count labels
    for bar, param in zip(bars, sorted_params):
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5000,
            f'{param:,}',
            ha='center',
            fontsize=9,
            rotation=90,
            color=color
        )
    
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(len(sorted_names)))
    ax1.set_xticklabels(sorted_names, rotation=45, ha='right')
    
    # Accuracy on right y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Test Accuracy', color=color, fontsize=12)
    line = ax2.plot(range(len(sorted_names)), sorted_accs, 'o-', color=color, linewidth=2, markersize=8)
    
    # Add accuracy labels
    for i, acc in enumerate(sorted_accs):
        ax2.text(
            i,
            acc + 0.003,
            f'{acc:.4f}',
            ha='center',
            fontsize=9,
            color=color
        )
    
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.85, 0.95)
    
    plt.title('Model Parameters vs. Test Accuracy', fontsize=15)
    plt.tight_layout()
    plt.savefig('parameters_vs_accuracy.png', dpi=300)
    plt.show()