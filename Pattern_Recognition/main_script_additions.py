# Add to imports:
from visualization_utils import (
    plot_confusion_matrix, 
    visualize_model_filters, 
    visualize_activation_maps,
    compare_parameters_and_performance
)

# Add this at the end of the main() function:
# Select best model for detailed analysis
best_model_idx = np.argmax(test_accuracies)
best_model_name = model_names[best_model_idx]
best_model = models[best_model_name][0]

# Detailed analysis of best model
print(f"\nDetailed analysis of best model: {best_model_name}")
plot_confusion_matrix(best_model, x_test, y_test)
visualize_model_filters(best_model)
visualize_activation_maps(best_model, x_test, image_index=42)  # Choose a representative image

# Compare parameters vs performance
param_counts = [count_parameters(models[name][0]) for name in model_names]
compare_parameters_and_performance(model_names, param_counts, test_accuracies)