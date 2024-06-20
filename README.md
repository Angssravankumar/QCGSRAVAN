# QCGS

Here's a README file for your quantum-enhanced SVM classifier project using the Wine dataset:

---

# Quantum-Enhanced SVM Classifier for Wine Dataset

This project demonstrates the application of a quantum-enhanced SVM classifier on the Wine dataset. We use a variational quantum circuit to create a quantum kernel and optimize it using kernel-target alignment. The quantum kernel is then used to train an SVM classifier.

## Project Structure

- **data**: Contains the Wine dataset (loaded from `sklearn`).
- **src**: Contains the main script `quantum_svm_wine.py` with all the necessary functions and the main execution code.
- **output**: Contains generated plots and results.

## Dependencies

The project requires the following Python libraries:
- `numpy`
- `matplotlib`
- `scikit-learn`
- `joblib`
- `pennylane`
- `scipy`

You can install these libraries using `pip`:
```bash
pip install numpy matplotlib scikit-learn joblib pennylane scipy
```

## Files

### `quantum_svm_wine.py`

This script contains the main execution code and the following functions:

1. **`quantum_map_wine(x, params)`**: Defines a simple quantum feature map.
2. **`variational_circuit_wine(x, params)`**: Defines the variational quantum circuit.
3. **`kernel_element_wine(x1, x2, params)`**: Computes the kernel matrix element.
4. **`qek_matrix_wine(X, params)`**: Computes the Quantum Enhanced Kernel (QEK) matrix.
5. **`alignment_wine(params, X, y)`**: Kernel-target alignment function.
6. **`compute_grid_kernel_matrix_wine(grid_points, X_train, params)`**: Computes the kernel matrix between grid points and training data.

### Usage

1. **Load the Wine dataset**:
   - Load the dataset using `load_wine()` from `sklearn.datasets`.
   - Select samples from two classes only (class 0 and class 1).
   - Use only the first two features for visualization.

2. **Define the Quantum Feature Map and Variational Circuit**:
   - Define a simple quantum feature map with rotation gates.
   - Define the variational quantum circuit using `pennylane`.

3. **Compute the QEK Matrix and Optimize Parameters**:
   - Initialize random parameters and optimize them using `scipy.optimize.minimize`.
   - Compute the optimized QEK matrix.

4. **Train the SVM Classifier**:
   - Split the dataset into training and testing sets.
   - Precompute the kernel matrix between the grid points and training data.
   - Train an SVM classifier on the optimized QEK matrix.
   - Plot the decision boundary.

5. **Evaluate the Classifier**:
   - Compute the kernel matrix for the test set.
   - Evaluate the classifier's accuracy on the test set.

### Example Plot

The script generates a plot showing the decision boundary of the SVM classifier with the quantum kernel:

![SVM Decision Boundary with Quantum Kernel](output/decision_boundary.png)

### Classification Accuracy

The script prints the classification accuracy on the test set:
```
Classification accuracy: 0.90
```

## How to Run

1. Ensure all dependencies are installed.
2. Run the `quantum_svm_wine.py` script:
   ```bash
   python quantum_svm_wine.py
   ```
3. The script will load the dataset, compute the quantum kernel, train the SVM classifier, and plot the decision boundary.

## References

- [PennyLane Documentation](https://pennylane.ai/qml/glossary.html)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Wine Dataset](https://archive.ics.uci.edu/ml/datasets/Wine)

---

This README provides a comprehensive overview of the project, including dependencies, usage instructions, and key functionalities. Feel free to modify it to better suit your needs!
