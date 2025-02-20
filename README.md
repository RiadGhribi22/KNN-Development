# KNN Development

## Description
This project implements a classifier based on the k-Nearest Neighbors (k-NN) algorithm with data normalization 
and performance evaluation. It also includes a bagging approach to improve model robustness.

## Prerequisites
- Python 3.x
- Standard modules: `math`, `csv`, `random`

## Files
- **main.py**: Contains the main code executing the k-NN algorithm with and without bagging.
- **dataset.csv**: CSV file containing the data to be classified.

## Features

### 1. Data Normalization
The function `normalisation(dataset, missing_value='?')` adjusts numerical attribute values to bring them within the [0,1] range, facilitating distance comparisons.

### 2. Dataset Loading
The function `load(filename)` loads a CSV file and converts numerical values to floats while keeping other values as strings.

### 3. Euclidean Distance
The function `euclide(x1, x2)` calculates the Euclidean distance between two points.

### 4. Finding Neighbors
The function `get_neighbors(dataset, inst, k)` finds the `k` nearest neighbors of a given instance using the Euclidean distance.

### 5. Class Prediction
The function `predicate(neighbors)` determines the majority class among the found neighbors.

### 6. Model Evaluation
The functions `evaluation(test_set, predicate_total)` and `calculate_P_R_F(test_set, predicate_total)` evaluate the precision, recall, and F-measure of the model.

### 7. Data Splitting and Classification
The function `split(dataset, percentage, k)` splits the data into training and test sets, applies the k-NN algorithm, and evaluates the results.

### 8. Bagging Implementation
The function `bagging(dataset, percentage, k, t)` applies the k-NN algorithm on multiple bootstrap subsets of the dataset to improve prediction stability.

## Execution

1. Modify the dataset file by replacing `filename` with the correct path in the main code:
   ```python
   filename = 'path/to/your/dataset.csv'
   ```
2. Run the main script:
   ```sh
   python knn.py
   ```
3. Observe the displayed results, comparing k-NN performance with and without bagging.

## Expected Results
The program will display statistics on correct/incorrect classification and precision, recall, and F-measure scores before and after applying bagging.

## Possible Improvements
- Optimization of missing value handling.
- Implementation of other distance metrics (Manhattan, Minkowski).
- Dynamic parameterization of `k` and `t` based on the dataset.

