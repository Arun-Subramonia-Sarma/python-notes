# NumPy - Complete Guide

A comprehensive guide to NumPy (Numerical Python), the fundamental package for scientific computing in Python.

## Table of Contents

- [Installation and Setup](#installation-and-setup)
- [NumPy Basics](#numpy-basics)
- [Array Creation](#array-creation)
- [Array Attributes and Properties](#array-attributes-and-properties)
- [Array Indexing and Slicing](#array-indexing-and-slicing)
- [Array Operations](#array-operations)
- [Mathematical Functions](#mathematical-functions)
- [Array Manipulation](#array-manipulation)
- [Broadcasting](#broadcasting)
- [Linear Algebra](#linear-algebra)
- [Statistics and Aggregation](#statistics-and-aggregation)
- [Random Numbers](#random-numbers)
- [File I/O](#file-io)
- [Performance Optimization](#performance-optimization)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)

## Installation and Setup

### Installation

```bash
# Using pip
pip install numpy

# Using UV (recommended for projects)
uv add numpy

# Using conda
conda install numpy

# Install with additional packages for scientific computing
uv add numpy scipy matplotlib pandas

# Check installation
python -c "import numpy as np; print(np.__version__)"
```

### Basic Import

```python
import numpy as np

# Check version and configuration
print(f"NumPy version: {np.__version__}")
print(f"NumPy configuration:")
np.show_config()
```

## NumPy Basics

### What is NumPy?

NumPy is the foundation of scientific computing in Python, providing:

- **N-dimensional arrays (ndarray)**: Efficient storage and operations
- **Broadcasting**: Operations on arrays of different shapes
- **Mathematical functions**: Comprehensive mathematical operations
- **Linear algebra**: Matrix operations and decompositions
- **Random number generation**: Statistical and random sampling
- **Integration**: Works seamlessly with other scientific Python packages

### Why Use NumPy?

```python
# Python list vs NumPy array comparison
import time

# Python list operations (slow)
python_list = list(range(1000000))
start_time = time.time()
result_list = [x * 2 for x in python_list]
list_time = time.time() - start_time

# NumPy array operations (fast)
numpy_array = np.arange(1000000)
start_time = time.time()
result_array = numpy_array * 2
numpy_time = time.time() - start_time

print(f"Python list time: {list_time:.4f} seconds")
print(f"NumPy array time: {numpy_time:.4f} seconds")
print(f"NumPy is {list_time/numpy_time:.1f}x faster")

# Memory efficiency
import sys
print(f"Python list memory: {sys.getsizeof(python_list)} bytes")
print(f"NumPy array memory: {numpy_array.nbytes} bytes")
```

## Array Creation

### Basic Array Creation

```python
# From Python lists
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"1D array: {arr1d}")
print(f"2D array:\n{arr2d}")
print(f"3D array:\n{arr3d}")

# Specifying data type
int_array = np.array([1, 2, 3], dtype=int)
float_array = np.array([1, 2, 3], dtype=float)
complex_array = np.array([1, 2, 3], dtype=complex)

print(f"Integer: {int_array.dtype}")
print(f"Float: {float_array.dtype}")
print(f"Complex: {complex_array.dtype}")
```

### Array Generation Functions

```python
# Zeros, ones, and empty arrays
zeros = np.zeros((3, 4))           # 3x4 array of zeros
ones = np.ones((2, 3, 4))         # 2x3x4 array of ones
empty = np.empty((2, 2))          # Uninitialized array
full = np.full((3, 3), 7)         # 3x3 array filled with 7

print(f"Zeros shape: {zeros.shape}")
print(f"Ones shape: {ones.shape}")
print(f"Full array:\n{full}")

# Identity matrices
identity = np.eye(4)              # 4x4 identity matrix
diagonal = np.diag([1, 2, 3, 4])  # Diagonal matrix

print(f"Identity matrix:\n{identity}")
print(f"Diagonal matrix:\n{diagonal}")

# Range arrays
arange_arr = np.arange(10)        # [0, 1, 2, ..., 9]
arange_step = np.arange(1, 10, 2) # [1, 3, 5, 7, 9]
linspace_arr = np.linspace(0, 1, 5) # 5 evenly spaced numbers from 0 to 1
logspace_arr = np.logspace(0, 2, 5)  # 5 logarithmically spaced numbers

print(f"Arange: {arange_arr}")
print(f"Arange with step: {arange_step}")
print(f"Linspace: {linspace_arr}")
print(f"Logspace: {logspace_arr}")

# Meshgrid for coordinate arrays
x = np.array([1, 2, 3])
y = np.array([4, 5])
X, Y = np.meshgrid(x, y)
print(f"X grid:\n{X}")
print(f"Y grid:\n{Y}")
```

### Random Array Creation

```python
# Random number generation
np.random.seed(42)  # For reproducible results

random_uniform = np.random.random((3, 3))      # Uniform [0, 1)
random_normal = np.random.randn(3, 3)          # Standard normal
random_integers = np.random.randint(1, 10, (3, 3))  # Random integers

print(f"Random uniform:\n{random_uniform}")
print(f"Random normal:\n{random_normal}")
print(f"Random integers:\n{random_integers}")

# Specific distributions
exponential = np.random.exponential(2, 1000)   # Exponential distribution
binomial = np.random.binomial(10, 0.5, 1000)   # Binomial distribution
choice = np.random.choice([1, 2, 3, 4, 5], size=10)  # Random choice

print(f"Exponential mean: {exponential.mean():.2f}")
print(f"Binomial mean: {binomial.mean():.2f}")
print(f"Random choice: {choice}")
```

## Array Attributes and Properties

### Basic Attributes

```python
# Create a sample array
arr = np.random.randint(1, 10, (3, 4, 2))

print(f"Array:\n{arr}")
print(f"Shape: {arr.shape}")           # Dimensions
print(f"Size: {arr.size}")             # Total number of elements
print(f"Ndim: {arr.ndim}")             # Number of dimensions
print(f"Dtype: {arr.dtype}")           # Data type
print(f"Itemsize: {arr.itemsize}")     # Size of each element in bytes
print(f"Nbytes: {arr.nbytes}")         # Total bytes consumed

# Memory layout
print(f"Flags:\n{arr.flags}")
print(f"Strides: {arr.strides}")       # Bytes to skip for each dimension
```

### Data Types

```python
# Common data types
int8_arr = np.array([1, 2, 3], dtype=np.int8)      # 8-bit integer
int32_arr = np.array([1, 2, 3], dtype=np.int32)    # 32-bit integer
float32_arr = np.array([1, 2, 3], dtype=np.float32) # 32-bit float
float64_arr = np.array([1, 2, 3], dtype=np.float64) # 64-bit float
bool_arr = np.array([True, False, True], dtype=bool)

print(f"int8 size: {int8_arr.itemsize} bytes")
print(f"int32 size: {int32_arr.itemsize} bytes")
print(f"float32 size: {float32_arr.itemsize} bytes")
print(f"float64 size: {float64_arr.itemsize} bytes")

# Type conversion
original = np.array([1.7, 2.3, 3.9])
as_int = original.astype(int)
as_str = original.astype(str)

print(f"Original: {original}")
print(f"As integer: {as_int}")
print(f"As string: {as_str}")
```

## Array Indexing and Slicing

### Basic Indexing

```python
# 1D array indexing
arr1d = np.array([10, 20, 30, 40, 50])
print(f"Array: {arr1d}")
print(f"First element: {arr1d[0]}")
print(f"Last element: {arr1d[-1]}")
print(f"Second to fourth: {arr1d[1:4]}")
print(f"Every second element: {arr1d[::2]}")
print(f"Reversed: {arr1d[::-1]}")

# 2D array indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D Array:\n{arr2d}")
print(f"Element at (1,2): {arr2d[1, 2]}")
print(f"First row: {arr2d[0, :]}")
print(f"Last column: {arr2d[:, -1]}")
print(f"Subarray:\n{arr2d[1:, 1:]}")

# 3D array indexing
arr3d = np.random.randint(1, 10, (2, 3, 4))
print(f"\n3D Array shape: {arr3d.shape}")
print(f"First 2D slice:\n{arr3d[0]}")
print(f"Element at (0,1,2): {arr3d[0, 1, 2]}")
```

### Advanced Indexing

```python
# Boolean indexing
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask = arr > 5
print(f"Array: {arr}")
print(f"Mask (> 5): {mask}")
print(f"Elements > 5: {arr[mask]}")
print(f"Elements > 5 (direct): {arr[arr > 5]}")

# Multiple conditions
complex_mask = (arr > 3) & (arr < 8)  # Must use & not 'and'
print(f"Elements between 3 and 8: {arr[complex_mask]}")

# Fancy indexing with arrays
indices = np.array([0, 2, 4, 6])
print(f"Elements at indices {indices}: {arr[indices]}")

# 2D fancy indexing
arr2d = np.arange(12).reshape(3, 4)
print(f"\n2D Array:\n{arr2d}")
print(f"Rows 0 and 2: \n{arr2d[[0, 2]]}")
print(f"Columns 1 and 3: \n{arr2d[:, [1, 3]]}")

# Combined fancy and boolean indexing
print(f"Values > 5 in specific rows:\n{arr2d[[1, 2]][arr2d[[1, 2]] > 5]}")
```

### Modifying Arrays

```python
# Modifying elements
arr = np.arange(10)
arr[5] = 99
arr[1:4] = [11, 12, 13]
print(f"Modified array: {arr}")

# Boolean indexing for modification
arr[arr > 50] = -1
print(f"After conditional modification: {arr}")

# 2D modifications
arr2d = np.arange(12).reshape(3, 4)
arr2d[1, :] = 0  # Set entire row to 0
arr2d[:, 2] = [10, 20, 30]  # Set entire column
print(f"Modified 2D array:\n{arr2d}")

# Views vs copies
original = np.arange(10)
view = original[2:8]      # Creates a view (shares memory)
copy = original[2:8].copy()  # Creates a copy

view[0] = 999
print(f"Original after view modification: {original}")
print(f"View: {view}")

copy[1] = 888
print(f"Original after copy modification: {original}")
print(f"Copy: {copy}")
```

## Array Operations

### Element-wise Operations

```python
# Arithmetic operations
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")  # Element-wise multiplication
print(f"a / b = {a / b}")
print(f"a ** b = {a ** b}")  # Element-wise power

# Operations with scalars
print(f"\na + 10 = {a + 10}")
print(f"a * 2 = {a * 2}")
print(f"a ** 2 = {a ** 2}")

# Comparison operations
print(f"\na > 2: {a > 2}")
print(f"a == b: {a == b}")
print(f"a <= 3: {a <= 3}")

# Logical operations
mask1 = a > 2
mask2 = a < 4
print(f"a > 2: {mask1}")
print(f"a < 4: {mask2}")
print(f"(a > 2) & (a < 4): {mask1 & mask2}")
print(f"Elements where (a > 2) & (a < 4): {a[mask1 & mask2]}")
```

### Matrix Operations

```python
# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")

# Different ways to do matrix multiplication
print(f"A @ B (matrix multiplication):\n{A @ B}")
print(f"np.dot(A, B):\n{np.dot(A, B)}")
print(f"A.dot(B):\n{A.dot(B)}")

# Element-wise vs matrix multiplication
print(f"A * B (element-wise):\n{A * B}")
print(f"A @ B (matrix multiplication):\n{A @ B}")

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"\nVector v1: {v1}")
print(f"Vector v2: {v2}")
print(f"Dot product: {np.dot(v1, v2)}")
print(f"Cross product: {np.cross(v1, v2)}")

# Matrix properties
print(f"\nMatrix A transpose:\n{A.T}")
print(f"Matrix A inverse:\n{np.linalg.inv(A)}")
print(f"Matrix A determinant: {np.linalg.det(A)}")
```

### Universal Functions (ufuncs)

```python
# Trigonometric functions
angles = np.array([0, np.pi/4, np.pi/2, np.pi])
print(f"Angles: {angles}")
print(f"sin(angles): {np.sin(angles)}")
print(f"cos(angles): {np.cos(angles)}")
print(f"tan(angles): {np.tan(angles)}")

# Exponential and logarithmic functions
x = np.array([1, 2, 3, 4, 5])
print(f"\nArray x: {x}")
print(f"exp(x): {np.exp(x)}")
print(f"log(x): {np.log(x)}")
print(f"log10(x): {np.log10(x)}")
print(f"sqrt(x): {np.sqrt(x)}")

# Rounding functions
decimals = np.array([1.234, 2.567, 3.891])
print(f"\nDecimals: {decimals}")
print(f"Round: {np.round(decimals, 2)}")
print(f"Floor: {np.floor(decimals)}")
print(f"Ceil: {np.ceil(decimals)}")
print(f"Truncate: {np.trunc(decimals)}")

# Absolute values and signs
signed = np.array([-3, -1, 0, 1, 3])
print(f"\nSigned array: {signed}")
print(f"Absolute: {np.abs(signed)}")
print(f"Sign: {np.sign(signed)}")
```

## Mathematical Functions

### Statistical Functions

```python
# Sample data
data = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])

print(f"Data:\n{data}")

# Basic statistics
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Standard deviation: {np.std(data)}")
print(f"Variance: {np.var(data)}")
print(f"Min: {np.min(data)}")
print(f"Max: {np.max(data)}")

# Axis-specific operations
print(f"\nMean along axis 0 (columns): {np.mean(data, axis=0)}")
print(f"Mean along axis 1 (rows): {np.mean(data, axis=1)}")
print(f"Sum along axis 0: {np.sum(data, axis=0)}")
print(f"Sum along axis 1: {np.sum(data, axis=1)}")

# Cumulative operations
print(f"\nCumulative sum: {np.cumsum(data.flatten())}")
print(f"Cumulative product: {np.cumprod([1, 2, 3, 4, 5])}")

# Finding positions
print(f"Position of min: {np.argmin(data)}")
print(f"Position of max: {np.argmax(data)}")
print(f"Position of min along axis 1: {np.argmin(data, axis=1)}")
```

### Advanced Mathematical Operations

```python
# Sorting
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"Original: {arr}")
print(f"Sorted: {np.sort(arr)}")
print(f"Sort indices: {np.argsort(arr)}")

# 2D sorting
arr2d = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
print(f"\n2D array:\n{arr2d}")
print(f"Sorted along axis 0:\n{np.sort(arr2d, axis=0)}")
print(f"Sorted along axis 1:\n{np.sort(arr2d, axis=1)}")

# Unique values
arr_with_duplicates = np.array([1, 2, 2, 3, 3, 3, 4])
unique_values, counts = np.unique(arr_with_duplicates, return_counts=True)
print(f"\nArray with duplicates: {arr_with_duplicates}")
print(f"Unique values: {unique_values}")
print(f"Counts: {counts}")

# Set operations
a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7])
print(f"\nArray a: {a}")
print(f"Array b: {b}")
print(f"Intersection: {np.intersect1d(a, b)}")
print(f"Union: {np.union1d(a, b)}")
print(f"Difference a-b: {np.setdiff1d(a, b)}")
print(f"Symmetric difference: {np.setxor1d(a, b)}")

# Polynomial operations
coefficients = [1, -2, 1]  # Represents x^2 - 2x + 1
x_values = np.array([0, 1, 2, 3])
y_values = np.polyval(coefficients, x_values)
print(f"\nPolynomial coefficients: {coefficients}")
print(f"x values: {x_values}")
print(f"y values: {y_values}")

# Roots of polynomial
roots = np.roots(coefficients)
print(f"Roots: {roots}")
```

## Array Manipulation

### Reshaping Arrays

```python
# Basic reshaping
arr = np.arange(12)
print(f"Original array: {arr}")
print(f"Reshaped to 3x4:\n{arr.reshape(3, 4)}")
print(f"Reshaped to 2x6:\n{arr.reshape(2, 6)}")
print(f"Reshaped to 2x2x3:\n{arr.reshape(2, 2, 3)}")

# Using -1 for automatic dimension calculation
print(f"Reshape with -1 (auto-calculate):\n{arr.reshape(4, -1)}")
print(f"Reshape to column vector:\n{arr.reshape(-1, 1)}")

# Flattening arrays
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2D array:\n{arr2d}")
print(f"Flattened: {arr2d.flatten()}")
print(f"Raveled: {arr2d.ravel()}")  # Similar to flatten but may return view

# Adding and removing dimensions
print(f"Original shape: {arr.shape}")
print(f"Add axis 0: {np.expand_dims(arr, axis=0).shape}")
print(f"Add axis 1: {np.expand_dims(arr, axis=1).shape}")
print(f"Using newaxis: {arr[np.newaxis, :].shape}")

# Squeezing dimensions (remove dimensions of size 1)
arr_with_single_dims = np.array([[[1], [2], [3]]])
print(f"Array with single dims shape: {arr_with_single_dims.shape}")
print(f"Squeezed shape: {np.squeeze(arr_with_single_dims).shape}")
```

### Combining Arrays

```python
# Concatenation
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(f"Array a:\n{a}")
print(f"Array b:\n{b}")

# Concatenate along different axes
print(f"Concatenate along axis 0 (vertical):\n{np.concatenate([a, b], axis=0)}")
print(f"Concatenate along axis 1 (horizontal):\n{np.concatenate([a, b], axis=1)}")

# Convenience functions
print(f"Vertical stack:\n{np.vstack([a, b])}")
print(f"Horizontal stack:\n{np.hstack([a, b])}")
print(f"Depth stack:\n{np.dstack([a, b])}")

# Stacking arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(f"\nArray 1: {arr1}")
print(f"Array 2: {arr2}")
print(f"Stack along new axis 0:\n{np.stack([arr1, arr2], axis=0)}")
print(f"Stack along new axis 1:\n{np.stack([arr1, arr2], axis=1)}")

# Repeating arrays
print(f"Repeat elements: {np.repeat([1, 2, 3], 3)}")
print(f"Tile array: {np.tile([1, 2, 3], 3)}")

# 2D tiling
arr_2d = np.array([[1, 2], [3, 4]])
print(f"Original 2D:\n{arr_2d}")
print(f"Tiled 2D:\n{np.tile(arr_2d, (2, 3))}")
```

### Splitting Arrays

```python
# Array to split
arr = np.arange(16).reshape(4, 4)
print(f"Array to split:\n{arr}")

# Split into equal parts
split_horizontal = np.hsplit(arr, 2)  # Split into 2 parts horizontally
split_vertical = np.vsplit(arr, 2)    # Split into 2 parts vertically

print(f"Horizontal split (2 parts):")
for i, part in enumerate(split_horizontal):
    print(f"Part {i}:\n{part}")

print(f"Vertical split (2 parts):")
for i, part in enumerate(split_vertical):
    print(f"Part {i}:\n{part}")

# Split at specific indices
arr1d = np.arange(10)
split_at_indices = np.split(arr1d, [3, 7])  # Split at positions 3 and 7
print(f"\n1D array: {arr1d}")
print(f"Split at indices [3, 7]: {split_at_indices}")

# Array splitting with unequal parts
unequal_split = np.array_split(arr1d, 3)  # Split into 3 (possibly unequal) parts
print(f"Unequal split into 3 parts: {unequal_split}")
```

## Broadcasting

### Broadcasting Rules

```python
# Broadcasting allows operations between arrays of different shapes
# Rules:
# 1. Arrays are aligned from the rightmost dimension
# 2. Dimensions of size 1 are stretched to match
# 3. Missing dimensions are added with size 1

# Scalar with array
arr = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10
print(f"Array shape: {arr.shape}")
print(f"Scalar + Array:\n{scalar + arr}")

# 1D array with 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
arr_1d = np.array([10, 20, 30])            # Shape: (3,)
print(f"\n2D array shape: {arr_2d.shape}")
print(f"1D array shape: {arr_1d.shape}")
print(f"2D + 1D:\n{arr_2d + arr_1d}")

# Column vector with row vector
col_vector = np.array([[1], [2], [3]])     # Shape: (3, 1)
row_vector = np.array([10, 20])            # Shape: (2,)
print(f"\nColumn vector shape: {col_vector.shape}")
print(f"Row vector shape: {row_vector.shape}")
print(f"Column + Row:\n{col_vector + row_vector}")

# Broadcasting with different operations
matrix = np.random.randint(1, 10, (3, 4))
col_means = np.mean(matrix, axis=0)        # Mean of each column
row_means = np.mean(matrix, axis=1, keepdims=True)  # Mean of each row

print(f"\nMatrix:\n{matrix}")
print(f"Column means: {col_means}")
print(f"Row means: {row_means.flatten()}")
print(f"Matrix - column means:\n{matrix - col_means}")
print(f"Matrix - row means:\n{matrix - row_means}")
```

### Broadcasting Examples

```python
# Normalizing data (z-score normalization)
data = np.random.normal(50, 15, (100, 5))  # 100 samples, 5 features
means = np.mean(data, axis=0)
stds = np.std(data, axis=0)
normalized_data = (data - means) / stds

print(f"Original data shape: {data.shape}")
print(f"Original means: {means}")
print(f"Original stds: {stds}")
print(f"Normalized means: {np.mean(normalized_data, axis=0)}")
print(f"Normalized stds: {np.std(normalized_data, axis=0)}")

# Distance calculation using broadcasting
points = np.random.rand(5, 2)  # 5 points in 2D
print(f"\nPoints:\n{points}")

# Calculate all pairwise distances
diff = points[:, np.newaxis] - points[np.newaxis, :]  # Broadcasting
distances = np.sqrt(np.sum(diff**2, axis=2))

print(f"Distance matrix:\n{distances}")

# Creating a multiplication table using broadcasting
x = np.arange(1, 11).reshape(-1, 1)  # Column vector [1, 2, ..., 10]
y = np.arange(1, 11)                 # Row vector [1, 2, ..., 10]
multiplication_table = x * y

print(f"\nMultiplication table (10x10):\n{multiplication_table}")
```

## Linear Algebra

### Basic Linear Algebra Operations

```python
# Matrix operations
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])  # Note: changed last element to make it invertible

B = np.array([[1, 0, 1],
              [0, 1, 1],
              [1, 1, 0]])

print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")

# Basic operations
print(f"A + B:\n{A + B}")
print(f"A - B:\n{A - B}")
print(f"A @ B (matrix multiplication):\n{A @ B}")

# Matrix properties
print(f"\nA transpose:\n{A.T}")
print(f"A diagonal: {np.diag(A)}")
print(f"A trace: {np.trace(A)}")
print(f"A determinant: {np.linalg.det(A):.4f}")

# Matrix inverse and pseudo-inverse
try:
    A_inv = np.linalg.inv(A)
    print(f"A inverse:\n{A_inv}")
    print(f"A @ A_inv (should be identity):\n{A @ A_inv}")
except np.linalg.LinAlgError:
    print("Matrix A is singular (not invertible)")
    A_pinv = np.linalg.pinv(A)  # Pseudo-inverse
    print(f"A pseudo-inverse:\n{A_pinv}")
```

### Eigenvalues and Eigenvectors

```python
# Symmetric matrix (guaranteed real eigenvalues)
symmetric_matrix = np.array([[4, 1, 1],
                           [1, 3, 2],
                           [1, 2, 3]])

print(f"Symmetric matrix:\n{symmetric_matrix}")

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(symmetric_matrix)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify eigenvalue equation: A * v = λ * v
for i in range(len(eigenvalues)):
    λ = eigenvalues[i]
    v = eigenvectors[:, i]
    Av = symmetric_matrix @ v
    λv = λ * v
    print(f"\nEigenvalue {i+1}: {λ:.4f}")
    print(f"A @ v = {Av}")
    print(f"λ * v = {λv}")
    print(f"Difference: {np.allclose(Av, λv)}")

# Eigenvalue decomposition
reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
print(f"\nReconstructed matrix:\n{reconstructed}")
print(f"Reconstruction accurate: {np.allclose(symmetric_matrix, reconstructed)}")
```

### Decompositions

```python
# Singular Value Decomposition (SVD)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]], dtype=float)

print(f"Original matrix ({matrix.shape}):\n{matrix}")

U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
print(f"U shape: {U.shape}")
print(f"Singular values: {s}")
print(f"Vt shape: {Vt.shape}")

# Reconstruct matrix
reconstructed = U @ np.diag(s) @ Vt
print(f"Reconstructed matrix:\n{reconstructed}")
print(f"Reconstruction accurate: {np.allclose(matrix, reconstructed)}")

# QR decomposition
square_matrix = np.random.rand(4, 4)
Q, R = np.linalg.qr(square_matrix)

print(f"\nQR Decomposition:")
print(f"Q (orthogonal):\n{Q}")
print(f"R (upper triangular):\n{R}")
print(f"Q @ R:\n{Q @ R}")
print(f"Original matrix:\n{square_matrix}")
print(f"QR reconstruction accurate: {np.allclose(square_matrix, Q @ R)}")

# Check orthogonality of Q
print(f"Q.T @ Q (should be identity):\n{Q.T @ Q}")
```

### Solving Linear Systems

```python
# Solve Ax = b
A = np.array([[2, 1, 1],
              [1, 3, 2],
              [1, 0, 0]], dtype=float)

b = np.array([4, 5, 6], dtype=float)

print(f"System Ax = b:")
print(f"A:\n{A}")
print(f"b: {b}")

# Direct solution
x = np.linalg.solve(A, b)
print(f"Solution x: {x}")

# Verify solution
verification = A @ x
print(f"A @ x = {verification}")
print(f"b = {b}")
print(f"Solution correct: {np.allclose(verification, b)}")

# Least squares solution (for overdetermined systems)
# Add more equations than unknowns
A_over = np.vstack([A, [[1, 1, 1]]])  # 4 equations, 3 unknowns
b_over = np.append(b, [7])

print(f"\nOverdetermined system:")
print(f"A_over shape: {A_over.shape}")
print(f"b_over: {b_over}")

x_lstsq, residuals, rank, s = np.linalg.lstsq(A_over, b_over, rcond=None)
print(f"Least squares solution: {x_lstsq}")
print(f"Residuals: {residuals}")
```

## Statistics and Aggregation

### Descriptive Statistics

```python
# Generate sample data
np.random.seed(42)
data = np.random.normal(100, 15, (1000, 5))  # 1000 samples, 5 features
print(f"Data shape: {data.shape}")

# Basic statistics
print(f"Mean: {np.mean(data, axis=0)}")
print(f"Median: {np.median(data, axis=0)}")
print(f"Standard deviation: {np.std(data, axis=0)}")
print(f"Variance: {np.var(data, axis=0)}")
print(f"Min: {np.min(data, axis=0)}")
print(f"Max: {np.max(data, axis=0)}")
print(f"Range: {np.ptp(data, axis=0)}")  # Peak-to-peak (max - min)

# Percentiles and quantiles
print(f"\nPercentiles:")
print(f"25th percentile: {np.percentile(data, 25, axis=0)}")
print(f"50th percentile (median): {np.percentile(data, 50, axis=0)}")
print(f"75th percentile: {np.percentile(data, 75, axis=0)}")

# Quantiles (same as percentiles but with fractions)
print(f"Quantiles [0.25, 0.5, 0.75]: \n{np.quantile(data, [0.25, 0.5, 0.75], axis=0)}")

# Histogram
feature_0 = data[:, 0]
hist, bin_edges = np.histogram(feature_0, bins=20)
print(f"\nHistogram of feature 0:")
print(f"Counts: {hist}")
print(f"Bin edges: {bin_edges}")
```

### Correlation and Covariance

```python
# Create correlated data
n_samples = 1000
x = np.random.normal(0, 1, n_samples)
y = 2 * x + np.random.normal(0, 0.5, n_samples)  # y correlated with x
z = np.random.normal(0, 1, n_samples)            # z independent

data_matrix = np.column_stack([x, y, z])
print(f"Data matrix shape: {data_matrix.shape}")

# Covariance matrix
cov_matrix = np.cov(data_matrix.T)  # Transpose for feature covariance
print(f"Covariance matrix:\n{cov_matrix}")

# Correlation matrix
corr_matrix = np.corrcoef(data_matrix.T)
print(f"Correlation matrix:\n{corr_matrix}")

# Manual correlation calculation
def correlation(x, y):
    """Calculate Pearson correlation coefficient."""
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    return np.sum(x_centered * y_centered) / (np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))

print(f"\nManual correlation x-y: {correlation(x, y):.4f}")
print(f"NumPy correlation x-y: {np.corrcoef(x, y)[0, 1]:.4f}")
```

### Advanced Statistical Functions

```python
# Binomial coefficient (combinations)
n, k = 10, 3
combinations = np.math.factorial(n) // (np.math.factorial(k) * np.math.factorial(n-k))
print(f"C({n}, {k}) = {combinations}")

# Using scipy for more advanced statistics (if available)
try:
    from scipy import stats
    
    # Sample data
    sample_data = np.random.normal(50, 10, 100)
    
    # Statistical tests
    # Normality test (Shapiro-Wilk)
    statistic, p_value = stats.shapiro(sample_data)
    print(f"Shapiro-Wilk test: statistic={statistic:.4f}, p-value={p_value:.4f}")
    
    # Descriptive statistics
    desc_stats = stats.describe(sample_data)
    print(f"Descriptive statistics: {desc_stats}")
    
except ImportError:
    print("SciPy not available for advanced statistical tests")

# Robust statistics (less sensitive to outliers)
# Add some outliers
data_with_outliers = np.append(data[:, 0], [200, 250, -100])

print(f"\nData with outliers:")
print(f"Mean: {np.mean(data_with_outliers):.2f}")
print(f"Median: {np.median(data_with_outliers):.2f}")  # More robust
print(f"Std: {np.std(data_with_outliers):.2f}")

# Percentile-based robust statistics
q75, q25 = np.percentile(data_with_outliers, [75, 25])
iqr = q75 - q25  # Interquartile range
print(f"IQR (robust measure of spread): {iqr:.2f}")
```

## Random Numbers

### Random Number Generation

```python
# Set seed for reproducibility
np.random.seed(42)

# Basic random number generation
print("Random number generation:")
print(f"Single random number: {np.random.random()}")
print(f"Random array (3x3): \n{np.random.random((3, 3))}")

# Different distributions
print(f"\nUniform [0, 1): {np.random.rand(5)}")
print(f"Uniform [low, high): {np.random.uniform(10, 20, 5)}")
print(f"Standard normal: {np.random.randn(5)}")
print(f"Normal (μ=50, σ=10): {np.random.normal(50, 10, 5)}")

# Integer random numbers
print(f"\nRandom integers [0, 10): {np.random.randint(0, 10, 10)}")
print(f"Random integers 2D: \n{np.random.randint(1, 7, (3, 3))}")  # Dice rolls

# Random choice and sampling
fruits = ['apple', 'banana', 'cherry', 'date', 'elderberry']
print(f"\nRandom choice: {np.random.choice(fruits)}")
print(f"Random choices (5): {np.random.choice(fruits, 5)}")
print(f"Random choices (no replacement): {np.random.choice(fruits, 3, replace=False)}")

# Weighted random choice
weights = [0.1, 0.2, 0.3, 0.3, 0.1]  # Probabilities for each fruit
print(f"Weighted random choice: {np.random.choice(fruits, 10, p=weights)}")
```

### Specific Distributions

```python
# Various probability distributions
np.random.seed(123)

print("Various probability distributions:")

# Binomial distribution
n_trials, prob_success = 10, 0.3
binomial_samples = np.random.binomial(n_trials, prob_success, 1000)
print(f"Binomial (n={n_trials}, p={prob_success}) mean: {np.mean(binomial_samples):.2f}")
print(f"Theoretical mean: {n_trials * prob_success}")

# Poisson distribution
lambda_param = 5
poisson_samples = np.random.poisson(lambda_param, 1000)
print(f"Poisson (λ={lambda_param}) mean: {np.mean(poisson_samples):.2f}")
print(f"Theoretical mean: {lambda_param}")

# Exponential distribution
scale_param = 2
exponential_samples = np.random.exponential(scale_param, 1000)
print(f"Exponential (scale={scale_param}) mean: {np.mean(exponential_samples):.2f}")
print(f"Theoretical mean: {scale_param}")

# Gamma distribution
shape, scale = 2, 2
gamma_samples = np.random.gamma(shape, scale, 1000)
print(f"Gamma (shape={shape}, scale={scale}) mean: {np.mean(gamma_samples):.2f}")
print(f"Theoretical mean: {shape * scale}")

# Beta distribution
alpha, beta = 2, 5
beta_samples = np.random.beta(alpha, beta, 1000)
print(f"Beta (α={alpha}, β={beta}) mean: {np.mean(beta_samples):.2f}")
print(f"Theoretical mean: {alpha / (alpha + beta):.2f}")
```

### Random Sampling and Shuffling

```python
# Random sampling without replacement
population = np.arange(100)
sample = np.random.choice(population, size=10, replace=False)
print(f"Random sample from population: {sample}")

# Shuffling arrays
deck = np.arange(52)  # Deck of cards (0-51)
print(f"Original deck: {deck[:10]}...")
np.random.shuffle(deck)  # In-place shuffle
print(f"Shuffled deck: {deck[:10]}...")

# Permutation (returns shuffled copy)
original = np.array([1, 2, 3, 4, 5])
shuffled_copy = np.random.permutation(original)
print(f"Original: {original}")
print(f"Shuffled copy: {shuffled_copy}")

# Random permutation of indices
n = 10
random_indices = np.random.permutation(n)
print(f"Random permutation of {n} indices: {random_indices}")

# Bootstrap sampling (sampling with replacement)
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
bootstrap_samples = []
for _ in range(5):
    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
    bootstrap_samples.append(bootstrap_sample)

print(f"\nBootstrap samples:")
for i, sample in enumerate(bootstrap_samples):
    print(f"Sample {i+1}: {sample}")
```

### Random State Management

```python
# Managing random state for reproducibility
print("Random state management:")

# Save current state
state = np.random.get_state()
random_numbers_1 = np.random.random(3)
print(f"Random numbers 1: {random_numbers_1}")

# Restore state and generate same numbers
np.random.set_state(state)
random_numbers_2 = np.random.random(3)
print(f"Random numbers 2: {random_numbers_2}")
print(f"Same numbers: {np.array_equal(random_numbers_1, random_numbers_2)}")

# Using RandomState object for multiple independent streams
rng1 = np.random.RandomState(42)
rng2 = np.random.RandomState(123)

print(f"\nIndependent random streams:")
print(f"Stream 1: {rng1.random(3)}")
print(f"Stream 2: {rng2.random(3)}")
print(f"Stream 1 again: {rng1.random(3)}")
print(f"Stream 2 again: {rng2.random(3)}")

# New Generator interface (NumPy 1.17+)
try:
    rng_new = np.random.default_rng(42)  # New interface
    print(f"New generator: {rng_new.random(3)}")
except AttributeError:
    print("New Generator interface not available (NumPy < 1.17)")
```

## File I/O

### Saving and Loading Arrays

```python
# Create sample data
data = np.random.randn(100, 5)
labels = np.random.randint(0, 3, 100)

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Save single array
np.save('data.npy', data)
np.save('labels.npy', labels)

# Load single array
loaded_data = np.load('data.npy')
loaded_labels = np.load('labels.npy')

print(f"Loaded data shape: {loaded_data.shape}")
print(f"Arrays equal: {np.array_equal(data, loaded_data)}")

# Save multiple arrays in a single file
np.savez('dataset.npz', data=data, labels=labels, info="Sample dataset")

# Load multiple arrays
loaded = np.load('dataset.npz')
print(f"Available arrays: {list(loaded.keys())}")
print(f"Data from npz: {loaded['data'].shape}")
print(f"Labels from npz: {loaded['labels'].shape}")

# Compressed saving (useful for large arrays)
large_sparse_data = np.zeros((1000, 1000))
large_sparse_data[::10, ::10] = 1  # Sparse data

np.savez_compressed('sparse_data.npz', sparse=large_sparse_data)

import os
print(f"Uncompressed size: {os.path.getsize('data.npy')} bytes")
print(f"Compressed size: {os.path.getsize('sparse_data.npz')} bytes")
```

### Text File I/O

```python
# Save to text file
data_small = np.random.randn(5, 3)
np.savetxt('data.txt', data_small, delimiter=',', fmt='%.4f')
np.savetxt('data.csv', data_small, delimiter=',', fmt='%.4f', 
           header='col1,col2,col3', comments='')

print("Data saved to text files")

# Load from text file
loaded_from_txt = np.loadtxt('data.txt', delimiter=',')
print(f"Loaded from text file:\n{loaded_from_txt}")

# Load CSV with headers (skip first line)
try:
    loaded_from_csv = np.loadtxt('data.csv', delimiter=',', skiprows=1)
    print(f"Loaded from CSV:\n{loaded_from_csv}")
except:
    print("Error loading CSV (might need to handle headers differently)")

# More complex text file loading
# Create a sample file with mixed data types
sample_data = """# Sample data file
# Name, Age, Score
Alice, 25, 85.5
Bob, 30, 92.3
Charlie, 22, 78.9"""

with open('mixed_data.txt', 'w') as f:
    f.write(sample_data)

# Load with specific data types
dtype_spec = [('name', 'U10'), ('age', 'i4'), ('score', 'f4')]
mixed_data = np.loadtxt('mixed_data.txt', delimiter=',', skiprows=2, 
                       dtype=dtype_spec, converters={0: lambda s: s.strip()})

print(f"Mixed data types:\n{mixed_data}")
print(f"Names: {mixed_data['name']}")
print(f"Ages: {mixed_data['age']}")
print(f"Scores: {mixed_data['score']}")

# Using genfromtxt for more robust loading
robust_data = np.genfromtxt('mixed_data.txt', delimiter=',', skip_header=2,
                           names=['name', 'age', 'score'],
                           dtype=['U10', 'i4', 'f4'],
                           autostrip=True)

print(f"Robust loaded data:\n{robust_data}")
```

### Memory Mapping

```python
# Memory mapping for large files
# Create a large array and save it
large_data = np.random.randn(1000, 1000)
np.save('large_data.npy', large_data)

# Memory map the file (doesn't load into memory immediately)
mmap_data = np.load('large_data.npy', mmap_mode='r')  # Read-only
print(f"Memory mapped data shape: {mmap_data.shape}")
print(f"Data type: {type(mmap_data)}")

# Access parts of the data (only loads what's needed)
subset = mmap_data[:10, :10]
print(f"Subset shape: {subset.shape}")

# Writable memory map
mmap_writable = np.load('large_data.npy', mmap_mode='r+')
mmap_writable[0, 0] = 999.0  # Modifies the file directly

print(f"Modified value: {mmap_writable[0, 0]}")

# Creating memory mapped arrays from scratch
# This creates a file that can be larger than available RAM
filename = 'memmap_array.dat'
shape = (100, 100)
dtype = np.float32

# Create memory mapped array
memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
memmap_array[:] = np.random.randn(*shape).astype(dtype)

# Access the memory mapped array
print(f"Memory mapped array shape: {memmap_array.shape}")
print(f"Memory usage (estimated): {memmap_array.nbytes} bytes")

# Close and reopen
del memmap_array  # Close the memmap

# Reopen existing memmap
reopened_memmap = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
print(f"Reopened memmap mean: {np.mean(reopened_memmap):.4f}")
```

## Performance Optimization

### Vectorization vs Loops

```python
import time

# Compare loop vs vectorized operations
n = 1000000
x = np.random.randn(n)
y = np.random.randn(n)

# Python loop (slow)
start_time = time.time()
result_loop = []
for i in range(len(x)):
    result_loop.append(x[i] * y[i] + np.sin(x[i]))
result_loop = np.array(result_loop)
loop_time = time.time() - start_time

# Vectorized operations (fast)
start_time = time.time()
result_vectorized = x * y + np.sin(x)
vectorized_time = time.time() - start_time

print(f"Loop time: {loop_time:.4f} seconds")
print(f"Vectorized time: {vectorized_time:.4f} seconds")
print(f"Speedup: {loop_time/vectorized_time:.1f}x")
print(f"Results equal: {np.allclose(result_loop, result_vectorized)}")
```

### Memory Layout and Views

```python
# Array memory layout affects performance
# Row-major (C-style) vs Column-major (Fortran-style)

# Create arrays with different memory layouts
size = (1000, 1000)
c_array = np.random.randn(*size)  # Default: C-style (row-major)
f_array = np.asfortranarray(c_array)  # Fortran-style (column-major)

print(f"C-style flags: {c_array.flags['C_CONTIGUOUS']}")
print(f"Fortran-style flags: {f_array.flags['F_CONTIGUOUS']}")

# Row-wise operations (faster on C-style arrays)
start_time = time.time()
row_sum_c = np.sum(c_array, axis=1)
c_row_time = time.time() - start_time

start_time = time.time()
row_sum_f = np.sum(f_array, axis=1)
f_row_time = time.time() - start_time

# Column-wise operations (faster on Fortran-style arrays)
start_time = time.time()
col_sum_c = np.sum(c_array, axis=0)
c_col_time = time.time() - start_time

start_time = time.time()
col_sum_f = np.sum(f_array, axis=0)
f_col_time = time.time() - start_time

print(f"\nRow operations:")
print(f"C-style time: {c_row_time:.4f}s")
print(f"F-style time: {f_row_time:.4f}s")

print(f"\nColumn operations:")
print(f"C-style time: {c_col_time:.4f}s")
print(f"F-style time: {f_col_time:.4f}s")

# Views vs copies
original = np.arange(1000000)

# Creating a view (shares memory)
start_time = time.time()
view = original[::2]  # Every second element
view_time = time.time() - start_time

# Creating a copy
start_time = time.time()
copy = original[::2].copy()
copy_time = time.time() - start_time

print(f"\nView creation time: {view_time:.6f}s")
print(f"Copy creation time: {copy_time:.6f}s")
print(f"View shares memory: {np.shares_memory(original, view)}")
print(f"Copy shares memory: {np.shares_memory(original, copy)}")
```

### Optimizing Common Operations

```python
# Efficient ways to perform common operations

# 1. Finding elements efficiently
large_array = np.random.randint(0, 1000, 100000)

# Method 1: Using boolean indexing (memory intensive for large arrays)
start_time = time.time()
indices_bool = np.where(large_array > 500)[0]
bool_time = time.time() - start_time

# Method 2: Using argwhere (more memory efficient)
start_time = time.time()
indices_arg = np.argwhere(large_array > 500).flatten()
arg_time = time.time() - start_time

print(f"Boolean indexing time: {bool_time:.4f}s")
print(f"Argwhere time: {arg_time:.4f}s")
print(f"Same results: {np.array_equal(indices_bool, indices_arg)}")

# 2. Efficient sorting and searching
data = np.random.randn(100000)

# Partial sorting (when you only need top-k elements)
k = 100
start_time = time.time()
full_sort = np.sort(data)[-k:]  # Full sort, take last k
full_sort_time = time.time() - start_time

start_time = time.time()
partial_sort = np.partition(data, -k)[-k:]  # Partial sort
partial_sort_time = time.time() - start_time

print(f"\nFull sort time: {full_sort_time:.4f}s")
print(f"Partial sort time: {partial_sort_time:.4f}s")
print(f"Speedup: {full_sort_time/partial_sort_time:.1f}x")

# 3. Efficient aggregations
matrix = np.random.randn(1000, 1000)

# Computing multiple statistics efficiently
start_time = time.time()
mean_separate = np.mean(matrix)
std_separate = np.std(matrix)
separate_time = time.time() - start_time

start_time = time.time()
# More efficient: compute mean once and use it for std
mean_efficient = np.mean(matrix)
std_efficient = np.sqrt(np.mean((matrix - mean_efficient)**2))
efficient_time = time.time() - start_time

print(f"\nSeparate computations time: {separate_time:.4f}s")
print(f"Efficient computation time: {efficient_time:.4f}s")
print(f"Results match: {np.isclose(std_separate, std_efficient)}")
```

### Memory Management

```python
# Memory usage optimization
import sys

def get_memory_usage(array):
    """Get memory usage of array in MB."""
    return array.nbytes / (1024 * 1024)

# Data type optimization
# Using smaller data types when possible
large_size = (10000, 1000)

# Default float64
float64_array = np.random.randn(*large_size)
print(f"Float64 memory: {get_memory_usage(float64_array):.2f} MB")

# Float32 (half the memory, usually sufficient precision)
float32_array = np.random.randn(*large_size).astype(np.float32)
print(f"Float32 memory: {get_memory_usage(float32_array):.2f} MB")

# Integer data
int64_array = np.random.randint(0, 1000, large_size, dtype=np.int64)
int32_array = np.random.randint(0, 1000, large_size, dtype=np.int32)
int16_array = np.random.randint(0, 1000, large_size, dtype=np.int16)

print(f"Int64 memory: {get_memory_usage(int64_array):.2f} MB")
print(f"Int32 memory: {get_memory_usage(int32_array):.2f} MB")
print(f"Int16 memory: {get_memory_usage(int16_array):.2f} MB")

# Memory mapping for very large datasets
# Create a function to process data in chunks
def process_large_dataset(filename, chunk_size=1000):
    """Process large dataset in chunks to manage memory."""
    # Assume we have a large dataset saved as memmap
    total_size = (100000, 100)
    
    # Create sample data
    memmap_data = np.memmap(filename, dtype=np.float32, mode='w+', shape=total_size)
    memmap_data[:] = np.random.randn(*total_size).astype(np.float32)
    
    # Process in chunks
    results = []
    for i in range(0, total_size[0], chunk_size):
        chunk = memmap_data[i:i+chunk_size]
        # Process chunk (e.g., compute mean)
        chunk_result = np.mean(chunk, axis=1)
        results.append(chunk_result)
    
    # Combine results
    final_result = np.concatenate(results)
    return final_result

# Example usage
result = process_large_dataset('large_dataset.dat')
print(f"\nProcessed large dataset, result shape: {result.shape}")

# Clean up temporary files
import os
temp_files = ['data.npy', 'labels.npy', 'dataset.npz', 'sparse_data.npz', 
              'data.txt', 'data.csv', 'mixed_data.txt', 'large_data.npy',
              'memmap_array.dat', 'large_dataset.dat']
for file in temp_files:
    try:
        os.remove(file)
    except FileNotFoundError:
        pass
```

## Real-World Examples

### Data Analysis Pipeline

```python
# Complete data analysis pipeline using NumPy

# 1. Data Generation (simulating real dataset)
np.random.seed(42)
n_samples = 10000
n_features = 5

# Generate synthetic dataset with known relationships
X = np.random.randn(n_samples, n_features)
# Create target variable with some noise
true_coefficients = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
y = X @ true_coefficients + np.random.randn(n_samples) * 0.1

print("Dataset created:")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"True coefficients: {true_coefficients}")

# 2. Data Exploration
print(f"\nData exploration:")
print(f"Feature means: {np.mean(X, axis=0)}")
print(f"Feature stds: {np.std(X, axis=0)}")
print(f"Target mean: {np.mean(y):.4f}")
print(f"Target std: {np.std(y):.4f}")

# Correlation analysis
correlation_matrix = np.corrcoef(X.T)
print(f"Feature correlation matrix:\n{correlation_matrix}")

# 3. Data Preprocessing
# Standardization (z-score normalization)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_standardized = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y_standardized = (y - y_mean) / y_std

print(f"\nAfter standardization:")
print(f"X mean: {np.mean(X_standardized, axis=0)}")
print(f"X std: {np.std(X_standardized, axis=0)}")

# 4. Train-Test Split
def train_test_split(X, y, test_size=0.2, random_state=None):
    """Simple train-test split implementation."""
    if random_state:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Random indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y_standardized, 
                                                   test_size=0.2, random_state=42)
print(f"\nTrain-test split:")
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 5. Linear Regression Implementation
def linear_regression_normal_equation(X, y):
    """Solve linear regression using normal equation."""
    # Add bias term (intercept)
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    
    # Normal equation: θ = (X^T X)^(-1) X^T y
    XTX = X_with_bias.T @ X_with_bias
    XTy = X_with_bias.T @ y
    theta = np.linalg.solve(XTX, XTy)
    
    return theta

# Train the model
coefficients = linear_regression_normal_equation(X_train, y_train)
print(f"\nLearned coefficients:")
print(f"Intercept: {coefficients[0]:.4f}")
print(f"Coefficients: {coefficients[1:]}")
print(f"True coefficients: {true_coefficients}")

# 6. Model Evaluation
def predict(X, coefficients):
    """Make predictions using linear regression model."""
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    return X_with_bias @ coefficients

def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def r_squared(y_true, y_pred):
    """Calculate R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Evaluate on train and test sets
y_train_pred = predict(X_train, coefficients)
y_test_pred = predict(X_test, coefficients)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r_squared(y_train, y_train_pred)
test_r2 = r_squared(y_test, y_test_pred)

print(f"\nModel Evaluation:")
print(f"Train MSE: {train_mse:.6f}")
print(f"Test MSE: {test_mse:.6f}")
print(f"Train R²: {train_r2:.6f}")
print(f"Test R²: {test_r2:.6f}")
```

### Image Processing Example

```python
# Image processing using NumPy (simulating image operations)

# Create a synthetic image
def create_synthetic_image(width=100, height=100):
    """Create a synthetic grayscale image."""
    x = np.linspace(-5, 5, width)
    y = np.linspace(-5, 5, height)
    X, Y = np.meshgrid(x, y)
    
    # Create pattern: combination of sine waves and gaussian
    pattern = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-(X**2 + Y**2)/10)
    
    # Normalize to 0-255 range
    image = ((pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255).astype(np.uint8)
    return image

# Create synthetic image
original_image = create_synthetic_image(200, 200)
print(f"Image shape: {original_image.shape}")
print(f"Image dtype: {original_image.dtype}")
print(f"Image range: [{original_image.min()}, {original_image.max()}]")

# Basic image operations
def apply_gaussian_blur(image, kernel_size=5):
    """Apply simple gaussian blur using convolution."""
    # Create gaussian kernel
    sigma = kernel_size / 3.0
    k = kernel_size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2*sigma**2))
    kernel = kernel / kernel.sum()
    
    # Apply convolution (simplified - real implementation would handle edges better)
    blurred = np.zeros_like(image)
    h, w = image.shape
    
    for i in range(k, h-k):
        for j in range(k, w-k):
            blurred[i, j] = np.sum(image[i-k:i+k+1, j-k:j+k+1] * kernel)
    
    return blurred.astype(np.uint8)

def apply_edge_detection(image):
    """Apply simple edge detection using Sobel operators."""
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    h, w = image.shape
    edges = np.zeros_like(image)
    
    # Apply Sobel operators
    for i in range(1, h-1):
        for j in range(1, w-1):
            gx = np.sum(image[i-1:i+2, j-1:j+2] * sobel_x)
            gy = np.sum(image[i-1:i+2, j-1:j+2] * sobel_y)
            edges[i, j] = min(255, np.sqrt(gx**2 + gy**2))
    
    return edges.astype(np.uint8)

# Apply image processing operations
blurred_image = apply_gaussian_blur(original_image)
edge_image = apply_edge_detection(original_image)

print(f"\nImage processing completed:")
print(f"Original image stats: mean={np.mean(original_image):.1f}, std={np.std(original_image):.1f}")
print(f"Blurred image stats: mean={np.mean(blurred_image):.1f}, std={np.std(blurred_image):.1f}")
print(f"Edge image stats: mean={np.mean(edge_image):.1f}, std={np.std(edge_image):.1f}")

# Histogram analysis
def compute_histogram(image, bins=256):
    """Compute image histogram."""
    hist = np.zeros(bins)
    for i in range(bins):
        hist[i] = np.sum(image == i)
    return hist

original_hist = compute_histogram(original_image)
print(f"Histogram computed - max count: {np.max(original_hist)}")
```

### Financial Analysis Example

```python
# Financial data analysis using NumPy

# Generate synthetic stock price data
def generate_stock_prices(initial_price=100, n_days=252, volatility=0.2, drift=0.05):
    """Generate synthetic stock prices using geometric Brownian motion."""
    dt = 1/252  # Daily time step (assuming 252 trading days per year)
    
    # Generate random returns
    random_returns = np.random.normal(
        (drift - 0.5 * volatility**2) * dt,
        volatility * np.sqrt(dt),
        n_days
    )
    
    # Calculate cumulative returns and prices
    log_returns = np.cumsum(random_returns)
    prices = initial_price * np.exp(log_returns)
    
    # Prepend initial price
    prices = np.concatenate([[initial_price], prices])
    
    return prices

# Generate data for multiple stocks
np.random.seed(42)
n_stocks = 5
n_days = 252
stock_names = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E']

# Generate price data
price_data = np.zeros((n_days + 1, n_stocks))
for i in range(n_stocks):
    price_data[:, i] = generate_stock_prices(
        initial_price=np.random.uniform(50, 150),
        volatility=np.random.uniform(0.15, 0.35),
        drift=np.random.uniform(0.02, 0.12)
    )

print("Stock price data generated:")
print(f"Data shape: {price_data.shape}")
print(f"Price ranges:")
for i, name in enumerate(stock_names):
    print(f"{name}: ${price_data[0, i]:.2f} -> ${price_data[-1, i]:.2f}")

# Calculate returns
def calculate_returns(prices):
    """Calculate daily returns from price data."""
    return (prices[1:] - prices[:-1]) / prices[:-1]

daily_returns = calculate_returns(price_data)
print(f"\nDaily returns shape: {daily_returns.shape}")

# Risk and return analysis
mean_returns = np.mean(daily_returns, axis=0) * 252  # Annualized
volatilities = np.std(daily_returns, axis=0) * np.sqrt(252)  # Annualized
sharpe_ratios = mean_returns / volatilities

print(f"\nAnnualized Statistics:")
for i, name in enumerate(stock_names):
    print(f"{name}: Return={mean_returns[i]:.2%}, Vol={volatilities[i]:.2%}, Sharpe={sharpe_ratios[i]:.2f}")

# Correlation analysis
correlation_matrix = np.corrcoef(daily_returns.T)
print(f"\nCorrelation Matrix:")
print("     ", " ".join(f"{name:>8}" for name in stock_names))
for i, name in enumerate(stock_names):
    correlations = " ".join(f"{correlation_matrix[i, j]:8.3f}" for j in range(n_stocks))
    print(f"{name}: {correlations}")

# Portfolio optimization (equal weight portfolio)
equal_weights = np.ones(n_stocks) / n_stocks
portfolio_returns = daily_returns @ equal_weights
portfolio_mean_return = np.mean(portfolio_returns) * 252
portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
portfolio_sharpe = portfolio_mean_return / portfolio_volatility

print(f"\nEqual Weight Portfolio:")
print(f"Expected Return: {portfolio_mean_return:.2%}")
print(f"Volatility: {portfolio_volatility:.2%}")
print(f"Sharpe Ratio: {portfolio_sharpe:.2f}")

# Value at Risk (VaR) calculation
def calculate_var(returns, confidence_level=0.05):
    """Calculate Value at Risk."""
    return np.percentile(returns, confidence_level * 100)

portfolio_var_95 = calculate_var(portfolio_returns, 0.05)
portfolio_var_99 = calculate_var(portfolio_returns, 0.01)

print(f"\nValue at Risk (VaR):")
print(f"95% VaR: {portfolio_var_95:.2%} (daily)")
print(f"99% VaR: {portfolio_var_99:.2%} (daily)")

# Maximum Drawdown calculation
def calculate_max_drawdown(prices):
    """Calculate maximum drawdown."""
    cumulative_returns = np.cumprod(1 + calculate_returns(prices))
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return np.min(drawdown)

portfolio_prices = price_data @ equal_weights
max_drawdown = calculate_max_drawdown(portfolio_prices)
print(f"Maximum Drawdown: {max_drawdown:.2%}")
```

### Scientific Computing Example

```python
# Scientific computing: Solving differential equations numerically

# Example: Population dynamics (Lotka-Volterra predator-prey model)
def lotka_volterra(state, t, alpha, beta, gamma, delta):
    """
    Lotka-Volterra predator-prey model.
    state = [prey_population, predator_population]
    """
    prey, predator = state
    
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    
    return np.array([dprey_dt, dpredator_dt])

# Implement simple Euler method for ODE solving
def euler_method(func, y0, t_span, args=(), n_steps=1000):
    """
    Solve ODE using Euler method.
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    
    t_values = np.linspace(t_start, t_end, n_steps + 1)
    y_values = np.zeros((n_steps + 1, len(y0)))
    y_values[0] = y0
    
    for i in range(n_steps):
        y_current = y_values[i]
        dy_dt = func(y_current, t_values[i], *args)
        y_values[i + 1] = y_current + dt * dy_dt
    
    return t_values, y_values

# Parameters for the model
alpha = 1.0   # prey growth rate
beta = 0.5    # predation rate
gamma = 0.75  # predator death rate
delta = 0.25  # predator efficiency

# Initial conditions
initial_prey = 10
initial_predator = 5
y0 = np.array([initial_prey, initial_predator])

# Solve the system
t_span = (0, 20)  # Time from 0 to 20
t_values, solution = euler_method(
    lotka_volterra, y0, t_span, 
    args=(alpha, beta, gamma, delta),
    n_steps=2000
)

prey_population = solution[:, 0]
predator_population = solution[:, 1]

print("Predator-Prey Model Results:")
print(f"Time span: {t_span[0]} to {t_span[1]}")
print(f"Initial prey population: {initial_prey}")
print(f"Initial predator population: {initial_predator}")
print(f"Final prey population: {prey_population[-1]:.2f}")
print(f"Final predator population: {predator_population[-1]:.2f}")

# Analyze oscillatory behavior
def find_peaks(data, min_distance=50):
    """Simple peak finding algorithm."""
    peaks = []
    for i in range(min_distance, len(data) - min_distance):
        if (data[i] > data[i-min_distance:i]).all() and (data[i] > data[i+1:i+min_distance+1]).all():
            peaks.append(i)
    return np.array(peaks)

prey_peaks = find_peaks(prey_population)
predator_peaks = find_peaks(predator_population)

if len(prey_peaks) > 1:
    prey_period = np.mean(np.diff(t_values[prey_peaks]))
    print(f"Approximate prey oscillation period: {prey_period:.2f}")

if len(predator_peaks) > 1:
    predator_period = np.mean(np.diff(t_values[predator_peaks]))
    print(f"Approximate predator oscillation period: {predator_period:.2f}")

# Phase space analysis
print(f"\nPhase space analysis:")
print(f"Prey population range: [{np.min(prey_population):.2f}, {np.max(prey_population):.2f}]")
print(f"Predator population range: [{np.min(predator_population):.2f}, {np.max(predator_population):.2f}]")

# Calculate conservation quantity (approximate for Lotka-Volterra)
# H = delta * prey - gamma * ln(prey) + beta * predator - alpha * ln(predator)
def conservation_quantity(prey, pred, alpha, beta, gamma, delta):
    """Calculate conservation quantity for Lotka-Volterra system."""
    return (delta * prey - gamma * np.log(prey) + 
            beta * pred - alpha * np.log(pred))

H_values = conservation_quantity(prey_population, predator_population, 
                               alpha, beta, gamma, delta)
H_variation = (np.max(H_values) - np.min(H_values)) / np.mean(H_values)
print(f"Conservation quantity variation: {H_variation:.6f} (should be ~0 for exact solution)")
```

## Best Practices

### Code Organization and Style

```python
# Good practices for NumPy code organization

import numpy as np
from typing import Tuple, Optional, Union

# 1. Use type hints for better code documentation
def standardize_data(data: np.ndarray, 
                    axis: Optional[int] = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize data to zero mean and unit variance.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data to standardize
    axis : int, optional
        Axis along which to compute mean and std (default: 0)
    
    Returns:
    --------
    standardized_data : np.ndarray
        Standardized data
    mean : np.ndarray
        Mean values used for standardization
    std : np.ndarray
        Standard deviation values used for standardization
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    standardized_data = (data - mean) / std
    
    return standardized_data, mean.squeeze(), std.squeeze()

# 2. Input validation and error handling
def safe_divide(numerator: np.ndarray, 
               denominator: np.ndarray, 
               fill_value: float = np.inf) -> np.ndarray:
    """
    Safely divide arrays, handling division by zero.
    
    Parameters:
    -----------
    numerator : np.ndarray
        Numerator array
    denominator : np.ndarray
        Denominator array
    fill_value : float
        Value to use when denominator is zero
    
    Returns:
    --------
    result : np.ndarray
        Result of division with safe handling of zeros
    
    Raises:
    -------
    ValueError
        If arrays are not broadcastable
    """
    numerator = np.asarray(numerator)
    denominator = np.asarray(denominator)
    
    # Check if arrays are broadcastable
    try:
        np.broadcast_arrays(numerator, denominator)
    except ValueError as e:
        raise ValueError(f"Arrays are not broadcastable: {e}")
    
    # Perform safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        result = np.where(denominator == 0, fill_value, result)
    
    return result

# 3. Vectorized operations over loops
def calculate_distances_vectorized(points1: np.ndarray, 
                                 points2: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise distances between two sets of points (vectorized).
    
    Parameters:
    -----------
    points1 : np.ndarray, shape (n, d)
        First set of points
    points2 : np.ndarray, shape (m, d)
        Second set of points
    
    Returns:
    --------
    distances : np.ndarray, shape (n, m)
        Pairwise distances
    """
    # Use broadcasting to compute all pairwise differences
    diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    return distances

# Example usage
if __name__ == "__main__":
    # Generate sample data
    sample_data = np.random.randn(1000, 5) * 10 + 5
    
    # Test standardization
    std_data, data_mean, data_std = standardize_data(sample_data)
    print(f"Original data mean: {np.mean(sample_data, axis=0)}")
    print(f"Standardized data mean: {np.mean(std_data, axis=0)}")
    print(f"Standardized data std: {np.std(std_data, axis=0)}")
    
    # Test safe division
    a = np.array([1, 2, 3, 4])
    b = np.array([2, 0, 1, 0])
    result = safe_divide(a, b, fill_value=999)
    print(f"Safe division result: {result}")
    
    # Test distance calculation
    points_a = np.random.randn(5, 2)
    points_b = np.random.randn(3, 2)
    distances = calculate_distances_vectorized(points_a, points_b)
    print(f"Distance matrix shape: {distances.shape}")
```

### Performance Best Practices

```python
# Performance optimization guidelines

# 1. Prefer in-place operations when possible
def normalize_inplace(array: np.ndarray) -> None:
    """Normalize array in-place to save memory."""
    array -= np.mean(array)
    array /= np.std(array)

# 2. Use appropriate data types
def optimize_data_types(data: np.ndarray) -> np.ndarray:
    """Optimize data types based on value ranges."""
    if np.issubdtype(data.dtype, np.integer):
        # Check if we can use smaller integer types
        min_val, max_val = np.min(data), np.max(data)
        
        if min_val >= 0:  # Unsigned integers
            if max_val < 256:
                return data.astype(np.uint8)
            elif max_val < 65536:
                return data.astype(np.uint16)
            elif max_val < 4294967296:
                return data.astype(np.uint32)
        else:  # Signed integers
            if -128 <= min_val and max_val < 128:
                return data.astype(np.int8)
            elif -32768 <= min_val and max_val < 32768:
                return data.astype(np.int16)
            elif -2147483648 <= min_val and max_val < 2147483648:
                return data.astype(np.int32)
    
    elif np.issubdtype(data.dtype, np.floating):
        # For floats, consider using float32 instead of float64 if precision allows
        if data.dtype == np.float64:
            float32_version = data.astype(np.float32)
            if np.allclose(data, float32_version, rtol=1e-6):
                return float32_version
    
    return data

# 3. Efficient array initialization
def create_arrays_efficiently():
    """Examples of efficient array creation."""
    n = 10000
    
    # Efficient: Pre-allocate and fill
    result = np.empty(n)
    result[:] = 5.0  # Fill with value
    
    # Efficient: Use built-in functions
    zeros_array = np.zeros(n)
    ones_array = np.ones(n)
    full_array = np.full(n, 3.14)
    
    # Less efficient: Growing arrays in loops
    # Don't do this:
    # result = np.array([])
    # for i in range(n):
    #     result = np.append(result, i)  # Creates new array each time!
    
    return result, zeros_array, ones_array, full_array

# 4. Memory-efficient operations
def process_large_array_efficiently(large_array: np.ndarray, 
                                  chunk_size: int = 1000) -> np.ndarray:
    """Process large array in chunks to manage memory usage."""
    result = np.empty_like(large_array)
    
    for i in range(0, len(large_array), chunk_size):
        chunk = large_array[i:i+chunk_size]
        # Process chunk (example: complex mathematical operation)
        result[i:i+chunk_size] = np.sin(chunk) * np.exp(-chunk**2)
    
    return result

# 5. Efficient boolean operations
def efficient_boolean_operations(data: np.ndarray) -> np.ndarray:
    """Examples of efficient boolean operations."""
    
    # Use numpy functions instead of Python's any/all on arrays
    # Efficient
    has_negative = np.any(data < 0)
    all_positive = np.all(data > 0)
    
    # Less efficient for large arrays
    # has_negative = any(x < 0 for x in data)
    
    # Use boolean indexing efficiently
    # Find elements in multiple ranges efficiently
    mask = (data > 10) & (data < 100) | (data > 1000)
    filtered_data = data[mask]
    
    # Count occurrences efficiently
    count_in_range = np.sum((data >= 50) & (data <= 150))
    
    return filtered_data

print("NumPy best practices examples created successfully!")
```

### Common Pitfalls and Solutions

```python
# Common NumPy pitfalls and how to avoid them

print("Common NumPy Pitfalls and Solutions:")

# 1. Views vs Copies confusion
print("\n1. Views vs Copies:")
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]        # This is a view
copy = original[1:4].copy() # This is a copy

view[0] = 999
print(f"After modifying view: original = {original}")  # Original is changed!

copy[0] = 888
print(f"After modifying copy: original = {original}")  # Original unchanged

# Solution: Explicitly copy when you need independence
independent_array = original[1:4].copy()

# 2. Broadcasting surprises
print("\n2. Broadcasting Issues:")
a = np.array([[1, 2, 3]])      # Shape: (1, 3)
b = np.array([[4], [5], [6]])  # Shape: (3, 1)
result = a + b                  # Shape: (3, 3) - might be unexpected!
print(f"a.shape: {a.shape}, b.shape: {b.shape}")
print(f"(a + b).shape: {result.shape}")
print(f"Result:\n{result}")

# Solution: Check shapes before operations or use explicit broadcasting
print(f"Explicit check: Can broadcast? {np.broadcast_shapes(a.shape, b.shape)}")

# 3. Integer division vs float division
print("\n3. Division Issues:")
int_array = np.array([1, 2, 3], dtype=int)
print(f"Integer array / 2: {int_array / 2}")      # This gives floats (good)
print(f"Integer array // 2: {int_array // 2}")   # Floor division (might be intended)

# Be explicit about desired behavior
float_result = int_array.astype(float) / 2
int_result = int_array // 2

# 4. Mutable default arguments with NumPy
print("\n4. Mutable Default Arguments:")

# Bad practice
def bad_function(arr=np.array([1, 2, 3])):
    arr[0] += 1
    return arr

result1 = bad_function()
result2 = bad_function()  # Reuses same array!
print(f"Second call result: {result2}")  # [3, 2, 3] instead of [2, 2, 3]

# Good practice
def good_function(arr=None):
    if arr is None:
        arr = np.array([1, 2, 3])
    else:
        arr = np.array(arr)  # Make a copy
    arr[0] += 1
    return arr

# 5. Inefficient element-wise operations
print("\n5. Inefficient Operations:")

# Inefficient: Element-wise operations in Python loops
large_array = np.random.randn(10000)

# Don't do this
# result = []
# for x in large_array:
#     result.append(x**2 + np.sin(x))
# result = np.array(result)

# Do this instead
efficient_result = large_array**2 + np.sin(large_array)

# 6. Memory issues with large arrays
print("\n6. Memory Management:")

def demonstrate_memory_efficiency():
    """Show memory-efficient practices."""
    # Instead of creating many intermediate arrays
    # x = np.random.randn(10000)
    # temp1 = x**2
    # temp2 = np.sin(temp1)
    # result = temp2 + x
    
    # Do operations in-place or in fewer steps
    x = np.random.randn(10000)
    np.power(x, 2, out=x)  # In-place squaring
    result = np.sin(x) + np.sqrt(x)  # Fewer intermediate arrays
    
    return result

# 7. Floating point precision issues
print("\n7. Floating Point Precision:")
a = 0.1 + 0.2
b = 0.3
print(f"0.1 + 0.2 == 0.3: {a == b}")  # False!
print(f"0.1 + 0.2 = {a}")
print(f"0.3 = {b}")

# Solution: Use numpy.isclose for floating point comparisons
print(f"Using np.isclose: {np.isclose(a, b)}")  # True

# For arrays
arr1 = np.array([0.1 + 0.2, 0.4 + 0.5])
arr2 = np.array([0.3, 0.9])
print(f"Array equality with tolerance: {np.allclose(arr1, arr2)}")

# 8. Axis confusion in reductions
print("\n8. Axis Confusion:")
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"Matrix shape: {matrix.shape}")
print(f"Sum axis=0 (columns): {np.sum(matrix, axis=0)}")  # Sum each column
print(f"Sum axis=1 (rows): {np.sum(matrix, axis=1)}")     # Sum each row

# Remember: axis=0 operates along rows (reducing row dimension)
#          axis=1 operates along columns (reducing column dimension)

print("\nBest practices applied successfully!")
```

## Summary

This comprehensive NumPy guide covers:

- **Installation and Setup**: Getting started with NumPy
- **Array Fundamentals**: Creation, properties, and basic operations  
- **Advanced Operations**: Broadcasting, linear algebra, statistics
- **Performance Optimization**: Vectorization, memory management, efficient coding
- **Real-World Applications**: Data analysis, image processing, financial modeling, scientific computing
- **Best Practices**: Code organization, common pitfalls, and solutions

NumPy is the foundation of the Python scientific computing ecosystem. Master these concepts and you'll be well-equipped to tackle complex numerical problems efficiently.

Key takeaways:
- **Always vectorize** operations instead of using Python loops
- **Understand broadcasting** to work with arrays of different shapes
- **Use appropriate data types** to optimize memory usage
- **Be aware of views vs copies** to avoid unexpected behavior
- **Leverage NumPy's extensive function library** for mathematical operations

For further learning, explore SciPy, Pandas, Matplotlib, and scikit-learn, which all build upon NumPy's foundation.