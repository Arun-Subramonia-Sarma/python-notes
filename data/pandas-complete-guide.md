# Pandas - Complete Guide

A comprehensive guide to Pandas, the powerful Python library for data manipulation and analysis.

## Table of Contents

- [Installation and Setup](#installation-and-setup)
- [Pandas Fundamentals](#pandas-fundamentals)
- [Series Operations](#series-operations)
- [DataFrame Operations](#dataframe-operations)
- [Data Loading and Saving](#data-loading-and-saving)
- [Data Inspection and Exploration](#data-inspection-and-exploration)
- [Data Cleaning](#data-cleaning)
- [Data Transformation](#data-transformation)
- [Grouping and Aggregation](#grouping-and-aggregation)
- [Merging and Joining](#merging-and-joining)
- [Time Series Analysis](#time-series-analysis)
- [Advanced Operations](#advanced-operations)
- [Performance Optimization](#performance-optimization)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)

## Installation and Setup

### Installation

```bash
# Using pip
pip install pandas

# Using UV (recommended for projects)
uv add pandas

# With optional dependencies
uv add "pandas[all]"  # Includes Excel, HDF5, and other optional dependencies

# Common data science stack
uv add pandas numpy matplotlib seaborn jupyter

# Check installation
python -c "import pandas as pd; print(pd.__version__)"
```

### Basic Import and Setup

```python
import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings

# Display settings
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', 20)       # Limit rows displayed
pd.set_option('display.precision', 2)       # Decimal precision
pd.set_option('display.width', None)        # Auto-detect terminal width

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
```

## Pandas Fundamentals

### Core Data Structures

Pandas has two primary data structures:

1. **Series**: One-dimensional labeled array
2. **DataFrame**: Two-dimensional labeled data structure (like a table)

```python
# Series creation
series_from_list = pd.Series([1, 2, 3, 4, 5])
series_with_index = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
series_from_dict = pd.Series({'a': 1, 'b': 2, 'c': 3})

print("Series from list:")
print(series_from_list)
print("\nSeries with custom index:")
print(series_with_index)
print("\nSeries from dictionary:")
print(series_from_dict)

# DataFrame creation
# From dictionary
data_dict = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Paris', 'Tokyo'],
    'Salary': [70000, 80000, 90000, 85000]
}
df_from_dict = pd.DataFrame(data_dict)

# From list of dictionaries
data_list = [
    {'Name': 'Alice', 'Age': 25, 'City': 'New York'},
    {'Name': 'Bob', 'Age': 30, 'City': 'London'},
    {'Name': 'Charlie', 'Age': 35, 'City': 'Paris'}
]
df_from_list = pd.DataFrame(data_list)

# From NumPy array
np_data = np.random.randn(4, 3)
df_from_numpy = pd.DataFrame(np_data, 
                           columns=['Col1', 'Col2', 'Col3'],
                           index=['Row1', 'Row2', 'Row3', 'Row4'])

print("\nDataFrame from dictionary:")
print(df_from_dict)
print("\nDataFrame from NumPy array:")
print(df_from_numpy.head())
```

### Index and Columns

```python
# Working with Index
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
})

print("Original DataFrame:")
print(df)
print(f"Index: {df.index.tolist()}")
print(f"Columns: {df.columns.tolist()}")

# Setting custom index
df_custom_index = df.set_index('A')
print("\nDataFrame with column 'A' as index:")
print(df_custom_index)

# Resetting index
df_reset = df_custom_index.reset_index()
print("\nDataFrame after resetting index:")
print(df_reset)

# Multi-level index (MultiIndex)
arrays = [['A', 'A', 'B', 'B'], ['X', 'Y', 'X', 'Y']]
tuples = list(zip(*arrays))
multi_index = pd.MultiIndex.from_tuples(tuples, names=['First', 'Second'])
df_multi = pd.DataFrame(np.random.randn(4, 3), 
                       index=multi_index, 
                       columns=['Col1', 'Col2', 'Col3'])
print("\nDataFrame with MultiIndex:")
print(df_multi)

# Renaming columns and index
df_renamed = df.rename(columns={'A': 'Alpha', 'B': 'Beta', 'C': 'Gamma'})
df_renamed.index = ['First', 'Second', 'Third', 'Fourth']
print("\nRenamed DataFrame:")
print(df_renamed)
```

## Series Operations

### Creating and Manipulating Series

```python
# Creating Series with different data types
numeric_series = pd.Series([1, 2, 3, 4, 5], name='Numbers')
string_series = pd.Series(['apple', 'banana', 'cherry', 'date'], name='Fruits')
boolean_series = pd.Series([True, False, True, False], name='Flags')
datetime_series = pd.Series(pd.date_range('2024-01-01', periods=4), name='Dates')

print("Numeric Series:")
print(numeric_series)
print(f"Data type: {numeric_series.dtype}")
print(f"Shape: {numeric_series.shape}")
print(f"Size: {numeric_series.size}")

# Series indexing and slicing
print(f"\nFirst element: {numeric_series[0]}")
print(f"Last element: {numeric_series.iloc[-1]}")
print(f"Elements 1-3: \n{numeric_series[1:4]}")
print(f"Elements at positions [0, 2, 4]: \n{numeric_series.iloc[[0, 2, 4]]}")

# Boolean indexing
print(f"Elements > 3: \n{numeric_series[numeric_series > 3]}")

# Series with custom index
custom_series = pd.Series([10, 20, 30, 40], 
                         index=['a', 'b', 'c', 'd'], 
                         name='CustomSeries')
print(f"\nCustom indexed series:\n{custom_series}")
print(f"Element 'b': {custom_series['b']}")
print(f"Elements 'a' and 'c': \n{custom_series[['a', 'c']]}")
```

### Series Methods and Operations

```python
# Mathematical operations
numbers = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='Numbers')

print("Mathematical operations:")
print(f"Sum: {numbers.sum()}")
print(f"Mean: {numbers.mean()}")
print(f"Median: {numbers.median()}")
print(f"Standard deviation: {numbers.std()}")
print(f"Min: {numbers.min()}")
print(f"Max: {numbers.max()}")

# Statistical methods
print(f"\nDescriptive statistics:")
print(numbers.describe())

# Value counts and unique values
fruits = pd.Series(['apple', 'banana', 'apple', 'cherry', 'banana', 'apple'])
print(f"\nFruit series: {fruits.tolist()}")
print(f"Value counts:\n{fruits.value_counts()}")
print(f"Unique values: {fruits.unique()}")
print(f"Number of unique values: {fruits.nunique()}")

# String operations
print(f"Uppercase fruits:\n{fruits.str.upper()}")
print(f"Fruits containing 'a':\n{fruits[fruits.str.contains('a')]}")
print(f"Fruit lengths:\n{fruits.str.len()}")

# Missing values
series_with_nan = pd.Series([1, 2, np.nan, 4, 5, np.nan])
print(f"\nSeries with NaN: {series_with_nan.tolist()}")
print(f"Is null: {series_with_nan.isnull().tolist()}")
print(f"Not null: {series_with_nan.notnull().tolist()}")
print(f"Drop NaN: {series_with_nan.dropna().tolist()}")
print(f"Fill NaN with 0: {series_with_nan.fillna(0).tolist()}")

# Sorting
unsorted_series = pd.Series([3, 1, 4, 1, 5, 9, 2, 6], 
                           index=['h', 'a', 'd', 'b', 'e', 'i', 'c', 'f'])
print(f"\nUnsorted series:\n{unsorted_series}")
print(f"Sort by values:\n{unsorted_series.sort_values()}")
print(f"Sort by index:\n{unsorted_series.sort_index()}")
```

## DataFrame Operations

### Creating and Basic Operations

```python
# Create a sample DataFrame
np.random.seed(42)
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'Age': [25, 30, 35, 28, 32, 29],
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'Finance'],
    'Salary': [50000, 80000, 90000, 75000, 60000, 85000],
    'Years_Experience': [2, 5, 8, 4, 6, 7],
    'Performance_Score': [8.5, 9.2, 7.8, 8.8, 9.0, 8.3]
})

print("Sample DataFrame:")
print(df)
print(f"\nDataFrame shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}")
```

### DataFrame Indexing and Selection

```python
# Column selection
print("Single column selection:")
print(df['Name'])
print(type(df['Name']))  # Returns Series

print("\nMultiple columns selection:")
print(df[['Name', 'Age', 'Salary']])
print(type(df[['Name', 'Age']]))  # Returns DataFrame

# Row selection
print("\nRow selection by position (iloc):")
print(df.iloc[0])  # First row
print(df.iloc[0:3])  # First three rows
print(df.iloc[[0, 2, 4]])  # Specific rows

print("\nRow selection by label (loc):")
df_indexed = df.set_index('Name')
print(df_indexed.loc['Alice'])  # Single row by label
print(df_indexed.loc[['Alice', 'Charlie']])  # Multiple rows by label

# Boolean indexing
print("\nBoolean indexing:")
high_salary = df[df['Salary'] > 70000]
print("Employees with salary > 70000:")
print(high_salary)

# Complex boolean conditions
complex_filter = df[(df['Age'] > 28) & (df['Department'] == 'IT')]
print("\nIT employees older than 28:")
print(complex_filter)

# Using query method (more readable for complex conditions)
query_result = df.query("Age > 30 and Salary < 90000")
print("\nUsing query method:")
print(query_result)

# Selecting specific cells
print(f"\nSpecific cell access:")
print(f"Alice's salary: {df.loc[df['Name'] == 'Alice', 'Salary'].iloc[0]}")
print(f"Using at method: {df[df['Name'] == 'Alice']['Salary'].iloc[0]}")
```

### Adding and Removing Data

```python
# Adding columns
df_modified = df.copy()

# Add calculated column
df_modified['Salary_per_Year_Experience'] = df_modified['Salary'] / df_modified['Years_Experience']

# Add column with conditions
df_modified['Salary_Category'] = pd.cut(df_modified['Salary'], 
                                      bins=[0, 60000, 80000, 100000],
                                      labels=['Low', 'Medium', 'High'])

# Add column using apply
df_modified['Performance_Level'] = df_modified['Performance_Score'].apply(
    lambda x: 'Excellent' if x >= 9.0 else 'Good' if x >= 8.5 else 'Average'
)

print("DataFrame with new columns:")
print(df_modified)

# Adding rows
new_employee = {
    'Name': 'Grace',
    'Age': 26,
    'Department': 'Marketing',
    'Salary': 55000,
    'Years_Experience': 3,
    'Performance_Score': 8.7
}

# Using pd.concat (recommended)
new_row_df = pd.DataFrame([new_employee])
df_with_new_row = pd.concat([df, new_row_df], ignore_index=True)

print("\nDataFrame with new row:")
print(df_with_new_row.tail(3))

# Removing columns
df_reduced = df_modified.drop(['Salary_per_Year_Experience'], axis=1)
print(f"\nAfter dropping column: {df_reduced.columns.tolist()}")

# Removing rows
df_filtered = df.drop(df[df['Age'] < 27].index)
print(f"\nAfter removing employees younger than 27:")
print(df_filtered)

# Remove duplicates (if any)
df_no_duplicates = df.drop_duplicates()
print(f"Shape after removing duplicates: {df_no_duplicates.shape}")
```

## Data Loading and Saving

### Reading from Various Sources

```python
# Create sample data for demonstration
sample_data = {
    'Date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'Product': np.random.choice(['A', 'B', 'C'], 100),
    'Sales': np.random.randint(100, 1000, 100),
    'Profit': np.random.randint(10, 100, 100),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100)
}
sample_df = pd.DataFrame(sample_data)

# Save to various formats
sample_df.to_csv('sample_data.csv', index=False)
sample_df.to_excel('sample_data.xlsx', index=False, sheet_name='Sales_Data')
sample_df.to_json('sample_data.json', orient='records', date_format='iso')

print("Sample data created and saved")
print(sample_df.head())

# Reading from CSV
print("\n1. Reading from CSV:")
df_csv = pd.read_csv('sample_data.csv')
print(f"Shape: {df_csv.shape}")
print(df_csv.head(3))

# CSV with custom parameters
df_csv_custom = pd.read_csv('sample_data.csv', 
                           parse_dates=['Date'],  # Parse dates
                           dtype={'Product': 'category'},  # Specify data types
                           nrows=10)  # Read only first 10 rows
print(f"Custom CSV read - Date column type: {df_csv_custom['Date'].dtype}")

# Reading from Excel
print("\n2. Reading from Excel:")
df_excel = pd.read_excel('sample_data.xlsx', sheet_name='Sales_Data')
print(f"Excel data shape: {df_excel.shape}")

# Reading from JSON
print("\n3. Reading from JSON:")
df_json = pd.read_json('sample_data.json')
print(f"JSON data shape: {df_json.shape}")
print(df_json.head(3))

# Reading from URL (example with a public dataset)
try:
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    df_url = pd.read_csv(url)
    print(f"\n4. Reading from URL:")
    print(f"Tips dataset shape: {df_url.shape}")
    print(df_url.head(3))
except Exception as e:
    print(f"Could not load data from URL: {e}")

# Reading from SQL (requires sqlalchemy)
try:
    import sqlite3
    
    # Create sample SQLite database
    conn = sqlite3.connect('sample.db')
    sample_df.to_sql('sales_data', conn, if_exists='replace', index=False)
    
    # Read from SQL
    df_sql = pd.read_sql_query("SELECT * FROM sales_data WHERE Sales > 500", conn)
    print(f"\n5. Reading from SQL:")
    print(f"SQL query result shape: {df_sql.shape}")
    conn.close()
    
except ImportError:
    print("\n5. SQL reading requires sqlite3 (install with: pip install sqlite3)")
```

### Saving Data

```python
# Different saving formats and options
df = sample_df.copy()

print("Saving data in various formats:")

# CSV options
df.to_csv('output.csv', index=False)  # Without index
df.to_csv('output_with_index.csv', index=True)  # With index
df.to_csv('output_custom.csv', 
         sep=';',  # Custom separator
         decimal=',',  # Custom decimal separator
         encoding='utf-8',  # Encoding
         columns=['Date', 'Product', 'Sales'])  # Specific columns only

# Excel with multiple sheets
with pd.ExcelWriter('output_multiple_sheets.xlsx') as writer:
    df[df['Product'] == 'A'].to_excel(writer, sheet_name='Product_A', index=False)
    df[df['Product'] == 'B'].to_excel(writer, sheet_name='Product_B', index=False)
    df[df['Product'] == 'C'].to_excel(writer, sheet_name='Product_C', index=False)

# JSON with different orientations
df.to_json('output_records.json', orient='records', date_format='iso')
df.to_json('output_index.json', orient='index', date_format='iso')
df.to_json('output_values.json', orient='values', date_format='iso')

# Parquet format (efficient for large datasets)
try:
    df.to_parquet('output.parquet', engine='pyarrow')
    print("Data saved in Parquet format")
except ImportError:
    print("Parquet format requires pyarrow: pip install pyarrow")

# Pickle format (preserves exact pandas objects)
df.to_pickle('output.pkl')

print("Data saved in various formats successfully")

# Reading back to verify
df_read_csv = pd.read_csv('output.csv')
df_read_pickle = pd.read_pickle('output.pkl')

print(f"Verification - CSV shape: {df_read_csv.shape}")
print(f"Verification - Pickle shape: {df_read_pickle.shape}")
print(f"Data types preserved in pickle: {df_read_pickle.dtypes}")
```

## Data Inspection and Exploration

### Basic Information Methods

```python
# Create a more complex dataset for exploration
np.random.seed(42)
n_rows = 1000

df = pd.DataFrame({
    'customer_id': range(1, n_rows + 1),
    'age': np.random.randint(18, 80, n_rows),
    'income': np.random.normal(50000, 20000, n_rows),
    'spending_score': np.random.randint(1, 100, n_rows),
    'gender': np.random.choice(['Male', 'Female'], n_rows),
    'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_rows),
    'purchase_amount': np.random.exponential(100, n_rows),
    'satisfaction_rating': np.random.choice([1, 2, 3, 4, 5], n_rows, p=[0.1, 0.15, 0.2, 0.35, 0.2]),
    'signup_date': pd.date_range('2020-01-01', periods=n_rows, freq='D')
})

# Add some missing values for demonstration
missing_indices = np.random.choice(df.index, size=50, replace=False)
df.loc[missing_indices[:25], 'income'] = np.nan
df.loc[missing_indices[25:], 'satisfaction_rating'] = np.nan

print("Dataset created for exploration")
print(f"Shape: {df.shape}")

# Basic information methods
print("\n1. head() and tail():")
print(df.head())
print("\nLast 3 rows:")
print(df.tail(3))

print("\n2. info() - Overview of the dataset:")
df.info()

print("\n3. describe() - Statistical summary:")
print(df.describe())

# Include non-numeric columns in describe
print("\n4. describe(include='all') - All columns:")
print(df.describe(include='all'))

print("\n5. Column data types:")
print(df.dtypes)

print("\n6. Dataset shape and size:")
print(f"Shape: {df.shape}")
print(f"Size: {df.size}")
print(f"Number of dimensions: {df.ndim}")

print("\n7. Missing values summary:")
print("Missing values per column:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")
print(f"Percentage of missing values: {(df.isnull().sum().sum() / df.size) * 100:.2f}%")

print("\n8. Memory usage:")
print(df.memory_usage(deep=True))
```

### Exploratory Data Analysis

```python
# Detailed exploration
print("Detailed Exploratory Data Analysis:")

# Unique values and cardinality
print("\n1. Unique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

print("\n2. Value counts for categorical columns:")
categorical_columns = ['gender', 'city', 'satisfaction_rating']
for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts())

# Numerical summaries
numerical_columns = ['age', 'income', 'spending_score', 'purchase_amount']
print("\n3. Detailed numerical summaries:")
for col in numerical_columns:
    series = df[col].dropna()
    print(f"\n{col}:")
    print(f"  Mean: {series.mean():.2f}")
    print(f"  Median: {series.median():.2f}")
    print(f"  Mode: {series.mode().iloc[0]:.2f}")
    print(f"  Standard Deviation: {series.std():.2f}")
    print(f"  Variance: {series.var():.2f}")
    print(f"  Skewness: {series.skew():.2f}")
    print(f"  Kurtosis: {series.kurtosis():.2f}")
    print(f"  Range: {series.max() - series.min():.2f}")
    print(f"  IQR: {series.quantile(0.75) - series.quantile(0.25):.2f}")

# Quantiles
print("\n4. Quantiles for numerical columns:")
print(df[numerical_columns].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

# Correlation analysis
print("\n5. Correlation matrix:")
correlation_matrix = df[numerical_columns].corr()
print(correlation_matrix)

# Find highly correlated pairs
print("\n6. Highly correlated pairs (|correlation| > 0.3):")
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        correlation = correlation_matrix.iloc[i, j]
        if abs(correlation) > 0.3:
            print(f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}: {correlation:.3f}")

# Data quality checks
print("\n7. Data Quality Checks:")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Check for outliers using IQR method
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

print("\nOutliers per numerical column:")
for col in numerical_columns:
    if col in df.columns:
        outliers = detect_outliers_iqr(df[col].dropna())
        print(f"{col}: {outliers.sum()} outliers")

# Sample rows for manual inspection
print("\n8. Random sample of 5 rows:")
print(df.sample(5))
```

## Data Cleaning

### Handling Missing Values

```python
# Create a dataset with various types of missing values
dirty_data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5, np.nan, 7],
    'B': [1.1, np.nan, 3.3, 4.4, np.nan, 6.6, 7.7],
    'C': ['a', 'b', None, 'd', 'e', '', 'g'],
    'D': [10, 20, 30, np.nan, 50, 60, 70],
    'E': [True, False, np.nan, True, False, np.nan, True]
})

print("Original dirty dataset:")
print(dirty_data)
print(f"\nMissing values:\n{dirty_data.isnull().sum()}")

# 1. Detecting missing values
print("\n1. Different ways to detect missing values:")
print(f"isnull(): \n{dirty_data.isnull()}")
print(f"isna(): \n{dirty_data.isna()}")  # Same as isnull()
print(f"notnull(): \n{dirty_data.notnull()}")

# Check for specific values
print(f"Empty strings in column C: {(dirty_data['C'] == '').sum()}")

# 2. Dropping missing values
print("\n2. Dropping missing values:")

# Drop rows with any missing values
clean_dropna_any = dirty_data.dropna()
print(f"Drop rows with any NaN - Shape: {clean_dropna_any.shape}")
print(clean_dropna_any)

# Drop rows where all values are missing
clean_dropna_all = dirty_data.dropna(how='all')
print(f"Drop rows where all are NaN - Shape: {clean_dropna_all.shape}")

# Drop columns with missing values
clean_dropna_columns = dirty_data.dropna(axis=1)
print(f"Drop columns with any NaN - Shape: {clean_dropna_columns.shape}")
print(clean_dropna_columns)

# Drop with threshold (keep rows with at least n non-null values)
clean_threshold = dirty_data.dropna(thresh=4)  # At least 4 non-null values
print(f"Drop rows with < 4 non-null values - Shape: {clean_threshold.shape}")

# 3. Filling missing values
print("\n3. Filling missing values:")

# Fill with constant value
fill_constant = dirty_data.fillna(0)
print("Fill with 0:")
print(fill_constant)

# Fill with different values for different columns
fill_dict = dirty_data.fillna({
    'A': dirty_data['A'].mean(),  # Mean for numeric
    'B': dirty_data['B'].median(),  # Median for numeric
    'C': 'Unknown',  # Constant for string
    'D': dirty_data['D'].interpolate(),  # Interpolation
    'E': dirty_data['E'].mode()[0]  # Mode for boolean
})
print("\nFill with column-specific methods:")
print(fill_dict)

# Forward fill and backward fill
print("\nForward fill (propagate last valid observation):")
print(dirty_data.fillna(method='ffill'))

print("\nBackward fill (use next valid observation):")
print(dirty_data.fillna(method='bfill'))

# Interpolation methods
numeric_series = pd.Series([1, np.nan, np.nan, 4, np.nan, 6])
print(f"\nOriginal series: {numeric_series.tolist()}")
print(f"Linear interpolation: {numeric_series.interpolate().tolist()}")
print(f"Polynomial interpolation: {numeric_series.interpolate(method='polynomial', order=2).tolist()}")

# 4. Advanced missing value handling
print("\n4. Advanced missing value handling:")

# Replace empty strings with NaN first
advanced_clean = dirty_data.replace('', np.nan)

# Use different strategies for different data types
for col in advanced_clean.columns:
    if advanced_clean[col].dtype in ['float64', 'int64']:
        # For numeric: fill with median
        advanced_clean[col] = advanced_clean[col].fillna(advanced_clean[col].median())
    elif advanced_clean[col].dtype == 'object':
        # For categorical: fill with mode or 'Unknown'
        mode_value = advanced_clean[col].mode()
        fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
        advanced_clean[col] = advanced_clean[col].fillna(fill_value)
    else:
        # For other types: forward fill
        advanced_clean[col] = advanced_clean[col].fillna(method='ffill')

print("Advanced cleaning result:")
print(advanced_clean)
print(f"Missing values after advanced cleaning:\n{advanced_clean.isnull().sum()}")
```

### Data Type Conversion and Validation

```python
# Create sample data with mixed types
mixed_data = pd.DataFrame({
    'integers_as_strings': ['1', '2', '3', '4', '5'],
    'floats_as_strings': ['1.1', '2.2', '3.3', '4.4', '5.5'],
    'dates_as_strings': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
    'booleans_as_strings': ['True', 'False', 'True', 'False', 'True'],
    'categories': ['A', 'B', 'A', 'C', 'B'],
    'mixed_numeric': [1, 2.5, 3, '4', 5.5],  # Mixed types
    'dirty_numeric': ['1', '2.5', 'invalid', '4', '5.5']  # With invalid values
})

print("Original mixed data types:")
print(mixed_data.dtypes)
print(mixed_data)

# 1. Basic type conversions
print("\n1. Basic type conversions:")

# Convert to appropriate types
converted_data = mixed_data.copy()

# Integer conversion
converted_data['integers_as_strings'] = pd.to_numeric(converted_data['integers_as_strings'])

# Float conversion
converted_data['floats_as_strings'] = pd.to_numeric(converted_data['floats_as_strings'])

# Date conversion
converted_data['dates_as_strings'] = pd.to_datetime(converted_data['dates_as_strings'])

# Boolean conversion
converted_data['booleans_as_strings'] = converted_data['booleans_as_strings'].map({'True': True, 'False': False})

# Category conversion
converted_data['categories'] = converted_data['categories'].astype('category')

print("After conversion:")
print(converted_data.dtypes)

# 2. Handling conversion errors
print("\n2. Handling conversion errors:")

# Convert with error handling
converted_data['mixed_numeric'] = pd.to_numeric(converted_data['mixed_numeric'], errors='coerce')
converted_data['dirty_numeric'] = pd.to_numeric(converted_data['dirty_numeric'], errors='coerce')

print("After handling mixed/dirty numeric:")
print(converted_data[['mixed_numeric', 'dirty_numeric']])
print(f"NaN values created: {converted_data[['mixed_numeric', 'dirty_numeric']].isnull().sum()}")

# 3. Data validation
print("\n3. Data validation:")

def validate_dataframe(df):
    """Perform comprehensive data validation."""
    validation_results = {}
    
    for col in df.columns:
        validation_results[col] = {
            'dtype': str(df[col].dtype),
            'null_count': df[col].isnull().sum(),
            'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'unique_count': df[col].nunique(),
            'memory_usage': df[col].memory_usage(deep=True)
        }
        
        # Type-specific validations
        if pd.api.types.is_numeric_dtype(df[col]):
            validation_results[col].update({
                'min': df[col].min(),
                'max': df[col].max(),
                'has_negative': (df[col] < 0).any(),
                'has_infinite': np.isinf(df[col]).any()
            })
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            validation_results[col].update({
                'min_date': df[col].min(),
                'max_date': df[col].max(),
                'date_range_days': (df[col].max() - df[col].min()).days
            })
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            validation_results[col].update({
                'max_length': df[col].astype(str).str.len().max(),
                'min_length': df[col].astype(str).str.len().min(),
                'has_whitespace_only': df[col].astype(str).str.strip().eq('').any()
            })
    
    return validation_results

# Validate the converted data
validation_report = validate_dataframe(converted_data)

print("Validation report:")
for col, report in validation_report.items():
    print(f"\nColumn: {col}")
    for key, value in report.items():
        print(f"  {key}: {value}")
```

### Handling Duplicates and Data Consistency

```python
# Create dataset with duplicates and inconsistencies
inconsistent_data = pd.DataFrame({
    'ID': [1, 2, 3, 2, 4, 5, 3, 6],
    'Name': ['Alice Smith', 'Bob Jones', 'Charlie Brown', 'bob jones', 'Diana Prince', 'Eve Wilson', 'charlie brown', 'Frank Miller'],
    'Email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', 'bob@email.com', 'diana@email.com', 'eve@email.com', 'charlie@email.com', 'frank@email.com'],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Los Angeles', 'Houston', 'Phoenix', 'chicago', 'Boston'],
    'Salary': [50000, 60000, 70000, 60000, 80000, 55000, 75000, 65000],  # Different salaries for same person
    'Join_Date': ['2021-01-15', '2021-02-20', '2021-03-10', '2021-02-20', '2021-04-05', '2021-05-12', '2021-03-10', '2021-06-08']
})

print("Dataset with duplicates and inconsistencies:")
print(inconsistent_data)

# 1. Identifying duplicates
print("\n1. Identifying duplicates:")

# Check for duplicate rows
print(f"Duplicate rows (all columns): {inconsistent_data.duplicated().sum()}")
print(f"Duplicate rows (subset): {inconsistent_data.duplicated(subset=['ID']).sum()}")

# Show duplicate rows
duplicates_all = inconsistent_data[inconsistent_data.duplicated(keep=False)]
print("\nAll rows that have duplicates:")
print(duplicates_all.sort_values('ID'))

# 2. Removing duplicates
print("\n2. Removing duplicates:")

# Remove exact duplicates
no_exact_duplicates = inconsistent_data.drop_duplicates()
print(f"After removing exact duplicates: {no_exact_duplicates.shape}")

# Remove duplicates based on subset of columns
no_id_duplicates = inconsistent_data.drop_duplicates(subset=['ID'], keep='first')
print(f"After removing ID duplicates (keep first): {no_id_duplicates.shape}")
print(no_id_duplicates)

# 3. Data normalization and standardization
print("\n3. Data normalization and standardization:")

normalized_data = inconsistent_data.copy()

# Normalize text data
normalized_data['Name'] = normalized_data['Name'].str.title()  # Title case
normalized_data['Email'] = normalized_data['Email'].str.lower()  # Lowercase emails
normalized_data['City'] = normalized_data['City'].str.title()   # Title case cities

# Remove extra whitespace
for col in ['Name', 'Email', 'City']:
    normalized_data[col] = normalized_data[col].str.strip()

print("After normalization:")
print(normalized_data)

# 4. Handling inconsistent duplicates
print("\n4. Handling inconsistent duplicates:")

def resolve_duplicates(df, id_col, resolution_strategy='most_recent'):
    """Resolve duplicates with different strategies."""
    
    if resolution_strategy == 'most_recent':
        # Keep the most recent record (assuming date column exists)
        df['Join_Date'] = pd.to_datetime(df['Join_Date'])
        df_resolved = df.sort_values('Join_Date').drop_duplicates(subset=[id_col], keep='last')
    
    elif resolution_strategy == 'highest_value':
        # Keep record with highest salary
        df_resolved = df.sort_values('Salary').drop_duplicates(subset=[id_col], keep='last')
    
    elif resolution_strategy == 'aggregate':
        # Aggregate numeric columns and keep most frequent categorical
        numeric_agg = df.select_dtypes(include=[np.number]).groupby(id_col).mean()
        categorical_agg = df.select_dtypes(exclude=[np.number]).groupby(id_col).agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        )
        df_resolved = pd.concat([categorical_agg, numeric_agg], axis=1).reset_index()
    
    else:
        df_resolved = df.drop_duplicates(subset=[id_col], keep='first')
    
    return df_resolved

# Apply different resolution strategies
resolved_most_recent = resolve_duplicates(normalized_data, 'ID', 'most_recent')
resolved_highest_salary = resolve_duplicates(normalized_data, 'ID', 'highest_value')

print("Resolved duplicates (most recent):")
print(resolved_most_recent.sort_values('ID'))

print("\nResolved duplicates (highest salary):")
print(resolved_highest_salary.sort_values('ID'))

# 5. Data quality report
print("\n5. Data quality summary:")

def data_quality_report(original_df, cleaned_df):
    """Generate a data quality report."""
    report = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'duplicate_rate': (original_df.duplicated().sum() / len(original_df)) * 100,
        'null_values_original': original_df.isnull().sum().sum(),
        'null_values_cleaned': cleaned_df.isnull().sum().sum()
    }
    return report

quality_report = data_quality_report(inconsistent_data, resolved_most_recent)
print("Data Quality Report:")
for key, value in quality_report.items():
    print(f"  {key}: {value}")
```

## Data Transformation

### Applying Functions and Mapping

```python
# Create sample dataset
np.random.seed(42)
df = pd.DataFrame({
    'employee_id': range(1, 101),
    'name': [f'Employee_{i}' for i in range(1, 101)],
    'age': np.random.randint(22, 65, 100),
    'salary': np.random.randint(40000, 120000, 100),
    'department': np.random.choice(['HR', 'IT', 'Finance', 'Marketing', 'Operations'], 100),
    'performance_score': np.random.uniform(1, 10, 100),
    'years_experience': np.random.randint(0, 20, 100)
})

print("Original dataset:")
print(df.head())

# 1. Apply function - element-wise transformations
print("\n1. Apply function - element-wise transformations:")

# Apply to single column
df['age_group'] = df['age'].apply(lambda x: 'Young' if x < 30 else 'Middle' if x < 50 else 'Senior')

# Apply with custom function
def categorize_salary(salary):
    if salary < 50000:
        return 'Low'
    elif salary < 80000:
        return 'Medium'
    else:
        return 'High'

df['salary_category'] = df['salary'].apply(categorize_salary)

# Apply to multiple columns (row-wise)
def calculate_bonus(row):
    base_bonus = row['salary'] * 0.1
    performance_multiplier = row['performance_score'] / 10
    experience_bonus = row['years_experience'] * 1000
    return base_bonus * performance_multiplier + experience_bonus

df['bonus'] = df.apply(calculate_bonus, axis=1)

print("After applying functions:")
print(df[['age', 'age_group', 'salary', 'salary_category', 'bonus']].head())

# 2. Mapping values
print("\n2. Mapping values:")

# Simple mapping with dictionary
department_codes = {
    'HR': 'H001',
    'IT': 'I002',
    'Finance': 'F003',
    'Marketing': 'M004',
    'Operations': 'O005'
}
df['department_code'] = df['department'].map(department_codes)

# Map with default value
df['department_code_safe'] = df['department'].map(department_codes).fillna('Unknown')

# Using replace for mapping (works with Series or dict)
grade_mapping = {'Low': 'C', 'Medium': 'B', 'High': 'A'}
df['salary_grade'] = df['salary_category'].replace(grade_mapping)

print("After mapping:")
print(df[['department', 'department_code', 'salary_category', 'salary_grade']].head())

# 3. String operations
print("\n3. String operations:")

# Extract parts of strings
df['first_name'] = df['name'].str.replace('Employee_', 'Emp_')
df['name_length'] = df['name'].str.len()
df['name_upper'] = df['name'].str.upper()

# String contains and extraction
df['has_number'] = df['name'].str.contains(r'\d+')
df['employee_number'] = df['name'].str.extract(r'Employee_(\d+)')

print("String operations:")
print(df[['name', 'first_name', 'name_length', 'employee_number']].head())

# 4. Conditional transformations
print("\n4. Conditional transformations:")

# Using numpy.where
df['promotion_eligible'] = np.where(
    (df['years_experience'] >= 5) & (df['performance_score'] >= 7.0),
    'Eligible',
    'Not Eligible'
)

# Using pandas.cut for binning
df['performance_quartile'] = pd.cut(df['performance_score'], 
                                   bins=4, 
                                   labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Using pandas.qcut for quantile-based binning
df['salary_quintile'] = pd.qcut(df['salary'], 
                               q=5, 
                               labels=['Lowest', 'Low', 'Medium', 'High', 'Highest'])

print("Conditional transformations:")
print(df[['years_experience', 'performance_score', 'promotion_eligible', 
          'performance_quartile', 'salary_quintile']].head())

# 5. Advanced transformations
print("\n5. Advanced transformations:")

# Transform multiple columns at once
def standardize_numeric_columns(df, columns):
    """Standardize numeric columns to z-scores."""
    df_standardized = df.copy()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df_standardized[f'{col}_standardized'] = (df[col] - mean) / std
    return df_standardized

# Standardize salary and performance score
numeric_cols = ['salary', 'performance_score']
df_with_std = standardize_numeric_columns(df, numeric_cols)

print("Standardized columns:")
print(df_with_std[['salary', 'salary_standardized', 
                   'performance_score', 'performance_score_standardized']].head())

# Ranking and percentiles
df['salary_rank'] = df['salary'].rank(ascending=False)
df['performance_percentile'] = df['performance_score'].rank(pct=True)

print("\nRankings and percentiles:")
print(df[['salary', 'salary_rank', 'performance_score', 'performance_percentile']].head())
```

### Reshaping Data

```python
# Create sample data for reshaping examples
wide_data = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Q1_Sales': [100, 150, 120, 180],
    'Q2_Sales': [110, 160, 130, 190],
    'Q3_Sales': [120, 170, 140, 200],
    'Q4_Sales': [130, 180, 150, 210]
})

print("Wide format data:")
print(wide_data)

# 1. Melt (Wide to Long)
print("\n1. Melt - Wide to Long format:")

long_data = pd.melt(wide_data, 
                   id_vars=['ID', 'Name'],  # Columns to keep as identifiers
                   value_vars=['Q1_Sales', 'Q2_Sales', 'Q3_Sales', 'Q4_Sales'],  # Columns to melt
                   var_name='Quarter',  # Name for the variable column
                   value_name='Sales')  # Name for the value column

print("Long format data:")
print(long_data)

# Clean up the Quarter column
long_data['Quarter'] = long_data['Quarter'].str.replace('_Sales', '')
print("\nCleaned long format:")
print(long_data.head(8))

# 2. Pivot (Long to Wide)
print("\n2. Pivot - Long to Wide format:")

# Pivot back to wide format
wide_again = long_data.pivot(index=['ID', 'Name'], 
                            columns='Quarter', 
                            values='Sales')

print("Pivoted back to wide:")
print(wide_again)

# Reset index to make it look like original
wide_again_clean = wide_again.reset_index()
wide_again_clean.columns.name = None  # Remove the column index name

print("Cleaned pivot result:")
print(wide_again_clean)

# 3. Pivot Table (with aggregation)
print("\n3. Pivot Table with aggregation:")

# Create more complex data for pivot table
sales_data = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=24, freq='M'),
    'Region': ['North', 'South'] * 12,
    'Product': ['A', 'B'] * 6 + ['A', 'B'] * 6,
    'Sales': np.random.randint(100, 1000, 24),
    'Profit': np.random.randint(10, 100, 24)
})

sales_data['Year'] = sales_data['Date'].dt.year
sales_data['Month'] = sales_data['Date'].dt.month

print("Sales data for pivot table:")
print(sales_data.head())

# Create pivot table with aggregation
pivot_table = pd.pivot_table(sales_data,
                            values=['Sales', 'Profit'],  # Values to aggregate
                            index='Region',  # Row grouping
                            columns='Product',  # Column grouping
                            aggfunc='mean',  # Aggregation function
                            fill_value=0)  # Fill missing values

print("\nPivot table:")
print(pivot_table)

# 4. Stack and Unstack
print("\n4. Stack and Unstack:")

# Create MultiIndex DataFrame
df_multi = wide_data.set_index(['ID', 'Name'])
print("MultiIndex DataFrame:")
print(df_multi)

# Stack (columns become part of index)
stacked = df_multi.stack()
print("\nStacked (Series):")
print(stacked.head())

# Convert back to DataFrame
stacked_df = stacked.reset_index()
stacked_df.columns = ['ID', 'Name', 'Quarter', 'Sales']
stacked_df['Quarter'] = stacked_df['Quarter'].str.replace('_Sales', '')
print("\nStacked as DataFrame:")
print(stacked_df.head())

# Unstack (index level becomes columns)
unstacked = stacked_df.set_index(['ID', 'Name', 'Quarter'])['Sales'].unstack()
print("\nUnstacked:")
print(unstacked)

# 5. Cross-tabulation
print("\n5. Cross-tabulation:")

# Create categorical data
categories_data = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female'], 100),
    'Department': np.random.choice(['HR', 'IT', 'Finance'], 100),
    'Satisfaction': np.random.choice(['Low', 'Medium', 'High'], 100),
    'Performance': np.random.choice(['Poor', 'Good', 'Excellent'], 100)
})

# Simple cross-tabulation
crosstab_simple = pd.crosstab(categories_data['Gender'], categories_data['Department'])
print("Simple cross-tabulation:")
print(crosstab_simple)

# Cross-tabulation with percentages
crosstab_pct = pd.crosstab(categories_data['Gender'], categories_data['Department'], 
                          normalize='index') * 100
print("\nCross-tabulation with percentages:")
print(crosstab_pct.round(1))

# Three-way cross-tabulation
crosstab_3way = pd.crosstab([categories_data['Gender'], categories_data['Department']], 
                           categories_data['Satisfaction'])
print("\nThree-way cross-tabulation:")
print(crosstab_3way)
```

### Working with Categorical Data

```python
# Create dataset with categorical variables
categorical_data = pd.DataFrame({
    'employee_id': range(1, 101),
    'department': np.random.choice(['HR', 'IT', 'Finance', 'Marketing'], 100),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
    'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 100),
    'satisfaction': np.random.choice(['Very Low', 'Low', 'Medium', 'High', 'Very High'], 100),
    'salary': np.random.randint(40000, 120000, 100)
})

print("Original data types:")
print(categorical_data.dtypes)
print("\nSample data:")
print(categorical_data.head())

# 1. Converting to categorical
print("\n1. Converting to categorical:")

# Convert specific columns to categorical
categorical_data['department'] = categorical_data['department'].astype('category')
categorical_data['education'] = pd.Categorical(categorical_data['education'])

# Convert with ordered categories
satisfaction_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
categorical_data['satisfaction'] = pd.Categorical(
    categorical_data['satisfaction'], 
    categories=satisfaction_order, 
    ordered=True
)

print("After conversion to categorical:")
print(categorical_data.dtypes)

# 2. Categorical properties
print("\n2. Categorical properties:")

print(f"Department categories: {categorical_data['department'].cat.categories.tolist()}")
print(f"Education categories: {categorical_data['education'].cat.categories.tolist()}")
print(f"Satisfaction categories (ordered): {categorical_data['satisfaction'].cat.categories.tolist()}")
print(f"Is satisfaction ordered: {categorical_data['satisfaction'].cat.ordered}")

# Memory usage comparison
memory_object = categorical_data['city'].memory_usage(deep=True)
categorical_data['city_cat'] = categorical_data['city'].astype('category')
memory_categorical = categorical_data['city_cat'].memory_usage(deep=True)

print(f"\nMemory usage - Object: {memory_object} bytes")
print(f"Memory usage - Categorical: {memory_categorical} bytes")
print(f"Memory savings: {((memory_object - memory_categorical) / memory_object) * 100:.1f}%")

# 3. Manipulating categories
print("\n3. Manipulating categories:")

# Add new category
categorical_data['department'] = categorical_data['department'].cat.add_categories(['Research'])
print(f"Department categories after adding: {categorical_data['department'].cat.categories.tolist()}")

# Remove unused categories
categorical_data['department'] = categorical_data['department'].cat.remove_unused_categories()
print(f"Department categories after removing unused: {categorical_data['department'].cat.categories.tolist()}")

# Rename categories
education_rename = {'High School': 'HS', 'Bachelor': 'BS', 'Master': 'MS', 'PhD': 'PhD'}
categorical_data['education'] = categorical_data['education'].cat.rename_categories(education_rename)
print(f"Education categories after renaming: {categorical_data['education'].cat.categories.tolist()}")

# Reorder categories
dept_order = ['HR', 'Finance', 'IT', 'Marketing']
categorical_data['department'] = categorical_data['department'].cat.reorder_categories(dept_order)
print(f"Department categories reordered: {categorical_data['department'].cat.categories.tolist()}")

# 4. One-hot encoding
print("\n4. One-hot encoding:")

# Get dummies (one-hot encoding)
department_dummies = pd.get_dummies(categorical_data['department'], prefix='dept')
education_dummies = pd.get_dummies(categorical_data['education'], prefix='edu')

print("Department dummies:")
print(department_dummies.head())

print("\nEducation dummies:")
print(education_dummies.head())

# Combine with original data
encoded_data = pd.concat([
    categorical_data[['employee_id', 'salary']], 
    department_dummies, 
    education_dummies
], axis=1)

print("\nCombined data with one-hot encoding:")
print(encoded_data.head())

# 5. Label encoding for ordered categories
print("\n5. Label encoding for ordered categories:")

# For ordered categorical, we can get numeric codes
satisfaction_codes = categorical_data['satisfaction'].cat.codes
print("Satisfaction as numeric codes:")
print(satisfaction_codes.head(10))

# Create a mapping
satisfaction_mapping = dict(enumerate(categorical_data['satisfaction'].cat.categories))
print(f"\nSatisfaction mapping: {satisfaction_mapping}")

# 6. Groupby operations with categoricals
print("\n6. Groupby operations with categoricals:")

# Group by categorical columns
dept_salary_stats = categorical_data.groupby('department')['salary'].agg(['mean', 'median', 'std'])
print("Salary statistics by department:")
print(dept_salary_stats.round(0))

# Cross-tabulation with categoricals
edu_dept_crosstab = pd.crosstab(categorical_data['education'], categorical_data['department'])
print("\nEducation vs Department cross-tabulation:")
print(edu_dept_crosstab)

# Value counts for categoricals
print("\nValue counts for satisfaction (ordered):")
print(categorical_data['satisfaction'].value_counts(sort=False))  # Keep categorical order
```

## Grouping and Aggregation

### Basic Groupby Operations

```python
# Create comprehensive dataset for grouping examples
np.random.seed(42)
n_records = 1000

df = pd.DataFrame({
    'employee_id': range(1, n_records + 1),
    'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR', 'Finance'], n_records),
    'team': np.random.choice(['Team_A', 'Team_B', 'Team_C'], n_records),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
    'salary': np.random.normal(70000, 20000, n_records),
    'bonus': np.random.normal(5000, 2000, n_records),
    'performance_score': np.random.uniform(1, 10, n_records),
    'years_experience': np.random.randint(0, 25, n_records),
    'age': np.random.randint(22, 65, n_records),
    'gender': np.random.choice(['Male', 'Female'], n_records),
    'hire_date': pd.date_range('2015-01-01', periods=n_records, freq='D')
})

# Ensure salary and bonus are positive
df['salary'] = np.abs(df['salary'])
df['bonus'] = np.abs(df['bonus'])

print("Sample data for grouping:")
print(df.head())
print(f"Dataset shape: {df.shape}")

# 1. Basic groupby operations
print("\n1. Basic groupby operations:")

# Group by single column
dept_groups = df.groupby('department')
print("Group by department:")
print(f"Number of groups: {dept_groups.ngroups}")
print(f"Group sizes:\n{dept_groups.size()}")

# Basic aggregations
print("\nMean salary by department:")
print(dept_groups['salary'].mean().round(0))

print("\nMultiple aggregations:")
dept_agg = dept_groups.agg({
    'salary': ['mean', 'median', 'std'],
    'bonus': ['mean', 'max'],
    'performance_score': ['mean', 'count']
})
print(dept_agg.round(2))

# 2. Multiple column grouping
print("\n2. Multiple column grouping:")

# Group by multiple columns
dept_region_groups = df.groupby(['department', 'region'])
print("Group by department and region:")
print(f"Number of groups: {dept_region_groups.ngroups}")

dept_region_agg = dept_region_groups.agg({
    'salary': 'mean',
    'performance_score': 'mean',
    'employee_id': 'count'  # Count of employees
}).round(2)
dept_region_agg.columns = ['avg_salary', 'avg_performance', 'employee_count']
print(dept_region_agg)

# 3. Different aggregation functions
print("\n3. Different aggregation functions:")

# Built-in aggregation functions
agg_functions = {
    'salary': ['mean', 'median', 'std', 'min', 'max', 'sum'],
    'bonus': ['mean', 'std'],
    'performance_score': ['mean', 'std'],
    'years_experience': ['mean', 'max'],
    'age': ['mean', 'min', 'max']
}

comprehensive_agg = df.groupby('department').agg(agg_functions)
print("Comprehensive aggregation by department:")
print(comprehensive_agg.round(2))

# 4. Custom aggregation functions
print("\n4. Custom aggregation functions:")

def salary_range(series):
    return series.max() - series.min()

def top_performer_threshold(series):
    return series.quantile(0.9)

custom_agg = df.groupby('department').agg({
    'salary': [salary_range, 'mean'],
    'performance_score': [top_performer_threshold, 'mean'],
    'years_experience': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean()  # Mode or mean
})

print("Custom aggregations:")
print(custom_agg.round(2))

# 5. Transform operations
print("\n5. Transform operations:")

# Transform to standardize within groups
df['salary_dept_zscore'] = df.groupby('department')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)

df['performance_dept_rank'] = df.groupby('department')['performance_score'].transform('rank')

df['salary_dept_pct'] = df.groupby('department')['salary'].transform(
    lambda x: x / x.sum()
)

print("Transform operations (first 10 rows):")
print(df[['department', 'salary', 'salary_dept_zscore', 
          'performance_score', 'performance_dept_rank']].head(10))

# 6. Filter operations
print("\n6. Filter operations:")

# Filter groups based on group characteristics
large_departments = df.groupby('department').filter(lambda x: len(x) > 180)
print(f"Employees in departments with >180 people: {len(large_departments)}")
print(f"Departments: {large_departments['department'].unique()}")

# Filter based on group statistics
high_performing_depts = df.groupby('department').filter(
    lambda x: x['performance_score'].mean() > 5.5
)
print(f"\nEmployees in high-performing departments: {len(high_performing_depts)}")
print(f"High-performing departments: {high_performing_depts['department'].unique()}")
```

### Advanced Groupby Operations

```python
# 1. Apply custom functions to groups
print("\n1. Apply custom functions to groups:")

def department_analysis(group):
    """Custom function to analyze each department."""
    result = pd.Series({
        'employee_count': len(group),
        'avg_salary': group['salary'].mean(),
        'salary_std': group['salary'].std(),
        'top_performer': group.loc[group['performance_score'].idxmax(), 'employee_id'],
        'avg_experience': group['years_experience'].mean(),
        'retention_score': len(group[group['years_experience'] > 5]) / len(group)
    })
    return result

dept_analysis = df.groupby('department').apply(department_analysis)
print("Department analysis:")
print(dept_analysis.round(3))

# 2. Rolling and expanding operations within groups
print("\n2. Rolling operations within groups:")

# Sort by hire date for meaningful rolling operations
df_sorted = df.sort_values(['department', 'hire_date'])

# Rolling average salary within each department
df_sorted['salary_rolling_3month'] = df_sorted.groupby('department')['salary'].rolling(
    window=90, min_periods=1
).mean().reset_index(0, drop=True)

# Expanding cumulative statistics
df_sorted['cumulative_avg_salary'] = df_sorted.groupby('department')['salary'].expanding().mean().reset_index(0, drop=True)

print("Rolling operations sample:")
print(df_sorted[['department', 'hire_date', 'salary', 'salary_rolling_3month', 'cumulative_avg_salary']].head(15))

# 3. Multi-level grouping with different aggregations
print("\n3. Multi-level grouping:")

# Group by multiple levels with different aggregations per level
multi_level_agg = df.groupby(['department', 'region', 'gender']).agg({
    'salary': ['mean', 'count'],
    'performance_score': 'mean',
    'years_experience': 'mean'
}).round(2)

print("Multi-level aggregation (first 20 rows):")
print(multi_level_agg.head(20))

# Unstack to create pivot-like structure
unstacked = multi_level_agg.unstack('gender')
print("\nUnstacked by gender:")
print(unstacked['salary']['mean'].head())

# 4. GroupBy with time-based operations
print("\n4. Time-based groupby operations:")

# Extract date components
df['hire_year'] = df['hire_date'].dt.year
df['hire_month'] = df['hire_date'].dt.month

# Time-based grouping
yearly_hiring = df.groupby(['department', 'hire_year']).agg({
    'employee_id': 'count',
    'salary': 'mean'
}).rename(columns={'employee_id': 'hires_count'})

print("Yearly hiring by department:")
print(yearly_hiring.head(15))

# 5. Percentile and quantile operations
print("\n5. Percentile operations:")

def calculate_percentiles(series):
    """Calculate multiple percentiles for a series."""
    return pd.Series({
        'p10': series.quantile(0.1),
        'p25': series.quantile(0.25),
        'p50': series.quantile(0.5),
        'p75': series.quantile(0.75),
        'p90': series.quantile(0.9)
    })

salary_percentiles = df.groupby('department')['salary'].apply(calculate_percentiles)
print("Salary percentiles by department:")
print(salary_percentiles.unstack().round(0))

# 6. Conditional aggregations
print("\n6. Conditional aggregations:")

def conditional_stats(group):
    """Calculate statistics based on conditions."""
    high_performers = group[group['performance_score'] >= 7]
    experienced = group[group['years_experience'] >= 5]
    
    return pd.Series({
        'total_employees': len(group),
        'high_performers': len(high_performers),
        'high_performer_avg_salary': high_performers['salary'].mean() if len(high_performers) > 0 else 0,
        'experienced_count': len(experienced),
        'experienced_avg_performance': experienced['performance_score'].mean() if len(experienced) > 0 else 0
    })

conditional_agg = df.groupby('department').apply(conditional_stats)
print("Conditional aggregations:")
print(conditional_agg.round(2))
```

### Pivot Tables and Cross-tabulation

```python
# Create more detailed dataset for pivot table examples
survey_data = pd.DataFrame({
    'respondent_id': range(1, 501),
    'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], 500),
    'gender': np.random.choice(['Male', 'Female', 'Other'], 500, p=[0.45, 0.45, 0.1]),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 500, p=[0.3, 0.4, 0.25, 0.05]),
    'income_bracket': np.random.choice(['<30k', '30k-50k', '50k-75k', '75k-100k', '>100k'], 500),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 500),
    'product_satisfaction': np.random.randint(1, 6, 500),  # 1-5 scale
    'service_satisfaction': np.random.randint(1, 6, 500),  # 1-5 scale
    'overall_satisfaction': np.random.randint(1, 6, 500),  # 1-5 scale
    'purchase_amount': np.random.exponential(100, 500),
    'visit_frequency': np.random.choice(['Daily', 'Weekly', 'Monthly', 'Rarely'], 500)
})

print("Survey data for pivot analysis:")
print(survey_data.head())

# 1. Basic pivot table
print("\n1. Basic pivot table:")

basic_pivot = pd.pivot_table(survey_data,
                           values='overall_satisfaction',
                           index='age_group',
                           columns='gender',
                           aggfunc='mean',
                           fill_value=0)

print("Average satisfaction by age group and gender:")
print(basic_pivot.round(2))

# 2. Multi-value pivot table
print("\n2. Multi-value pivot table:")

multi_value_pivot = pd.pivot_table(survey_data,
                                 values=['product_satisfaction', 'service_satisfaction', 'purchase_amount'],
                                 index='region',
                                 columns='education',
                                 aggfunc='mean',
                                 fill_value=0)

print("Multi-value pivot (first level - product satisfaction):")
print(multi_value_pivot['product_satisfaction'].round(2))

# 3. Pivot with multiple aggregation functions
print("\n3. Pivot with multiple functions:")

multi_func_pivot = pd.pivot_table(survey_data,
                                values='purchase_amount',
                                index='income_bracket',
                                columns='visit_frequency',
                                aggfunc=['mean', 'count', 'std'],
                                fill_value=0)

print("Purchase amount statistics by income and visit frequency:")
print(multi_func_pivot['mean'].round(2))

# 4. Pivot with margins (totals)
print("\n4. Pivot with margins:")

pivot_with_margins = pd.pivot_table(survey_data,
                                  values='overall_satisfaction',
                                  index='region',
                                  columns='age_group',
                                  aggfunc='mean',
                                  margins=True,
                                  margins_name='Overall')

print("Satisfaction with regional and age group totals:")
print(pivot_with_margins.round(2))

# 5. Cross-tabulation with statistical tests
print("\n5. Cross-tabulation analysis:")

# Basic cross-tabulation
crosstab = pd.crosstab(survey_data['gender'], survey_data['education'])
print("Gender vs Education cross-tabulation:")
print(crosstab)

# Normalized cross-tabulation
crosstab_normalized = pd.crosstab(survey_data['gender'], survey_data['education'], 
                                normalize='index') * 100
print("\nPercentage distribution within gender:")
print(crosstab_normalized.round(1))

# Cross-tabulation with values
crosstab_with_values = pd.crosstab(survey_data['age_group'], 
                                 survey_data['income_bracket'],
                                 values=survey_data['purchase_amount'],
                                 aggfunc='mean')
print("\nAverage purchase amount by age group and income:")
print(crosstab_with_values.round(2))

# 6. Advanced pivot operations
print("\n6. Advanced pivot operations:")

# Pivot with custom aggregation function
def satisfaction_score(series):
    """Custom satisfaction scoring function."""
    return (series >= 4).mean() * 100  # Percentage satisfied (rating 4+)

custom_pivot = pd.pivot_table(survey_data,
                            values='overall_satisfaction',
                            index='region',
                            columns='visit_frequency',
                            aggfunc=satisfaction_score,
                            fill_value=0)

print("Percentage of satisfied customers (rating 4+) by region and visit frequency:")
print(custom_pivot.round(1))

# Stack/unstack pivot results for different views
stacked_pivot = basic_pivot.stack()
print("\nStacked pivot (Series format):")
print(stacked_pivot.head(10))

# Reset index to convert back to DataFrame
pivot_df = basic_pivot.reset_index()
pivot_melted = pd.melt(pivot_df, 
                      id_vars='age_group',
                      var_name='gender',
                      value_name='satisfaction')
print("\nMelted pivot back to long format:")
print(pivot_melted.head())
```

## Merging and Joining

### Different Types of Joins

```python
# Create sample datasets for merging examples
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'customer_name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson'],
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'signup_date': pd.date_range('2023-01-01', periods=5, freq='30D')
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105, 106, 107],
    'customer_id': [1, 2, 2, 3, 6, 1, 4],  # Note: customer_id 6 doesn't exist in customers
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Tablet', 'Phone', 'Headphones'],
    'amount': [1200, 25, 75, 300, 500, 800, 150],
    'order_date': pd.date_range('2023-02-01', periods=7, freq='15D')
})

products = pd.DataFrame({
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Tablet', 'Phone', 'Headphones', 'Printer'],
    'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics', 'Electronics', 'Accessories', 'Electronics'],
    'price': [1200, 25, 75, 300, 500, 800, 150, 200]
})

print("Sample datasets:")
print("Customers:")
print(customers)
print("\nOrders:")
print(orders)
print("\nProducts:")
print(products)

# 1. Inner Join
print("\n1. Inner Join:")
inner_join = pd.merge(customers, orders, on='customer_id', how='inner')
print("Customers with orders (inner join):")
print(inner_join)

# 2. Left Join
print("\n2. Left Join:")
left_join = pd.merge(customers, orders, on='customer_id', how='left')
print("All customers with their orders (left join):")
print(left_join)

# 3. Right Join
print("\n3. Right Join:")
right_join = pd.merge(customers, orders, on='customer_id', how='right')
print("All orders with customer info (right join):")
print(right_join)

# 4. Outer Join
print("\n4. Outer Join:")
outer_join = pd.merge(customers, orders, on='customer_id', how='outer')
print("All customers and all orders (outer join):")
print(outer_join)

# 5. Merging on different column names
print("\n5. Merging on different column names:")

# Create dataset with different column name
customer_profiles = pd.DataFrame({
    'cust_id': [1, 2, 3, 4, 5],
    'age': [25, 30, 35, 28, 32],
    'income': [50000, 75000, 90000, 60000, 80000]
})

# Merge using left_on and right_on
merge_different_names = pd.merge(customers, customer_profiles, 
                               left_on='customer_id', right_on='cust_id', how='inner')
print("Merge with different column names:")
print(merge_different_names)

# 6. Multiple column joins
print("\n6. Multiple column joins:")

# Create data for multiple column join
order_details = pd.DataFrame({
    'customer_id': [1, 2, 2, 3, 1, 4],
    'order_date': pd.to_datetime(['2023-02-01', '2023-02-16', '2023-02-16', '2023-03-03', '2023-02-01', '2023-04-07']),
    'item_count': [1, 2, 1, 1, 1, 3],
    'shipping_cost': [10, 15, 5, 12, 8, 20]
})

# First ensure order_date columns have same name and type
orders_with_date = orders.copy()
orders_with_date['order_date'] = pd.to_datetime(orders_with_date['order_date'])

multi_join = pd.merge(orders_with_date, order_details, 
                     on=['customer_id', 'order_date'], how='inner')
print("Multiple column join:")
print(multi_join)
```

### Advanced Merging Techniques

```python
# 1. Index-based merging
print("\n1. Index-based merging:")

customers_indexed = customers.set_index('customer_id')
orders_indexed = orders.set_index('customer_id')

index_merge = pd.merge(customers_indexed, orders_indexed, 
                      left_index=True, right_index=True, how='inner')
print("Index-based merge:")
print(index_merge)

# 2. Handling duplicate keys
print("\n2. Handling duplicate keys:")

# Create data with duplicate keys
duplicate_orders = pd.DataFrame({
    'customer_id': [1, 1, 2, 2, 3],
    'order_type': ['Online', 'In-store', 'Online', 'Phone', 'Online'],
    'discount': [0.1, 0.05, 0.15, 0.0, 0.2]
})

duplicate_merge = pd.merge(customers, duplicate_orders, on='customer_id', how='left')
print("Merge with duplicate keys:")
print(duplicate_merge)

# 3. Suffix handling for overlapping columns
print("\n3. Suffix handling:")

# Create overlapping columns
customers_extended = customers.copy()
customers_extended['order_date'] = pd.date_range('2023-01-15', periods=5, freq='45D')

suffix_merge = pd.merge(customers_extended, orders, on='customer_id', 
                       how='inner', suffixes=('_customer', '_order'))
print("Merge with suffixes:")
print(suffix_merge)

# 4. Validation during merge
print("\n4. Merge validation:")

try:
    # This will raise an error if there are unexpected duplicates
    validated_merge = pd.merge(customers, orders, on='customer_id', 
                              how='left', validate='one_to_many')
    print("Validation successful - one customer to many orders relationship confirmed")
except Exception as e:
    print(f"Validation failed: {e}")

# 5. Indicator to track merge source
print("\n5. Merge with indicator:")

indicator_merge = pd.merge(customers, orders, on='customer_id', 
                          how='outer', indicator=True)
print("Merge with source indicator:")
print(indicator_merge[['customer_name', 'product', '_merge']].fillna('N/A'))

print("\nMerge indicator summary:")
print(indicator_merge['_merge'].value_counts())

# 6. Concat for combining DataFrames
print("\n6. Concatenation:")

# Vertical concatenation (stacking)
more_customers = pd.DataFrame({
    'customer_id': [6, 7, 8],
    'customer_name': ['Frank Miller', 'Grace Lee', 'Henry Davis'],
    'city': ['Seattle', 'Denver', 'Miami'],
    'signup_date': pd.date_range('2023-06-01', periods=3, freq='20D')
})

vertical_concat = pd.concat([customers, more_customers], ignore_index=True)
print("Vertical concatenation:")
print(vertical_concat)

# Horizontal concatenation
customer_details = pd.DataFrame({
    'phone': ['555-0101', '555-0102', '555-0103', '555-0104', '555-0105'],
    'email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', 'diana@email.com', 'eve@email.com']
})

horizontal_concat = pd.concat([customers, customer_details], axis=1)
print("\nHorizontal concatenation:")
print(horizontal_concat)

# 7. Complex multi-table joins
print("\n7. Complex multi-table joins:")

# Chain multiple merges
comprehensive_data = (orders
                     .merge(customers, on='customer_id', how='left')
                     .merge(products, on='product', how='left')
                     .merge(customer_profiles.rename(columns={'cust_id': 'customer_id'}), 
                           on='customer_id', how='left'))

print("Comprehensive multi-table join:")
print(comprehensive_data[['customer_name', 'product', 'category', 'amount', 'age', 'income']].head())

# 8. Handling many-to-many relationships
print("\n8. Many-to-many relationships:")

# Create many-to-many scenario
customer_interests = pd.DataFrame({
    'customer_id': [1, 1, 2, 2, 2, 3, 4, 4],
    'interest': ['Technology', 'Gaming', 'Technology', 'Sports', 'Gaming', 'Books', 'Technology', 'Music']
})

product_categories_detailed = pd.DataFrame({
    'interest': ['Technology', 'Gaming', 'Sports', 'Books', 'Music'],
    'related_category': ['Electronics', 'Electronics', 'Accessories', 'Electronics', 'Accessories']
})

many_to_many = pd.merge(customer_interests, product_categories_detailed, on='interest', how='inner')
print("Many-to-many relationship result:")
print(many_to_many)

# Aggregate many-to-many results
aggregated_interests = (many_to_many
                       .groupby(['customer_id', 'related_category'])
                       .agg({'interest': 'count'})
                       .rename(columns={'interest': 'interest_count'})
                       .reset_index())
print("\nAggregated many-to-many results:")
print(aggregated_interests)
```

## Time Series Analysis

### Working with Datetime Data

```python
# Create comprehensive time series dataset
date_range = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
np.random.seed(42)

# Generate realistic time series data with trends and seasonality
n_days = len(date_range)
trend = np.linspace(100, 200, n_days)  # Linear trend
seasonal = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)  # Annual seasonality
noise = np.random.normal(0, 10, n_days)  # Random noise
values = trend + seasonal + noise

ts_data = pd.DataFrame({
    'date': date_range,
    'sales': np.maximum(values, 0),  # Ensure non-negative sales
    'temperature': 20 + 15 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + np.random.normal(0, 5, n_days),
    'marketing_spend': np.random.exponential(1000, n_days),
    'day_of_week': date_range.day_of_week,
    'month': date_range.month
})

# Set date as index
ts_data = ts_data.set_index('date')

print("Time series dataset:")
print(ts_data.head(10))
print(f"Date range: {ts_data.index.min()} to {ts_data.index.max()}")
print(f"Frequency: {pd.infer_freq(ts_data.index)}")

# 1. Basic datetime operations
print("\n1. Basic datetime operations:")

# Extract datetime components
ts_data['year'] = ts_data.index.year
ts_data['month_name'] = ts_data.index.month_name()
ts_data['day_name'] = ts_data.index.day_name()
ts_data['quarter'] = ts_data.index.quarter
ts_data['week_of_year'] = ts_data.index.week
ts_data['is_weekend'] = ts_data.index.weekday >= 5

print("Datetime components:")
print(ts_data[['sales', 'year', 'month_name', 'day_name', 'quarter', 'is_weekend']].head())

# 2. Resampling operations
print("\n2. Resampling operations:")

# Monthly aggregations
monthly_data = ts_data.resample('M').agg({
    'sales': ['mean', 'sum', 'std'],
    'temperature': 'mean',
    'marketing_spend': 'sum'
})

print("Monthly resampling:")
print(monthly_data.head())

# Quarterly data
quarterly_data = ts_data.resample('Q').agg({
    'sales': 'sum',
    'temperature': 'mean'
})

print("\nQuarterly data:")
print(quarterly_data.head())

# Custom resampling with business days
business_weekly = ts_data.resample('W-FRI').agg({  # Week ending on Friday
    'sales': 'sum',
    'marketing_spend': 'sum'
})

print("\nWeekly business data (Friday cutoff):")
print(business_weekly.head())

# 3. Rolling operations
print("\n3. Rolling operations:")

# Rolling averages
ts_data['sales_7day_avg'] = ts_data['sales'].rolling(window=7).mean()
ts_data['sales_30day_avg'] = ts_data['sales'].rolling(window=30).mean()
ts_data['sales_7day_std'] = ts_data['sales'].rolling(window=7).std()

# Expanding operations (cumulative)
ts_data['sales_cumulative_avg'] = ts_data['sales'].expanding().mean()
ts_data['sales_cumulative_max'] = ts_data['sales'].expanding().max()

print("Rolling and expanding operations:")
print(ts_data[['sales', 'sales_7day_avg', 'sales_30day_avg', 'sales_cumulative_avg']].head(20))

# 4. Lag operations and shifts
print("\n4. Lag operations:")

# Create lag features
ts_data['sales_lag1'] = ts_data['sales'].shift(1)    # Previous day
ts_data['sales_lag7'] = ts_data['sales'].shift(7)    # Same day last week
ts_data['sales_lag365'] = ts_data['sales'].shift(365)  # Same day last year

# Lead features (future values)
ts_data['sales_lead1'] = ts_data['sales'].shift(-1)   # Next day

# Percentage change
ts_data['sales_pct_change'] = ts_data['sales'].pct_change()
ts_data['sales_diff'] = ts_data['sales'].diff()       # First difference

print("Lag and difference operations:")
print(ts_data[['sales', 'sales_lag1', 'sales_lag7', 'sales_pct_change']].head(10))

# 5. Seasonal decomposition (simplified)
print("\n5. Seasonal analysis:")

# Group by different time periods
monthly_avg = ts_data.groupby(ts_data.index.month)['sales'].mean()
day_of_week_avg = ts_data.groupby(ts_data.index.dayofweek)['sales'].mean()
quarterly_avg = ts_data.groupby(ts_data.index.quarter)['sales'].mean()

print("Average sales by month:")
print(monthly_avg.round(2))

print("\nAverage sales by day of week:")
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
day_of_week_avg.index = day_names
print(day_of_week_avg.round(2))

# 6. Time-based filtering and slicing
print("\n6. Time-based filtering:")

# Select specific year
sales_2023 = ts_data.loc['2023']
print(f"2023 data points: {len(sales_2023)}")

# Select date range
q1_2023 = ts_data.loc['2023-01-01':'2023-03-31']
print(f"Q1 2023 data points: {len(q1_2023)}")

# Filter by conditions
high_sales_days = ts_data[ts_data['sales'] > ts_data['sales'].quantile(0.9)]
print(f"High sales days (top 10%): {len(high_sales_days)}")

# Weekend vs weekday comparison
weekend_avg = ts_data[ts_data['is_weekend']]['sales'].mean()
weekday_avg = ts_data[~ts_data['is_weekend']]['sales'].mean()
print(f"Weekend average sales: {weekend_avg:.2f}")
print(f"Weekday average sales: {weekday_avg:.2f}")
```

### Time Series Analysis and Forecasting

```python
# 1. Trend analysis
print("\n1. Trend analysis:")

# Calculate year-over-year growth
yearly_data = ts_data.resample('Y')['sales'].sum()
yearly_growth = yearly_data.pct_change() * 100

print("Yearly sales and growth:")
for year, (sales, growth) in zip(yearly_data.index.year, zip(yearly_data.values, yearly_growth.values)):
    if not np.isnan(growth):
        print(f"{year}: Sales = {sales:.0f}, Growth = {growth:.1f}%")

# Trend decomposition using moving averages
ts_data['trend'] = ts_data['sales'].rolling(window=365, center=True).mean()
ts_data['detrended'] = ts_data['sales'] - ts_data['trend']

print("\nTrend analysis sample:")
print(ts_data[['sales', 'trend', 'detrended']].iloc[200:210])

# 2. Correlation with external factors
print("\n2. Correlation analysis:")

# Calculate correlations with different lags
correlations = {}
for lag in range(0, 8):
    corr = ts_data['sales'].corr(ts_data['temperature'].shift(lag))
    correlations[f'temp_lag_{lag}'] = corr

print("Sales correlation with temperature at different lags:")
for lag, corr in correlations.items():
    print(f"{lag}: {corr:.3f}")

# Marketing spend correlation
marketing_corr = ts_data['sales'].corr(ts_data['marketing_spend'])
print(f"\nSales correlation with marketing spend: {marketing_corr:.3f}")

# 3. Anomaly detection
print("\n3. Anomaly detection:")

# Statistical method using z-score
z_scores = np.abs((ts_data['sales'] - ts_data['sales'].mean()) / ts_data['sales'].std())
anomalies_zscore = ts_data[z_scores > 3]

print(f"Anomalies using z-score (>3): {len(anomalies_zscore)}")

# Using IQR method
Q1 = ts_data['sales'].quantile(0.25)
Q3 = ts_data['sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

anomalies_iqr = ts_data[(ts_data['sales'] < lower_bound) | (ts_data['sales'] > upper_bound)]
print(f"Anomalies using IQR method: {len(anomalies_iqr)}")

# Rolling statistical anomalies
rolling_mean = ts_data['sales'].rolling(window=30).mean()
rolling_std = ts_data['sales'].rolling(window=30).std()
rolling_z = (ts_data['sales'] - rolling_mean) / rolling_std

anomalies_rolling = ts_data[np.abs(rolling_z) > 2.5]
print(f"Anomalies using rolling statistics: {len(anomalies_rolling)}")

# 4. Simple forecasting methods
print("\n4. Simple forecasting:")

# Split data for forecasting
train_end = '2024-06-30'
train_data = ts_data.loc[:train_end]
test_data = ts_data.loc[train_end:]

print(f"Training data: {len(train_data)} days")
print(f"Test data: {len(test_data)} days")

# Simple moving average forecast
def simple_ma_forecast(series, window=30, periods=30):
    """Simple moving average forecast."""
    last_values = series.tail(window)
    forecast = [last_values.mean()] * periods
    return forecast

# Seasonal naive forecast (same day last year)
def seasonal_naive_forecast(series, seasonal_period=365, periods=30):
    """Seasonal naive forecast."""
    if len(series) >= seasonal_period:
        seasonal_values = series.tail(seasonal_period).head(periods)
        return seasonal_values.values
    else:
        return [series.mean()] * periods

# Generate forecasts
ma_forecast = simple_ma_forecast(train_data['sales'], periods=len(test_data))
seasonal_forecast = seasonal_naive_forecast(train_data['sales'], periods=len(test_data))

# Calculate forecast accuracy
def calculate_mape(actual, forecast):
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs((actual - forecast) / actual)) * 100

def calculate_mae(actual, forecast):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(actual - forecast))

# Evaluate forecasts
if len(test_data) > 0:
    ma_mape = calculate_mape(test_data['sales'].values[:len(ma_forecast)], ma_forecast)
    seasonal_mape = calculate_mape(test_data['sales'].values[:len(seasonal_forecast)], seasonal_forecast)
    
    print(f"Moving Average MAPE: {ma_mape:.2f}%")
    print(f"Seasonal Naive MAPE: {seasonal_mape:.2f}%")

# 5. Time series patterns
print("\n5. Pattern analysis:")

# Identify patterns by different time periods
patterns = {
    'Month': ts_data.groupby(ts_data.index.month)['sales'].mean(),
    'Day of Week': ts_data.groupby(ts_data.index.dayofweek)['sales'].mean(),
    'Quarter': ts_data.groupby(ts_data.index.quarter)['sales'].mean(),
    'Hour': ts_data.groupby(ts_data.index.hour)['sales'].mean() if 'hour' in ts_data.index.names else None
}

for period, pattern in patterns.items():
    if pattern is not None and len(pattern) > 0:
        print(f"\n{period} patterns:")
        print(pattern.round(2))

# Identify strongest seasonal pattern
monthly_var = patterns['Month'].var()
dow_var = patterns['Day of Week'].var()
quarterly_var = patterns['Quarter'].var()

print(f"\nSeasonal variation (variance):")
print(f"Monthly: {monthly_var:.2f}")
print(f"Day of Week: {dow_var:.2f}")
print(f"Quarterly: {quarterly_var:.2f}")

strongest_pattern = max([('Monthly', monthly_var), ('Day of Week', dow_var), ('Quarterly', quarterly_var)], 
                       key=lambda x: x[1])
print(f"Strongest seasonal pattern: {strongest_pattern[0]}")
```

## Advanced Operations

### Multi-level Indexing (MultiIndex)

```python
# Create comprehensive multi-index dataset
np.random.seed(42)

# Generate hierarchical data
companies = ['Apple', 'Google', 'Microsoft', 'Amazon']
metrics = ['Revenue', 'Profit', 'Employees', 'Market_Cap']
years = list(range(2020, 2025))
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

# Create multi-index from product of iterables
index_tuples = []
data_values = []

for company in companies:
    for year in years:
        for quarter in quarters:
            for metric in metrics:
                index_tuples.append((company, year, quarter, metric))
                if metric == 'Revenue':
                    base_value = np.random.normal(50000, 10000)
                elif metric == 'Profit':
                    base_value = np.random.normal(10000, 3000)
                elif metric == 'Employees':
                    base_value = np.random.normal(100000, 20000)
                else:  # Market_Cap
                    base_value = np.random.normal(1000000, 200000)
                
                # Add some trend and seasonality
                trend = year - 2020
                seasonal = 0.1 if quarter in ['Q1', 'Q4'] else -0.05
                value = base_value * (1 + 0.1 * trend + seasonal)
                data_values.append(max(value, 0))

# Create MultiIndex
multi_index = pd.MultiIndex.from_tuples(index_tuples, 
                                       names=['Company', 'Year', 'Quarter', 'Metric'])

# Create Series with MultiIndex
multi_series = pd.Series(data_values, index=multi_index, name='Value')

print("Multi-index Series structure:")
print(f"Index levels: {multi_series.index.nlevels}")
print(f"Index names: {multi_series.index.names}")
print(multi_series.head(10))

# Convert to DataFrame for easier manipulation
multi_df = multi_series.unstack('Metric')

print("\nMulti-index DataFrame:")
print(multi_df.head())
print(f"Shape: {multi_df.shape}")

# 1. Indexing and selection with MultiIndex
print("\n1. Multi-index selection:")

# Select single company
apple_data = multi_df.loc['Apple']
print("Apple data:")
print(apple_data.head())

# Select multiple companies
tech_giants = multi_df.loc[['Apple', 'Google']]
print(f"\nApple and Google data shape: {tech_giants.shape}")

# Cross-section selection
q1_2023_data = multi_df.xs((2023, 'Q1'), level=['Year', 'Quarter'])
print("\nQ1 2023 data for all companies:")
print(q1_2023_data)

# Boolean indexing with MultiIndex
high_revenue = multi_df[multi_df['Revenue'] > 60000]
print(f"\nHigh revenue periods: {len(high_revenue)}")
print(high_revenue[['Revenue', 'Profit']].head())

# 2. Level operations
print("\n2. Level operations:")

# Swap levels
swapped_df = multi_df.swaplevel('Year', 'Quarter').sort_index()
print("After swapping Year and Quarter levels:")
print(swapped_df.head())

# Drop level
single_level_df = multi_df.droplevel('Quarter')
print(f"\nAfter dropping Quarter level - shape: {single_level_df.shape}")

# Reset index
flat_df = multi_df.reset_index()
print(f"Flattened DataFrame shape: {flat_df.shape}")
print(flat_df.head())

# 3. Aggregations with MultiIndex
print("\n3. Aggregations with MultiIndex:")

# Aggregate across levels
yearly_avg = multi_df.groupby(level=['Company', 'Year']).mean()
print("Yearly averages:")
print(yearly_avg.head())

# Aggregate across companies
company_avg = multi_df.groupby(level='Company').mean()
print("\nCompany averages:")
print(company_avg)

# Custom aggregations
def growth_rate(series):
    if len(series) > 1:
        return (series.iloc[-1] / series.iloc[0] - 1) * 100
    return 0

revenue_growth = multi_df.groupby(level='Company')['Revenue'].apply(growth_rate)
print("\nRevenue growth by company:")
print(revenue_growth.round(2))

# 4. Stacking and unstacking
print("\n4. Stacking and unstacking:")

# Stack to move column level to index
stacked_df = multi_df.stack()
print(f"Stacked shape: {stacked_df.shape}")
print(stacked_df.head())

# Unstack specific level
unstacked_company = multi_df.unstack('Company')
print(f"\nUnstacked by Company - shape: {unstacked_company.shape}")
print(unstacked_company.head())

# Partial unstacking
partial_unstack = multi_df.unstack(['Year', 'Quarter'])
print(f"Partial unstack shape: {partial_unstack.shape}")

# 5. Multi-index with time series
print("\n5. Multi-index time series:")

# Create time-based multi-index
dates = pd.date_range('2023-01-01', '2024-12-31', freq='M')
companies_ts = ['AAPL', 'GOOGL', 'MSFT']
metrics_ts = ['Price', 'Volume']

ts_index = pd.MultiIndex.from_product([dates, companies_ts, metrics_ts],
                                     names=['Date', 'Ticker', 'Metric'])

ts_values = np.random.randn(len(ts_index))
multi_ts = pd.Series(ts_values, index=ts_index)

# Unstack for time series analysis
ts_df = multi_ts.unstack(['Ticker', 'Metric'])
print("Time series with multi-index:")
print(ts_df.head())

# Resample multi-index time series
quarterly_ts = ts_df.resample('Q').mean()
print("\nQuarterly resampled multi-index data:")
print(quarterly_ts.head())

# 6. Performance with MultiIndex
print("\n6. Performance considerations:")

# Sorting index for better performance
sorted_multi_df = multi_df.sort_index()
print("Index is monotonic after sorting:", sorted_multi_df.index.is_monotonic_increasing)

# Using query with MultiIndex (after reset_index)
flat_for_query = multi_df.reset_index()
query_result = flat_for_query.query("Company == 'Apple' and Year >= 2022 and Revenue > 45000")
print(f"\nQuery result shape: {query_result.shape}")
print(query_result[['Company', 'Year', 'Quarter', 'Revenue']].head())
```

### Custom Functions and Apply Operations

```python
# Create complex dataset for custom function examples
np.random.seed(42)

employee_data = pd.DataFrame({
    'employee_id': range(1, 1001),
    'name': [f'Employee_{i}' for i in range(1, 1001)],
    'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], 1000),
    'level': np.random.choice(['Junior', 'Mid', 'Senior', 'Lead', 'Manager'], 1000),
    'salary': np.random.normal(75000, 25000, 1000),
    'bonus': np.random.normal(7500, 3000, 1000),
    'performance_score': np.random.uniform(1, 10, 1000),
    'years_experience': np.random.randint(0, 25, 1000),
    'projects_completed': np.random.poisson(5, 1000),
    'training_hours': np.random.exponential(20, 1000),
    'hire_date': pd.date_range('2010-01-01', periods=1000, freq='D'),
    'birth_date': pd.date_range('1960-01-01', periods=1000, freq='W')
})

# Ensure non-negative values
employee_data['salary'] = np.abs(employee_data['salary'])
employee_data['bonus'] = np.abs(employee_data['bonus'])

print("Employee dataset for custom functions:")
print(employee_data.head())
print(f"Shape: {employee_data.shape}")

# 1. Element-wise custom functions
print("\n1. Element-wise custom functions:")

# Custom salary categorization
def categorize_salary(salary):
    """Categorize salary into bands."""
    if salary < 50000:
        return 'Low'
    elif salary < 75000:
        return 'Medium-Low'
    elif salary < 100000:
        return 'Medium-High'
    elif salary < 125000:
        return 'High'
    else:
        return 'Very High'

employee_data['salary_category'] = employee_data['salary'].apply(categorize_salary)

# Performance rating with complex logic
def calculate_performance_rating(score, projects, training):
    """Calculate comprehensive performance rating."""
    base_rating = score / 2  # Scale down from 10 to 5
    
    # Adjust for project completion
    if projects >= 8:
        base_rating += 0.5
    elif projects >= 5:
        base_rating += 0.2
    elif projects < 3:
        base_rating -= 0.3
    
    # Adjust for training hours
    if training >= 30:
        base_rating += 0.3
    elif training >= 20:
        base_rating += 0.1
    elif training < 10:
        base_rating -= 0.2
    
    return min(max(base_rating, 1), 5)  # Clamp between 1 and 5

employee_data['comprehensive_rating'] = employee_data.apply(
    lambda row: calculate_performance_rating(
        row['performance_score'], 
        row['projects_completed'], 
        row['training_hours']
    ), axis=1
)

print("Custom function results:")
print(employee_data[['salary', 'salary_category', 'performance_score', 
                    'projects_completed', 'comprehensive_rating']].head())

# 2. Row-wise operations
print("\n2. Row-wise operations:")

def employee_summary(row):
    """Generate comprehensive employee summary."""
    total_comp = row['salary'] + row['bonus']
    years_service = (pd.Timestamp.now() - row['hire_date']).days / 365.25
    age = (pd.Timestamp.now() - row['birth_date']).days / 365.25
    
    summary = {
        'total_compensation': total_comp,
        'years_of_service': years_service,
        'age': age,
        'compensation_per_year': total_comp / max(years_service, 1),
        'projects_per_year': row['projects_completed'] / max(row['years_experience'], 1),
        'is_high_performer': row['comprehensive_rating'] >= 4.0,
        'retention_risk': 'High' if (years_service > 10 and row['comprehensive_rating'] < 3.5) else 'Low'
    }
    return pd.Series(summary)

# Apply row-wise function
employee_summary_df = employee_data.apply(employee_summary, axis=1)
employee_combined = pd.concat([employee_data, employee_summary_df], axis=1)

print("Row-wise operation results:")
print(employee_combined[['name', 'total_compensation', 'years_of_service', 
                        'age', 'is_high_performer', 'retention_risk']].head())

# 3. Group-wise custom functions
print("\n3. Group-wise custom functions:")

def department_analysis(group):
    """Comprehensive department analysis."""
    return pd.Series({
        'headcount': len(group),
        'avg_salary': group['salary'].mean(),
        'salary_std': group['salary'].std(),
        'top_performer_salary': group.loc[group['comprehensive_rating'].idxmax(), 'salary'],
        'avg_experience': group['years_experience'].mean(),
        'high_performer_rate': (group['comprehensive_rating'] >= 4.0).mean(),
        'avg_projects': group['projects_completed'].mean(),
        'total_training_hours': group['training_hours'].sum(),
        'retention_risk_count': (group['retention_risk'] == 'High').sum(),
        'salary_budget': group['salary'].sum()
    })

dept_analysis = employee_combined.groupby('department').apply(department_analysis)
print("Department analysis:")
print(dept_analysis.round(2))

# 4. Custom aggregation functions
print("\n4. Custom aggregation functions:")

def salary_percentile_range(series):
    """Calculate salary percentile range."""
    p90 = series.quantile(0.9)
    p10 = series.quantile(0.1)
    return p90 - p10

def performance_consistency(series):
    """Calculate performance consistency score."""
    return 1 / (1 + series.std())  # Higher score for more consistent performance

custom_agg = employee_combined.groupby('level').agg({
    'salary': [salary_percentile_range, 'mean', 'std'],
    'comprehensive_rating': [performance_consistency, 'mean'],
    'years_experience': 'mean',
    'projects_completed': 'sum'
})

print("Custom aggregation by level:")
print(custom_agg.round(3))

# 5. Transform operations with custom functions
print("\n5. Transform operations:")

def standardize_within_group(series):
    """Standardize values within group."""
    return (series - series.mean()) / series.std()

def rank_within_group(series):
    """Rank within group (1 = highest)."""
    return series.rank(ascending=False)

# Apply transforms
employee_combined['salary_dept_zscore'] = (employee_combined
                                          .groupby('department')['salary']
                                          .transform(standardize_within_group))

employee_combined['performance_dept_rank'] = (employee_combined
                                             .groupby('department')['comprehensive_rating']
                                             .transform(rank_within_group))

employee_combined['salary_level_percentile'] = (employee_combined
                                               .groupby('level')['salary']
                                               .transform(lambda x: x.rank(pct=True)))

print("Transform operations:")
print(employee_combined[['name', 'department', 'level', 'salary', 
                        'salary_dept_zscore', 'performance_dept_rank', 
                        'salary_level_percentile']].head(10))

# 6. Complex filtering with custom functions
print("\n6. Complex filtering:")

def is_promotion_candidate(group):
    """Identify promotion candidates within each department."""
    # Criteria: top 20% performance, above median experience, high project completion
    performance_threshold = group['comprehensive_rating'].quantile(0.8)
    experience_threshold = group['years_experience'].median()
    projects_threshold = group['projects_completed'].quantile(0.7)
    
    candidates = group[
        (group['comprehensive_rating'] >= performance_threshold) &
        (group['years_experience'] >= experience_threshold) &
        (group['projects_completed'] >= projects_threshold)
    ]
    
    return len(candidates) > 0

# Apply filter
promotion_candidates = (employee_combined
                       .groupby('department')
                       .filter(is_promotion_candidate))

promotion_ready = (employee_combined
                  .groupby('department')
                  .apply(lambda g: g[
                      (g['comprehensive_rating'] >= g['comprehensive_rating'].quantile(0.8)) &
                      (g['years_experience'] >= g['years_experience'].median()) &
                      (g['projects_completed'] >= g['projects_completed'].quantile(0.7))
                  ]))

print(f"Promotion candidates identified: {len(promotion_ready)}")
print("Sample promotion candidates:")
if len(promotion_ready) > 0:
    print(promotion_ready[['name', 'department', 'level', 'comprehensive_rating', 
                          'years_experience', 'projects_completed']].head())

# 7. Performance optimization for custom functions
print("\n7. Performance optimization:")

# Vectorized version vs. apply
def slow_calculation(row):
    """Intentionally slow calculation."""
    result = 0
    for i in range(100):
        result += row['salary'] * 0.01
    return result

def vectorized_calculation(salary_series):
    """Vectorized equivalent."""
    return salary_series * 1.0  # Equivalent to the loop above

# Timing comparison (commented out to avoid long execution)
# import time
# 
# start_time = time.time()
# slow_result = employee_combined.apply(slow_calculation, axis=1)
# slow_time = time.time() - start_time
# 
# start_time = time.time()
# fast_result = vectorized_calculation(employee_combined['salary'])
# fast_time = time.time() - start_time
# 
# print(f"Slow method time: {slow_time:.4f} seconds")
# print(f"Fast method time: {fast_time:.4f} seconds")
# print(f"Speedup: {slow_time / fast_time:.1f}x")

print("Performance optimization techniques demonstrated")
print("Always prefer vectorized operations when possible!")
```

## Performance Optimization

### Memory Usage and Data Types

```python
import psutil
import os

# Get memory usage function
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Create large dataset to demonstrate optimization
print("Memory Optimization Techniques")
print("=" * 50)

initial_memory = get_memory_usage()
print(f"Initial memory usage: {initial_memory:.2f} MB")

# Create large dataset
np.random.seed(42)
n_rows = 100000

# Inefficient data types (default behavior)
inefficient_df = pd.DataFrame({
    'id': range(n_rows),  # int64 by default
    'category': np.random.choice(['A', 'B', 'C'], n_rows),  # object
    'subcategory': np.random.choice(['X', 'Y', 'Z'], n_rows),  # object
    'value': np.random.randn(n_rows),  # float64
    'flag': np.random.choice([True, False], n_rows),  # bool
    'small_int': np.random.randint(0, 100, n_rows),  # int64
    'price': np.random.uniform(10, 1000, n_rows),  # float64
    'rating': np.random.randint(1, 6, n_rows)  # int64 (but could be smaller)
})

after_creation_memory = get_memory_usage()
print(f"Memory after creating inefficient DataFrame: {after_creation_memory:.2f} MB")
print(f"DataFrame memory usage: {(after_creation_memory - initial_memory):.2f} MB")

# 1. Data type optimization
print("\n1. Data Type Optimization:")
print("Original data types and memory usage:")
print(inefficient_df.dtypes)
print(f"Memory usage: {inefficient_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Optimize data types
efficient_df = inefficient_df.copy()

# Convert to categorical for low-cardinality strings
efficient_df['category'] = efficient_df['category'].astype('category')
efficient_df['subcategory'] = efficient_df['subcategory'].astype('category')

# Use smaller integer types
efficient_df['small_int'] = efficient_df['small_int'].astype('int8')  # 0-100 fits in int8
efficient_df['rating'] = efficient_df['rating'].astype('int8')  # 1-5 fits in int8

# Use float32 instead of float64 when precision allows
efficient_df['value'] = efficient_df['value'].astype('float32')
efficient_df['price'] = efficient_df['price'].astype('float32')

# ID column optimization (if needed for indexing)
if efficient_df['id'].max() < 2**31:
    efficient_df['id'] = efficient_df['id'].astype('int32')

print("\nOptimized data types and memory usage:")
print(efficient_df.dtypes)
print(f"Memory usage: {efficient_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Calculate savings
original_memory = inefficient_df.memory_usage(deep=True).sum()
optimized_memory = efficient_df.memory_usage(deep=True).sum()
savings = (original_memory - optimized_memory) / original_memory * 100

print(f"Memory savings: {savings:.1f}%")

# 2. Chunking for large datasets
print("\n2. Chunking techniques:")

def process_in_chunks(df, chunk_size=10000):
    """Process DataFrame in chunks to reduce memory usage."""
    results = []
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        
        # Example processing: calculate statistics for each chunk
        chunk_stats = {
            'chunk_start': i,
            'chunk_end': min(i + chunk_size, len(df)),
            'mean_value': chunk['value'].mean(),
            'mean_price': chunk['price'].mean(),
            'category_a_count': (chunk['category'] == 'A').sum()
        }
        results.append(chunk_stats)
    
    return pd.DataFrame(results)

chunk_results = process_in_chunks(efficient_df)
print("Chunk processing results:")
print(chunk_results.head())

# 3. Using iterators for memory efficiency
print("\n3. Iterator-based processing:")

def calculate_rolling_stats_iterator(df, window=1000):
    """Calculate rolling statistics using iterator approach."""
    stats_list = []
    
    for i in range(window, len(df), window//2):  # 50% overlap
        window_data = df.iloc[i-window:i]
        
        stats = {
            'end_index': i,
            'mean_value': window_data['value'].mean(),
            'std_value': window_data['value'].std(),
            'mean_price': window_data['price'].mean(),
            'category_distribution': window_data['category'].value_counts().to_dict()
        }
        stats_list.append(stats)
    
    return stats_list

iterator_stats = calculate_rolling_stats_iterator(efficient_df)
print(f"Iterator processing completed: {len(iterator_stats)} windows processed")
print("Sample iterator result:")
print(iterator_stats[0])
```

### Performance Benchmarking

```python
import time
from functools import wraps

def benchmark(func):
    """Decorator to benchmark function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Create test datasets
print("\nPerformance Benchmarking:")
print("=" * 30)

# Small and large datasets for comparison
small_df = pd.DataFrame({
    'A': np.random.randn(1000),
    'B': np.random.randn(1000),
    'C': np.random.choice(['X', 'Y', 'Z'], 1000),
    'D': np.random.randint(1, 100, 1000)
})

large_df = pd.DataFrame({
    'A': np.random.randn(100000),
    'B': np.random.randn(100000),
    'C': np.random.choice(['X', 'Y', 'Z'], 100000),
    'D': np.random.randint(1, 100, 100000)
})

# 1. Vectorized operations vs loops
print("\n1. Vectorized vs Loop Operations:")

@benchmark
def slow_loop_calculation(df):
    """Slow loop-based calculation."""
    result = []
    for idx, row in df.iterrows():
        calc = row['A'] * row['B'] + row['D']
        result.append(calc)
    return result

@benchmark
def fast_vectorized_calculation(df):
    """Fast vectorized calculation."""
    return df['A'] * df['B'] + df['D']

@benchmark
def medium_apply_calculation(df):
    """Medium speed apply calculation."""
    return df.apply(lambda row: row['A'] * row['B'] + row['D'], axis=1)

print("Small dataset (1K rows):")
_ = slow_loop_calculation(small_df)
_ = medium_apply_calculation(small_df)
_ = fast_vectorized_calculation(small_df)

print("\nLarge dataset (100K rows):")
# Skip slow loop for large dataset to save time
# _ = slow_loop_calculation(large_df)  # Too slow!
_ = medium_apply_calculation(large_df)
_ = fast_vectorized_calculation(large_df)

# 2. Efficient filtering techniques
print("\n2. Filtering Techniques:")

@benchmark
def slow_filtering(df):
    """Slow iterative filtering."""
    result = df[df.apply(lambda row: row['A'] > 0 and row['C'] == 'X', axis=1)]
    return len(result)

@benchmark
def fast_filtering(df):
    """Fast boolean indexing."""
    result = df[(df['A'] > 0) & (df['C'] == 'X')]
    return len(result)

@benchmark
def query_filtering(df):
    """Query-based filtering."""
    result = df.query("A > 0 and C == 'X'")
    return len(result)

print("Filtering performance:")
print(f"Slow filtering result: {slow_filtering(small_df)} rows")
print(f"Fast filtering result: {fast_filtering(small_df)} rows") 
print(f"Query filtering result: {query_filtering(small_df)} rows")

# 3. GroupBy performance
print("\n3. GroupBy Performance:")

@benchmark
def groupby_multiple_agg(df):
    """Multiple aggregations in groupby."""
    return df.groupby('C').agg({
        'A': ['mean', 'std'],
        'B': ['sum', 'count'],
        'D': 'max'
    })

@benchmark
def groupby_single_then_combine(df):
    """Single aggregations then combine."""
    result1 = df.groupby('C')['A'].agg(['mean', 'std'])
    result2 = df.groupby('C')['B'].agg(['sum', 'count'])
    result3 = df.groupby('C')['D'].max()
    return pd.concat([result1, result2, result3], axis=1)

@benchmark
def groupby_custom_agg(df):
    """Custom aggregation function."""
    def custom_stats(group):
        return pd.Series({
            'A_mean': group['A'].mean(),
            'A_std': group['A'].std(),
            'B_sum': group['B'].sum(),
            'B_count': len(group),
            'D_max': group['D'].max()
        })
    return df.groupby('C').apply(custom_stats)

print("GroupBy performance comparison:")
result1 = groupby_multiple_agg(large_df)
result2 = groupby_single_then_combine(large_df)
result3 = groupby_custom_agg(large_df)

# 4. Memory-efficient operations
print("\n4. Memory-Efficient Operations:")

@benchmark
def memory_inefficient_merge(df1, df2):
    """Creates many intermediate DataFrames."""
    temp1 = df1.copy()
    temp2 = df2.copy()
    temp3 = temp1.merge(temp2, on='key', how='left')
    temp4 = temp3.dropna()
    temp5 = temp4.groupby('category').mean()
    return temp5

@benchmark
def memory_efficient_merge(df1, df2):
    """Chain operations to minimize intermediate objects."""
    return (df1.merge(df2, on='key', how='left')
           .dropna()
           .groupby('category')
           .mean())

# Create test data for merge
df1_test = pd.DataFrame({
    'key': range(10000),
    'category': np.random.choice(['A', 'B', 'C'], 10000),
    'value1': np.random.randn(10000)
})

df2_test = pd.DataFrame({
    'key': range(5000, 15000),  # Partial overlap
    'value2': np.random.randn(10000)
})

print("Memory efficiency comparison:")
result_inefficient = memory_inefficient_merge(df1_test, df2_test)
result_efficient = memory_efficient_merge(df1_test, df2_test)

# 5. Index optimization
print("\n5. Index Optimization:")

# Create dataset for indexing tests
index_test_df = pd.DataFrame({
    'id': range(50000),
    'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 50000),
    'value': np.random.randn(50000),
    'timestamp': pd.date_range('2023-01-01', periods=50000, freq='min')
})

@benchmark
def search_without_index(df):
    """Search without index (slow)."""
    results = []
    for search_id in [100, 1000, 5000, 10000]:
        result = df[df['id'] == search_id]
        results.append(len(result))
    return sum(results)

@benchmark
def search_with_index(df):
    """Search with index (fast)."""
    df_indexed = df.set_index('id')
    results = []
    for search_id in [100, 1000, 5000, 10000]:
        result = df_indexed.loc[[search_id]]
        results.append(len(result))
    return sum(results)

print("Index optimization:")
no_index_result = search_without_index(index_test_df)
with_index_result = search_with_index(index_test_df)

# 6. String operations optimization
print("\n6. String Operations:")

string_df = pd.DataFrame({
    'text': ['hello world', 'pandas optimization', 'performance testing'] * 10000
})

@benchmark
def slow_string_ops(df):
    """Slow string operations."""
    result = df['text'].apply(lambda x: x.upper().replace(' ', '_'))
    return len(result)

@benchmark
def fast_string_ops(df):
    """Fast vectorized string operations."""
    result = df['text'].str.upper().str.replace(' ', '_')
    return len(result)

print("String operations:")
slow_result = slow_string_ops(string_df)
fast_result = fast_string_ops(string_df)

## Real-World Examples

### Complete Data Analysis Project

```python
# Comprehensive data analysis project demonstrating pandas capabilities
print("Real-World Data Analysis Project")
print("=" * 50)

# Generate realistic sales dataset
np.random.seed(42)
n_records = 10000

# Create realistic sales data
sales_data = pd.DataFrame({
    'order_id': range(1, n_records + 1),
    'customer_id': np.random.randint(1, 2000, n_records),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_records),
    'product_name': [f'Product_{np.random.randint(1, 1000)}' for _ in range(n_records)],
    'quantity': np.random.randint(1, 10, n_records),
    'unit_price': np.random.exponential(50, n_records) + 10,
    'order_date': pd.date_range('2022-01-01', periods=n_records, freq='H'),
    'customer_age': np.random.randint(18, 80, n_records),
    'customer_gender': np.random.choice(['M', 'F'], n_records),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
    'sales_rep': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'], n_records),
    'shipping_cost': np.random.exponential(10, n_records),
    'discount_pct': np.random.choice([0, 5, 10, 15, 20], n_records, p=[0.4, 0.3, 0.15, 0.1, 0.05])
})

# Calculate derived fields
sales_data['gross_amount'] = sales_data['quantity'] * sales_data['unit_price']
sales_data['discount_amount'] = sales_data['gross_amount'] * (sales_data['discount_pct'] / 100)
sales_data['net_amount'] = sales_data['gross_amount'] - sales_data['discount_amount']
sales_data['total_amount'] = sales_data['net_amount'] + sales_data['shipping_cost']

print("Sales dataset created:")
print(sales_data.head())
print(f"Dataset shape: {sales_data.shape}")

# 1. Data Quality Assessment
print("\n1. Data Quality Assessment:")

def data_quality_report(df):
    """Generate comprehensive data quality report."""
    report = []
    
    for column in df.columns:
        col_report = {
            'column': column,
            'dtype': str(df[column].dtype),
            'non_null_count': df[column].count(),
            'null_count': df[column].isnull().sum(),
            'null_percentage': (df[column].isnull().sum() / len(df)) * 100,
            'unique_count': df[column].nunique(),
            'unique_percentage': (df[column].nunique() / len(df)) * 100
        }
        
        if pd.api.types.is_numeric_dtype(df[column]):
            col_report.update({
                'mean': df[column].mean(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'zeros': (df[column] == 0).sum(),
                'negative': (df[column] < 0).sum()
            })
        
        report.append(col_report)
    
    return pd.DataFrame(report)

quality_report = data_quality_report(sales_data)
print("Data quality summary:")
print(quality_report[['column', 'dtype', 'null_count', 'unique_count']].head(10))

# 2. Business Metrics Calculation
print("\n2. Business Metrics Calculation:")

# Key business metrics
total_revenue = sales_data['net_amount'].sum()
total_orders = len(sales_data)
average_order_value = sales_data['net_amount'].mean()
total_customers = sales_data['customer_id'].nunique()
customer_lifetime_value = total_revenue / total_customers

print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Total Orders: {total_orders:,}")
print(f"Average Order Value: ${average_order_value:.2f}")
print(f"Total Customers: {total_customers:,}")
print(f"Customer Lifetime Value: ${customer_lifetime_value:.2f}")

# Monthly trends
sales_data['year_month'] = sales_data['order_date'].dt.to_period('M')
monthly_metrics = sales_data.groupby('year_month').agg({
    'net_amount': ['sum', 'mean', 'count'],
    'customer_id': 'nunique',
    'quantity': 'sum'
}).round(2)

monthly_metrics.columns = ['total_revenue', 'avg_order_value', 'order_count', 
                          'unique_customers', 'total_quantity']
print("\nMonthly trends:")
print(monthly_metrics.head())

# 3. Customer Segmentation
print("\n3. Customer Segmentation:")

# RFM Analysis (Recency, Frequency, Monetary)
reference_date = sales_data['order_date'].max()

rfm_data = sales_data.groupby('customer_id').agg({
    'order_date': lambda x: (reference_date - x.max()).days,  # Recency
    'order_id': 'count',  # Frequency
    'net_amount': 'sum'   # Monetary
}).rename(columns={
    'order_date': 'recency',
    'order_id': 'frequency', 
    'net_amount': 'monetary'
})

# Create RFM scores (1-5 scale)
rfm_data['recency_score'] = pd.qcut(rfm_data['recency'], 5, labels=[5,4,3,2,1])
rfm_data['frequency_score'] = pd.qcut(rfm_data['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm_data['monetary_score'] = pd.qcut(rfm_data['monetary'], 5, labels=[1,2,3,4,5])

# Combined RFM score
rfm_data['rfm_score'] = (rfm_data['recency_score'].astype(str) + 
                        rfm_data['frequency_score'].astype(str) + 
                        rfm_data['monetary_score'].astype(str))

# Customer segments based on RFM
def segment_customers(row):
    """Segment customers based on RFM scores."""
    if row['rfm_score'] in ['555', '554', '544', '545', '454', '455', '445']:
        return 'Champions'
    elif row['rfm_score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
        return 'Loyal Customers'
    elif row['rfm_score'] in ['512', '511', '422', '421', '412', '411']:
        return 'Potential Loyalists'
    elif row['rfm_score'] in ['533', '532', '531', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313']:
        return 'New Customers'
    elif row['rfm_score'] in ['155', '154', '144', '214', '215', '115', '114']:
        return 'At Risk'
    elif row['rfm_score'] in ['155', '154', '144', '214', '215', '115']:
        return 'Cannot Lose Them'
    else:
        return 'Others'

rfm_data['customer_segment'] = rfm_data.apply(segment_customers, axis=1)

segment_summary = rfm_data.groupby('customer_segment').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean'
}).round(2)

segment_counts = rfm_data['customer_segment'].value_counts()

print("Customer segmentation results:")
print(segment_counts)
print("\nSegment characteristics:")
print(segment_summary)

# 4. Product Performance Analysis
print("\n4. Product Performance Analysis:")

product_performance = sales_data.groupby('product_category').agg({
    'net_amount': ['sum', 'mean'],
    'quantity': 'sum',
    'order_id': 'count',
    'customer_id': 'nunique'
}).round(2)

product_performance.columns = ['total_revenue', 'avg_order_value', 
                              'total_quantity', 'total_orders', 'unique_customers']

# Calculate market share
product_performance['revenue_share'] = (
    product_performance['total_revenue'] / product_performance['total_revenue'].sum() * 100
).round(2)

# Sort by total revenue
product_performance = product_performance.sort_values('total_revenue', ascending=False)

print("Product category performance:")
print(product_performance)

# 5. Sales Representative Performance
print("\n5. Sales Representative Performance:")

rep_performance = sales_data.groupby('sales_rep').agg({
    'net_amount': ['sum', 'mean'],
    'order_id': 'count',
    'customer_id': 'nunique',
    'discount_amount': 'sum'
}).round(2)

rep_performance.columns = ['total_revenue', 'avg_order_value', 
                          'total_orders', 'unique_customers', 'total_discounts']

# Calculate performance metrics
rep_performance['revenue_per_order'] = (
    rep_performance['total_revenue'] / rep_performance['total_orders']
).round(2)

rep_performance['customers_per_order'] = (
    rep_performance['unique_customers'] / rep_performance['total_orders']
).round(3)

rep_performance = rep_performance.sort_values('total_revenue', ascending=False)

print("Sales representative performance:")
print(rep_performance)

# 6. Time-based Analysis
print("\n6. Time-based Analysis:")

# Add time components
sales_data['hour'] = sales_data['order_date'].dt.hour
sales_data['day_of_week'] = sales_data['order_date'].dt.day_name()
sales_data['month'] = sales_data['order_date'].dt.month_name()

# Hourly patterns
hourly_sales = sales_data.groupby('hour').agg({
    'net_amount': ['sum', 'mean'],
    'order_id': 'count'
}).round(2)
hourly_sales.columns = ['total_revenue', 'avg_order_value', 'order_count']

print("Peak hours analysis:")
print(hourly_sales.nlargest(5, 'total_revenue'))

# Day of week patterns
dow_sales = sales_data.groupby('day_of_week').agg({
    'net_amount': ['sum', 'mean'],
    'order_id': 'count'
}).round(2)
dow_sales.columns = ['total_revenue', 'avg_order_value', 'order_count']

# Reorder by actual day sequence
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_sales = dow_sales.reindex(day_order)

print("\nDay of week patterns:")
print(dow_sales)

# 7. Regional Analysis
print("\n7. Regional Analysis:")

regional_analysis = sales_data.groupby('region').agg({
    'net_amount': ['sum', 'mean'],
    'customer_id': 'nunique',
    'order_id': 'count',
    'unit_price': 'mean',
    'discount_pct': 'mean'
}).round(2)

regional_analysis.columns = ['total_revenue', 'avg_order_value', 'unique_customers', 
                           'total_orders', 'avg_unit_price', 'avg_discount']

# Add market share
regional_analysis['market_share'] = (
    regional_analysis['total_revenue'] / regional_analysis['total_revenue'].sum() * 100
).round(2)

print("Regional performance:")
print(regional_analysis)

# 8. Customer Demographics Analysis
print("\n8. Demographics Analysis:")

# Age group analysis
sales_data['age_group'] = pd.cut(sales_data['customer_age'], 
                               bins=[0, 25, 35, 45, 55, 100], 
                               labels=['18-25', '26-35', '36-45', '46-55', '55+'])

demo_analysis = sales_data.groupby(['age_group', 'customer_gender']).agg({
    'net_amount': ['sum', 'mean'],
    'order_id': 'count',
    'customer_id': 'nunique'
}).round(2)

demo_analysis.columns = ['total_revenue', 'avg_order_value', 'total_orders', 'unique_customers']

print("Demographics analysis (by age group and gender):")
print(demo_analysis)

# 9. Cohort Analysis (simplified)
print("\n9. Customer Cohort Analysis:")

# First purchase date for each customer
customer_first_purchase = sales_data.groupby('customer_id')['order_date'].min().reset_index()
customer_first_purchase.columns = ['customer_id', 'first_purchase_date']
customer_first_purchase['first_purchase_month'] = customer_first_purchase['first_purchase_date'].dt.to_period('M')

# Merge back with sales data
sales_with_cohort = sales_data.merge(customer_first_purchase, on='customer_id')
sales_with_cohort['order_month'] = sales_with_cohort['order_date'].dt.to_period('M')

# Calculate period number (months since first purchase)
sales_with_cohort['period_number'] = (
    sales_with_cohort['order_month'] - sales_with_cohort['first_purchase_month']
).apply(attrgetter('n'))

# Cohort table
cohort_data = sales_with_cohort.groupby(['first_purchase_month', 'period_number'])['customer_id'].nunique().reset_index()
cohort_counts = cohort_data.pivot(index='first_purchase_month', 
                                columns='period_number', 
                                values='customer_id')

# Calculate cohort sizes
cohort_sizes = customer_first_purchase.groupby('first_purchase_month')['customer_id'].nunique()

# Calculate retention rates
cohort_table = cohort_counts.divide(cohort_sizes, axis=0)

print("Customer cohort retention rates (first 6 months):")
print(cohort_table.iloc[:5, :6].round(3))  # First 5 cohorts, first 6 periods

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("Key insights and recommendations can be derived from the above analysis")
```

### E-commerce Dashboard Data Preparation

```python
# Prepare data for dashboard/visualization
print("\nE-commerce Dashboard Data Preparation")
print("=" * 50)

# 1. Key Performance Indicators (KPIs)
def calculate_kpis(df, comparison_period_days=30):
    """Calculate key performance indicators with period comparison."""
    
    # Current period (last 30 days)
    end_date = df['order_date'].max()
    current_start = end_date - pd.Timedelta(days=comparison_period_days)
    current_data = df[df['order_date'] >= current_start]
    
    # Previous period (30 days before current period)
    previous_start = current_start - pd.Timedelta(days=comparison_period_days)
    previous_end = current_start
    previous_data = df[(df['order_date'] >= previous_start) & (df['order_date'] < previous_end)]
    
    def period_metrics(data):
        return {
            'total_revenue': data['net_amount'].sum(),
            'total_orders': len(data),
            'unique_customers': data['customer_id'].nunique(),
            'avg_order_value': data['net_amount'].mean(),
            'conversion_rate': data['customer_id'].nunique() / len(data) if len(data) > 0 else 0
        }
    
    current_metrics = period_metrics(current_data)
    previous_metrics = period_metrics(previous_data)
    
    # Calculate percentage changes
    kpis = {}
    for metric in current_metrics.keys():
        current_val = current_metrics[metric]
        previous_val = previous_metrics[metric]
        
        if previous_val != 0:
            pct_change = ((current_val - previous_val) / previous_val) * 100
        else:
            pct_change = 0
        
        kpis[metric] = {
            'current': current_val,
            'previous': previous_val,
            'change_pct': pct_change,
            'trend': 'up' if pct_change > 0 else 'down' if pct_change < 0 else 'flat'
        }
    
    return kpis

dashboard_kpis = calculate_kpis(sales_data)

print("Dashboard KPIs:")
for metric, values in dashboard_kpis.items():
    print(f"{metric.replace('_', ' ').title()}:")
    print(f"  Current: {values['current']:,.2f}")
    print(f"  Previous: {values['previous']:,.2f}")
    print(f"  Change: {values['change_pct']:+.1f}% ({values['trend']})")
    print()

# 2. Time series data for charts
daily_sales = sales_data.groupby(sales_data['order_date'].dt.date).agg({
    'net_amount': 'sum',
    'order_id': 'count',
    'customer_id': 'nunique'
}).reset_index()

daily_sales.columns = ['date', 'revenue', 'orders', 'customers']
daily_sales['date'] = pd.to_datetime(daily_sales['date'])

print("Daily sales trend (last 10 days):")
print(daily_sales.tail(10))

# 3. Product category performance for pie charts
category_performance = sales_data.groupby('product_category').agg({
    'net_amount': 'sum',
    'order_id': 'count'
}).reset_index()

category_performance.columns = ['category', 'revenue', 'orders']
category_performance['revenue_pct'] = (
    category_performance['revenue'] / category_performance['revenue'].sum() * 100
).round(2)

print("\nCategory performance for charts:")
print(category_performance)

# 4. Geographic data for maps
geo_data = sales_data.groupby('region').agg({
    'net_amount': ['sum', 'mean'],
    'customer_id': 'nunique',
    'order_id': 'count'
}).round(2)

geo_data.columns = ['total_revenue', 'avg_revenue', 'customers', 'orders']
geo_data = geo_data.reset_index()

print("\nGeographic data:")
print(geo_data)

# 5. Customer segments for targeted marketing
segment_summary = rfm_data.groupby('customer_segment').agg({
    'monetary': ['sum', 'mean', 'count'],
    'frequency': 'mean',
    'recency': 'mean'
}).round(2)

segment_summary.columns = ['total_value', 'avg_value', 'customer_count', 'avg_frequency', 'avg_recency']
segment_summary = segment_summary.reset_index()

print("\nCustomer segments summary:")
print(segment_summary)

print("\nDashboard data preparation complete!")
print("Data is ready for visualization in tools like:")
print("- Matplotlib/Seaborn for static charts")
print("- Plotly for interactive dashboards") 
print("- Streamlit/Dash for web applications")
print("- Tableau/Power BI for business intelligence")
```

## Best Practices

### Code Organization and Style

```python
# Best practices for pandas code organization and style
print("Pandas Best Practices")
print("=" * 30)

# 1. Import conventions
print("1. Import Conventions:")
print("""
# Standard imports
import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')  # Only for production code

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20) 
pd.set_option('display.precision', 2)
pd.set_option('display.width', None)
""")

# 2. Function design best practices
def load_and_clean_sales_data(file_path, date_columns=None, categorical_columns=None):
    """
    Load and perform basic cleaning of sales data.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
    date_columns : list, optional
        Columns to parse as dates
    categorical_columns : list, optional
        Columns to convert to categorical type
        
    Returns:
    --------
    pd.DataFrame
        Cleaned sales data
        
    Examples:
    ---------
    >>> df = load_and_clean_sales_data('sales.csv', 
    ...                               date_columns=['order_date'],
    ...                               categorical_columns=['category'])
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Parse dates
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert to categorical
        if categorical_columns:
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].astype('category')
        
        # Basic cleaning
        df = df.dropna(subset=['id'] if 'id' in df.columns else df.columns[:1])
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def calculate_business_metrics(df, revenue_col='revenue', date_col='date'):
    """
    Calculate standard business metrics from sales data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Sales data
    revenue_col : str
        Name of revenue column
    date_col : str
        Name of date column
        
    Returns:
    --------
    dict
        Dictionary containing business metrics
    """
    if df.empty:
        return {}
    
    metrics = {
        'total_revenue': df[revenue_col].sum(),
        'avg_daily_revenue': df.groupby(df[date_col].dt.date)[revenue_col].sum().mean(),
        'total_transactions': len(df),
        'avg_transaction_value': df[revenue_col].mean()
    }
    
    return metrics

print("2. Function Design:")
print("✓ Use descriptive function names")
print("✓ Include comprehensive docstrings") 
print("✓ Add type hints when possible")
print("✓ Handle errors gracefully")
print("✓ Return consistent data types")

# 3. Data validation patterns
def validate_sales_data(df):
    """Validate sales data quality."""
    validation_results = {
        'passed': True,
        'issues': []
    }
    
    # Check for required columns
    required_columns = ['order_id', 'customer_id', 'amount', 'date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_results['passed'] = False
        validation_results['issues'].append(f"Missing columns: {missing_columns}")
    
    # Check for negative amounts
    if 'amount' in df.columns and (df['amount'] < 0).any():
        validation_results['passed'] = False
        validation_results['issues'].append("Negative amounts found")
    
    # Check for duplicate order IDs
    if 'order_id' in df.columns and df['order_id'].duplicated().any():
        validation_results['passed'] = False
        validation_results['issues'].append("Duplicate order IDs found")
    
    return validation_results

print("\n3. Data Validation:")
print("✓ Validate data before processing")
print("✓ Check for required columns")
print("✓ Validate data types and ranges")
print("✓ Check for duplicates and inconsistencies")

# 4. Error handling patterns
def safe_groupby_operation(df, groupby_cols, agg_dict):
    """Safely perform groupby operations with error handling."""
    try:
        # Validate inputs
        if not isinstance(groupby_cols, list):
            groupby_cols = [groupby_cols]
        
        missing_cols = [col for col in groupby_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Groupby columns not found: {missing_cols}")
        
        # Perform groupby
        result = df.groupby(groupby_cols).agg(agg_dict)
        return result
        
    except Exception as e:
        print(f"Error in groupby operation: {e}")
        return pd.DataFrame()

print("\n4. Error Handling:")
print("✓ Use try-except blocks for file operations")
print("✓ Validate inputs before processing")
print("✓ Provide meaningful error messages")
print("✓ Return consistent fallback values")

# 5. Performance best practices
print("\n5. Performance Best Practices:")

performance_tips = {
    'Memory Usage': [
        'Use appropriate data types (int8, int16, category)',
        'Process data in chunks for large datasets',
        'Use iterators when possible',
        'Drop unnecessary columns early'
    ],
    'Computation': [
        'Prefer vectorized operations over loops',
        'Use pandas string methods instead of apply',
        'Set indexes for frequent lookups',
        'Chain operations to reduce intermediate objects'
    ],
    'I/O Operations': [
        'Use parquet format for large datasets',
        'Specify dtypes when reading CSV files',
        'Use compression for storage',
        'Read only necessary columns'
    ]
}

for category, tips in performance_tips.items():
    print(f"\n{category}:")
    for tip in tips:
        print(f"  ✓ {tip}")

# 6. Code organization patterns
print("\n6. Code Organization:")

example_structure = """
# Recommended project structure:
project/
├── data/
│   ├── raw/           # Original data files
│   ├── processed/     # Cleaned data files
│   └── external/      # External reference data
├── src/
│   ├── data_loading.py    # Data loading functions
│   ├── data_cleaning.py   # Data cleaning functions
│   ├── analysis.py        # Analysis functions
│   └── visualization.py   # Plotting functions
├── notebooks/
│   ├── exploration.ipynb  # Initial exploration
│   └── analysis.ipynb     # Main analysis
├── tests/
│   └── test_data_functions.py
└── requirements.txt
"""

print(example_structure)

# 7. Documentation best practices
print("7. Documentation:")
documentation_example = '''
def calculate_customer_lifetime_value(df, customer_col='customer_id', 
                                    revenue_col='revenue', date_col='date'):
    """
    Calculate Customer Lifetime Value (CLV) for each customer.
    
    CLV is calculated as the total revenue per customer divided by
    the time span of their activity in years.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Customer transaction data
    customer_col : str, default 'customer_id'
        Column name containing customer identifiers
    revenue_col : str, default 'revenue'
        Column name containing revenue amounts
    date_col : str, default 'date'
        Column name containing transaction dates
        
    Returns:
    --------
    pd.Series
        Customer lifetime value indexed by customer ID
        
    Examples:
    ---------
    >>> transactions = pd.DataFrame({
    ...     'customer_id': [1, 1, 2, 2],
    ...     'revenue': [100, 150, 200, 300],
    ...     'date': pd.date_range('2023-01-01', periods=4, freq='M')
    ... })
    >>> clv = calculate_customer_lifetime_value(transactions)
    >>> print(clv)
    customer_id
    1    83.33
    2    166.67
    dtype: float64
    
    Notes:
    ------
    - Customers with activity spanning less than 1 year are normalized to 1 year
    - Missing values in revenue are treated as 0
    - Requires at least 2 transactions per customer for meaningful results
    """
    # Implementation here...
    pass
'''

print("✓ Write comprehensive docstrings")
print("✓ Include parameter descriptions and types")
print("✓ Provide usage examples")
print("✓ Document assumptions and limitations")
print("✓ Include return value descriptions")

# 8. Testing patterns
print("\n8. Testing Best Practices:")
testing_example = '''
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

def test_data_cleaning_removes_nulls():
    """Test that data cleaning function removes null values."""
    # Arrange
    dirty_data = pd.DataFrame({
        'id': [1, 2, None, 4],
        'value': [10, None, 30, 40]
    })
    
    # Act
    clean_data = clean_dataframe(dirty_data)
    
    # Assert
    assert clean_data.isnull().sum().sum() == 0
    assert len(clean_data) < len(dirty_data)

def test_calculate_metrics_with_empty_dataframe():
    """Test metrics calculation with edge case."""
    empty_df = pd.DataFrame()
    metrics = calculate_business_metrics(empty_df)
    assert metrics == {}
'''

print("✓ Test edge cases (empty DataFrames, null values)")
print("✓ Use pandas.testing for DataFrame comparisons")
print("✓ Test both normal and error conditions")
print("✓ Mock external dependencies")

print("\n" + "="*50)
print("BEST PRACTICES SUMMARY")
print("="*50)
print("Following these practices will lead to:")
print("✓ More maintainable code")
print("✓ Better performance")
print("✓ Fewer bugs and errors") 
print("✓ Easier collaboration")
print("✓ More reliable data analysis")

# Cleanup temporary files
import os
temp_files = ['sample_data.csv', 'sample_data.xlsx', 'sample_data.json', 
              'output.csv', 'output.pkl', 'output.parquet',
              'sample.db', 'output_multiple_sheets.xlsx']

for file in temp_files:
    try:
        os.remove(file)
    except FileNotFoundError:
        pass

print("\nGuide complete! You now have comprehensive knowledge of pandas.")
```

This completes the comprehensive Pandas guide! The guide covers everything from basic concepts to advanced techniques, real-world applications, and best practices for professional data analysis with pandas.