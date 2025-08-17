# Python Program Constructs - Complete Guide

A comprehensive guide to Python programming constructs including control flow, functions, classes, and advanced programming patterns.

## Table of Contents

- [Variables and Data Types](#variables-and-data-types)
- [Control Flow Statements](#control-flow-statements)
- [Functions](#functions)
- [Classes and Objects](#classes-and-objects)
- [Modules and Packages](#modules-and-packages)
- [Exception Handling](#exception-handling)
- [Comprehensions](#comprehensions)
- [Decorators](#decorators)
- [Context Managers](#context-managers)
- [Generators and Iterators](#generators-and-iterators)
- [Advanced Constructs](#advanced-constructs)
- [Design Patterns](#design-patterns)
- [Best Practices](#best-practices)

## Variables and Data Types

### Variable Declaration and Assignment

```python
# Basic variable assignment
name = "Alice"
age = 25
height = 5.6
is_student = True

# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 0  # All get the same value

# Unpacking assignment
coordinates = (10, 20)
x, y = coordinates

# Extended unpacking (Python 3+)
numbers = [1, 2, 3, 4, 5]
first, *middle, last = numbers  # first=1, middle=[2,3,4], last=5
```

### Data Types Overview

```python
# Built-in types
integer = 42
float_num = 3.14159
string = "Hello, World!"
boolean = True
none_value = None

# Collections
my_list = [1, 2, 3, 4, 5]
my_tuple = (1, 2, 3)
my_dict = {"name": "John", "age": 30}
my_set = {1, 2, 3, 4, 5}

# Type checking
print(type(integer))    # <class 'int'>
print(isinstance(float_num, float))  # True

# Type conversion
str_num = "123"
converted = int(str_num)  # 123
float_converted = float(str_num)  # 123.0
```

### Variable Scope and Namespaces

```python
# Global scope
global_var = "I'm global"

def function_example():
    # Local scope
    local_var = "I'm local"
    
    # Access global variable
    global global_var
    global_var = "Modified global"
    
    # Nonlocal scope (for nested functions)
    def nested_function():
        nonlocal local_var
        local_var = "Modified local"
    
    nested_function()
    print(local_var)  # "Modified local"

# Class scope
class MyClass:
    class_var = "I'm a class variable"
    
    def __init__(self):
        self.instance_var = "I'm an instance variable"
```

## Control Flow Statements

### Conditional Statements

```python
# Basic if-else
age = 18
if age >= 18:
    print("Adult")
else:
    print("Minor")

# if-elif-else chain
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

# Ternary operator (conditional expression)
status = "adult" if age >= 18 else "minor"

# Multiple conditions
temperature = 25
humidity = 60
if temperature > 20 and humidity < 70:
    print("Nice weather")

# Checking for membership
vowels = "aeiou"
letter = "a"
if letter in vowels:
    print("It's a vowel")

# Truthiness checking
data = [1, 2, 3]
if data:  # True if list is not empty
    print("Data exists")

# Match statement (Python 3.10+)
def handle_http_status(status):
    match status:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case 500 | 502 | 503:
            return "Server Error"
        case code if 400 <= code < 500:
            return "Client Error"
        case _:  # Default case
            return "Unknown Status"
```

### Loop Constructs

#### For Loops

```python
# Basic for loop
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(fruit)

# With enumerate for index and value
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# Range-based loops
for i in range(5):          # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 6):       # 1, 2, 3, 4, 5
    print(i)

for i in range(0, 10, 2):   # 0, 2, 4, 6, 8
    print(i)

# Iterating over dictionaries
person = {"name": "John", "age": 30, "city": "New York"}
for key in person:
    print(f"{key}: {person[key]}")

for key, value in person.items():
    print(f"{key}: {value}")

# Nested loops
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for row in matrix:
    for element in row:
        print(element, end=" ")
    print()  # New line after each row

# Loop with else clause
for i in range(5):
    if i == 3:
        break
else:
    print("Loop completed normally")  # Won't execute if break is hit
```

#### While Loops

```python
# Basic while loop
count = 0
while count < 5:
    print(count)
    count += 1

# While with condition
user_input = ""
while user_input != "quit":
    user_input = input("Enter command (or 'quit' to exit): ")
    if user_input != "quit":
        print(f"You entered: {user_input}")

# While-else
attempts = 0
max_attempts = 3
while attempts < max_attempts:
    password = input("Enter password: ")
    if password == "secret":
        print("Access granted!")
        break
    attempts += 1
else:
    print("Access denied! Too many attempts.")

# Infinite loop with break
while True:
    choice = input("Enter choice (1-3, 'q' to quit): ")
    if choice == 'q':
        break
    elif choice in ['1', '2', '3']:
        print(f"You chose {choice}")
    else:
        print("Invalid choice")
```

#### Loop Control Statements

```python
# break and continue
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Skip even numbers, stop at 8
for num in numbers:
    if num % 2 == 0:
        continue  # Skip rest of loop iteration
    if num > 7:
        break     # Exit loop completely
    print(num)    # Prints: 1, 3, 5, 7

# pass statement (placeholder)
for i in range(5):
    if i == 2:
        pass  # Do nothing, placeholder for future code
    else:
        print(i)
```

## Functions

### Function Definition and Calling

```python
# Basic function
def greet(name):
    """Function to greet a person."""
    return f"Hello, {name}!"

# Call the function
message = greet("Alice")
print(message)

# Function with multiple parameters
def calculate_area(length, width):
    """Calculate area of rectangle."""
    return length * width

area = calculate_area(5, 3)  # 15

# Function with default parameters
def greet_with_title(name, title="Mr./Ms."):
    return f"Hello, {title} {name}!"

print(greet_with_title("Smith"))           # Uses default title
print(greet_with_title("Smith", "Dr."))    # Uses provided title

# Variable number of arguments
def sum_all(*args):
    """Sum all provided arguments."""
    return sum(args)

result = sum_all(1, 2, 3, 4, 5)  # 15

# Keyword arguments
def create_profile(**kwargs):
    """Create profile from keyword arguments."""
    profile = {}
    for key, value in kwargs.items():
        profile[key] = value
    return profile

profile = create_profile(name="John", age=30, city="NYC")

# Mixed arguments
def complex_function(required_arg, default_arg="default", *args, **kwargs):
    print(f"Required: {required_arg}")
    print(f"Default: {default_arg}")
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

complex_function("test", "custom", 1, 2, 3, name="John", age=30)
```

### Advanced Function Features

```python
# Function annotations (type hints)
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

# Lambda functions (anonymous functions)
square = lambda x: x ** 2
print(square(5))  # 25

# Lambda with multiple arguments
multiply = lambda x, y: x * y
print(multiply(3, 4))  # 12

# Functions as first-class objects
def operation(func, x, y):
    return func(x, y)

result = operation(lambda a, b: a + b, 5, 3)  # 8

# Nested functions
def outer_function(x):
    def inner_function(y):
        return x + y
    return inner_function

add_5 = outer_function(5)
result = add_5(3)  # 8

# Closures
def create_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

double = create_multiplier(2)
triple = create_multiplier(3)
print(double(5))  # 10
print(triple(5))  # 15
```

### Function Decorators (Preview)

```python
# Simple decorator
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()  # Prints before, hello, after
```

## Classes and Objects

### Class Definition and Object Creation

```python
# Basic class definition
class Person:
    # Class variable (shared by all instances)
    species = "Homo sapiens"
    
    def __init__(self, name, age):
        """Constructor method."""
        # Instance variables
        self.name = name
        self.age = age
    
    def introduce(self):
        """Instance method."""
        return f"Hi, I'm {self.name} and I'm {self.age} years old."
    
    def have_birthday(self):
        """Method that modifies instance state."""
        self.age += 1
    
    @classmethod
    def from_string(cls, person_str):
        """Class method - alternative constructor."""
        name, age = person_str.split('-')
        return cls(name, int(age))
    
    @staticmethod
    def is_adult(age):
        """Static method - utility function."""
        return age >= 18

# Create objects
person1 = Person("Alice", 25)
person2 = Person("Bob", 17)

print(person1.introduce())  # Hi, I'm Alice and I'm 25 years old.
print(person2.is_adult(person2.age))  # False

# Using class method
person3 = Person.from_string("Charlie-30")
print(person3.name)  # Charlie
```

### Inheritance

```python
# Base class
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        return "Some generic animal sound"
    
    def info(self):
        return f"{self.name} is a {self.species}"

# Derived class
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Canine")  # Call parent constructor
        self.breed = breed
    
    def make_sound(self):  # Override parent method
        return "Woof!"
    
    def fetch(self):  # New method specific to Dog
        return f"{self.name} is fetching the ball!"

# Multiple inheritance
class Flyable:
    def fly(self):
        return "Flying through the air!"

class Bird(Animal, Flyable):
    def __init__(self, name, wingspan):
        super().__init__(name, "Avian")
        self.wingspan = wingspan
    
    def make_sound(self):
        return "Tweet!"

# Usage
dog = Dog("Buddy", "Golden Retriever")
print(dog.info())        # Buddy is a Canine
print(dog.make_sound())  # Woof!
print(dog.fetch())       # Buddy is fetching the ball!

bird = Bird("Eagle", 2.5)
print(bird.make_sound()) # Tweet!
print(bird.fly())        # Flying through the air!
```

### Special Methods (Magic Methods)

```python
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
    
    def __str__(self):
        """String representation for end users."""
        return f"{self.title} by {self.author}"
    
    def __repr__(self):
        """String representation for developers."""
        return f"Book('{self.title}', '{self.author}', {self.pages})"
    
    def __len__(self):
        """Length of the book."""
        return self.pages
    
    def __eq__(self, other):
        """Equality comparison."""
        if isinstance(other, Book):
            return (self.title == other.title and 
                   self.author == other.author)
        return False
    
    def __lt__(self, other):
        """Less than comparison (for sorting)."""
        return self.pages < other.pages
    
    def __add__(self, other):
        """Addition operation."""
        if isinstance(other, Book):
            combined_title = f"{self.title} & {other.title}"
            combined_author = f"{self.author} & {other.author}"
            combined_pages = self.pages + other.pages
            return Book(combined_title, combined_author, combined_pages)
        return NotImplemented

# Usage
book1 = Book("1984", "George Orwell", 328)
book2 = Book("Animal Farm", "George Orwell", 112)

print(str(book1))     # 1984 by George Orwell
print(len(book1))     # 328
print(book1 < book2)  # False
combined = book1 + book2
print(combined.title) # 1984 & Animal Farm
```

### Properties and Encapsulation

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius  # Protected attribute
        self.__pi = 3.14159   # Private attribute
    
    @property
    def radius(self):
        """Getter for radius."""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Setter for radius with validation."""
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        """Computed property."""
        return self.__pi * self._radius ** 2
    
    @property
    def circumference(self):
        """Another computed property."""
        return 2 * self.__pi * self._radius

# Usage
circle = Circle(5)
print(circle.area)         # 78.53975
print(circle.circumference) # 31.4159

circle.radius = 10
print(circle.area)         # 314.159

# circle.radius = -5  # Would raise ValueError
```

## Modules and Packages

### Module Creation and Import

```python
# File: math_utils.py
"""A utility module for mathematical operations."""

PI = 3.14159

def calculate_area(radius):
    """Calculate area of a circle."""
    return PI * radius ** 2

def calculate_circumference(radius):
    """Calculate circumference of a circle."""
    return 2 * PI * radius

class Calculator:
    @staticmethod
    def add(a, b):
        return a + b
    
    @staticmethod
    def multiply(a, b):
        return a * b
```

```python
# File: main.py - Different ways to import

# Import entire module
import math_utils
area = math_utils.calculate_area(5)

# Import specific functions
from math_utils import calculate_area, PI
area = calculate_area(5)

# Import with alias
import math_utils as math
area = math.calculate_area(5)

# Import specific items with alias
from math_utils import calculate_area as calc_area
area = calc_area(5)

# Import all (not recommended)
from math_utils import *
area = calculate_area(5)  # Now directly available

# Conditional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if HAS_NUMPY:
    # Use numpy functions
    pass
else:
    # Use alternative implementation
    pass
```

### Package Structure

```
my_package/
├── __init__.py          # Makes it a package
├── core/
│   ├── __init__.py
│   ├── engine.py
│   └── utils.py
├── plugins/
│   ├── __init__.py
│   ├── plugin1.py
│   └── plugin2.py
└── tests/
    ├── __init__.py
    └── test_core.py
```

```python
# my_package/__init__.py
"""Main package initialization."""
from .core.engine import Engine
from .core.utils import utility_function

__version__ = "1.0.0"
__author__ = "Your Name"

# Make commonly used items available at package level
__all__ = ['Engine', 'utility_function']
```

```python
# Usage of package
import my_package
engine = my_package.Engine()

# Or import specific modules
from my_package.core import engine
from my_package.plugins import plugin1
```

## Exception Handling

### Basic Exception Handling

```python
# Basic try-except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exception types
try:
    value = int(input("Enter a number: "))
    result = 10 / value
except ValueError:
    print("Invalid input! Please enter a number.")
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Catching multiple exceptions
try:
    # Some risky operation
    pass
except (ValueError, TypeError) as e:
    print(f"Input error: {e}")

# Generic exception handling
try:
    # Some operation
    pass
except Exception as e:
    print(f"An error occurred: {e}")

# Complete try-except-else-finally block
try:
    file = open("data.txt", "r")
    data = file.read()
except FileNotFoundError:
    print("File not found!")
    data = None
except PermissionError:
    print("Permission denied!")
    data = None
else:
    # Executed if no exception occurs
    print("File read successfully!")
finally:
    # Always executed
    if 'file' in locals():
        file.close()
```

### Custom Exceptions

```python
# Define custom exception classes
class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class AgeValidationError(ValidationError):
    """Exception for age validation errors."""
    def __init__(self, age, message="Invalid age"):
        self.age = age
        self.message = message
        super().__init__(self.message)

# Function that raises custom exception
def validate_age(age):
    if not isinstance(age, int):
        raise AgeValidationError(age, "Age must be an integer")
    if age < 0:
        raise AgeValidationError(age, "Age cannot be negative")
    if age > 150:
        raise AgeValidationError(age, "Age seems unrealistic")
    return True

# Using custom exceptions
try:
    validate_age(-5)
except AgeValidationError as e:
    print(f"Age validation failed: {e.message}")
    print(f"Provided age: {e.age}")

# Re-raising exceptions
def process_data(data):
    try:
        # Some processing
        if not data:
            raise ValueError("Empty data")
    except ValueError:
        print("Logging error...")
        raise  # Re-raise the same exception

# Exception chaining
def convert_and_process(value):
    try:
        number = int(value)
    except ValueError as e:
        raise ValidationError("Could not convert to number") from e
```

## Comprehensions

### List Comprehensions

```python
# Basic list comprehension
squares = [x**2 for x in range(10)]
# Equivalent to:
# squares = []
# for x in range(10):
#     squares.append(x**2)

# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Nested loops
matrix = [[i*j for j in range(3)] for i in range(3)]
# Creates: [[0, 0, 0], [0, 1, 2], [0, 2, 4]]

# Flattening nested lists
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for sublist in nested for item in sublist]
# Result: [1, 2, 3, 4, 5, 6, 7, 8, 9]

# String processing
words = ["hello", "world", "python"]
capitalized = [word.capitalize() for word in words]
lengths = [len(word) for word in words]

# Complex example
people = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 17},
    {"name": "Charlie", "age": 30}
]
adult_names = [person["name"] for person in people if person["age"] >= 18]
```

### Dictionary Comprehensions

```python
# Basic dictionary comprehension
square_dict = {x: x**2 for x in range(5)}
# Result: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# From two lists
keys = ["a", "b", "c"]
values = [1, 2, 3]
my_dict = {k: v for k, v in zip(keys, values)}

# With condition
filtered_dict = {k: v for k, v in my_dict.items() if v > 1}

# String manipulation
text = "hello world"
char_count = {char: text.count(char) for char in set(text)}

# Inverting dictionary
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
```

### Set Comprehensions

```python
# Basic set comprehension
unique_squares = {x**2 for x in range(-5, 6)}
# Negative and positive numbers with same square are deduplicated

# With condition
unique_remainders = {x % 5 for x in range(100)}
# Result: {0, 1, 2, 3, 4}

# From string
unique_chars = {char.lower() for char in "Hello World" if char.isalpha()}
```

### Generator Expressions

```python
# Generator expression (similar to list comprehension but lazy)
squares_gen = (x**2 for x in range(10))

# Generator expressions are memory efficient
import sys
list_comp = [x**2 for x in range(1000)]
gen_exp = (x**2 for x in range(1000))
print(sys.getsizeof(list_comp))  # Much larger
print(sys.getsizeof(gen_exp))    # Much smaller

# Using generator in functions
def sum_squares(n):
    return sum(x**2 for x in range(n))

# Generator with conditions
even_squares = (x**2 for x in range(100) if x % 2 == 0)
```

## Decorators

### Basic Decorators

```python
# Simple decorator
def my_decorator(func):
    def wrapper():
        print("Something before the function")
        func()
        print("Something after the function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# Equivalent to: say_hello = my_decorator(say_hello)

say_hello()  # Executes the decorated version

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")  # Prints greeting 3 times

# Preserving function metadata
import functools

def my_decorator(func):
    @functools.wraps(func)  # Preserves original function's metadata
    def wrapper(*args, **kwargs):
        print("Before function")
        result = func(*args, **kwargs)
        print("After function")
        return result
    return wrapper

@my_decorator
def add(x, y):
    """Add two numbers."""
    return x + y

print(add.__name__)  # 'add' (without wraps, it would be 'wrapper')
print(add.__doc__)   # 'Add two numbers.'
```

### Practical Decorators

```python
import time
import functools

# Timer decorator
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Retry decorator
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

# Cache decorator (simple memoization)
def cache(func):
    cached_results = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args in cached_results:
            return cached_results[args]
        result = func(*args)
        cached_results[args] = result
        return result
    return wrapper

# Usage
@timer
@retry(max_attempts=3)
@cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Class-based decorators
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} has been called {self.count} times")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")
```

## Context Managers

### Using Context Managers

```python
# File handling with context manager
with open("data.txt", "r") as file:
    content = file.read()
    # File is automatically closed when exiting the with block

# Multiple context managers
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    data = infile.read()
    outfile.write(data.upper())

# Exception handling in context managers
try:
    with open("nonexistent.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("File not found")
# File is still properly closed even if exception occurs
```

### Creating Custom Context Managers

```python
# Context manager class
class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    def __enter__(self):
        print("Connecting to database...")
        # Simulate database connection
        self.connection = f"Connected to {self.connection_string}"
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection...")
        # Cleanup code here
        self.connection = None
        # Return False to propagate any exceptions

# Usage
with DatabaseConnection("postgresql://localhost") as conn:
    print(f"Using connection: {conn}")
    # Connection is automatically closed

# Context manager using contextlib
from contextlib import contextmanager

@contextmanager
def timer_context():
    start_time = time.time()
    try:
        yield start_time
    finally:
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")

# Usage
with timer_context() as start_time:
    # Some time-consuming operation
    time.sleep(1)
    print("Operation completed")

# Suppress exceptions context manager
from contextlib import suppress

with suppress(FileNotFoundError):
    with open("nonexistent.txt", "r") as file:
        content = file.read()
# FileNotFoundError is suppressed, program continues

# Temporary directory context manager
import tempfile
import os

with tempfile.TemporaryDirectory() as temp_dir:
    temp_file = os.path.join(temp_dir, "temp.txt")
    with open(temp_file, "w") as f:
        f.write("Temporary data")
    # Directory and all files are automatically deleted
```

## Generators and Iterators

### Generators

```python
# Generator function
def countdown(n):
    while n > 0:
        yield n
        n -= 1

# Using generator
for num in countdown(5):
    print(num)  # Prints 5, 4, 3, 2, 1

# Generator expressions
squares = (x**2 for x in range(10))
print(next(squares))  # 0
print(next(squares))  # 1

# Infinite generator
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Use with islice to limit
from itertools import islice
first_10_fibs = list(islice(fibonacci(), 10))

# Generator with send()
def echo():
    while True:
        value = yield
        if value is not None:
            print(f"Received: {value}")

gen = echo()
next(gen)  # Prime the generator
gen.send("Hello")  # Sends value to generator

# Generator for file processing
def read_large_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# Memory efficient processing
# for line in read_large_file("huge_file.txt"):
#     process(line)
```

### Custom Iterators

```python
# Iterator class
class CountDown:
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

# Usage
for num in CountDown(3):
    print(num)  # Prints 3, 2, 1

# Range-like iterator
class Range:
    def __init__(self, start, end, step=1):
        self.start = start
        self.end = end
        self.step = step
    
    def __iter__(self):
        current = self.start
        while current < self.end:
            yield current
            current += self.step

# Iterator tools
from itertools import cycle, count, repeat, chain, combinations, permutations

# Infinite iterators
colors = cycle(['red', 'green', 'blue'])  # Cycles through colors infinitely
counter = count(start=1, step=2)          # 1, 3, 5, 7, ...
repeater = repeat('hello', times=3)       # 'hello', 'hello', 'hello'

# Combining iterators
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = chain(list1, list2)  # 1, 2, 3, 4, 5, 6

# Combinatorial iterators
items = ['A', 'B', 'C']
combos = list(combinations(items, 2))  # [('A', 'B'), ('A', 'C'), ('B', 'C')]
perms = list(permutations(items, 2))   # [('A', 'B'), ('A', 'C'), ('B', 'A'), ...]
```

## Advanced Constructs

### Metaclasses

```python
# Basic metaclass
class Meta(type):
    def __new__(mcs, name, bases, namespace):
        # Modify class creation
        namespace['class_id'] = f"{name}_{id(namespace)}"
        return super().__new__(mcs, name, bases, namespace)

class MyClass(metaclass=Meta):
    pass

print(MyClass.class_id)  # MyClass_<some_id>

# Practical metaclass - Singleton pattern
class Singleton(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=Singleton):
    def __init__(self):
        print("Creating database connection")

db1 = Database()  # Creates new instance
db2 = Database()  # Returns existing instance
print(db1 is db2)  # True
```

### Descriptors

```python
# Descriptor class
class Temperature:
    def __init__(self, initial_temp=0):
        self.temp = initial_temp
    
    def __get__(self, instance, owner):
        return self.temp
    
    def __set__(self, instance, value):
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self.temp = value

class Substance:
    temperature = Temperature(25)  # Descriptor instance

# Usage
water = Substance()
print(water.temperature)  # 25
water.temperature = 100   # Sets temperature
# water.temperature = -300  # Would raise ValueError

# Property as descriptor
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2
```

### Abstract Base Classes

```python
from abc import ABC, abstractmethod

# Abstract base class
class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass
    
    # Concrete method
    def description(self):
        return f"This is a shape with area {self.area()}"

# Concrete implementation
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

# Can't instantiate abstract class
# shape = Shape()  # TypeError

# Can instantiate concrete class
rect = Rectangle(5, 3)
print(rect.area())        # 15
print(rect.description()) # This is a shape with area 15
```

## Design Patterns

### Commonly Used Python Design Patterns

```python
# 1. Factory Pattern
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type.lower() == "dog":
            return Dog()
        elif animal_type.lower() == "cat":
            return Cat()
        else:
            raise ValueError("Unknown animal type")

# Usage
factory = AnimalFactory()
dog = factory.create_animal("dog")
print(dog.speak())  # Woof!

# 2. Observer Pattern
class Observer:
    def update(self, subject):
        pass

class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)
    
    def set_state(self, state):
        self._state = state
        self.notify()
    
    def get_state(self):
        return self._state

class EmailNotifier(Observer):
    def update(self, subject):
        print(f"Email: State changed to {subject.get_state()}")

# 3. Strategy Pattern
class PaymentStrategy:
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} with credit card"

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} with PayPal"

class ShoppingCart:
    def __init__(self, payment_strategy):
        self.payment_strategy = payment_strategy
    
    def checkout(self, amount):
        return self.payment_strategy.pay(amount)

# Usage
cart = ShoppingCart(CreditCardPayment())
print(cart.checkout(100))  # Paid $100 with credit card
```

## Best Practices

### Code Organization

```python
# 1. Module structure
"""
module_name.py

Module docstring explaining the purpose and usage of the module.
"""

# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import requests
import pandas as pd

# Local imports
from .utils import helper_function
from .models import DataModel

# Constants
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

# Module-level variables
_cache = {}

# Classes and functions
class ExampleClass:
    """Class docstring."""
    pass

def main():
    """Main function."""
    pass

if __name__ == "__main__":
    main()
```

### Naming Conventions

```python
# Variables and functions: snake_case
user_name = "john_doe"
def calculate_total_price():
    pass

# Classes: PascalCase
class UserAccount:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_CONNECTIONS = 100
API_BASE_URL = "https://api.example.com"

# Private/internal: prefix with underscore
class MyClass:
    def __init__(self):
        self.public_var = "public"
        self._protected_var = "protected"  # Convention only
        self.__private_var = "private"     # Name mangling

# Meaningful names
# Bad
def calc(x, y):
    return x * y * 0.1

# Good
def calculate_discount(price, quantity, discount_rate=0.1):
    return price * quantity * discount_rate
```

### Error Handling Best Practices

```python
# Specific exception handling
try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print("Config file not found, using defaults")
    config = get_default_config()
except json.JSONDecodeError as e:
    print(f"Invalid JSON in config file: {e}")
    config = get_default_config()
except Exception as e:
    print(f"Unexpected error reading config: {e}")
    raise

# Custom exceptions with context
class ValidationError(Exception):
    def __init__(self, field, value, message):
        self.field = field
        self.value = value
        super().__init__(message)

def validate_email(email):
    if "@" not in email:
        raise ValidationError("email", email, "Email must contain @ symbol")

# Early returns for error conditions
def process_user_data(user_data):
    if not user_data:
        return {"error": "No user data provided"}
    
    if "email" not in user_data:
        return {"error": "Email is required"}
    
    # Process data
    return {"success": True, "processed": user_data}
```

### Performance Considerations

```python
# Use list comprehensions for simple transformations
# Good
squared = [x**2 for x in numbers]

# Less efficient
squared = []
for x in numbers:
    squared.append(x**2)

# Use generators for large datasets
def process_large_file(filename):
    with open(filename, 'r') as f:
        for line in f:  # Generator, not loading entire file
            yield process_line(line)

# Use set/dict for membership testing
# Slow for large collections
if item in large_list:  # O(n)
    pass

# Fast for large collections
if item in large_set:   # O(1) average
    pass

# Cache expensive computations
import functools

@functools.lru_cache(maxsize=128)
def expensive_calculation(n):
    # Some expensive operation
    return sum(i**2 for i in range(n))
```

### Documentation

```python
def calculate_compound_interest(principal: float, rate: float, 
                              time: int, compound_frequency: int = 12) -> float:
    """
    Calculate compound interest.
    
    Args:
        principal: Initial amount of money
        rate: Annual interest rate (as a decimal)
        time: Time period in years
        compound_frequency: Number of times interest is compounded per year
    
    Returns:
        Final amount after compound interest
    
    Raises:
        ValueError: If any parameter is negative
    
    Examples:
        >>> calculate_compound_interest(1000, 0.05, 3, 12)
        1161.62
    """
    if any(param < 0 for param in [principal, rate, time, compound_frequency]):
        raise ValueError("All parameters must be non-negative")
    
    return principal * (1 + rate / compound_frequency) ** (compound_frequency * time)
```

This comprehensive guide covers all major Python programming constructs with practical examples and best practices for effective Python programming!