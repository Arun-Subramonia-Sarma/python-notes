# Python Operators - Complete Guide

A comprehensive reference for all Python operators, their usage, precedence, and practical examples.

## Table of Contents

- [Arithmetic Operators](#arithmetic-operators)
- [Comparison Operators](#comparison-operators)
- [Logical Operators](#logical-operators)
- [Assignment Operators](#assignment-operators)
- [Bitwise Operators](#bitwise-operators)
- [Membership Operators](#membership-operators)
- [Identity Operators](#identity-operators)
- [Operator Precedence](#operator-precedence)
- [Special Operators](#special-operators)
- [Operator Overloading](#operator-overloading)
- [Practical Examples](#practical-examples)
- [Best Practices](#best-practices)

## Arithmetic Operators

Arithmetic operators perform mathematical operations on numeric values.

### Basic Arithmetic

| Operator | Name | Example | Result |
|----------|------|---------|--------|
| `+` | Addition | `5 + 3` | `8` |
| `-` | Subtraction | `5 - 3` | `2` |
| `*` | Multiplication | `5 * 3` | `15` |
| `/` | Division (float) | `7 / 2` | `3.5` |
| `//` | Floor Division | `7 // 2` | `3` |
| `%` | Modulus | `7 % 3` | `1` |
| `**` | Exponentiation | `2 ** 3` | `8` |

### Examples and Use Cases

```python
# Basic arithmetic
a = 10
b = 3

print(f"Addition: {a + b}")        # 13
print(f"Subtraction: {a - b}")     # 7
print(f"Multiplication: {a * b}")  # 30
print(f"Division: {a / b}")        # 3.3333...
print(f"Floor division: {a // b}") # 3
print(f"Modulus: {a % b}")         # 1
print(f"Exponentiation: {a ** b}") # 1000

# Practical uses
# Check if number is even
def is_even(n):
    return n % 2 == 0

# Calculate compound interest
principal = 1000
rate = 0.05
time = 3
compound_interest = principal * (1 + rate) ** time

# Split into groups
total_items = 25
group_size = 4
full_groups = total_items // group_size  # 6
remaining = total_items % group_size     # 1
```

### String Arithmetic

```python
# String concatenation
first_name = "John"
last_name = "Doe"
full_name = first_name + " " + last_name  # "John Doe"

# String repetition
border = "-" * 20  # "--------------------"
greeting = "Hello! " * 3  # "Hello! Hello! Hello! "

# List/tuple arithmetic
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2  # [1, 2, 3, 4, 5, 6]
repeated = [0] * 5        # [0, 0, 0, 0, 0]
```

## Comparison Operators

Comparison operators compare two values and return Boolean results.

### Basic Comparisons

| Operator | Name | Example | Result |
|----------|------|---------|--------|
| `==` | Equal | `5 == 5` | `True` |
| `!=` | Not equal | `5 != 3` | `True` |
| `>` | Greater than | `5 > 3` | `True` |
| `<` | Less than | `3 < 5` | `True` |
| `>=` | Greater than or equal | `5 >= 5` | `True` |
| `<=` | Less than or equal | `3 <= 5` | `True` |

### Examples and Edge Cases

```python
# Basic comparisons
x = 10
y = 20

print(x == y)   # False
print(x != y)   # True
print(x < y)    # True
print(x > y)    # False
print(x <= 10)  # True
print(x >= 10)  # True

# String comparisons (lexicographic)
print("apple" < "banana")   # True
print("Apple" < "apple")    # True (uppercase comes before lowercase)
print("10" < "9")           # True (string comparison, not numeric)

# List comparisons (element by element)
print([1, 2, 3] < [1, 2, 4])   # True
print([1, 2] < [1, 2, 3])      # True (shorter is less if prefix matches)

# Chained comparisons
age = 25
print(18 <= age <= 65)  # True (equivalent to: age >= 18 and age <= 65)

# Floating point precision issues
print(0.1 + 0.2 == 0.3)  # False! (floating point precision)
import math
print(math.isclose(0.1 + 0.2, 0.3))  # True (better for float comparison)

# None comparisons
value = None
print(value is None)     # True (preferred)
print(value == None)     # True (works but not recommended)
```

## Logical Operators

Logical operators combine Boolean expressions.

### Basic Logical Operators

| Operator | Description | Example | Result |
|----------|------------|---------|--------|
| `and` | Returns True if both operands are True | `True and False` | `False` |
| `or` | Returns True if at least one operand is True | `True or False` | `True` |
| `not` | Returns the opposite Boolean value | `not True` | `False` |

### Truth Tables

```python
# AND truth table
print(True and True)    # True
print(True and False)   # False
print(False and True)   # False
print(False and False)  # False

# OR truth table
print(True or True)     # True
print(True or False)    # True
print(False or True)    # True
print(False or False)   # False

# NOT truth table
print(not True)         # False
print(not False)        # True
```

### Short-Circuit Evaluation

```python
# AND short-circuits on first False
def false_func():
    print("false_func called")
    return False

def true_func():
    print("true_func called")
    return True

# Only false_func is called, true_func is skipped
result = false_func() and true_func()

# OR short-circuits on first True
result = true_func() or false_func()  # Only true_func called

# Practical use - safe navigation
user = {"name": "John"}
# Safe way to check nested attributes
if user and "profile" in user and user["profile"].get("age"):
    print(f"Age: {user['profile']['age']}")
```

### Complex Logical Expressions

```python
# Complex conditions
age = 25
income = 50000
credit_score = 750

# Loan eligibility
eligible = (age >= 18 and age <= 65) and (income >= 30000 or credit_score >= 700)

# Using parentheses for clarity
valid_user = (username and password) and (is_active or is_admin)

# De Morgan's Laws
# not (A and B) == (not A) or (not B)
# not (A or B) == (not A) and (not B)
x, y = True, False
print(not (x and y) == (not x or not y))  # True
print(not (x or y) == (not x and not y))  # True
```

## Assignment Operators

Assignment operators assign values to variables.

### Basic Assignment

| Operator | Example | Equivalent To |
|----------|---------|---------------|
| `=` | `x = 5` | - |
| `+=` | `x += 3` | `x = x + 3` |
| `-=` | `x -= 3` | `x = x - 3` |
| `*=` | `x *= 3` | `x = x * 3` |
| `/=` | `x /= 3` | `x = x / 3` |
| `//=` | `x //= 3` | `x = x // 3` |
| `%=` | `x %= 3` | `x = x % 3` |
| `**=` | `x **= 3` | `x = x ** 3` |

### Examples

```python
# Basic assignment
x = 10

# Augmented assignment
x += 5    # x is now 15
x -= 3    # x is now 12
x *= 2    # x is now 24
x /= 4    # x is now 6.0
x //= 2   # x is now 3.0
x %= 2    # x is now 1.0
x **= 3   # x is now 1.0

# String augmented assignment
message = "Hello"
message += " World"  # "Hello World"

# List augmented assignment
numbers = [1, 2, 3]
numbers += [4, 5]    # [1, 2, 3, 4, 5]
numbers *= 2         # [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

# Multiple assignment
a = b = c = 10  # All variables get the same value

# Tuple unpacking assignment
x, y = 5, 10
x, y = y, x  # Swap values

# Extended unpacking
first, *middle, last = [1, 2, 3, 4, 5]  # first=1, middle=[2,3,4], last=5
```

### Bitwise Assignment

| Operator | Example | Equivalent To |
|----------|---------|---------------|
| `&=` | `x &= 3` | `x = x & 3` |
| `\|=` | `x \|= 3` | `x = x \| 3` |
| `^=` | `x ^= 3` | `x = x ^ 3` |
| `<<=` | `x <<= 3` | `x = x << 3` |
| `>>=` | `x >>= 3` | `x = x >> 3` |

```python
# Bitwise assignment examples
flags = 0b1010  # Binary: 1010
flags &= 0b1100  # AND: 1000
flags |= 0b0011  # OR: 1011
flags ^= 0b1111  # XOR: 0100
flags <<= 2      # Left shift: 10000
flags >>= 1      # Right shift: 1000
```

## Bitwise Operators

Bitwise operators work on binary representations of numbers.

### Basic Bitwise Operators

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `&` | AND | Sets bit if both bits are 1 | `12 & 10 = 8` |
| `\|` | OR | Sets bit if at least one bit is 1 | `12 \| 10 = 14` |
| `^` | XOR | Sets bit if bits are different | `12 ^ 10 = 6` |
| `~` | NOT | Inverts all bits | `~12 = -13` |
| `<<` | Left Shift | Shifts bits left | `12 << 2 = 48` |
| `>>` | Right Shift | Shifts bits right | `12 >> 2 = 3` |

### Detailed Examples

```python
# Binary representations
a = 12  # Binary: 1100
b = 10  # Binary: 1010

print(f"a = {a:08b}")      # 00001100
print(f"b = {b:08b}")      # 00001010

# Bitwise AND
result = a & b  # 8 (Binary: 1000)
print(f"a & b = {result:08b} = {result}")

# Bitwise OR
result = a | b  # 14 (Binary: 1110)
print(f"a | b = {result:08b} = {result}")

# Bitwise XOR
result = a ^ b  # 6 (Binary: 0110)
print(f"a ^ b = {result:08b} = {result}")

# Bitwise NOT (Two's complement)
result = ~a  # -13
print(f"~a = {result}")

# Left shift (multiply by 2^n)
result = a << 2  # 48 (multiply by 4)
print(f"a << 2 = {result}")

# Right shift (divide by 2^n)
result = a >> 2  # 3 (divide by 4)
print(f"a >> 2 = {result}")
```

### Practical Bitwise Applications

```python
# Flag operations
READ = 1    # 001
WRITE = 2   # 010
EXECUTE = 4 # 100

# Set permissions
permissions = READ | WRITE  # 011 (3)

# Check if has permission
has_read = bool(permissions & READ)    # True
has_execute = bool(permissions & EXECUTE)  # False

# Add permission
permissions |= EXECUTE  # Now 111 (7)

# Remove permission
permissions &= ~WRITE  # Now 101 (5)

# Toggle permission
permissions ^= READ  # Toggle read permission

# Bit manipulation tricks
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def count_set_bits(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

# Swap without temporary variable
x, y = 5, 10
x ^= y
y ^= x
x ^= y
print(f"x={x}, y={y}")  # x=10, y=5
```

## Membership Operators

Membership operators test if a value is a member of a sequence.

### Basic Membership

| Operator | Description | Example |
|----------|-------------|---------|
| `in` | Returns True if value is found in sequence | `'a' in 'apple'` |
| `not in` | Returns True if value is not found in sequence | `'z' not in 'apple'` |

### Examples with Different Data Types

```python
# String membership
text = "Hello World"
print('H' in text)        # True
print('hello' in text)    # False (case sensitive)
print('World' in text)    # True
print('xyz' not in text)  # True

# List membership
fruits = ['apple', 'banana', 'orange']
print('apple' in fruits)     # True
print('grape' not in fruits) # True

# Tuple membership
coordinates = (10, 20, 30)
print(20 in coordinates)  # True
print(40 in coordinates)  # False

# Dictionary membership (checks keys by default)
person = {'name': 'John', 'age': 25}
print('name' in person)     # True
print('John' in person)     # False (checks keys, not values)
print('John' in person.values())  # True (checks values)

# Set membership (very fast)
numbers = {1, 2, 3, 4, 5}
print(3 in numbers)  # True - O(1) average time complexity

# Range membership
print(5 in range(1, 10))   # True
print(15 in range(1, 10))  # False

# Nested membership
matrix = [[1, 2], [3, 4], [5, 6]]
print([3, 4] in matrix)  # True
print(3 in [item for row in matrix for item in row])  # True
```

### Practical Applications

```python
# Input validation
valid_choices = ['yes', 'no', 'maybe']
user_input = input("Enter choice: ").lower()
if user_input in valid_choices:
    print("Valid choice")

# Filtering
vowels = 'aeiou'
def count_vowels(text):
    return sum(1 for char in text.lower() if char in vowels)

# Permission checking
user_roles = ['admin', 'editor']
if 'admin' in user_roles:
    print("Has admin access")

# Blacklist checking
banned_words = {'spam', 'virus', 'malware'}
message = "This is a normal message"
if any(word in message.lower() for word in banned_words):
    print("Message contains banned words")
```

## Identity Operators

Identity operators compare the memory locations of objects.

### Basic Identity

| Operator | Description | Example |
|----------|-------------|---------|
| `is` | Returns True if both variables point to the same object | `x is y` |
| `is not` | Returns True if variables point to different objects | `x is not y` |

### Understanding Object Identity

```python
# Integer caching (-5 to 256)
a = 100
b = 100
print(a is b)  # True (same object due to integer caching)

a = 1000
b = 1000
print(a is b)  # False (different objects)
print(a == b)  # True (same value)

# String interning
x = "hello"
y = "hello"
print(x is y)  # True (string interning)

x = "hello world"
y = "hello world"
print(x is y)  # May be True or False (depends on Python implementation)

# Lists (always different objects)
list1 = [1, 2, 3]
list2 = [1, 2, 3]
print(list1 is list2)  # False
print(list1 == list2)  # True

# None comparison (always use 'is')
value = None
print(value is None)     # Correct way
print(value == None)     # Works but not recommended
```

### Practical Examples

```python
# Singleton pattern check
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True

# Mutable default argument pitfall
def append_to_list(item, target_list=[]):  # Dangerous!
    target_list.append(item)
    return target_list

list1 = append_to_list(1)
list2 = append_to_list(2)
print(list1 is list2)  # True (same object!)
print(list1)  # [1, 2] - unexpected!

# Correct way
def append_to_list_safe(item, target_list=None):
    if target_list is None:
        target_list = []
    target_list.append(item)
    return target_list

# Copying vs referencing
original = [1, 2, 3]
reference = original        # Same object
shallow_copy = original[:]  # Different object, same content
import copy
deep_copy = copy.deepcopy(original)

print(original is reference)    # True
print(original is shallow_copy) # False
print(original == shallow_copy) # True
```

## Operator Precedence

Operator precedence determines the order of operations in expressions.

### Precedence Table (High to Low)

| Precedence | Operators | Description |
|------------|-----------|-------------|
| 1 | `()` | Parentheses |
| 2 | `**` | Exponentiation |
| 3 | `+x`, `-x`, `~x` | Unary plus, minus, bitwise NOT |
| 4 | `*`, `/`, `//`, `%` | Multiplication, division, floor division, modulus |
| 5 | `+`, `-` | Addition, subtraction |
| 6 | `<<`, `>>` | Bitwise shifts |
| 7 | `&` | Bitwise AND |
| 8 | `^` | Bitwise XOR |
| 9 | `\|` | Bitwise OR |
| 10 | `==`, `!=`, `<`, `<=`, `>`, `>=`, `is`, `is not`, `in`, `not in` | Comparisons |
| 11 | `not` | Logical NOT |
| 12 | `and` | Logical AND |
| 13 | `or` | Logical OR |

### Examples

```python
# Without parentheses - follows precedence rules
result = 2 + 3 * 4      # 14 (not 20)
result = 2 ** 3 ** 2    # 512 (2 ** (3 ** 2), right associative)
result = 10 - 4 - 2     # 4 (left associative)

# Logical operator precedence
result = True or False and False  # True (and has higher precedence)
result = (True or False) and False  # False (with parentheses)

# Complex expression
x = 5
y = 10
result = x < 7 and y > 8 or x == 5  # True
# Equivalent to: ((x < 7) and (y > 8)) or (x == 5)

# Bitwise vs comparison precedence
result = 8 & 4 == 4  # False (equivalent to 8 & (4 == 4) = 8 & True = 8)
result = (8 & 4) == 4  # True (correct way)

# Best practice: use parentheses for clarity
total = (price * quantity) + (tax_rate * price * quantity)
valid = (age >= 18) and (has_license or is_supervised)
```

## Special Operators

### Walrus Operator (`:=`) - Python 3.8+

Assignment expression that assigns and returns a value.

```python
# Traditional approach
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squared = []
for n in numbers:
    if n * n > 25:
        squared.append(n * n)

# With walrus operator
squared = [square for n in numbers if (square := n * n) > 25]

# In while loops
import random
while (value := random.randint(1, 10)) != 7:
    print(f"Got {value}, trying again...")
print("Finally got 7!")

# In if statements
data = {"name": "John", "age": 25}
if (age := data.get("age")) and age >= 18:
    print(f"Adult: {age} years old")
```

### Ternary Operator

Conditional expression for inline if-else.

```python
# Basic ternary
age = 20
status = "adult" if age >= 18 else "minor"

# Nested ternary (use sparingly)
grade = 85
letter = "A" if grade >= 90 else "B" if grade >= 80 else "C" if grade >= 70 else "F"

# With function calls
def expensive_calculation():
    return sum(range(1000000))

result = expensive_calculation() if should_calculate else 0

# Ternary vs regular if-else
# Ternary: for simple assignments
value = x if condition else y

# Regular if-else: for complex logic
if condition:
    # Multiple statements
    log_action()
    value = complex_calculation(x)
    update_database()
else:
    value = y
```

## Operator Overloading

Define how operators work with custom classes.

### Magic Methods for Operators

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # Arithmetic operators
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        raise TypeError("Can only multiply by scalar")
    
    def __truediv__(self, scalar):
        return Vector(self.x / scalar, self.y / scalar)
    
    # Comparison operators
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other):
        return self.magnitude() < other.magnitude()
    
    # Unary operators
    def __neg__(self):
        return Vector(-self.x, -self.y)
    
    def __abs__(self):
        return self.magnitude()
    
    # String representation
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def magnitude(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

# Usage
v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)      # Vector(4, 6)
print(v1 - v2)      # Vector(2, 2)
print(v1 * 2)       # Vector(6, 8)
print(v1 / 2)       # Vector(1.5, 2.0)
print(v1 == v2)     # False
print(v1 < v2)      # False (5.0 < 2.236...)
print(-v1)          # Vector(-3, -4)
print(abs(v1))      # 5.0
```

### Complete Operator Magic Methods

```python
class CompleteOperators:
    def __init__(self, value):
        self.value = value
    
    # Arithmetic
    def __add__(self, other): return CompleteOperators(self.value + other.value)
    def __sub__(self, other): return CompleteOperators(self.value - other.value)
    def __mul__(self, other): return CompleteOperators(self.value * other.value)
    def __truediv__(self, other): return CompleteOperators(self.value / other.value)
    def __floordiv__(self, other): return CompleteOperators(self.value // other.value)
    def __mod__(self, other): return CompleteOperators(self.value % other.value)
    def __pow__(self, other): return CompleteOperators(self.value ** other.value)
    
    # Reverse arithmetic (for when left operand doesn't support operation)
    def __radd__(self, other): return CompleteOperators(other + self.value)
    def __rsub__(self, other): return CompleteOperators(other - self.value)
    def __rmul__(self, other): return CompleteOperators(other * self.value)
    
    # In-place operators
    def __iadd__(self, other): 
        self.value += other.value
        return self
    def __isub__(self, other): 
        self.value -= other.value
        return self
    
    # Unary operators
    def __neg__(self): return CompleteOperators(-self.value)
    def __pos__(self): return CompleteOperators(+self.value)
    def __abs__(self): return CompleteOperators(abs(self.value))
    
    # Bitwise operators
    def __and__(self, other): return CompleteOperators(self.value & other.value)
    def __or__(self, other): return CompleteOperators(self.value | other.value)
    def __xor__(self, other): return CompleteOperators(self.value ^ other.value)
    def __lshift__(self, other): return CompleteOperators(self.value << other.value)
    def __rshift__(self, other): return CompleteOperators(self.value >> other.value)
    def __invert__(self): return CompleteOperators(~self.value)
    
    # Comparison operators
    def __eq__(self, other): return self.value == other.value
    def __ne__(self, other): return self.value != other.value
    def __lt__(self, other): return self.value < other.value
    def __le__(self, other): return self.value <= other.value
    def __gt__(self, other): return self.value > other.value
    def __ge__(self, other): return self.value >= other.value
    
    def __str__(self):
        return f"CompleteOperators({self.value})"
```

## Practical Examples

### Real-World Use Cases

```python
# 1. Data Validation Pipeline
class ValidationError(Exception):
    pass

def validate_user_data(data):
    # Using logical operators for validation
    if not (data.get('email') and '@' in data['email']):
        raise ValidationError("Invalid email")
    
    if not (data.get('age') and 13 <= data['age'] <= 120):
        raise ValidationError("Invalid age")
    
    if not (data.get('password') and len(data['password']) >= 8):
        raise ValidationError("Password too short")
    
    return True

# 2. Bit Flags for Permissions
class Permissions:
    READ = 1      # 001
    WRITE = 2     # 010
    EXECUTE = 4   # 100
    DELETE = 8    # 1000
    
    def __init__(self, flags=0):
        self.flags = flags
    
    def grant(self, permission):
        self.flags |= permission
    
    def revoke(self, permission):
        self.flags &= ~permission
    
    def has(self, permission):
        return bool(self.flags & permission)
    
    def toggle(self, permission):
        self.flags ^= permission

# Usage
perms = Permissions()
perms.grant(Permissions.READ | Permissions.WRITE)
print(perms.has(Permissions.READ))   # True
print(perms.has(Permissions.DELETE)) # False

# 3. Mathematical Calculations
def calculate_loan_payment(principal, rate, years):
    # Using arithmetic operators for financial calculations
    monthly_rate = rate / 12
    num_payments = years * 12
    
    if rate == 0:
        return principal / num_payments
    
    # Loan payment formula
    payment = (principal * monthly_rate * 
              (1 + monthly_rate) ** num_payments) / \
              ((1 + monthly_rate) ** num_payments - 1)
    
    return payment

# 4. Text Processing
def analyze_text(text):
    words = text.lower().split()
    
    # Using membership and comparison operators
    analysis = {
        'word_count': len(words),
        'has_python': 'python' in words,
        'short_words': sum(1 for word in words if len(word) <= 4),
        'long_words': sum(1 for word in words if len(word) > 10),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
    }
    
    return analysis

# 5. Game State Management
class GameState:
    def __init__(self):
        self.player_health = 100
        self.player_mana = 50
        self.inventory = []
        self.level = 1
    
    def can_cast_spell(self, spell_cost):
        return self.player_mana >= spell_cost and self.player_health > 0
    
    def can_use_item(self, item_name):
        return item_name in self.inventory and self.player_health > 0
    
    def level_up_check(self, experience):
        # Using arithmetic and comparison operators
        required_exp = self.level * 100
        if experience >= required_exp:
            self.level += 1
            return True
        return False
```

## Best Practices

### 1. Readability and Clarity

```python
# Bad: Hard to read
result = a and b or c and d or e and f

# Good: Use parentheses for clarity
result = (a and b) or (c and d) or (e and f)

# Bad: Complex ternary
value = x if a else y if b else z if c else w

# Good: Use regular if-else for complex logic
if a:
    value = x
elif b:
    value = y
elif c:
    value = z
else:
    value = w
```

### 2. Operator-Specific Best Practices

```python
# Use 'is' for None comparisons
if value is None:  # Good
    pass

if value == None:  # Works but not recommended
    pass

# Use 'in' for membership tests instead of multiple comparisons
# Bad
if status == 'active' or status == 'pending' or status == 'waiting':
    pass

# Good
if status in {'active', 'pending', 'waiting'}:
    pass

# Floating point comparisons
import math
if math.isclose(a, b, rel_tol=1e-9):  # Good
    pass

if a == b:  # Bad for floats
    pass

# Short-circuit evaluation for performance
# Expensive operation should be second
if cheap_condition() and expensive_operation():
    pass
```

### 3. Common Pitfalls

```python
# Pitfall 1: Mutable default arguments
def bad_function(items=[]):  # Don't do this
    items.append(1)
    return items

def good_function(items=None):  # Do this instead
    if items is None:
        items = []
    items.append(1)
    return items

# Pitfall 2: Chained comparisons misunderstanding
# This doesn't work as expected:
# if 1 < age < 5 or 65:  # Wrong! Always True because of 'or 65'

# Correct way:
if (1 < age < 5) or (age >= 65):
    pass

# Pitfall 3: Integer division confusion
# Python 2 vs Python 3
result = 7 / 2    # 3.5 in Python 3, 3 in Python 2
result = 7 // 2   # 3 in both versions (floor division)

# Pitfall 4: Bitwise vs logical operators
# Wrong: using bitwise for boolean logic
if condition1 & condition2:  # Don't do this
    pass

# Correct: using logical operators
if condition1 and condition2:  # Do this
    pass
```

### 4. Performance Considerations

```python
# Use appropriate data structures for membership tests
# Slow for large datasets
large_list = list(range(10000))
if 9999 in large_list:  # O(n) operation
    pass

# Fast for large datasets
large_set = set(range(10000))
if 9999 in large_set:   # O(1) average operation
    pass

# Use generator expressions for memory efficiency
# Memory intensive
squares = [x**2 for x in range(1000000)]

# Memory efficient
squares = (x**2 for x in range(1000000))

# Use appropriate comparison operators
# For sorting, implement __lt__ instead of __cmp__
class Person:
    def __init__(self, age):
        self.age = age
    
    def __lt__(self, other):
        return self.age < other.age  # Enables sorting
```

## Quick Reference

### Operator Cheat Sheet

```python
# Arithmetic: + - * / // % **
result = 10 + 5 - 2 * 3 / 2 // 1 % 4 ** 2

# Comparison: == != < <= > >=
is_valid = age >= 18 and score > 75

# Logical: and or not
can_proceed = has_permission and (is_admin or is_owner) and not is_blocked

# Bitwise: & | ^ ~ << >>
flags = flag1 | flag2 & ~flag3

# Assignment: = += -= *= /= //= %= **= &= |= ^= <<= >>=
count += 1
balance *= 1.05

# Membership: in not in
if username in valid_users and email not in blocked_emails:
    pass

# Identity: is is not
if value is None or result is not False:
    pass

# Precedence (remember PEMDAS + logic):
# Parentheses -> Exponents -> Multiplication/Division -> Addition/Subtraction -> 
# Shifts -> Bitwise -> Comparison -> Logical NOT -> Logical AND -> Logical OR
```

This comprehensive guide covers all Python operators with practical examples and best practices for effective usage!