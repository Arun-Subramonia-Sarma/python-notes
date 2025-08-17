# Pydantic Complete Guide - Comprehensive Coverage

Pydantic is a Python library that provides data validation and parsing using Python type hints. This guide covers every aspect of Pydantic from basic usage to advanced enterprise patterns.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Basic Models](#basic-models)
- [Field Types and Constraints](#field-types-and-constraints)
- [Validation Scenarios](#validation-scenarios)
- [Advanced Model Features](#advanced-model-features)
- [JSON Handling](#json-handling)
- [Configuration and Settings](#configuration-and-settings)
- [Error Handling and Debugging](#error-handling-and-debugging)
- [Performance Optimization](#performance-optimization)
- [Integration Patterns](#integration-patterns)
- [Testing Strategies](#testing-strategies)
- [Migration and Compatibility](#migration-and-compatibility)
- [Enterprise Patterns](#enterprise-patterns)
- [Troubleshooting](#troubleshooting)

## Installation

```bash
# Basic installation
pip install pydantic

# With email validation
pip install "pydantic[email]"

# With dotenv support
pip install "pydantic[dotenv]"

# Complete installation with all optional dependencies
pip install "pydantic[email,dotenv]"

# Development installation with type stubs
pip install pydantic[email,dotenv] types-pydantic

# For Pydantic v2 (latest)
pip install "pydantic>=2.0"

# For legacy Pydantic v1
pip install "pydantic<2.0"
```

## Quick Start

### Basic Example

```python
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field

class User(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    signup_date: Optional[datetime] = None
    is_active: bool = True
    tags: List[str] = []

# Valid data
user_data = {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "signup_date": "2024-01-15T10:00:00",
    "tags": ["developer", "python"]
}

user = User(**user_data)
print(user.model_dump_json(indent=2))
```

### Error Handling from the Start

```python
from pydantic import ValidationError

try:
    # Invalid data
    invalid_user = User(
        id="not_a_number",  # Should be int
        name="",            # Too short
        email="invalid",    # Invalid email
        tags="not_a_list"   # Should be list
    )
except ValidationError as e:
    print("Validation failed:")
    for error in e.errors():
        print(f"  {error['loc']}: {error['msg']}")
```

## Core Concepts

### BaseModel Fundamentals

```python
from pydantic import BaseModel
from typing import Any, Dict

class CoreModel(BaseModel):
    """Understanding BaseModel core functionality"""
    
    def __init__(self, **data: Any):
        # Custom initialization logic
        if 'computed_field' not in data:
            data['computed_field'] = self._compute_value(data)
        super().__init__(**data)
    
    @staticmethod
    def _compute_value(data: Dict[str, Any]) -> str:
        return f"computed_{data.get('id', 'unknown')}"
    
    # Model methods
    def custom_method(self) -> str:
        return f"Custom logic for {self.__class__.__name__}"
    
    # Property access
    @property
    def display_name(self) -> str:
        return getattr(self, 'name', 'Unknown')
```

### Type System Coverage

```python
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from enum import Enum, IntEnum
from pathlib import Path
from uuid import UUID, uuid4
from typing import (
    List, Dict, Set, Tuple, Optional, Union, Any, Literal, 
    ForwardRef, ClassVar, Final, Annotated
)
from pydantic import BaseModel, Field, constr, conint, confloat

# Enum types
class StatusEnum(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"

class PriorityEnum(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

# Comprehensive type coverage
class AllTypesModel(BaseModel):
    # Basic Python types
    string_field: str
    integer_field: int
    float_field: float
    boolean_field: bool
    bytes_field: bytes
    
    # None and Optional
    optional_string: Optional[str] = None
    nullable_int: Union[int, None] = None
    
    # Collections
    list_of_strings: List[str]
    set_of_ints: Set[int]
    tuple_fixed: Tuple[str, int, bool]
    tuple_variable: Tuple[str, ...]
    dict_string_to_int: Dict[str, int]
    dict_any: Dict[str, Any]
    
    # Union types
    string_or_int: Union[str, int]
    multiple_union: Union[str, int, float, bool]
    
    # Literal types
    literal_values: Literal["option1", "option2", "option3"]
    
    # Specialized types
    decimal_value: Decimal
    uuid_field: UUID
    path_field: Path
    
    # Date and time
    date_field: date
    time_field: time
    datetime_field: datetime
    timedelta_field: timedelta
    
    # Enums
    status: StatusEnum
    priority: PriorityEnum
    
    # Annotated types with constraints
    constrained_string: Annotated[str, Field(min_length=5, max_length=50)]
    positive_int: Annotated[int, Field(gt=0)]
    percentage: Annotated[float, Field(ge=0, le=100)]
    
    # Advanced annotations
    email_pattern: Annotated[str, Field(regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')]
    
    # Class variables (not validated)
    class_constant: ClassVar[str] = "This is not validated"
    
    # Forward references
    self_reference: Optional['AllTypesModel'] = None
    
    class Config:
        # Allow arbitrary types
        arbitrary_types_allowed = True
        # Update forward references
        extra = 'forbid'

# Enable forward reference resolution
AllTypesModel.model_rebuild()
```

## Basic Models

### Model Inheritance Patterns

```python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generic, TypeVar, Optional, List
from pydantic import BaseModel, Field

T = TypeVar('T')

# Abstract base model
class TimestampedModel(BaseModel, ABC):
    """Base model with timestamp functionality"""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    def update_timestamp(self):
        self.updated_at = datetime.now()
    
    @abstractmethod
    def get_identifier(self) -> str:
        pass

# Mixin pattern
class SoftDeleteMixin(BaseModel):
    """Soft delete functionality"""
    is_deleted: bool = False
    deleted_at: Optional[datetime] = None
    
    def soft_delete(self):
        self.is_deleted = True
        self.deleted_at = datetime.now()

# Multiple inheritance
class User(TimestampedModel, SoftDeleteMixin):
    id: int
    name: str
    email: str
    
    def get_identifier(self) -> str:
        return f"user_{self.id}"

# Generic models
class Repository(BaseModel, Generic[T]):
    """Generic repository pattern"""
    items: List[T] = []
    total_count: int = 0
    
    def add_item(self, item: T):
        self.items.append(item)
        self.total_count += 1

# Usage
user_repo = Repository[User](items=[
    User(id=1, name="John", email="john@example.com")
])
```

### Nested and Recursive Models

```python
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Deeply nested structures
class Address(BaseModel):
    street: str
    city: str
    state: str
    country: str
    postal_code: str
    coordinates: Optional[Tuple[float, float]] = None

class ContactInfo(BaseModel):
    email: str
    phone: Optional[str] = None
    address: Address

class Company(BaseModel):
    name: str
    industry: str
    contact: ContactInfo

class Employee(BaseModel):
    id: int
    name: str
    position: str
    company: Company
    direct_reports: List['Employee'] = []
    manager: Optional['Employee'] = None

# Tree-like structures
class TreeNode(BaseModel):
    """Recursive tree structure"""
    id: str
    value: Any
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = []
    metadata: Dict[str, Any] = {}
    
    def add_child(self, child: 'TreeNode'):
        child.parent = self
        self.children.append(child)
    
    def get_depth(self) -> int:
        if not self.parent:
            return 0
        return self.parent.get_depth() + 1

# Enable forward references
Employee.model_rebuild()
TreeNode.model_rebuild()

# Complex nested validation
class NestedValidationModel(BaseModel):
    """Model with complex nested validation rules"""
    
    class NestedConfig(BaseModel):
        enabled: bool
        settings: Dict[str, Union[str, int, bool]]
        
        @validator('settings')
        def validate_settings(cls, v):
            required_keys = {'timeout', 'retries', 'debug'}
            if not required_keys.issubset(v.keys()):
                missing = required_keys - v.keys()
                raise ValueError(f'Missing required settings: {missing}')
            return v
    
    name: str
    config: NestedConfig
    profiles: List[NestedConfig] = []
    
    @root_validator
    def validate_profiles(cls, values):
        config = values.get('config')
        profiles = values.get('profiles', [])
        
        if config and profiles:
            # Ensure at least one profile is enabled if main config is disabled
            if not config.enabled and not any(p.enabled for p in profiles):
                raise ValueError('At least one configuration must be enabled')
        
        return values
```

## Field Types and Constraints

### Comprehensive Field Constraints

```python
from pydantic import BaseModel, Field, constr, conint, confloat, conlist
from typing import List, Optional
import re
from datetime import datetime, date

class ConstrainedFieldsModel(BaseModel):
    # String constraints
    username: constr(
        min_length=3, 
        max_length=20, 
        regex=r'^[a-zA-Z0-9_]+$',
        strip_whitespace=True
    )
    
    # Alternative Field syntax
    email: str = Field(
        ..., 
        regex=r'^[\w\.-]+@[\w\.-]+\.\w+$',
        description="Valid email address",
        example="user@example.com"
    )
    
    # Numeric constraints
    age: conint(ge=0, le=150)  # 0 <= age <= 150
    score: confloat(gt=0, lt=100)  # 0 < score < 100
    rating: float = Field(..., ge=1, le=5, multiple_of=0.5)
    
    # List constraints
    tags: conlist(str, min_items=1, max_items=10)
    coordinates: conlist(float, min_items=2, max_items=3)
    
    # Custom field with multiple constraints
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password with complexity requirements"
    )
    
    # Date constraints
    birth_date: date = Field(..., description="Birth date")
    appointment: datetime = Field(
        ..., 
        description="Future appointment time"
    )
    
    @validator('password')
    def validate_password_complexity(cls, v):
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain special character')
        return v
    
    @validator('appointment')
    def validate_future_date(cls, v):
        if v <= datetime.now():
            raise ValueError('Appointment must be in the future')
        return v
    
    @validator('birth_date')
    def validate_reasonable_age(cls, v):
        today = date.today()
        age = today.year - v.year - ((today.month, today.day) < (v.month, v.day))
        if age > 150 or age < 0:
            raise ValueError('Invalid birth date')
        return v
```

### Custom Field Types

```python
from pydantic import BaseModel, validator
from typing import NewType, Union
import phonenumbers
from phonenumbers import NumberParseException

# Custom types with validation
PhoneNumber = NewType('PhoneNumber', str)
CurrencyCode = NewType('CurrencyCode', str)
CountryCode = NewType('CountryCode', str)

class CustomTypesModel(BaseModel):
    phone: PhoneNumber
    currency: CurrencyCode
    country: CountryCode
    
    @validator('phone')
    def validate_phone_number(cls, v):
        try:
            parsed = phonenumbers.parse(v, None)
            if not phonenumbers.is_valid_number(parsed):
                raise ValueError('Invalid phone number')
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        except NumberParseException:
            raise ValueError('Invalid phone number format')
    
    @validator('currency')
    def validate_currency_code(cls, v):
        # ISO 4217 currency codes
        valid_currencies = {
            'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'INR'
        }
        if v.upper() not in valid_currencies:
            raise ValueError(f'Invalid currency code. Valid codes: {valid_currencies}')
        return v.upper()
    
    @validator('country')
    def validate_country_code(cls, v):
        # ISO 3166-1 alpha-2 country codes (sample)
        valid_countries = {
            'US', 'GB', 'DE', 'FR', 'JP', 'CA', 'AU', 'IN', 'BR', 'MX'
        }
        if v.upper() not in valid_countries:
            raise ValueError(f'Invalid country code. Valid codes: {valid_countries}')
        return v.upper()

# Complex custom field with transformation
class TransformingModel(BaseModel):
    # Automatically clean and format data
    title: str = Field(..., description="Title that gets auto-formatted")
    slug: str = Field(..., description="URL-friendly slug")
    
    @validator('title', pre=True)
    def clean_title(cls, v):
        if isinstance(v, str):
            # Clean up title
            return ' '.join(v.strip().split())
        return v
    
    @validator('slug', pre=True, always=True)
    def generate_slug(cls, v, values):
        if not v and 'title' in values:
            # Auto-generate slug from title
            import re
            slug = re.sub(r'[^\w\s-]', '', values['title'].lower())
            slug = re.sub(r'[\s_-]+', '-', slug)
            return slug.strip('-')
        return v
```

## Validation Scenarios

### Pre and Post Validation Patterns

```python
from pydantic import BaseModel, validator, root_validator
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

class ValidationPatternsModel(BaseModel):
    """Comprehensive validation pattern examples"""
    
    # Raw input that needs preprocessing
    raw_json: str
    processed_data: Optional[Dict[str, Any]] = None
    
    # Fields that depend on each other
    password: str
    password_confirm: str
    
    # Conditional validation
    account_type: str  # 'premium' or 'basic'
    premium_features: Optional[List[str]] = None
    
    # Complex business rules
    start_date: datetime
    end_date: datetime
    duration_days: Optional[int] = None
    
    @validator('raw_json', pre=True)
    def validate_json_string(cls, v):
        """Pre-validation: ensure valid JSON string"""
        if isinstance(v, dict):
            return json.dumps(v)
        if not isinstance(v, str):
            v = str(v)
        
        try:
            json.loads(v)  # Validate JSON format
            return v
        except json.JSONDecodeError:
            raise ValueError('Invalid JSON format')
    
    @validator('processed_data', pre=True, always=True)
    def process_json_data(cls, v, values):
        """Pre-validation: automatically process JSON"""
        if v is None and 'raw_json' in values:
            try:
                return json.loads(values['raw_json'])
            except json.JSONDecodeError:
                return None
        return v
    
    @validator('premium_features')
    def validate_premium_features(cls, v, values):
        """Conditional validation based on account type"""
        account_type = values.get('account_type')
        
        if account_type == 'basic' and v:
            raise ValueError('Basic accounts cannot have premium features')
        
        if account_type == 'premium' and not v:
            raise ValueError('Premium accounts must specify features')
        
        return v
    
    @validator('duration_days', always=True)
    def calculate_duration(cls, v, values):
        """Auto-calculate field based on other fields"""
        start_date = values.get('start_date')
        end_date = values.get('end_date')
        
        if start_date and end_date:
            duration = (end_date - start_date).days
            if v is not None and v != duration:
                raise ValueError(f'Duration mismatch: calculated {duration}, provided {v}')
            return duration
        
        return v
    
    @root_validator
    def validate_password_match(cls, values):
        """Root validation for related fields"""
        password = values.get('password')
        password_confirm = values.get('password_confirm')
        
        if password and password_confirm and password != password_confirm:
            raise ValueError('Passwords do not match')
        
        return values
    
    @root_validator
    def validate_date_logic(cls, values):
        """Complex business rule validation"""
        start_date = values.get('start_date')
        end_date = values.get('end_date')
        
        if start_date and end_date:
            if end_date <= start_date:
                raise ValueError('End date must be after start date')
            
            # Business rule: maximum duration
            max_days = 365
            if (end_date - start_date).days > max_days:
                raise ValueError(f'Maximum duration is {max_days} days')
        
        return values
```

### Dynamic Validation

```python
from pydantic import BaseModel, validator
from typing import Dict, Any, Callable, Optional
import importlib

class DynamicValidationModel(BaseModel):
    """Model with dynamic validation rules"""
    
    data_type: str
    value: Any
    validation_rules: Dict[str, Any] = {}
    
    @validator('value')
    def dynamic_validation(cls, v, values):
        """Apply validation based on data_type"""
        data_type = values.get('data_type')
        rules = values.get('validation_rules', {})
        
        if data_type == 'email':
            import re
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', str(v)):
                raise ValueError('Invalid email format')
        
        elif data_type == 'phone':
            # Remove non-digits
            digits = re.sub(r'\D', '', str(v))
            if len(digits) < 10 or len(digits) > 15:
                raise ValueError('Invalid phone number length')
            return f"+{digits}"
        
        elif data_type == 'number':
            try:
                v = float(v)
                if 'min' in rules and v < rules['min']:
                    raise ValueError(f'Value must be >= {rules["min"]}')
                if 'max' in rules and v > rules['max']:
                    raise ValueError(f'Value must be <= {rules["max"]}')
                return v
            except (ValueError, TypeError):
                raise ValueError('Invalid number format')
        
        elif data_type == 'custom':
            # Load custom validator from rules
            if 'validator_module' in rules and 'validator_function' in rules:
                try:
                    module = importlib.import_module(rules['validator_module'])
                    validator_func = getattr(module, rules['validator_function'])
                    return validator_func(v)
                except (ImportError, AttributeError):
                    raise ValueError('Custom validator not found')
        
        return v

# Plugin-based validation system
class PluginValidationModel(BaseModel):
    """Model with pluggable validation system"""
    
    field_type: str
    field_value: Any
    
    # Registry of validators
    _validators: Dict[str, Callable] = {}
    
    @classmethod
    def register_validator(cls, field_type: str, validator_func: Callable):
        """Register a custom validator for a field type"""
        cls._validators[field_type] = validator_func
    
    @validator('field_value')
    def apply_registered_validator(cls, v, values):
        field_type = values.get('field_type')
        if field_type in cls._validators:
            return cls._validators[field_type](v)
        return v

# Register custom validators
def validate_isbn(value):
    """Validate ISBN format"""
    isbn = str(value).replace('-', '').replace(' ', '')
    if len(isbn) not in [10, 13]:
        raise ValueError('ISBN must be 10 or 13 digits')
    return isbn

PluginValidationModel.register_validator('isbn', validate_isbn)
```

### Async Validation

```python
import asyncio
import aiohttp
from pydantic import BaseModel, validator
from typing import Optional

class AsyncValidationModel(BaseModel):
    """Model with async validation (simulation)"""
    
    username: str
    email: str
    domain_verified: bool = False
    
    class Config:
        # Note: Pydantic validators are inherently sync
        # This is a pattern for post-validation async checks
        validate_assignment = True
    
    @validator('email')
    def basic_email_validation(cls, v):
        """Sync validation first"""
        import re
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', v):
            raise ValueError('Invalid email format')
        return v
    
    async def verify_domain(self) -> bool:
        """Async method for additional verification"""
        domain = self.email.split('@')[1]
        
        # Simulate async domain verification
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f'http://{domain}', timeout=5) as response:
                    self.domain_verified = response.status == 200
                    return self.domain_verified
            except:
                self.domain_verified = False
                return False
    
    async def verify_username_unique(self) -> bool:
        """Simulate async username uniqueness check"""
        # This would typically check against a database
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Mock check - in reality, query your database
        taken_usernames = ['admin', 'user', 'test']
        return self.username.lower() not in taken_usernames

# Usage pattern for async validation
async def create_user_with_validation(user_data):
    """Create user with async validation checks"""
    
    # First, basic validation
    user = AsyncValidationModel(**user_data)
    
    # Then async validation
    domain_ok = await user.verify_domain()
    username_ok = await user.verify_username_unique()
    
    if not domain_ok:
        raise ValueError('Domain verification failed')
    
    if not username_ok:
        raise ValueError('Username already taken')
    
    return user
```

## Advanced Model Features

### Model Configuration Deep Dive

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import json

class AdvancedConfigModel(BaseModel):
    """Comprehensive model configuration examples"""
    
    name: str
    value: int = Field(alias='val')
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        # Field behavior
        allow_population_by_field_name = True  # Allow both 'value' and 'val'
        validate_assignment = True  # Validate when fields are assigned
        use_enum_values = True  # Use enum values in serialization
        
        # Extra fields
        extra = 'forbid'  # Options: 'allow', 'forbid', 'ignore'
        
        # Case sensitivity
        case_sensitive = False
        
        # Validation behavior
        validate_all = True  # Validate all fields even if some fail
        anystr_strip_whitespace = True  # Strip whitespace from strings
        
        # JSON handling
        json_loads = json.loads
        json_dumps = lambda v, *, default: json.dumps(v, default=default, indent=2)
        
        # Custom JSON encoders
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            set: lambda v: list(v),  # Convert sets to lists for JSON
        }
        
        # Schema customization
        title = "Advanced Configuration Model"
        description = "Model demonstrating advanced configuration options"
        
        # Field ordering
        fields = {
            'name': {
                'description': 'The name field',
                'example': 'John Doe',
                'title': 'Full Name'
            },
            'value': {
                'description': 'Numeric value',
                'example': 42,
                'alias': 'val'
            }
        }
        
        # Schema extras
        schema_extra = {
            "examples": [
                {
                    "name": "Sample Item",
                    "val": 100,
                    "metadata": {"category": "test"}
                }
            ]
        }

# Different configuration patterns
class StrictModel(BaseModel):
    """Ultra-strict validation"""
    name: str
    age: int
    
    class Config:
        extra = 'forbid'
        validate_assignment = True
        str_strip_whitespace = True
        validate_all = True

class FlexibleModel(BaseModel):
    """Flexible model allowing extra fields"""
    name: str
    
    class Config:
        extra = 'allow'  # Allow additional fields
        case_sensitive = False

class PerformanceModel(BaseModel):
    """Optimized for performance"""
    id: int
    name: str
    
    class Config:
        # Disable expensive features for better performance
        validate_assignment = False
        copy_on_model_validation = False
        validate_all = False
```

### Model Factories and Builders

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Type, TypeVar, Generic
from datetime import datetime
import uuid

T = TypeVar('T', bound=BaseModel)

class ModelFactory(Generic[T]):
    """Factory pattern for model creation"""
    
    def __init__(self, model_class: Type[T]):
        self.model_class = model_class
        self.default_values: Dict[str, Any] = {}
        self.faker_mapping: Dict[str, callable] = {}
    
    def with_defaults(self, **defaults) -> 'ModelFactory[T]':
        """Set default values for fields"""
        self.default_values.update(defaults)
        return self
    
    def with_faker(self, field: str, faker_func: callable) -> 'ModelFactory[T]':
        """Map field to faker function"""
        self.faker_mapping[field] = faker_func
        return self
    
    def build(self, **overrides) -> T:
        """Build model instance"""
        data = self.default_values.copy()
        
        # Apply faker functions
        for field, faker_func in self.faker_mapping.items():
            if field not in data and field not in overrides:
                data[field] = faker_func()
        
        # Apply overrides
        data.update(overrides)
        
        return self.model_class(**data)
    
    def build_batch(self, count: int, **overrides) -> List[T]:
        """Build multiple instances"""
        return [self.build(**overrides) for _ in range(count)]

# Example model for factory
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    age: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)

# Using the factory
user_factory = (ModelFactory(User)
    .with_defaults(age=25)
    .with_faker('name', lambda: 'Generated User')
    .with_faker('email', lambda: f'user{uuid.uuid4().hex[:8]}@example.com'))

# Create users
user1 = user_factory.build(name='John Doe')
users = user_factory.build_batch(5)

# Builder pattern for complex models
class ComplexModelBuilder:
    """Builder pattern for complex model construction"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._data = {}
        return self
    
    def with_user_info(self, name: str, email: str):
        self._data.update({'name': name, 'email': email})
        return self
    
    def with_preferences(self, **preferences):
        self._data['preferences'] = preferences
        return self
    
    def with_metadata(self, **metadata):
        self._data.setdefault('metadata', {}).update(metadata)
        return self
    
    def build_user(self) -> User:
        if not all(key in self._data for key in ['name', 'email']):
            raise ValueError('Name and email are required')
        
        user = User(**self._data)
        self.reset()  # Reset for next build
        return user

# Usage
builder = ComplexModelBuilder()
user = (builder
    .with_user_info('Jane Doe', 'jane@example.com')
    .with_preferences(theme='dark', notifications=True)
    .with_metadata(source='web', campaign='signup')
    .build_user())
```

### Model Composition and Mixins

```python
from pydantic import BaseModel, Field
from typing import Optional, Protocol, runtime_checkable
from datetime import datetime
from abc import ABC, abstractmethod

# Protocol-based composition
@runtime_checkable
class Timestamped(Protocol):
    created_at: datetime
    updated_at: Optional[datetime]

@runtime_checkable
class Identifiable(Protocol):
    id: str

# Mixin classes
class TimestampMixin(BaseModel):
    """Mixin for timestamp functionality"""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    def touch(self):
        """Update the timestamp"""
        self.updated_at = datetime.now()

class SoftDeleteMixin(BaseModel):
    """Mixin for soft delete functionality"""
    is_deleted: bool = False
    deleted_at: Optional[datetime] = None
    
    def delete(self):
        """Soft delete the record"""
        self.is_deleted = True
        self.deleted_at = datetime.now()
    
    def restore(self):
        """Restore a soft-deleted record"""
        self.is_deleted = False
        self.deleted_at = None

class AuditMixin(BaseModel):
    """Mixin for audit trail"""
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    version: int = 1
    
    def increment_version(self, updated_by: str):
        """Increment version for audit trail"""
        self.version += 1
        self.updated_by = updated_by

# Composed models
class BaseEntity(TimestampMixin, SoftDeleteMixin, AuditMixin):
    """Base entity with common functionality"""
    id: str = Field(..., description="Unique identifier")

class Product(BaseEntity):
    """Product model with all mixins"""
    name: str
    price: float
    description: Optional[str] = None
    
    # Product-specific methods
    def update_price(self, new_price: float, updated_by: str):
        """Update price with audit trail"""
        self.price = new_price
        self.touch()
        self.increment_version(updated_by)

# Composition through delegation
class ModelManager:
    """Manager class for model operations"""
    
    def __init__(self, model_instance: BaseModel):
        self.instance = model_instance
    
    def validate_and_save(self) -> bool:
        """Validate model and perform save operations"""
        try:
            # Trigger validation
            self.instance.dict()
            
            # Perform timestamp operations if supported
            if isinstance(self.instance, Timestamped):
                if hasattr(self.instance, 'touch'):
                    self.instance.touch()
            
            return True
        except Exception as e:
            print(f"Validation failed: {e}")
            return False
    
    def safe_delete(self) -> bool:
        """Safely delete model"""
        if hasattr(self.instance, 'delete'):
            self.instance.delete()
            return True
        return False
```

## JSON Handling

### Advanced JSON Serialization

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Set, List
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
import json
from uuid import UUID

class StatusEnum(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

class AdvancedJSONModel(BaseModel):
    """Model with comprehensive JSON handling"""
    
    # Basic fields
    id: UUID
    name: str
    status: StatusEnum
    
    # Complex types
    metadata: Dict[str, Any] = {}
    tags: Set[str] = set()
    scores: List[float] = []
    
    # Date/time fields
    created_at: datetime
    birth_date: Optional[date] = None
    
    # Decimal for precise numbers
    price: Decimal
    
    # Optional fields
    description: Optional[str] = None
    
    class Config:
        # Custom JSON encoders
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            Decimal: float,
            set: list,  # Convert sets to lists for JSON
        }
        
        # Use enum values
        use_enum_values = True

# Serialization control
class SerializationControlModel(BaseModel):
    """Model demonstrating serialization control"""
    
    public_field: str
    private_field: str = Field(..., exclude=True)  # Never serialize
    admin_field: str = Field(..., description="Admin only field")
    computed_field: Optional[str] = None
    
    # Sensitive data
    password_hash: str = Field(..., alias="pwd", write_only=True)
    
    @property
    def computed_value(self) -> str:
        """Computed property for serialization"""
        return f"computed_{self.public_field}"
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Custom dict method with computed fields"""
        data = super().dict(**kwargs)
        
        # Add computed fields
        if 'exclude' not in kwargs or 'computed_field' not in kwargs.get('exclude', set()):
            data['computed_field'] = self.computed_value
        
        return data
    
    def json_for_user(self, is_admin: bool = False) -> str:
        """Context-aware JSON serialization"""
        exclude = {'private_field', 'password_hash'}
        if not is_admin:
            exclude.add('admin_field')
        
        return self.json(exclude=exclude)

# Custom JSON handling
class CustomJSONModel(BaseModel):
    """Model with custom JSON serialization logic"""
    
    name: str
    data: Dict[str, Any]
    
    @classmethod
    def parse_raw(cls, b: bytes, **kwargs):
        """Custom JSON parsing with preprocessing"""
        # Preprocess JSON before parsing
        json_str = b.decode('utf-8')
        
        # Handle special cases in JSON
        json_str = json_str.replace('null', '"null"')  # Example preprocessing
        
        return cls.parse_obj(json.loads(json_str))
    
    def json(self, **kwargs) -> str:
        """Custom JSON serialization with post-processing"""
        # Get standard JSON
        json_str = super().json(**kwargs)
        
        # Post-process if needed
        # Example: pretty printing in development
        if kwargs.get('indent'):
            data = json.loads(json_str)
            return json.dumps(data, indent=kwargs['indent'], sort_keys=True)
        
        return json_str

# Streaming JSON for large datasets
class StreamingJSONModel(BaseModel):
    """Model for handling large JSON datasets"""
    
    items: List[Dict[str, Any]] = []
    total_count: int = 0
    
    @classmethod
    def from_json_stream(cls, json_stream):
        """Create model from streaming JSON"""
        items = []
        for line in json_stream:
            try:
                item = json.loads(line.strip())
                items.append(item)
            except json.JSONDecodeError:
                continue  # Skip invalid lines
        
        return cls(items=items, total_count=len(items))
    
    def to_json_lines(self) -> str:
        """Export as JSON lines format"""
        lines = []
        for item in self.items:
            lines.append(json.dumps(item))
        return '\n'.join(lines)
```

### Schema Generation and Documentation

```python
from pydantic import BaseModel, Field, Schema
from typing import List, Optional, Dict, Any
from enum import Enum

class APIDocumentationModel(BaseModel):
    """Model with comprehensive documentation for API schema generation"""
    
    user_id: int = Field(
        ..., 
        title="User ID",
        description="Unique identifier for the user",
        example=12345,
        ge=1
    )
    
    full_name: str = Field(
        ...,
        title="Full Name",
        description="User's complete name",
        example="John Doe Smith",
        min_length=2,
        max_length=100
    )
    
    email: str = Field(
        ...,
        title="Email Address",
        description="Valid email address for communication",
        example="john.doe@example.com",
        regex=r'^[\w\.-]+@[\w\.-]+\.\w+$'
    )
    
    age: Optional[int] = Field(
        None,
        title="Age",
        description="User's age in years",
        example=25,
        ge=0,
        le=150
    )
    
    preferences: Dict[str, Any] = Field(
        default_factory=dict,
        title="User Preferences",
        description="Customizable user preferences",
        example={"theme": "dark", "notifications": True}
    )
    
    tags: List[str] = Field(
        default_factory=list,
        title="Tags",
        description="List of tags associated with the user",
        example=["developer", "python", "ai"],
        max_items=10
    )
    
    class Config:
        title = "User Profile"
        description = "Complete user profile information"
        schema_extra = {
            "examples": [
                {
                    "user_id": 12345,
                    "full_name": "John Doe",
                    "email": "john.doe@example.com",
                    "age": 30,
                    "preferences": {
                        "theme": "dark",
                        "language": "en",
                        "notifications": True
                    },
                    "tags": ["developer", "python"]
                },
                {
                    "user_id": 67890,
                    "full_name": "Jane Smith",
                    "email": "jane.smith@example.com",
                    "preferences": {},
                    "tags": []
                }
            ]
        }

# Generate and inspect schema
def generate_api_documentation():
    """Generate comprehensive API documentation"""
    
    schema = APIDocumentationModel.schema()
    
    print("=== Generated JSON Schema ===")
    print(json.dumps(schema, indent=2))
    
    print("\n=== Field Information ===")
    for field_name, field_info in schema.get('properties', {}).items():
        print(f"Field: {field_name}")
        print(f"  Type: {field_info.get('type', 'unknown')}")
        print(f"  Title: {field_info.get('title', 'N/A')}")
        print(f"  Description: {field_info.get('description', 'N/A')}")
        print(f"  Example: {field_info.get('example', 'N/A')}")
        print(f"  Required: {field_name in schema.get('required', [])}")
        print()

# Custom schema modifications
class CustomSchemaModel(BaseModel):
    """Model with custom schema modifications"""
    
    secret_field: str = Field(..., description="This field won't appear in public schema")
    public_field: str
    
    @classmethod
    def schema(cls, by_alias: bool = True, ref_template: str = '#/definitions/{model}') -> Dict[str, Any]:
        """Custom schema generation"""
        schema = super().schema(by_alias=by_alias, ref_template=ref_template)
        
        # Remove sensitive fields from public schema
        if 'properties' in schema and 'secret_field' in schema['properties']:
            del schema['properties']['secret_field']
            
            # Update required fields
            if 'required' in schema and 'secret_field' in schema['required']:
                schema['required'].remove('secret_field')
        
        # Add custom schema information
        schema['custom_info'] = {
            'version': '1.0',
            'generated_at': datetime.now().isoformat()
        }
        
        return schema
```

## Configuration and Settings

### Environment-Based Configuration

```python
from pydantic import BaseSettings, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import os
from functools import lru_cache

class DatabaseSettings(BaseSettings):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    username: str
    password: str
    database: str
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    # Computed property
    @property
    def url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_prefix = "DB_"  # DB_HOST, DB_PORT, etc.

class RedisSettings(BaseSettings):
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 10
    
    class Config:
        env_prefix = "REDIS_"

class LoggingSettings(BaseSettings):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[Path] = None
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    @validator('level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Invalid log level. Choose from: {valid_levels}')
        return v.upper()
    
    class Config:
        env_prefix = "LOG_"

class SecuritySettings(BaseSettings):
    """Security configuration"""
    secret_key: str = Field(..., min_length=32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    allowed_hosts: List[str] = ["localhost", "127.0.0.1"]
    cors_origins: List[str] = []
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v
    
    class Config:
        env_prefix = "SEC_"

class AppSettings(BaseSettings):
    """Main application settings"""
    app_name: str = "My Application"
    version: str = "1.0.0"
    debug: bool = False
    environment: str = Field("development", env="ENV")
    
    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    security: SecuritySettings
    
    # Feature flags
    feature_flags: Dict[str, bool] = Field(default_factory=dict)
    
    # External service URLs
    external_apis: Dict[str, str] = Field(default_factory=dict)
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of: {valid_envs}')
        return v
    
    @root_validator
    def validate_production_settings(cls, values):
        """Additional validation for production environment"""
        env = values.get('environment')
        debug = values.get('debug')
        
        if env == 'production':
            if debug:
                raise ValueError('Debug mode cannot be enabled in production')
            
            # Ensure security settings are properly configured for production
            security = values.get('security')
            if security and len(security.secret_key) < 64:
                raise ValueError('Production secret key should be at least 64 characters')
        
        return values
    
    class Config:
        env_file = ['.env.local', '.env']  # Load multiple env files
        env_file_encoding = 'utf-8'
        case_sensitive = False
        
        # Custom env loading
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            """Customize settings sources priority"""
            return (
                init_settings,  # Highest priority
                env_settings,   # Environment variables
                file_secret_settings,  # Lowest priority
            )

# Singleton pattern for settings
@lru_cache()
def get_settings() -> AppSettings:
    """Get application settings (cached)"""
    return AppSettings()

# Environment-specific settings
class DevelopmentSettings(AppSettings):
    """Development environment settings"""
    debug: bool = True
    
    class Config(AppSettings.Config):
        env_file = '.env.development'

class ProductionSettings(AppSettings):
    """Production environment settings"""
    debug: bool = False
    
    class Config(AppSettings.Config):
        env_file = '.env.production'

def get_settings_for_environment(env: str) -> AppSettings:
    """Factory function for environment-specific settings"""
    settings_map = {
        'development': DevelopmentSettings,
        'staging': AppSettings,
        'production': ProductionSettings,
    }
    
    settings_class = settings_map.get(env, AppSettings)
    return settings_class()

# Dynamic configuration
class DynamicConfigModel(BaseSettings):
    """Configuration that can be updated at runtime"""
    
    # Core settings
    api_rate_limit: int = 1000
    cache_ttl: int = 3600
    worker_count: int = 4
    
    # Feature toggles
    new_feature_enabled: bool = False
    maintenance_mode: bool = False
    
    # Runtime modifiable settings
    _modifiable_fields = {
        'api_rate_limit', 'cache_ttl', 'new_feature_enabled', 'maintenance_mode'
    }
    
    def update_setting(self, key: str, value: Any) -> bool:
        """Update a setting at runtime"""
        if key not in self._modifiable_fields:
            raise ValueError(f'Setting {key} cannot be modified at runtime')
        
        # Validate the new value
        try:
            # Create a new instance to validate
            new_data = self.dict()
            new_data[key] = value
            validated = self.__class__(**new_data)
            
            # If validation passes, update this instance
            setattr(self, key, value)
            return True
            
        except Exception as e:
            raise ValueError(f'Invalid value for {key}: {e}')
    
    def reload_from_env(self):
        """Reload configuration from environment"""
        new_settings = self.__class__()
        for field in self._modifiable_fields:
            setattr(self, field, getattr(new_settings, field))
    
    class Config:
        env_file = '.env'
        validate_assignment = True  # Validate when settings change
```

### Configuration Management Patterns

```python
from pydantic import BaseSettings, Field
from typing import Dict, Any, Optional, Type, TypeVar, Generic
from pathlib import Path
import json
import yaml
from abc import ABC, abstractmethod

T = TypeVar('T', bound=BaseSettings)

class ConfigLoader(Generic[T], ABC):
    """Abstract configuration loader"""
    
    def __init__(self, config_class: Type[T]):
        self.config_class = config_class
    
    @abstractmethod
    def load(self) -> T:
        """Load configuration from source"""
        pass

class FileConfigLoader(ConfigLoader[T]):
    """Load configuration from files"""
    
    def __init__(self, config_class: Type[T], file_path: Path):
        super().__init__(config_class)
        self.file_path = file_path
    
    def load(self) -> T:
        """Load from JSON or YAML file"""
        if not self.file_path.exists():
            return self.config_class()
        
        with open(self.file_path, 'r') as f:
            if self.file_path.suffix == '.json':
                data = json.load(f)
            elif self.file_path.suffix in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f'Unsupported file type: {self.file_path.suffix}')
        
        return self.config_class(**data)

class EnvironmentConfigLoader(ConfigLoader[T]):
    """Load configuration from environment variables"""
    
    def load(self) -> T:
        return self.config_class()

class MultiSourceConfigLoader(ConfigLoader[T]):
    """Load configuration from multiple sources with priority"""
    
    def __init__(self, config_class: Type[T], loaders: List[ConfigLoader[T]]):
        super().__init__(config_class)
        self.loaders = loaders
    
    def load(self) -> T:
        """Load from multiple sources, later sources override earlier ones"""
        config_data = {}
        
        for loader in self.loaders:
            try:
                config = loader.load()
                config_data.update(config.dict())
            except Exception as e:
                print(f"Failed to load from {loader}: {e}")
                continue
        
        return self.config_class(**config_data)

# Configuration with hot-reload
class HotReloadConfig(BaseSettings):
    """Configuration that supports hot reloading"""
    
    api_key: str
    timeout: int = 30
    debug: bool = False
    
    # Internal tracking
    _file_path: Optional[Path] = None
    _last_modified: Optional[float] = None
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'HotReloadConfig':
        """Create configuration from file with hot reload capability"""
        instance = cls()
        instance._file_path = file_path
        instance.reload()
        return instance
    
    def reload(self) -> bool:
        """Reload configuration if file has changed"""
        if not self._file_path or not self._file_path.exists():
            return False
        
        current_modified = self._file_path.stat().st_mtime
        if self._last_modified and current_modified <= self._last_modified:
            return False  # No changes
        
        try:
            with open(self._file_path, 'r') as f:
                data = json.load(f)
            
            # Update fields
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            self._last_modified = current_modified
            return True
            
        except Exception as e:
            print(f"Failed to reload config: {e}")
            return False
    
    def should_reload(self) -> bool:
        """Check if configuration should be reloaded"""
        if not self._file_path:
            return False
        
        current_modified = self._file_path.stat().st_mtime
        return current_modified > (self._last_modified or 0)

# Configuration validation and transformation
class ValidatedConfig(BaseSettings):
    """Configuration with comprehensive validation"""
    
    # Database settings
    database_url: str
    connection_pool_size: int = Field(10, ge=1, le=100)
    
    # Cache settings
    cache_backend: str = Field("redis", regex=r'^(redis|memcached|memory)$')
    cache_ttl: int = Field(3600, gt=0)
    
    # Security settings
    allowed_origins: List[str] = Field(default_factory=list)
    jwt_secret: str = Field(..., min_length=32)
    
    @validator('database_url')
    def validate_database_url(cls, v):
        """Validate database URL format"""
        import re
        pattern = r'^(postgresql|mysql|sqlite)://.*'
        if not re.match(pattern, v):
            raise ValueError('Invalid database URL format')
        return v
    
    @validator('allowed_origins')
    def validate_origins(cls, v):
        """Validate origin URLs"""
        import re
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        
        for origin in v:
            if origin != '*' and not re.match(url_pattern, origin):
                raise ValueError(f'Invalid origin URL: {origin}')
        
        return v
    
    @root_validator
    def validate_security_settings(cls, values):
        """Validate security configuration consistency"""
        allowed_origins = values.get('allowed_origins', [])
        
        # In production, don't allow wildcard origins
        if '*' in allowed_origins and not values.get('debug', False):
            raise ValueError('Wildcard origins not allowed in production')
        
        return values
    
    class Config:
        # Custom validation message
        error_msg_templates = {
            'value_error.missing': 'This field is required',
            'value_error.number.not_ge': 'Value must be greater than or equal to {limit_value}',
            'value_error.str.regex': 'Value does not match required pattern',
        }
```

## Error Handling and Debugging

### Comprehensive Error Handling

```python
from pydantic import BaseModel, ValidationError, validator, root_validator, Field
from typing import List, Dict, Any, Optional, Union
import traceback
from datetime import datetime

class DetailedValidationError:
    """Enhanced error information for debugging"""
    
    def __init__(self, validation_error: ValidationError, context: Dict[str, Any] = None):
        self.original_error = validation_error
        self.context = context or {}
        self.timestamp = datetime.now()
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Generate detailed error report"""
        errors = []
        
        for error in self.original_error.errors():
            field_path = ' -> '.join(str(loc) for loc in error['loc'])
            
            error_detail = {
                'field': field_path,
                'error_type': error['type'],
                'message': error['msg'],
                'invalid_value': error.get('input'),
                'constraint': error.get('ctx', {}),
            }
            
            errors.append(error_detail)
        
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_errors': len(errors),
            'errors': errors,
            'context': self.context,
            'raw_error': str(self.original_error)
        }
    
    def format_for_user(self) -> str:
        """Format error message for end users"""
        if len(self.original_error.errors()) == 1:
            error = self.original_error.errors()[0]
            field = ' -> '.join(str(loc) for loc in error['loc'])
            return f"Error in {field}: {error['msg']}"
        
        return f"Found {len(self.original_error.errors())} validation errors. Please check your input."
    
    def format_for_developer(self) -> str:
        """Format error message for developers"""
        lines = ["Validation Error Details:"]
        
        for i, error in enumerate(self.original_error.errors(), 1):
            field_path = ' -> '.join(str(loc) for loc in error['loc'])
            lines.append(f"  {i}. Field: {field_path}")
            lines.append(f"     Type: {error['type']}")
            lines.append(f"     Message: {error['msg']}")
            lines.append(f"     Value: {error.get('input', 'N/A')}")
            lines.append("")
        
        return '\n'.join(lines)

class ErrorHandlingModel(BaseModel):
    """Model demonstrating comprehensive error handling"""
    
    name: str = Field(..., min_length=2, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: int = Field(..., ge=0, le=150)
    tags: List[str] = Field(..., min_items=1)
    metadata: Dict[str, Union[str, int, bool]] = {}
    
    @validator('name')
    def validate_name_content(cls, v):
        """Custom validation with detailed error messages"""
        if not v.strip():
            raise ValueError("Name cannot be empty or whitespace only")
        
        if any(char.isdigit() for char in v):
            raise ValueError("Name cannot contain numbers")
        
        forbidden_chars = ['@', '#', '$', '%', '&']
        if any(char in v for char in forbidden_chars):
            raise ValueError(f"Name cannot contain these characters: {', '.join(forbidden_chars)}")
        
        return v.strip()
    
    @validator('tags')
    def validate_tags_content(cls, v):
        """Validate individual tags"""
        if not v:
            raise ValueError("At least one tag is required")
        
        for i, tag in enumerate(v):
            if not tag.strip():
                raise ValueError(f"Tag at position {i+1} cannot be empty")
            
            if len(tag) > 20:
                raise ValueError(f"Tag at position {i+1} is too long (max 20 characters)")
        
        # Check for duplicates
        unique_tags = set(tag.lower() for tag in v)
        if len(unique_tags) != len(v):
            raise ValueError("Duplicate tags are not allowed")
        
        return v
    
    @root_validator
    def validate_metadata_consistency(cls, values):
        """Validate metadata based on other fields"""
        metadata = values.get('metadata', {})
        name = values.get('name', '')
        
        # Business rule: if name starts with 'Admin', metadata must have 'role': 'admin'
        if name.startswith('Admin') and metadata.get('role') != 'admin':
            raise ValueError("Users with 'Admin' prefix must have role='admin' in metadata")
        
        return values

def safe_model_creation(model_class, data: Dict[str, Any], context: Dict[str, Any] = None) -> Union[BaseModel, DetailedValidationError]:
    """Safely create model with detailed error handling"""
    try:
        return model_class(**data)
    
    except ValidationError as e:
        return DetailedValidationError(e, context)
    
    except Exception as e:
        # Handle unexpected errors
        synthetic_error = ValidationError([{
            'loc': ('__root__',),
            'msg': f'Unexpected error: {str(e)}',
            'type': 'unexpected_error'
        }], model_class)
        
        return DetailedValidationError(synthetic_error, {
            'original_exception': str(e),
            'traceback': traceback.format_exc(),
            **(context or {})
        })

# Error recovery and partial validation
class PartialValidationModel(BaseModel):
    """Model that supports partial validation and error recovery"""
    
    name: str
    email: str
    age: Optional[int] = None
    phone: Optional[str] = None
    
    @classmethod
    def create_with_partial_data(cls, data: Dict[str, Any]) -> 'PartialValidationResult':
        """Create model allowing partial success"""
        result = PartialValidationResult()
        validated_data = {}
        
        # Try to validate each field individually
        for field_name, field_value in data.items():
            try:
                # Create a minimal model to validate just this field
                temp_data = {field_name: field_value}
                # Fill other required fields with defaults
                if field_name != 'name':
                    temp_data['name'] = 'temp'
                if field_name != 'email':
                    temp_data['email'] = 'temp@example.com'
                
                # Validate with temporary model
                temp_model = cls(**temp_data)
                validated_data[field_name] = getattr(temp_model, field_name)
                result.valid_fields[field_name] = validated_data[field_name]
                
            except ValidationError as e:
                # Field validation failed
                field_errors = [err for err in e.errors() if field_name in str(err['loc'])]
                result.field_errors[field_name] = field_errors
            
            except Exception as e:
                result.field_errors[field_name] = [{'msg': str(e), 'type': 'unexpected'}]
        
        # Try to create the full model with valid data
        try:
            if validated_data:
                result.model = cls(**validated_data)
        except ValidationError as e:
            result.model_errors = e.errors()
        
        return result

class PartialValidationResult:
    """Result of partial validation"""
    
    def __init__(self):
        self.model: Optional[PartialValidationModel] = None
        self.valid_fields: Dict[str, Any] = {}
        self.field_errors: Dict[str, List[Dict[str, Any]]] = {}
        self.model_errors: List[Dict[str, Any]] = []
    
    @property
    def is_complete_success(self) -> bool:
        return self.model is not None and not self.field_errors
    
    @property
    def has_partial_success(self) -> bool:
        return bool(self.valid_fields)
    
    def get_error_summary(self) -> Dict[str, Any]:
        return {
            'field_errors': self.field_errors,
            'model_errors': self.model_errors,
            'valid_field_count': len(self.valid_fields),
            'error_field_count': len(self.field_errors)
        }

# Advanced debugging utilities
class DebugModel(BaseModel):
    """Model with built-in debugging capabilities"""
    
    name: str
    value: int
    metadata: Dict[str, Any] = {}
    
    _debug_info: Dict[str, Any] = {}
    
    def __init__(self, **data):
        # Capture debug information
        self._debug_info = {
            'input_data': data.copy(),
            'validation_start': datetime.now(),
            'field_transformations': {}
        }
        
        try:
            super().__init__(**data)
            self._debug_info['validation_success'] = True
            self._debug_info['validation_end'] = datetime.now()
        except Exception as e:
            self._debug_info['validation_success'] = False
            self._debug_info['validation_error'] = str(e)
            self._debug_info['validation_end'] = datetime.now()
            raise
    
    @validator('name', pre=True)
    def debug_name_transformation(cls, v):
        """Track name transformations for debugging"""
        original = v
        processed = str(v).strip().title()
        
        # This would be stored in instance if we had access
        # In practice, you'd use a different approach for tracking
        return processed
    
    def get_debug_report(self) -> Dict[str, Any]:
        """Get comprehensive debugging information"""
        duration = None
        if 'validation_start' in self._debug_info and 'validation_end' in self._debug_info:
            duration = (self._debug_info['validation_end'] - self._debug_info['validation_start']).total_seconds()
        
        return {
            'validation_duration_seconds': duration,
            'input_data': self._debug_info.get('input_data'),
            'output_data': self.dict() if self._debug_info.get('validation_success') else None,
            'validation_success': self._debug_info.get('validation_success'),
            'validation_error': self._debug_info.get('validation_error'),
            'field_transformations': self._debug_info.get('field_transformations', {}),
            'model_fields': list(self.__fields__.keys()),
            'field_types': {k: str(v.type_) for k, v in self.__fields__.items()}
        }

# Exception chaining for complex validation
class ValidationException(Exception):
    """Custom exception with validation context"""
    
    def __init__(self, message: str, field: str = None, value: Any = None, original_exception: Exception = None):
        self.field = field
        self.value = value
        self.original_exception = original_exception
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message': str(self),
            'field': self.field,
            'value': self.value,
            'original_error': str(self.original_exception) if self.original_exception else None
        }

class ChainedValidationModel(BaseModel):
    """Model with chained validation that provides context"""
    
    email: str
    username: str
    
    @validator('email')
    def validate_email_format(cls, v):
        try:
            import re
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+, v):
                raise ValidationException(
                    "Invalid email format",
                    field='email',
                    value=v
                )
            return v
        except ValidationException:
            raise
        except Exception as e:
            raise ValidationException(
                "Email validation failed due to unexpected error",
                field='email',
                value=v,
                original_exception=e
            )
    
    @validator('username')
    def validate_username_availability(cls, v):
        try:
            # Simulate checking username availability
            taken_usernames = ['admin', 'root', 'user']
            if v.lower() in taken_usernames:
                raise ValidationException(
                    f"Username '{v}' is not available",
                    field='username',
                    value=v
                )
            return v
        except ValidationException:
            raise
        except Exception as e:
            raise ValidationException(
                "Username validation failed",
                field='username',
                value=v,
                original_exception=e
            )

## Performance Optimization

### Memory and Speed Optimization

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, ClassVar
import sys
from functools import lru_cache
import orjson  # High-performance JSON library

# Memory-optimized model
class OptimizedModel(BaseModel):
    """Model optimized for memory usage"""
    
    # Use __slots__ to reduce memory overhead
    __slots__ = ('_id', '_name', '_value')
    
    id: int
    name: str
    value: float
    
    class Config:
        # Optimize for performance
        validate_assignment = False  # Skip validation on assignment
        copy_on_model_validation = False  # Avoid unnecessary copying
        validate_all = False  # Stop on first validation error
        allow_reuse = True  # Reuse validators
        
        # Use faster JSON library if available
        json_loads = orjson.loads if 'orjson' in sys.modules else None
        json_dumps = (
            lambda v, *, default: orjson.dumps(v, default=default).decode()
            if 'orjson' in sys.modules else None
        )

# Cached validation patterns
class CachedValidationModel(BaseModel):
    """Model with cached expensive operations"""
    
    data: str
    computed_hash: Optional[str] = None
    
    @lru_cache(maxsize=1000)
    @classmethod
    def _expensive_validation(cls, value: str) -> bool:
        """Cached expensive validation operation"""
        # Simulate expensive validation
        import hashlib
        return len(hashlib.sha256(value.encode()).hexdigest()) == 64
    
    @validator('data')
    def validate_data_expensive(cls, v):
        if not cls._expensive_validation(v):
            raise ValueError('Data validation failed')
        return v
    
    @property
    @lru_cache(maxsize=None)
    def computed_value(self) -> str:
        """Expensive computed property with caching"""
        import hashlib
        return hashlib.sha256(self.data.encode()).hexdigest()

# Bulk processing optimization
class BulkProcessingModel(BaseModel):
    """Model optimized for bulk operations"""
    
    items: List[Dict[str, Any]]
    
    @classmethod
    def validate_bulk(cls, items_data: List[Dict[str, Any]]) -> List[Union['BulkProcessingModel', ValidationError]]:
        """Bulk validation with error collection"""
        results = []
        
        for item_data in items_data:
            try:
                result = cls(**item_data)
                results.append(result)
            except ValidationError as e:
                results.append(e)
        
        return results
    
    @classmethod
    def create_batch(cls, items_data: List[Dict[str, Any]]) -> 'BatchResult':
        """Create batch with success/failure tracking"""
        batch_result = BatchResult()
        
        for i, item_data in enumerate(items_data):
            try:
                model = cls(**item_data)
                batch_result.successful.append(model)
            except ValidationError as e:
                batch_result.failed.append({
                    'index': i,
                    'data': item_data,
                    'error': e
                })
        
        return batch_result

class BatchResult:
    """Result of batch processing"""
    
    def __init__(self):
        self.successful: List[BulkProcessingModel] = []
        self.failed: List[Dict[str, Any]] = []
    
    @property
    def success_rate(self) -> float:
        total = len(self.successful) + len(self.failed)
        return len(self.successful) / total if total > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_processed': len(self.successful) + len(self.failed),
            'successful': len(self.successful),
            'failed': len(self.failed),
            'success_rate': self.success_rate
        }

# Lazy loading and streaming
class LazyLoadModel(BaseModel):
    """Model with lazy loading capabilities"""
    
    id: str
    name: str
    _heavy_data: Optional[Dict[str, Any]] = None
    
    class Config:
        # Don't include private fields in dict/json output
        underscore_attrs_are_private = True
    
    @property
    def heavy_data(self) -> Dict[str, Any]:
        """Lazy load expensive data"""
        if self._heavy_data is None:
            # Simulate loading from external source
            self._heavy_data = self._load_heavy_data()
        return self._heavy_data
    
    def _load_heavy_data(self) -> Dict[str, Any]:
        """Simulate expensive data loading"""
        # In practice, this would load from database, API, etc.
        return {
            'large_dataset': list(range(1000)),
            'computed_metrics': {'avg': 500, 'sum': 499500}
        }
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Custom dict method that doesn't trigger lazy loading"""
        data = super().dict(**kwargs)
        
        # Only include heavy_data if it's already loaded
        if self._heavy_data is not None:
            data['heavy_data'] = self._heavy_data
        
        return data

## Integration Patterns

### FastAPI Integration

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import HTTPException
from enum import Enum

# API Models
class HTTPErrorResponse(BaseModel):
    """Standardized error response"""
    error: bool = True
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "error": True,
                "message": "Validation failed",
                "details": {
                    "field_errors": ["name: field required"]
                },
                "timestamp": "2024-01-01T12:00:00"
            }
        }

class PaginationRequest(BaseModel):
    """Standardized pagination parameters"""
    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(10, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("asc", regex="^(asc|desc)$")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

class PaginatedResponse(BaseModel):
    """Standardized paginated response"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool
    
    @classmethod
    def create(cls, items: List[Any], total: int, pagination: PaginationRequest):
        total_pages = (total + pagination.page_size - 1) // pagination.page_size
        
        return cls(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
            total_pages=total_pages,
            has_next=pagination.page < total_pages,
            has_previous=pagination.page > 1
        )

# Request/Response Models
class UserCreateRequest(BaseModel):
    """Request model for creating users"""
    name: str = Field(..., min_length=2, max_length=100, example="John Doe")
    email: str = Field(..., example="john@example.com")
    age: Optional[int] = Field(None, ge=13, le=120, example=25)
    tags: List[str] = Field(default_factory=list, max_items=10)
    
    @validator('email')
    def validate_email(cls, v):
        import re
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+, v):
            raise ValueError('Invalid email format')
        return v.lower()

class UserResponse(BaseModel):
    """Response model for user data"""
    id: str
    name: str
    email: str
    age: Optional[int]
    tags: List[str]
    created_at: datetime
    is_active: bool = True
    
    class Config:
        # Enable ORM mode for SQLAlchemy integration
        from_attributes = True  # Pydantic v2
        # orm_mode = True  # Pydantic v1
        
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UserUpdateRequest(BaseModel):
    """Request model for updating users"""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[str] = None
    age: Optional[int] = Field(None, ge=13, le=120)
    tags: Optional[List[str]] = Field(None, max_items=10)
    
    @validator('email')
    def validate_email(cls, v):
        if v is not None:
            import re
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+, v):
                raise ValueError('Invalid email format')
            return v.lower()
        return v

# Custom exception handling for FastAPI
def create_validation_error_handler():
    """Create custom validation error handler for FastAPI"""
    
    def validation_exception_handler(request, exc):
        """Convert Pydantic ValidationError to HTTP 422"""
        errors = []
        for error in exc.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            errors.append({
                "field": field_path,
                "message": error["msg"],
                "type": error["type"]
            })
        
        return HTTPErrorResponse(
            message="Validation failed",
            details={"field_errors": errors}
        )
    
    return validation_exception_handler

### Database Integration (SQLAlchemy)

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

Base = declarative_base()

# SQLAlchemy model
class UserTable(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    age = Column(Integer, nullable=True)
    bio = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Pydantic models for different use cases
class UserBase(BaseModel):
    """Base user model with common fields"""
    name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., max_length=255)
    age: Optional[int] = Field(None, ge=13, le=120)
    bio: Optional[str] = Field(None, max_length=1000)

class UserCreate(UserBase):
    """Model for creating users"""
    pass

class UserUpdate(BaseModel):
    """Model for updating users (all fields optional)"""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[str] = Field(None, max_length=255)
    age: Optional[int] = Field(None, ge=13, le=120)
    bio: Optional[str] = Field(None, max_length=1000)
    is_active: Optional[bool] = None

class UserInDB(UserBase):
    """Model representing user as stored in database"""
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True  # Enable ORM mode

class UserPublic(UserBase):
    """Model for public user representation"""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Repository pattern with Pydantic integration
class UserRepository:
    """Repository for user operations with Pydantic models"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(self, user_create: UserCreate) -> UserInDB:
        """Create user from Pydantic model"""
        # Convert Pydantic model to dict
        user_data = user_create.dict()
        
        # Create SQLAlchemy model
        db_user = UserTable(**user_data)
        
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        
        # Convert back to Pydantic model
        return UserInDB.from_orm(db_user)
    
    def get_user(self, user_id: int) -> Optional[UserInDB]:
        """Get user by ID"""
        db_user = self.db.query(UserTable).filter(UserTable.id == user_id).first()
        return UserInDB.from_orm(db_user) if db_user else None
    
    def update_user(self, user_id: int, user_update: UserUpdate) -> Optional[UserInDB]:
        """Update user with Pydantic model"""
        db_user = self.db.query(UserTable).filter(UserTable.id == user_id).first()
        if not db_user:
            return None
        
        # Update only provided fields
        update_data = user_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_user, field, value)
        
        self.db.commit()
        self.db.refresh(db_user)
        
        return UserInDB.from_orm(db_user)
    
    def list_users(self, skip: int = 0, limit: int = 100) -> List[UserInDB]:
        """List users with pagination"""
        db_users = (
            self.db.query(UserTable)
            .offset(skip)
            .limit(limit)
            .all()
        )
        
        return [UserInDB.from_orm(user) for user in db_users]

### Message Queue Integration

```python
import json
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, Union, Literal
from datetime import datetime
from enum import Enum
import uuid

class MessagePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class BaseMessage(BaseModel):
    """Base message for queue systems"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: MessagePriority = MessagePriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True

class UserCreatedMessage(BaseMessage):
    """Message for user creation events"""
    type: Literal["user.created"] = "user.created"
    user_id: str
    user_email: str
    user_name: str
    
class OrderProcessedMessage(BaseMessage):
    """Message for order processing events"""
    type: Literal["order.processed"] = "order.processed"
    order_id: str
    user_id: str
    total_amount: float
    currency: str = "USD"
    
    @validator('currency')
    def validate_currency(cls, v):
        valid_currencies = ['USD', 'EUR', 'GBP', 'JPY']
        if v not in valid_currencies:
            raise ValueError(f'Invalid currency: {v}')
        return v

class MessageProcessor:
    """Process messages with Pydantic validation"""
    
    MESSAGE_TYPES = {
        "user.created": UserCreatedMessage,
        "order.processed": OrderProcessedMessage,
    }
    
    def process_message(self, raw_message: Union[str, dict]) -> Optional[BaseMessage]:
        """Process raw message into Pydantic model"""
        try:
            # Parse JSON if string
            if isinstance(raw_message, str):
                message_data = json.loads(raw_message)
            else:
                message_data = raw_message
            
            # Determine message type
            message_type = message_data.get('type')
            if not message_type:
                raise ValueError("Message type not specified")
            
            # Get appropriate model class
            model_class = self.MESSAGE_TYPES.get(message_type, BaseMessage)
            
            # Validate and create message
            message = model_class(**message_data)
            
            # Process based on type
            return self._handle_message(message)
            
        except Exception as e:
            print(f"Failed to process message: {e}")
            return None
    
    def _handle_message(self, message: BaseMessage) -> BaseMessage:
        """Handle processed message"""
        if isinstance(message, UserCreatedMessage):
            print(f"Processing user creation: {message.user_name}")
            # Handle user creation logic
            
        elif isinstance(message, OrderProcessedMessage):
            print(f"Processing order: {message.order_id} for ${message.total_amount}")
            # Handle order processing logic
        
        return message

## Testing Strategies

### Comprehensive Testing Patterns

```python
import pytest
from pydantic import ValidationError
from typing import Any, Dict, List
from datetime import datetime, timedelta

class TestModelValidation:
    """Comprehensive test suite for Pydantic models"""
    
    @pytest.fixture
    def valid_user_data(self) -> Dict[str, Any]:
        """Valid user data fixture"""
        return {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30,
            "tags": ["developer", "python"]
        }
    
    @pytest.fixture
    def user_model_class(self):
        """User model fixture"""
        from typing import List
        
        class TestUser(BaseModel):
            name: str = Field(..., min_length=2, max_length=50)
            email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+)
            age: Optional[int] = Field(None, ge=13, le=120)
            tags: List[str] = Field(default_factory=list)
            
            @validator('name')
            def validate_name(cls, v):
                if not v.strip():
                    raise ValueError("Name cannot be empty")
                return v.strip()
        
        return TestUser
    
    def test_valid_data_creation(self, user_model_class, valid_user_data):
        """Test model creation with valid data"""
        user = user_model_class(**valid_user_data)
        
        assert user.name == "John Doe"
        assert user.email == "john@example.com"
        assert user.age == 30
        assert user.tags == ["developer", "python"]
    
    def test_required_field_validation(self, user_model_class):
        """Test validation of required fields"""
        with pytest.raises(ValidationError) as exc_info:
            user_model_class()
        
        errors = exc_info.value.errors()
        required_fields = {error['loc'][0] for error in errors if error['type'] == 'missing'}
        
        assert 'name' in required_fields
        assert 'email' in required_fields
    
    def test_field_constraint_validation(self, user_model_class):
        """Test field constraint validation"""
        # Test string length constraints
        with pytest.raises(ValidationError) as exc_info:
            user_model_class(name="A", email="john@example.com")
        
        assert any("at least 2 characters" in str(error) for error in exc_info.value.errors())
        
        # Test age constraints
        with pytest.raises(ValidationError):
            user_model_class(name="John", email="john@example.com", age=5)  # Too young
        
        with pytest.raises(ValidationError):
            user_model_class(name="John", email="john@example.com", age=200)  # Too old
    
    def test_email_format_validation(self, user_model_class):
        """Test email format validation"""
        invalid_emails = [
            "invalid-email",
            "@domain.com",
            "user@",
            "user.domain.com",
            ""
        ]
        
        for invalid_email in invalid_emails:
            with pytest.raises(ValidationError):
                user_model_class(name="John", email=invalid_email)
    
    def test_custom_validator(self, user_model_class):
        """Test custom validator logic"""
        # Test whitespace handling
        user = user_model_class(
            name="  John Doe  ",  # Extra whitespace
            email="john@example.com"
        )
        assert user.name == "John Doe"  # Should be stripped
        
        # Test empty string after stripping
        with pytest.raises(ValidationError):
            user_model_class(name="   ", email="john@example.com")  # Only whitespace
    
    @pytest.mark.parametrize("test_data,expected_error_count", [
        ({"name": "", "email": "invalid"}, 2),  # Two validation errors
        ({"name": "John"}, 1),  # Missing email
        ({"email": "john@example.com"}, 1),  # Missing name
        ({}, 2),  # Missing both required fields
    ])
    def test_multiple_validation_errors(self, user_model_class, test_data, expected_error_count):
        """Test handling of multiple validation errors"""
        with pytest.raises(ValidationError) as exc_info:
            user_model_class(**test_data)
        
        assert len(exc_info.value.errors()) >= expected_error_count
    
    def test_serialization_deserialization(self, user_model_class, valid_user_data):
        """Test model serialization and deserialization"""
        original_user = user_model_class(**valid_user_data)
        
        # Test dict serialization
        user_dict = original_user.dict()
        recreated_user = user_model_class(**user_dict)
        assert original_user == recreated_user
        
        # Test JSON serialization
        user_json = original_user.json()
        recreated_from_json = user_model_class.parse_raw(user_json)
        assert original_user == recreated_from_json

# Property-based testing with Hypothesis
try:
    from hypothesis import given, strategies as st
    from hypothesis.strategies import text, integers, emails
    
    class TestPropertyBasedValidation:
        """Property-based testing for robust validation"""
        
        @given(
            name=text(min_size=2, max_size=50),
            age=integers(min_value=13, max_value=120),
            email=emails()
        )
        def test_valid_user_properties(self, name, age, email):
            """Property-based test for valid user creation"""
            if name.strip():  # Only test with non-empty names
                user_data = {
                    "name": name,
                    "email": email,
                    "age": age
                }
                
                # This should not raise an exception
                user = ErrorHandlingModel(**user_data)
                assert user.name.strip() == name.strip()
                assert user.age == age
        
        @given(
            name=text(max_size=1),  # Too short
            email=text()  # Not a valid email
        )
        def test_invalid_user_properties(self, name, email):
            """Property-based test for invalid user data"""
            with pytest.raises(ValidationError):
                ErrorHandlingModel(name=name, email=email, age=25, tags=["test"])

except ImportError:
    print("Hypothesis not available - skipping property-based tests")

# Mock and fixture patterns
class TestWithMocks:
    """Test patterns using mocks and fixtures"""
    
    @pytest.fixture
    def mock_external_service(self, monkeypatch):
        """Mock external service calls"""
        def mock_validate_email_exists(email: str) -> bool:
            # Mock implementation
            return email != "nonexistent@example.com"
        
        monkeypatch.setattr(
            "your_module.validate_email_exists", 
            mock_validate_email_exists
        )
    
    def test_with_external_dependency(self, mock_external_service):
        """Test model that depends on external service"""
        # This would test a model that validates email existence
        # using the mocked service
        pass

# Performance testing
class TestModelPerformance:
    """Performance testing for Pydantic models"""
    
    def test_bulk_validation_performance(self, user_model_class):
        """Test performance of bulk model creation"""
        import time
        
        # Generate test data
        test_data = [
            {
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "age": 20 + (i % 50),
                "tags": [f"tag{j}" for j in range(i % 5)]
            }
            for i in range(1000)
        ]
        
        # Measure creation time
        start_time = time.time()
        users = [user_model_class(**data) for data in test_data]
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"Created {len(users)} users in {duration:.3f} seconds")
        
        # Assert reasonable performance (adjust threshold as needed)
        assert duration < 1.0  # Should create 1000 users in less than 1 second
    
    def test_serialization_performance(self, user_model_class, valid_user_data):
        """Test serialization performance"""
        import time
        
        users = [user_model_class(**valid_user_data) for _ in range(1000)]
        
        # Test dict serialization
        start_time = time.time()
        dicts = [user.dict() for user in users]
        dict_time = time.time() - start_time
        
        # Test JSON serialization
        start_time = time.time()
        jsons = [user.json() for user in users]
        json_time = time.time() - start_time
        
        print(f"Dict serialization: {dict_time:.3f}s")
        print(f"JSON serialization: {json_time:.3f}s")
        
        assert dict_time < 0.1
        assert json_time < 0.2

## Migration and Compatibility

### Version Management

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Union
from enum import Enum
import json
from datetime import datetime

class ModelVersion(str, Enum):
    V1_0 = "1.0"
    V1_1 = "1.1"  
    V2_0 = "2.0"

class VersionedModel(BaseModel):
    """Base class for versioned models"""
    
    _version: ModelVersion = Field(ModelVersion.V2_0, alias="version")
    
    @classmethod
    def parse_versioned(cls, data: Union[str, dict]) -> 'VersionedModel':
        """Parse data handling different versions"""
        if isinstance(data, str):
            data = json.loads(data)
        
        version = data.get('version', '1.0')
        
        if version == '1.0':
            return cls._migrate_from_v1_0(data)
        elif version == '1.1':
            return cls._migrate_from_v1_1(data)
        elif version == '2.0':
            return cls(**data)
        else:
            raise ValueError(f"Unsupported version: {version}")
    
    @classmethod
    def _migrate_from_v1_0(cls, data: dict) -> 'VersionedModel':
        """Migration from version 1.0"""
        # Example migration logic
        migrated_data = data.copy()
        
        # Handle field renames
        if 'old_field_name' in migrated_data:
            migrated_data['new_field_name'] = migrated_data.pop('old_field_name')
        
        # Handle structure changes
        if 'metadata' not in migrated_data:
            migrated_data['metadata'] = {}
        
        # Update version
        migrated_data['version'] = ModelVersion.V2_0
        
        return cls(**migrated_data)
    
    @classmethod  
    def _migrate_from_v1_1(cls, data: dict) -> 'VersionedModel':
        """Migration from version 1.1"""
        migrated_data = data.copy()
        migrated_data['version'] = ModelVersion.V2_0
        return cls(**migrated_data)

# Backward compatibility patterns
class UserV1(BaseModel):
    """Legacy user model (v1)"""
    id: int
    name: str
    email: str
    
class UserV2(BaseModel):
    """Current user model (v2) with backward compatibility"""
    id: str  # Changed from int to str
    name: str
    email: str
    full_name: Optional[str] = None  # New field
    metadata: Dict[str, Any] = Field(default_factory=dict)  # New field
    
    @classmethod
    def from_v1(cls, v1_user: UserV1) -> 'UserV2':
        """Convert from V1 model"""
        return cls(
            id=str(v1_user.id),  # Convert int to str
            name=v1_user.name,
            email=v1_user.email,
            full_name=v1_user.name,  # Use name as full_name
            metadata={}  # Default empty metadata
        )
    
    def to_v1_compatible(self) -> Dict[str, Any]:
        """Convert to V1-compatible format"""
        return {
            "id": int(self.id) if self.id.isdigit() else 0,
            "name": self.name,
            "email": self.email
        }

# Schema evolution patterns
class EvolutionAwareModel(BaseModel):
    """Model that handles schema evolution gracefully"""
    
    # Core fields (never remove these)
    id: str
    created_at: datetime
    
    # Evolved fields with defaults for backward compatibility
    status: Optional[str] = "active"  # Added in v1.1
    priority: Optional[int] = 1  # Added in v1.2
    
    # Future fields (prepare for evolution)
    _extra_data: Dict[str, Any] = Field(default_factory=dict, alias="extra")
    
    class Config:
        extra = 'allow'  # Allow extra fields for forward compatibility
        
        # Custom field aliases for backward compatibility  
        fields = {
            '_extra_data': 'extra'
        }
    
    def get_field_safely(self, field_name: str, default: Any = None) -> Any:
        """Safely get field value with fallback"""
        return getattr(self, field_name, self._extra_data.get(field_name, default))
    
    def set_field_safely(self, field_name: str, value: Any):
        """Safely set field value"""
        if hasattr(self, field_name):
            setattr(self, field_name, value)
        else:
            self._extra_data[field_name] = value

## Enterprise Patterns

### Audit and Compliance

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, ClassVar
from datetime import datetime
from enum import Enum
import hashlib
import json
from uuid import uuid4

class AuditAction(str, Enum):
    CREATE = "create"
    UPDATE = "update"  
    DELETE = "delete"
    READ = "read"

class ComplianceLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class AuditLog(BaseModel):
    """Comprehensive audit logging model"""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str
    action: AuditAction
    resource_type: str
    resource_id: str
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    compliance_level: ComplianceLevel = ComplianceLevel.INTERNAL
    
    # Data integrity
    checksum: Optional[str] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for data integrity"""
        data_for_hash = {
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
        }
        
        hash_string = json.dumps(data_for_hash, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify data integrity"""
        expected_checksum = self._calculate_checksum()
        return self.checksum == expected_checksum

class AuditableModel(BaseModel):
    """Base model with built-in auditing capabilities"""
    
    # Audit fields
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    version: int = 1
    
    # Compliance
    compliance_level: ComplianceLevel = ComplianceLevel.INTERNAL
    data_retention_days: Optional[int] = None
    
    # Audit log reference
    _audit_logs: ClassVar[List[AuditLog]] = []
    
    def update_with_audit(self, updated_by: str, **updates) -> 'AuditableModel':
        """Update model with full audit trail"""
        old_values = self.dict()
        
        # Apply updates
        for field, value in updates.items():
            if hasattr(self, field):
                setattr(self, field, value)
        
        # Update audit fields
        self.updated_by = updated_by
        self.updated_at = datetime.utcnow()
        self.version += 1
        
        # Create audit log
        audit_log = AuditLog(
            user_id=updated_by,
            action=AuditAction.UPDATE,
            resource_type=self.__class__.__name__,
            resource_id=str(getattr(self, 'id', 'unknown')),
            old_values=old_values,
            new_values=self.dict(),
            compliance_level=self.compliance_level
        )
        
        self._audit_logs.append(audit_log)
        return self
    
    def get_audit_trail(self) -> List[AuditLog]:
        """Get complete audit trail for this model"""
        resource_id = str(getattr(self, 'id', 'unknown'))
        return [
            log for log in self._audit_logs 
            if log.resource_id == resource_id
        ]

# GDPR Compliance model
class GDPRCompliantModel(BaseModel):
    """Model with GDPR compliance features"""
    
    # Personal data fields
    first_name: Optional[str] = None
    last_name: Optional[str] = None  
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    
    # GDPR metadata
    consent_given: bool = False
    consent_date: Optional[datetime] = None
    consent_version: str = "1.0"
    data_processing_purposes: List[str] = Field(default_factory=list)
    
    # Data subject rights
    anonymized: bool = False
    anonymization_date: Optional[datetime] = None
    deletion_requested: bool = False
    deletion_request_date: Optional[datetime] = None
    
    # Sensitive data classification
    _pii_fields: ClassVar[List[str]] = [
        'first_name', 'last_name', 'email', 'phone', 'address'
    ]
    
    def anonymize_pii(self) -> 'GDPRCompliantModel':
        """Anonymize personally identifiable information"""
        if self.anonymized:
            return self
        
        for field in self._pii_fields:
            if hasattr(self, field) and getattr(self, field):
                # Replace with anonymized placeholder
                setattr(self, field, f"[ANONYMIZED_{field.upper()}]")
        
        self.anonymized = True
        self.anonymization_date = datetime.utcnow()
        return self
    
    def export_personal_data(self) -> Dict[str, Any]:
        """Export personal data for GDPR data portability"""
        personal_data = {}
        
        for field in self._pii_fields:
            value = getattr(self, field, None)
            if value and not self.anonymized:
                personal_data[field] = value
        
        return {
            'personal_data': personal_data,
            'consent_info': {
                'consent_given': self.consent_given,
                'consent_date': self.consent_date.isoformat() if self.consent_date else None,
                'consent_version': self.consent_version,
                'purposes': self.data_processing_purposes
            },
            'export_date': datetime.utcnow().isoformat()
        }
    
    def request_deletion(self) -> bool:
        """Request data deletion (right to be forgotten)"""
        self.deletion_requested = True
        self.deletion_request_date = datetime.utcnow()
        return True
    
    @validator('consent_given')
    def validate_consent_requirements(cls, v, values):
        """Ensure consent is properly documented"""
        if v and not values.get('consent_date'):
            raise ValueError('Consent date must be provided when consent is given')
        
        if v and not values.get('data_processing_purposes'):
            raise ValueError('Data processing purposes must be specified')
        
        return v

# Multi-tenant model
class TenantAwareModel(BaseModel):
    """Model with multi-tenancy support"""
    
    tenant_id: str = Field(..., description="Tenant identifier")
    
    # Tenant isolation validation
    @validator('tenant_id')
    def validate_tenant_access(cls, v):
        """Validate tenant access (would integrate with auth system)"""
        # In practice, this would check against current user's tenant access
        if not v:
            raise ValueError("Tenant ID is required")
        return v
    
    class Config:
        # Ensure tenant_id is always included in serialization
        fields = {
            'tenant_id': {'exclude': False}
        }
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Ensure tenant context is preserved"""
        # Always include tenant_id unless explicitly excluded
        if 'exclude' in kwargs and isinstance(kwargs['exclude'], set):
            kwargs['exclude'].discard('tenant_id')
        elif 'exclude' in kwargs and isinstance(kwargs['exclude'], dict):
            kwargs['exclude'].pop('tenant_id', None)
        
        return super().dict(**kwargs)

## Troubleshooting

### Common Issues and Solutions

```python
from pydantic import BaseModel, ValidationError, Field, validator
from typing import Any, Dict, Optional, List, Union
import traceback
from datetime import datetime

class DiagnosticModel(BaseModel):
    """Model for troubleshooting common Pydantic issues"""
    
    # Issue 1: Field assignment validation not working
    value: int = Field(..., gt=0)
    
    class Config:
        # Solution: Enable validate_assignment
        validate_assignment = True
    
    # Issue 2: Custom validator not being called
    name: str
    
    @validator('name', pre=False, always=False)  # Common mistake: always=False by default
    def validate_name(cls, v):
        # Solution: Use always=True if you want validation even for None values
        if v is None:
            return "Default Name"
        return v

# Common validation pitfalls
class PitfallDemonstrationModel(BaseModel):
    """Demonstrates common validation pitfalls and solutions"""
    
    # Pitfall 1: Mutable default arguments
    # WRONG:
    # items: List[str] = []  # This creates a shared list!
    
    # CORRECT:
    items: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Pitfall 2: Validator order dependencies
    raw_value: str
    processed_value: Optional[str] = None
    
    @validator('raw_value')
    def clean_raw_value(cls, v):
        return v.strip() if v else v
    
    @validator('processed_value', pre=False, always=True)
    def process_value(cls, v, values):
        # This validator depends on raw_value being processed first
        raw = values.get('raw_value', '')
        return f"processed_{raw}" if raw else None
    
    # Pitfall 3: Circular validation dependencies
    field_a: Optional[str] = None
    field_b: Optional[str] = None
    
    @validator('field_a')
    def validate_field_a(cls, v, values):
        # Be careful with circular dependencies
        if not v and not values.get('field_b'):
            raise ValueError("Either field_a or field_b must be provided")
        return v

def debug_validation_error(model_class: type, data: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive validation error debugging"""
    debug_info = {
        'input_data': data,
        'model_class': model_class.__name__,
        'validation_errors': [],
        'field_info': {},
        'suggestions': []
    }
    
    # Get field information
    for field_name, field in model_class.__fields__.items():
        debug_info['field_info'][field_name] = {
            'type': str(field.type_),
            'required': field.required,
            'default': field.default,
            'validators': [v.__name__ for v in field.validators] if field.validators else []
        }
    
    try:
        instance = model_class(**data)
        debug_info['success'] = True
        debug_info['created_instance'] = instance.dict()
        
    except ValidationError as e:
        debug_info['success'] = False
        
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            error_info = {
                'field': field_path,
                'error_type': error['type'],
                'message': error['msg'],
                'input_value': error.get('input'),
                'context': error.get('ctx', {})
            }
            
            debug_info['validation_errors'].append(error_info)
            
            # Generate suggestions based on error type
            if error['type'] == 'missing':
                debug_info['suggestions'].append(
                    f"Add required field '{field_path}' to your input data"
                )
            elif error['type'] == 'value_error.number.not_gt':
                min_val = error.get('ctx', {}).get('limit_value', 0)
                debug_info['suggestions'].append(
                    f"Field '{field_path}' must be greater than {min_val}"
                )
            elif error['type'] == 'type_error':
                expected_type = debug_info['field_info'].get(field_path, {}).get('type', 'unknown')
                debug_info['suggestions'].append(
                    f"Field '{field_path}' expects type {expected_type}"
                )
    
    except Exception as e:
        debug_info['success'] = False
        debug_info['unexpected_error'] = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    return debug_info

# Performance troubleshooting
class PerformanceTroubleshootingModel(BaseModel):
    """Model for identifying performance issues"""
    
    data: List[Dict[str, Any]] = []
    
    # Performance issue: Expensive validation on large lists
    @validator('data')
    def validate_data_expensive(cls, v):
        """This validator might be slow for large lists"""
        start_time = datetime.now()
        
        # Simulate expensive validation
        for item in v:
            # Expensive operation per item
            pass
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if duration > 1.0:  # Log slow validations
            print(f"Slow validation detected: {duration:.2f} seconds for {len(v)} items")
        
        return v
    
    # Better approach: Sample validation for large datasets
    @validator('data', pre=True)
    def validate_data_sample(cls, v):
        """Validate a sample of large datasets for better performance"""
        if isinstance(v, list) and len(v) > 1000:
            # Validate only a sample for very large datasets
            sample_size = min(100, len(v))
            sample = v[:sample_size]
            
            # Perform expensive validation on sample only
            for item in sample:
                if not isinstance(item, dict):
                    raise ValueError("All items must be dictionaries")
        
        return v

# Memory usage troubleshooting
def analyze_model_memory_usage(model_class: type, sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze memory usage patterns of a model"""
    import sys
    import gc
    
    # Create baseline
    gc.collect()
    baseline_objects = len(gc.get_objects())
    
    # Create model instances
    instances = []
    for i in range(100):
        instance_data = sample_data.copy()
        instance_data['id'] = i  # Vary the data
        instances.append(model_class(**instance_data))
    
    # Measure memory usage
    gc.collect()
    after_objects = len(gc.get_objects())
    
    # Calculate size of a single instance
    single_instance_size = sys.getsizeof(instances[0])
    
    analysis = {
        'objects_created': after_objects - baseline_objects,
        'instances_created': len(instances),
        'single_instance_size_bytes': single_instance_size,
        'estimated_total_size_bytes': single_instance_size * len(instances),
        'objects_per_instance': (after_objects - baseline_objects) / len(instances),
        'recommendations': []
    }
    
    # Generate recommendations
    if analysis['objects_per_instance'] > 10:
        analysis['recommendations'].append(
            "Consider using __slots__ to reduce memory overhead"
        )
    
    if single_instance_size > 1000:
        analysis['recommendations'].append(
            "Instance size is large - consider splitting into smaller models"
        )
    
    return analysis

# Debugging utilities
class ModelDebugger:
    """Utility class for debugging Pydantic models"""
    
    @staticmethod
    def trace_validation(model_class: type, data: Dict[str, Any]) -> List[str]:
        """Trace validation process step by step"""
        trace = []
        
        try:
            trace.append(f"Starting validation for {model_class.__name__}")
            trace.append(f"Input data keys: {list(data.keys())}")
            
            # Check required fields
            required_fields = [
                name for name, field in model_class.__fields__.items()
                if field.required
            ]
            trace.append(f"Required fields: {required_fields}")
            
            missing_required = [
                field for field in required_fields
                if field not in data
            ]
            if missing_required:
                trace.append(f"Missing required fields: {missing_required}")
            
            # Attempt validation
            instance = model_class(**data)
            trace.append("Validation successful")
            trace.append(f"Created instance: {instance}")
            
        except ValidationError as e:
            trace.append(f"Validation failed with {len(e.errors())} errors:")
            for error in e.errors():
                trace.append(f"  - {error}")
        
        except Exception as e:
            trace.append(f"Unexpected error: {e}")
        
        return trace
    
    @staticmethod
    def compare_models(model1: BaseModel, model2: BaseModel) -> Dict[str, Any]:
        """Compare two model instances and highlight differences"""
        dict1 = model1.dict()
        dict2 = model2.dict()
        
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        comparison = {
            'identical': True,
            'differences': {},
            'only_in_model1': [],
            'only_in_model2': []
        }
        
        for key in all_keys:
            if key only in dict1:
                comparison['only_in_model1'].append(key)
                comparison['identical'] = False
            elif key only in dict2:
                comparison['only_in_model2'].append(key)
                comparison['identical'] = False
            elif dict1[key] != dict2[key]:
                comparison['differences'][key] = {
                    'model1': dict1[key],
                    'model2': dict2[key]
                }
                comparison['identical'] = False
        
        return comparison

---

## Complete Example Application

```python
"""
Complete example: User Management System with Pydantic
Demonstrates real-world usage patterns and best practices
"""

from pydantic import BaseModel, BaseSettings, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
import asyncio
import json

# Enums
class UserRole(str, Enum):
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"

class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

# Configuration
class AppSettings(BaseSettings):
    app_name: str = "User Management System"
    debug: bool = False
    max_users: int = 10000
    token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"

# Base models
class TimestampedModel(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

# Core models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=30, regex=r'^[a-zA-Z0-9_]+)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+)
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str = Field(..., min_length=1, max_length=100)
    role: UserRole = UserRole.USER
    
    @validator('email')
    def validate_email_domain(cls, v):
        """Validate email domain"""
        allowed_domains = ['gmail.com', 'example.com', 'company.com']
        domain = v.split('@')[1]
        if domain not in allowed_domains:
            raise ValueError(f'Email domain must be one of: {allowed_domains}')
        return v.lower()
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password complexity"""
        import re
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain special character')
        return v

class User(TimestampedModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    username: str
    email: str
    password_hash: str = Field(..., exclude=True)  # Never serialize password
    full_name: str
    role: UserRole
    status: UserStatus = UserStatus.ACTIVE
    last_login: Optional[datetime] = None
    login_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_create_request(cls, user_create: UserCreate, password_hash: str) -> 'User':
        """Create User from UserCreate request"""
        return cls(
            username=user_create.username,
            email=user_create.email,
            password_hash=password_hash,
            full_name=user_create.full_name,
            role=user_create.role
        )
    
    def update_login(self):
        """Update login statistics"""
        self.last_login = datetime.utcnow()
        self.login_count += 1
        self.updated_at = datetime.utcnow()
    
    def to_public(self) -> 'UserPublic':
        """Convert to public representation"""
        return UserPublic(
            id=self.id,
            username=self.username,
            full_name=self.full_name,
            role=self.role,
            status=self.status,
            created_at=self.created_at,
            last_login=self.last_login
        )

class UserPublic(BaseModel):
    """Public user representation (safe for API responses)"""
    id: str
    username: str
    full_name: str
    role: UserRole
    status: UserStatus
    created_at: datetime
    last_login: Optional[datetime]

class UserUpdate(BaseModel):
    """User update request"""
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[str] = Field(None, regex=r'^[\w\.-]+@[\w\.-]+\.\w+)
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('email')
    def validate_email_if_provided(cls, v):
        if v is not None:
            return v.lower()
        return v

# API Models
class APIResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class UserListResponse(APIResponse):
    """User list API response"""
    data: Optional[List[UserPublic]] = None
    total: int = 0
    page: int = 1
    page_size: int = 10

# Service layer
class UserService:
    """User management service with Pydantic models"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.users: Dict[str, User] = {}  # In-memory storage for demo
    
    async def create_user(self, user_create: UserCreate) -> Union[User, List[str]]:
        """Create a new user with validation"""
        errors = []
        
        # Check if user already exists
        if any(u.username == user_create.username for u in self.users.values()):
            errors.append("Username already exists")
        
        if any(u.email == user_create.email for u in self.users.values()):
            errors.append("Email already exists")
        
        # Check user limit
        if len(self.users) >= self.settings.max_users:
            errors.append(f"Maximum user limit ({self.settings.max_users}) reached")
        
        if errors:
            return errors
        
        # Hash password (simplified for demo)
        password_hash = f"hashed_{user_create.password}"
        
        # Create user
        user = User.from_create_request(user_create, password_hash)
        self.users[user.id] = user
        
        return user
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    async def update_user(self, user_id: str, user_update: UserUpdate) -> Union[User, List[str]]:
        """Update user with validation"""
        user = self.users.get(user_id)
        if not user:
            return ["User not found"]
        
        errors = []
        
        # Check email uniqueness if being updated
        if user_update.email and user_update.email != user.email:
            if any(u.email == user_update.email for u in self.users.values() if u.id != user_id):
                errors.append("Email already exists")
        
        if errors:
            return errors
        
        # Apply updates
        update_data = user_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        return user
    
    async def list_users(self, page: int = 1, page_size: int = 10, role: Optional[UserRole] = None) -> UserListResponse:
        """List users with pagination and filtering"""
        users_list = list(self.users.values())
        
        # Filter by role if specified
        if role:
            users_list = [u for u in users_list if u.role == role]
        
        total = len(users_list)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        users_page = users_list[start_idx:end_idx]
        public_users = [user.to_public() for user in users_page]
        
        return UserListResponse(
            success=True,
            message="Users retrieved successfully",
            data=public_users,
            total=total,
            page=page,
            page_size=page_size
        )
    
    async def authenticate_user(self, username: str, password: str) -> Union[User, str]:
        """Authenticate user and return user or error message"""
        user = next((u for u in self.users.values() if u.username == username), None)
        
        if not user:
            return "Invalid username or password"
        
        if user.status != UserStatus.ACTIVE:
            return f"Account is {user.status.value}"
        
        # Verify password (simplified)
        expected_hash = f"hashed_{password}"
        if user.password_hash != expected_hash:
            return "Invalid username or password"
        
        # Update login info
        user.update_login()
        
        return user

# Main application example
async def main():
    """Example usage of the user management system"""
    
    # Initialize settings and service
    settings = AppSettings()
    user_service = UserService(settings)
    
    print(f"=== {settings.app_name} Demo ===\n")
    
    # Create test users
    test_users = [
        {
            "username": "admin_user",
            "email": "admin@company.com",
            "password": "AdminPass123!",
            "full_name": "System Administrator",
            "role": UserRole.ADMIN
        },
        {
            "username": "john_doe",
            "email": "john@gmail.com", 
            "password": "SecurePass456!",
            "full_name": "John Doe",
            "role": UserRole.USER
        },
        {
            "username": "jane_mod",
            "email": "jane@example.com",
            "password": "ModPass789!",
            "full_name": "Jane Smith",
            "role": UserRole.MODERATOR
        }
    ]
    
    created_users = []
    
    # Create users
    print("Creating users:")
    for user_data in test_users:
        try:
            user_create = UserCreate(**user_data)
            result = await user_service.create_user(user_create)
            
            if isinstance(result, User):
                created_users.append(result)
                print(f"✓ Created user: {result.username} ({result.full_name})")
            else:
                print(f"✗ Failed to create {user_data['username']}: {result}")
                
        except ValidationError as e:
            print(f"✗ Validation failed for {user_data['username']}:")
            for error in e.errors():
                print(f"  - {error['loc'][-1]}: {error['msg']}")
    
    print(f"\nCreated {len(created_users)} users successfully.\n")
    
    # List all users
    user_list = await user_service.list_users(page_size=10)
    print("All users:")
    for user in user_list.data:
        print(f"- {user.username}: {user.full_name} ({user.role.value})")
    print()
    
    # Test authentication
    print("Testing authentication:")
    auth_result = await user_service.authenticate_user("john_doe", "SecurePass456!")
    if isinstance(auth_result, User):
        print(f"✓ Authentication successful for {auth_result.username}")
        print(f"  Login count: {auth_result.login_count}")
        print(f"  Last login: {auth_result.last_login}")
    else:
        print(f"✗ Authentication failed: {auth_result}")
    
    # Test user update
    print("\nTesting user update:")
    if created_users:
        user_to_update = created_users[0]
        update_data = UserUpdate(
            full_name="Updated Administrator Name",
            metadata={"updated": True, "version": 2}
        )
        
        update_result = await user_service.update_user(user_to_update.id, update_data)
        if isinstance(update_result, User):
            print(f"✓ Updated user {update_result.username}")
            print(f"  New name: {update_result.full_name}")
            print(f"  Metadata: {update_result.metadata}")
        else:
            print(f"✗ Update failed: {update_result}")
    
    # Demonstrate error handling
    print("\nTesting validation errors:")
    try:
        invalid_user = UserCreate(
            username="ab",  # Too short
            email="invalid-email",  # Invalid format
            password="weak",  # Too weak
            full_name="",  # Empty
            role=UserRole.USER
        )
    except ValidationError as e:
        print("Expected validation errors:")
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error['loc'])
            print(f"  - {field}: {error['msg']}")
    
    # Test filtering
    print(f"\nFiltering users by role (ADMIN):")
    admin_users = await user_service.list_users(role=UserRole.ADMIN)
    for user in admin_users.data:
        print(f"- {user.username}: {user.full_name}")
    
    print(f"\n=== Demo completed ===")
    print(f"Total users created: {len(user_service.users)}")
    print(f"Settings: {settings.dict()}")

if __name__ == "__main__":
    asyncio.run(main())