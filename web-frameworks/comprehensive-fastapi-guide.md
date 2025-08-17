# Comprehensive FastAPI Development Guide

## Table of Contents

1. [Introduction to FastAPI](#chapter-1-introduction-to-fastapi)
2. [Environment Setup and Installation](#chapter-2-environment-setup-and-installation)
3. [FastAPI Fundamentals](#chapter-3-fastapi-fundamentals)
4. [Path Operations and Request Handling](#chapter-4-path-operations-and-request-handling)
5. [Data Validation and Pydantic Models](#chapter-5-data-validation-and-pydantic-models)
6. [Response Models and Data Serialization](#chapter-6-response-models-and-data-serialization)
7. [Database Integration and Async Operations](#chapter-7-database-integration-and-async-operations)
8. [Authentication and Security](#chapter-8-authentication-and-security)
9. [File Handling and Background Tasks](#chapter-9-file-handling-and-background-tasks)
10. [WebSocket and Real-time Communication](#chapter-10-websocket-and-real-time-communication)
11. [Testing FastAPI Applications](#chapter-11-testing-fastapi-applications)
12. [Documentation and OpenAPI](#chapter-12-documentation-and-openapi)
13. [Deployment and Production](#chapter-13-deployment-and-production)
14. [Advanced Features and Patterns](#chapter-14-advanced-features-and-patterns)
15. [Real-world FastAPI Project](#chapter-15-real-world-fastapi-project)

## Chapter 1: Introduction to FastAPI

FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.

### 1.1 What is FastAPI?

FastAPI is designed to be:
- **Fast**: Very high performance, on par with NodeJS and Go
- **Fast to code**: Increase development speed by about 200% to 300%
- **Fewer bugs**: Reduce about 40% of human (developer) induced errors
- **Intuitive**: Great editor support with completion everywhere
- **Easy**: Designed to be easy to use and learn
- **Short**: Minimize code duplication
- **Robust**: Get production-ready code with automatic interactive documentation
- **Standards-based**: Based on OpenAPI and JSON Schema

### 1.2 Key Features

- **Automatic API documentation**: Interactive API documentation with Swagger UI and ReDoc
- **Type hints**: Full Python type hints support with validation
- **Async support**: Native async/await support for high performance
- **Dependency injection**: Powerful dependency injection system
- **Security utilities**: OAuth2, JWT tokens, API keys, and more
- **WebSocket support**: Built-in WebSocket support
- **GraphQL support**: GraphQL integration available
- **Standards compliance**: Based on OpenAPI (formerly Swagger) and JSON Schema

### 1.3 FastAPI vs Other Frameworks

Comparison with other Python web frameworks:

| Feature | FastAPI | Flask | Django | Django REST |
|---------|---------|-------|--------|-------------|
| Performance | Very High | Medium | Medium | Medium |
| Async Support | Native | Via extensions | Limited | Limited |
| Type Hints | Full support | None | None | None |
| Auto Documentation | Yes | No | No | Yes |
| Validation | Automatic | Manual | Manual | Some |
| Learning Curve | Easy | Easy | Steep | Medium |

### 1.4 When to Use FastAPI

FastAPI is ideal for:
- **API-first applications**: Building REST APIs and microservices
- **High-performance requirements**: Applications needing high throughput
- **Modern Python development**: Projects using Python 3.7+ with type hints
- **Rapid prototyping**: Quick API development with automatic documentation
- **Data validation**: Applications requiring strict data validation
- **Async applications**: I/O intensive applications benefiting from async operations

## Chapter 2: Environment Setup and Installation

Setting up your development environment for FastAPI development.

### 2.1 Prerequisites

Requirements:
- Python 3.7 or higher
- pip (Python package manager)
- Virtual environment (recommended)
- Code editor with Python support (VS Code, PyCharm, etc.)

### 2.2 Virtual Environment Setup

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv fastapi-env

# Activate virtual environment
# On Windows
fastapi-env\Scripts\activate

# On macOS/Linux
source fastapi-env/bin/activate
```

### 2.3 Installing FastAPI

Install FastAPI and ASGI server:

```bash
# Install FastAPI
pip install fastapi

# Install ASGI server (Uvicorn)
pip install "uvicorn[standard]"

# Or install both together
pip install "fastapi[all]"
```

The `fastapi[all]` installation includes:
- `uvicorn` - ASGI server
- `pydantic[email]` - Email validation
- `python-multipart` - Form parsing
- `jinja2` - Template support
- `python-jose[cryptography]` - JWT tokens
- `passlib[bcrypt]` - Password hashing
- `aiofiles` - Async file operations

### 2.4 Development Dependencies

Additional development tools:

```bash
# Testing
pip install pytest pytest-asyncio httpx

# Code formatting and linting
pip install black isort flake8 mypy

# Database (optional)
pip install sqlalchemy alembic asyncpg  # PostgreSQL
pip install aiosqlite  # SQLite

# Additional utilities
pip install python-dotenv  # Environment variables
pip install email-validator  # Email validation
pip install pillow  # Image processing
```

### 2.5 Project Structure

Recommended project structure:

```
fastapi_project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   └── users.py
│   ├── dependencies.py
│   ├── database.py
│   └── config.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

### 2.6 Basic FastAPI Application

Create your first FastAPI app (`app/main.py`):

```python
from fastapi import FastAPI

# Create FastAPI instance
app = FastAPI(
    title="My FastAPI App",
    description="A comprehensive FastAPI application",
    version="1.0.0"
)

# Basic route
@app.get("/")
async def root():
    """Root endpoint returning welcome message"""
    return {"message": "Welcome to FastAPI!"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

### 2.7 Running the Application

Start the development server:

```bash
# Using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python
python app/main.py
```

Access your application:
- API: http://localhost:8000
- Interactive docs (Swagger UI): http://localhost:8000/docs
- Alternative docs (ReDoc): http://localhost:8000/redoc
- OpenAPI schema: http://localhost:8000/openapi.json

### 2.8 Configuration Management

Environment configuration (`app/config.py`):

```python
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application settings
    app_name: str = "FastAPI Application"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database settings
    database_url: Optional[str] = None
    
    # Security settings
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # API settings
    api_v1_prefix: str = "/api/v1"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()
```

Environment file (`.env`):

```env
# Application
APP_NAME=My FastAPI Application
DEBUG=True

# Database
DATABASE_URL=sqlite:///./app.db

# Security
SECRET_KEY=your-super-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Server
HOST=127.0.0.1
PORT=8000
```

## Chapter 3: FastAPI Fundamentals

Understanding the core concepts and building blocks of FastAPI.

### 3.1 FastAPI Instance

Creating and configuring the FastAPI application:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI instance with metadata
app = FastAPI(
    title="FastAPI Application",
    description="A comprehensive API built with FastAPI",
    version="1.0.0",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "API Support",
        "url": "http://www.example.com/contact/",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    # OpenAPI URL (set to None to disable)
    openapi_url="/api/v1/openapi.json",
    # Documentation URLs
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3.2 Path Operations

FastAPI supports all HTTP methods:

```python
from fastapi import FastAPI, HTTPException
from typing import Optional

app = FastAPI()

# GET operation
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    """Get item by ID with optional query parameter"""
    if item_id < 1:
        raise HTTPException(status_code=400, detail="Item ID must be positive")
    
    result = {"item_id": item_id}
    if q:
        result.update({"q": q})
    return result

# POST operation
@app.post("/items/")
async def create_item(item: dict):
    """Create a new item"""
    return {"message": "Item created", "item": item}

# PUT operation
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: dict):
    """Update an existing item"""
    return {"item_id": item_id, "item": item}

# DELETE operation
@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    """Delete an item"""
    return {"message": f"Item {item_id} deleted"}

# PATCH operation
@app.patch("/items/{item_id}")
async def patch_item(item_id: int, item: dict):
    """Partially update an item"""
    return {"item_id": item_id, "updated_fields": item}
```

### 3.3 Path Parameters

Working with path parameters:

```python
from fastapi import FastAPI, Path
from typing import Annotated

app = FastAPI()

# Simple path parameter
@app.get("/users/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}

# Path parameter with validation
@app.get("/items/{item_id}")
async def read_item(
    item_id: Annotated[int, Path(title="The ID of the item", ge=1, le=1000)]
):
    return {"item_id": item_id}

# Multiple path parameters
@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(user_id: int, item_id: str):
    return {"user_id": user_id, "item_id": item_id}

# Path parameter with predefined values (Enum)
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}
    
    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}
    
    return {"model_name": model_name, "message": "Have some residuals"}
```

### 3.4 Query Parameters

Handling query parameters:

```python
from fastapi import FastAPI, Query
from typing import Optional, List, Annotated

app = FastAPI()

# Basic query parameters
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# Optional query parameter
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Optional[str] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}

# Query parameter with validation
@app.get("/items/search")
async def search_items(
    q: Annotated[
        Optional[str], 
        Query(
            title="Query string",
            description="Query string for the items to search",
            min_length=3,
            max_length=50,
            regex="^[a-zA-Z0-9 ]+$"
        )
    ] = None
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results

# Multiple query parameters of same name
@app.get("/items/")
async def read_items(
    q: Annotated[Optional[List[str]], Query()] = None
):
    query_items = {"q": q}
    return query_items

# Required query parameter
@app.get("/items/required")
async def read_items_required(q: Annotated[str, Query(min_length=3)]):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    results.update({"q": q})
    return results
```

## Chapter 4: Path Operations and Request Handling

Advanced path operations and request handling techniques.

### 4.1 Request Body

Handling request bodies with Pydantic models:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

app = FastAPI()

# Pydantic model for request body
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    tags: list[str] = []

class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None

# POST with request body
@app.post("/items/")
async def create_item(item: Item):
    """Create item with validated request body"""
    # Access item attributes
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict

# PUT with request body
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item, q: Optional[str] = None):
    """Update item combining path param, query param, and request body"""
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result

# Multiple request body parameters
@app.post("/users/{user_id}/items/")
async def create_item_for_user(user_id: int, item: Item, user: User):
    """Create item for user with multiple request body models"""
    return {
        "user_id": user_id,
        "item": item,
        "user": user
    }
```

### 4.2 Form Data

Handling HTML form data:

```python
from fastapi import FastAPI, Form, File, UploadFile
from typing import Annotated

app = FastAPI()

# Simple form data
@app.post("/login/")
async def login(
    username: Annotated[str, Form()],
    password: Annotated[str, Form()]
):
    """Handle login form"""
    return {"username": username}

# Form with file upload
@app.post("/submit-form/")
async def submit_form(
    name: Annotated[str, Form()],
    email: Annotated[str, Form()],
    message: Annotated[str, Form()],
    file: Annotated[UploadFile, File()]
):
    """Handle form submission with file upload"""
    return {
        "name": name,
        "email": email,
        "message": message,
        "filename": file.filename,
        "content_type": file.content_type
    }
```

### 4.3 Headers and Cookies

Working with HTTP headers and cookies:

```python
from fastapi import FastAPI, Header, Cookie, Response
from typing import Annotated, Optional

app = FastAPI()

# Reading headers
@app.get("/headers/")
async def read_headers(
    user_agent: Annotated[Optional[str], Header()] = None,
    accept_language: Annotated[Optional[str], Header()] = None,
    authorization: Annotated[Optional[str], Header()] = None
):
    """Read HTTP headers"""
    return {
        "User-Agent": user_agent,
        "Accept-Language": accept_language,
        "Authorization": authorization
    }

# Reading cookies
@app.get("/cookies/")
async def read_cookies(
    session_id: Annotated[Optional[str], Cookie()] = None,
    tracking_id: Annotated[Optional[str], Cookie()] = None
):
    """Read cookies"""
    return {
        "session_id": session_id,
        "tracking_id": tracking_id
    }

# Setting cookies
@app.post("/set-cookie/")
async def set_cookie(response: Response):
    """Set cookies in response"""
    response.set_cookie(
        key="session_id", 
        value="abc123", 
        max_age=3600,
        httponly=True
    )
    return {"message": "Cookie set"}
```

### 4.4 Response Models

Defining response models and status codes:

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List

app = FastAPI()

class ItemResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float

class ItemCreate(BaseModel):
    name: str
    description: Optional[str] = None
    price: float

# Response model
@app.post("/items/", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
async def create_item(item: ItemCreate):
    """Create item with response model"""
    # Simulate item creation with ID
    created_item = ItemResponse(
        id=1,
        name=item.name,
        description=item.description,
        price=item.price
    )
    return created_item

# List response model
@app.get("/items/", response_model=List[ItemResponse])
async def read_items():
    """Get items list"""
    return [
        ItemResponse(id=1, name="Item 1", price=10.5),
        ItemResponse(id=2, name="Item 2", price=20.5)
    ]

# Custom status codes
@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: int):
    """Delete item"""
    return  # Empty response with 204 status
```

## Chapter 5: Data Validation and Pydantic Models

Comprehensive data validation using Pydantic models.

### 5.1 Basic Pydantic Models

Creating and using Pydantic models:

```python
from pydantic import BaseModel, validator, Field, EmailStr
from typing import Optional, List
from datetime import datetime
from enum import Enum

# Basic model
class User(BaseModel):
    id: Optional[int] = None
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None
    age: int = Field(..., ge=0, le=120)
    is_active: bool = True
    created_at: Optional[datetime] = None

# Model with enum
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"

class UserWithRole(User):
    role: UserRole = UserRole.USER

# Nested models
class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str = Field(..., regex=r'^\d{5}(-\d{4})?$')

class UserWithAddress(User):
    address: Optional[Address] = None

# Usage
@app.post("/users/", response_model=UserWithRole)
async def create_user(user: UserWithAddress):
    """Create user with validation"""
    user.created_at = datetime.now()
    return user
```

### 5.2 Advanced Validation

Custom validators and advanced validation:

```python
from pydantic import BaseModel, validator, root_validator
from typing import List
import re

class UserRegistration(BaseModel):
    username: str
    email: str
    password: str
    password_confirm: str
    tags: List[str] = []

    @validator('username')
    def username_alphanumeric(cls, v):
        """Validate username is alphanumeric"""
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username must be alphanumeric')
        return v

    @validator('email')
    def validate_email(cls, v):
        """Custom email validation"""
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()

    @validator('password')
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain digit')
        return v

    @root_validator
    def validate_passwords_match(cls, values):
        """Validate passwords match"""
        password = values.get('password')
        password_confirm = values.get('password_confirm')
        if password != password_confirm:
            raise ValueError('Passwords do not match')
        return values

    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags"""
        if len(v) > 5:
            raise ValueError('Maximum 5 tags allowed')
        return [tag.lower().strip() for tag in v]
```

### 5.3 Model Configuration

Configuring Pydantic model behavior:

```python
from pydantic import BaseModel, Field
from datetime import datetime

class ConfiguredModel(BaseModel):
    name: str
    value: float
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        # Allow population by field name and alias
        allow_population_by_field_name = True
        
        # Validate assignment
        validate_assignment = True
        
        # Use enum values instead of enum names
        use_enum_values = True
        
        # JSON encoders for custom types
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        
        # Schema customization
        schema_extra = {
            "example": {
                "name": "Example Item",
                "value": 42.0,
                "created_at": "2024-01-01T12:00:00"
            }
        }

# Model inheritance
class BaseItem(BaseModel):
    name: str
    description: Optional[str] = None
    
    class Config:
        orm_mode = True  # Enable ORM mode for SQLAlchemy

class Item(BaseItem):
    price: float
    category_id: int

class ItemUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
```

## Chapter 6: Response Models and Data Serialization

Advanced response handling and data serialization.

### 6.1 Response Models

Creating flexible response models:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Union
from datetime import datetime

app = FastAPI()

# Base response models
class BaseResponse(BaseModel):
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseResponse):
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[dict] = None

class SuccessResponse(BaseResponse):
    success: bool = True
    data: Optional[dict] = None

# Specific response models
class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    created_at: datetime

class UserListResponse(BaseModel):
    users: List[UserResponse]
    total: int
    page: int
    per_page: int

# Union response types
@app.get("/users/{user_id}", response_model=Union[UserResponse, ErrorResponse])
async def get_user(user_id: int):
    """Get user with flexible response"""
    if user_id < 1:
        return ErrorResponse(
            message="Invalid user ID",
            error_code="INVALID_ID",
            details={"provided_id": user_id}
        )
    
    # Simulate user retrieval
    user = UserResponse(
        id=user_id,
        username=f"user{user_id}",
        email=f"user{user_id}@example.com",
        is_active=True,
        created_at=datetime.now()
    )
    return user
```

### 6.2 Response Model Features

Advanced response model features:

```python
from pydantic import BaseModel
from typing import Optional

# Response model with field exclusion
class UserFull(BaseModel):
    id: int
    username: str
    email: str
    password_hash: str  # Sensitive field
    is_active: bool

class UserPublic(BaseModel):
    id: int
    username: str
    is_active: bool

# Multiple response models
@app.get("/users/{user_id}/public", response_model=UserPublic)
async def get_user_public(user_id: int):
    """Get user public info only"""
    user_full = UserFull(
        id=user_id,
        username=f"user{user_id}",
        email=f"user{user_id}@example.com",
        password_hash="secret_hash",
        is_active=True
    )
    return user_full  # FastAPI will exclude non-public fields

# Response model with include/exclude
@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: int, include_email: bool = False):
    """Get user profile with conditional fields"""
    user = {
        "id": user_id,
        "username": f"user{user_id}",
        "email": f"user{user_id}@example.com",
        "full_name": f"User {user_id}",
        "bio": "User biography"
    }
    
    if not include_email:
        user.pop("email", None)
    
    return user
```

## Chapter 7: Database Integration and Async Operations

Integrating databases with FastAPI and leveraging async operations.

### 7.1 Database Setup

Setting up database with SQLAlchemy and Alembic:

```python
# app/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import databases
from app.config import settings

# Async database setup
DATABASE_URL = settings.database_url
database = databases.Database(DATABASE_URL)

# SQLAlchemy setup
Base = declarative_base()

# Sync engine (for migrations)
engine = create_engine(DATABASE_URL.replace("asyncpg://", "postgresql://"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Async engine
async_engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

# Dependency
async def get_database():
    async with database.transaction():
        yield database

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
```

### 7.2 Database Models

Creating SQLAlchemy models:

```python
# app/models/user.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    posts = relationship("Post", back_populates="author")

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=False)
    content = Column(Text, nullable=False)
    published = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Foreign key
    author_id = Column(Integer, ForeignKey("users.id"))

    # Relationships
    author = relationship("User", back_populates="posts")
```

### 7.3 Database Operations

CRUD operations with async/await:

```python
# app/crud/user.py
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.future import select as future_select
from typing import List, Optional
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.core.security import get_password_hash

class UserCRUD:
    async def get_user(self, db: AsyncSession, user_id: int) -> Optional[User]:
        """Get user by ID"""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_user_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        """Get user by email"""
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_users(
        self, db: AsyncSession, skip: int = 0, limit: int = 100
    ) -> List[User]:
        """Get users list"""
        result = await db.execute(
            select(User).offset(skip).limit(limit).order_by(User.created_at.desc())
        )
        return result.scalars().all()

    async def create_user(self, db: AsyncSession, user: UserCreate) -> User:
        """Create new user"""
        hashed_password = get_password_hash(user.password)
        db_user = User(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            hashed_password=hashed_password,
        )
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        return db_user

    async def update_user(
        self, db: AsyncSession, user_id: int, user_update: UserUpdate
    ) -> Optional[User]:
        """Update user"""
        result = await db.execute(select(User).where(User.id == user_id))
        db_user = result.scalar_one_or_none()
        
        if db_user:
            update_data = user_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_user, field, value)
            
            await db.commit()
            await db.refresh(db_user)
        
        return db_user

    async def delete_user(self, db: AsyncSession, user_id: int) -> bool:
        """Delete user"""
        result = await db.execute(select(User).where(User.id == user_id))
        db_user = result.scalar_one_or_none()
        
        if db_user:
            await db.delete(db_user)
            await db.commit()
            return True
        return False

user_crud = UserCRUD()
```

### 7.4 API Endpoints with Database

Creating API endpoints with database operations:

```python
# app/routers/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.database import get_db
from app.crud.user import user_crud
from app.schemas.user import User, UserCreate, UserUpdate
from app.core.security import get_current_user

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(
    user: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create new user"""
    # Check if user exists
    existing_user = await user_crud.get_user_by_email(db, email=user.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    return await user_crud.create_user(db, user=user)

@router.get("/", response_model=List[User])
async def read_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """Get users list"""
    users = await user_crud.get_users(db, skip=skip, limit=limit)
    return users

@router.get("/{user_id}", response_model=User)
async def read_user(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get user by ID"""
    user = await user_crud.get_user(db, user_id=user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@router.put("/{user_id}", response_model=User)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update user"""
    if current_user.id != user_id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    user = await user_crud.update_user(db, user_id=user_id, user_update=user_update)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete user"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    success = await user_crud.delete_user(db, user_id=user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
```

## Chapter 8: Authentication and Security

Implementing comprehensive authentication and security in FastAPI applications.

### 8.1 Password Security

Secure password hashing and verification:

```python
# app/core/security.py
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from app.config import settings

# Password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        if username is None:
            return None
        return {"username": username}
    except JWTError:
        return None
```

### 8.2 Authentication Dependencies

Creating authentication dependencies:

```python
# app/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from app.database import get_db
from app.models.user import User
from app.crud.user import user_crud
from app.core.security import verify_token

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = verify_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception
    
    user = await user_crud.get_user_by_username(db, username=token_data["username"])
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_superuser(current_user: User = Depends(get_current_user)) -> User:
    """Get current superuser"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user
```

### 8.3 Authentication Endpoints

Login and authentication routes:

```python
# app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta
from app.database import get_db
from app.schemas.auth import Token, UserLogin, UserRegister
from app.schemas.user import User as UserSchema
from app.crud.user import user_crud
from app.core.security import verify_password, create_access_token
from app.dependencies import get_current_active_user
from app.config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=UserSchema)
async def register(
    user_data: UserRegister,
    db: AsyncSession = Depends(get_db)
):
    """Register new user"""
    # Check if user exists
    existing_user = await user_crud.get_user_by_email(db, email=user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    existing_user = await user_crud.get_user_by_username(db, username=user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create user
    user = await user_crud.create_user(db, user=user_data)
    return user

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Login and get access token"""
    user = await user_crud.get_user_by_username(db, username=form_data.username)
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.post("/login/json", response_model=Token)
async def login_json(
    user_credentials: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """Login with JSON body"""
    user = await user_crud.get_user_by_email(db, email=user_credentials.email)
    
    if not user or not verify_password(user_credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.get("/me", response_model=UserSchema)
async def read_users_me(current_user: UserSchema = Depends(get_current_active_user)):
    """Get current user"""
    return current_user
```

### 8.4 OAuth2 and Third-party Authentication

OAuth2 integration example:

```python
# app/core/oauth.py
from authlib.integrations.starlette_client import OAuth
from fastapi import Request
from app.config import settings

oauth = OAuth()

# Google OAuth
oauth.register(
    name='google',
    client_id=settings.google_client_id,
    client_secret=settings.google_client_secret,
    server_metadata_url='https://accounts.google.com/.well-known/openid_configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

# GitHub OAuth
oauth.register(
    name='github',
    client_id=settings.github_client_id,
    client_secret=settings.github_client_secret,
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)

# OAuth routes
@router.get("/oauth/{provider}")
async def oauth_login(request: Request, provider: str):
    """Initiate OAuth login"""
    if provider not in ['google', 'github']:
        raise HTTPException(status_code=400, detail="Unsupported OAuth provider")
    
    client = oauth.create_client(provider)
    redirect_uri = request.url_for('oauth_callback', provider=provider)
    return await client.authorize_redirect(request, redirect_uri)

@router.get("/oauth/{provider}/callback")
async def oauth_callback(request: Request, provider: str, db: AsyncSession = Depends(get_db)):
    """Handle OAuth callback"""
    client = oauth.create_client(provider)
    token = await client.authorize_access_token(request)
    
    if provider == 'google':
        user_info = token.get('userinfo')
        email = user_info.get('email')
        name = user_info.get('name')
    elif provider == 'github':
        user_response = await client.get('user', token=token)
        user_info = user_response.json()
        email = user_info.get('email')
        name = user_info.get('name')
    
    # Create or get user
    user = await user_crud.get_user_by_email(db, email=email)
    if not user:
        user_data = UserCreate(
            email=email,
            username=email.split('@')[0],
            full_name=name,
            password="oauth_user"  # OAuth users don't have passwords
        )
        user = await user_crud.create_user(db, user=user_data)
    
    # Create access token
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}
```

## Chapter 9: File Handling and Background Tasks

Managing file uploads and background task processing.

### 9.1 File Upload Handling

Comprehensive file upload implementation:

```python
# app/routers/files.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import FileResponse
from typing import List, Optional
import aiofiles
import os
import uuid
import mimetypes
from pathlib import Path
import shutil
from PIL import Image
from app.dependencies import get_current_active_user
from app.schemas.user import User

router = APIRouter(prefix="/files", tags=["files"])

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".pdf", ".txt", ".docx"}

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size (this is approximate, actual size check happens during upload)
    if hasattr(file.file, 'seek') and hasattr(file.file, 'tell'):
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """Upload a single file"""
    validate_file(file)
    
    # Generate unique filename
    file_ext = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename
    
    try:
        # Save file asynchronously
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Get file info
        file_size = len(content)
        content_type = mimetypes.guess_type(file.filename)[0]
        
        return {
            "filename": file.filename,
            "unique_filename": unique_filename,
            "content_type": content_type,
            "size": file_size,
            "description": description,
            "uploaded_by": current_user.username
        }
    
    except Exception as e:
        # Clean up file if it was created
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@router.post("/upload-multiple")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """Upload multiple files"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    
    for file in files:
        try:
            validate_file(file)
            
            # Generate unique filename
            file_ext = Path(file.filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = UPLOAD_DIR / unique_filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            results.append({
                "filename": file.filename,
                "unique_filename": unique_filename,
                "size": len(content),
                "status": "success"
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
    
    return {"uploaded_files": results}

@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download a file"""
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@router.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    resize: Optional[bool] = Form(False),
    width: Optional[int] = Form(800),
    height: Optional[int] = Form(600),
    current_user: User = Depends(get_current_active_user)
):
    """Upload and optionally resize image"""
    # Validate image file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    validate_file(file)
    
    # Generate unique filename
    file_ext = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename
    
    try:
        content = await file.read()
        
        if resize:
            # Resize image using Pillow
            image = Image.open(io.BytesIO(content))
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Save resized image
            output = io.BytesIO()
            image.save(output, format=image.format)
            content = output.getvalue()
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        return {
            "filename": file.filename,
            "unique_filename": unique_filename,
            "size": len(content),
            "resized": resize,
            "dimensions": {"width": width, "height": height} if resize else None
        }
    
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
```

### 9.2 Background Tasks

Implementing background tasks with FastAPI:

```python
# app/core/tasks.py
from fastapi import BackgroundTasks
import asyncio
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List
from app.config import settings

logger = logging.getLogger(__name__)

async def send_email_background(
    to_email: str, 
    subject: str, 
    message: str
):
    """Send email in background"""
    try:
        # Simulate email sending delay
        await asyncio.sleep(2)
        
        msg = MIMEMultipart()
        msg['From'] = settings.smtp_user
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP(settings.smtp_server, settings.smtp_port)
        server.starttls()
        server.login(settings.smtp_user, settings.smtp_password)
        text = msg.as_string()
        server.sendmail(settings.smtp_user, to_email, text)
        server.quit()
        
        logger.info(f"Email sent to {to_email}")
    
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {str(e)}")

def process_file_background(file_path: str, user_id: int):
    """Process uploaded file in background"""
    try:
        # Simulate file processing
        import time
        time.sleep(5)
        
        logger.info(f"File {file_path} processed for user {user_id}")
    
    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {str(e)}")

async def cleanup_old_files():
    """Clean up old files"""
    try:
        from pathlib import Path
        import time
        
        upload_dir = Path("uploads")
        cutoff_time = time.time() - (7 * 24 * 60 * 60)  # 7 days ago
        
        for file_path in upload_dir.glob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                logger.info(f"Deleted old file: {file_path}")
    
    except Exception as e:
        logger.error(f"Failed to cleanup files: {str(e)}")

# Background task endpoints
@router.post("/send-notification")
async def send_notification(
    to_emails: List[str],
    subject: str,
    message: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Send notification emails in background"""
    for email in to_emails:
        background_tasks.add_task(send_email_background, email, subject, message)
    
    return {"message": f"Notification will be sent to {len(to_emails)} recipients"}

@router.post("/process-file")
async def process_file_endpoint(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_active_user)
):
    """Upload file and process in background"""
    # Save file first
    file_ext = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Add background processing task
    background_tasks.add_task(process_file_background, str(file_path), current_user.id)
    
    return {
        "message": "File uploaded and will be processed in background",
        "filename": unique_filename
    }
```

### 9.3 Celery Integration

Advanced background tasks with Celery:

```python
# app/core/celery_app.py
from celery import Celery
from app.config import settings

celery_app = Celery(
    "fastapi_app",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=['app.core.tasks']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    worker_disable_rate_limits=True,
    worker_prefetch_multiplier=1,
)

# app/core/celery_tasks.py
from celery import current_task
from app.core.celery_app import celery_app
import time

@celery_app.task
def long_running_task(duration: int):
    """Long running task example"""
    for i in range(duration):
        time.sleep(1)
        current_task.update_state(
            state='PROGRESS',
            meta={'current': i, 'total': duration}
        )
    
    return {'status': 'completed', 'result': f'Task completed in {duration} seconds'}

@celery_app.task
def send_email_task(to_email: str, subject: str, message: str):
    """Send email task"""
    try:
        # Email sending logic here
        time.sleep(2)  # Simulate email sending
        return {'status': 'success', 'email': to_email}
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

# Endpoints using Celery tasks
@router.post("/start-long-task")
async def start_long_task(duration: int = 10):
    """Start long running task"""
    task = long_running_task.delay(duration)
    return {"task_id": task.id, "status": "started"}

@router.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': 'In progress...'
        }
    else:
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),
        }
    
    return response
```

## Chapter 10: WebSocket and Real-time Communication

Implementing real-time features with WebSockets in FastAPI.

### 10.1 Basic WebSocket Implementation

Simple WebSocket endpoint:

```python
# app/routers/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict
import json
import asyncio

router = APIRouter()

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str = None):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            self.user_connections[user_id] = websocket

    def disconnect(self, websocket: WebSocket, user_id: str = None):
        """Remove WebSocket connection"""
        self.active_connections.remove(websocket)
        if user_id and user_id in self.user_connections:
            del self.user_connections[user_id]

    async def send_personal_message(self, message: str, user_id: str):
        """Send message to specific user"""
        if user_id in self.user_connections:
            websocket = self.user_connections[user_id]
            await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Broadcast message to all connections"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove broken connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message based on type
            if message_data.get("type") == "chat":
                await handle_chat_message(message_data, user_id)
            elif message_data.get("type") == "notification":
                await handle_notification(message_data, user_id)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        await manager.broadcast(f"User {user_id} disconnected")

async def handle_chat_message(message_data: dict, user_id: str):
    """Handle chat messages"""
    message = {
        "type": "chat",
        "user_id": user_id,
        "message": message_data.get("message"),
        "timestamp": message_data.get("timestamp")
    }
    
    target_user = message_data.get("target_user")
    if target_user:
        # Send to specific user
        await manager.send_personal_message(json.dumps(message), target_user)
    else:
        # Broadcast to all
        await manager.broadcast(json.dumps(message))

async def handle_notification(message_data: dict, user_id: str):
    """Handle notifications"""
    notification = {
        "type": "notification",
        "from_user": user_id,
        "message": message_data.get("message"),
        "timestamp": message_data.get("timestamp")
    }
    
    target_users = message_data.get("target_users", [])
    for target_user in target_users:
        await manager.send_personal_message(json.dumps(notification), target_user)
```

### 10.2 Real-time Chat Application

Complete chat system implementation:

```python
# app/models/chat.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class ChatRoom(Base):
    __tablename__ = "chat_rooms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    created_by = Column(Integer, ForeignKey("users.id"))

    # Relationships
    messages = relationship("ChatMessage", back_populates="room")
    creator = relationship("User")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    
    # Foreign keys
    room_id = Column(Integer, ForeignKey("chat_rooms.id"))
    user_id = Column(Integer, ForeignKey("users.id"))

    # Relationships
    room = relationship("ChatRoom", back_populates="messages")
    user = relationship("User")

# Enhanced WebSocket manager for chat
class ChatManager:
    def __init__(self):
        self.room_connections: Dict[int, List[WebSocket]] = {}
        self.user_connections: Dict[str, WebSocket] = {}

    async def connect_to_room(self, websocket: WebSocket, room_id: int, user_id: str):
        """Connect user to chat room"""
        await websocket.accept()
        
        if room_id not in self.room_connections:
            self.room_connections[room_id] = []
        
        self.room_connections[room_id].append(websocket)
        self.user_connections[f"{user_id}_{room_id}"] = websocket

    def disconnect_from_room(self, websocket: WebSocket, room_id: int, user_id: str):
        """Disconnect user from chat room"""
        if room_id in self.room_connections:
            self.room_connections[room_id].remove(websocket)
        
        key = f"{user_id}_{room_id}"
        if key in self.user_connections:
            del self.user_connections[key]

    async def send_to_room(self, message: str, room_id: int):
        """Send message to all users in room"""
        if room_id in self.room_connections:
            for connection in self.room_connections[room_id].copy():
                try:
                    await connection.send_text(message)
                except:
                    self.room_connections[room_id].remove(connection)

chat_manager = ChatManager()

@router.websocket("/ws/chat/{room_id}/{user_id}")
async def chat_websocket(websocket: WebSocket, room_id: int, user_id: str):
    """WebSocket for chat room"""
    await chat_manager.connect_to_room(websocket, room_id, user_id)
    
    # Notify room about new user
    join_message = {
        "type": "user_joined",
        "user_id": user_id,
        "message": f"User {user_id} joined the chat",
        "timestamp": datetime.now().isoformat()
    }
    await chat_manager.send_to_room(json.dumps(join_message), room_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Save message to database
            # (database operations would go here)
            
            # Broadcast message to room
            chat_message = {
                "type": "chat_message",
                "user_id": user_id,
                "message": message_data.get("message"),
                "timestamp": datetime.now().isoformat()
            }
            
            await chat_manager.send_to_room(json.dumps(chat_message), room_id)
    
    except WebSocketDisconnect:
        chat_manager.disconnect_from_room(websocket, room_id, user_id)
        
        # Notify room about user leaving
        leave_message = {
            "type": "user_left",
            "user_id": user_id,
            "message": f"User {user_id} left the chat",
            "timestamp": datetime.now().isoformat()
        }
        await chat_manager.send_to_room(json.dumps(leave_message), room_id)
```

## Chapter 11: Testing FastAPI Applications

Comprehensive testing strategies for FastAPI applications.

### 11.1 Basic Testing Setup

Setting up the test environment:

```python
# tests/conftest.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app.models import User, Post
from app.core.security import get_password_hash

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
    },
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client():
    """Create test client"""
    with TestClient(app) as c:
        yield c

@pytest.fixture
async def async_client():
    """Create async test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def db():
    """Create test database"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_user(db):
    """Create test user"""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=get_password_hash("testpassword"),
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
```

### 11.2 Testing API Endpoints

Testing CRUD operations:

```python
# tests/test_users.py
import pytest
from httpx import AsyncClient
from app.core.security import create_access_token

class TestUsers:
    """Test user endpoints"""

    async def test_create_user(self, async_client: AsyncClient):
        """Test user creation"""
        user_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "password123",
            "full_name": "New User"
        }
        
        response = await async_client.post("/users/", json=user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == user_data["username"]
        assert data["email"] == user_data["email"]
        assert "id" in data
        assert "password" not in data

    async def test_create_user_duplicate_email(self, async_client: AsyncClient, test_user):
        """Test user creation with duplicate email"""
        user_data = {
            "username": "anotheruser",
            "email": test_user.email,  # Same email as test_user
            "password": "password123"
        }
        
        response = await async_client.post("/users/", json=user_data)
        
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]

    async def test_get_user(self, async_client: AsyncClient, test_user):
        """Test getting user by ID"""
        response = await async_client.get(f"/users/{test_user.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_user.id
        assert data["username"] == test_user.username

    async def test_get_user_not_found(self, async_client: AsyncClient):
        """Test getting non-existent user"""
        response = await async_client.get("/users/999")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    async def test_get_users_list(self, async_client: AsyncClient, test_user):
        """Test getting users list"""
        response = await async_client.get("/users/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    async def test_update_user_authorized(self, async_client: AsyncClient, test_user):
        """Test updating user with authorization"""
        # Create access token
        token = create_access_token(data={"sub": test_user.username})
        headers = {"Authorization": f"Bearer {token}"}
        
        update_data = {"full_name": "Updated Name"}
        
        response = await async_client.put(
            f"/users/{test_user.id}",
            json=update_data,
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["full_name"] == "Updated Name"

    async def test_update_user_unauthorized(self, async_client: AsyncClient, test_user):
        """Test updating user without authorization"""
        update_data = {"full_name": "Updated Name"}
        
        response = await async_client.put(f"/users/{test_user.id}", json=update_data)
        
        assert response.status_code == 401

    async def test_delete_user_forbidden(self, async_client: AsyncClient, test_user):
        """Test deleting user without superuser permissions"""
        token = create_access_token(data={"sub": test_user.username})
        headers = {"Authorization": f"Bearer {token}"}
        
        response = await async_client.delete(f"/users/{test_user.id}", headers=headers)
        
        assert response.status_code == 403
```

### 11.3 Testing Authentication

Authentication and security tests:

```python
# tests/test_auth.py
import pytest
from httpx import AsyncClient
from app.core.security import create_access_token, verify_token

class TestAuth:
    """Test authentication endpoints"""

    async def test_register_user(self, async_client: AsyncClient):
        """Test user registration"""
        user_data = {
            "username": "newuser",
            "email": "new@example.com",
            "password": "password123",
            "password_confirm": "password123"
        }
        
        response = await async_client.post("/auth/register", json=user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == user_data["username"]
        assert data["email"] == user_data["email"]

    async def test_login_success(self, async_client: AsyncClient, test_user):
        """Test successful login"""
        login_data = {
            "email": test_user.email,
            "password": "testpassword"
        }
        
        response = await async_client.post("/auth/login/json", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    async def test_login_invalid_credentials(self, async_client: AsyncClient, test_user):
        """Test login with invalid credentials"""
        login_data = {
            "email": test_user.email,
            "password": "wrongpassword"
        }
        
        response = await async_client.post("/auth/login/json", json=login_data)
        
        assert response.status_code == 401
        assert "Incorrect" in response.json()["detail"]

    async def test_get_current_user(self, async_client: AsyncClient, test_user):
        """Test getting current user info"""
        token = create_access_token(data={"sub": test_user.username})
        headers = {"Authorization": f"Bearer {token}"}
        
        response = await async_client.get("/auth/me", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user.username

    async def test_invalid_token(self, async_client: AsyncClient):
        """Test with invalid token"""
        headers = {"Authorization": "Bearer invalid_token"}
        
        response = await async_client.get("/auth/me", headers=headers)
        
        assert response.status_code == 401

class TestSecurity:
    """Test security functions"""

    def test_create_and_verify_token(self, test_user):
        """Test token creation and verification"""
        token = create_access_token(data={"sub": test_user.username})
        
        assert token is not None
        assert isinstance(token, str)
        
        # Verify token
        token_data = verify_token(token)
        assert token_data is not None
        assert token_data["username"] == test_user.username

    def test_verify_invalid_token(self):
        """Test verification of invalid token"""
        result = verify_token("invalid_token")
        assert result is None

    def test_expired_token(self, test_user):
        """Test expired token"""
        from datetime import timedelta
        
        # Create token that expires immediately
        token = create_access_token(
            data={"sub": test_user.username},
            expires_delta=timedelta(seconds=-1)
        )
        
        # Should be invalid due to expiration
        result = verify_token(token)
        assert result is None
```

### 11.4 Testing WebSockets

WebSocket testing:

```python
# tests/test_websockets.py
import pytest
from fastapi.testclient import TestClient
import json

class TestWebSockets:
    """Test WebSocket functionality"""

    def test_websocket_connection(self, client):
        """Test basic WebSocket connection"""
        with client.websocket_connect("/ws/testuser") as websocket:
            # Send message
            message = {
                "type": "chat",
                "message": "Hello WebSocket!",
                "timestamp": "2024-01-01T12:00:00"
            }
            websocket.send_text(json.dumps(message))
            
            # Should receive the message back (broadcast)
            data = websocket.receive_text()
            received_message = json.loads(data)
            
            assert received_message["type"] == "chat"
            assert received_message["message"] == "Hello WebSocket!"
            assert received_message["user_id"] == "testuser"

    def test_websocket_disconnect(self, client):
        """Test WebSocket disconnection"""
        with client.websocket_connect("/ws/testuser") as websocket:
            websocket.close()
            
        # Connection should be closed gracefully

    def test_chat_room_websocket(self, client):
        """Test chat room WebSocket"""
        with client.websocket_connect("/ws/chat/1/testuser") as websocket:
            # Should receive join message
            data = websocket.receive_text()
            join_message = json.loads(data)
            
            assert join_message["type"] == "user_joined"
            assert join_message["user_id"] == "testuser"
            
            # Send chat message
            chat_message = {
                "message": "Hello chat room!"
            }
            websocket.send_text(json.dumps(chat_message))
            
            # Should receive the chat message back
            data = websocket.receive_text()
            received = json.loads(data)
            
            assert received["type"] == "chat_message"
            assert received["message"] == "Hello chat room!"

@pytest.mark.asyncio
async def test_multiple_websocket_connections(client):
    """Test multiple WebSocket connections"""
    connections = []
    
    try:
        # Create multiple connections
        for i in range(3):
            ws = client.websocket_connect(f"/ws/user{i}")
            await ws.__aenter__()
            connections.append(ws)
        
        # Send message from first connection
        message = {
            "type": "chat",
            "message": "Broadcast message",
            "timestamp": "2024-01-01T12:00:00"
        }
        await connections[0].send_text(json.dumps(message))
        
        # All connections should receive the broadcast
        for ws in connections:
            data = await ws.receive_text()
            received = json.loads(data)
            assert received["message"] == "Broadcast message"
    
    finally:
        # Clean up connections
        for ws in connections:
            await ws.__aexit__(None, None, None)
```

## Chapter 12: Documentation and OpenAPI

Leveraging FastAPI's automatic documentation features.

### 12.1 Advanced OpenAPI Configuration

Customizing OpenAPI schema:

```python
# app/main.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

def custom_openapi():
    """Custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="FastAPI Comprehensive API",
        version="2.0.0",
        description="""
        ## FastAPI Comprehensive API
        
        This is a comprehensive FastAPI application demonstrating:
        
        * **Authentication**: JWT token-based authentication
        * **CRUD Operations**: Full Create, Read, Update, Delete operations
        * **File Upload**: Handle file uploads with validation
        * **WebSocket**: Real-time communication
        * **Background Tasks**: Async task processing
        
        ### Authentication
        
        Most endpoints require authentication. Use the `/auth/login` endpoint to get an access token.
        
        ### Rate Limiting
        
        API endpoints are rate-limited to prevent abuse.
        """,
        routes=app.routes,
        contact={
            "name": "API Support",
            "url": "http://www.example.com/contact/",
            "email": "support@example.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        servers=[
            {
                "url": "https://api.example.com",
                "description": "Production server"
            },
            {
                "url": "https://staging-api.example.com",
                "description": "Staging server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            }
        ]
    )
    
    # Customize schema
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "HTTPBearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Custom documentation pages
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Custom ReDoc"""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )
```

### 12.2 Detailed Endpoint Documentation

Advanced endpoint documentation:

```python
# app/routers/documented_users.py
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from typing import List, Optional
from app.schemas.user import User, UserCreate, UserUpdate
from app.dependencies import get_current_active_user

router = APIRouter(prefix="/users", tags=["users"])

@router.post(
    "/",
    response_model=User,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description="Create a new user with the provided information",
    response_description="The created user",
    responses={
        201: {
            "description": "User created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": 1,
                        "username": "johndoe",
                        "email": "john@example.com",
                        "full_name": "John Doe",
                        "is_active": True,
                        "created_at": "2024-01-01T12:00:00"
                    }
                }
            }
        },
        400: {
            "description": "Bad request - validation error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Email already registered"
                    }
                }
            }
        },
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "email"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def create_user(
    user: UserCreate = Body(
        ...,
        example={
            "username": "johndoe",
            "email": "john@example.com",
            "password": "secretpassword",
            "full_name": "John Doe"
        }
    )
):
    """
    Create a new user account.
    
    **Parameters:**
    - **username**: Unique username (3-50 characters, alphanumeric + underscore)
    - **email**: Valid email address
    - **password**: Strong password (minimum 8 characters)
    - **full_name**: Optional full name
    
    **Returns:**
    - **User object**: Created user with generated ID and timestamp
    
    **Raises:**
    - **400**: If email or username already exists
    - **422**: If validation fails
    """
    # Implementation here
    pass

@router.get(
    "/{user_id}",
    response_model=User,
    summary="Get user by ID",
    description="Retrieve a specific user by their ID",
    responses={
        404: {"description": "User not found"}
    }
)
async def get_user(
    user_id: int = Path(
        ...,
        title="User ID",
        description="The ID of the user to retrieve",
        ge=1,
        example=1
    )
):
    """
    Get a specific user by ID.
    
    **Path Parameters:**
    - **user_id**: The unique identifier of the user (must be positive integer)
    
    **Returns:**
    - **User object**: Complete user information
    
    **Raises:**
    - **404**: If user with given ID doesn't exist
    """
    # Implementation here
    pass

@router.get(
    "/",
    response_model=List[User],
    summary="List users",
    description="Get a paginated list of users",
)
async def list_users(
    skip: int = Query(
        0,
        title="Skip",
        description="Number of users to skip (for pagination)",
        ge=0,
        example=0
    ),
    limit: int = Query(
        100,
        title="Limit",
        description="Maximum number of users to return",
        ge=1,
        le=1000,
        example=10
    ),
    search: Optional[str] = Query(
        None,
        title="Search",
        description="Search term to filter users by username or email",
        min_length=2,
        max_length=50,
        example="john"
    ),
    is_active: Optional[bool] = Query(
        None,
        title="Active Status",
        description="Filter by active status",
        example=True
    )
):
    """
    Retrieve a list of users with optional filtering and pagination.
    
    **Query Parameters:**
    - **skip**: Number of users to skip (default: 0)
    - **limit**: Maximum users to return (default: 100, max: 1000)
    - **search**: Filter by username or email containing this text
    - **is_active**: Filter by active status
    
    **Returns:**
    - **List of User objects**: Paginated list of users matching criteria
    
    **Example:**
    ```
    GET /users?skip=0&limit=10&search=john&is_active=true
    ```
    """
    # Implementation here
    pass
```

## Chapter 13: Deployment and Production

Deploying FastAPI applications to production environments.

### 13.1 Production Configuration

Production-ready configuration:

```python
# app/config.py
from pydantic import BaseSettings, validator
from typing import Optional, List
import secrets

class Settings(BaseSettings):
    """Production settings"""
    
    # Application
    app_name: str = "FastAPI Production App"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Security
    secret_key: str = secrets.token_urlsafe(32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Database
    database_url: str
    database_pool_size: int = 20
    database_max_overflow: int = 0
    
    # CORS
    backend_cors_origins: List[str] = []
    
    @validator("backend_cors_origins", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Redis (for caching and sessions)
    redis_url: str = "redis://localhost:6379"
    
    # Email
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    
    # File upload
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    upload_dir: str = "uploads"
    
    # Logging
    log_level: str = "INFO"
    
    # Monitoring
    sentry_dsn: Optional[str] = None
    enable_metrics: bool = True
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### 13.2 Docker Deployment

Production Dockerfile:

```dockerfile
# Dockerfile
FROM python:3.11-slim as requirements-stage

WORKDIR /tmp

RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Create user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY --from=requirements-stage /tmp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy application
COPY ./app /app/app

# Create directories and set permissions
RUN mkdir -p /app/uploads /app/logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Docker Compose for production:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/fastapi_prod
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=${SECRET_KEY}
      - SENTRY_DSN=${SENTRY_DSN}
    depends_on:
      - db
      - redis
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: fastapi_prod
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
      - ./static:/var/www/static
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 13.3 Nginx Configuration

Nginx reverse proxy configuration:

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream fastapi_app {
        server app:8000;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;
    
    server {
        listen 80;
        server_name api.yourdomain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name api.yourdomain.com;
        
        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # Gzip compression
        gzip on;
        gzip_vary on;
        gzip_min_length 1024;
        gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
        
        # Static files
        location /static/ {
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
        
        # API endpoints
        location / {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://fastapi_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeout settings
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
        
        # Login endpoint with stricter rate limiting
        location /auth/login {
            limit_req zone=login burst=5 nodelay;
            
            proxy_pass http://fastapi_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # WebSocket support
        location /ws {
            proxy_pass http://fastapi_app;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### 13.4 Monitoring and Logging

Production monitoring setup:

```python
# app/core/monitoring.py
import logging
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import sentry_sdk
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
from app.config import settings

# Sentry setup
if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        integrations=[
            FastApiIntegration(auto_enabling_integrations=True),
            SqlalchemyIntegration(),
        ],
        traces_sample_rate=0.1,
    )

# Prometheus metrics
REQUEST_COUNT = Counter(
    'fastapi_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'fastapi_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

# Logging setup
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/app/logs/app.log')
        ]
    )

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring requests"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        if settings.enable_metrics:
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
        
        # Add response headers
        response.headers["X-Process-Time"] = str(duration)
        
        return response

# Health check with detailed status
async def health_check():
    """Comprehensive health check"""
    from app.database import engine
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.app_version,
        "checks": {}
    }
    
    try:
        # Database check
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    try:
        # Redis check
        import aioredis
        redis = aioredis.from_url(settings.redis_url)
        await redis.ping()
        await redis.close()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
    
    return health_status

# Metrics endpoint
def get_metrics():
    """Get Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

## Chapter 14: Advanced Features and Patterns

Advanced FastAPI patterns and real-world implementation techniques.

### 14.1 Middleware and Request Processing

Custom middleware patterns:

```python
# app/middleware/custom.py
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import uuid
import redis.asyncio as redis
from app.config import settings

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.redis = redis.from_url(settings.redis_url)
    
    async def dispatch(self, request: Request, call_next):
        if not settings.rate_limit_enabled:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host
        if "x-forwarded-for" in request.headers:
            client_ip = request.headers["x-forwarded-for"].split(",")[0]
        
        # Create rate limit key
        key = f"rate_limit:{client_ip}"
        
        try:
            # Get current count
            current = await self.redis.get(key)
            
            if current is None:
                # First request in window
                await self.redis.setex(key, self.period, 1)
            else:
                current_count = int(current)
                if current_count >= self.calls:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded"}
                    )
                
                # Increment counter
                await self.redis.incr(key)
        
        except Exception:
            # Redis unavailable, allow request
            pass
        
        return await call_next(request)

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request"""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response

class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with advanced options"""
    
    def __init__(self, app, allow_origins=None, allow_methods=None):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE"]
    
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
            return response
        
        response = await call_next(request)
        
        # Add CORS headers
        if origin in self.allow_origins or "*" in self.allow_origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response
```

### 14.2 Advanced Dependency Injection

Sophisticated dependency patterns:

```python
# app/dependencies/advanced.py
from fastapi import Depends, HTTPException, Query
from typing import Optional, Dict, Any
from functools import lru_cache
import redis.asyncio as redis
from app.config import settings
from app.database import AsyncSession, get_db

# Cached dependencies
@lru_cache()
def get_settings():
    """Cached settings dependency"""
    return settings

# Redis dependency
async def get_redis() -> redis.Redis:
    """Get Redis connection"""
    client = redis.from_url(settings.redis_url)
    try:
        yield client
    finally:
        await client.close()

# Pagination dependency
class PaginationParams:
    def __init__(
        self,
        skip: int = Query(0, ge=0, description="Items to skip"),
        limit: int = Query(50, ge=1, le=100, description="Items per page")
    ):
        self.skip = skip
        self.limit = limit
        self.offset = skip
        self.size = limit

# Filtering dependency
class FilterParams:
    def __init__(
        self,
        search: Optional[str] = Query(None, min_length=1, max_length=100),
        sort_by: str = Query("created_at", description="Sort field"),
        sort_order: str = Query("desc", regex="^(asc|desc)$"),
        filters: Dict[str, Any] = Query({})
    ):
        self.search = search
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.filters = filters

# Permission dependency
class RequirePermissions:
    def __init__(self, *permissions):
        self.permissions = permissions
    
    def __call__(self, current_user=Depends(get_current_user)):
        user_permissions = set(current_user.permissions)
        required_permissions = set(self.permissions)
        
        if not required_permissions.issubset(user_permissions):
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions"
            )
        
        return current_user

# Cache dependency
class CacheManager:
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
    
    async def get(self, key: str, redis_client=Depends(get_redis)):
        """Get cached value"""
        try:
            value = await redis_client.get(key)
            if value:
                import json
                return json.loads(value)
        except Exception:
            pass
        return None
    
    async def set(self, key: str, value: Any, redis_client=Depends(get_redis)):
        """Set cached value"""
        try:
            import json
            await redis_client.setex(key, self.ttl, json.dumps(value, default=str))
        except Exception:
            pass

# Usage examples
require_admin = RequirePermissions("admin")
require_write = RequirePermissions("write")
cache_5min = CacheManager(ttl=300)

@router.get("/admin-only")
async def admin_endpoint(user=Depends(require_admin)):
    return {"message": "Admin access granted"}

@router.get("/users/cached")
async def get_users_cached(
    pagination: PaginationParams = Depends(),
    filters: FilterParams = Depends(),
    cache: CacheManager = Depends(cache_5min)
):
    # Try cache first
    cache_key = f"users:{pagination.skip}:{pagination.limit}:{filters.search}"
    cached_result = await cache.get(cache_key)
    if cached_result:
        return cached_result
    
    # Fetch from database
    # ... database query logic ...
    result = {"users": [], "total": 0}
    
    # Cache result
    await cache.set(cache_key, result)
    
    return result
```

### 14.3 Event System and Webhooks

Event-driven architecture:

```python
# app/events/system.py
from typing import Dict, List, Callable, Any
import asyncio
import json
from datetime import datetime
from app.models import Event
from app.database import get_db

class EventManager:
    """Event management system"""
    
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}
        self.webhooks: Dict[str, List[str]] = {}
    
    def on(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def webhook(self, event_type: str, url: str):
        """Register webhook for event type"""
        if event_type not in self.webhooks:
            self.webhooks[event_type] = []
        self.webhooks[event_type].append(url)
    
    async def emit(self, event_type: str, data: Dict[str, Any]):
        """Emit event"""
        event = Event(
            type=event_type,
            data=data,
            timestamp=datetime.utcnow()
        )
        
        # Save to database
        async with get_db() as db:
            db.add(event)
            await db.commit()
        
        # Call handlers
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    print(f"Error in event handler: {e}")
        
        # Send webhooks
        if event_type in self.webhooks:
            for webhook_url in self.webhooks[event_type]:
                await self.send_webhook(webhook_url, event)
    
    async def send_webhook(self, url: str, event: Event):
        """Send webhook notification"""
        import httpx
        
        payload = {
            "event_type": event.type,
            "data": event.data,
            "timestamp": event.timestamp.isoformat()
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=payload,
                    timeout=10.0,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
        except Exception as e:
            print(f"Webhook error: {e}")

# Global event manager
events = EventManager()

# Event handlers
@events.on("user.created")
async def send_welcome_email(event):
    """Send welcome email when user is created"""
    user_data = event.data
    # Send welcome email logic
    print(f"Sending welcome email to {user_data['email']}")

@events.on("order.completed")
async def process_order_completion(event):
    """Process completed order"""
    order_data = event.data
    # Order completion logic
    print(f"Processing completed order {order_data['order_id']}")

# Usage in endpoints
@router.post("/users/")
async def create_user(user_data: UserCreate):
    # Create user logic
    user = await create_user_in_db(user_data)
    
    # Emit event
    await events.emit("user.created", {
        "user_id": user.id,
        "email": user.email,
        "username": user.username
    })
    
    return user
```

## Chapter 15: Real-world FastAPI Project

A complete e-commerce API demonstrating all concepts.

### 15.1 Project Structure

Complete production project structure:

```
ecommerce_api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── product.py
│   │   ├── order.py
│   │   └── base.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── product.py
│   │   └── order.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── users.py
│   │   ├── products.py
│   │   └── orders.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py
│   │   ├── deps.py
│   │   └── config.py
│   ├── crud/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── product.py
│   │   └── order.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── payment.py
│   │   ├── email.py
│   │   └── inventory.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
├── alembic/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

This comprehensive FastAPI guide now covers all essential aspects of building production-ready APIs with FastAPI, from basic concepts through advanced deployment patterns and real-world implementation strategies. The guide provides over 2,800 lines of detailed content with practical examples, best practices, and complete code implementations.