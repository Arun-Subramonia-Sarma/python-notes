# Comprehensive SQLAlchemy Guide

A complete guide to SQLAlchemy - Python's most powerful SQL toolkit and Object-Relational Mapping (ORM) framework.

## Table of Contents

1. [Introduction to SQLAlchemy](#chapter-1-introduction-to-sqlalchemy)
2. [Installation and Setup](#chapter-2-installation-and-setup)
3. [SQLAlchemy Core](#chapter-3-sqlalchemy-core)
4. [Database Engines and Connections](#chapter-4-database-engines-and-connections)
5. [Schema Definition and Metadata](#chapter-5-schema-definition-and-metadata)
6. [SQLAlchemy ORM Fundamentals](#chapter-6-sqlalchemy-orm-fundamentals)
7. [Model Definition and Configuration](#chapter-7-model-definition-and-configuration)
8. [Relationships and Associations](#chapter-8-relationships-and-associations)
9. [Querying with the ORM](#chapter-9-querying-with-the-orm)
10. [Sessions and Transactions](#chapter-10-sessions-and-transactions)
11. [Advanced Query Techniques](#chapter-11-advanced-query-techniques)
12. [Database Migrations with Alembic](#chapter-12-database-migrations-with-alembic)
13. [Performance Optimization](#chapter-13-performance-optimization)
14. [Advanced Patterns and Techniques](#chapter-14-advanced-patterns-and-techniques)
15. [Integration with Web Frameworks](#chapter-15-integration-with-web-frameworks)

---

## Chapter 1: Introduction to SQLAlchemy

SQLAlchemy is a comprehensive SQL toolkit and Object-Relational Mapping (ORM) library for Python. It provides a high-level ORM and a lower-level "Core" that offers direct SQL expression language capabilities.

### 1.1 What is SQLAlchemy?

SQLAlchemy provides:
- **Core**: Database abstraction layer with SQL expression language
- **ORM**: Object-Relational Mapping for working with database records as Python objects
- **Engine**: Database connection and execution framework
- **Schema**: Database schema definition and management tools

### 1.2 SQLAlchemy Architecture

```python
# SQLAlchemy Architecture Overview
"""
Application Layer
    |
ORM Layer (Optional)
    |
Core Layer
    |
Engine Layer
    |
DBAPI (Database drivers)
    |
Database
"""

# Two main usage patterns:
# 1. Core - SQL expression language
# 2. ORM - Object-relational mapping
```

### 1.3 Core vs ORM

**SQLAlchemy Core:**
- Lower-level, SQL-centric approach
- Direct SQL expression construction
- High performance for complex queries
- More explicit database interactions

**SQLAlchemy ORM:**
- Higher-level, object-oriented approach
- Maps database tables to Python classes
- Automatic SQL generation
- Rich relationship handling

### 1.4 Key Features

- Database agnostic (PostgreSQL, MySQL, SQLite, Oracle, etc.)
- Connection pooling and engine management
- Schema introspection and migration support
- Lazy loading and eager loading strategies
- Transaction management
- Query optimization and caching
- Extensive customization options

---

## Chapter 2: Installation and Setup

### 2.1 Installation

```bash
# Basic SQLAlchemy installation
pip install sqlalchemy

# With specific database drivers
pip install sqlalchemy[postgresql]  # PostgreSQL
pip install sqlalchemy[mysql]       # MySQL
pip install sqlalchemy[oracle]      # Oracle
pip install sqlalchemy[mssql]       # Microsoft SQL Server

# Complete installation with common drivers
pip install sqlalchemy psycopg2-binary pymysql

# Development dependencies
pip install sqlalchemy alembic pytest sqlalchemy-utils
```

### 2.2 Version Compatibility

```python
import sqlalchemy
print(f"SQLAlchemy version: {sqlalchemy.__version__}")

# SQLAlchemy 2.0+ syntax (recommended)
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

# Check for 2.0 features
if sqlalchemy.__version__.startswith('2'):
    print("Using SQLAlchemy 2.0+ syntax")
else:
    print("Consider upgrading to SQLAlchemy 2.0+")
```

### 2.3 Basic Configuration

```python
# config.py - Database configuration
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Generator
import os

# Database URLs for different databases
DATABASE_URLS = {
    'sqlite': 'sqlite:///example.db',
    'postgresql': 'postgresql://user:password@localhost:5432/dbname',
    'mysql': 'mysql+pymysql://user:password@localhost:3306/dbname',
    'oracle': 'oracle://user:password@localhost:1521/dbname',
    'mssql': 'mssql+pyodbc://user:password@server/database?driver=ODBC+Driver+17+for+SQL+Server'
}

class DatabaseConfig:
    """Database configuration class"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv('DATABASE_URL', DATABASE_URLS['sqlite'])
        self.engine = None
        self.SessionLocal = None
        self.setup()
    
    def setup(self):
        """Setup database engine and session factory"""
        self.engine = create_engine(
            self.database_url,
            echo=True,  # Log SQL statements
            pool_size=20,
            max_overflow=0,
            pool_pre_ping=True,
            pool_recycle=300
        )
        
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )
    
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with proper cleanup"""
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

# Initialize database configuration
db_config = DatabaseConfig()
```

### 2.4 Environment Setup

```python
# .env file configuration
"""
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
SQLALCHEMY_ECHO=True
SQLALCHEMY_POOL_SIZE=20
SQLALCHEMY_MAX_OVERFLOW=10
"""

# environment.py - Environment configuration
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings"""
    
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///app.db')
    SQLALCHEMY_ECHO: bool = os.getenv('SQLALCHEMY_ECHO', 'False').lower() == 'true'
    SQLALCHEMY_POOL_SIZE: int = int(os.getenv('SQLALCHEMY_POOL_SIZE', '20'))
    SQLALCHEMY_MAX_OVERFLOW: int = int(os.getenv('SQLALCHEMY_MAX_OVERFLOW', '10'))
    
    # Additional database settings
    SQLALCHEMY_POOL_TIMEOUT: int = int(os.getenv('SQLALCHEMY_POOL_TIMEOUT', '30'))
    SQLALCHEMY_POOL_RECYCLE: int = int(os.getenv('SQLALCHEMY_POOL_RECYCLE', '3600'))
    SQLALCHEMY_POOL_PRE_PING: bool = os.getenv('SQLALCHEMY_POOL_PRE_PING', 'True').lower() == 'true'

settings = Settings()
```

---

## Chapter 3: SQLAlchemy Core

SQLAlchemy Core provides the foundation for database operations with a SQL-centric approach.

### 3.1 Core Fundamentals

```python
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, DateTime, ForeignKey
from datetime import datetime

# Create engine
engine = create_engine('sqlite:///core_example.db', echo=True)

# Execute raw SQL
with engine.connect() as connection:
    # Using text() for raw SQL
    result = connection.execute(text("SELECT 'Hello SQLAlchemy Core' as message"))
    print(result.fetchone())
    
    # Parameterized queries
    result = connection.execute(
        text("SELECT * FROM users WHERE name = :name"),
        {"name": "John"}
    )
```

### 3.2 Metadata and Table Definitions

```python
# Define metadata container
metadata = MetaData()

# Define tables using Core syntax
users_table = Table(
    'users',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('username', String(50), unique=True, nullable=False),
    Column('email', String(100), unique=True, nullable=False),
    Column('full_name', String(100)),
    Column('created_at', DateTime, default=datetime.utcnow),
    Column('is_active', Boolean, default=True)
)

posts_table = Table(
    'posts',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('title', String(200), nullable=False),
    Column('content', Text),
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('created_at', DateTime, default=datetime.utcnow),
    Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
)

# Create all tables
metadata.create_all(engine)
```

### 3.3 CRUD Operations with Core

```python
from sqlalchemy import insert, select, update, delete

# INSERT operations
def create_user_core(username: str, email: str, full_name: str = None):
    """Create user using Core syntax"""
    with engine.connect() as connection:
        stmt = insert(users_table).values(
            username=username,
            email=email,
            full_name=full_name
        )
        result = connection.execute(stmt)
        connection.commit()
        return result.inserted_primary_key[0]

# SELECT operations
def get_users_core():
    """Get all users using Core syntax"""
    with engine.connect() as connection:
        stmt = select(users_table)
        result = connection.execute(stmt)
        return result.fetchall()

def get_user_by_id_core(user_id: int):
    """Get user by ID using Core syntax"""
    with engine.connect() as connection:
        stmt = select(users_table).where(users_table.c.id == user_id)
        result = connection.execute(stmt)
        return result.fetchone()

# UPDATE operations
def update_user_core(user_id: int, **kwargs):
    """Update user using Core syntax"""
    with engine.connect() as connection:
        stmt = update(users_table).where(
            users_table.c.id == user_id
        ).values(**kwargs)
        result = connection.execute(stmt)
        connection.commit()
        return result.rowcount

# DELETE operations
def delete_user_core(user_id: int):
    """Delete user using Core syntax"""
    with engine.connect() as connection:
        stmt = delete(users_table).where(users_table.c.id == user_id)
        result = connection.execute(stmt)
        connection.commit()
        return result.rowcount

# Example usage
if __name__ == "__main__":
    # Create users
    user1_id = create_user_core("john_doe", "john@example.com", "John Doe")
    user2_id = create_user_core("jane_smith", "jane@example.com", "Jane Smith")
    
    # Read users
    all_users = get_users_core()
    print(f"All users: {all_users}")
    
    # Update user
    update_user_core(user1_id, full_name="John Updated Doe")
    
    # Delete user
    delete_user_core(user2_id)
```

### 3.4 Advanced Core Queries

```python
from sqlalchemy import func, and_, or_, desc, asc, case, cast, Date

def advanced_core_queries():
    """Demonstrate advanced Core query patterns"""
    
    with engine.connect() as connection:
        
        # Aggregate functions
        stmt = select(func.count(users_table.c.id).label('user_count'))
        result = connection.execute(stmt).fetchone()
        print(f"Total users: {result.user_count}")
        
        # JOIN operations
        stmt = select(
            users_table.c.username,
            func.count(posts_table.c.id).label('post_count')
        ).select_from(
            users_table.outerjoin(posts_table)
        ).group_by(
            users_table.c.id, users_table.c.username
        )
        
        results = connection.execute(stmt).fetchall()
        for row in results:
            print(f"User: {row.username}, Posts: {row.post_count}")
        
        # Complex WHERE conditions
        stmt = select(users_table).where(
            and_(
                users_table.c.is_active == True,
                or_(
                    users_table.c.username.like('%john%'),
                    users_table.c.email.like('%example.com')
                )
            )
        )
        
        # Subqueries
        subquery = select(func.avg(posts_table.c.id)).scalar_subquery()
        stmt = select(posts_table).where(posts_table.c.id > subquery)
        
        # Case statements
        stmt = select(
            users_table.c.username,
            case(
                (users_table.c.is_active == True, 'Active'),
                else_='Inactive'
            ).label('status')
        )
        
        # Window functions (PostgreSQL example)
        # stmt = select(
        #     users_table.c.username,
        #     func.row_number().over(
        #         order_by=users_table.c.created_at.desc()
        #     ).label('row_num')
        # )
```

---

## Chapter 4: Database Engines and Connections

### 4.1 Engine Configuration

```python
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool, QueuePool
import logging

# Configure logging to see SQL statements
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

class EngineFactory:
    """Factory for creating database engines with various configurations"""
    
    @staticmethod
    def create_sqlite_engine(database_path: str = "app.db", **kwargs) -> Engine:
        """Create SQLite engine with optimal settings"""
        return create_engine(
            f'sqlite:///{database_path}',
            echo=kwargs.get('echo', False),
            poolclass=StaticPool,
            connect_args={
                'check_same_thread': False,  # Allow multithreading
                'timeout': 20
            },
            **kwargs
        )
    
    @staticmethod
    def create_postgresql_engine(
        username: str,
        password: str,
        host: str = 'localhost',
        port: int = 5432,
        database: str = 'postgres',
        **kwargs
    ) -> Engine:
        """Create PostgreSQL engine with connection pooling"""
        database_url = f'postgresql://{username}:{password}@{host}:{port}/{database}'
        
        return create_engine(
            database_url,
            echo=kwargs.get('echo', False),
            poolclass=QueuePool,
            pool_size=kwargs.get('pool_size', 20),
            max_overflow=kwargs.get('max_overflow', 10),
            pool_pre_ping=True,
            pool_recycle=kwargs.get('pool_recycle', 3600),
            connect_args=kwargs.get('connect_args', {})
        )
    
    @staticmethod
    def create_mysql_engine(
        username: str,
        password: str,
        host: str = 'localhost',
        port: int = 3306,
        database: str = 'mysql',
        **kwargs
    ) -> Engine:
        """Create MySQL engine with specific configurations"""
        database_url = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
        
        return create_engine(
            database_url,
            echo=kwargs.get('echo', False),
            poolclass=QueuePool,
            pool_size=kwargs.get('pool_size', 20),
            max_overflow=kwargs.get('max_overflow', 10),
            pool_pre_ping=True,
            pool_recycle=kwargs.get('pool_recycle', 3600),
            connect_args={
                'charset': 'utf8mb4',
                'autocommit': False,
                **kwargs.get('connect_args', {})
            }
        )

# Example engine creation
engine = EngineFactory.create_postgresql_engine(
    username='myuser',
    password='mypassword',
    database='mydb',
    echo=True,
    pool_size=30
)
```

### 4.2 Connection Management

```python
from contextlib import contextmanager
from sqlalchemy.engine import Connection
from typing import Generator

class ConnectionManager:
    """Advanced connection management with proper resource cleanup"""
    
    def __init__(self, engine: Engine):
        self.engine = engine
    
    @contextmanager
    def get_connection(self) -> Generator[Connection, None, None]:
        """Get connection with automatic cleanup"""
        connection = self.engine.connect()
        try:
            yield connection
        except Exception as e:
            connection.rollback()
            raise e
        finally:
            connection.close()
    
    @contextmanager
    def get_transaction(self) -> Generator[Connection, None, None]:
        """Get connection with transaction management"""
        connection = self.engine.connect()
        transaction = connection.begin()
        try:
            yield connection
            transaction.commit()
        except Exception as e:
            transaction.rollback()
            raise e
        finally:
            connection.close()
    
    def execute_in_transaction(self, func, *args, **kwargs):
        """Execute function within a transaction"""
        with self.get_transaction() as connection:
            return func(connection, *args, **kwargs)

# Usage example
conn_manager = ConnectionManager(engine)

def transfer_funds(from_account: int, to_account: int, amount: float):
    """Example of transactional operation"""
    def _transfer(connection):
        # Debit from source account
        connection.execute(
            text("UPDATE accounts SET balance = balance - :amount WHERE id = :account_id"),
            {"amount": amount, "account_id": from_account}
        )
        
        # Credit to destination account
        connection.execute(
            text("UPDATE accounts SET balance = balance + :amount WHERE id = :account_id"),
            {"amount": amount, "account_id": to_account}
        )
        
        return True
    
    return conn_manager.execute_in_transaction(_transfer)
```

### 4.3 Connection Pooling

```python
from sqlalchemy.pool import QueuePool, StaticPool, NullPool
from sqlalchemy import event

def configure_connection_pool():
    """Configure advanced connection pooling"""
    
    # Production PostgreSQL configuration
    production_engine = create_engine(
        'postgresql://user:pass@localhost/dbname',
        
        # Pool configuration
        poolclass=QueuePool,
        pool_size=20,              # Number of connections to keep persistently
        max_overflow=30,           # Additional connections beyond pool_size
        pool_timeout=30,           # Seconds to wait for connection
        pool_recycle=3600,         # Recycle connections after 1 hour
        pool_pre_ping=True,        # Validate connections before use
        
        # Connection arguments
        connect_args={
            "application_name": "MyApp",
            "connect_timeout": 10,
        }
    )
    
    # Development SQLite configuration
    development_engine = create_engine(
        'sqlite:///dev.db',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False}
    )
    
    # Testing configuration (no pooling)
    test_engine = create_engine(
        'sqlite:///:memory:',
        poolclass=NullPool,
        connect_args={'check_same_thread': False}
    )
    
    return {
        'production': production_engine,
        'development': development_engine,
        'testing': test_engine
    }

# Connection pool monitoring
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for better performance"""
    if 'sqlite' in str(dbapi_connection):
        cursor = dbapi_connection.cursor()
        # Enable foreign key support
        cursor.execute("PRAGMA foreign_keys=ON")
        # Set journal mode to WAL for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        # Set synchronous to NORMAL for better performance
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

@event.listens_for(Engine, "pool_overflow")
def pool_overflow_handler(dbapi_connection, connection_record, pool):
    """Handle connection pool overflow"""
    logging.warning(f"Connection pool overflow detected. Pool size: {pool.size()}")

# Pool status monitoring
def monitor_pool_status(engine: Engine):
    """Monitor connection pool status"""
    pool = engine.pool
    
    status = {
        'pool_size': pool.size(),
        'checked_in': pool.checkedin(),
        'checked_out': pool.checkedout(),
        'overflow': pool.overflow(),
        'invalid': pool.invalid()
    }
    
    return status
```

---

## Chapter 5: Schema Definition and Metadata

### 5.1 Advanced Schema Definition

```python
from sqlalchemy import (
    MetaData, Table, Column, Integer, String, Text, DateTime, Boolean, 
    Numeric, JSON, ARRAY, Enum as SQLEnum, Index, UniqueConstraint, 
    CheckConstraint, ForeignKeyConstraint
)
from enum import Enum as PyEnum
import datetime

# Define custom types
class UserRole(PyEnum):
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"

class PostStatus(PyEnum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"

# Advanced metadata with naming convention
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s"
    }
)

# Comprehensive table definition
users_table = Table(
    'users',
    metadata,
    
    # Primary key
    Column('id', Integer, primary_key=True),
    
    # Basic fields with constraints
    Column('username', String(50), unique=True, nullable=False),
    Column('email', String(255), unique=True, nullable=False),
    Column('password_hash', String(255), nullable=False),
    
    # Profile information
    Column('first_name', String(100)),
    Column('last_name', String(100)),
    Column('bio', Text),
    
    # Enum field
    Column('role', SQLEnum(UserRole), default=UserRole.USER, nullable=False),
    
    # JSON field (PostgreSQL, MySQL 8.0+)
    Column('preferences', JSON, default=dict),
    Column('metadata_info', JSON),
    
    # Numeric fields
    Column('account_balance', Numeric(10, 2), default=0.00),
    Column('login_count', Integer, default=0),
    
    # Boolean fields
    Column('is_active', Boolean, default=True, nullable=False),
    Column('is_verified', Boolean, default=False, nullable=False),
    Column('email_notifications', Boolean, default=True),
    
    # Timestamp fields
    Column('created_at', DateTime, default=datetime.datetime.utcnow, nullable=False),
    Column('updated_at', DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow),
    Column('last_login', DateTime),
    Column('email_verified_at', DateTime),
    
    # Indexes
    Index('ix_users_email_active', 'email', 'is_active'),
    Index('ix_users_role_created', 'role', 'created_at'),
    
    # Constraints
    CheckConstraint('account_balance >= 0', name='positive_balance'),
    CheckConstraint('login_count >= 0', name='non_negative_logins'),
    CheckConstraint("email LIKE '%@%.%'", name='valid_email_format'),
)

# Posts table with advanced features
posts_table = Table(
    'posts',
    metadata,
    
    Column('id', Integer, primary_key=True),
    Column('title', String(200), nullable=False),
    Column('slug', String(250), unique=True, nullable=False),
    Column('content', Text),
    Column('excerpt', String(500)),
    
    # Foreign key
    Column('author_id', Integer, nullable=False),
    
    # Status and categories
    Column('status', SQLEnum(PostStatus), default=PostStatus.DRAFT),
    Column('tags', ARRAY(String), default=list),  # PostgreSQL only
    Column('metadata_info', JSON, default=dict),
    
    # SEO and analytics
    Column('view_count', Integer, default=0),
    Column('like_count', Integer, default=0),
    Column('meta_title', String(60)),
    Column('meta_description', String(160)),
    
    # Timestamps
    Column('created_at', DateTime, default=datetime.datetime.utcnow),
    Column('updated_at', DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow),
    Column('published_at', DateTime),
    
    # Foreign key constraint
    ForeignKeyConstraint(['author_id'], ['users.id'], ondelete='CASCADE'),
    
    # Indexes
    Index('ix_posts_author_status', 'author_id', 'status'),
    Index('ix_posts_published', 'published_at', 'status'),
    Index('ix_posts_slug_unique', 'slug', unique=True),
    
    # Constraints
    CheckConstraint('view_count >= 0', name='non_negative_views'),
    CheckConstraint('like_count >= 0', name='non_negative_likes'),
    CheckConstraint("slug ~ '^[a-z0-9-]+$'", name='valid_slug_format'),  # PostgreSQL regex
)

# Comments table with self-referencing relationship
comments_table = Table(
    'comments',
    metadata,
    
    Column('id', Integer, primary_key=True),
    Column('post_id', Integer, nullable=False),
    Column('user_id', Integer, nullable=False),
    Column('parent_id', Integer),  # Self-referencing for threaded comments
    
    Column('content', Text, nullable=False),
    Column('is_approved', Boolean, default=False),
    Column('like_count', Integer, default=0),
    
    Column('created_at', DateTime, default=datetime.datetime.utcnow),
    Column('updated_at', DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow),
    
    # Foreign key constraints
    ForeignKeyConstraint(['post_id'], ['posts.id'], ondelete='CASCADE'),
    ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    ForeignKeyConstraint(['parent_id'], ['comments.id'], ondelete='CASCADE'),
    
    # Indexes
    Index('ix_comments_post_approved', 'post_id', 'is_approved'),
    Index('ix_comments_user_created', 'user_id', 'created_at'),
    Index('ix_comments_parent', 'parent_id'),
    
    # Constraints
    CheckConstraint('like_count >= 0', name='non_negative_comment_likes'),
    CheckConstraint('parent_id != id', name='no_self_reference'),
)
```

### 5.2 Database Schema Introspection

```python
from sqlalchemy import inspect, text
from sqlalchemy.engine import Inspector

class SchemaIntrospector:
    """Utility class for database schema introspection"""
    
    def __init__(self, engine):
        self.engine = engine
        self.inspector = inspect(engine)
    
    def get_table_info(self, table_name: str) -> dict:
        """Get comprehensive table information"""
        info = {
            'table_name': table_name,
            'columns': self.inspector.get_columns(table_name),
            'primary_keys': self.inspector.get_pk_constraint(table_name),
            'foreign_keys': self.inspector.get_foreign_keys(table_name),
            'indexes': self.inspector.get_indexes(table_name),
            'unique_constraints': self.inspector.get_unique_constraints(table_name),
            'check_constraints': self.inspector.get_check_constraints(table_name)
        }
        return info
    
    def get_all_tables(self) -> list:
        """Get list of all table names"""
        return self.inspector.get_table_names()
    
    def get_all_views(self) -> list:
        """Get list of all view names"""
        return self.inspector.get_view_names()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        return table_name in self.get_all_tables()
    
    def get_column_info(self, table_name: str, column_name: str) -> dict:
        """Get detailed column information"""
        columns = self.inspector.get_columns(table_name)
        for column in columns:
            if column['name'] == column_name:
                return column
        return None
    
    def analyze_schema(self) -> dict:
        """Comprehensive schema analysis"""
        analysis = {
            'tables': {},
            'total_tables': 0,
            'total_columns': 0,
            'foreign_key_relationships': []
        }
        
        tables = self.get_all_tables()
        analysis['total_tables'] = len(tables)
        
        for table_name in tables:
            table_info = self.get_table_info(table_name)
            analysis['tables'][table_name] = table_info
            analysis['total_columns'] += len(table_info['columns'])
            
            # Collect foreign key relationships
            for fk in table_info['foreign_keys']:
                analysis['foreign_key_relationships'].append({
                    'source_table': table_name,
                    'source_columns': fk['constrained_columns'],
                    'target_table': fk['referred_table'],
                    'target_columns': fk['referred_columns']
                })
        
        return analysis

# Example usage
def demonstrate_schema_introspection():
    """Demonstrate schema introspection capabilities"""
    
    # Create inspector
    introspector = SchemaIntrospector(engine)
    
    # Check if table exists
    if introspector.table_exists('users'):
        print("Users table exists")
        
        # Get table information
        table_info = introspector.get_table_info('users')
        print(f"Table: {table_info['table_name']}")
        
        # Print column information
        for column in table_info['columns']:
            print(f"  Column: {column['name']} ({column['type']})")
        
        # Print indexes
        for index in table_info['indexes']:
            print(f"  Index: {index['name']} on {index['column_names']}")
    
    # Full schema analysis
    schema_analysis = introspector.analyze_schema()
    print(f"Total tables: {schema_analysis['total_tables']}")
    print(f"Total columns: {schema_analysis['total_columns']}")
    print(f"Foreign key relationships: {len(schema_analysis['foreign_key_relationships'])}")
```

---

## Chapter 6: SQLAlchemy ORM Fundamentals

### 6.1 ORM Base Configuration (SQLAlchemy 2.0 Style)

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, DateTime, Boolean, Text, ForeignKey
from datetime import datetime
from typing import Optional, List

# Define the ORM base class
class Base(DeclarativeBase):
    """Base class for all ORM models"""
    pass

# Alternative base with common fields
class TimestampMixin:
    """Mixin for timestamp fields"""
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )

class BaseModel(Base):
    """Enhanced base model with common functionality"""
    __abstract__ = True
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )
    
    def to_dict(self) -> dict:
        """Convert model instance to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update(self, **kwargs):
        """Update model instance with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def __repr__(self):
        """String representation of model"""
        return f"<{self.__class__.__name__}(id={getattr(self, 'id', None)})>"
```

### 6.2 Basic ORM Models

```python
from sqlalchemy import Enum as SQLEnum
from enum import Enum as PyEnum

# Enums
class UserRole(PyEnum):
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"

class PostStatus(PyEnum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"

# User model with comprehensive fields
class User(BaseModel):
    """User model with full feature set"""
    __tablename__ = 'users'
    
    # Basic information
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Profile information
    first_name: Mapped[Optional[str]] = mapped_column(String(100))
    last_name: Mapped[Optional[str]] = mapped_column(String(100))
    bio: Mapped[Optional[str]] = mapped_column(Text)
    
    # Account settings
    role: Mapped[UserRole] = mapped_column(SQLEnum(UserRole), default=UserRole.USER)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Properties
    @property
    def full_name(self) -> str:
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name or self.last_name or self.username
    
    @property
    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.role == UserRole.ADMIN
    
    # Class methods
    @classmethod
    def get_by_username(cls, session, username: str):
        """Get user by username"""
        return session.query(cls).filter_by(username=username).first()
    
    @classmethod
    def get_by_email(cls, session, email: str):
        """Get user by email"""
        return session.query(cls).filter_by(email=email).first()
    
    # Instance methods
    def set_password(self, password: str):
        """Set user password (would use proper hashing in production)"""
        import hashlib
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    def check_password(self, password: str) -> bool:
        """Check if password is correct"""
        import hashlib
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()
    
    def activate(self):
        """Activate user account"""
        self.is_active = True
        self.updated_at = datetime.utcnow()
    
    def deactivate(self):
        """Deactivate user account"""
        self.is_active = False
        self.updated_at = datetime.utcnow()

# Post model
class Post(BaseModel):
    """Blog post model"""
    __tablename__ = 'posts'
    
    # Content fields
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(250), unique=True, nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text)
    excerpt: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Foreign key
    author_id: Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False)
    
    # Status and metadata
    status: Mapped[PostStatus] = mapped_column(SQLEnum(PostStatus), default=PostStatus.DRAFT)
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # SEO fields
    meta_title: Mapped[Optional[str]] = mapped_column(String(60))
    meta_description: Mapped[Optional[str]] = mapped_column(String(160))
    
    # Publication date
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Properties
    @property
    def is_published(self) -> bool:
        """Check if post is published"""
        return self.status == PostStatus.PUBLISHED
    
    @property
    def word_count(self) -> int:
        """Calculate approximate word count"""
        if not self.content:
            return 0
        return len(self.content.split())
    
    @property
    def reading_time(self) -> int:
        """Estimate reading time in minutes (assuming 200 words per minute)"""
        return max(1, self.word_count // 200)
    
    # Class methods
    @classmethod
    def get_published(cls, session):
        """Get all published posts"""
        return session.query(cls).filter_by(status=PostStatus.PUBLISHED).all()
    
    @classmethod
    def get_by_slug(cls, session, slug: str):
        """Get post by slug"""
        return session.query(cls).filter_by(slug=slug).first()
    
    # Instance methods
    def publish(self):
        """Publish the post"""
        self.status = PostStatus.PUBLISHED
        self.published_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def unpublish(self):
        """Unpublish the post"""
        self.status = PostStatus.DRAFT
        self.published_at = None
        self.updated_at = datetime.utcnow()
    
    def increment_views(self):
        """Increment view count"""
        self.view_count += 1
    
    def increment_likes(self):
        """Increment like count"""
        self.like_count += 1
    
    def generate_slug(self):
        """Generate slug from title"""
        import re
        if not self.title:
            return ""
        
        # Convert to lowercase and replace spaces with hyphens
        slug = re.sub(r'[^\w\s-]', '', self.title.lower())
        slug = re.sub(r'[-\s]+', '-', slug).strip('-')
        
        # Ensure uniqueness would require database check
        self.slug = slug
        return slug

# Comment model with self-referencing relationship
class Comment(BaseModel):
    """Comment model with threading support"""
    __tablename__ = 'comments'
    
    # Foreign keys
    post_id: Mapped[int] = mapped_column(ForeignKey('posts.id'), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False)
    parent_id: Mapped[Optional[int]] = mapped_column(ForeignKey('comments.id'))
    
    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Moderation
    is_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Properties
    @property
    def is_reply(self) -> bool:
        """Check if comment is a reply to another comment"""
        return self.parent_id is not None
    
    @property
    def depth_level(self) -> int:
        """Calculate comment depth level (for threading)"""
        if not self.parent_id:
            return 0
        # This would require recursive query in practice
        return 1
    
    # Methods
    def approve(self):
        """Approve comment"""
        self.is_approved = True
        self.updated_at = datetime.utcnow()
    
    def reject(self):
        """Reject comment"""
        self.is_approved = False
        self.updated_at = datetime.utcnow()
```

### 6.3 Model Validation and Custom Fields

```python
from sqlalchemy import event, CheckConstraint
from sqlalchemy.orm import validates
from typing import Union
import re

class ValidatedUser(BaseModel):
    """User model with comprehensive validation"""
    __tablename__ = 'validated_users'
    
    username: Mapped[str] = mapped_column(String(50), unique=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    age: Mapped[Optional[int]] = mapped_column(Integer)
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    website: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Add database-level constraints
    __table_args__ = (
        CheckConstraint('age >= 13 AND age <= 120', name='valid_age_range'),
        CheckConstraint("email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'", name='valid_email'),
        CheckConstraint("username ~ '^[a-zA-Z0-9_]{3,50}$'", name='valid_username'),
    )
    
    @validates('username')
    def validate_username(self, key, value):
        """Validate username format"""
        if not value:
            raise ValueError("Username is required")
        
        if len(value) < 3:
            raise ValueError("Username must be at least 3 characters long")
        
        if len(value) > 50:
            raise ValueError("Username cannot exceed 50 characters")
        
        if not re.match(r'^[a-zA-Z0-9_]+$', value):
            raise ValueError("Username can only contain letters, numbers, and underscores")
        
        return value.lower()
    
    @validates('email')
    def validate_email(self, key, value):
        """Validate email format"""
        if not value:
            raise ValueError("Email is required")
        
        email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
        if not re.match(email_pattern, value):
            raise ValueError("Invalid email format")
        
        return value.lower()
    
    @validates('age')
    def validate_age(self, key, value):
        """Validate age range"""
        if value is not None:
            if value < 13:
                raise ValueError("Age must be at least 13")
            if value > 120:
                raise ValueError("Age cannot exceed 120")
        
        return value
    
    @validates('phone')
    def validate_phone(self, key, value):
        """Validate phone number format"""
        if value:
            # Remove all non-digit characters for validation
            digits_only = re.sub(r'\D', '', value)
            if len(digits_only) < 10:
                raise ValueError("Phone number must have at least 10 digits")
            
            # Format phone number
            if len(digits_only) == 10:
                return f"({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
            elif len(digits_only) == 11 and digits_only[0] == '1':
                return f"+1 ({digits_only[1:4]}) {digits_only[4:7]}-{digits_only[7:]}"
        
        return value
    
    @validates('website')
    def validate_website(self, key, value):
        """Validate website URL"""
        if value:
            if not value.startswith(('http://', 'https://')):
                value = f'https://{value}'
            
            url_pattern = r'^https?:\/\/(?:[-\w.])+(?:\:[0-9]+)?(?:\/(?:[\w\/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
            if not re.match(url_pattern, value):
                raise ValueError("Invalid website URL")
        
        return value

# Event listeners for additional validation
@event.listens_for(ValidatedUser, 'before_insert')
@event.listens_for(ValidatedUser, 'before_update')
def validate_user_before_save(mapper, connection, target):
    """Additional validation before database operations"""
    
    # Check for duplicate usernames (case-insensitive)
    if target.username:
        existing = connection.execute(
            text("SELECT id FROM validated_users WHERE LOWER(username) = :username AND id != :user_id"),
            {"username": target.username.lower(), "user_id": target.id or 0}
        ).fetchone()
        
        if existing:
            raise ValueError(f"Username '{target.username}' is already taken")
    
    # Check for duplicate emails
    if target.email:
        existing = connection.execute(
            text("SELECT id FROM validated_users WHERE LOWER(email) = :email AND id != :user_id"),
            {"email": target.email.lower(), "user_id": target.id or 0}
        ).fetchone()
        
        if existing:
            raise ValueError(f"Email '{target.email}' is already registered")
```

---

## Chapter 7: Model Definition and Configuration

### 7.1 Advanced Model Configuration

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, declared_attr
from sqlalchemy import String, Integer, DateTime, Boolean, Index
from datetime import datetime

class Base(DeclarativeBase):
    """Enhanced base class with advanced configuration"""
    
    # Global naming conventions
    __mapper_args__ = {
        "eager_defaults": True  # Fetch server defaults immediately after insert
    }

class AuditMixin:
    """Mixin for audit trail functionality"""
    
    @declared_attr
    def created_by(cls) -> Mapped[Optional[int]]:
        return mapped_column(Integer)
    
    @declared_attr
    def updated_by(cls) -> Mapped[Optional[int]]:
        return mapped_column(Integer)
    
    @declared_attr
    def created_at(cls) -> Mapped[datetime]:
        return mapped_column(DateTime, default=datetime.utcnow)
    
    @declared_attr
    def updated_at(cls) -> Mapped[datetime]:
        return mapped_column(
            DateTime, 
            default=datetime.utcnow, 
            onupdate=datetime.utcnow
        )

class SoftDeleteMixin:
    """Mixin for soft delete functionality"""
    
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    
    def soft_delete(self):
        """Soft delete the record"""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
    
    def restore(self):
        """Restore soft deleted record"""
        self.is_deleted = False
        self.deleted_at = None
    
    @property
    def is_active(self) -> bool:
        """Check if record is active (not soft deleted)"""
        return not self.is_deleted

class SlugMixin:
    """Mixin for URL-friendly slugs"""
    
    slug: Mapped[str] = mapped_column(String(250), unique=True)
    
    def generate_slug(self, source_field: str = 'title'):
        """Generate URL-friendly slug from source field"""
        import re
        source_value = getattr(self, source_field, '')
        if not source_value:
            return ""
        
        # Convert to lowercase and replace spaces/special chars with hyphens
        slug = re.sub(r'[^\w\s-]', '', source_value.lower())
        slug = re.sub(r'[-\s]+', '-', slug).strip('-')
        
        # Truncate if too long
        if len(slug) > 245:  # Leave room for uniqueness suffix
            slug = slug[:245]
        
        self.slug = slug
        return slug

# Advanced model with multiple mixins
class Article(Base, AuditMixin, SoftDeleteMixin, SlugMixin):
    """Article model with comprehensive features"""
    __tablename__ = 'articles'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text)
    
    # Table-specific configurations
    __table_args__ = (
        Index('ix_articles_title_active', 'title', 'is_deleted'),
        Index('ix_articles_slug_active', 'slug', 'is_deleted'),
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'}
    )
    
    # Mapper configurations
    __mapper_args__ = {
        'eager_defaults': True,
        'confirm_deleted_rows': False  # For bulk operations
    }
    
    @property
    def is_published(self) -> bool:
        """Check if article is published and active"""
        return not self.is_deleted and bool(self.content)
```

### 7.2 Custom Field Types and Validators

```python
from sqlalchemy import TypeDecorator, String, JSON, Text, event
from sqlalchemy.ext.mutable import MutableDict
import json

class EncryptedType(TypeDecorator):
    """Custom type for encrypted fields"""
    impl = String
    cache_ok = True
    
    def __init__(self, secret_key: str = None, **kwargs):
        self.secret_key = secret_key or "default-secret-key"
        super().__init__(**kwargs)
    
    def process_bind_param(self, value, dialect):
        """Encrypt value before storing"""
        if value is not None:
            # Simple encryption (use proper encryption in production)
            import hashlib
            return hashlib.sha256(f"{self.secret_key}{value}".encode()).hexdigest()
        return value
    
    def process_result_value(self, value, dialect):
        """Value is already encrypted, return as-is"""
        return value

class JSONEncodedDict(TypeDecorator):
    """Custom JSON field with validation"""
    impl = Text
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        """Serialize dict to JSON"""
        if value is not None:
            return json.dumps(value, ensure_ascii=False)
        return value
    
    def process_result_value(self, value, dialect):
        """Deserialize JSON to dict"""
        if value is not None:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

class TagListType(TypeDecorator):
    """Custom type for tag lists"""
    impl = String
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        """Convert list to comma-separated string"""
        if value is not None:
            if isinstance(value, list):
                return ','.join(str(tag).strip() for tag in value if tag)
            return str(value)
        return value
    
    def process_result_value(self, value, dialect):
        """Convert comma-separated string to list"""
        if value is not None:
            return [tag.strip() for tag in value.split(',') if tag.strip()]
        return []

# Model using custom types
class Product(Base):
    """Product model with custom field types"""
    __tablename__ = 'products'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    
    # Custom encrypted field
    secret_data: Mapped[Optional[str]] = mapped_column(
        EncryptedType(secret_key="product-secret")
    )
    
    # Custom JSON field
    attributes: Mapped[dict] = mapped_column(
        JSONEncodedDict, 
        default=dict
    )
    
    # Custom tag list field
    tags: Mapped[list] = mapped_column(TagListType, default=list)
    
    # Mutable JSON field (SQLAlchemy built-in)
    metadata_info: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict
    )
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

# Validation using SQLAlchemy events
@event.listens_for(Product, 'before_insert')
@event.listens_for(Product, 'before_update')
def validate_product(mapper, connection, target):
    """Validate product data before save"""
    
    # Validate name
    if not target.name or len(target.name.strip()) < 3:
        raise ValueError("Product name must be at least 3 characters long")
    
    # Validate attributes
    if target.attributes:
        required_attrs = ['category', 'price']
        for attr in required_attrs:
            if attr not in target.attributes:
                raise ValueError(f"Missing required attribute: {attr}")
    
    # Validate tags
    if target.tags:
        if len(target.tags) > 10:
            raise ValueError("Maximum 10 tags allowed")
        
        for tag in target.tags:
            if len(tag) > 50:
                raise ValueError("Tag length cannot exceed 50 characters")

# Example usage
def demonstrate_custom_types():
    """Demonstrate custom field types"""
    from sqlalchemy.orm import Session
    
    with Session(engine) as session:
        # Create product with custom fields
        product = Product(
            name="Example Product",
            secret_data="sensitive information",
            attributes={
                "category": "electronics",
                "price": 99.99,
                "warranty": "2 years"
            },
            tags=["electronics", "gadgets", "popular"],
            metadata_info={"featured": True, "discount": 10}
        )
        
        session.add(product)
        session.commit()
        
        # Retrieve and verify
        retrieved = session.get(Product, product.id)
        print(f"Name: {retrieved.name}")
        print(f"Encrypted data: {retrieved.secret_data}")  # Shows hash
        print(f"Attributes: {retrieved.attributes}")  # Automatically deserialized
        print(f"Tags: {retrieved.tags}")  # Automatically converted to list
        print(f"Metadata: {retrieved.metadata_info}")  # Mutable dict
```

---

## Chapter 8: Relationships and Associations

### 8.1 Basic Relationships

```python
from sqlalchemy.orm import relationship, backref
from sqlalchemy import ForeignKey, Table
from typing import List

# One-to-Many Relationship
class User(Base):
    """User model with relationships"""
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    email: Mapped[str] = mapped_column(String(100), unique=True)
    
    # One-to-Many: One user has many posts
    posts: Mapped[List["Post"]] = relationship(
        "Post",
        back_populates="author",
        cascade="all, delete-orphan",  # Delete posts when user is deleted
        lazy="dynamic"  # Load as query object, not list
    )
    
    # One-to-Many: One user has many comments
    comments: Mapped[List["Comment"]] = relationship(
        "Comment",
        back_populates="user",
        cascade="all, delete-orphan"
    )

class Post(Base):
    """Post model with relationships"""
    __tablename__ = 'posts'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[Optional[str]] = mapped_column(Text)
    
    # Foreign key to user
    author_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    
    # Many-to-One: Many posts belong to one user
    author: Mapped["User"] = relationship(
        "User",
        back_populates="posts"
    )
    
    # One-to-Many: One post has many comments
    comments: Mapped[List["Comment"]] = relationship(
        "Comment",
        back_populates="post",
        cascade="all, delete-orphan",
        order_by="Comment.created_at"  # Order comments by creation time
    )

class Comment(Base):
    """Comment model with multiple relationships"""
    __tablename__ = 'comments'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Foreign keys
    post_id: Mapped[int] = mapped_column(ForeignKey('posts.id'))
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    parent_id: Mapped[Optional[int]] = mapped_column(ForeignKey('comments.id'))
    
    # Many-to-One relationships
    post: Mapped["Post"] = relationship("Post", back_populates="comments")
    user: Mapped["User"] = relationship("User", back_populates="comments")
    
    # Self-referential relationship for comment threading
    parent: Mapped[Optional["Comment"]] = relationship(
        "Comment",
        back_populates="replies",
        remote_side="Comment.id"  # Specify the remote side for self-reference
    )
    
    replies: Mapped[List["Comment"]] = relationship(
        "Comment",
        back_populates="parent",
        cascade="all, delete-orphan"
    )
```

### 8.2 Many-to-Many Relationships

```python
# Association table for Many-to-Many relationship
post_tags_association = Table(
    'post_tags',
    Base.metadata,
    Column('post_id', Integer, ForeignKey('posts.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True),
    # Additional columns in association table
    Column('created_at', DateTime, default=datetime.utcnow),
    Column('created_by', Integer, ForeignKey('users.id'))
)

class Tag(Base):
    """Tag model for Many-to-Many relationship"""
    __tablename__ = 'tags'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    color: Mapped[Optional[str]] = mapped_column(String(7))  # Hex color code
    
    # Many-to-Many: Tags can be on many posts, posts can have many tags
    posts: Mapped[List["Post"]] = relationship(
        "Post",
        secondary=post_tags_association,
        back_populates="tags",
        lazy="dynamic"
    )

# Update Post model to include tags relationship
class Post(Base):
    __tablename__ = 'posts'
    
    # ... existing fields ...
    
    # Many-to-Many relationship with tags
    tags: Mapped[List["Tag"]] = relationship(
        "Tag",
        secondary=post_tags_association,
        back_populates="posts",
        lazy="subquery"  # Load tags with posts in single query
    )

# Advanced Many-to-Many with Association Object
class UserRole(Base):
    """Association object for User-Role Many-to-Many relationship"""
    __tablename__ = 'user_roles'
    
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), primary_key=True)
    role_id: Mapped[int] = mapped_column(ForeignKey('roles.id'), primary_key=True)
    
    # Additional fields in association
    assigned_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    assigned_by: Mapped[int] = mapped_column(ForeignKey('users.id'))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships to parent objects
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    role: Mapped["Role"] = relationship("Role")
    assigner: Mapped["User"] = relationship("User", foreign_keys=[assigned_by])

class Role(Base):
    """Role model"""
    __tablename__ = 'roles'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    permissions: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Association object relationship
    users: Mapped[List["UserRole"]] = relationship("UserRole", back_populates="role")

# Update User model for role relationship
class User(Base):
    __tablename__ = 'users'
    
    # ... existing fields ...
    
    # Association object relationship
    roles: Mapped[List["UserRole"]] = relationship(
        "UserRole",
        back_populates="user",
        foreign_keys="UserRole.user_id"
    )
    
    # Convenience property to get active roles
    @property
    def active_roles(self) -> List["Role"]:
        """Get user's active roles"""
        return [
            user_role.role for user_role in self.roles 
            if user_role.is_active and (
                user_role.expires_at is None or 
                user_role.expires_at > datetime.utcnow()
            )
        ]
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has specific role"""
        return any(role.name == role_name for role in self.active_roles)
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        for role in self.active_roles:
            if role.permissions.get(permission, False):
                return True
        return False
```

### 8.3 Advanced Relationship Patterns

```python
from sqlalchemy.orm import relationship, backref, selectinload, joinedload
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import select, func

class Category(Base):
    """Hierarchical category model with self-referential relationship"""
    __tablename__ = 'categories'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    slug: Mapped[str] = mapped_column(String(120), unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Self-referential foreign key
    parent_id: Mapped[Optional[int]] = mapped_column(ForeignKey('categories.id'))
    
    # Self-referential relationships
    parent: Mapped[Optional["Category"]] = relationship(
        "Category",
        back_populates="children",
        remote_side="Category.id"
    )
    
    children: Mapped[List["Category"]] = relationship(
        "Category",
        back_populates="parent",
        cascade="all, delete-orphan"
    )
    
    # Relationship to products
    products: Mapped[List["Product"]] = relationship(
        "Product",
        back_populates="category",
        lazy="dynamic"
    )
    
    @property
    def path(self) -> List["Category"]:
        """Get path from root to this category"""
        path = []
        current = self
        while current:
            path.insert(0, current)
            current = current.parent
        return path
    
    @property
    def path_names(self) -> List[str]:
        """Get path names as list"""
        return [cat.name for cat in self.path]
    
    @property
    def full_path(self) -> str:
        """Get full path as string"""
        return " > ".join(self.path_names)
    
    def get_descendants(self, session) -> List["Category"]:
        """Get all descendant categories recursively"""
        descendants = []
        
        def _collect_descendants(category):
            for child in category.children:
                descendants.append(child)
                _collect_descendants(child)
        
        _collect_descendants(self)
        return descendants
    
    @hybrid_property
    def product_count(self):
        """Count products in this category"""
        return self.products.count()
    
    @product_count.expression
    def product_count(cls):
        """SQL expression for product count"""
        return (
            select(func.count(Product.id))
            .where(Product.category_id == cls.id)
            .scalar_subquery()
        )

class Product(Base):
    """Product model with advanced relationships"""
    __tablename__ = 'products'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    price: Mapped[decimal.Decimal] = mapped_column(Numeric(10, 2))
    
    # Foreign key to category
    category_id: Mapped[Optional[int]] = mapped_column(ForeignKey('categories.id'))
    
    # Many-to-One relationship
    category: Mapped[Optional["Category"]] = relationship(
        "Category",
        back_populates="products"
    )

# Polymorphic relationships
class Contact(Base):
    """Base contact model using polymorphism"""
    __tablename__ = 'contacts'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(100))
    type: Mapped[str] = mapped_column(String(50))  # Discriminator column
    
    __mapper_args__ = {
        'polymorphic_identity': 'contact',
        'polymorphic_on': type
    }

class Customer(Contact):
    """Customer model inheriting from Contact"""
    __tablename__ = 'customers'
    
    id: Mapped[int] = mapped_column(ForeignKey('contacts.id'), primary_key=True)
    credit_limit: Mapped[decimal.Decimal] = mapped_column(Numeric(10, 2))
    
    # One-to-Many relationship
    orders: Mapped[List["Order"]] = relationship("Order", back_populates="customer")
    
    __mapper_args__ = {
        'polymorphic_identity': 'customer',
    }

class Supplier(Contact):
    """Supplier model inheriting from Contact"""
    __tablename__ = 'suppliers'
    
    id: Mapped[int] = mapped_column(ForeignKey('contacts.id'), primary_key=True)
    payment_terms: Mapped[str] = mapped_column(String(100))
    
    # Many-to-Many relationship with products
    products: Mapped[List["Product"]] = relationship(
        "Product",
        secondary="product_suppliers",
        back_populates="suppliers"
    )
    
    __mapper_args__ = {
        'polymorphic_identity': 'supplier',
    }

class Order(Base):
    """Order model with polymorphic relationship"""
    __tablename__ = 'orders'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    total: Mapped[decimal.Decimal] = mapped_column(Numeric(10, 2))
    
    # Foreign key to customer
    customer_id: Mapped[int] = mapped_column(ForeignKey('customers.id'))
    
    # Relationship to customer (which is polymorphic)
    customer: Mapped["Customer"] = relationship("Customer", back_populates="orders")
```

### 8.4 Relationship Loading Strategies

```python
from sqlalchemy.orm import selectinload, joinedload, subqueryload, contains_eager

class RelationshipLoader:
    """Utility class for demonstrating relationship loading strategies"""
    
    def __init__(self, session):
        self.session = session
    
    def lazy_loading_example(self):
        """Lazy loading (default) - load related objects on access"""
        # This will execute one query for users
        users = self.session.query(User).all()
        
        # This will execute additional queries for each user's posts (N+1 problem)
        for user in users:
            print(f"User {user.username} has {len(user.posts)} posts")
    
    def eager_loading_joinedload(self):
        """Joined loading - single query with LEFT OUTER JOIN"""
        users = (
            self.session.query(User)
            .options(joinedload(User.posts))
            .all()
        )
        
        # No additional queries needed
        for user in users:
            print(f"User {user.username} has {len(user.posts)} posts")
    
    def eager_loading_selectinload(self):
        """Select IN loading - separate optimized query"""
        users = (
            self.session.query(User)
            .options(selectinload(User.posts))
            .all()
        )
        
        # Two queries total: one for users, one for all related posts
        for user in users:
            print(f"User {user.username} has {len(user.posts)} posts")
    
    def eager_loading_subqueryload(self):
        """Subquery loading - uses subquery to fetch related objects"""
        users = (
            self.session.query(User)
            .options(subqueryload(User.posts))
            .all()
        )
        
        for user in users:
            print(f"User {user.username} has {len(user.posts)} posts")
    
    def nested_loading(self):
        """Load nested relationships"""
        posts = (
            self.session.query(Post)
            .options(
                joinedload(Post.author),  # Load author with posts
                selectinload(Post.comments).selectinload(Comment.user)  # Load comments and their users
            )
            .all()
        )
        
        for post in posts:
            print(f"Post: {post.title} by {post.author.username}")
            for comment in post.comments:
                print(f"  Comment by {comment.user.username}: {comment.content[:50]}...")
    
    def conditional_loading(self):
        """Load relationships conditionally"""
        from sqlalchemy import and_
        
        # Load only published posts for users
        users = (
            self.session.query(User)
            .options(
                selectinload(User.posts).where(Post.status == 'published')
            )
            .all()
        )
        
        for user in users:
            print(f"User {user.username} has {len(user.posts)} published posts")
    
    def contains_eager_example(self):
        """Use contains_eager for manual joins"""
        # Manual JOIN with contains_eager
        query = (
            self.session.query(User)
            .join(User.posts)
            .options(contains_eager(User.posts))
            .filter(Post.status == 'published')
        )
        
        users_with_published_posts = query.all()
        
        for user in users_with_published_posts:
            print(f"User {user.username} with published posts:")
            for post in user.posts:
                print(f"  - {post.title}")

# Example usage with different loading strategies
def demonstrate_loading_strategies():
    """Demonstrate different loading strategies"""
    from sqlalchemy.orm import Session
    
    with Session(engine) as session:
        loader = RelationshipLoader(session)
        
        print("=== Lazy Loading ===")
        loader.lazy_loading_example()
        
        print("\n=== Joined Loading ===")
        loader.eager_loading_joinedload()
        
        print("\n=== Select IN Loading ===")
        loader.eager_loading_selectinload()
        
        print("\n=== Nested Loading ===")
        loader.nested_loading()
        
        print("\n=== Conditional Loading ===")
        loader.conditional_loading()
```

---

## Chapter 9: Querying with the ORM

### 9.1 Basic Querying

```python
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, not_, func, desc, asc, case, cast, Date

# Basic query operations
class QueryExamples:
    """Comprehensive querying examples"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def basic_queries(self):
        """Basic query operations"""
        
        # Get all users
        all_users = self.session.query(User).all()
        
        # Get first user
        first_user = self.session.query(User).first()
        
        # Get user by primary key
        user_by_id = self.session.get(User, 1)
        
        # Get one user (raises exception if not found or multiple found)
        try:
            one_user = self.session.query(User).filter(User.email == "test@example.com").one()
        except NoResultFound:
            print("No user found with that email")
        except MultipleResultsFound:
            print("Multiple users found with that email")
        
        # Get one or none
        user_or_none = self.session.query(User).filter(User.email == "test@example.com").one_or_none()
        
        # Count queries
        user_count = self.session.query(User).count()
        active_user_count = self.session.query(User).filter(User.is_active == True).count()
        
        return {
            'all_users': all_users,
            'first_user': first_user,
            'user_by_id': user_by_id,
            'user_count': user_count,
            'active_user_count': active_user_count
        }
    
    def filtering_examples(self):
        """Various filtering examples"""
        
        # Basic equality filtering
        active_users = self.session.query(User).filter(User.is_active == True).all()
        
        # Multiple conditions (AND)
        admin_users = (
            self.session.query(User)
            .filter(User.is_active == True)
            .filter(User.role == UserRole.ADMIN)
            .all()
        )
        
        # Using and_() explicitly
        active_admins = (
            self.session.query(User)
            .filter(and_(User.is_active == True, User.role == UserRole.ADMIN))
            .all()
        )
        
        # OR conditions
        admin_or_moderator = (
            self.session.query(User)
            .filter(or_(User.role == UserRole.ADMIN, User.role == UserRole.MODERATOR))
            .all()
        )
        
        # NOT conditions
        non_admin_users = (
            self.session.query(User)
            .filter(not_(User.role == UserRole.ADMIN))
            .all()
        )
        
        # IN clause
        specific_users = (
            self.session.query(User)
            .filter(User.id.in_([1, 2, 3, 4, 5]))
            .all()
        )
        
        # LIKE patterns
        users_with_gmail = (
            self.session.query(User)
            .filter(User.email.like('%gmail.com'))
            .all()
        )
        
        # ILIKE (case-insensitive LIKE)
        users_named_john = (
            self.session.query(User)
            .filter(User.first_name.ilike('john%'))
            .all()
        )
        
        # IS NULL / IS NOT NULL
        users_without_last_name = (
            self.session.query(User)
            .filter(User.last_name.is_(None))
            .all()
        )
        
        users_with_last_name = (
            self.session.query(User)
            .filter(User.last_name.isnot(None))
            .all()
        )
        
        # Comparison operators
        recent_users = (
            self.session.query(User)
            .filter(User.created_at >= datetime(2024, 1, 1))
            .all()
        )
        
        return {
            'active_users': active_users,
            'admin_users': admin_users,
            'admin_or_moderator': admin_or_moderator,
            'users_with_gmail': users_with_gmail,
            'recent_users': recent_users
        }
    
    def ordering_examples(self):
        """Examples of ordering results"""
        
        # Simple ordering
        users_by_username = (
            self.session.query(User)
            .order_by(User.username)
            .all()
        )
        
        # Descending order
        users_by_created_desc = (
            self.session.query(User)
            .order_by(desc(User.created_at))
            .all()
        )
        
        # Multiple ordering criteria
        users_ordered = (
            self.session.query(User)
            .order_by(User.role, desc(User.created_at), User.username)
            .all()
        )
        
        # Ordering by computed values
        users_by_name_length = (
            self.session.query(User)
            .order_by(func.length(User.username))
            .all()
        )
        
        # Case-based ordering
        users_custom_order = (
            self.session.query(User)
            .order_by(
                case(
                    (User.role == UserRole.ADMIN, 1),
                    (User.role == UserRole.MODERATOR, 2),
                    else_=3
                ),
                User.username
            )
            .all()
        )
        
        return {
            'users_by_username': users_by_username,
            'users_by_created_desc': users_by_created_desc,
            'users_ordered': users_ordered,
            'users_custom_order': users_custom_order
        }
    
    def pagination_examples(self):
        """Examples of pagination"""
        
        # Basic limit and offset
        page_size = 10
        page_number = 1  # 1-indexed
        offset = (page_number - 1) * page_size
        
        paginated_users = (
            self.session.query(User)
            .order_by(User.created_at.desc())
            .offset(offset)
            .limit(page_size)
            .all()
        )
        
        # Get total count for pagination info
        total_users = self.session.query(User).count()
        total_pages = (total_users + page_size - 1) // page_size  # Ceiling division
        
        # Cursor-based pagination (more efficient for large datasets)
        last_id = 100  # From previous page
        cursor_paginated = (
            self.session.query(User)
            .filter(User.id > last_id)
            .order_by(User.id)
            .limit(page_size)
            .all()
        )
        
        return {
            'paginated_users': paginated_users,
            'total_users': total_users,
            'total_pages': total_pages,
            'cursor_paginated': cursor_paginated
        }

def demonstrate_basic_queries():
    """Demonstrate basic querying functionality"""
    with Session(engine) as session:
        query_examples = QueryExamples(session)
        
        # Basic queries
        basic_results = query_examples.basic_queries()
        print(f"Total users: {basic_results['user_count']}")
        print(f"Active users: {basic_results['active_user_count']}")
        
        # Filtering
        filter_results = query_examples.filtering_examples()
        print(f"Admin users: {len(filter_results['admin_users'])}")
        
        # Ordering
        order_results = query_examples.ordering_examples()
        print(f"First user by username: {order_results['users_by_username'][0].username if order_results['users_by_username'] else 'None'}")
        
        # Pagination
        page_results = query_examples.pagination_examples()
        print(f"Page 1 users: {len(page_results['paginated_users'])}")
```

### 9.2 Advanced Query Techniques

```python
from sqlalchemy import text, literal, bindparam, union, union_all, except_, intersect
from sqlalchemy.orm import aliased

class AdvancedQueryExamples:
    """Advanced querying techniques"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def aggregate_queries(self):
        """Aggregation and grouping examples"""
        
        # Count posts per user
        posts_per_user = (
            self.session.query(User.username, func.count(Post.id).label('post_count'))
            .outerjoin(User.posts)
            .group_by(User.id, User.username)
            .all()
        )
        
        # Average, min, max
        post_stats = (
            self.session.query(
                func.count(Post.id).label('total_posts'),
                func.avg(func.length(Post.content)).label('avg_content_length'),
                func.min(Post.created_at).label('first_post'),
                func.max(Post.created_at).label('latest_post')
            )
            .first()
        )
        
        # Group by with having clause
        prolific_users = (
            self.session.query(User.username, func.count(Post.id).label('post_count'))
            .join(User.posts)
            .group_by(User.id, User.username)
            .having(func.count(Post.id) > 5)
            .order_by(desc('post_count'))
            .all()
        )
        
        # Complex aggregations with multiple tables
        user_engagement = (
            self.session.query(
                User.username,
                func.count(Post.id).label('posts'),
                func.count(Comment.id).label('comments'),
                func.coalesce(func.sum(Post.view_count), 0).label('total_views')
            )
            .outerjoin(User.posts)
            .outerjoin(User.comments)
            .group_by(User.id, User.username)
            .all()
        )
        
        return {
            'posts_per_user': posts_per_user,
            'post_stats': post_stats,
            'prolific_users': prolific_users,
            'user_engagement': user_engagement
        }
    
    def subquery_examples(self):
        """Subquery examples"""
        
        # Scalar subquery
        avg_post_count = (
            self.session.query(func.count(Post.id) / func.count(func.distinct(Post.author_id)))
            .scalar_subquery()
        )
        
        users_above_average = (
            self.session.query(User)
            .join(User.posts)
            .group_by(User.id)
            .having(func.count(Post.id) > avg_post_count)
            .all()
        )
        
        # Correlated subquery
        users_with_recent_posts = (
            self.session.query(User)
            .filter(
                self.session.query(Post.id)
                .filter(Post.author_id == User.id)
                .filter(Post.created_at >= datetime.now() - timedelta(days=30))
                .exists()
            )
            .all()
        )
        
        # Subquery with IN
        active_post_authors = (
            self.session.query(User.id)
            .join(User.posts)
            .filter(Post.status == PostStatus.PUBLISHED)
            .subquery()
        )
        
        active_users = (
            self.session.query(User)
            .filter(User.id.in_(select(active_post_authors.c.id)))
            .all()
        )
        
        return {
            'users_above_average': users_above_average,
            'users_with_recent_posts': users_with_recent_posts,
            'active_users': active_users
        }
    
    def window_function_examples(self):
        """Window function examples (PostgreSQL, SQL Server, etc.)"""
        
        # Row number
        users_with_row_numbers = (
            self.session.query(
                User.username,
                User.created_at,
                func.row_number().over(order_by=User.created_at).label('row_num')
            )
            .all()
        )
        
        # Rank users by post count
        user_post_rankings = (
            self.session.query(
                User.username,
                func.count(Post.id).label('post_count'),
                func.rank().over(order_by=desc(func.count(Post.id))).label('rank'),
                func.dense_rank().over(order_by=desc(func.count(Post.id))).label('dense_rank')
            )
            .outerjoin(User.posts)
            .group_by(User.id, User.username)
            .all()
        )
        
        # Percentile calculations
        user_percentiles = (
            self.session.query(
                User.username,
                func.count(Post.id).label('post_count'),
                func.percent_rank().over(order_by=func.count(Post.id)).label('percentile')
            )
            .outerjoin(User.posts)
            .group_by(User.id, User.username)
            .all()
        )
        
        # Moving averages
        daily_post_stats = (
            self.session.query(
                func.date_trunc('day', Post.created_at).label('day'),
                func.count(Post.id).label('posts_count'),
                func.avg(func.count(Post.id)).over(
                    order_by=func.date_trunc('day', Post.created_at),
                    rows=(6, 0)  # 7-day moving average
                ).label('moving_avg')
            )
            .group_by(func.date_trunc('day', Post.created_at))
            .order_by('day')
            .all()
        )
        
        return {
            'users_with_row_numbers': users_with_row_numbers,
            'user_post_rankings': user_post_rankings,
            'user_percentiles': user_percentiles,
            'daily_post_stats': daily_post_stats
        }
    
    def cte_examples(self):
        """Common Table Expression (CTE) examples"""
        
        # Recursive CTE for hierarchical data
        def get_category_hierarchy(self, root_category_id):
            """Get category hierarchy using recursive CTE"""
            
            # Base case: root category
            base_query = (
                self.session.query(
                    Category.id,
                    Category.name,
                    Category.parent_id,
                    literal(0).label('level'),
                    Category.name.label('path')
                )
                .filter(Category.id == root_category_id)
            )
            
            # Recursive case: children
            recursive_query = (
                self.session.query(
                    Category.id,
                    Category.name,
                    Category.parent_id,
                    (base_query.c.level + 1).label('level'),
                    (base_query.c.path + ' > ' + Category.name).label('path')
                )
                .join(base_query, Category.parent_id == base_query.c.id)
            )
            
            # Note: Actual recursive CTE syntax varies by database
            # This is a simplified example - real implementation would use database-specific syntax
            
            return base_query.union_all(recursive_query)
        
        # Non-recursive CTE
        monthly_stats_cte = (
            self.session.query(
                func.date_trunc('month', Post.created_at).label('month'),
                func.count(Post.id).label('post_count'),
                func.count(func.distinct(Post.author_id)).label('author_count')
            )
            .group_by(func.date_trunc('month', Post.created_at))
            .cte('monthly_stats')
        )
        
        monthly_analysis = (
            self.session.query(
                monthly_stats_cte.c.month,
                monthly_stats_cte.c.post_count,
                monthly_stats_cte.c.author_count,
                (monthly_stats_cte.c.post_count / monthly_stats_cte.c.author_count).label('posts_per_author')
            )
            .order_by(monthly_stats_cte.c.month)
            .all()
        )
        
        return {
            'monthly_analysis': monthly_analysis
        }
    
    def union_examples(self):
        """UNION and set operation examples"""
        
        # UNION - combine results from multiple queries
        active_entities = (
            self.session.query(User.id, User.username, literal('user').label('type'))
            .filter(User.is_active == True)
            .union(
                self.session.query(Post.id, Post.title, literal('post').label('type'))
                .filter(Post.status == PostStatus.PUBLISHED)
            )
            .all()
        )
        
        # UNION ALL - include duplicates
        all_names = (
            self.session.query(User.username.label('name'))
            .union_all(
                self.session.query(Post.title.label('name'))
            )
            .all()
        )
        
        # EXCEPT - items in first query but not second
        users_without_posts = (
            self.session.query(User.id)
            .except_(
                self.session.query(User.id).join(User.posts)
            )
            .all()
        )
        
        # INTERSECT - items common to both queries
        common_words = (
            self.session.query(func.unnest(func.string_to_array(Post.title, ' ')).label('word'))
            .intersect(
                self.session.query(func.unnest(func.string_to_array(Post.content, ' ')).label('word'))
            )
            .all()
        )
        
        return {
            'active_entities': active_entities,
            'all_names': all_names,
            'users_without_posts': users_without_posts
        }
    
    def raw_sql_examples(self):
        """Raw SQL integration examples"""
        
        # Simple raw SQL
        result = self.session.execute(
            text("SELECT COUNT(*) as user_count FROM users WHERE is_active = :active"),
            {"active": True}
        )
        user_count = result.fetchone().user_count
        
        # Raw SQL with ORM objects
        users = self.session.execute(
            text("SELECT * FROM users WHERE created_at > :date ORDER BY username"),
            {"date": datetime(2024, 1, 1)}
        ).fetchall()
        
        # Raw SQL returning ORM objects
        user_objects = (
            self.session.query(User)
            .from_statement(
                text("SELECT * FROM users WHERE email LIKE :pattern")
            )
            .params(pattern="%@gmail.com")
            .all()
        )
        
        # Complex raw SQL with parameters
        complex_stats = self.session.execute(
            text("""
                WITH user_stats AS (
                    SELECT 
                        u.id,
                        u.username,
                        COUNT(p.id) as post_count,
                        AVG(p.view_count) as avg_views
                    FROM users u
                    LEFT JOIN posts p ON u.id = p.author_id
                    WHERE u.created_at >= :start_date
                    GROUP BY u.id, u.username
                )
                SELECT 
                    username,
                    post_count,
                    avg_views,
                    CASE 
                        WHEN post_count > 10 THEN 'prolific'
                        WHEN post_count > 5 THEN 'active'
                        ELSE 'casual'
                    END as user_type
                FROM user_stats
                ORDER BY post_count DESC
            """),
            {"start_date": datetime(2024, 1, 1)}
        ).fetchall()
        
        return {
            'user_count': user_count,
            'raw_users': users,
            'user_objects': user_objects,
            'complex_stats': complex_stats
        }

def demonstrate_advanced_queries():
    """Demonstrate advanced querying techniques"""
    with Session(engine) as session:
        advanced_queries = AdvancedQueryExamples(session)
        
        # Aggregations
        agg_results = advanced_queries.aggregate_queries()
        print(f"Post statistics: {agg_results['post_stats']}")
        
        # Subqueries
        sub_results = advanced_queries.subquery_examples()
        print(f"Users above average posts: {len(sub_results['users_above_average'])}")
        
        # Window functions
        window_results = advanced_queries.window_function_examples()
        print(f"Top user by posts: {window_results['user_post_rankings'][0] if window_results['user_post_rankings'] else 'None'}")
        
        # Raw SQL
        raw_results = advanced_queries.raw_sql_examples()
        print(f"Active user count (raw SQL): {raw_results['user_count']}")
```

---

## Chapter 10: Sessions and Transactions

### 10.1 Session Management

```python
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.orm.exc import DetachedInstanceError
from contextlib import contextmanager
from threading import local
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SessionManager:
    """Advanced session management class"""
    
    def __init__(self, engine, **kwargs):
        self.engine = engine
        self.session_factory = sessionmaker(bind=engine, **kwargs)
        
        # Thread-local session for web applications
        self.scoped_session_factory = scoped_session(self.session_factory)
        
    @contextmanager
    def get_session(self):
        """Context manager for session with automatic cleanup"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    @contextmanager
    def get_transaction(self):
        """Context manager for explicit transaction control"""
        session = self.session_factory()
        try:
            # Begin transaction explicitly
            transaction = session.begin()
            yield session
            transaction.commit()
        except Exception as e:
            transaction.rollback()
            logger.error(f"Transaction error: {e}")
            raise
        finally:
            session.close()
    
    def get_scoped_session(self):
        """Get thread-local session (for web applications)"""
        return self.scoped_session_factory()
    
    def remove_scoped_session(self):
        """Remove thread-local session"""
        self.scoped_session_factory.remove()

# Global session manager instance
session_manager = SessionManager(
    engine,
    expire_on_commit=False,  # Don't expire objects after commit
    autoflush=True,          # Automatically flush before queries
    autocommit=False         # Don't auto-commit transactions
)

class SessionOperations:
    """Examples of session operations and lifecycle management"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def object_lifecycle_examples(self):
        """Demonstrate object lifecycle in sessions"""
        
        # Create new object (transient state)
        new_user = User(
            username="newuser",
            email="newuser@example.com",
            password_hash="hashed_password"
        )
        
        logger.info(f"New user state: {inspect(new_user).transient}")  # True
        
        # Add to session (persistent state)
        self.session.add(new_user)
        logger.info(f"User after add - pending: {inspect(new_user).pending}")  # True
        
        # Flush to database (still in transaction)
        self.session.flush()
        logger.info(f"User after flush - persistent: {inspect(new_user).persistent}")  # True
        logger.info(f"User ID after flush: {new_user.id}")  # Now has ID
        
        # Commit transaction
        self.session.commit()
        logger.info(f"User after commit - persistent: {inspect(new_user).persistent}")  # True
        
        # Detach object from session
        self.session.expunge(new_user)
        logger.info(f"User after expunge - detached: {inspect(new_user).detached}")  # True
        
        # Trying to access relationships on detached object will raise error
        try:
            posts = new_user.posts  # This will raise DetachedInstanceError
        except DetachedInstanceError as e:
            logger.warning(f"Detached instance error: {e}")
        
        # Merge detached object back into session
        merged_user = self.session.merge(new_user)
        logger.info(f"Merged user - persistent: {inspect(merged_user).persistent}")  # True
        
        return new_user
    
    def state_management_examples(self):
        """Examples of object state management"""
        
        # Load existing user
        user = self.session.get(User, 1)
        if not user:
            return None
        
        # Make changes (dirty state)
        original_username = user.username
        user.username = "modified_username"
        logger.info(f"User is dirty: {inspect(user).modified}")  # True
        
        # Check which attributes are modified
        logger.info(f"Modified attributes: {inspect(user).attrs.username.history}")
        
        # Refresh object from database (loses changes)
        self.session.refresh(user)
        logger.info(f"Username after refresh: {user.username}")  # Original value
        
        # Make changes again
        user.username = "new_username"
        
        # Expunge and merge pattern for detached updates
        self.session.expunge(user)
        user.email = "updated@example.com"  # Modify detached object
        
        merged_user = self.session.merge(user)  # Merge changes back
        self.session.commit()
        
        return merged_user
    
    def bulk_operations(self):
        """Bulk operations for performance"""
        
        # Bulk insert
        users_data = [
            {"username": f"user_{i}", "email": f"user_{i}@example.com", "password_hash": "hash"}
            for i in range(1000, 1100)
        ]
        
        # Bulk insert with Core
        self.session.execute(
            insert(User.__table__),
            users_data
        )
        
        # Bulk update
        self.session.query(User).filter(
            User.username.like('user_%')
        ).update(
            {"is_verified": True},
            synchronize_session=False  # Don't synchronize with session
        )
        
        # Bulk delete
        deleted_count = self.session.query(User).filter(
            User.username.like('user_%')
        ).delete(synchronize_session=False)
        
        logger.info(f"Deleted {deleted_count} users")
        
        return deleted_count
    
    def identity_map_examples(self):
        """Demonstrate session identity map"""
        
        # Load same object multiple times - should be same instance
        user1 = self.session.get(User, 1)
        user2 = self.session.get(User, 1)
        
        logger.info(f"Same object instance: {user1 is user2}")  # True
        
        # Query same object - still same instance
        user3 = self.session.query(User).filter(User.id == 1).first()
        logger.info(f"Query returns same instance: {user1 is user3}")  # True
        
        # After expunge, new queries create new instances
        self.session.expunge(user1)
        user4 = self.session.get(User, 1)
        logger.info(f"After expunge, new instance: {user1 is user4}")  # False
        
        return user1, user4

def demonstrate_session_management():
    """Demonstrate session management patterns"""
    
    # Context manager pattern
    with session_manager.get_session() as session:
        ops = SessionOperations(session)
        user = ops.object_lifecycle_examples()
        logger.info(f"Created user: {user.username}")
    
    # Transaction pattern
    with session_manager.get_transaction() as session:
        ops = SessionOperations(session)
        modified_user = ops.state_management_examples()
        if modified_user:
            logger.info(f"Modified user: {modified_user.username}")
    
    # Manual session management
    session = Session(engine)
    try:
        ops = SessionOperations(session)
        ops.bulk_operations()
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Bulk operation failed: {e}")
    finally:
        session.close()
```

### 10.2 Transaction Management

```python
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, DataError
import time

class TransactionManager:
    """Advanced transaction management patterns"""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
    
    def nested_transactions_example(self):
        """Demonstrate nested transactions with savepoints"""
        
        with self.session_manager.get_session() as session:
            # Start main transaction
            user = User(username="main_user", email="main@example.com", password_hash="hash")
            session.add(user)
            session.flush()  # Get user ID
            
            # Create savepoint
            savepoint = session.begin_nested()
            
            try:
                # Nested transaction work
                post1 = Post(title="Post 1", author_id=user.id)
                post2 = Post(title="Post 2", author_id=user.id)
                
                session.add_all([post1, post2])
                session.flush()
                
                # Simulate error in nested transaction
                if post1.id % 2 == 0:  # Arbitrary condition for demo
                    raise ValueError("Simulated error in nested transaction")
                
                savepoint.commit()  # Commit savepoint
                logger.info("Nested transaction committed")
                
            except Exception as e:
                savepoint.rollback()  # Rollback to savepoint
                logger.error(f"Nested transaction rolled back: {e}")
                
                # Continue with main transaction
                post3 = Post(title="Recovery Post", author_id=user.id)
                session.add(post3)
            
            # Main transaction completes
            logger.info("Main transaction completed")
    
    def retry_transaction_pattern(self, max_retries: int = 3):
        """Transaction with retry logic for deadlocks/conflicts"""
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                with self.session_manager.get_transaction() as session:
                    # Simulate work that might cause conflicts
                    user = session.query(User).filter(User.id == 1).with_for_update().first()
                    if user:
                        user.login_count += 1
                        time.sleep(0.1)  # Simulate processing time
                        session.flush()
                
                logger.info("Transaction completed successfully")
                return True
                
            except (IntegrityError, DataError) as e:
                retry_count += 1
                logger.warning(f"Transaction failed (attempt {retry_count}): {e}")
                
                if retry_count >= max_retries:
                    logger.error("Max retries reached, transaction failed")
                    raise
                
                # Exponential backoff
                wait_time = (2 ** retry_count) * 0.1
                time.sleep(wait_time)
        
        return False
    
    def distributed_transaction_pattern(self):
        """Pattern for distributed transactions (simplified)"""
        
        # Phase 1: Prepare all resources
        session1 = self.session_manager.session_factory()
        session2 = self.session_manager.session_factory()
        
        prepared_sessions = []
        
        try:
            # Prepare first session
            user = User(username="dist_user1", email="dist1@example.com", password_hash="hash")
            session1.add(user)
            session1.flush()
            prepared_sessions.append(session1)
            
            # Prepare second session
            user2 = User(username="dist_user2", email="dist2@example.com", password_hash="hash")
            session2.add(user2)
            session2.flush()
            prepared_sessions.append(session2)
            
            # Phase 2: Commit all (simplified - real distributed transactions are more complex)
            for session in prepared_sessions:
                session.commit()
            
            logger.info("Distributed transaction completed")
            return True
            
        except Exception as e:
            # Rollback all sessions
            for session in prepared_sessions:
                try:
                    session.rollback()
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")
            
            logger.error(f"Distributed transaction failed: {e}")
            return False
            
        finally:
            # Clean up sessions
            for session in [session1, session2]:
                session.close()
    
    def compensation_transaction_pattern(self):
        """Compensation pattern for saga transactions"""
        
        compensations = []
        
        try:
            # Step 1: Create user
            with self.session_manager.get_session() as session:
                user = User(username="saga_user", email="saga@example.com", password_hash="hash")
                session.add(user)
                session.flush()
                user_id = user.id
                
                # Add compensation for user creation
                compensations.append(lambda: self._delete_user(user_id))
            
            # Step 2: Create user profile
            with self.session_manager.get_session() as session:
                user = session.get(User, user_id)
                user.bio = "This is a saga user"
                session.flush()
                
                # Add compensation for profile update
                compensations.append(lambda: self._clear_user_bio(user_id))
            
            # Step 3: Send welcome email (simulated)
            # This step might fail
            if True:  # Simulate failure
                raise Exception("Email service unavailable")
            
            logger.info("Saga transaction completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Saga transaction failed: {e}")
            
            # Execute compensations in reverse order
            for compensation in reversed(compensations):
                try:
                    compensation()
                except Exception as comp_error:
                    logger.error(f"Compensation failed: {comp_error}")
            
            return False
    
    def _delete_user(self, user_id: int):
        """Compensation: delete user"""
        with self.session_manager.get_session() as session:
            user = session.get(User, user_id)
            if user:
                session.delete(user)
                logger.info(f"Compensated: Deleted user {user_id}")
    
    def _clear_user_bio(self, user_id: int):
        """Compensation: clear user bio"""
        with self.session_manager.get_session() as session:
            user = session.get(User, user_id)
            if user:
                user.bio = None
                logger.info(f"Compensated: Cleared bio for user {user_id}")

def demonstrate_transaction_patterns():
    """Demonstrate various transaction patterns"""
    
    tx_manager = TransactionManager(session_manager)
    
    # Nested transactions
    logger.info("=== Nested Transactions ===")
    tx_manager.nested_transactions_example()
    
    # Retry pattern
    logger.info("=== Retry Pattern ===")
    success = tx_manager.retry_transaction_pattern()
    logger.info(f"Retry pattern result: {success}")
    
    # Distributed transactions
    logger.info("=== Distributed Transactions ===")
    success = tx_manager.distributed_transaction_pattern()
    logger.info(f"Distributed transaction result: {success}")
    
    # Compensation pattern
    logger.info("=== Compensation Pattern ===")
    success = tx_manager.compensation_transaction_pattern()
    logger.info(f"Compensation pattern result: {success}")
```

---

## Chapter 11: Advanced Query Techniques

### 11.1 Complex Joins and Subqueries

```python
from sqlalchemy import func, exists, case, literal_column, text
from sqlalchemy.orm import aliased, contains_eager, selectinload, subqueryload

# Complex join scenarios
def advanced_joins_example(session):
    """Advanced join patterns and techniques"""
    
    # Self-join with aliases
    parent_user = aliased(User)
    child_user = aliased(User)
    
    # Find users and their managers
    query = session.query(child_user, parent_user).join(
        parent_user, child_user.manager_id == parent_user.id
    )
    
    for employee, manager in query.all():
        print(f"{employee.username} reports to {manager.username}")
    
    # Multiple table joins with conditions
    complex_query = (
        session.query(User.username, func.count(Order.id).label('order_count'))
        .join(Order, User.id == Order.user_id)
        .join(Product, Order.product_id == Product.id)
        .filter(Product.price > 100)
        .group_by(User.id)
        .having(func.count(Order.id) > 2)
        .order_by(func.count(Order.id).desc())
    )
    
    # Correlated subqueries
    subq = (
        session.query(func.avg(Order.total_amount))
        .filter(Order.user_id == User.id)
        .correlate(User)
        .scalar_subquery()
    )
    
    high_value_customers = (
        session.query(User)
        .filter(subq > 1000)
    )
    
    return high_value_customers.all()

# Window functions
def window_functions_example(session):
    """Using window functions for analytics"""
    
    from sqlalchemy import func
    
    # Rank users by order total within each month
    query = (
        session.query(
            User.username,
            func.extract('month', Order.created_at).label('month'),
            func.sum(Order.total_amount).label('total'),
            func.rank().over(
                partition_by=func.extract('month', Order.created_at),
                order_by=func.sum(Order.total_amount).desc()
            ).label('rank')
        )
        .join(Order)
        .group_by(User.id, func.extract('month', Order.created_at))
        .order_by('month', 'rank')
    )
    
    return query.all()

# Common Table Expressions (CTEs)
def cte_example(session):
    """Using Common Table Expressions"""
    
    # Recursive CTE for hierarchical data
    if session.bind.dialect.name == 'postgresql':
        cte_query = text("""
            WITH RECURSIVE employee_hierarchy AS (
                -- Base case: top-level employees
                SELECT id, username, manager_id, 0 as level
                FROM users 
                WHERE manager_id IS NULL
                
                UNION ALL
                
                -- Recursive case
                SELECT u.id, u.username, u.manager_id, eh.level + 1
                FROM users u
                JOIN employee_hierarchy eh ON u.manager_id = eh.id
            )
            SELECT * FROM employee_hierarchy ORDER BY level, username
        """)
        
        result = session.execute(cte_query)
        return result.fetchall()
    
    return []

# Advanced filtering with EXISTS
def exists_subquery_example(session):
    """Using EXISTS for advanced filtering"""
    
    # Users who have placed orders
    has_orders = session.query(User).filter(
        exists().where(Order.user_id == User.id)
    )
    
    # Users who haven't placed any orders
    no_orders = session.query(User).filter(
        ~exists().where(Order.user_id == User.id)
    )
    
    # Users with high-value orders
    high_value_orders = session.query(User).filter(
        exists().where(
            (Order.user_id == User.id) & 
            (Order.total_amount > 1000)
        )
    )
    
    return {
        'has_orders': has_orders.count(),
        'no_orders': no_orders.count(),
        'high_value': high_value_orders.count()
    }
```

### 11.2 Query Optimization Techniques

```python
from sqlalchemy.orm import joinedload, selectinload, subqueryload
from sqlalchemy import Index, inspect
import time

class QueryOptimizer:
    """Advanced query optimization patterns"""
    
    def __init__(self, session):
        self.session = session
    
    def eager_loading_strategies(self):
        """Different eager loading strategies and their use cases"""
        
        # Joined load - single query with JOIN
        print("1. Joined Load (single query):")
        start = time.time()
        users_joined = (
            self.session.query(User)
            .options(joinedload(User.orders))
            .limit(10)
            .all()
        )
        print(f"   Time: {time.time() - start:.3f}s")
        print(f"   Users loaded: {len(users_joined)}")
        
        # Select in load - separate query for relationships
        print("\n2. Select In Load (separate optimized query):")
        start = time.time()
        users_selectin = (
            self.session.query(User)
            .options(selectinload(User.orders))
            .limit(10)
            .all()
        )
        print(f"   Time: {time.time() - start:.3f}s")
        print(f"   Users loaded: {len(users_selectin)}")
        
        # Subquery load - subquery for relationships
        print("\n3. Subquery Load:")
        start = time.time()
        users_subquery = (
            self.session.query(User)
            .options(subqueryload(User.orders))
            .limit(10)
            .all()
        )
        print(f"   Time: {time.time() - start:.3f}s")
        print(f"   Users loaded: {len(users_subquery)}")
    
    def query_execution_analysis(self):
        """Analyze query execution plans"""
        
        # Enable query logging
        import logging
        logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
        
        # Complex query for analysis
        query = (
            self.session.query(User)
            .join(Order)
            .join(Product)
            .filter(Product.price > 50)
            .filter(User.created_at > '2023-01-01')
            .options(joinedload(User.orders))
        )
        
        # Get the compiled query
        compiled = query.statement.compile(
            dialect=self.session.bind.dialect,
            compile_kwargs={"literal_binds": True}
        )
        
        print("Compiled SQL:")
        print(str(compiled))
        
        # Execute and measure
        start = time.time()
        results = query.all()
        execution_time = time.time() - start
        
        print(f"\nExecution time: {execution_time:.3f}s")
        print(f"Results count: {len(results)}")
        
        return results
    
    def batch_operations(self):
        """Efficient batch insert/update operations"""
        
        # Bulk insert
        print("1. Bulk Insert:")
        start = time.time()
        
        users_data = [
            {'username': f'user_{i}', 'email': f'user{i}@example.com'}
            for i in range(1000, 2000)
        ]
        
        self.session.bulk_insert_mappings(User, users_data)
        self.session.commit()
        
        print(f"   Bulk insert time: {time.time() - start:.3f}s")
        
        # Bulk update
        print("\n2. Bulk Update:")
        start = time.time()
        
        self.session.bulk_update_mappings(
            User, 
            [
                {'id': user.id, 'last_login': func.now()}
                for user in self.session.query(User).filter(User.id.between(1000, 1100)).all()
            ]
        )
        self.session.commit()
        
        print(f"   Bulk update time: {time.time() - start:.3f}s")
    
    def pagination_strategies(self):
        """Efficient pagination techniques"""
        
        # Standard LIMIT/OFFSET pagination
        def offset_pagination(page, per_page):
            return (
                self.session.query(User)
                .order_by(User.id)
                .offset(page * per_page)
                .limit(per_page)
                .all()
            )
        
        # Cursor-based pagination (more efficient for large datasets)
        def cursor_pagination(last_id=None, per_page=20):
            query = self.session.query(User).order_by(User.id)
            
            if last_id:
                query = query.filter(User.id > last_id)
            
            return query.limit(per_page).all()
        
        # Compare performance
        print("Pagination Performance Comparison:")
        
        # Test offset pagination
        start = time.time()
        page_1000 = offset_pagination(1000, 20)  # Skip 20,000 records
        offset_time = time.time() - start
        print(f"Offset pagination (page 1000): {offset_time:.3f}s")
        
        # Test cursor pagination
        start = time.time()
        cursor_page = cursor_pagination(20000, 20)  # After ID 20,000
        cursor_time = time.time() - start
        print(f"Cursor pagination (after ID 20000): {cursor_time:.3f}s")
        
        return {
            'offset_results': len(page_1000),
            'cursor_results': len(cursor_page),
            'performance_improvement': f"{(offset_time / cursor_time):.1f}x faster"
        }

# Query caching strategies
class QueryCache:
    """Implement query result caching"""
    
    def __init__(self):
        self.cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def cached_query(self, session, cache_key, query_func, ttl=300):
        """Cache query results with TTL"""
        import time
        import hashlib
        import pickle
        
        # Generate cache key
        if isinstance(cache_key, str):
            key = cache_key
        else:
            key = hashlib.md5(str(cache_key).encode()).hexdigest()
        
        # Check cache
        now = time.time()
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            if now - timestamp < ttl:
                self.cache_stats['hits'] += 1
                return pickle.loads(cached_data)
        
        # Execute query and cache result
        self.cache_stats['misses'] += 1
        result = query_func(session)
        self.cache[key] = (pickle.dumps(result), now)
        
        return result
    
    def get_cache_stats(self):
        """Get cache hit/miss statistics"""
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        if total == 0:
            return {'hit_rate': 0, 'total_queries': 0}
        
        return {
            'hit_rate': self.cache_stats['hits'] / total,
            'total_queries': total,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses']
        }
```

---

## Chapter 12: Database Migrations with Alembic

### 12.1 Alembic Setup and Configuration

```python
# alembic/env.py configuration
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Add your model import path
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

from models import Base  # Import your Base from models
from config import DATABASE_URL

# Alembic Config object
config = context.config

# Set database URL
config.set_main_option('sqlalchemy.url', DATABASE_URL)

# Interpret the config file for logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_schemas=True
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_schemas=True
        )

        with context.begin_transaction():
            context.run_migrations()

# Run migrations
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### 12.2 Migration Management

```python
"""
Migration management utilities and best practices
"""
import os
import subprocess
from pathlib import Path
from typing import List, Optional
from sqlalchemy import create_engine, text
import logging

class MigrationManager:
    """Manage database migrations with Alembic"""
    
    def __init__(self, database_url: str, alembic_dir: str = "alembic"):
        self.database_url = database_url
        self.alembic_dir = Path(alembic_dir)
        self.logger = logging.getLogger(__name__)
    
    def create_migration(self, message: str, auto: bool = True) -> str:
        """Create a new migration"""
        try:
            if auto:
                cmd = ["alembic", "revision", "--autogenerate", "-m", message]
            else:
                cmd = ["alembic", "revision", "-m", message]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Extract revision ID from output
            output_lines = result.stdout.strip().split('\n')
            revision_line = [line for line in output_lines if 'Generating' in line]
            
            if revision_line:
                revision_id = revision_line[0].split('/')[-1].split('_')[0]
                self.logger.info(f"Created migration: {revision_id} - {message}")
                return revision_id
            
            return "unknown"
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create migration: {e.stderr}")
            raise
    
    def apply_migrations(self, revision: str = "head") -> bool:
        """Apply migrations to database"""
        try:
            cmd = ["alembic", "upgrade", revision]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            self.logger.info(f"Successfully applied migrations to {revision}")
            return True
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Migration failed: {e.stderr}")
            return False
    
    def rollback_migration(self, revision: str) -> bool:
        """Rollback to a specific migration"""
        try:
            cmd = ["alembic", "downgrade", revision]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            self.logger.info(f"Successfully rolled back to {revision}")
            return True
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Rollback failed: {e.stderr}")
            return False
    
    def get_migration_history(self) -> List[dict]:
        """Get migration history"""
        try:
            cmd = ["alembic", "history", "--verbose"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse history output
            history = []
            current_migration = {}
            
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line.startswith('Rev:'):
                    if current_migration:
                        history.append(current_migration)
                    current_migration = {'revision': line.split(':')[1].strip()}
                elif line.startswith('Parent:'):
                    current_migration['parent'] = line.split(':')[1].strip()
                elif line.startswith('Branch:'):
                    current_migration['branch'] = line.split(':')[1].strip()
                elif line.startswith('Description:'):
                    current_migration['description'] = line.split(':', 1)[1].strip()
            
            if current_migration:
                history.append(current_migration)
            
            return history
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get history: {e.stderr}")
            return []
    
    def get_current_revision(self) -> Optional[str]:
        """Get current database revision"""
        try:
            cmd = ["alembic", "current"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            output = result.stdout.strip()
            if output:
                return output.split(' ')[0]
            return None
        
        except subprocess.CalledProcessError:
            return None
    
    def check_migration_status(self) -> dict:
        """Check if database needs migrations"""
        try:
            # Get current revision
            current = self.get_current_revision()
            
            # Get head revision
            cmd = ["alembic", "heads"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            head = result.stdout.strip().split(' ')[0]
            
            # Check for pending migrations
            cmd = ["alembic", "show", "head"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return {
                'current_revision': current,
                'head_revision': head,
                'up_to_date': current == head,
                'needs_migration': current != head
            }
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Status check failed: {e.stderr}")
            return {'error': str(e)}
    
    def create_backup_before_migration(self) -> str:
        """Create database backup before applying migrations"""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"backup_before_migration_{timestamp}.sql"
        
        try:
            # This is PostgreSQL-specific, adapt for other databases
            if 'postgresql' in self.database_url:
                import urllib.parse
                parsed = urllib.parse.urlparse(self.database_url)
                
                cmd = [
                    "pg_dump",
                    "-h", parsed.hostname or "localhost",
                    "-p", str(parsed.port or 5432),
                    "-U", parsed.username,
                    "-d", parsed.path[1:],  # Remove leading slash
                    "-f", backup_file,
                    "--verbose"
                ]
                
                env = os.environ.copy()
                env['PGPASSWORD'] = parsed.password
                
                subprocess.run(cmd, env=env, check=True)
                self.logger.info(f"Database backup created: {backup_file}")
                return backup_file
            
            else:
                self.logger.warning("Backup only implemented for PostgreSQL")
                return ""
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Backup failed: {e}")
            return ""

# Custom migration templates
def create_custom_migration_template():
    """Create custom migration template with best practices"""
    
    template = '''"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# Migration metadata
MIGRATION_ID = "${up_revision}"
MIGRATION_DESCRIPTION = "${message}"

def upgrade() -> None:
    """Apply migration changes"""
    
    # Add migration validation
    _validate_migration_preconditions()
    
    try:
        ${upgrades if upgrades else "pass"}
        
        # Log successful migration
        _log_migration_success("upgrade")
        
    except Exception as e:
        # Log migration failure
        _log_migration_failure("upgrade", str(e))
        raise

def downgrade() -> None:
    """Rollback migration changes"""
    
    try:
        ${downgrades if downgrades else "pass"}
        
        # Log successful rollback
        _log_migration_success("downgrade")
        
    except Exception as e:
        # Log rollback failure
        _log_migration_failure("downgrade", str(e))
        raise

def _validate_migration_preconditions():
    """Validate conditions before applying migration"""
    
    # Check database connection
    connection = op.get_bind()
    if not connection:
        raise RuntimeError("No database connection available")
    
    # Add custom validation logic here
    pass

def _log_migration_success(operation: str):
    """Log successful migration operation"""
    connection = op.get_bind()
    connection.execute(sa.text(f"""
        INSERT INTO migration_log (revision_id, description, operation, status, timestamp)
        VALUES ('{MIGRATION_ID}', '{MIGRATION_DESCRIPTION}', '{operation}', 'success', NOW())
    """))

def _log_migration_failure(operation: str, error: str):
    """Log failed migration operation"""
    try:
        connection = op.get_bind()
        connection.execute(sa.text(f"""
            INSERT INTO migration_log (revision_id, description, operation, status, error, timestamp)
            VALUES ('{MIGRATION_ID}', '{MIGRATION_DESCRIPTION}', '{operation}', 'failed', '{error}', NOW())
        """))
    except:
        # Ignore logging failures during migration failures
        pass
'''
    
    return template

# Data migrations
class DataMigration:
    """Handle data migrations and transformations"""
    
    def __init__(self, connection):
        self.connection = connection
        self.logger = logging.getLogger(__name__)
    
    def migrate_user_data(self):
        """Example data migration for user table changes"""
        
        # Step 1: Add new column
        self.connection.execute(text("""
            ALTER TABLE users 
            ADD COLUMN full_name VARCHAR(200)
        """))
        
        # Step 2: Populate new column from existing data
        self.connection.execute(text("""
            UPDATE users 
            SET full_name = CONCAT(first_name, ' ', last_name)
            WHERE first_name IS NOT NULL AND last_name IS NOT NULL
        """))
        
        # Step 3: Handle edge cases
        self.connection.execute(text("""
            UPDATE users 
            SET full_name = COALESCE(first_name, last_name, 'Unknown')
            WHERE full_name IS NULL
        """))
        
        self.logger.info("User data migration completed")
    
    def batch_data_transformation(self, table_name: str, batch_size: int = 1000):
        """Perform batch data transformation to avoid locking large tables"""
        
        # Get total count
        result = self.connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        total_rows = result.scalar()
        
        processed = 0
        while processed < total_rows:
            # Process batch
            self.connection.execute(text(f"""
                UPDATE {table_name}
                SET updated_at = NOW()
                WHERE id IN (
                    SELECT id FROM {table_name}
                    ORDER BY id
                    LIMIT {batch_size}
                    OFFSET {processed}
                )
            """))
            
            processed += batch_size
            self.logger.info(f"Processed {min(processed, total_rows)}/{total_rows} rows")
            
            # Commit batch to release locks
            self.connection.commit()

# Migration testing utilities
def test_migration_integrity():
    """Test migration integrity and rollback safety"""
    
    def test_forward_backward_migration(migration_manager: MigrationManager):
        """Test that migration can be applied and rolled back safely"""
        
        # Get current state
        initial_state = migration_manager.get_current_revision()
        
        # Apply migration
        success = migration_manager.apply_migrations("head")
        assert success, "Forward migration failed"
        
        # Verify migration applied
        current_state = migration_manager.get_current_revision()
        assert current_state != initial_state, "Migration didn't change database state"
        
        # Rollback migration
        success = migration_manager.rollback_migration(initial_state)
        assert success, "Rollback migration failed"
        
        # Verify rollback
        final_state = migration_manager.get_current_revision()
        assert final_state == initial_state, "Rollback didn't restore original state"
        
        print("✓ Migration forward/backward test passed")
    
    return test_forward_backward_migration
```

---

## Chapter 13: Performance Optimization

### 13.1 Database Performance Tuning

```python
from sqlalchemy import event, Index, text, func, inspect
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.engine import Engine
import time
import logging

class DatabasePerformanceMonitor:
    """Monitor and optimize database performance"""
    
    def __init__(self, engine):
        self.engine = engine
        self.query_stats = {}
        self.slow_query_threshold = 1.0  # seconds
        self.logger = logging.getLogger(__name__)
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Setup performance monitoring hooks"""
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
            context._query_statement = statement
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            if hasattr(context, '_query_start_time'):
                execution_time = time.time() - context._query_start_time
                
                # Log slow queries
                if execution_time > self.slow_query_threshold:
                    self.logger.warning(f"Slow query ({execution_time:.3f}s): {statement[:100]}...")
                
                # Collect statistics
                self._collect_query_stats(statement, execution_time)
    
    def _collect_query_stats(self, statement: str, execution_time: float):
        """Collect query execution statistics"""
        # Normalize query for statistics
        query_type = statement.strip().split()[0].upper()
        
        if query_type not in self.query_stats:
            self.query_stats[query_type] = {
                'count': 0,
                'total_time': 0,
                'avg_time': 0,
                'max_time': 0,
                'min_time': float('inf')
            }
        
        stats = self.query_stats[query_type]
        stats['count'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['max_time'] = max(stats['max_time'], execution_time)
        stats['min_time'] = min(stats['min_time'], execution_time)
    
    def get_performance_report(self) -> dict:
        """Get comprehensive performance report"""
        
        # Connection pool stats
        pool_stats = {}
        if hasattr(self.engine.pool, 'size'):
            pool_stats = {
                'pool_size': self.engine.pool.size(),
                'checked_in': self.engine.pool.checkedin(),
                'checked_out': self.engine.pool.checkedout(),
                'overflow': self.engine.pool.overflow(),
                'invalid': self.engine.pool.invalid()
            }
        
        return {
            'query_statistics': self.query_stats,
            'connection_pool': pool_stats,
            'slow_query_threshold': self.slow_query_threshold
        }
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest performance optimizations"""
        suggestions = []
        
        # Analyze query patterns
        for query_type, stats in self.query_stats.items():
            if stats['avg_time'] > self.slow_query_threshold:
                suggestions.append(
                    f"Consider optimizing {query_type} queries (avg: {stats['avg_time']:.3f}s)"
                )
            
            if query_type == 'SELECT' and stats['count'] > 1000:
                suggestions.append(
                    "High SELECT query volume - consider implementing query caching"
                )
        
        # Check connection pool utilization
        report = self.get_performance_report()
        pool_stats = report['connection_pool']
        
        if pool_stats and pool_stats.get('checked_out', 0) > pool_stats.get('pool_size', 0) * 0.8:
            suggestions.append("Connection pool utilization is high - consider increasing pool size")
        
        return suggestions

class IndexOptimizer:
    """Analyze and optimize database indexes"""
    
    def __init__(self, session):
        self.session = session
        self.logger = logging.getLogger(__name__)
    
    def analyze_missing_indexes(self, table_name: str) -> List[dict]:
        """Analyze missing indexes based on query patterns"""
        
        # This is PostgreSQL-specific - adapt for other databases
        if self.session.bind.dialect.name != 'postgresql':
            return []
        
        missing_indexes_query = text("""
            SELECT 
                schemaname,
                tablename,
                attname,
                n_distinct,
                correlation
            FROM pg_stats 
            WHERE schemaname = 'public' 
            AND tablename = :table_name
            AND n_distinct > 100
            ORDER BY n_distinct DESC
        """)
        
        result = self.session.execute(missing_indexes_query, {'table_name': table_name})
        
        suggestions = []
        for row in result:
            suggestions.append({
                'table': row.tablename,
                'column': row.attname,
                'distinct_values': row.n_distinct,
                'correlation': row.correlation,
                'suggestion': f"CREATE INDEX idx_{row.tablename}_{row.attname} ON {row.tablename}({row.attname})"
            })
        
        return suggestions
    
    def analyze_unused_indexes(self) -> List[dict]:
        """Find unused indexes that can be dropped"""
        
        if self.session.bind.dialect.name != 'postgresql':
            return []
        
        unused_indexes_query = text("""
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_tup_read,
                idx_tup_fetch
            FROM pg_stat_user_indexes 
            WHERE idx_tup_read = 0 
            AND idx_tup_fetch = 0
            AND indexname NOT LIKE '%_pkey'
        """)
        
        result = self.session.execute(unused_indexes_query)
        
        unused = []
        for row in result:
            unused.append({
                'schema': row.schemaname,
                'table': row.tablename,
                'index': row.indexname,
                'suggestion': f"Consider dropping unused index: DROP INDEX {row.indexname}"
            })
        
        return unused
    
    def create_composite_index(self, table_name: str, columns: List[str], index_name: str = None):
        """Create composite index for multiple columns"""
        
        if not index_name:
            index_name = f"idx_{table_name}_{'_'.join(columns)}"
        
        # Create index
        create_index_sql = text(f"""
            CREATE INDEX CONCURRENTLY {index_name} 
            ON {table_name} ({', '.join(columns)})
        """)
        
        try:
            self.session.execute(create_index_sql)
            self.session.commit()
            self.logger.info(f"Created composite index: {index_name}")
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Failed to create index {index_name}: {e}")

class QueryOptimizer:
    """Optimize SQLAlchemy queries for better performance"""
    
    def __init__(self, session):
        self.session = session
    
    def optimize_large_result_sets(self, query, batch_size: int = 1000):
        """Process large result sets in batches to avoid memory issues"""
        
        # Get total count first
        total = query.count()
        processed = 0
        
        while processed < total:
            batch = query.offset(processed).limit(batch_size).all()
            yield batch
            processed += len(batch)
            
            # Clear session periodically to prevent memory buildup
            if processed % (batch_size * 10) == 0:
                self.session.expunge_all()
    
    def optimize_n_plus_one(self, base_query, *relationships):
        """Eliminate N+1 queries using eager loading"""
        
        from sqlalchemy.orm import joinedload, selectinload
        
        # Automatically choose loading strategy based on relationship cardinality
        optimized_query = base_query
        
        for relationship in relationships:
            # Use selectinload for one-to-many, joinedload for many-to-one
            # This is a simplified heuristic
            optimized_query = optimized_query.options(selectinload(relationship))
        
        return optimized_query
    
    def create_materialized_view(self, view_name: str, query):
        """Create materialized view for expensive queries"""
        
        if self.session.bind.dialect.name == 'postgresql':
            # Get the SQL from the query
            compiled_query = str(query.statement.compile(
                dialect=self.session.bind.dialect,
                compile_kwargs={"literal_binds": True}
            ))
            
            # Create materialized view
            create_view_sql = text(f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS {view_name} AS
                {compiled_query}
            """)
            
            try:
                self.session.execute(create_view_sql)
                self.session.commit()
                
                # Create refresh function
                refresh_sql = text(f"REFRESH MATERIALIZED VIEW {view_name}")
                
                return {
                    'view_name': view_name,
                    'created': True,
                    'refresh_command': str(refresh_sql)
                }
            
            except Exception as e:
                self.session.rollback()
                return {'error': str(e)}
        
        return {'error': 'Materialized views only supported on PostgreSQL'}

# Connection pooling optimization
def configure_optimal_pool(database_url: str, **pool_kwargs):
    """Configure optimal connection pool settings"""
    
    default_pool_config = {
        'poolclass': QueuePool,
        'pool_size': 20,           # Number of connections to maintain
        'max_overflow': 30,        # Additional connections beyond pool_size
        'pool_recycle': 3600,      # Recycle connections after 1 hour
        'pool_pre_ping': True,     # Validate connections before use
        'pool_timeout': 30,        # Timeout for getting connection from pool
    }
    
    # Override with provided kwargs
    pool_config = {**default_pool_config, **pool_kwargs}
    
    from sqlalchemy import create_engine
    engine = create_engine(database_url, **pool_config)
    
    return engine

# Memory optimization for large datasets
class MemoryOptimizedQuery:
    """Handle memory-efficient querying for large datasets"""
    
    def __init__(self, session):
        self.session = session
    
    def stream_query_results(self, query, chunk_size: int = 1000):
        """Stream query results to avoid loading all data into memory"""
        
        # Enable streaming
        query = query.execution_options(stream_results=True)
        result = self.session.execute(query.statement)
        
        while True:
            chunk = result.fetchmany(chunk_size)
            if not chunk:
                break
            yield chunk
    
    def process_large_table(self, model_class, process_func, batch_size: int = 1000):
        """Process large tables in batches with memory management"""
        
        # Get primary key column
        pk_column = inspect(model_class).primary_key[0]
        
        # Find min and max IDs
        min_id = self.session.query(func.min(pk_column)).scalar()
        max_id = self.session.query(func.max(pk_column)).scalar()
        
        if min_id is None:
            return
        
        current_id = min_id
        processed_count = 0
        
        while current_id <= max_id:
            # Process batch
            batch = (
                self.session.query(model_class)
                .filter(pk_column.between(current_id, current_id + batch_size - 1))
                .all()
            )
            
            if batch:
                process_func(batch)
                processed_count += len(batch)
                
                # Clear session to free memory
                self.session.expunge_all()
                
                print(f"Processed {processed_count} records...")
            
            current_id += batch_size
        
        return processed_count

def demonstrate_performance_optimization(session):
    """Demonstrate performance optimization techniques"""
    
    print("=== Database Performance Optimization Demo ===")
    
    # Setup monitoring
    monitor = DatabasePerformanceMonitor(session.bind)
    
    # Simulate some queries
    print("\n1. Running sample queries...")
    users = session.query(User).limit(10).all()
    orders = session.query(Order).join(User).limit(20).all()
    
    # Get performance report
    print("\n2. Performance Report:")
    report = monitor.get_performance_report()
    for query_type, stats in report['query_statistics'].items():
        print(f"   {query_type}: {stats['count']} queries, avg {stats['avg_time']:.3f}s")
    
    # Get optimization suggestions
    print("\n3. Optimization Suggestions:")
    suggestions = monitor.suggest_optimizations()
    for suggestion in suggestions:
        print(f"   • {suggestion}")
    
    # Index analysis
    print("\n4. Index Analysis:")
    optimizer = IndexOptimizer(session)
    missing_indexes = optimizer.analyze_missing_indexes('users')
    for index_info in missing_indexes[:3]:  # Show top 3
        print(f"   Missing index suggestion: {index_info['suggestion']}")
    
    # Memory optimization demo
    print("\n5. Memory Optimization:")
    memory_optimizer = MemoryOptimizedQuery(session)
    
    def sample_processor(batch):
        print(f"   Processing batch of {len(batch)} records")
    
    if session.query(User).count() > 0:
        processed = memory_optimizer.process_large_table(User, sample_processor, 5)
        print(f"   Processed {processed} total records in batches")
```

---

## Chapter 14: Advanced Patterns and Techniques

### 14.1 Custom Types and Validators

```python
from sqlalchemy import TypeDecorator, String, Integer, DateTime, Text
from sqlalchemy.types import UserDefinedType
from sqlalchemy.orm import validates
import json
import uuid
from datetime import datetime
from typing import Any, Optional
import hashlib
import re

# Custom column types
class JSONType(TypeDecorator):
    """Custom JSON column type with automatic serialization"""
    
    impl = Text
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        """Convert Python object to JSON string for storage"""
        if value is not None:
            return json.dumps(value, default=str)
        return value
    
    def process_result_value(self, value, dialect):
        """Convert JSON string back to Python object"""
        if value is not None:
            return json.loads(value)
        return value

class EncryptedType(TypeDecorator):
    """Custom encrypted column type"""
    
    impl = String
    cache_ok = True
    
    def __init__(self, secret_key: str, **kwargs):
        self.secret_key = secret_key
        super().__init__(**kwargs)
    
    def process_bind_param(self, value, dialect):
        """Encrypt value before storage"""
        if value is not None:
            # Simple encryption (use proper encryption in production)
            import hashlib
            return hashlib.sha256((str(value) + self.secret_key).encode()).hexdigest()
        return value
    
    def process_result_value(self, value, dialect):
        """Return encrypted value (can't decrypt with SHA256)"""
        # In real implementation, use reversible encryption
        return value

class UUIDType(TypeDecorator):
    """Custom UUID column type"""
    
    impl = String(36)
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        """Convert UUID to string"""
        if value is not None:
            return str(value)
        return value
    
    def process_result_value(self, value, dialect):
        """Convert string back to UUID"""
        if value is not None:
            return uuid.UUID(value)
        return value

class EmailType(TypeDecorator):
    """Custom email column type with validation"""
    
    impl = String(255)
    cache_ok = True
    
    EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    def process_bind_param(self, value, dialect):
        """Validate and normalize email before storage"""
        if value is not None:
            # Normalize to lowercase
            email = str(value).strip().lower()
            
            # Validate format
            if not self.EMAIL_REGEX.match(email):
                raise ValueError(f"Invalid email format: {value}")
            
            return email
        return value

# Advanced model with custom types and validators
class AdvancedUser(Base):
    """User model with advanced features"""
    
    __tablename__ = 'advanced_users'
    
    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    email = Column(EmailType, unique=True, nullable=False)
    password_hash = Column(EncryptedType('secret_key'))
    profile = Column(JSONType)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    login_attempts = Column(Integer, default=0)
    is_locked = Column(Boolean, default=False)
    
    # Audit fields
    created_by = Column(String(100))
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(100))
    version = Column(Integer, default=1)
    
    @validates('email')
    def validate_email(self, key, email):
        """Additional email validation"""
        if not email:
            raise ValueError("Email is required")
        
        # Check for common typos
        common_typos = {
            'gmail.co': 'gmail.com',
            'yahoo.co': 'yahoo.com',
            'hotmail.co': 'hotmail.com'
        }
        
        for typo, correction in common_typos.items():
            if typo in email:
                suggested = email.replace(typo, correction)
                raise ValueError(f"Possible typo in email. Did you mean: {suggested}?")
        
        return email
    
    @validates('profile')
    def validate_profile(self, key, profile):
        """Validate profile JSON structure"""
        if profile is not None:
            required_fields = ['first_name', 'last_name']
            for field in required_fields:
                if field not in profile:
                    raise ValueError(f"Profile missing required field: {field}")
        
        return profile
    
    def increment_login_attempts(self):
        """Increment login attempts and lock if necessary"""
        self.login_attempts += 1
        if self.login_attempts >= 5:
            self.is_locked = True
    
    def reset_login_attempts(self):
        """Reset login attempts on successful login"""
        self.login_attempts = 0
        self.is_locked = False
        self.last_login = datetime.utcnow()
    
    def to_dict(self, include_sensitive=False):
        """Convert to dictionary for JSON serialization"""
        data = {
            'id': str(self.id),
            'email': self.email,
            'profile': self.profile,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_locked': self.is_locked
        }
        
        if include_sensitive:
            data.update({
                'login_attempts': self.login_attempts,
                'password_hash': self.password_hash
            })
        
        return data

# Model versioning and auditing
class AuditMixin:
    """Mixin for automatic audit trail"""
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(100))
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(100))
    version = Column(Integer, default=1)
    
    def increment_version(self):
        """Increment version on update"""
        self.version += 1
        self.updated_at = datetime.utcnow()

class SoftDeleteMixin:
    """Mixin for soft delete functionality"""
    
    deleted_at = Column(DateTime)
    deleted_by = Column(String(100))
    is_deleted = Column(Boolean, default=False)
    
    def soft_delete(self, deleted_by: str = None):
        """Mark record as deleted without removing from database"""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        self.deleted_by = deleted_by
    
    def restore(self):
        """Restore soft-deleted record"""
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = None

# Advanced query patterns
class AdvancedQueryPatterns:
    """Advanced querying techniques and patterns"""
    
    def __init__(self, session):
        self.session = session
    
    def fuzzy_search(self, model_class, field, search_term, threshold=0.7):
        """Fuzzy string matching for search"""
        
        # PostgreSQL-specific fuzzy search using similarity
        if self.session.bind.dialect.name == 'postgresql':
            from sqlalchemy import text, func
            
            # Enable fuzzy string extension
            self.session.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            
            query = (
                self.session.query(model_class)
                .filter(func.similarity(field, search_term) > threshold)
                .order_by(func.similarity(field, search_term).desc())
            )
            
            return query.all()
        
        # Fallback for other databases
        return (
            self.session.query(model_class)
            .filter(field.ilike(f'%{search_term}%'))
            .all()
        )
    
    def full_text_search(self, model_class, fields, search_term):
        """Full-text search across multiple fields"""
        
        if self.session.bind.dialect.name == 'postgresql':
            from sqlalchemy import func, text
            
            # Create search vector from multiple fields
            search_vector = func.to_tsvector(
                'english',
                func.concat_ws(' ', *fields)
            )
            
            search_query = func.plainto_tsquery('english', search_term)
            
            query = (
                self.session.query(
                    model_class,
                    func.ts_rank(search_vector, search_query).label('rank')
                )
                .filter(search_vector.match(search_term))
                .order_by(text('rank DESC'))
            )
            
            return query.all()
        
        # Fallback implementation
        from sqlalchemy import or_
        
        conditions = [field.ilike(f'%{search_term}%') for field in fields]
        return (
            self.session.query(model_class)
            .filter(or_(*conditions))
            .all()
        )
    
    def temporal_queries(self, model_class, date_field, period='month'):
        """Query data grouped by time periods"""
        
        from sqlalchemy import func, extract
        
        if period == 'month':
            period_func = func.date_trunc('month', date_field)
        elif period == 'week':
            period_func = func.date_trunc('week', date_field)
        elif period == 'day':
            period_func = func.date_trunc('day', date_field)
        else:
            period_func = date_field
        
        query = (
            self.session.query(
                period_func.label('period'),
                func.count().label('count')
            )
            .group_by('period')
            .order_by('period')
        )
        
        return query.all()
    
    def hierarchical_queries(self, model_class, parent_id_field, target_id):
        """Query hierarchical data (tree structures)"""
        
        # Simple recursive approach for small datasets
        def get_descendants(parent_id, level=0, max_level=10):
            if level > max_level:  # Prevent infinite recursion
                return []
            
            children = (
                self.session.query(model_class)
                .filter(parent_id_field == parent_id)
                .all()
            )
            
            result = []
            for child in children:
                result.append((child, level))
                # Recursively get descendants
                result.extend(get_descendants(child.id, level + 1, max_level))
            
            return result
        
        return get_descendants(target_id)

# Connection and session management patterns
class SessionManager:
    """Advanced session management patterns"""
    
    def __init__(self, engine):
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=engine)
    
    def read_only_session(self):
        """Create read-only session for queries"""
        session = self.SessionLocal()
        session.connection(execution_options={'isolation_level': 'AUTOCOMMIT'})
        return session
    
    def batch_session(self, batch_size=1000):
        """Session optimized for batch operations"""
        session = self.SessionLocal()
        # Disable autoflush for performance
        session.autoflush = False
        return session
    
    def transactional_session(self):
        """Session with explicit transaction management"""
        session = self.SessionLocal()
        session.begin()
        return session

class DatabaseSharding:
    """Implement database sharding patterns"""
    
    def __init__(self, shard_engines):
        self.shard_engines = shard_engines
        self.shard_count = len(shard_engines)
    
    def get_shard_key(self, user_id):
        """Determine shard based on user ID"""
        return hash(str(user_id)) % self.shard_count
    
    def get_session_for_user(self, user_id):
        """Get session for specific user's shard"""
        shard_key = self.get_shard_key(user_id)
        engine = self.shard_engines[shard_key]
        SessionLocal = sessionmaker(bind=engine)
        return SessionLocal()
    
    def query_all_shards(self, query_func):
        """Execute query across all shards and combine results"""
        results = []
        
        for engine in self.shard_engines:
            SessionLocal = sessionmaker(bind=engine)
            session = SessionLocal()
            try:
                shard_results = query_func(session)
                results.extend(shard_results)
            finally:
                session.close()
        
        return results

def demonstrate_advanced_patterns(session):
    """Demonstrate advanced SQLAlchemy patterns"""
    
    print("=== Advanced SQLAlchemy Patterns Demo ===")
    
    # Custom types demonstration
    print("\n1. Custom Types:")
    try:
        # Create advanced user
        advanced_user = AdvancedUser(
            email="test@example.com",
            profile={'first_name': 'John', 'last_name': 'Doe', 'age': 30},
            created_by='system'
        )
        
        session.add(advanced_user)
        session.commit()
        
        print(f"   Created user with UUID: {advanced_user.id}")
        print(f"   Profile JSON: {advanced_user.profile}")
        
    except Exception as e:
        session.rollback()
        print(f"   Error: {e}")
    
    # Advanced query patterns
    print("\n2. Advanced Query Patterns:")
    query_patterns = AdvancedQueryPatterns(session)
    
    # Fuzzy search example
    fuzzy_results = query_patterns.fuzzy_search(User, User.username, 'john', 0.5)
    print(f"   Fuzzy search results: {len(fuzzy_results)}")
    
    # Temporal queries
    if session.query(User).count() > 0:
        temporal_results = query_patterns.temporal_queries(User, User.created_at, 'month')
        print(f"   Temporal query results: {len(temporal_results)}")
    
    print("\n3. Session Management:")
    session_manager = SessionManager(session.bind)
    
    # Read-only session
    readonly_session = session_manager.read_only_session()
    user_count = readonly_session.query(User).count()
    readonly_session.close()
    print(f"   Read-only query result: {user_count} users")
    
    print("   Advanced patterns demonstration completed!")
```

---

## Chapter 15: Integration with Web Frameworks

### 15.1 FastAPI Integration

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# FastAPI + SQLAlchemy setup
DATABASE_URL = "sqlite:///./fastapi_example.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(title="SQLAlchemy + FastAPI", version="1.0.0")
security = HTTPBearer()

# Pydantic models for API
class UserCreate(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    
    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None

# Dependency injection
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get current authenticated user"""
    # Simple token validation (implement proper JWT in production)
    token = credentials.credentials
    
    if token != "valid-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    # Return user or raise exception
    user = db.query(User).filter(User.id == 1).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user

# API endpoints
@app.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    
    # Check if user already exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@app.get("/users/", response_model=List[UserResponse])
async def list_users(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all users"""
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user by ID"""
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@app.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int, 
    user_update: UserUpdate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update user"""
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update only provided fields
    update_data = user_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_user, field, value)
    
    db.commit()
    db.refresh(db_user)
    return db_user

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete user"""
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    db.delete(db_user)
    db.commit()

# Health check endpoint
@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Advanced FastAPI features
@app.middleware("http")
async def add_db_session_middleware(request, call_next):
    """Add database session to request context"""
    response = await call_next(request)
    return response

# Background tasks with SQLAlchemy
from fastapi import BackgroundTasks

@app.post("/users/{user_id}/send-email")
async def send_user_email(
    user_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Send email to user (background task)"""
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    def send_email_task(email: str, username: str):
        """Background task to send email"""
        # Simulate email sending
        import time
        time.sleep(2)  # Simulate email service delay
        print(f"Email sent to {email} for user {username}")
    
    background_tasks.add_task(send_email_task, user.email, user.username)
    return {"message": "Email will be sent in the background"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 15.2 Flask Integration

```python
from flask import Flask, request, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os

# Flask app setup
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flask_example.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # Change in production
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)

# Extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
jwt = JWTManager(app)

# Flask-SQLAlchemy models
class FlaskUser(db.Model):
    """User model for Flask integration"""
    
    __tablename__ = 'flask_users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    full_name = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    posts = db.relationship('FlaskPost', backref='author', lazy='dynamic')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'posts_count': self.posts.count()
        }

class FlaskPost(db.Model):
    """Post model for Flask integration"""
    
    __tablename__ = 'flask_posts'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign key
    user_id = db.Column(db.Integer, db.ForeignKey('flask_users.id'), nullable=False)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'author': self.author.username if self.author else None
        }

# Request context and error handling
@app.before_request
def before_request():
    """Before each request"""
    g.request_start_time = datetime.utcnow()

@app.after_request
def after_request(response):
    """After each request"""
    if hasattr(g, 'request_start_time'):
        duration = datetime.utcnow() - g.request_start_time
        app.logger.info(f"Request completed in {duration.total_seconds():.3f}s")
    return response

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(400)
def bad_request(error):
    """400 error handler"""
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

# Authentication endpoints
@app.route('/api/register', methods=['POST'])
def register():
    """Register new user"""
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Username and password required'}), 400
    
    # Check if user exists
    if FlaskUser.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if FlaskUser.query.filter_by(email=data.get('email')).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    # Create user
    user = FlaskUser(
        username=data['username'],
        email=data.get('email', ''),
        full_name=data.get('full_name', '')
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify(user.to_dict()), 201

@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Username and password required'}), 400
    
    user = FlaskUser.query.filter_by(username=data['username']).first()
    
    if user and user.check_password(data['password']):
        access_token = create_access_token(identity=user.id)
        return jsonify({
            'access_token': access_token,
            'user': user.to_dict()
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401

# User endpoints
@app.route('/api/users', methods=['GET'])
@jwt_required()
def list_users():
    """List all users"""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 10, type=int), 100)
    
    users = FlaskUser.query.paginate(
        page=page, 
        per_page=per_page, 
        error_out=False
    )
    
    return jsonify({
        'users': [user.to_dict() for user in users.items],
        'pagination': {
            'page': users.page,
            'pages': users.pages,
            'per_page': users.per_page,
            'total': users.total,
            'has_next': users.has_next,
            'has_prev': users.has_prev
        }
    })

@app.route('/api/users/<int:user_id>', methods=['GET'])
@jwt_required()
def get_user(user_id):
    """Get user by ID"""
    user = FlaskUser.query.get_or_404(user_id)
    return jsonify(user.to_dict())

@app.route('/api/users/<int:user_id>', methods=['PUT'])
@jwt_required()
def update_user(user_id):
    """Update user"""
    current_user_id = get_jwt_identity()
    
    # Check if user can update this profile
    if current_user_id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    user = FlaskUser.query.get_or_404(user_id)
    data = request.get_json()
    
    # Update allowed fields
    if 'full_name' in data:
        user.full_name = data['full_name']
    if 'email' in data:
        # Check email uniqueness
        existing_user = FlaskUser.query.filter_by(email=data['email']).first()
        if existing_user and existing_user.id != user_id:
            return jsonify({'error': 'Email already in use'}), 400
        user.email = data['email']
    
    db.session.commit()
    return jsonify(user.to_dict())

# Post endpoints
@app.route('/api/posts', methods=['POST'])
@jwt_required()
def create_post():
    """Create new post"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    if not data or not data.get('title') or not data.get('content'):
        return jsonify({'error': 'Title and content required'}), 400
    
    post = FlaskPost(
        title=data['title'],
        content=data['content'],
        user_id=current_user_id
    )
    
    db.session.add(post)
    db.session.commit()
    
    return jsonify(post.to_dict()), 201

@app.route('/api/posts', methods=['GET'])
def list_posts():
    """List all posts"""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 10, type=int), 100)
    
    posts = FlaskPost.query.order_by(FlaskPost.created_at.desc()).paginate(
        page=page,
        per_page=per_page,
        error_out=False
    )
    
    return jsonify({
        'posts': [post.to_dict() for post in posts.items],
        'pagination': {
            'page': posts.page,
            'pages': posts.pages,
            'per_page': posts.per_page,
            'total': posts.total
        }
    })

@app.route('/api/posts/<int:post_id>', methods=['GET'])
def get_post(post_id):
    """Get post by ID"""
    post = FlaskPost.query.get_or_404(post_id)
    return jsonify(post.to_dict())

@app.route('/api/posts/<int:post_id>', methods=['PUT'])
@jwt_required()
def update_post(post_id):
    """Update post"""
    current_user_id = get_jwt_identity()
    post = FlaskPost.query.get_or_404(post_id)
    
    # Check if user owns this post
    if post.user_id != current_user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    
    if 'title' in data:
        post.title = data['title']
    if 'content' in data:
        post.content = data['content']
    
    post.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify(post.to_dict())

@app.route('/api/posts/<int:post_id>', methods=['DELETE'])
@jwt_required()
def delete_post(post_id):
    """Delete post"""
    current_user_id = get_jwt_identity()
    post = FlaskPost.query.get_or_404(post_id)
    
    # Check if user owns this post
    if post.user_id != current_user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    db.session.delete(post)
    db.session.commit()
    
    return '', 204

# Health check
@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db.session.execute('SELECT 1')
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# CLI commands for database management
@app.cli.command()
def init_db():
    """Initialize database"""
    db.create_all()
    print("Database initialized!")

@app.cli.command()
def seed_data():
    """Seed database with sample data"""
    
    # Create sample user
    user = FlaskUser(
        username='admin',
        email='admin@example.com',
        full_name='Administrator'
    )
    user.set_password('admin123')
    
    db.session.add(user)
    db.session.commit()
    
    # Create sample post
    post = FlaskPost(
        title='Welcome to the Blog',
        content='This is a sample blog post created during database seeding.',
        user_id=user.id
    )
    
    db.session.add(post)
    db.session.commit()
    
    print("Sample data created!")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    app.run(debug=True)
```

---

This completes the comprehensive SQLAlchemy guide with all 15 chapters! The guide now covers:

✅ **Complete Chapters 1-15:**
- Chapters 1-10: Foundation and core concepts (already existed)
- **Chapter 11**: Advanced Query Techniques (window functions, CTEs, complex joins)
- **Chapter 12**: Database Migrations with Alembic (migration management, data migrations)
- **Chapter 13**: Performance Optimization (monitoring, indexing, query optimization)
- **Chapter 14**: Advanced Patterns (custom types, validators, sharding, session management)  
- **Chapter 15**: Web Framework Integration (FastAPI and Flask with SQLAlchemy)

The SQLAlchemy guide is now complete and comprehensive, covering everything from basic concepts through advanced enterprise patterns and web framework integration!
