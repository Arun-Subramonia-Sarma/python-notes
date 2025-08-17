# Comprehensive Flask Framework Guide

## Table of Contents

1. [Introduction to Flask](#chapter-1-introduction-to-flask)
2. [Environment Setup and Installation](#chapter-2-environment-setup-and-installation)
3. [Flask Fundamentals](#chapter-3-flask-fundamentals)
4. [Routing and URL Handling](#chapter-4-routing-and-url-handling)
5. [Request Handling and Response Generation](#chapter-5-request-handling-and-response-generation)
6. [Templates and Jinja2](#chapter-6-templates-and-jinja2)
7. [Static Files and Asset Management](#chapter-7-static-files-and-asset-management)
8. [Forms and User Input](#chapter-8-forms-and-user-input)
9. [Database Integration](#chapter-9-database-integration)
10. [User Authentication and Authorization](#chapter-10-user-authentication-and-authorization)
11. [Session Management and Cookies](#chapter-11-session-management-and-cookies)
12. [Error Handling and Logging](#chapter-12-error-handling-and-logging)
13. [Flask Extensions](#chapter-13-flask-extensions)
14. [RESTful APIs with Flask](#chapter-14-restful-apis-with-flask)
15. [Testing Flask Applications](#chapter-15-testing-flask-applications)
16. [Security Best Practices](#chapter-16-security-best-practices)
17. [Performance Optimization](#chapter-17-performance-optimization)
18. [Deployment and Production](#chapter-18-deployment-and-production)
19. [Real-World Projects](#chapter-19-real-world-projects)
20. [Advanced Topics](#chapter-20-advanced-topics)

---

## Chapter 1: Introduction to Flask

Flask is a lightweight and flexible Python web framework that provides the essential tools and features needed to build web applications quickly and efficiently.

### 1.1 What is Flask?

Flask is a micro web framework written in Python. It's based on the Werkzeug WSGI toolkit and Jinja2 template engine.

**Key Features:**
- Lightweight and minimalist
- Flexible and extensible
- Built-in development server and debugger
- RESTful request dispatching
- Jinja2 templating
- Secure cookie support (client-side sessions)
- WSGI compliant
- Unicode-based

### 1.2 Flask vs Other Frameworks

```python
# web-frameworks/examples/01_framework_comparison.py

"""
Framework Comparison: Flask vs Django vs FastAPI
"""

# Flask - Minimal and Flexible
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, Flask!'

# Key Benefits:
# - Simple and easy to learn
# - Flexible architecture
# - Extensive ecosystem
# - Perfect for small to medium applications
# - Great for microservices

if __name__ == '__main__':
    app.run(debug=True)
```

### 1.3 When to Choose Flask

**Use Flask when:**
- Building small to medium-sized applications
- Need flexibility in architecture decisions
- Creating APIs and microservices
- Prototyping web applications
- Learning web development concepts
- Building custom solutions

**Consider alternatives when:**
- Building large, complex applications (Django)
- Need high-performance async APIs (FastAPI)
- Require extensive built-in features (Django)

---

## Chapter 2: Environment Setup and Installation

### 2.1 Python Environment Setup

```bash
# web-frameworks/setup/setup_flask_environment.sh

#!/bin/bash
echo "Setting up Flask Development Environment"

# Check Python version
python3 --version

# Create project directory
mkdir flask-comprehensive-guide
cd flask-comprehensive-guide

# Create virtual environment
python3 -m venv flask_env
source flask_env/bin/activate  # On Windows: flask_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

echo "✅ Virtual environment created and activated!"
```

### 2.2 Flask Installation and Dependencies

```bash
# Install Flask and essential packages
pip install Flask

# Install development dependencies
pip install Flask-SQLAlchemy Flask-Migrate Flask-Login Flask-WTF
pip install Flask-Mail Flask-Admin Flask-CORS Flask-JWT-Extended
pip install pytest pytest-flask python-dotenv

# Install additional useful packages
pip install requests gunicorn redis celery

# Create requirements.txt
pip freeze > requirements.txt

echo "✅ Flask and dependencies installed!"
```

### 2.3 Project Structure Setup

```
flask-comprehensive-guide/
├── app/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── auth.py
│   │   └── api.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   └── auth/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   ├── forms/
│   │   ├── __init__.py
│   │   └── auth.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_routes.py
│   └── conftest.py
├── migrations/
├── config.py
├── run.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

### 2.4 Configuration Setup

```python
# web-frameworks/examples/02_configuration.py

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Basic Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    # Mail configuration
    MAIL_SERVER = os.environ.get('MAIL_SERVER') or 'localhost'
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 25)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'false').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # Security
    WTF_CSRF_ENABLED = True
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    
    # Application specific
    POSTS_PER_PAGE = 10
    LANGUAGES = ['en', 'es', 'fr']
    
    # Redis configuration
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    
    # Celery configuration
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL') or 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND') or 'redis://localhost:6379/0'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Disable security features for development
    SESSION_COOKIE_SECURE = False
    WTF_CSRF_ENABLED = False
    
    # Development database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or 'sqlite:///dev.db'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    
    # Use in-memory database for testing
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable CSRF protection for testing
    WTF_CSRF_ENABLED = False
    
    # Disable email sending in tests
    MAIL_SUPPRESS_SEND = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1 year
    
    # Production database (should use PostgreSQL or MySQL)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://user:pass@localhost/proddb'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])
```

---

## Chapter 3: Flask Fundamentals

### 3.1 Basic Flask Application Structure

```python
# web-frameworks/examples/03_basic_app.py

from flask import Flask, render_template, request, redirect, url_for, flash
import os

# Create Flask application instance
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Application context and request context demonstration
@app.before_first_request
def before_first_request():
    """Execute before the first request"""
    print("Application starting up...")

@app.before_request
def before_request():
    """Execute before each request"""
    print(f"Processing request: {request.method} {request.path}")

@app.after_request
def after_request(response):
    """Execute after each request"""
    print(f"Response status: {response.status_code}")
    return response

@app.teardown_request
def teardown_request(exception):
    """Execute during request teardown"""
    if exception:
        print(f"Request ended with exception: {exception}")

# Basic route
@app.route('/')
def index():
    """Home page"""
    return '<h1>Welcome to Flask!</h1>'

# Route with template
@app.route('/home')
def home():
    """Home page with template"""
    user = {'name': 'Flask User'}
    return render_template('home.html', user=user)

# Route with parameters
@app.route('/user/<username>')
def user_profile(username):
    """User profile page"""
    return f'<h1>Hello, {username}!</h1>'

# Route with multiple HTTP methods
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """Contact form"""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Process form data
        flash(f'Thank you {name}! Your message has been sent.')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('errors/500.html'), 500

# Custom CLI commands
@app.cli.command()
def init_db():
    """Initialize the database"""
    print("Initializing database...")
    # Database initialization logic here

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### 3.2 Application Factory Pattern

```python
# web-frameworks/examples/03_application_factory.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from config import get_config

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()

def create_app(config_name=None):
    """Application factory function"""
    
    # Create Flask instance
    app = Flask(__name__)
    
    # Load configuration
    if config_name:
        app.config.from_object(config_name)
    else:
        app.config.from_object(get_config())
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Configure login manager
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    
    # Register blueprints
    from app.routes.main import main_bp
    from app.routes.auth import auth_bp
    from app.routes.api import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register CLI commands
    register_cli_commands(app)
    
    # Add context processors
    @app.context_processor
    def utility_processor():
        """Add utility functions to template context"""
        return dict(
            enumerate=enumerate,
            len=len,
            str=str
        )
    
    return app

def register_error_handlers(app):
    """Register error handlers"""
    
    @app.errorhandler(403)
    def forbidden(error):
        return render_template('errors/403.html'), 403
    
    @app.errorhandler(404)
    def not_found(error):
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template('errors/500.html'), 500

def register_cli_commands(app):
    """Register CLI commands"""
    
    @app.cli.command()
    def deploy():
        """Deploy the application"""
        print("Deploying application...")
        
        # Create database tables
        db.create_all()
        
        # Add default data
        create_default_data()
        
        print("Deployment completed!")
    
    @app.cli.command()
    def test():
        """Run tests"""
        import pytest
        pytest.main(['-v', 'tests/'])

def create_default_data():
    """Create default application data"""
    # Add default users, roles, etc.
    pass

# Usage example
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
```

### 3.3 Flask Extensions Integration

```python
# web-frameworks/examples/03_extensions_setup.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_mail import Mail
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from flask_compress import Compress
from flask_talisman import Talisman

# Extension instances
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
mail = Mail()
csrf = CSRFProtect()
cache = Cache()
compress = Compress()

def create_app_with_extensions(config_name=None):
    """Create app with all extensions configured"""
    
    app = Flask(__name__)
    
    # Load configuration
    if config_name:
        app.config.from_object(config_name)
    else:
        from config import get_config
        app.config.from_object(get_config())
    
    # Initialize core extensions
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Security extensions
    csrf.init_app(app)
    Talisman(app, force_https=False)  # Set to True in production
    
    # Performance extensions
    cache.init_app(app, config={
        'CACHE_TYPE': 'redis',
        'CACHE_REDIS_URL': app.config.get('REDIS_URL')
    })
    compress.init_app(app)
    
    # Rate limiting
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=["1000 per hour", "100 per minute"]
    )
    
    # Authentication
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        from app.models.user import User
        return User.query.get(int(user_id))
    
    # Email
    mail.init_app(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register shell context
    @app.shell_context_processor
    def make_shell_context():
        return {
            'db': db,
            'User': User,
            'Post': Post
        }
    
    return app

def register_blueprints(app):
    """Register application blueprints"""
    
    from app.routes.main import main_bp
    from app.routes.auth import auth_bp
    from app.routes.api import api_bp
    from app.routes.admin import admin_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    app.register_blueprint(admin_bp, url_prefix='/admin')

# Demonstration of extension usage
def demonstrate_extensions():
    """Show how to use various Flask extensions"""
    
    app = create_app_with_extensions()
    
    with app.app_context():
        # Database operations
        user = User(username='demo', email='demo@example.com')
        db.session.add(user)
        db.session.commit()
        
        # Caching
        @cache.memoize(timeout=300)
        def expensive_operation(param):
            import time
            time.sleep(2)  # Simulate expensive operation
            return f"Result for {param}"
        
        # Rate limiting example
        @app.route('/api/limited')
        @limiter.limit("5 per minute")
        def limited_endpoint():
            return {'message': 'This endpoint is rate limited'}
        
        # Email example
        from flask_mail import Message
        
        def send_email(subject, recipients, body):
            msg = Message(subject=subject, recipients=recipients)
            msg.body = body
            mail.send(msg)

if __name__ == '__main__':
    app = create_app_with_extensions()
    app.run(debug=True)
```

---

## Chapter 4: Routing and URL Handling

### 4.1 Basic Routing Patterns

```python
# web-frameworks/examples/04_routing_basics.py

from flask import Flask, request, redirect, url_for, abort
from werkzeug.routing import Rule

app = Flask(__name__)

# Static routes
@app.route('/')
def index():
    """Home page"""
    return 'Home Page'

@app.route('/about')
def about():
    """About page"""
    return 'About Page'

# Routes with variables
@app.route('/user/<username>')
def user_profile(username):
    """User profile with string parameter"""
    return f'User: {username}'

@app.route('/post/<int:post_id>')
def show_post(post_id):
    """Show post with integer parameter"""
    return f'Post ID: {post_id}'

@app.route('/user/<int:user_id>/posts/<int:post_id>')
def user_post(user_id, post_id):
    """Nested parameters"""
    return f'User {user_id}, Post {post_id}'

# Route with multiple variable types
@app.route('/file/<path:filename>')
def serve_file(filename):
    """Serve file with path parameter (includes slashes)"""
    return f'File: {filename}'

@app.route('/tag/<uuid:tag_id>')
def show_tag(tag_id):
    """Route with UUID parameter"""
    return f'Tag: {tag_id}'

# HTTP methods
@app.route('/submit', methods=['GET', 'POST', 'PUT', 'DELETE'])
def handle_submit():
    """Handle multiple HTTP methods"""
    if request.method == 'GET':
        return 'GET request'
    elif request.method == 'POST':
        return 'POST request'
    elif request.method == 'PUT':
        return 'PUT request'
    elif request.method == 'DELETE':
        return 'DELETE request'

# Route with defaults
@app.route('/page/')
@app.route('/page/<int:page_num>')
def show_page(page_num=1):
    """Route with default parameter"""
    return f'Page: {page_num}'

# Trailing slashes
@app.route('/projects/')
def projects():
    """Route with trailing slash (canonical)"""
    return 'Projects page'

@app.route('/tasks')  # No trailing slash
def tasks():
    """Route without trailing slash"""
    return 'Tasks page'

# Route with query parameters
@app.route('/search')
def search():
    """Handle query parameters"""
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    return f'Search: {query}, Page: {page}, Per page: {per_page}'

# Redirects
@app.route('/old-url')
def old_url():
    """Redirect to new URL"""
    return redirect(url_for('new_url'))

@app.route('/new-url')
def new_url():
    """New URL destination"""
    return 'New URL content'

# Error handling
@app.route('/protected')
def protected():
    """Protected route that might abort"""
    # Simulate authentication check
    authenticated = request.args.get('auth') == 'true'
    
    if not authenticated:
        abort(403)  # Forbidden
    
    return 'Protected content'

# Dynamic URL building
@app.route('/build-urls')
def build_urls():
    """Demonstrate URL building"""
    urls = {
        'home': url_for('index'),
        'about': url_for('about'),
        'user': url_for('user_profile', username='johndoe'),
        'post': url_for('show_post', post_id=123),
        'search': url_for('search', q='flask', page=2),
    }
    
    html = '<h1>Generated URLs:</h1>'
    for name, url in urls.items():
        html += f'<p><strong>{name}:</strong> <a href="{url}">{url}</a></p>'
    
    return html

if __name__ == '__main__':
    app.run(debug=True)
```

### 4.2 Advanced Routing Techniques

```python
# web-frameworks/examples/04_advanced_routing.py

from flask import Flask, request, g
from werkzeug.routing import BaseConverter
import re

app = Flask(__name__)

# Custom URL converters
class ListConverter(BaseConverter):
    """Convert URL segment to Python list"""
    
    def to_python(self, value):
        return value.split(',')
    
    def to_url(self, values):
        return ','.join(BaseConverter.to_url(self, value) for value in values)

class RegexConverter(BaseConverter):
    """Custom regex converter"""
    
    def __init__(self, url_map, *items):
        super(RegexConverter, self).__init__(url_map)
        self.regex = items[0]

# Register custom converters
app.url_map.converters['list'] = ListConverter
app.url_map.converters['regex'] = RegexConverter

# Using custom converters
@app.route('/tags/<list:tag_names>')
def show_tags(tag_names):
    """Route using list converter"""
    return f'Tags: {", ".join(tag_names)}'

@app.route('/user/<regex("[a-z]{3,}"):username>')
def user_regex(username):
    """Route using regex converter"""
    return f'User: {username}'

# Subdomain routing
@app.route('/admin', subdomain='admin')
def admin_panel():
    """Admin subdomain route"""
    return 'Admin Panel'

@app.route('/', subdomain='<subdomain>')
def subdomain_index(subdomain):
    """Dynamic subdomain routing"""
    return f'Subdomain: {subdomain}'

# Host-based routing
@app.route('/', host='api.example.com')
def api_index():
    """API host routing"""
    return {'message': 'API endpoint'}

# Route registration alternatives
def alternative_route():
    """Alternative way to register routes"""
    return 'Alternative route'

app.add_url_rule('/alternative', 'alternative', alternative_route)

# Class-based views
from flask.views import View, MethodView

class ListView(View):
    """Class-based view"""
    
    def dispatch_request(self):
        return 'List view'

app.add_url_rule('/list', view_func=ListView.as_view('list'))

class ItemAPI(MethodView):
    """RESTful class-based view"""
    
    def get(self, item_id=None):
        if item_id is None:
            return 'List all items'
        return f'Get item {item_id}'
    
    def post(self):
        return 'Create new item'
    
    def put(self, item_id):
        return f'Update item {item_id}'
    
    def delete(self, item_id):
        return f'Delete item {item_id}'

# Register class-based view
item_view = ItemAPI.as_view('item_api')
app.add_url_rule('/items/', defaults={'item_id': None},
                 view_func=item_view, methods=['GET'])
app.add_url_rule('/items/', view_func=item_view, methods=['POST'])
app.add_url_rule('/items/<int:item_id>', view_func=item_view,
                 methods=['GET', 'PUT', 'DELETE'])

# Route groups using blueprints
from flask import Blueprint

# Create blueprint
api_v1 = Blueprint('api_v1', __name__)

@api_v1.route('/users')
def list_users():
    return 'API v1 - List users'

@api_v1.route('/users/<int:user_id>')
def get_user(user_id):
    return f'API v1 - Get user {user_id}'

# Register blueprint
app.register_blueprint(api_v1, url_prefix='/api/v1')

# Route with multiple decorators
def login_required(f):
    """Simple login required decorator"""
    def decorated_function(*args, **kwargs):
        if not getattr(g, 'user', None):
            return 'Login required', 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/dashboard')
@login_required
def dashboard():
    """Protected dashboard route"""
    return 'Dashboard content'

# URL preprocessing
@app.url_defaults
def add_language_code(endpoint, values):
    """Add language code to URLs"""
    if 'lang_code' in values or not g.lang_code:
        return
    if app.url_map.is_endpoint_expecting(endpoint, 'lang_code'):
        values['lang_code'] = g.lang_code

@app.url_value_preprocessor
def pull_lang_code(endpoint, values):
    """Extract language code from URLs"""
    if values is not None:
        g.lang_code = values.pop('lang_code', 'en')

@app.route('/<lang_code>/about')
def about_i18n(lang_code):
    """Internationalized about page"""
    return f'About page in {lang_code}'

# Route testing
def test_routes():
    """Test route generation"""
    with app.test_client() as client:
        # Test various routes
        response = client.get('/tags/python,flask,web')
        print(f"Tags response: {response.data}")
        
        response = client.get('/user/alice')
        print(f"User response: {response.data}")

if __name__ == '__main__':
    # Print all registered routes
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.rule} -> {rule.endpoint} [{', '.join(rule.methods)}]")
    
    app.run(debug=True)
```

### 4.3 URL Generation and Reverse Routing

```python
# web-frameworks/examples/04_url_generation.py

from flask import Flask, url_for, request, redirect, render_template_string
from urllib.parse import urlparse, urljoin

app = Flask(__name__)

# Basic URL generation
@app.route('/')
def index():
    """Generate URLs for various endpoints"""
    
    urls = {
        # Basic URL generation
        'about': url_for('about'),
        'contact': url_for('contact'),
        
        # URLs with parameters
        'user_profile': url_for('user_profile', username='johndoe'),
        'edit_post': url_for('edit_post', post_id=123),
        
        # URLs with query parameters
        'search': url_for('search', q='flask tutorial', page=2),
        'filter': url_for('filter_items', category='books', sort='date'),
        
        # External URLs
        'external': url_for('external_site', _external=True),
        
        # URLs with fragments
        'section': url_for('documentation', _anchor='installation'),
        
        # Scheme-specific URLs
        'secure': url_for('secure_endpoint', _scheme='https'),
    }
    
    # Generate HTML with links
    html = '<h1>URL Generation Examples</h1>'
    for name, url in urls.items():
        html += f'<p><strong>{name}:</strong> <a href="{url}">{url}</a></p>'
    
    return html

@app.route('/about')
def about():
    return 'About page'

@app.route('/contact')
def contact():
    return 'Contact page'

@app.route('/user/<username>')
def user_profile(username):
    return f'User: {username}'

@app.route('/post/<int:post_id>/edit')
def edit_post(post_id):
    return f'Edit post: {post_id}'

@app.route('/search')
def search():
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)
    return f'Search: {query} (Page {page})'

@app.route('/items')
def filter_items():
    category = request.args.get('category', 'all')
    sort = request.args.get('sort', 'name')
    return f'Items - Category: {category}, Sort: {sort}'

@app.route('/external')
def external_site():
    return 'External site simulation'

@app.route('/docs')
def documentation():
    return 'Documentation page'

@app.route('/secure')
def secure_endpoint():
    return 'Secure endpoint'

# Dynamic URL building in templates
@app.route('/navigation')
def navigation():
    """Demonstrate URL building in templates"""
    
    template = """
    <nav>
        <ul>
            <li><a href="{{ url_for('index') }}">Home</a></li>
            <li><a href="{{ url_for('about') }}">About</a></li>
            <li><a href="{{ url_for('contact') }}">Contact</a></li>
            <li><a href="{{ url_for('user_profile', username='guest') }}">Profile</a></li>
        </ul>
    </nav>
    
    <h2>Search Form</h2>
    <form action="{{ url_for('search') }}" method="get">
        <input type="text" name="q" placeholder="Search...">
        <button type="submit">Search</button>
    </form>
    """
    
    return render_template_string(template)

# URL manipulation utilities
def is_safe_url(target):
    """Check if URL is safe for redirects"""
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc

@app.route('/safe-redirect')
def safe_redirect():
    """Demonstrate safe URL redirection"""
    next_url = request.args.get('next')
    
    if next_url and is_safe_url(next_url):
        return redirect(next_url)
    
    # Default redirect
    return redirect(url_for('index'))

# URL building with application context
def build_urls_outside_request():
    """Build URLs outside of request context"""
    
    with app.app_context():
        # This works outside of a request
        about_url = url_for('about')
        user_url = url_for('user_profile', username='admin')
        
        return {
            'about': about_url,
            'user': user_url
        }

# URL testing and validation
@app.route('/test-urls')
def test_urls():
    """Test URL generation"""
    
    # Test internal URL generation
    urls = []
    
    # Basic routes
    urls.append(('Home', url_for('index')))
    urls.append(('About', url_for('about')))
    
    # Parameterized routes
    urls.append(('User Profile', url_for('user_profile', username='testuser')))
    urls.append(('Edit Post', url_for('edit_post', post_id=456)))
    
    # Routes with query parameters
    urls.append(('Search', url_for('search', q='test query')))
    urls.append(('Filtered Items', url_for('filter_items', category='tech', sort='date')))
    
    # External URLs
    urls.append(('External', url_for('external_site', _external=True)))
    
    # Build HTML response
    html = '<h1>URL Testing</h1><ul>'
    for title, url in urls:
        html += f'<li><strong>{title}:</strong> <code>{url}</code></li>'
    html += '</ul>'
    
    # Test URL building outside request context
    external_urls = build_urls_outside_request()
    html += '<h2>URLs built outside request context:</h2><ul>'
    for title, url in external_urls.items():
        html += f'<li><strong>{title}:</strong> <code>{url}</code></li>'
    html += '</ul>'
    
    return html

# Blueprint URL generation
from flask import Blueprint

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/dashboard')
def dashboard():
    return 'Admin Dashboard'

@admin_bp.route('/users')
def users():
    return 'Admin Users'

app.register_blueprint(admin_bp)

@app.route('/admin-links')
def admin_links():
    """Generate URLs for blueprint routes"""
    
    admin_urls = {
        'dashboard': url_for('admin.dashboard'),
        'users': url_for('admin.users'),
    }
    
    html = '<h1>Admin Links</h1>'
    for name, url in admin_urls.items():
        html += f'<p><strong>{name}:</strong> <a href="{url}">{url}</a></p>'
    
    return html

if __name__ == '__main__':
    # Print URL map
    print("URL Map:")
    with app.app_context():
        for rule in app.url_map.iter_rules():
            options = {}
            for arg in rule.arguments:
                options[arg] = f"<{arg}>"
            
            methods = ','.join(rule.methods)
            url = url_for(rule.endpoint, **options)
            print(f"{methods:15} {rule.endpoint:25} {url}")
    
    app.run(debug=True)
```

This comprehensive Flask guide provides detailed coverage of all major Flask topics with practical examples. The guide covers everything from basic setup to advanced production patterns, making it perfect for developers at any level who want to master Flask web development.

---

## Chapter 5: Request Handling and Response Generation

### 5.1 Request Object and Data Access

```python
# web-frameworks/examples/05_request_handling.py

from flask import Flask, request, jsonify, make_response, abort, send_file
import json
from datetime import datetime
import tempfile
import os

app = Flask(__name__)

# Basic request information
@app.route('/request-info', methods=['GET', 'POST', 'PUT', 'DELETE'])
def request_info():
    """Display comprehensive request information"""
    
    info = {
        # Basic request info
        'method': request.method,
        'url': request.url,
        'base_url': request.base_url,
        'url_root': request.url_root,
        'path': request.path,
        'full_path': request.full_path,
        'endpoint': request.endpoint,
        
        # Headers
        'headers': dict(request.headers),
        'user_agent': str(request.user_agent),
        'content_type': request.content_type,
        'content_length': request.content_length,
        
        # Client info
        'remote_addr': request.remote_addr,
        'environ': {
            'REMOTE_HOST': request.environ.get('REMOTE_HOST'),
            'HTTP_X_FORWARDED_FOR': request.environ.get('HTTP_X_FORWARDED_FOR'),
            'HTTP_USER_AGENT': request.environ.get('HTTP_USER_AGENT'),
        },
        
        # Query parameters
        'args': dict(request.args),
        'query_string': request.query_string.decode('utf-8'),
        
        # Cookies
        'cookies': dict(request.cookies),
        
        # Timestamp
        'timestamp': datetime.now().isoformat()
    }
    
    # Add form data for POST requests
    if request.method == 'POST' and request.form:
        info['form_data'] = dict(request.form)
    
    # Add JSON data if present
    if request.is_json:
        info['json_data'] = request.get_json()
    
    return jsonify(info)

# Form data handling
@app.route('/form-demo', methods=['GET', 'POST'])
def form_demo():
    """Demonstrate form data handling"""
    
    if request.method == 'GET':
        return '''
        <form method="POST" enctype="multipart/form-data">
            <p>Name: <input type="text" name="name" required></p>
            <p>Email: <input type="email" name="email" required></p>
            <p>Age: <input type="number" name="age" min="1" max="120"></p>
            <p>Bio: <textarea name="bio" rows="4" cols="50"></textarea></p>
            <p>Skills: 
                <input type="checkbox" name="skills" value="python"> Python
                <input type="checkbox" name="skills" value="javascript"> JavaScript
                <input type="checkbox" name="skills" value="sql"> SQL
            </p>
            <p>Experience:
                <input type="radio" name="experience" value="beginner"> Beginner
                <input type="radio" name="experience" value="intermediate"> Intermediate
                <input type="radio" name="experience" value="advanced"> Advanced
            </p>
            <p>Resume: <input type="file" name="resume"></p>
            <p><input type="submit" value="Submit"></p>
        </form>
        '''
    
    # Process form data
    form_data = {
        'name': request.form.get('name'),
        'email': request.form.get('email'),
        'age': request.form.get('age', type=int),
        'bio': request.form.get('bio'),
        'skills': request.form.getlist('skills'),
        'experience': request.form.get('experience'),
    }
    
    # Handle file upload
    if 'resume' in request.files:
        resume = request.files['resume']
        if resume.filename:
            # Save file (in production, use secure filename and proper storage)
            resume_path = os.path.join(tempfile.gettempdir(), resume.filename)
            resume.save(resume_path)
            form_data['resume_uploaded'] = resume.filename
            form_data['resume_size'] = os.path.getsize(resume_path)
    
    return jsonify({
        'status': 'success',
        'form_data': form_data,
        'files': [f.filename for f in request.files.values() if f.filename]
    })

# JSON request handling
@app.route('/api/json-demo', methods=['POST'])
def json_demo():
    """Demonstrate JSON request handling"""
    
    # Check if request contains JSON
    if not request.is_json:
        return jsonify({'error': 'Request must contain JSON'}), 400
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields
            }), 400
        
        # Process the data
        response_data = {
            'status': 'success',
            'received_data': data,
            'processed_at': datetime.now().isoformat(),
            'validation': {
                'name_length': len(data['name']),
                'email_valid': '@' in data['email'],
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# File upload handling
@app.route('/upload', methods=['GET', 'POST'])
def file_upload():
    """Handle file uploads with validation"""
    
    if request.method == 'GET':
        return '''
        <h2>File Upload Demo</h2>
        <form method="POST" enctype="multipart/form-data">
            <p>Select files: <input type="file" name="files" multiple></p>
            <p>Description: <input type="text" name="description"></p>
            <p><input type="submit" value="Upload"></p>
        </form>
        '''
    
    # Check if files were uploaded
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    uploaded_files = request.files.getlist('files')
    description = request.form.get('description', '')
    
    # Process uploaded files
    results = []
    allowed_extensions = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}
    max_file_size = 10 * 1024 * 1024  # 10MB
    
    for file in uploaded_files:
        if file.filename == '':
            continue
        
        # Validate file
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            results.append({
                'filename': file.filename,
                'status': 'error',
                'message': f'File type not allowed. Allowed: {", ".join(allowed_extensions)}'
            })
            continue
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > max_file_size:
            results.append({
                'filename': file.filename,
                'status': 'error',
                'message': f'File too large. Maximum size: {max_file_size // (1024*1024)}MB'
            })
            continue
        
        # Save file
        try:
            save_path = os.path.join(tempfile.gettempdir(), file.filename)
            file.save(save_path)
            
            results.append({
                'filename': file.filename,
                'status': 'success',
                'size': file_size,
                'path': save_path,
                'type': file_ext
            })
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'status': 'error',
                'message': str(e)
            })
    
    return jsonify({
        'description': description,
        'files': results,
        'summary': {
            'total': len(uploaded_files),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'error')
        }
    })

# Query parameter handling
@app.route('/search')
def search():
    """Advanced query parameter handling"""
    
    # Get parameters with defaults and type conversion
    query = request.args.get('q', '', type=str)
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    sort_by = request.args.get('sort_by', 'relevance', type=str)
    order = request.args.get('order', 'desc', type=str)
    filters = request.args.getlist('filter')  # Multiple values
    include_archived = request.args.get('include_archived', False, type=bool)
    
    # Validate parameters
    errors = []
    
    if not query:
        errors.append('Query parameter "q" is required')
    
    if page < 1:
        errors.append('Page must be >= 1')
    
    if per_page < 1 or per_page > 100:
        errors.append('Per page must be between 1 and 100')
    
    if sort_by not in ['relevance', 'date', 'name']:
        errors.append('Sort by must be one of: relevance, date, name')
    
    if order not in ['asc', 'desc']:
        errors.append('Order must be either "asc" or "desc"')
    
    if errors:
        return jsonify({'errors': errors}), 400
    
    # Simulate search results
    results = {
        'query': query,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': 150,  # Mock total
            'pages': 15
        },
        'sorting': {
            'sort_by': sort_by,
            'order': order
        },
        'filters': filters,
        'include_archived': include_archived,
        'results': [
            f'Result {i} for "{query}"' 
            for i in range((page-1)*per_page + 1, min(page*per_page + 1, 151))
        ]
    }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
```

### 5.2 Response Generation and Customization

```python
# web-frameworks/examples/05_response_generation.py

from flask import Flask, jsonify, make_response, render_template_string, send_file, abort
from flask import stream_template, Response
import json
import io
import csv
from datetime import datetime, timedelta
import mimetypes

app = Flask(__name__)

# Basic response types
@app.route('/responses/text')
def text_response():
    """Return plain text response"""
    return 'This is a plain text response'

@app.route('/responses/html')
def html_response():
    """Return HTML response"""
    return '<h1>HTML Response</h1><p>This is an HTML response</p>'

@app.route('/responses/json')
def json_response():
    """Return JSON response"""
    data = {
        'message': 'This is a JSON response',
        'timestamp': datetime.now().isoformat(),
        'data': {
            'items': [1, 2, 3, 4, 5],
            'total': 5
        }
    }
    return jsonify(data)

# Custom response with headers
@app.route('/responses/custom')
def custom_response():
    """Create custom response with headers"""
    
    # Create response object
    response = make_response('Custom response with headers')
    
    # Set custom headers
    response.headers['X-Custom-Header'] = 'Custom Value'
    response.headers['X-Timestamp'] = str(datetime.now())
    response.headers['Content-Type'] = 'text/plain; charset=utf-8'
    
    # Set caching headers
    response.headers['Cache-Control'] = 'public, max-age=3600'
    response.headers['Expires'] = (datetime.now() + timedelta(hours=1)).strftime('%a, %d %b %Y %H:%M:%S GMT')
    
    return response

# Response with status codes
@app.route('/responses/created', methods=['POST'])
def created_response():
    """Return 201 Created response"""
    data = {'message': 'Resource created successfully', 'id': 123}
    return jsonify(data), 201

@app.route('/responses/not-found')
def not_found_response():
    """Return 404 Not Found response"""
    return jsonify({'error': 'Resource not found'}), 404

@app.route('/responses/error')
def error_response():
    """Return 500 Internal Server Error response"""
    return jsonify({'error': 'Internal server error'}), 500

# Redirect responses
@app.route('/responses/redirect')
def redirect_response():
    """Return redirect response"""
    response = make_response('', 302)
    response.headers['Location'] = '/responses/text'
    return response

# Cookie handling
@app.route('/cookies/set')
def set_cookie():
    """Set cookies in response"""
    
    response = make_response('Cookies have been set')
    
    # Simple cookie
    response.set_cookie('username', 'johndoe')
    
    # Cookie with options
    response.set_cookie(
        'session_token',
        'abc123def456',
        max_age=3600,  # 1 hour
        secure=False,  # Set to True in production with HTTPS
        httponly=True,  # Prevent JavaScript access
        samesite='Lax'
    )
    
    # Cookie with expiration date
    expires = datetime.now() + timedelta(days=30)
    response.set_cookie(
        'preferences',
        json.dumps({'theme': 'dark', 'language': 'en'}),
        expires=expires,
        path='/'
    )
    
    return response

@app.route('/cookies/get')
def get_cookies():
    """Read cookies from request"""
    cookies = dict(request.cookies)
    return jsonify({
        'cookies': cookies,
        'username': cookies.get('username'),
        'preferences': json.loads(cookies.get('preferences', '{}'))
    })

@app.route('/cookies/delete')
def delete_cookie():
    """Delete cookies"""
    response = make_response('Cookies deleted')
    
    # Delete cookies by setting them to expire in the past
    response.set_cookie('username', '', expires=0)
    response.set_cookie('session_token', '', expires=0)
    response.set_cookie('preferences', '', expires=0)
    
    return response

# File responses
@app.route('/download/text')
def download_text():
    """Generate and download a text file"""
    
    # Create file content
    content = "This is a generated text file\n"
    content += f"Generated at: {datetime.now()}\n"
    content += "Lines of text...\n" * 10
    
    # Create file-like object
    file_obj = io.BytesIO()
    file_obj.write(content.encode('utf-8'))
    file_obj.seek(0)
    
    return send_file(
        file_obj,
        as_attachment=True,
        download_name='generated_file.txt',
        mimetype='text/plain'
    )

@app.route('/download/csv')
def download_csv():
    """Generate and download a CSV file"""
    
    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write headers
    writer.writerow(['ID', 'Name', 'Email', 'Created'])
    
    # Write sample data
    for i in range(1, 101):
        writer.writerow([
            i,
            f'User {i}',
            f'user{i}@example.com',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])
    
    # Create response
    csv_data = output.getvalue()
    response = make_response(csv_data)
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment; filename=users.csv'
    
    return response

# Streaming responses
@app.route('/stream/text')
def stream_text():
    """Stream text response"""
    
    def generate():
        for i in range(10):
            yield f"Line {i}\n"
            import time
            time.sleep(0.5)  # Simulate processing delay
    
    return Response(generate(), mimetype='text/plain')

@app.route('/stream/json')
def stream_json():
    """Stream JSON response"""
    
    def generate():
        yield '{"items": ['
        
        for i in range(5):
            if i > 0:
                yield ','
            yield json.dumps({
                'id': i,
                'name': f'Item {i}',
                'timestamp': datetime.now().isoformat()
            })
            import time
            time.sleep(0.3)
        
        yield ']}'
    
    return Response(generate(), mimetype='application/json')

# Template streaming
@app.route('/stream/template')
def stream_template_response():
    """Stream template response"""
    
    def generate_items():
        for i in range(20):
            yield {
                'id': i,
                'name': f'Item {i}',
                'description': f'Description for item {i}'
            }
            import time
            time.sleep(0.1)
    
    template = '''
    <!DOCTYPE html>
    <html>
    <head><title>Streamed Template</title></head>
    <body>
        <h1>Items (Streamed)</h1>
        <ul>
        {% for item in items %}
            <li><strong>{{ item.name }}</strong>: {{ item.description }}</li>
        {% endfor %}
        </ul>
    </body>
    </html>
    '''
    
    return Response(
        stream_template(template, items=generate_items()),
        mimetype='text/html'
    )

# Content negotiation
@app.route('/api/data')
def content_negotiation():
    """Respond with different formats based on Accept header"""
    
    data = {
        'id': 1,
        'name': 'Sample Item',
        'description': 'This is a sample item',
        'created_at': datetime.now().isoformat()
    }
    
    # Check Accept header
    if request.headers.get('Accept') == 'application/xml':
        # Generate XML response
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<item>
    <id>{data['id']}</id>
    <name>{data['name']}</name>
    <description>{data['description']}</description>
    <created_at>{data['created_at']}</created_at>
</item>"""
        response = make_response(xml_content)
        response.headers['Content-Type'] = 'application/xml'
        return response
    
    elif 'text/csv' in request.headers.get('Accept', ''):
        # Generate CSV response
        csv_content = f"id,name,description,created_at\n{data['id']},{data['name']},{data['description']},{data['created_at']}"
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        return response
    
    else:
        # Default to JSON
        return jsonify(data)

# Response compression
@app.route('/responses/large')
def large_response():
    """Generate large response (good for testing compression)"""
    
    content = {
        'message': 'This is a large response',
        'data': ['Item ' + str(i) for i in range(10000)],
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'size': 10000
        }
    }
    
    response = jsonify(content)
    
    # Add headers for compression testing
    response.headers['X-Content-Size'] = len(response.get_data())
    
    return response

# Error responses
@app.route('/responses/validation-error')
def validation_error():
    """Return structured validation error"""
    
    error_response = {
        'error': {
            'code': 'VALIDATION_ERROR',
            'message': 'Request validation failed',
            'details': [
                {
                    'field': 'email',
                    'message': 'Invalid email format'
                },
                {
                    'field': 'age',
                    'message': 'Age must be between 18 and 120'
                }
            ]
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(error_response), 422

if __name__ == '__main__':
    app.run(debug=True)
```

## Chapter 6: Templates and Jinja2

Flask uses the Jinja2 templating engine to render dynamic HTML content.

### 6.1 Template Basics

Create a templates directory:
```bash
mkdir templates
```

Basic template (`templates/index.html`):
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title or 'Flask App' }}</title>
</head>
<body>
    <h1>{{ heading }}</h1>
    <p>{{ message }}</p>
</body>
</html>
```

Using templates in Flask:
```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', 
                         title='Home Page',
                         heading='Welcome to Flask!',
                         message='This is rendered from a template.')

@app.route('/user/<name>')
def user_profile(name):
    return render_template('user.html', username=name)
```

### 6.2 Template Inheritance

Base template (`templates/base.html`):
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Flask App{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar">
        <div class="nav-container">
            <a href="{{ url_for('home') }}" class="nav-brand">Flask App</a>
            <ul class="nav-menu">
                <li><a href="{{ url_for('home') }}">Home</a></li>
                <li><a href="{{ url_for('about') }}">About</a></li>
                <li><a href="{{ url_for('contact') }}">Contact</a></li>
            </ul>
        </div>
    </nav>
    
    <main class="content">
        {% block content %}{% endblock %}
    </main>
    
    <footer>
        <p>&copy; 2024 Flask Application</p>
    </footer>
</body>
</html>
```

Child template (`templates/home.html`):
```html
{% extends "base.html" %}

{% block title %}Home - Flask App{% endblock %}

{% block content %}
<div class="hero">
    <h1>{{ heading }}</h1>
    <p>{{ description }}</p>
    <a href="{{ url_for('about') }}" class="btn btn-primary">Learn More</a>
</div>

<div class="features">
    <h2>Features</h2>
    <div class="feature-grid">
        {% for feature in features %}
        <div class="feature-card">
            <h3>{{ feature.title }}</h3>
            <p>{{ feature.description }}</p>
        </div>
        {% endfor %}
    </div>
</div>
{% endblock %}
```

### 6.3 Template Variables and Filters

Using variables and filters:
```html
<!-- Variables -->
<h1>Hello, {{ name }}!</h1>
<p>Today is {{ date }}</p>

<!-- Filters -->
<p>{{ message | upper }}</p>
<p>{{ price | round(2) }}</p>
<p>{{ content | truncate(100) }}</p>
<p>{{ timestamp | strftime('%Y-%m-%d') }}</p>

<!-- Conditional rendering -->
{% if user.is_authenticated %}
    <p>Welcome back, {{ user.name }}!</p>
{% else %}
    <p>Please log in.</p>
{% endif %}

<!-- Loops -->
<ul>
{% for item in items %}
    <li>{{ item.name }} - ${{ item.price }}</li>
{% else %}
    <li>No items found.</li>
{% endfor %}
</ul>
```

Custom filters:
```python
from flask import Flask
from datetime import datetime
import locale

app = Flask(__name__)

@app.template_filter('datetime')
def datetime_filter(timestamp):
    """Custom filter to format datetime"""
    return timestamp.strftime('%B %d, %Y at %I:%M %p')

@app.template_filter('currency')
def currency_filter(amount):
    """Format currency"""
    return f"${amount:,.2f}"

# Register filters
app.jinja_env.filters['datetime'] = datetime_filter
app.jinja_env.filters['currency'] = currency_filter
```

## Chapter 7: Static Files and Asset Management

Flask serves static files (CSS, JavaScript, images) from the static directory.

### 7.1 Static File Structure

Recommended directory structure:
```
my_flask_app/
├── app.py
├── static/
│   ├── css/
│   │   ├── style.css
│   │   └── bootstrap.min.css
│   ├── js/
│   │   ├── main.js
│   │   └── jquery.min.js
│   ├── img/
│   │   ├── logo.png
│   │   └── favicon.ico
│   └── fonts/
│       └── custom-font.woff2
└── templates/
    └── base.html
```

### 7.2 Serving Static Files

Basic static file usage:
```html
<!-- CSS -->
<link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">

<!-- JavaScript -->
<script src="{{ url_for('static', filename='js/main.js') }}"></script>

<!-- Images -->
<img src="{{ url_for('static', filename='img/logo.png') }}" alt="Logo">

<!-- Fonts -->
<link href="{{ url_for('static', filename='fonts/custom-font.woff2') }}" rel="preload">
```

## Chapter 8: Forms and User Input

Handling forms is crucial for interactive web applications.

### 8.1 Basic Form Handling

Simple form processing:
```python
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Basic validation
        if not name or not email or not message:
            flash('All fields are required!', 'error')
            return render_template('contact.html')
        
        # Process form (send email, save to database, etc.)
        send_contact_email(name, email, message)
        flash('Message sent successfully!', 'success')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

def send_contact_email(name, email, message):
    # Email sending logic here
    pass
```

### 8.2 Flask-WTF for Advanced Forms

Install Flask-WTF:
```bash
pip install Flask-WTF
```

Form class definition:
```python
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField, SelectField, BooleanField
from wtforms.validators import DataRequired, Email, Length

class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    subject = SelectField('Subject', choices=[
        ('general', 'General Inquiry'),
        ('support', 'Support'),
        ('feedback', 'Feedback')
    ])
    message = TextAreaField('Message', validators=[DataRequired(), Length(min=10, max=500)])
    newsletter = BooleanField('Subscribe to newsletter')
    submit = SubmitField('Send Message')
```

Using forms in views:
```python
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    form = ContactForm()
    
    if form.validate_on_submit():
        # Form data is valid
        name = form.name.data
        email = form.email.data
        subject = form.subject.data
        message = form.message.data
        newsletter = form.newsletter.data
        
        # Process form data
        send_contact_email(name, email, subject, message)
        
        if newsletter:
            subscribe_to_newsletter(email)
        
        flash('Message sent successfully!', 'success')
        return redirect(url_for('contact'))
    
    return render_template('contact_form.html', form=form)
```

Form template with WTF:
```html
{% extends "base.html" %}

{% block content %}
<div class="contact-form">
    <h2>Contact Us</h2>
    
    <form method="POST" novalidate>
        {{ form.hidden_tag() }}
        
        <div class="form-group">
            {{ form.name.label(class="form-label") }}
            {{ form.name(class="form-control") }}
            {% if form.name.errors %}
                <ul class="errors">
                    {% for error in form.name.errors %}
                        <li>{{ error }}</li>
                    {% endfor %}
                </ul>
            {% endif %}
        </div>
        
        <div class="form-group">
            {{ form.email.label(class="form-label") }}
            {{ form.email(class="form-control") }}
            {% if form.email.errors %}
                <ul class="errors">
                    {% for error in form.email.errors %}
                        <li>{{ error }}</li>
                    {% endfor %}
                </ul>
            {% endif %}
        </div>
        
        <div class="form-group">
            {{ form.subject.label(class="form-label") }}
            {{ form.subject(class="form-control") }}
        </div>
        
        <div class="form-group">
            {{ form.message.label(class="form-label") }}
            {{ form.message(class="form-control", rows="5") }}
            {% if form.message.errors %}
                <ul class="errors">
                    {% for error in form.message.errors %}
                        <li>{{ error }}</li>
                    {% endfor %}
                </ul>
            {% endif %}
        </div>
        
        <div class="form-group">
            {{ form.newsletter() }}
            {{ form.newsletter.label() }}
        </div>
        
        {{ form.submit(class="btn btn-primary") }}
    </form>
</div>
{% endblock %}
```

## Chapter 9: Database Integration

Flask applications commonly use databases to store and retrieve data.

### 9.1 Flask-SQLAlchemy Setup

Install Flask-SQLAlchemy:
```bash
pip install Flask-SQLAlchemy
```

Basic database configuration:
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    posts = db.relationship('Post', backref='author', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign key
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    def __repr__(self):
        return f'<Post {self.title}>'

# Create tables
with app.app_context():
    db.create_all()
```

### 9.2 Database Operations (CRUD)

Create operations:
```python
@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    
    # Validation
    if not data.get('username') or not data.get('email'):
        return jsonify({'error': 'Username and email required'}), 400
    
    # Check if user exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    # Create new user
    user = User(
        username=data['username'],
        email=data['email'],
        password_hash=hash_password(data.get('password', ''))
    )
    
    try:
        db.session.add(user)
        db.session.commit()
        return jsonify({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'created_at': user.created_at.isoformat()
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/posts', methods=['POST'])
def create_post():
    data = request.get_json()
    
    # Validation
    if not data.get('title') or not data.get('content') or not data.get('user_id'):
        return jsonify({'error': 'Title, content, and user_id required'}), 400
    
    # Verify user exists
    user = User.query.get(data['user_id'])
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Create post
    post = Post(
        title=data['title'],
        content=data['content'],
        user_id=data['user_id']
    )
    
    try:
        db.session.add(post)
        db.session.commit()
        return jsonify({
            'id': post.id,
            'title': post.title,
            'content': post.content,
            'author': post.author.username,
            'created_at': post.created_at.isoformat()
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
```

Read operations:
```python
@app.route('/users')
def get_users():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search = request.args.get('search', '', type=str)
    
    # Build query
    query = User.query
    
    if search:
        query = query.filter(
            User.username.contains(search) | 
            User.email.contains(search)
        )
    
    # Paginate
    pagination = query.paginate(
        page=page, 
        per_page=per_page, 
        error_out=False
    )
    
    users = pagination.items
    
    return jsonify({
        'users': [{
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'created_at': user.created_at.isoformat(),
            'post_count': len(user.posts)
        } for user in users],
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': pagination.total,
            'pages': pagination.pages,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev
        }
    })

@app.route('/users/<int:user_id>')
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'created_at': user.created_at.isoformat(),
        'posts': [{
            'id': post.id,
            'title': post.title,
            'created_at': post.created_at.isoformat()
        } for post in user.posts]
    })

@app.route('/posts')
def get_posts():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    author_id = request.args.get('author_id', type=int)
    
    # Build query
    query = Post.query
    
    if author_id:
        query = query.filter_by(user_id=author_id)
    
    # Order by creation date (newest first)
    query = query.order_by(Post.created_at.desc())
    
    # Paginate
    pagination = query.paginate(
        page=page,
        per_page=per_page,
        error_out=False
    )
    
    posts = pagination.items
    
    return jsonify({
        'posts': [{
            'id': post.id,
            'title': post.title,
            'content': post.content[:200] + '...' if len(post.content) > 200 else post.content,
            'author': post.author.username,
            'created_at': post.created_at.isoformat(),
            'updated_at': post.updated_at.isoformat()
        } for post in posts],
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': pagination.total,
            'pages': pagination.pages
        }
    })
```

Update operations:
```python
@app.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    # Update fields
    if 'username' in data:
        # Check username uniqueness
        existing = User.query.filter(
            User.username == data['username'], 
            User.id != user_id
        ).first()
        if existing:
            return jsonify({'error': 'Username already exists'}), 400
        user.username = data['username']
    
    if 'email' in data:
        # Check email uniqueness
        existing = User.query.filter(
            User.email == data['email'], 
            User.id != user_id
        ).first()
        if existing:
            return jsonify({'error': 'Email already exists'}), 400
        user.email = data['email']
    
    try:
        db.session.commit()
        return jsonify({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'updated_at': datetime.utcnow().isoformat()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/posts/<int:post_id>', methods=['PUT'])
def update_post(post_id):
    post = Post.query.get_or_404(post_id)
    data = request.get_json()
    
    # Update fields
    if 'title' in data:
        post.title = data['title']
    if 'content' in data:
        post.content = data['content']
    
    # Update timestamp
    post.updated_at = datetime.utcnow()
    
    try:
        db.session.commit()
        return jsonify({
            'id': post.id,
            'title': post.title,
            'content': post.content,
            'updated_at': post.updated_at.isoformat()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
```

Delete operations:
```python
@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    
    try:
        # Delete associated posts first
        Post.query.filter_by(user_id=user_id).delete()
        
        # Delete user
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({'message': 'User deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/posts/<int:post_id>', methods=['DELETE'])
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    
    try:
        db.session.delete(post)
        db.session.commit()
        return jsonify({'message': 'Post deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
```

### 9.3 Database Migrations with Flask-Migrate

Install Flask-Migrate:
```bash
pip install Flask-Migrate
```

Setup migrations:
```python
from flask_migrate import Migrate

migrate = Migrate(app, db)
```

Initialize migration repository:
```bash
flask db init
```

Create initial migration:
```bash
flask db migrate -m "Initial migration"
```

Apply migration:
```bash
flask db upgrade
```

## Chapter 10: Authentication and Security

Implementing user authentication and security measures.

### 10.1 Password Hashing

Use Werkzeug for password hashing:
```python
from werkzeug.security import generate_password_hash, check_password_hash

def hash_password(password):
    """Hash a password"""
    return generate_password_hash(password)

def verify_password(password_hash, password):
    """Verify a password against its hash"""
    return check_password_hash(password_hash, password)

# Update User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
```

### 10.2 Session Management

Basic session handling:
```python
from flask import session

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    # Find user
    user = User.query.filter_by(email=email).first()
    
    if user and user.check_password(password):
        # Create session
        session['user_id'] = user.id
        session['username'] = user.username
        session.permanent = True  # Use permanent session
        
        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        })
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'})

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'created_at': user.created_at.isoformat()
    })
```

### 10.3 Flask-Login Integration

Install Flask-Login:
```bash
pip install Flask-Login
```

Setup Flask-Login:
```python
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Update User model to inherit from UserMixin
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Updated login route
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    remember = data.get('remember', False)
    
    user = User.query.filter_by(email=email).first()
    
    if user and user.check_password(password):
        login_user(user, remember=remember)
        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        })
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'})

@app.route('/protected')
@login_required
def protected():
    return jsonify({
        'message': 'This is a protected route',
        'user': {
            'id': current_user.id,
            'username': current_user.username
        }
    })
```

### 10.4 JWT Authentication

Install PyJWT:
```bash
pip install PyJWT
```

JWT implementation:
```python
import jwt
from datetime import datetime, timedelta
from functools import wraps

def generate_token(user_id):
    """Generate JWT token"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=24),
        'iat': datetime.utcnow()
    }
    
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'Token missing'}), 401
        
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            user_id = verify_token(token)
            if not user_id:
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            # Get user and add to request context
            current_user = User.query.get(user_id)
            if not current_user:
                return jsonify({'error': 'User not found'}), 404
            
            request.current_user = current_user
            
        except Exception as e:
            return jsonify({'error': 'Token verification failed'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

# JWT login endpoint
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    user = User.query.filter_by(email=email).first()
    
    if user and user.check_password(password):
        token = generate_token(user.id)
        return jsonify({
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        })
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

# Protected API endpoint
@app.route('/api/profile')
@token_required
def api_profile():
    user = request.current_user
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'created_at': user.created_at.isoformat()
    })
```

## Chapter 11: Testing Flask Applications

Testing is crucial for maintaining reliable Flask applications.

### 11.1 Basic Testing Setup

Install testing dependencies:
```bash
pip install pytest pytest-flask
```

Basic test configuration (`conftest.py`):
```python
import pytest
from app import create_app, db
from app.models import User, Post

@pytest.fixture
def app():
    """Create application for testing"""
    app = create_app('testing')
    
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Create test runner"""
    return app.test_cli_runner()

@pytest.fixture
def user(app):
    """Create test user"""
    user = User(username='testuser', email='test@example.com')
    user.set_password('testpass')
    db.session.add(user)
    db.session.commit()
    return user
```

### 11.2 Unit Testing Routes

Testing basic routes:
```python
# tests/test_routes.py
import json

def test_home_page(client):
    """Test home page loads"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Welcome' in response.data

def test_api_users_get(client):
    """Test getting users list"""
    response = client.get('/users')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'users' in data
    assert 'pagination' in data

def test_create_user(client):
    """Test user creation"""
    user_data = {
        'username': 'newuser',
        'email': 'new@example.com',
        'password': 'newpass'
    }
    
    response = client.post('/users', 
                          json=user_data,
                          content_type='application/json')
    
    assert response.status_code == 201
    data = json.loads(response.data)
    assert data['username'] == 'newuser'
    assert data['email'] == 'new@example.com'

def test_create_user_validation(client):
    """Test user creation validation"""
    # Missing required fields
    response = client.post('/users', 
                          json={},
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_get_user(client, user):
    """Test getting specific user"""
    response = client.get(f'/users/{user.id}')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['username'] == user.username
    assert data['email'] == user.email

def test_update_user(client, user):
    """Test user update"""
    update_data = {'username': 'updateduser'}
    
    response = client.put(f'/users/{user.id}',
                         json=update_data,
                         content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['username'] == 'updateduser'

def test_delete_user(client, user):
    """Test user deletion"""
    response = client.delete(f'/users/{user.id}')
    assert response.status_code == 200
    
    # Verify user is deleted
    response = client.get(f'/users/{user.id}')
    assert response.status_code == 404
```

### 11.3 Testing Authentication

Authentication testing:
```python
# tests/test_auth.py
import json

def test_login_success(client, user):
    """Test successful login"""
    login_data = {
        'email': user.email,
        'password': 'testpass'
    }
    
    response = client.post('/api/login',
                          json=login_data,
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'token' in data
    assert data['user']['email'] == user.email

def test_login_invalid_credentials(client, user):
    """Test login with invalid credentials"""
    login_data = {
        'email': user.email,
        'password': 'wrongpass'
    }
    
    response = client.post('/api/login',
                          json=login_data,
                          content_type='application/json')
    
    assert response.status_code == 401
    data = json.loads(response.data)
    assert 'error' in data

def test_protected_route_without_token(client):
    """Test accessing protected route without token"""
    response = client.get('/api/profile')
    assert response.status_code == 401

def test_protected_route_with_token(client, user):
    """Test accessing protected route with valid token"""
    # Login to get token
    login_data = {
        'email': user.email,
        'password': 'testpass'
    }
    
    login_response = client.post('/api/login',
                                json=login_data,
                                content_type='application/json')
    
    token = json.loads(login_response.data)['token']
    
    # Access protected route
    headers = {'Authorization': f'Bearer {token}'}
    response = client.get('/api/profile', headers=headers)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['email'] == user.email
```

### 11.4 Database Testing

Testing database operations:
```python
# tests/test_models.py
from app.models import User, Post
from app import db

def test_user_creation(app):
    """Test user model creation"""
    with app.app_context():
        user = User(username='testuser', email='test@example.com')
        user.set_password('testpass')
        
        db.session.add(user)
        db.session.commit()
        
        # Verify user was created
        created_user = User.query.filter_by(email='test@example.com').first()
        assert created_user is not None
        assert created_user.username == 'testuser'
        assert created_user.check_password('testpass')

def test_user_password_hashing(app):
    """Test password hashing"""
    with app.app_context():
        user = User(username='testuser', email='test@example.com')
        user.set_password('secret')
        
        # Password should be hashed
        assert user.password_hash != 'secret'
        assert user.check_password('secret')
        assert not user.check_password('wrong')

def test_post_creation(app, user):
    """Test post creation with relationship"""
    with app.app_context():
        post = Post(
            title='Test Post',
            content='This is test content',
            user_id=user.id
        )
        
        db.session.add(post)
        db.session.commit()
        
        # Verify post was created
        created_post = Post.query.filter_by(title='Test Post').first()
        assert created_post is not None
        assert created_post.author.username == user.username

def test_user_posts_relationship(app, user):
    """Test user-posts relationship"""
    with app.app_context():
        # Create posts
        post1 = Post(title='Post 1', content='Content 1', user_id=user.id)
        post2 = Post(title='Post 2', content='Content 2', user_id=user.id)
        
        db.session.add_all([post1, post2])
        db.session.commit()
        
        # Verify relationship
        user_posts = user.posts
        assert len(user_posts) == 2
        assert post1 in user_posts
        assert post2 in user_posts
```

## Chapter 12: Deployment and Production

Deploying Flask applications to production environments.

### 12.1 Production Configuration

Production configuration (`config.py`):
```python
import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Mail configuration
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite:///app_dev.db'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://user:password@localhost/app_prod'
    
    # Security headers
    SECURITY_HEADERS = {
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'SAMEORIGIN',
        'X-XSS-Protection': '1; mode=block',
        'Content-Security-Policy': "default-src 'self'"
    }

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
```

Application factory with configuration:
```python
# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from config import config

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()

def create_app(config_name=None):
    """Application factory"""
    app = Flask(__name__)
    
    # Load configuration
    config_name = config_name or os.environ.get('FLASK_ENV', 'default')
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    # Register blueprints
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Add security headers (production)
    if app.config.get('SECURITY_HEADERS'):
        @app.after_request
        def add_security_headers(response):
            for header, value in app.config['SECURITY_HEADERS'].items():
                response.headers[header] = value
            return response
    
    return app
```

### 12.2 WSGI Server Setup

Using Gunicorn for production:
```bash
pip install gunicorn
```

Gunicorn configuration (`gunicorn.conf.py`):
```python
# Gunicorn configuration
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'flask_app'

# Server mechanics
preload_app = True
daemon = False
pidfile = 'gunicorn.pid'
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None
```

Starting the application:
```bash
# Development
flask run

# Production with Gunicorn
gunicorn -c gunicorn.conf.py "app:create_app('production')"
```

### 12.3 Docker Deployment

Dockerfile:
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:create_app('production')"]
```

Docker Compose (`docker-compose.yml`):
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://user:password@db:5432/flask_app
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: flask_app
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./static:/var/www/static
    depends_on:
      - web
    restart: unless-stopped

volumes:
  postgres_data:
```

### 12.4 Monitoring and Logging

Application logging setup:
```python
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(app):
    """Setup application logging"""
    if not app.debug and not app.testing:
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        # File handler
        file_handler = RotatingFileHandler(
            'logs/flask_app.log',
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('Flask application startup')

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check database connection
        db.session.execute('SELECT 1')
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0'
        })
    except Exception as e:
        app.logger.error(f'Health check failed: {str(e)}')
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500
```

## Chapter 13: Complete Flask Project Example

A complete Flask blog application demonstrating all concepts.

### 13.1 Project Structure

Complete project structure:
```
flask_blog/
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── main/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── forms.py
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── forms.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   └── templates/
│       ├── base.html
│       ├── index.html
│       ├── auth/
│       │   ├── login.html
│       │   └── register.html
│       └── blog/
│           ├── post.html
│           └── create_post.html
├── migrations/
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_auth.py
│   └── test_routes.py
├── static/
│   ├── css/
│   ├── js/
│   └── img/
├── config.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── gunicorn.conf.py
└── flask_blog.py
```

### 13.2 Main Application Entry Point

Main application file (`flask_blog.py`):
```python
import os
from app import create_app, db
from app.models import User, Post
from flask_migrate import upgrade

app = create_app(os.getenv('FLASK_CONFIG') or 'default')

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Post': Post}

@app.cli.command()
def deploy():
    """Run deployment tasks"""
    # Migrate database to latest revision
    upgrade()
    
    # Create or update roles
    from app.models import Role
    Role.insert_roles()

if __name__ == '__main__':
    app.run()
```

This comprehensive Flask guide covers all essential aspects of Flask web development, from basic concepts to advanced production deployment, providing developers with the knowledge needed to build robust, scalable web applications.