# LangChain Complete Guide: From Basic to Advanced

*A comprehensive, step-by-step guide to mastering LangChain development*

## Table of Contents

- [Chapter 1: Installation and Environment Setup](#chapter-1-installation-and-environment-setup)
- [Chapter 2: Understanding LangChain Core Concepts](#chapter-2-understanding-langchain-core-concepts)
- [Chapter 3: Language Model Integration](#chapter-3-language-model-integration)
- [Chapter 4: Prompts and Prompt Templates](#chapter-4-prompts-and-prompt-templates)
- [Chapter 5: Chains - Building Sequential Operations](#chapter-5-chains---building-sequential-operations)
- [Chapter 6: Memory Management](#chapter-6-memory-management)
- [Chapter 7: Document Processing and Retrieval](#chapter-7-document-processing-and-retrieval)
- [Chapter 8: Vector Stores and Embeddings](#chapter-8-vector-stores-and-embeddings)
- [Chapter 9: Agents and Tools](#chapter-9-agents-and-tools)
- [Chapter 10: Advanced Patterns and Optimization](#chapter-10-advanced-patterns-and-optimization)
- [Chapter 11: Real-World Applications](#chapter-11-real-world-applications)
- [Chapter 12: Testing Strategies](#chapter-12-testing-strategies)
- [Chapter 13: Production Deployment](#chapter-13-production-deployment)
- [Chapter 14: Debugging and Troubleshooting](#chapter-14-debugging-and-troubleshooting)
- [Chapter 15: LangSmith - Observability and Debugging](#chapter-15-langsmith---observability-and-debugging)

---

## Chapter 1: Installation and Environment Setup

### 1.1 Prerequisites and System Requirements

Before diving into LangChain development, ensure your system meets these requirements:

#### System Requirements
- **Python**: 3.8 or higher (3.11+ recommended)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large models)
- **Storage**: At least 10GB free space
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)

#### Required Knowledge
- Basic Python programming
- Understanding of APIs and REST services
- Familiarity with JSON and data structures
- Basic understanding of machine learning concepts (helpful but not required)

### 1.2 Step-by-Step Installation

#### Step 1: Create Virtual Environment

```bash
# Create a new virtual environment
python -m venv langchain_env

# Activate the environment
# On Windows:
langchain_env\Scripts\activate

# On macOS/Linux:
source langchain_env/bin/activate
```

#### Step 2: Core LangChain Installation

```bash
# Install core LangChain
pip install langchain

# Install LangChain Community (additional integrations)
pip install langchain-community

# Install LangChain Core (fundamental components)
pip install langchain-core

# Install OpenAI integration
pip install langchain-openai

# Install additional useful packages
pip install python-dotenv
pip install requests
pip install numpy
pip install pandas
```

#### Step 3: Optional Dependencies for Advanced Features

```bash
# For vector stores and embeddings
pip install chromadb
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install pinecone-client

# For document processing
pip install pypdf
pip install docx2txt
pip install unstructured

# For web scraping
pip install beautifulsoup4
pip install selenium

# For async support
pip install aiohttp
pip install asyncio

# For production deployment
pip install fastapi
pip install uvicorn
pip install gunicorn
```

### 1.3 Environment Configuration

#### Create Environment Variables File

Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=your_project_name
```

#### Installation Verification Script

Create `verify_installation.py`:

```python
# verify_installation.py
import sys
import subprocess
import importlib
from typing import Dict, List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_package_installation(packages: List[str]) -> Dict[str, Tuple[bool, str]]:
    """Check if required packages are installed"""
    results = {}
    
    for package in packages:
        try:
            module = importlib.import_module(package.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            results[package] = (True, f"✅ {package} {version}")
        except ImportError:
            results[package] = (False, f"❌ {package} not installed")
    
    return results

def check_environment_variables(env_vars: List[str]) -> Dict[str, Tuple[bool, str]]:
    """Check if environment variables are set"""
    import os
    results = {}
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Hide sensitive values
            display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            results[var] = (True, f"✅ {var}={display_value}")
        else:
            results[var] = (False, f"❌ {var} not set")
    
    return results

def test_basic_langchain_functionality():
    """Test basic LangChain functionality"""
    try:
        from langchain.schema import BaseMessage
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        
        # Test prompt template
        template = PromptTemplate(
            input_variables=["topic"],
            template="Tell me about {topic}"
        )
        
        prompt = template.format(topic="Python")
        
        print("✅ Basic LangChain functionality working")
        print(f"   Sample prompt: {prompt[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Basic LangChain functionality failed: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection"""
    try:
        import os
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  OpenAI API key not found - skipping connection test")
            return False
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            max_tokens=10,
            temperature=0
        )
        
        response = llm.invoke("Hello")
        print("✅ OpenAI connection successful")
        print(f"   Response: {response.content[:30]}...")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

def main():
    """Run comprehensive installation verification"""
    print("🔍 LangChain Installation Verification")
    print("=" * 50)
    
    # Check Python version
    python_ok, python_msg = check_python_version()
    print(f"\n📋 Python Version:")
    print(f"   {python_msg}")
    
    if not python_ok:
        print("\n❌ Python version incompatible. Please upgrade to Python 3.8+")
        return
    
    # Check core packages
    core_packages = [
        'langchain',
        'langchain_community',
        'langchain_core',
        'langchain_openai',
        'dotenv',
        'requests'
    ]
    
    print(f"\n📦 Core Packages:")
    package_results = check_package_installation(core_packages)
    for package, (success, message) in package_results.items():
        print(f"   {message}")
    
    # Check optional packages
    optional_packages = [
        'chromadb',
        'faiss',
        'numpy',
        'pandas',
        'pypdf',
        'fastapi'
    ]
    
    print(f"\n📦 Optional Packages:")
    optional_results = check_package_installation(optional_packages)
    for package, (success, message) in optional_results.items():
        print(f"   {message}")
    
    # Check environment variables
    env_vars = [
        'OPENAI_API_KEY',
        'LANGCHAIN_API_KEY',
        'LANGCHAIN_PROJECT'
    ]
    
    print(f"\n🔧 Environment Variables:")
    env_results = check_environment_variables(env_vars)
    for var, (success, message) in env_results.items():
        print(f"   {message}")
    
    # Test basic functionality
    print(f"\n🧪 Functionality Tests:")
    test_basic_langchain_functionality()
    test_openai_connection()
    
    # Summary
    total_core = len(core_packages)
    successful_core = sum(1 for success, _ in package_results.values() if success)
    
    print(f"\n📊 Summary:")
    print(f"   Core packages: {successful_core}/{total_core} installed")
    
    if successful_core == total_core:
        print("🎉 Installation verification complete! You're ready to start with LangChain.")
    else:
        print("⚠️  Some core packages missing. Please install missing packages.")
        print("\nTo install missing packages:")
        for package, (success, _) in package_results.items():
            if not success:
                print(f"   pip install {package}")

if __name__ == "__main__":
    main()
```

Run the verification:

```bash
python verify_installation.py
```

### 1.4 Project Structure Setup

Create a comprehensive project structure for LangChain development:

```bash
# Create project directory structure
mkdir -p langchain_project/{src,tests,config,data,docs,scripts,examples}

# Navigate to project
cd langchain_project

# Create subdirectories
mkdir -p src/{chains,agents,prompts,memory,retrievers,tools,utils}
mkdir -p tests/{unit,integration,e2e}
mkdir -p data/{documents,embeddings,indexes}
mkdir -p config/{development,production}
mkdir -p examples/{basic,intermediate,advanced}
```

#### Complete Directory Structure

```
langchain_project/
├── src/                          # Source code
│   ├── __init__.py
│   ├── chains/                   # Custom chains
│   │   ├── __init__.py
│   │   ├── base_chain.py
│   │   └── custom_chains.py
│   ├── agents/                   # Custom agents
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   └── custom_agents.py
│   ├── prompts/                  # Prompt templates
│   │   ├── __init__.py
│   │   └── templates.py
│   ├── memory/                   # Memory implementations
│   │   ├── __init__.py
│   │   └── custom_memory.py
│   ├── retrievers/               # Document retrievers
│   │   ├── __init__.py
│   │   └── custom_retrievers.py
│   ├── tools/                    # Custom tools
│   │   ├── __init__.py
│   │   └── custom_tools.py
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       └── helpers.py
├── tests/                        # Test files
│   ├── __init__.py
│   ├── conftest.py              # Pytest configuration
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── e2e/                     # End-to-end tests
├── config/                       # Configuration files
│   ├── development/
│   │   ├── .env.development
│   │   └── config.yaml
│   └── production/
│       ├── .env.production
│       └── config.yaml
├── data/                         # Data files
│   ├── documents/               # Source documents
│   ├── embeddings/              # Pre-computed embeddings
│   └── indexes/                 # Vector store indexes
├── docs/                         # Documentation
│   ├── api/                     # API documentation
│   ├── guides/                  # User guides
│   └── README.md
├── scripts/                      # Utility scripts
│   ├── setup.py
│   ├── deploy.py
│   └── data_preprocessing.py
├── examples/                     # Example implementations
│   ├── basic/
│   ├── intermediate/
│   └── advanced/
├── requirements.txt              # Python dependencies
├── requirements-dev.txt          # Development dependencies
├── .env                         # Environment variables
├── .gitignore                   # Git ignore rules
├── pytest.ini                  # Pytest configuration
├── setup.py                     # Package setup
└── README.md                    # Project documentation
```

### 1.5 Initial Configuration Files

#### requirements.txt
```txt
# Core LangChain packages
langchain>=0.1.0
langchain-community>=0.0.10
langchain-core>=0.1.0
langchain-openai>=0.0.5

# Environment and utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
requests>=2.28.0
httpx>=0.24.0

# Document processing
pypdf>=3.0.0
python-docx>=0.8.11
unstructured>=0.10.0

# Vector stores and embeddings
chromadb>=0.4.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0

# Async support
aiohttp>=3.8.0
asyncio
```

#### requirements-dev.txt
```txt
# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0

# Code quality
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0
mypy>=1.0.0

# Documentation
sphinx>=7.0.0
sphinx-rtd-theme>=1.3.0

# Development tools
jupyter>=1.0.0
ipython>=8.0.0
```

#### .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
langchain_env/

# Environment variables
.env
.env.local
.env.development
.env.production

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Data files
data/documents/*.pdf
data/documents/*.docx
data/embeddings/
data/indexes/

# Logs
logs/
*.log

# Jupyter
.ipynb_checkpoints

# Testing
.coverage
htmlcov/
.pytest_cache/

# OS
.DS_Store
Thumbs.db
```

#### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=html
    --cov-report=term-missing
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    requires_api: Tests that require API keys
```

#### Basic Configuration Module

Create `src/utils/config.py`:

```python
# src/utils/config.py
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import yaml

class Config:
    """Configuration management for LangChain applications"""
    
    def __init__(self, env: str = "development"):
        self.env = env
        self._load_environment_variables()
        self._load_config_file()
    
    def _load_environment_variables(self):
        """Load environment variables from .env files"""
        # Load base .env file
        load_dotenv()
        
        # Load environment-specific .env file
        env_file = f".env.{self.env}"
        if os.path.exists(env_file):
            load_dotenv(env_file, override=True)
    
    def _load_config_file(self):
        """Load configuration from YAML file"""
        config_path = Path(f"config/{self.env}/config.yaml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        # First check environment variables
        env_value = os.getenv(key.upper())
        if env_value is not None:
            return env_value
        
        # Then check config file
        keys = key.lower().split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key"""
        return self.get('OPENAI_API_KEY')
    
    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Get Anthropic API key"""
        return self.get('ANTHROPIC_API_KEY')
    
    @property
    def huggingface_api_token(self) -> Optional[str]:
        """Get HuggingFace API token"""
        return self.get('HUGGINGFACE_API_TOKEN')
    
    @property
    def langchain_api_key(self) -> Optional[str]:
        """Get LangChain API key"""
        return self.get('LANGCHAIN_API_KEY')
    
    @property
    def database_url(self) -> str:
        """Get database URL"""
        return self.get('DATABASE_URL', 'sqlite:///langchain.db')
    
    def get_llm_config(self, provider: str = 'openai') -> Dict[str, Any]:
        """Get LLM configuration"""
        base_config = self.config.get('llm', {}).get(provider, {})
        
        if provider == 'openai':
            base_config['openai_api_key'] = self.openai_api_key
        elif provider == 'anthropic':
            base_config['anthropic_api_key'] = self.anthropic_api_key
        
        return base_config

# Global config instance
config = Config()
```

#### Create Basic Configuration YAML

Create `config/development/config.yaml`:

```yaml
# Development Configuration
app:
  name: "LangChain Development"
  debug: true
  log_level: "DEBUG"

llm:
  openai:
    model_name: "gpt-3.5-turbo"
    temperature: 0.7
    max_tokens: 1000
    timeout: 30
  
  anthropic:
    model_name: "claude-3-sonnet-20240229"
    max_tokens: 1000
    temperature: 0.7

embeddings:
  provider: "openai"
  model: "text-embedding-ada-002"
  chunk_size: 1000
  chunk_overlap: 200

vector_store:
  provider: "chroma"
  collection_name: "langchain_docs"
  persist_directory: "./data/chroma_db"

memory:
  type: "conversation_buffer"
  max_token_limit: 2000

retrieval:
  search_type: "similarity"
  k: 4
  score_threshold: 0.5

agents:
  max_iterations: 10
  max_execution_time: 300
```

### 1.6 Basic Hello World Example

Create `examples/basic/hello_world.py`:

```python
# examples/basic/hello_world.py
"""
Basic LangChain Hello World Example
This demonstrates the most fundamental LangChain concepts
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseMessage

# Load environment variables
load_dotenv()

def basic_llm_example():
    """Basic LLM interaction example"""
    print("🤖 Basic LLM Example")
    print("-" * 30)
    
    try:
        # Initialize the LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100
        )
        
        # Simple message
        response = llm.invoke("Hello! Tell me a fun fact about Python programming.")
        
        print(f"LLM Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def prompt_template_example():
    """Prompt template example"""
    print("\n📝 Prompt Template Example")
    print("-" * 30)
    
    try:
        # Create a prompt template
        template = PromptTemplate(
            input_variables=["topic", "audience"],
            template="Explain {topic} to a {audience} in simple terms."
        )
        
        # Format the prompt
        prompt = template.format(
            topic="machine learning",
            audience="5-year-old"
        )
        
        print(f"Formatted Prompt: {prompt}")
        
        # Use with LLM
        llm = ChatOpenAI(temperature=0.5, max_tokens=150)
        response = llm.invoke(prompt)
        
        print(f"LLM Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def chain_example():
    """Simple chain example"""
    print("\n🔗 Chain Example")
    print("-" * 30)
    
    try:
        # Initialize LLM
        llm = ChatOpenAI(temperature=0.7, max_tokens=200)
        
        # Create prompt template
        prompt = PromptTemplate(
            input_variables=["product"],
            template="Write a creative marketing slogan for {product}. Make it catchy and memorable."
        )
        
        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run chain
        result = chain.run(product="eco-friendly water bottles")
        
        print(f"Marketing Slogan: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function to run all examples"""
    print("🚀 LangChain Hello World Examples")
    print("=" * 40)
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Run examples
    examples = [
        basic_llm_example,
        prompt_template_example,
        chain_example
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Example failed: {e}")
    
    print("\n✅ Hello World examples complete!")
    print("\nNext steps:")
    print("1. Explore the examples/ directory for more complex examples")
    print("2. Read the comprehensive guide chapters")
    print("3. Build your first LangChain application")

if __name__ == "__main__":
    main()
```

Run the hello world example:

```bash
python examples/basic/hello_world.py
```

### 1.7 Development Environment Setup

#### Set up pre-commit hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

Install and setup:

```bash
pip install pre-commit
pre-commit install
```

#### Create Makefile for common tasks

```makefile
# Makefile

.PHONY: install test lint format clean docs

# Installation
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

install-prod:
	pip install -r requirements.txt

# Testing
test:
	pytest tests/

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

test-coverage:
	pytest --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src tests
	mypy src

format:
	black src tests examples
	isort src tests examples

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/

# Development
dev-setup:
	python -m venv venv
	source venv/bin/activate && pip install -r requirements-dev.txt
	pre-commit install

# Verification
verify:
	python verify_installation.py
```

---

## Chapter 2: Understanding LangChain Core Concepts

### 2.1 What is LangChain?

LangChain is a powerful framework designed to simplify the development of applications using large language models (LLMs). It provides a comprehensive toolkit for building, deploying, and maintaining LLM-powered applications.

#### Key Features of LangChain:

1. **Modular Architecture**: Components that can be mixed and matched
2. **Chain Abstraction**: Sequential operations made simple
3. **Memory Management**: Maintain context across interactions
4. **Tool Integration**: Connect LLMs with external systems
5. **Agent Framework**: Autonomous decision-making capabilities
6. **Document Processing**: Advanced text handling and retrieval

### 2.2 Core Components Overview

Create `examples/basic/core_components.py`:

```python
# examples/basic/core_components.py
"""
Understanding LangChain Core Components
This example demonstrates the fundamental building blocks of LangChain
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# Core LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent

load_dotenv()

class LangChainComponentsDemo:
    """Demonstration of core LangChain components"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=200
        )
        print("🔧 LangChain Components Demo Initialized")
    
    def demonstrate_llms(self):
        """1. Language Model Interface"""
        print("\n1️⃣ Language Model Interface")
        print("-" * 40)
        
        try:
            # Direct LLM call
            response = self.llm.invoke("What is artificial intelligence?")
            print(f"LLM Response: {response.content[:100]}...")
            
            # Batch processing
            messages = [
                "What is Python?",
                "What is machine learning?",
                "What is natural language processing?"
            ]
            
            responses = self.llm.batch(messages)
            print(f"\nBatch processing: {len(responses)} responses generated")
            
            return True
            
        except Exception as e:
            print(f"❌ LLM demonstration failed: {e}")
            return False
    
    def demonstrate_prompts(self):
        """2. Prompt Templates"""
        print("\n2️⃣ Prompt Templates")
        print("-" * 40)
        
        try:
            # Basic prompt template
            basic_template = PromptTemplate(
                input_variables=["topic"],
                template="Explain {topic} in simple terms with an example."
            )
            
            prompt = basic_template.format(topic="blockchain")
            print(f"Basic Template: {prompt[:60]}...")
            
            # Chat prompt template
            chat_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful technical writer."),
                ("user", "Write a brief explanation of {concept} for {audience}")
            ])
            
            chat_prompt = chat_template.format_messages(
                concept="quantum computing",
                audience="beginners"
            )
            
            print(f"Chat Template: {len(chat_prompt)} messages created")
            
            # Use with LLM
            response = self.llm.invoke(chat_prompt)
            print(f"Response: {response.content[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Prompt demonstration failed: {e}")
            return False
    
    def demonstrate_chains(self):
        """3. Chains for Sequential Operations"""
        print("\n3️⃣ Chains")
        print("-" * 40)
        
        try:
            # Simple LLM Chain
            prompt = PromptTemplate(
                input_variables=["product"],
                template="Generate 3 creative names for a {product} startup:"
            )
            
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                verbose=True
            )
            
            result = chain.run(product="sustainable fashion")
            print(f"Chain Result: {result[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Chain demonstration failed: {e}")
            return False
    
    def demonstrate_memory(self):
        """4. Memory for Context Management"""
        print("\n4️⃣ Memory")
        print("-" * 40)
        
        try:
            # Initialize memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Create conversational chain
            template = """
            Previous conversation:
            {chat_history}
            
            Human: {human_input}
            Assistant:"""
            
            prompt = PromptTemplate(
                input_variables=["chat_history", "human_input"],
                template=template
            )
            
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                memory=memory,
                verbose=True
            )
            
            # Simulate conversation
            response1 = chain.run(human_input="Hi, I'm learning about AI")
            print(f"Response 1: {response1[:60]}...")
            
            response2 = chain.run(human_input="What did I just tell you about?")
            print(f"Response 2: {response2[:60]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Memory demonstration failed: {e}")
            return False
    
    def demonstrate_documents(self):
        """5. Document Processing"""
        print("\n5️⃣ Document Processing")
        print("-" * 40)
        
        try:
            # Create sample document
            sample_text = """
            LangChain is a framework for developing applications powered by language models.
            It provides tools for prompt management, chains, memory, and agents.
            LangChain makes it easy to build complex AI applications with minimal code.
            The framework supports various language models and provides abstractions
            for common patterns in LLM application development.
            """
            
            # Save sample document
            with open("temp_doc.txt", "w") as f:
                f.write(sample_text)
            
            # Load document
            loader = TextLoader("temp_doc.txt")
            documents = loader.load()
            
            print(f"Loaded document: {len(documents)} document(s)")
            print(f"Content preview: {documents[0].page_content[:60]}...")
            
            # Split document
            text_splitter = CharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=20
            )
            
            splits = text_splitter.split_documents(documents)
            print(f"Document splits: {len(splits)} chunks created")
            
            # Clean up
            os.remove("temp_doc.txt")
            
            return True
            
        except Exception as e:
            print(f"❌ Document demonstration failed: {e}")
            return False
    
    def demonstrate_embeddings(self):
        """6. Embeddings and Vector Stores"""
        print("\n6️⃣ Embeddings and Vector Stores")
        print("-" * 40)
        
        try:
            # Initialize embeddings
            embeddings = OpenAIEmbeddings()
            
            # Sample texts
            texts = [
                "LangChain is a framework for LLM applications",
                "Python is a programming language",
                "Machine learning is a subset of AI",
                "Vector databases store high-dimensional data"
            ]
            
            # Create vector store
            vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=embeddings,
                persist_directory="./temp_chroma"
            )
            
            print(f"Vector store created with {len(texts)} documents")
            
            # Search similar documents
            query = "What is LangChain?"
            results = vectorstore.similarity_search(query, k=2)
            
            print(f"Search results for '{query}':")
            for i, doc in enumerate(results):
                print(f"  {i+1}. {doc.page_content}")
            
            # Clean up
            import shutil
            shutil.rmtree("./temp_chroma", ignore_errors=True)
            
            return True
            
        except Exception as e:
            print(f"❌ Embeddings demonstration failed: {e}")
            return False
    
    def demonstrate_tools(self):
        """7. Tools for External Integration"""
        print("\n7️⃣ Tools")
        print("-" * 40)
        
        try:
            # Define custom tools
            def calculator(expression: str) -> str:
                """Calculate mathematical expressions"""
                try:
                    result = eval(expression)  # Note: Use safely in production
                    return f"Result: {result}"
                except:
                    return "Invalid expression"
            
            def word_counter(text: str) -> str:
                """Count words in text"""
                word_count = len(text.split())
                return f"Word count: {word_count}"
            
            # Create tool objects
            calc_tool = Tool(
                name="Calculator",
                description="Calculate mathematical expressions",
                func=calculator
            )
            
            counter_tool = Tool(
                name="WordCounter",
                description="Count words in text",
                func=word_counter
            )
            
            tools = [calc_tool, counter_tool]
            
            print(f"Created {len(tools)} custom tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Test tools
            calc_result = calc_tool.run("2 + 2 * 3")
            count_result = counter_tool.run("This is a sample text")
            
            print(f"Calculator: {calc_result}")
            print(f"Word Counter: {count_result}")
            
            return True
            
        except Exception as e:
            print(f"❌ Tools demonstration failed: {e}")
            return False
    
    def run_all_demonstrations(self):
        """Run all component demonstrations"""
        print("🚀 LangChain Core Components Demonstration")
        print("=" * 50)
        
        demonstrations = [
            self.demonstrate_llms,
            self.demonstrate_prompts,
            self.demonstrate_chains,
            self.demonstrate_memory,
            self.demonstrate_documents,
            self.demonstrate_embeddings,
            self.demonstrate_tools
        ]
        
        results = []
        for demo in demonstrations:
            try:
                success = demo()
                results.append(success)
            except Exception as e:
                print(f"❌ Demonstration failed: {e}")
                results.append(False)
        
        # Summary
        successful = sum(results)
        total = len(results)
        
        print(f"\n📊 Summary: {successful}/{total} demonstrations successful")
        
        if successful == total:
            print("🎉 All core components working correctly!")
        else:
            print("⚠️  Some components may need attention")

def main():
    """Main function"""
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found")
        print("Please set your OpenAI API key in the .env file")
        return
    
    demo = LangChainComponentsDemo()
    demo.run_all_demonstrations()

if __name__ == "__main__":
    main()
```

### 2.3 LangChain Architecture Patterns

Create `examples/basic/architecture_patterns.py`:

```python
# examples/basic/architecture_patterns.py
"""
LangChain Architecture Patterns
Common patterns and best practices for structuring LangChain applications
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

class ComponentType(Enum):
    """Types of LangChain components"""
    LLM = "llm"
    PROMPT = "prompt"
    CHAIN = "chain"
    MEMORY = "memory"
    TOOL = "tool"

@dataclass
class ComponentConfig:
    """Configuration for LangChain components"""
    name: str
    type: ComponentType
    config: Dict[str, Any]
    dependencies: List[str] = None

class BaseLangChainComponent(ABC):
    """Base class for all LangChain components"""
    
    def __init__(self, config: ComponentConfig):
        self.config = config
        self.name = config.name
        self.type = config.type
        self.dependencies = config.dependencies or []
        self._component = None
    
    @abstractmethod
    def initialize(self, **kwargs) -> Any:
        """Initialize the component"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate component configuration"""
        pass
    
    def get_component(self):
        """Get the initialized component"""
        if self._component is None:
            self._component = self.initialize()
        return self._component

class LLMComponent(BaseLangChainComponent):
    """LLM component wrapper"""
    
    def initialize(self, **kwargs):
        """Initialize LLM"""
        config = self.config.config
        
        provider = config.get('provider', 'openai')
        
        if provider == 'openai':
            return ChatOpenAI(
                model_name=config.get('model_name', 'gpt-3.5-turbo'),
                temperature=config.get('temperature', 0.7),
                max_tokens=config.get('max_tokens', 1000),
                **config.get('extra_params', {})
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def validate(self) -> bool:
        """Validate LLM configuration"""
        required_keys = ['provider']
        config = self.config.config
        
        for key in required_keys:
            if key not in config:
                return False
        
        return True

class PromptComponent(BaseLangChainComponent):
    """Prompt template component wrapper"""
    
    def initialize(self, **kwargs):
        """Initialize prompt template"""
        config = self.config.config
        
        return PromptTemplate(
            input_variables=config.get('input_variables', []),
            template=config.get('template', ''),
            **config.get('extra_params', {})
        )
    
    def validate(self) -> bool:
        """Validate prompt configuration"""
        required_keys = ['input_variables', 'template']
        config = self.config.config
        
        for key in required_keys:
            if key not in config:
                return False
        
        return True

class ChainComponent(BaseLangChainComponent):
    """Chain component wrapper"""
    
    def initialize(self, llm=None, prompt=None, memory=None, **kwargs):
        """Initialize chain"""
        config = self.config.config
        chain_type = config.get('type', 'llm')
        
        if chain_type == 'llm':
            return LLMChain(
                llm=llm,
                prompt=prompt,
                memory=memory,
                **config.get('extra_params', {})
            )
        else:
            raise ValueError(f"Unsupported chain type: {chain_type}")
    
    def validate(self) -> bool:
        """Validate chain configuration"""
        required_keys = ['type']
        config = self.config.config
        
        for key in required_keys:
            if key not in config:
                return False
        
        return True

class MemoryComponent(BaseLangChainComponent):
    """Memory component wrapper"""
    
    def initialize(self, **kwargs):
        """Initialize memory"""
        config = self.config.config
        memory_type = config.get('type', 'buffer')
        
        if memory_type == 'buffer':
            return ConversationBufferMemory(
                memory_key=config.get('memory_key', 'history'),
                return_messages=config.get('return_messages', False),
                **config.get('extra_params', {})
            )
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")
    
    def validate(self) -> bool:
        """Validate memory configuration"""
        required_keys = ['type']
        config = self.config.config
        
        for key in required_keys:
            if key not in config:
                return False
        
        return True

class ComponentFactory:
    """Factory for creating LangChain components"""
    
    @staticmethod
    def create_component(config: ComponentConfig) -> BaseLangChainComponent:
        """Create component based on type"""
        component_classes = {
            ComponentType.LLM: LLMComponent,
            ComponentType.PROMPT: PromptComponent,
            ComponentType.CHAIN: ChainComponent,
            ComponentType.MEMORY: MemoryComponent,
        }
        
        component_class = component_classes.get(config.type)
        if not component_class:
            raise ValueError(f"Unsupported component type: {config.type}")
        
        component = component_class(config)
        
        if not component.validate():
            raise ValueError(f"Invalid configuration for {config.name}")
        
        return component

class LangChainApplication:
    """Main application class following dependency injection pattern"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.components: Dict[str, BaseLangChainComponent] = {}
        self.initialized_components: Dict[str, Any] = {}
    
    def register_component(self, component_config: ComponentConfig):
        """Register a component"""
        component = ComponentFactory.create_component(component_config)
        self.components[component_config.name] = component
    
    def initialize_component(self, name: str) -> Any:
        """Initialize a component and its dependencies"""
        if name in self.initialized_components:
            return self.initialized_components[name]
        
        if name not in self.components:
            raise ValueError(f"Component {name} not found")
        
        component = self.components[name]
        
        # Initialize dependencies first
        kwargs = {}
        for dep_name in component.dependencies:
            kwargs[dep_name] = self.initialize_component(dep_name)
        
        # Initialize component
        initialized = component.initialize(**kwargs)
        self.initialized_components[name] = initialized
        
        return initialized
    
    def get_component(self, name: str) -> Any:
        """Get an initialized component"""
        return self.initialize_component(name)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the application"""
        # This should be implemented by specific applications
        raise NotImplementedError("Subclasses must implement run method")

class ChatbotApplication(LangChainApplication):
    """Example chatbot application"""
    
    def __init__(self):
        super().__init__({})
        self._setup_components()
    
    def _setup_components(self):
        """Setup chatbot components"""
        # LLM component
        llm_config = ComponentConfig(
            name="llm",
            type=ComponentType.LLM,
            config={
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 200
            }
        )
        self.register_component(llm_config)
        
        # Prompt component
        prompt_config = ComponentConfig(
            name="prompt",
            type=ComponentType.PROMPT,
            config={
                "input_variables": ["history", "human_input"],
                "template": """
                Previous conversation:
                {history}
                
                Human: {human_input}
                Assistant: I'll help you with that.
                """
            }
        )
        self.register_component(prompt_config)
        
        # Memory component
        memory_config = ComponentConfig(
            name="memory",
            type=ComponentType.MEMORY,
            config={
                "type": "buffer",
                "memory_key": "history",
                "return_messages": False
            }
        )
        self.register_component(memory_config)
        
        # Chain component
        chain_config = ComponentConfig(
            name="chain",
            type=ComponentType.CHAIN,
            config={
                "type": "llm"
            },
            dependencies=["llm", "prompt", "memory"]
        )
        self.register_component(chain_config)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run chatbot"""
        chain = self.get_component("chain")
        user_input = input_data.get("message", "")
        
        try:
            response = chain.run(human_input=user_input)
            return {
                "success": True,
                "response": response,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": str(e)
            }

def demonstrate_architecture_patterns():
    """Demonstrate LangChain architecture patterns"""
    print("🏗️ LangChain Architecture Patterns")
    print("=" * 40)
    
    try:
        # Create chatbot application
        chatbot = ChatbotApplication()
        
        print("✅ Chatbot application initialized")
        
        # Test conversation
        test_inputs = [
            "Hello, how are you?",
            "What can you help me with?",
            "Tell me a joke"
        ]
        
        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n💬 Turn {i}")
            print(f"User: {user_input}")
            
            result = chatbot.run({"message": user_input})
            
            if result["success"]:
                print(f"Bot: {result['response'][:100]}...")
            else:
                print(f"Error: {result['error']}")
        
        print("\n✅ Architecture pattern demonstration complete")
        
    except Exception as e:
        print(f"❌ Architecture demonstration failed: {e}")

if __name__ == "__main__":
    demonstrate_architecture_patterns()
```

---

## Chapter 3: Language Model Integration

### 3.1 Understanding Language Model Providers

LangChain supports multiple language model providers, each with their own strengths and use cases. This chapter covers how to integrate and work with different LLM providers effectively.

#### Supported LLM Providers

Create `examples/intermediate/llm_providers.py`:

```python
# examples/intermediate/llm_providers.py
"""
Language Model Provider Integration Examples
Demonstrates how to work with different LLM providers in LangChain
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# LangChain LLM imports
from langchain_openai import ChatOpenAI, OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import HuggingFacePipeline
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

load_dotenv()

class LLMProviderManager:
    """Manager for different LLM providers"""
    
    def __init__(self):
        self.providers: Dict[str, Any] = {}
        self.initialize_providers()
    
    def initialize_providers(self):
        """Initialize all available LLM providers"""
        
        # OpenAI Provider
        if os.getenv('OPENAI_API_KEY'):
            self.providers['openai_chat'] = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000,
                streaming=False
            )
            
            self.providers['openai_completion'] = OpenAI(
                model_name="gpt-3.5-turbo-instruct",
                temperature=0.7,
                max_tokens=1000
            )
        
        # Anthropic Provider
        if os.getenv('ANTHROPIC_API_KEY'):
            self.providers['anthropic'] = ChatAnthropic(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.7
            )
        
        # HuggingFace Provider (local)
        try:
            self.providers['huggingface_local'] = HuggingFacePipeline.from_model_id(
                model_id="microsoft/DialoGPT-medium",
                task="text-generation",
                model_kwargs={"temperature": 0.7, "max_length": 1000}
            )
        except Exception as e:
            print(f"⚠️  HuggingFace local model not available: {e}")
    
    def get_provider(self, name: str) -> Optional[Any]:
        """Get a specific provider"""
        return self.providers.get(name)
    
    def list_providers(self) -> List[str]:
        """List available providers"""
        return list(self.providers.keys())
    
    def test_provider(self, provider_name: str, prompt: str = "Hello, how are you?") -> Dict[str, Any]:
        """Test a specific provider"""
        provider = self.get_provider(provider_name)
        
        if not provider:
            return {
                "success": False,
                "error": f"Provider {provider_name} not available",
                "response": None,
                "latency": None
            }
        
        try:
            import time
            start_time = time.time()
            
            # Handle different provider types
            if hasattr(provider, 'invoke'):
                if 'chat' in provider_name.lower():
                    response = provider.invoke([HumanMessage(content=prompt)])
                    result = response.content
                else:
                    response = provider.invoke(prompt)
                    result = response
            else:
                result = provider(prompt)
            
            end_time = time.time()
            latency = end_time - start_time
            
            return {
                "success": True,
                "error": None,
                "response": result,
                "latency": latency
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": None,
                "latency": None
            }

class LLMComparisonTool:
    """Tool for comparing different LLM providers"""
    
    def __init__(self, manager: LLMProviderManager):
        self.manager = manager
    
    def compare_providers(self, prompt: str, providers: List[str] = None) -> Dict[str, Any]:
        """Compare responses from different providers"""
        if providers is None:
            providers = self.manager.list_providers()
        
        results = {}
        
        print(f"🔍 Comparing LLM Providers")
        print(f"Prompt: {prompt}")
        print("-" * 50)
        
        for provider_name in providers:
            print(f"\n🤖 Testing {provider_name}...")
            result = self.manager.test_provider(provider_name, prompt)
            results[provider_name] = result
            
            if result["success"]:
                print(f"✅ Success (Latency: {result['latency']:.2f}s)")
                print(f"Response: {result['response'][:100]}...")
            else:
                print(f"❌ Failed: {result['error']}")
        
        return results
    
    def benchmark_performance(self, prompts: List[str], providers: List[str] = None) -> Dict[str, Any]:
        """Benchmark performance across multiple prompts"""
        if providers is None:
            providers = self.manager.list_providers()
        
        benchmark_results = {provider: [] for provider in providers}
        
        print(f"\n📊 Performance Benchmark")
        print(f"Testing {len(prompts)} prompts across {len(providers)} providers")
        print("-" * 50)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\nPrompt {i}/{len(prompts)}: {prompt[:30]}...")
            
            for provider_name in providers:
                result = self.manager.test_provider(provider_name, prompt)
                benchmark_results[provider_name].append(result)
        
        # Calculate statistics
        stats = {}
        for provider_name, results in benchmark_results.items():
            successful_results = [r for r in results if r["success"]]
            
            if successful_results:
                latencies = [r["latency"] for r in successful_results]
                stats[provider_name] = {
                    "success_rate": len(successful_results) / len(results),
                    "avg_latency": sum(latencies) / len(latencies),
                    "min_latency": min(latencies),
                    "max_latency": max(latencies)
                }
            else:
                stats[provider_name] = {
                    "success_rate": 0,
                    "avg_latency": None,
                    "min_latency": None,
                    "max_latency": None
                }
        
        return {
            "raw_results": benchmark_results,
            "statistics": stats
        }

def demonstrate_streaming():
    """Demonstrate streaming responses"""
    print("\n🌊 Streaming Response Demo")
    print("-" * 30)
    
    try:
        # Initialize streaming LLM
        streaming_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        print("Streaming response:")
        response = streaming_llm.invoke([
            HumanMessage(content="Write a short poem about artificial intelligence")
        ])
        
        print(f"\nComplete response received: {len(response.content)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming demo failed: {e}")
        return False

def demonstrate_batch_processing():
    """Demonstrate batch processing with LLMs"""
    print("\n📦 Batch Processing Demo")
    print("-" * 30)
    
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=100
        )
        
        # Batch of prompts
        prompts = [
            [HumanMessage(content="What is Python?")],
            [HumanMessage(content="What is JavaScript?")],
            [HumanMessage(content="What is machine learning?")],
            [HumanMessage(content="What is blockchain?")]
        ]
        
        print(f"Processing batch of {len(prompts)} prompts...")
        
        import time
        start_time = time.time()
        
        responses = llm.batch(prompts)
        
        end_time = time.time()
        
        print(f"Batch completed in {end_time - start_time:.2f} seconds")
        print(f"Responses generated: {len(responses)}")
        
        for i, response in enumerate(responses, 1):
            print(f"{i}. {response.content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processing demo failed: {e}")
        return False

def demonstrate_custom_llm():
    """Demonstrate creating a custom LLM wrapper"""
    print("\n🔧 Custom LLM Wrapper Demo")
    print("-" * 30)
    
    from langchain.llms.base import LLM
    from typing import Any, List, Optional
    import requests
    
    class CustomAPILLM(LLM):
        """Custom LLM that calls an external API"""
        
        api_url: str = "https://api.example.com/generate"  # Example URL
        api_key: str = ""
        model_name: str = "custom-model"
        
        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            """Call the custom API"""
            # This is a mock implementation
            # In practice, you would make an actual API call
            return f"Custom API Response to: {prompt[:30]}... [This is a mock response]"
        
        @property
        def _llm_type(self) -> str:
            return "custom-api"
        
        def _identifying_params(self) -> Dict[str, Any]:
            return {
                "api_url": self.api_url,
                "model_name": self.model_name
            }
    
    try:
        # Initialize custom LLM
        custom_llm = CustomAPILLM(
            api_key="your-api-key",
            model_name="my-custom-model"
        )
        
        # Test custom LLM
        response = custom_llm.invoke("Tell me about artificial intelligence")
        print(f"Custom LLM Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom LLM demo failed: {e}")
        return False

def main():
    """Main demonstration function"""
    print("🤖 LangChain LLM Integration Guide")
    print("=" * 40)
    
    # Initialize LLM manager
    manager = LLMProviderManager()
    
    print(f"Available providers: {manager.list_providers()}")
    
    if not manager.list_providers():
        print("❌ No LLM providers available. Please check your API keys.")
        return
    
    # Initialize comparison tool
    comparison_tool = LLMComparisonTool(manager)
    
    # Compare providers
    test_prompt = "Explain quantum computing in simple terms"
    comparison_results = comparison_tool.compare_providers(test_prompt)
    
    # Benchmark performance
    benchmark_prompts = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain neural networks briefly"
    ]
    
    benchmark_results = comparison_tool.benchmark_performance(benchmark_prompts)
    
    print("\n📊 Benchmark Statistics:")
    for provider, stats in benchmark_results["statistics"].items():
        if stats["avg_latency"] is not None:
            print(f"{provider}: {stats['success_rate']:.1%} success, {stats['avg_latency']:.2f}s avg latency")
        else:
            print(f"{provider}: {stats['success_rate']:.1%} success")
    
    # Additional demonstrations
    demonstrate_streaming()
    demonstrate_batch_processing()
    demonstrate_custom_llm()
    
    print("\n✅ LLM Integration demonstration complete!")

if __name__ == "__main__":
    main()
```

### 3.2 Advanced LLM Configuration

Create `src/llm/advanced_config.py`:

```python
# src/llm/advanced_config.py
"""
Advanced LLM Configuration and Management
Provides sophisticated LLM configuration and management capabilities
"""

import os
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI, OpenAI
from langchain_anthropic import ChatAnthropic
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage, LLMResult
from langchain.cache import InMemoryCache, SQLiteCache
from langchain import globals

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

class LLMType(Enum):
    """Types of language models"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"

@dataclass
class LLMConfig:
    """Configuration for LLM instances"""
    provider: LLMProvider
    model_name: str
    llm_type: LLMType
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    streaming: bool = False
    callbacks: List[BaseCallbackHandler] = field(default_factory=list)
    cache: bool = False
    retry_attempts: int = 3
    timeout: int = 30
    custom_params: Dict[str, Any] = field(default_factory=dict)

class CustomCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for monitoring LLM calls"""
    
    def __init__(self, name: str = "CustomCallback"):
        self.name = name
        self.call_count = 0
        self.total_tokens = 0
        self.errors = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running"""
        self.call_count += 1
        print(f"🚀 {self.name}: Starting LLM call #{self.call_count}")
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM ends running"""
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            tokens = token_usage.get('total_tokens', 0)
            self.total_tokens += tokens
            print(f"✅ {self.name}: Call completed. Tokens used: {tokens}")
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Called when LLM errors"""
        self.errors.append(str(error))
        print(f"❌ {self.name}: LLM error: {error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get callback statistics"""
        return {
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "error_count": len(self.errors),
            "errors": self.errors
        }

class LLMFactory:
    """Factory for creating configured LLM instances"""
    
    @staticmethod
    def create_llm(config: LLMConfig) -> Any:
        """Create LLM instance from configuration"""
        
        # Setup caching if enabled
        if config.cache:
            globals.set_llm_cache(InMemoryCache())
        
        # Common parameters
        common_params = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "callbacks": config.callbacks,
            **config.custom_params
        }
        
        # Provider-specific creation
        if config.provider == LLMProvider.OPENAI:
            if config.llm_type == LLMType.CHAT:
                return ChatOpenAI(
                    model_name=config.model_name,
                    streaming=config.streaming,
                    **common_params
                )
            elif config.llm_type == LLMType.COMPLETION:
                return OpenAI(
                    model_name=config.model_name,
                    **common_params
                )
        
        elif config.provider == LLMProvider.ANTHROPIC:
            if config.llm_type == LLMType.CHAT:
                return ChatAnthropic(
                    model=config.model_name,
                    **common_params
                )
        
        else:
            raise ValueError(f"Unsupported provider/type combination: {config.provider}/{config.llm_type}")

class LLMManager:
    """Advanced LLM management with features like pooling, fallbacks, and monitoring"""
    
    def __init__(self):
        self.llms: Dict[str, Any] = {}
        self.configs: Dict[str, LLMConfig] = {}
        self.callbacks: Dict[str, CustomCallbackHandler] = {}
        self.fallback_chains: Dict[str, List[str]] = {}
    
    def register_llm(self, name: str, config: LLMConfig) -> None:
        """Register an LLM with the manager"""
        # Add monitoring callback
        callback = CustomCallbackHandler(f"{name}_monitor")
        config.callbacks.append(callback)
        
        # Create and register LLM
        llm = LLMFactory.create_llm(config)
        
        self.llms[name] = llm
        self.configs[name] = config
        self.callbacks[name] = callback
        
        print(f"✅ Registered LLM: {name} ({config.provider.value}/{config.model_name})")
    
    def set_fallback_chain(self, primary: str, fallbacks: List[str]) -> None:
        """Set fallback chain for an LLM"""
        self.fallback_chains[primary] = fallbacks
        print(f"🔄 Set fallback chain for {primary}: {' -> '.join(fallbacks)}")
    
    def invoke_with_fallback(self, name: str, prompt: Union[str, List[BaseMessage]]) -> str:
        """Invoke LLM with fallback support"""
        llm_chain = [name] + self.fallback_chains.get(name, [])
        
        for llm_name in llm_chain:
            if llm_name not in self.llms:
                print(f"⚠️  LLM {llm_name} not found, skipping")
                continue
            
            try:
                llm = self.llms[llm_name]
                print(f"🔄 Trying LLM: {llm_name}")
                
                response = llm.invoke(prompt)
                
                if hasattr(response, 'content'):
                    return response.content
                else:
                    return str(response)
                
            except Exception as e:
                print(f"❌ LLM {llm_name} failed: {e}")
                continue
        
        raise Exception("All LLMs in fallback chain failed")
    
    def get_llm_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for an LLM"""
        if name in self.callbacks:
            return self.callbacks[name].get_stats()
        return {}
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all registered LLMs"""
        return {name: self.get_llm_stats(name) for name in self.llms.keys()}

class LLMLoadBalancer:
    """Load balancer for distributing requests across multiple LLMs"""
    
    def __init__(self, llms: Dict[str, Any], strategy: str = "round_robin"):
        self.llms = llms
        self.strategy = strategy
        self.current_index = 0
        self.request_counts = {name: 0 for name in llms.keys()}
    
    def select_llm(self) -> tuple[str, Any]:
        """Select LLM based on load balancing strategy"""
        llm_names = list(self.llms.keys())
        
        if self.strategy == "round_robin":
            name = llm_names[self.current_index % len(llm_names)]
            self.current_index += 1
        
        elif self.strategy == "least_requests":
            name = min(self.request_counts.keys(), 
                      key=lambda x: self.request_counts[x])
        
        else:
            name = llm_names[0]  # Default to first
        
        self.request_counts[name] += 1
        return name, self.llms[name]
    
    def invoke(self, prompt: Union[str, List[BaseMessage]]) -> str:
        """Invoke using load balancing"""
        name, llm = self.select_llm()
        
        print(f"🎯 Selected LLM: {name} (requests: {self.request_counts[name]})")
        
        response = llm.invoke(prompt)
        
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)

def demonstrate_advanced_llm_config():
    """Demonstrate advanced LLM configuration features"""
    print("⚙️ Advanced LLM Configuration Demo")
    print("=" * 40)
    
    try:
        # Initialize LLM manager
        manager = LLMManager()
        
        # Register multiple LLMs with different configurations
        configs = [
            LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                llm_type=LLMType.CHAT,
                temperature=0.3,
                max_tokens=200,
                cache=True
            ),
            LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4",
                llm_type=LLMType.CHAT,
                temperature=0.7,
                max_tokens=300,
                cache=True
            )
        ]
        
        # Register LLMs
        for i, config in enumerate(configs):
            manager.register_llm(f"llm_{i+1}", config)
        
        # Set fallback chain
        manager.set_fallback_chain("llm_2", ["llm_1"])  # GPT-4 falls back to GPT-3.5
        
        # Test with fallback
        test_prompt = "Explain the concept of machine learning in 2 sentences."
        
        response = manager.invoke_with_fallback("llm_1", test_prompt)
        print(f"\nResponse: {response}")
        
        # Show statistics
        print("\n📊 LLM Statistics:")
        stats = manager.get_all_stats()
        for llm_name, llm_stats in stats.items():
            print(f"{llm_name}: {llm_stats['call_count']} calls, {llm_stats['total_tokens']} tokens")
        
        # Demonstrate load balancing
        print("\n⚖️ Load Balancing Demo:")
        
        if len(manager.llms) > 1:
            load_balancer = LLMLoadBalancer(manager.llms, strategy="round_robin")
            
            for i in range(3):
                response = load_balancer.invoke(f"Test message {i+1}")
                print(f"Response {i+1}: {response[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced configuration demo failed: {e}")
        return False

if __name__ == "__main__":
    demonstrate_advanced_llm_config()
```

### 3.3 LLM Performance Optimization

Create `src/llm/performance_optimization.py`:

```python
# src/llm/performance_optimization.py
"""
LLM Performance Optimization Techniques
Techniques for optimizing LLM performance, caching, and efficiency
"""

import time
import asyncio
import hashlib
from typing import Dict, Any, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import pickle
import os

from langchain_openai import ChatOpenAI
from langchain.cache import InMemoryCache, SQLiteCache
from langchain import globals
from langchain.schema import BaseMessage, HumanMessage

@dataclass
class PerformanceMetrics:
    """Performance metrics for LLM operations"""
    total_calls: int = 0
    total_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    tokens_used: int = 0

class AdvancedCache:
    """Advanced caching system for LLM responses"""
    
    def __init__(self, cache_dir: str = "./cache", max_size: int = 1000):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_counts: Dict[str, int] = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        self.load_cache()
    
    def _hash_key(self, prompt: str, model: str, **kwargs) -> str:
        """Create hash key for caching"""
        cache_input = f"{prompt}:{model}:{sorted(kwargs.items())}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, **kwargs) -> Optional[str]:
        """Get cached response"""
        key = self._hash_key(prompt, model, **kwargs)
        
        if key in self.cache:
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            return self.cache[key]
        
        return None
    
    def set(self, prompt: str, model: str, response: str, **kwargs) -> None:
        """Set cached response"""
        key = self._hash_key(prompt, model, **kwargs)
        
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = response
        self.access_counts[key] = 1
        self.save_cache()
    
    def _evict_lru(self):
        """Evict least recently used items"""
        if not self.access_counts:
            return
        
        lru_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        del self.cache[lru_key]
        del self.access_counts[lru_key]
    
    def save_cache(self):
        """Save cache to disk"""
        cache_file = os.path.join(self.cache_dir, "llm_cache.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'cache': self.cache,
                'access_counts': self.access_counts
            }, f)
    
    def load_cache(self):
        """Load cache from disk"""
        cache_file = os.path.join(self.cache_dir, "llm_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    self.access_counts = data.get('access_counts', {})
            except:
                # If loading fails, start with empty cache
                pass
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_counts.clear()
        self.save_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_accesses = sum(self.access_counts.values())
        return {
            "cache_size": len(self.cache),
            "total_accesses": total_accesses,
            "hit_rate": len(self.cache) / max(total_accesses, 1),
            "max_size": self.max_size
        }

class OptimizedLLMWrapper:
    """Optimized wrapper for LLM with caching and batching"""
    
    def __init__(self, llm, cache: AdvancedCache = None):
        self.llm = llm
        self.cache = cache or AdvancedCache()
        self.metrics = PerformanceMetrics()
    
    def invoke(self, prompt: Union[str, List[BaseMessage]], **kwargs) -> str:
        """Optimized invoke with caching"""
        self.metrics.total_calls += 1
        
        # Convert prompt to string for caching
        prompt_str = prompt if isinstance(prompt, str) else str(prompt)
        model_name = getattr(self.llm, 'model_name', 'unknown')
        
        # Check cache first
        cached_response = self.cache.get(prompt_str, model_name, **kwargs)
        if cached_response:
            self.metrics.cache_hits += 1
            return cached_response
        
        # Cache miss - call LLM
        self.metrics.cache_misses += 1
        
        try:
            start_time = time.time()
            response = self.llm.invoke(prompt)
            end_time = time.time()
            
            self.metrics.total_latency += (end_time - start_time)
            
            # Extract response content
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Cache the response
            self.cache.set(prompt_str, model_name, response_content, **kwargs)
            
            return response_content
            
        except Exception as e:
            self.metrics.errors += 1
            raise e
    
    async def ainvoke(self, prompt: Union[str, List[BaseMessage]], **kwargs) -> str:
        """Async optimized invoke"""
        # For now, wrap sync invoke - could be optimized further
        return self.invoke(prompt, **kwargs)
    
    def batch_invoke(self, prompts: List[Union[str, List[BaseMessage]]], 
                    max_workers: int = 5) -> List[str]:
        """Batch processing with thread pool"""
        
        def process_prompt(prompt):
            return self.invoke(prompt)
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {
                executor.submit(process_prompt, prompt): prompt 
                for prompt in prompts
            }
            
            for future in as_completed(future_to_prompt):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Batch processing error: {e}")
                    results.append(f"Error: {e}")
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_latency = (self.metrics.total_latency / max(self.metrics.total_calls, 1))
        cache_hit_rate = (self.metrics.cache_hits / max(self.metrics.total_calls, 1))
        
        return {
            "total_calls": self.metrics.total_calls,
            "average_latency": avg_latency,
            "cache_hit_rate": cache_hit_rate,
            "cache_stats": self.cache.get_stats(),
            "errors": self.metrics.errors
        }

class LLMOptimizer:
    """Optimizer for LLM configurations and parameters"""
    
    def __init__(self, llm_factory_func):
        self.llm_factory_func = llm_factory_func
        self.test_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "How do neural networks work?",
            "What is the difference between AI and ML?"
        ]
    
    def optimize_temperature(self, temperature_range: tuple = (0.0, 1.0), 
                           steps: int = 5) -> float:
        """Find optimal temperature for consistent responses"""
        
        temperatures = [temperature_range[0] + i * (temperature_range[1] - temperature_range[0]) / (steps - 1) 
                       for i in range(steps)]
        
        results = {}
        
        print("🌡️ Optimizing temperature parameter...")
        
        for temp in temperatures:
            llm = self.llm_factory_func(temperature=temp, max_tokens=100)
            wrapper = OptimizedLLMWrapper(llm)
            
            # Test consistency - run same prompt multiple times
            test_prompt = self.test_prompts[0]
            responses = []
            
            for _ in range(3):
                response = wrapper.invoke(test_prompt)
                responses.append(response)
            
            # Calculate response diversity (simple metric)
            diversity = len(set(responses)) / len(responses)
            
            results[temp] = {
                'diversity': diversity,
                'responses': responses
            }
            
            print(f"Temperature {temp:.2f}: Diversity = {diversity:.2f}")
        
        # Find optimal temperature (balance between consistency and creativity)
        optimal_temp = min(results.keys(), key=lambda t: abs(results[t]['diversity'] - 0.7))
        
        print(f"✅ Optimal temperature: {optimal_temp:.2f}")
        return optimal_temp
    
    def benchmark_token_limits(self, token_limits: List[int]) -> Dict[int, Dict[str, Any]]:
        """Benchmark different token limits"""
        
        results = {}
        
        print("📊 Benchmarking token limits...")
        
        for max_tokens in token_limits:
            llm = self.llm_factory_func(max_tokens=max_tokens, temperature=0.7)
            wrapper = OptimizedLLMWrapper(llm)
            
            total_time = 0
            responses = []
            
            for prompt in self.test_prompts:
                start_time = time.time()
                response = wrapper.invoke(prompt)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                responses.append(len(response))
            
            avg_latency = total_time / len(self.test_prompts)
            avg_response_length = sum(responses) / len(responses)
            
            results[max_tokens] = {
                'avg_latency': avg_latency,
                'avg_response_length': avg_response_length,
                'efficiency': avg_response_length / avg_latency  # chars per second
            }
            
            print(f"Max tokens {max_tokens}: {avg_latency:.2f}s avg, {avg_response_length:.0f} chars avg")
        
        return results

async def demonstrate_async_optimization():
    """Demonstrate async LLM optimization"""
    print("\n⚡ Async LLM Optimization Demo")
    print("-" * 30)
    
    try:
        # Create async-capable LLMs
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=150
        )
        
        wrapper = OptimizedLLMWrapper(llm)
        
        # Test prompts
        prompts = [
            "What is Python programming?",
            "Explain data structures briefly.",
            "What are algorithms?",
            "Define software engineering."
        ]
        
        # Async concurrent processing
        print("Processing prompts concurrently...")
        start_time = time.time()
        
        tasks = [wrapper.ainvoke(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        print(f"Concurrent processing completed in {end_time - start_time:.2f} seconds")
        print(f"Processed {len(responses)} prompts")
        
        # Compare with sequential processing
        print("\nComparing with sequential processing...")
        start_time = time.time()
        
        sequential_responses = []
        for prompt in prompts:
            response = wrapper.invoke(prompt)
            sequential_responses.append(response)
        
        end_time = time.time()
        
        print(f"Sequential processing completed in {end_time - start_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Async optimization demo failed: {e}")
        return False

def demonstrate_performance_optimization():
    """Main demonstration of performance optimization techniques"""
    print("🚀 LLM Performance Optimization Demo")
    print("=" * 40)
    
    try:
        # Initialize optimized LLM
        base_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=200
        )
        
        cache = AdvancedCache(max_size=100)
        optimized_llm = OptimizedLLMWrapper(base_llm, cache)
        
        # Test caching performance
        print("🗄️ Testing cache performance...")
        
        test_prompts = [
            "What is machine learning?",
            "Explain artificial intelligence.",
            "What is machine learning?",  # Repeat for cache hit
            "Define neural networks.",
            "What is machine learning?"   # Another repeat
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            start_time = time.time()
            response = optimized_llm.invoke(prompt)
            end_time = time.time()
            
            print(f"Prompt {i}: {end_time - start_time:.3f}s - {response[:30]}...")
        
        # Show performance metrics
        metrics = optimized_llm.get_performance_metrics()
        print(f"\n📊 Performance Metrics:")
        print(f"Total calls: {metrics['total_calls']}")
        print(f"Average latency: {metrics['average_latency']:.3f}s")
        print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
        print(f"Cache size: {metrics['cache_stats']['cache_size']}")
        
        # Test batch processing
        print(f"\n📦 Testing batch processing...")
        batch_prompts = [
            "What is Python?",
            "What is JavaScript?",
            "What is Go?",
            "What is Rust?"
        ]
        
        start_time = time.time()
        batch_responses = optimized_llm.batch_invoke(batch_prompts, max_workers=3)
        end_time = time.time()
        
        print(f"Batch processing completed in {end_time - start_time:.2f}s")
        print(f"Processed {len(batch_responses)} prompts")
        
        # Demonstrate parameter optimization
        def llm_factory(**kwargs):
            return ChatOpenAI(
                model_name="gpt-3.5-turbo",
                **kwargs
            )
        
        optimizer = LLMOptimizer(llm_factory)
        
        # Optimize temperature
        optimal_temp = optimizer.optimize_temperature(steps=3)
        
        # Benchmark token limits
        token_results = optimizer.benchmark_token_limits([100, 200, 500])
        
        print(f"\n📈 Token limit benchmarks:")
        for tokens, stats in token_results.items():
            print(f"{tokens} tokens: {stats['efficiency']:.1f} chars/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance optimization demo failed: {e}")
        return False

def main():
    """Main function"""
    print("⚡ LLM Performance Optimization Guide")
    print("=" * 45)
    
    # Run demonstrations
    demonstrate_performance_optimization()
    
    # Run async demonstration
    print("\n" + "="*45)
    asyncio.run(demonstrate_async_optimization())
    
    print("\n✅ Performance optimization demonstration complete!")
    print("\n💡 Key takeaways:")
    print("- Use caching for repeated queries")
    print("- Batch process multiple requests when possible")
    print("- Optimize temperature and token limits for your use case")
    print("- Use async processing for concurrent operations")
    print("- Monitor performance metrics to identify bottlenecks")

if __name__ == "__main__":
    main()
```

---

## Chapter 4: Prompts and Prompt Templates

### 4.1 Understanding Prompt Engineering

Effective prompting is crucial for getting the best results from language models. This chapter covers advanced prompt engineering techniques and template management.

Create `examples/intermediate/prompt_engineering.py`:

```python
# examples/intermediate/prompt_engineering.py
"""
Advanced Prompt Engineering Techniques
Comprehensive guide to creating effective prompts and templates
"""

import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate, 
    ChatPromptTemplate, 
    MessagesPlaceholder,
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate
)
from langchain.prompts.example_selector import (
    LengthBasedExampleSelector,
    SemanticSimilarityExampleSelector
)
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field

load_dotenv()

class PromptType(Enum):
    """Types of prompts"""
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    COMPLETION = "completion"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    GENERATION = "generation"

@dataclass
class PromptMetrics:
    """Metrics for evaluating prompt performance"""
    accuracy: float
    consistency: float
    relevance: float
    efficiency: float
    
class AdvancedPromptTemplate:
    """Advanced prompt template with validation and optimization"""
    
    def __init__(self, template: str, input_variables: List[str], 
                 prompt_type: PromptType, examples: List[Dict] = None):
        self.template = template
        self.input_variables = input_variables
        self.prompt_type = prompt_type
        self.examples = examples or []
        self.performance_history = []
    
    def validate(self) -> bool:
        """Validate prompt template"""
        # Check if all variables are present in template
        for var in self.input_variables:
            if f"{{{var}}}" not in self.template:
                return False
        return True
    
    def format(self, **kwargs) -> str:
        """Format prompt with given variables"""
        if not self.validate():
            raise ValueError("Invalid prompt template")
        
        return self.template.format(**kwargs)
    
    def add_example(self, example: Dict):
        """Add example to prompt"""
        self.examples.append(example)
    
    def get_few_shot_prompt(self, max_examples: int = 5) -> FewShotPromptTemplate:
        """Create few-shot prompt template"""
        if not self.examples:
            raise ValueError("No examples available for few-shot prompting")
        
        example_template = PromptTemplate(
            input_variables=list(self.examples[0].keys()),
            template="\n".join([f"{k}: {{{k}}}" for k in self.examples[0].keys()])
        )
        
        return FewShotPromptTemplate(
            examples=self.examples[:max_examples],
            example_prompt=example_template,
            prefix=self.template,
            suffix="Input: {input}\nOutput:",
            input_variables=["input"]
        )

class PromptOptimizer:
    """Optimizer for improving prompt performance"""
    
    def __init__(self, llm):
        self.llm = llm
        self.test_cases = []
    
    def add_test_case(self, input_data: Dict, expected_output: str, 
                     weight: float = 1.0):
        """Add test case for prompt evaluation"""
        self.test_cases.append({
            "input": input_data,
            "expected": expected_output,
            "weight": weight
        })
    
    def evaluate_prompt(self, prompt_template: PromptTemplate) -> PromptMetrics:
        """Evaluate prompt template performance"""
        if not self.test_cases:
            raise ValueError("No test cases available")
        
        results = []
        
        for test_case in self.test_cases:
            try:
                # Format and run prompt
                formatted_prompt = prompt_template.format(**test_case["input"])
                response = self.llm.invoke(formatted_prompt)
                
                # Extract response content
                response_content = response.content if hasattr(response, 'content') else str(response)
                
                # Simple similarity score (in practice, use more sophisticated metrics)
                similarity = self._calculate_similarity(
                    response_content, 
                    test_case["expected"]
                )
                
                results.append({
                    "similarity": similarity,
                    "weight": test_case["weight"]
                })
                
            except Exception as e:
                results.append({
                    "similarity": 0.0,
                    "weight": test_case["weight"]
                })
        
        # Calculate weighted metrics
        weighted_accuracy = sum(r["similarity"] * r["weight"] for r in results) / sum(r["weight"] for r in results)
        
        return PromptMetrics(
            accuracy=weighted_accuracy,
            consistency=self._calculate_consistency(results),
            relevance=weighted_accuracy,  # Simplified
            efficiency=1.0  # Would need timing data
        )
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simplified)"""
        # Simple word overlap - in practice, use embeddings or other metrics
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_consistency(self, results: List[Dict]) -> float:
        """Calculate consistency of results"""
        if len(results) < 2:
            return 1.0
        
        similarities = [r["similarity"] for r in results]
        mean_sim = sum(similarities) / len(similarities)
        variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
        
        return 1.0 / (1.0 + variance)  # Higher consistency = lower variance
    
    def optimize_prompt_variants(self, base_prompt: str, 
                                variations: List[str]) -> tuple[str, PromptMetrics]:
        """Compare prompt variations and return the best one"""
        best_prompt = base_prompt
        best_metrics = None
        
        all_prompts = [base_prompt] + variations
        
        print(f"🔍 Evaluating {len(all_prompts)} prompt variations...")
        
        for i, prompt in enumerate(all_prompts):
            template = PromptTemplate(
                input_variables=["input"],  # Simplified
                template=prompt
            )
            
            metrics = self.evaluate_prompt(template)
            
            print(f"Variant {i+1}: Accuracy={metrics.accuracy:.3f}, Consistency={metrics.consistency:.3f}")
            
            if best_metrics is None or metrics.accuracy > best_metrics.accuracy:
                best_prompt = prompt
                best_metrics = metrics
        
        return best_prompt, best_metrics

# Output Parsing Models
class PersonInfo(BaseModel):
    """Person information extraction model"""
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's occupation")
    location: str = Field(description="Person's location")

class SentimentAnalysis(BaseModel):
    """Sentiment analysis model"""
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")
    key_emotions: List[str] = Field(description="List of key emotions detected")

class PromptEngineering:
    """Main class for demonstrating prompt engineering techniques"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500
        )
        print("🔧 Prompt Engineering initialized")
    
    def demonstrate_basic_prompting(self):
        """Demonstrate basic prompting techniques"""
        print("\n1️⃣ Basic Prompting Techniques")
        print("-" * 40)
        
        techniques = {
            "Direct": "Translate 'Hello, world!' to French.",
            
            "With Context": """You are a professional translator specializing in French.
Translate the following English text to French: 'Hello, world!'""",
            
            "Step-by-step": """Translate 'Hello, world!' to French. Follow these steps:
1. Identify the key words
2. Find French equivalents
3. Arrange in proper French syntax
4. Provide the final translation""",
            
            "With Examples": """Translate English to French. Here are examples:
English: "Good morning" → French: "Bonjour"
English: "Thank you" → French: "Merci"
English: "Hello, world!" → French: """
        }
        
        for technique, prompt in techniques.items():
            print(f"\n🔹 {technique} Technique:")
            print(f"Prompt: {prompt[:50]}...")
            
            try:
                response = self.llm.invoke(prompt)
                print(f"Response: {response.content}")
            except Exception as e:
                print(f"Error: {e}")
        
        return True
    
    def demonstrate_few_shot_prompting(self):
        """Demonstrate few-shot prompting with examples"""
        print("\n2️⃣ Few-Shot Prompting")
        print("-" * 40)
        
        # Define examples for classification task
        examples = [
            {"text": "I love this product! It's amazing!", "sentiment": "positive"},
            {"text": "This is terrible. Worst purchase ever.", "sentiment": "negative"},
            {"text": "It's okay, nothing special.", "sentiment": "neutral"},
            {"text": "Absolutely fantastic! Highly recommend!", "sentiment": "positive"}
        ]
        
        # Create example template
        example_template = PromptTemplate(
            input_variables=["text", "sentiment"],
            template="Text: {text}\nSentiment: {sentiment}"
        )
        
        # Create few-shot prompt
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix="Classify the sentiment of the following texts:",
            suffix="Text: {text}\nSentiment:",
            input_variables=["text"]
        )
        
        # Test few-shot prompting
        test_texts = [
            "This product exceeded my expectations!",
            "I'm not sure how I feel about this.",
            "Complete waste of money!"
        ]
        
        for text in test_texts:
            prompt = few_shot_prompt.format(text=text)
            print(f"\nTest: {text}")
            
            try:
                response = self.llm.invoke(prompt)
                print(f"Predicted Sentiment: {response.content.strip()}")
            except Exception as e:
                print(f"Error: {e}")
        
        return True
    
    def demonstrate_chain_of_thought(self):
        """Demonstrate chain-of-thought prompting"""
        print("\n3️⃣ Chain-of-Thought Prompting")
        print("-" * 40)
        
        # Math problem with chain of thought
        cot_prompt = """
Solve this step by step:

Problem: A store has 120 apples. They sell 30% in the morning and 25% of the remaining apples in the afternoon. How many apples are left?

Let me think through this step by step:

1. Start with 120 apples
2. Morning sales: 30% of 120 = 0.30 × 120 = 36 apples
3. Remaining after morning: 120 - 36 = 84 apples
4. Afternoon sales: 25% of 84 = 0.25 × 84 = 21 apples
5. Final remaining: 84 - 21 = 63 apples

Answer: 63 apples

Now solve this problem step by step:
Problem: {problem}
"""
        
        template = PromptTemplate(
            input_variables=["problem"],
            template=cot_prompt
        )
        
        test_problem = "A library has 500 books. They lend out 40% on Monday and 30% of the remaining on Tuesday. How many books are left?"
        
        prompt = template.format(problem=test_problem)
        print(f"Problem: {test_problem}")
        
        try:
            response = self.llm.invoke(prompt)
            print(f"Solution:\n{response.content}")
        except Exception as e:
            print(f"Error: {e}")
        
        return True
    
    def demonstrate_structured_output(self):
        """Demonstrate structured output with parsers"""
        print("\n4️⃣ Structured Output Parsing")
        print("-" * 40)
        
        # Person information extraction
        person_parser = PydanticOutputParser(pydantic_object=PersonInfo)
        
        person_prompt = PromptTemplate(
            template="""Extract person information from the following text.
            
Text: {text}

{format_instructions}""",
            input_variables=["text"],
            partial_variables={"format_instructions": person_parser.get_format_instructions()}
        )
        
        test_text = "John Smith is a 35-year-old software engineer living in San Francisco."
        
        prompt = person_prompt.format(text=test_text)
        print(f"Extracting from: {test_text}")
        
        try:
            response = self.llm.invoke(prompt)
            parsed_response = person_parser.parse(response.content)
            
            print("Extracted Information:")
            print(f"Name: {parsed_response.name}")
            print(f"Age: {parsed_response.age}")
            print(f"Occupation: {parsed_response.occupation}")
            print(f"Location: {parsed_response.location}")
            
        except Exception as e:
            print(f"Error: {e}")
            # Try with fixing parser
            try:
                fixing_parser = OutputFixingParser.from_llm(parser=person_parser, llm=self.llm)
                parsed_response = fixing_parser.parse(response.content)
                print("Fixed parsing successful")
            except:
                print("Parsing failed completely")
        
        return True
    
    def demonstrate_conversation_prompting(self):
        """Demonstrate conversation-style prompting"""
        print("\n5️⃣ Conversation Prompting")
        print("-" * 40)
        
        # Create conversation template
        conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful customer service representative for a tech company. Be professional but friendly."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Simulate conversation history
        chat_history = [
            HumanMessage(content="Hi, I'm having issues with my laptop"),
            AIMessage(content="Hello! I'm sorry to hear you're having laptop issues. Can you describe what specific problems you're experiencing?"),
            HumanMessage(content="It keeps freezing and running slowly")
        ]
        
        # Format conversation prompt
        messages = conversation_prompt.format_messages(
            chat_history=chat_history,
            input="What are some troubleshooting steps I can try?"
        )
        
        print("Conversation Context:")
        for msg in chat_history:
            role = "Customer" if isinstance(msg, HumanMessage) else "Agent"
            print(f"{role}: {msg.content}")
        
        print("\nNew Customer Message: What are some troubleshooting steps I can try?")
        
        try:
            response = self.llm.invoke(messages)
            print(f"Agent Response: {response.content}")
        except Exception as e:
            print(f"Error: {e}")
        
        return True
    
    def demonstrate_prompt_optimization(self):
        """Demonstrate prompt optimization techniques"""
        print("\n6️⃣ Prompt Optimization")
        print("-" * 40)
        
        # Create optimizer
        optimizer = PromptOptimizer(self.llm)
        
        # Add test cases for summarization task
        test_cases = [
            {
                "input": {"text": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals."},
                "expected": "AI is machine intelligence that perceives environments and takes goal-directed actions.",
                "weight": 1.0
            },
            {
                "input": {"text": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."},
                "expected": "Machine learning automates model building by learning patterns from data with minimal human input.",
                "weight": 1.0
            }
        ]
        
        for test_case in test_cases:
            optimizer.add_test_case(**test_case)
        
        # Define prompt variations
        base_prompt = "Summarize the following text in one sentence: {text}"
        
        variations = [
            "Provide a concise one-sentence summary of: {text}",
            "In exactly one sentence, summarize: {text}",
            "Create a brief summary (one sentence only) of this text: {text}",
            "Distill the main idea of this text into a single sentence: {text}"
        ]
        
        # Optimize prompts
        best_prompt, best_metrics = optimizer.optimize_prompt_variants(base_prompt, variations)
        
        print(f"\nBest Prompt: {best_prompt}")
        print(f"Accuracy: {best_metrics.accuracy:.3f}")
        print(f"Consistency: {best_metrics.consistency:.3f}")
        
        return True
    
    def run_all_demonstrations(self):
        """Run all prompt engineering demonstrations"""
        print("📝 Advanced Prompt Engineering Guide")
        print("=" * 50)
        
        demonstrations = [
            self.demonstrate_basic_prompting,
            self.demonstrate_few_shot_prompting,
            self.demonstrate_chain_of_thought,
            self.demonstrate_structured_output,
            self.demonstrate_conversation_prompting,
            self.demonstrate_prompt_optimization
        ]
        
        results = []
        for demo in demonstrations:
            try:
                success = demo()
                results.append(success)
            except Exception as e:
                print(f"❌ Demonstration failed: {e}")
                results.append(False)
        
        # Summary
        successful = sum(results)
        total = len(results)
        
        print(f"\n📊 Summary: {successful}/{total} demonstrations successful")
        
        if successful == total:
            print("🎉 All prompt engineering techniques demonstrated successfully!")
        else:
            print("⚠️  Some demonstrations may need attention")
        
        print("\n💡 Key Prompt Engineering Tips:")
        print("1. Be specific and clear in your instructions")
        print("2. Provide examples when possible (few-shot prompting)")
        print("3. Use step-by-step reasoning for complex tasks")
        print("4. Structure your output with parsers")
        print("5. Test and optimize your prompts with real data")
        print("6. Consider conversation context for multi-turn interactions")

def main():
    """Main function"""
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found")
        print("Please set your OpenAI API key in the .env file")
        return
    
    prompt_engineering = PromptEngineering()
    prompt_engineering.run_all_demonstrations()

if __name__ == "__main__":
    main()
```

### 4.2 Advanced Template Management

Create `src/prompts/template_manager.py`:

```python
# src/prompts/template_manager.py
"""
Advanced Template Management System
Centralized management of prompt templates with versioning and optimization
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import hashlib

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

class TemplateType(Enum):
    """Types of prompt templates"""
    SIMPLE = "simple"
    CHAT = "chat"
    FEW_SHOT = "few_shot"
    CONVERSATION = "conversation"

@dataclass
class TemplateMetadata:
    """Metadata for prompt templates"""
    name: str
    version: str
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    template_type: TemplateType
    input_variables: List[str]
    tags: List[str]
    performance_score: Optional[float] = None

@dataclass
class TemplateVersion:
    """Version information for templates"""
    template: str
    metadata: TemplateMetadata
    examples: List[Dict[str, Any]]
    test_cases: List[Dict[str, Any]]

class TemplateManager:
    """Advanced template management system"""
    
    def __init__(self, templates_dir: str = "./templates"):
        self.templates_dir = templates_dir
        self.templates: Dict[str, Dict[str, TemplateVersion]] = {}
        self.llm = ChatOpenAI(temperature=0.3)
        
        # Create templates directory
        os.makedirs(templates_dir, exist_ok=True)
        
        # Load existing templates
        self.load_templates()
    
    def create_template(self, name: str, template: str, 
                       template_type: TemplateType,
                       input_variables: List[str],
                       description: str = "",
                       author: str = "Unknown",
                       tags: List[str] = None,
                       examples: List[Dict] = None,
                       test_cases: List[Dict] = None) -> str:
        """Create a new template version"""
        
        # Generate version hash
        version = self._generate_version_hash(template)
        
        # Create metadata
        metadata = TemplateMetadata(
            name=name,
            version=version,
            description=description,
            author=author,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            template_type=template_type,
            input_variables=input_variables,
            tags=tags or []
        )
        
        # Create template version
        template_version = TemplateVersion(
            template=template,
            metadata=metadata,
            examples=examples or [],
            test_cases=test_cases or []
        )
        
        # Store template
        if name not in self.templates:
            self.templates[name] = {}
        
        self.templates[name][version] = template_version
        
        # Save to disk
        self._save_template(name, version, template_version)
        
        print(f"✅ Created template '{name}' version {version[:8]}")
        return version
    
    def update_template(self, name: str, template: str,
                       description: str = None,
                       examples: List[Dict] = None,
                       test_cases: List[Dict] = None) -> str:
        """Update an existing template"""
        
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        # Get latest version
        latest_version = self.get_latest_version(name)
        old_template = self.templates[name][latest_version]
        
        # Create new version
        version = self._generate_version_hash(template)
        
        # Update metadata
        metadata = old_template.metadata
        metadata.version = version
        metadata.updated_at = datetime.now()
        
        if description:
            metadata.description = description
        
        # Create new template version
        template_version = TemplateVersion(
            template=template,
            metadata=metadata,
            examples=examples or old_template.examples,
            test_cases=test_cases or old_template.test_cases
        )
        
        # Store template
        self.templates[name][version] = template_version
        
        # Save to disk
        self._save_template(name, version, template_version)
        
        print(f"✅ Updated template '{name}' to version {version[:8]}")
        return version
    
    def get_template(self, name: str, version: str = None) -> PromptTemplate:
        """Get a template by name and version"""
        
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        if version is None:
            version = self.get_latest_version(name)
        
        if version not in self.templates[name]:
            raise ValueError(f"Version '{version}' of template '{name}' not found")
        
        template_version = self.templates[name][version]
        
        # Create appropriate template type
        if template_version.metadata.template_type == TemplateType.SIMPLE:
            return PromptTemplate(
                template=template_version.template,
                input_variables=template_version.metadata.input_variables
            )
        elif template_version.metadata.template_type == TemplateType.CHAT:
            return ChatPromptTemplate.from_template(template_version.template)
        else:
            # Default to simple template
            return PromptTemplate(
                template=template_version.template,
                input_variables=template_version.metadata.input_variables
            )
    
    def get_latest_version(self, name: str) -> str:
        """Get the latest version of a template"""
        
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        versions = list(self.templates[name].keys())
        # Sort by creation time
        versions.sort(key=lambda v: self.templates[name][v].metadata.created_at, reverse=True)
        
        return versions[0]
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all templates with metadata"""
        
        templates_list = []
        
        for name, versions in self.templates.items():
            latest_version = self.get_latest_version(name)
            template_version = versions[latest_version]
            
            templates_list.append({
                "name": name,
                "latest_version": latest_version[:8],
                "description": template_version.metadata.description,
                "type": template_version.metadata.template_type.value,
                "author": template_version.metadata.author,
                "created_at": template_version.metadata.created_at.isoformat(),
                "tags": template_version.metadata.tags,
                "versions_count": len(versions)
            })
        
        return templates_list
    
    def test_template(self, name: str, version: str = None) -> Dict[str, Any]:
        """Test a template with its test cases"""
        
        template = self.get_template(name, version)
        version = version or self.get_latest_version(name)
        template_version = self.templates[name][version]
        
        if not template_version.test_cases:
            return {"success": False, "error": "No test cases defined"}
        
        results = []
        
        for i, test_case in enumerate(template_version.test_cases):
            try:
                # Format template
                formatted_prompt = template.format(**test_case["input"])
                
                # Get LLM response
                response = self.llm.invoke(formatted_prompt)
                response_content = response.content if hasattr(response, 'content') else str(response)
                
                # Check expected output if provided
                success = True
                if "expected" in test_case:
                    # Simple keyword matching - can be enhanced
                    expected_keywords = test_case["expected"].lower().split()
                    response_keywords = response_content.lower().split()
                    
                    matches = sum(1 for keyword in expected_keywords if keyword in response_keywords)
                    success = matches / len(expected_keywords) > 0.5  # 50% keyword match
                
                results.append({
                    "test_case": i + 1,
                    "input": test_case["input"],
                    "response": response_content[:200] + "..." if len(response_content) > 200 else response_content,
                    "success": success
                })
                
            except Exception as e:
                results.append({
                    "test_case": i + 1,
                    "input": test_case["input"],
                    "error": str(e),
                    "success": False
                })
        
        # Calculate overall success rate
        success_rate = sum(1 for r in results if r.get("success", False)) / len(results)
        
        return {
            "success": True,
            "template_name": name,
            "version": version[:8],
            "test_results": results,
            "success_rate": success_rate
        }
    
    def search_templates(self, query: str = None, tags: List[str] = None,
                        template_type: TemplateType = None) -> List[Dict[str, Any]]:
        """Search templates by query, tags, or type"""
        
        templates_list = self.list_templates()
        filtered_templates = []
        
        for template_info in templates_list:
            include_template = True
            
            # Filter by query
            if query:
                query_lower = query.lower()
                if (query_lower not in template_info["name"].lower() and
                    query_lower not in template_info["description"].lower()):
                    include_template = False
            
            # Filter by tags
            if tags and include_template:
                template_tags = [tag.lower() for tag in template_info["tags"]]
                query_tags = [tag.lower() for tag in tags]
                
                if not any(tag in template_tags for tag in query_tags):
                    include_template = False
            
            # Filter by type
            if template_type and include_template:
                if template_info["type"] != template_type.value:
                    include_template = False
            
            if include_template:
                filtered_templates.append(template_info)
        
        return filtered_templates
    
    def _generate_version_hash(self, template: str) -> str:
        """Generate version hash for template"""
        template_hash = hashlib.md5(template.encode()).hexdigest()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{template_hash[:8]}"
    
    def _save_template(self, name: str, version: str, template_version: TemplateVersion):
        """Save template to disk"""
        template_dir = os.path.join(self.templates_dir, name)
        os.makedirs(template_dir, exist_ok=True)
        
        template_file = os.path.join(template_dir, f"{version}.yaml")
        
        # Convert to dict for serialization
        template_data = {
            "template": template_version.template,
            "metadata": asdict(template_version.metadata),
            "examples": template_version.examples,
            "test_cases": template_version.test_cases
        }
        
        # Convert datetime objects to strings
        template_data["metadata"]["created_at"] = template_version.metadata.created_at.isoformat()
        template_data["metadata"]["updated_at"] = template_version.metadata.updated_at.isoformat()
        template_data["metadata"]["template_type"] = template_version.metadata.template_type.value
        
        with open(template_file, 'w') as f:
            yaml.dump(template_data, f, default_flow_style=False)
    
    def load_templates(self):
        """Load templates from disk"""
        
        if not os.path.exists(self.templates_dir):
            return
        
        for template_name in os.listdir(self.templates_dir):
            template_dir = os.path.join(self.templates_dir, template_name)
            
            if not os.path.isdir(template_dir):
                continue
            
            self.templates[template_name] = {}
            
            for version_file in os.listdir(template_dir):
                if not version_file.endswith('.yaml'):
                    continue
                
                version = version_file[:-5]  # Remove .yaml extension
                
                try:
                    with open(os.path.join(template_dir, version_file), 'r') as f:
                        template_data = yaml.safe_load(f)
                    
                    # Convert back to objects
                    metadata_dict = template_data["metadata"]
                    metadata_dict["created_at"] = datetime.fromisoformat(metadata_dict["created_at"])
                    metadata_dict["updated_at"] = datetime.fromisoformat(metadata_dict["updated_at"])
                    metadata_dict["template_type"] = TemplateType(metadata_dict["template_type"])
                    
                    metadata = TemplateMetadata(**metadata_dict)
                    
                    template_version = TemplateVersion(
                        template=template_data["template"],
                        metadata=metadata,
                        examples=template_data.get("examples", []),
                        test_cases=template_data.get("test_cases", [])
                    )
                    
                    self.templates[template_name][version] = template_version
                    
                except Exception as e:
                    print(f"⚠️  Failed to load template {template_name}/{version}: {e}")

def demonstrate_template_management():
    """Demonstrate advanced template management"""
    print("📋 Advanced Template Management Demo")
    print("=" * 40)
    
    try:
        # Initialize template manager
        manager = TemplateManager("./demo_templates")
        
        # Create some example templates
        print("Creating example templates...")
        
        # Simple classification template
        classification_template = """
Classify the following text into one of these categories: {categories}

Text: {text}

Category:"""
        
        manager.create_template(
            name="text_classifier",
            template=classification_template,
            template_type=TemplateType.SIMPLE,
            input_variables=["categories", "text"],
            description="Classify text into predefined categories",
            author="Demo",
            tags=["classification", "nlp"],
            test_cases=[
                {
                    "input": {
                        "categories": "positive, negative, neutral",
                        "text": "I love this product!"
                    },
                    "expected": "positive"
                },
                {
                    "input": {
                        "categories": "technology, sports, politics",
                        "text": "The new iPhone features are amazing"
                    },
                    "expected": "technology"
                }
            ]
        )
        
        # Summarization template
        summary_template = """
Please provide a concise summary of the following text in {max_sentences} sentences:

Text: {text}

Summary:"""
        
        manager.create_template(
            name="text_summarizer",
            template=summary_template,
            template_type=TemplateType.SIMPLE,
            input_variables=["text", "max_sentences"],
            description="Summarize text in specified number of sentences",
            author="Demo",
            tags=["summarization", "nlp"],
            test_cases=[
                {
                    "input": {
                        "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals. AI research has been highly successful in developing effective techniques for solving a wide range of problems, from game playing to medical diagnosis.",
                        "max_sentences": "1"
                    },
                    "expected": "AI is machine intelligence used to solve various problems"
                }
            ]
        )
        
        # List all templates
        print("\n📄 Available Templates:")
        templates = manager.list_templates()
        for template in templates:
            print(f"  - {template['name']} (v{template['latest_version']}): {template['description']}")
        
        # Test a template
        print(f"\n🧪 Testing text_classifier template...")
        test_results = manager.test_template("text_classifier")
        
        if test_results["success"]:
            print(f"Success rate: {test_results['success_rate']:.1%}")
            for result in test_results["test_results"]:
                print(f"  Test {result['test_case']}: {'✅' if result.get('success') else '❌'}")
        
        # Search templates
        print(f"\n🔍 Searching templates with tag 'nlp'...")
        search_results = manager.search_templates(tags=["nlp"])
        for result in search_results:
            print(f"  - {result['name']}: {result['description']}")
        
        # Use a template
        print(f"\n🚀 Using text_classifier template...")
        classifier_template = manager.get_template("text_classifier")
        
        formatted_prompt = classifier_template.format(
            categories="positive, negative, neutral",
            text="This movie was absolutely terrible!"
        )
        
        print("Formatted prompt:")
        print(formatted_prompt)
        
        return True
        
    except Exception as e:
        print(f"❌ Template management demo failed: {e}")
        return False

if __name__ == "__main__":
    demonstrate_template_management()
```

---

## Chapter 5: Chains - Building Sequential Operations

### 5.1 Understanding Chain Types

Chains are one of the most powerful features in LangChain, allowing you to combine multiple operations into sequential workflows. This chapter covers all types of chains and how to build custom ones.

Create `examples/intermediate/chain_types.py`:

```python
# examples/intermediate/chain_types.py
"""
Comprehensive Chain Types and Patterns
Demonstrates different types of chains and their use cases
"""

import os
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import (
    LLMChain,
    SimpleSequentialChain,
    SequentialChain,
    TransformChain,
    ConversationChain
)
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseOutputParser
from langchain.callbacks import StdOutCallbackHandler

load_dotenv()

class ChainDemo:
    """Demonstration of different chain types"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=300
        )
        print("🔗 Chain Demo initialized")
    
    def demonstrate_llm_chain(self):
        """1. Basic LLM Chain"""
        print("\n1️⃣ LLM Chain")
        print("-" * 30)
        
        try:
            # Create prompt template
            prompt = PromptTemplate(
                template="Write a {adjective} story about {subject} in {word_count} words.",
                input_variables=["adjective", "subject", "word_count"]
            )
            
            # Create LLM chain
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                verbose=True
            )
            
            # Run chain
            result = chain.run(
                adjective="mysterious",
                subject="a hidden library",
                word_count="50"
            )
            
            print(f"Generated Story:\n{result}")
            return True
            
        except Exception as e:
            print(f"❌ LLM Chain demo failed: {e}")
            return False
    
    def demonstrate_simple_sequential_chain(self):
        """2. Simple Sequential Chain"""
        print("\n2️⃣ Simple Sequential Chain")
        print("-" * 30)
        
        try:
            # First chain: Generate a topic
            topic_template = PromptTemplate(
                template="Generate a creative {category} topic for a blog post:",
                input_variables=["category"]
            )
            topic_chain = LLMChain(llm=self.llm, prompt=topic_template)
            
            # Second chain: Create outline from topic
            outline_template = PromptTemplate(
                template="Create a detailed outline for this blog post topic: {topic}",
                input_variables=["topic"]
            )
            outline_chain = LLMChain(llm=self.llm, prompt=outline_template)
            
            # Third chain: Write introduction from outline
            intro_template = PromptTemplate(
                template="Write an engaging introduction based on this outline: {outline}",
                input_variables=["outline"]
            )
            intro_chain = LLMChain(llm=self.llm, prompt=intro_template)
            
            # Combine into sequential chain
            overall_chain = SimpleSequentialChain(
                chains=[topic_chain, outline_chain, intro_chain],
                verbose=True
            )
            
            # Run the chain
            result = overall_chain.run("technology")
            
            print(f"Final Result (Introduction):\n{result}")
            return True
            
        except Exception as e:
            print(f"❌ Simple Sequential Chain demo failed: {e}")
            return False
    
    def demonstrate_sequential_chain(self):
        """3. Sequential Chain with Multiple Inputs/Outputs"""
        print("\n3️⃣ Sequential Chain")
        print("-" * 30)
        
        try:
            # First chain: Analyze text sentiment
            sentiment_template = PromptTemplate(
                template="Analyze the sentiment of this text: {text}\nSentiment:",
                input_variables=["text"]
            )
            sentiment_chain = LLMChain(
                llm=self.llm, 
                prompt=sentiment_template,
                output_key="sentiment"
            )
            
            # Second chain: Generate response based on sentiment
            response_template = PromptTemplate(
                template="""Based on the original text and its sentiment, generate an appropriate response.

Original text: {text}
Sentiment: {sentiment}

Response:""",
                input_variables=["text", "sentiment"]
            )
            response_chain = LLMChain(
                llm=self.llm,
                prompt=response_template,
                output_key="response"
            )
            
            # Third chain: Suggest follow-up actions
            action_template = PromptTemplate(
                template="""Based on the text, sentiment, and response, suggest 2 follow-up actions.

Text: {text}
Sentiment: {sentiment}
Response: {response}

Follow-up actions:""",
                input_variables=["text", "sentiment", "response"]
            )
            action_chain = LLMChain(
                llm=self.llm,
                prompt=action_template,
                output_key="actions"
            )
            
            # Combine into sequential chain
            overall_chain = SequentialChain(
                chains=[sentiment_chain, response_chain, action_chain],
                input_variables=["text"],
                output_variables=["sentiment", "response", "actions"],
                verbose=True
            )
            
            # Run the chain
            result = overall_chain({
                "text": "I'm really frustrated with the poor customer service I received today."
            })
            
            print(f"Results:")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Response: {result['response']}")
            print(f"Actions: {result['actions']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Sequential Chain demo failed: {e}")
            return False
    
    def demonstrate_transform_chain(self):
        """4. Transform Chain for Data Processing"""
        print("\n4️⃣ Transform Chain")
        print("-" * 30)
        
        try:
            # Define transformation function
            def clean_text(inputs: Dict[str, str]) -> Dict[str, str]:
                text = inputs["text"]
                
                # Clean the text
                cleaned_text = text.strip()
                cleaned_text = ' '.join(cleaned_text.split())  # Normalize whitespace
                cleaned_text = cleaned_text.replace('\n', ' ')
                
                # Count words and characters
                word_count = len(cleaned_text.split())
                char_count = len(cleaned_text)
                
                return {
                    "cleaned_text": cleaned_text,
                    "word_count": word_count,
                    "char_count": char_count,
                    "original_text": text
                }
            
            # Create transform chain
            transform_chain = TransformChain(
                input_variables=["text"],
                output_variables=["cleaned_text", "word_count", "char_count", "original_text"],
                transform=clean_text
            )
            
            # Create summarization chain
            summary_template = PromptTemplate(
                template="""Summarize the following text in exactly 20 words:

Text: {cleaned_text}
(Word count: {word_count}, Character count: {char_count})

Summary:""",
                input_variables=["cleaned_text", "word_count", "char_count"]
            )
            summary_chain = LLMChain(llm=self.llm, prompt=summary_template)
            
            # Combine chains
            overall_chain = SequentialChain(
                chains=[transform_chain, summary_chain],
                input_variables=["text"],
                output_variables=["cleaned_text", "word_count", "char_count", "text"]
            )
            
            # Test with messy text
            messy_text = """   This    is some    
            text with    irregular   spacing
            and    line breaks.       It needs    to be cleaned   
            before   processing.    """
            
            result = overall_chain.run(text=messy_text)
            
            print(f"Original text: {repr(messy_text)}")
            print(f"Cleaned text: {result}")
            
            return True
            
        except Exception as e:
            print(f"❌ Transform Chain demo failed: {e}")
            return False
    
    def demonstrate_router_chain(self):
        """5. Router Chain for Dynamic Routing"""
        print("\n5️⃣ Router Chain")
        print("-" * 30)
        
        try:
            # Define different prompt templates for different tasks
            physics_template = """You are a physics expert. Answer this physics question clearly and accurately:

Question: {input}
Answer:"""
            
            math_template = """You are a mathematics expert. Solve this math problem step by step:

Problem: {input}
Solution:"""
            
            history_template = """You are a history expert. Provide detailed historical information about:

Topic: {input}
Information:"""
            
            # Create prompt infos for router
            prompt_infos = [
                {
                    "name": "physics",
                    "description": "Good for answering questions about physics, mechanics, thermodynamics, etc.",
                    "prompt_template": physics_template
                },
                {
                    "name": "math",
                    "description": "Good for solving mathematical problems, equations, calculations, etc.",
                    "prompt_template": math_template
                },
                {
                    "name": "history",
                    "description": "Good for questions about historical events, figures, and periods.",
                    "prompt_template": history_template
                }
            ]
            
            # Create destination chains
            destination_chains = {}
            for p_info in prompt_infos:
                name = p_info["name"]
                prompt_template = p_info["prompt_template"]
                prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
                chain = LLMChain(llm=self.llm, prompt=prompt)
                destination_chains[name] = chain
            
            # Create default chain
            default_prompt = PromptTemplate(
                template="I don't know how to respond to: {input}",
                input_variables=["input"]
            )
            default_chain = LLMChain(llm=self.llm, prompt=default_prompt)
            
            # Create multi-prompt chain
            chain = MultiPromptChain(
                router_chain=LLMRouterChain.from_llm(self.llm, prompt_infos),
                destination_chains=destination_chains,
                default_chain=default_chain,
                verbose=True
            )
            
            # Test different types of questions
            test_questions = [
                "What is Newton's second law of motion?",
                "Solve for x: 2x + 5 = 15",
                "Who was Napoleon Bonaparte?",
                "What's the weather like today?"  # Should go to default
            ]
            
            for question in test_questions:
                print(f"\nQuestion: {question}")
                try:
                    response = chain.run(question)
                    print(f"Response: {response[:200]}...")
                except Exception as e:
                    print(f"Error: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Router Chain demo failed: {e}")
            return False
    
    def demonstrate_conversation_chain(self):
        """6. Conversation Chain with Memory"""
        print("\n6️⃣ Conversation Chain")
        print("-" * 30)
        
        try:
            # Create memory
            memory = ConversationBufferMemory()
            
            # Create conversation chain
            conversation = ConversationChain(
                llm=self.llm,
                memory=memory,
                verbose=True
            )
            
            # Simulate conversation
            conversations = [
                "Hi, I'm learning about machine learning.",
                "What are the main types of machine learning?",
                "Can you explain supervised learning in more detail?",
                "What did I first tell you I was learning about?"
            ]
            
            for i, user_input in enumerate(conversations, 1):
                print(f"\nTurn {i}")
                print(f"User: {user_input}")
                
                response = conversation.predict(input=user_input)
                print(f"Assistant: {response}")
            
            return True
            
        except Exception as e:
            print(f"❌ Conversation Chain demo failed: {e}")
            return False
    
    def demonstrate_custom_chain(self):
        """7. Custom Chain Implementation"""
        print("\n7️⃣ Custom Chain")
        print("-" * 30)
        
        try:
            from langchain.chains.base import Chain
            
            class EmailGeneratorChain(Chain):
                """Custom chain for generating professional emails"""
                
                llm: ChatOpenAI
                
                @property
                def input_keys(self) -> List[str]:
                    return ["recipient", "subject", "main_points", "tone"]
                
                @property
                def output_keys(self) -> List[str]:
                    return ["email", "subject_line"]
                
                def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
                    recipient = inputs["recipient"]
                    subject = inputs["subject"]
                    main_points = inputs["main_points"]
                    tone = inputs["tone"]
                    
                    # Generate subject line
                    subject_prompt = f"Create a professional email subject line about: {subject}"
                    subject_response = self.llm.invoke(subject_prompt)
                    subject_line = subject_response.content.strip()
                    
                    # Generate email body
                    email_prompt = f"""Write a {tone} professional email with the following details:

Recipient: {recipient}
Subject: {subject}
Main points to cover: {main_points}

Email:"""
                    
                    email_response = self.llm.invoke(email_prompt)
                    email_body = email_response.content
                    
                    return {
                        "email": email_body,
                        "subject_line": subject_line
                    }
            
            # Create and test custom chain
            email_chain = EmailGeneratorChain(llm=self.llm)
            
            result = email_chain({
                "recipient": "team members",
                "subject": "project update and next steps",
                "main_points": "completed phase 1, starting phase 2, need feedback by Friday",
                "tone": "friendly but professional"
            })
            
            print(f"Generated Subject: {result['subject_line']}")
            print(f"Generated Email:\n{result['email']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Custom Chain demo failed: {e}")
            return False
    
    def run_all_demonstrations(self):
        """Run all chain demonstrations"""
        print("🔗 Chain Types and Patterns Demo")
        print("=" * 40)
        
        demonstrations = [
            self.demonstrate_llm_chain,
            self.demonstrate_simple_sequential_chain,
            self.demonstrate_sequential_chain,
            self.demonstrate_transform_chain,
            self.demonstrate_router_chain,
            self.demonstrate_conversation_chain,
            self.demonstrate_custom_chain
        ]
        
        results = []
        for demo in demonstrations:
            try:
                success = demo()
                results.append(success)
            except Exception as e:
                print(f"❌ Demonstration failed: {e}")
                results.append(False)
        
        # Summary
        successful = sum(results)
        total = len(results)
        
        print(f"\n📊 Summary: {successful}/{total} chain demonstrations successful")
        
        if successful == total:
            print("🎉 All chain types demonstrated successfully!")
        else:
            print("⚠️  Some chain demonstrations may need attention")
        
        print("\n💡 Key Chain Concepts:")
        print("1. LLMChain - Basic building block for single LLM operations")
        print("2. Sequential Chains - Chain operations in sequence")
        print("3. Router Chains - Dynamic routing based on input")
        print("4. Transform Chains - Data processing and transformation")
        print("5. Custom Chains - Build specialized chain logic")
        print("6. Memory Integration - Maintain context across calls")

def main():
    """Main function"""
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found")
        print("Please set your OpenAI API key in the .env file")
        return
    
    chain_demo = ChainDemo()
    chain_demo.run_all_demonstrations()

if __name__ == "__main__":
    main()
```

## Chapter 6: Memory Management - Persistent Context

Memory management in LangChain allows your applications to maintain context across conversations and interactions.

### 6.1 Basic Memory Types

```python
# ai/langchain/examples/06_memory_management.py

from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory
)
from langchain.schema import BaseMessage
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
import os
from typing import List, Dict, Any
from datetime import datetime

class MemoryDemonstrator:
    """Comprehensive memory management demonstrations"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.3)
        self.setup_demonstrations()
    
    def setup_demonstrations(self):
        """Set up different memory types for comparison"""
        self.memory_types = {
            'buffer': ConversationBufferMemory(return_messages=True),
            'window': ConversationBufferWindowMemory(k=3, return_messages=True),
            'summary': ConversationSummaryMemory(llm=self.llm, return_messages=True),
            'summary_buffer': ConversationSummaryBufferMemory(
                llm=self.llm, 
                max_token_limit=100,
                return_messages=True
            ),
            'token_buffer': ConversationTokenBufferMemory(
                llm=self.llm,
                max_token_limit=100,
                return_messages=True
            )
        }
    
    def demonstrate_buffer_memory(self):
        """Basic buffer memory - stores all messages"""
        print("=== Buffer Memory Demo ===")
        
        memory = self.memory_types['buffer']
        chain = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=True
        )
        
        # Simulate conversation
        responses = []
        conversation = [
            "My name is Alice and I love programming",
            "What's my name?",
            "What do I love doing?",
            "Tell me about my interests"
        ]
        
        for message in conversation:
            response = chain.predict(input=message)
            responses.append(response)
            print(f"User: {message}")
            print(f"AI: {response}\n")
        
        # Show memory contents
        print("Memory Buffer Contents:")
        print(memory.buffer)
        return responses

### 6.2 Custom Memory Implementation

```python
class CustomProjectMemory(ConversationBufferMemory):
    """Custom memory for project-specific context"""
    
    def __init__(self, project_context: Dict[str, Any] = None):
        super().__init__(return_messages=True)
        self.project_context = project_context or {}
        self.user_preferences = {}
        self.session_metadata = {}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Enhanced context saving with project awareness"""
        # Save standard conversation
        super().save_context(inputs, outputs)
        
        # Extract and store project-relevant information
        user_input = inputs.get('input', '')
        ai_output = outputs.get('response', '')
        
        # Update user preferences
        self._extract_preferences(user_input, ai_output)
        
        # Update project context
        self._update_project_context(user_input, ai_output)
    
    def _extract_preferences(self, user_input: str, ai_output: str):
        """Extract user preferences from conversation"""
        preference_keywords = {
            'language': ['python', 'javascript', 'java', 'go', 'rust'],
            'framework': ['django', 'flask', 'fastapi', 'react', 'vue', 'angular'],
            'database': ['postgresql', 'mysql', 'mongodb', 'redis'],
            'style': ['functional', 'object-oriented', 'minimal', 'verbose']
        }
        
        text = (user_input + ' ' + ai_output).lower()
        
        for category, keywords in preference_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if category not in self.user_preferences:
                        self.user_preferences[category] = set()
                    self.user_preferences[category].add(keyword)

def main():
    """Run all memory demonstrations"""
    print("LangChain Memory Management Comprehensive Demo")
    print("=" * 50)
    
    # Set up API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    demonstrator = MemoryDemonstrator()
    
    try:
        # Basic memory types
        demonstrator.demonstrate_buffer_memory()
        
        # Custom memory example
        print("=== Custom Memory Demo ===")
        custom_memory = CustomProjectMemory({
            'project_name': 'AI Assistant',
            'tech_stack': ['python', 'langchain', 'openai']
        })
        
        chain = ConversationChain(
            llm=demonstrator.llm,
            memory=custom_memory,
            verbose=True
        )
        
        # Simulate project-focused conversation
        project_conversation = [
            "I'm working on a Python web application using FastAPI",
            "I need help with database migrations in SQLAlchemy",
            "I prefer using pytest for testing",
            "Can you help me write tests for my API endpoints?"
        ]
        
        for msg in project_conversation:
            response = chain.predict(input=msg)
            print(f"User: {msg}")
            print(f"AI: {response}\n")
        
    except Exception as e:
        print(f"Error in memory demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

## Chapter 7: Document Processing and Retrieval

LangChain excels at processing and retrieving information from various document types.

### 7.1 Document Loaders and Processing

```python
# ai/langchain/examples/07_document_processing.py

from langchain.document_loaders import (
    TextLoader, PDFLoader, CSVLoader, JSONLoader,
    DirectoryLoader, UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import (
    CharacterTextSplitter, RecursiveCharacterTextSplitter,
    TokenTextSplitter, MarkdownHeaderTextSplitter
)
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

class DocumentProcessor:
    """Comprehensive document processing system"""
    
    def __init__(self, embeddings_model: str = "text-embedding-ada-002"):
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.llm = ChatOpenAI(temperature=0.1)
        self.processed_documents: List[Document] = []
        self.vector_store: Optional[Any] = None
    
    def load_documents_from_directory(self, directory_path: str, 
                                    file_types: List[str] = None) -> List[Document]:
        """Load documents from directory with specific file types"""
        if file_types is None:
            file_types = ['txt', 'md', 'pdf', 'csv', 'json', 'html']
        
        documents = []
        
        for file_type in file_types:
            try:
                if file_type == 'txt':
                    loader = DirectoryLoader(
                        directory_path, 
                        glob=f"**/*.{file_type}",
                        loader_cls=TextLoader
                    )
                elif file_type == 'md':
                    loader = DirectoryLoader(
                        directory_path,
                        glob=f"**/*.{file_type}",
                        loader_cls=UnstructuredMarkdownLoader
                    )
                else:
                    continue
                
                file_docs = loader.load()
                documents.extend(file_docs)
                print(f"Loaded {len(file_docs)} {file_type} documents")
                
            except Exception as e:
                print(f"Error loading {file_type} files: {e}")
        
        return documents
    
    def smart_text_splitting(self, documents: List[Document]) -> List[Document]:
        """Intelligent text splitting based on document type and content"""
        split_documents = []
        
        for doc in documents:
            content = doc.page_content
            source = doc.metadata.get('source', '')
            
            if source.endswith('.md'):
                # Use markdown-aware splitting
                splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[
                        ("#", "Header 1"),
                        ("##", "Header 2"),
                        ("###", "Header 3"),
                    ]
                )
                splits = splitter.split_text(content)
            else:
                # Default splitting
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100
                )
                splits = splitter.split_documents([doc])
            
            split_documents.extend(splits)
        
        return split_documents

def create_sample_documents():
    """Create sample documents for demonstration"""
    docs_dir = Path("./sample_docs")
    docs_dir.mkdir(exist_ok=True)
    
    sample_files = {
        "python_guide.md": """# Python Programming Guide

## Introduction
Python is a versatile programming language.

## Functions
```python
def greet(name):
    return f"Hello, {name}!"
```
""",
        "project_readme.txt": """Project Setup Instructions

1. Install dependencies: pip install -r requirements.txt
2. Set environment variables
3. Run database migrations
"""
    }
    
    for filename, content in sample_files.items():
        with open(docs_dir / filename, 'w') as f:
            f.write(content)
    
    return str(docs_dir)

def main():
    """Run comprehensive document processing demonstration"""
    print("LangChain Document Processing & Retrieval Demo")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        sample_docs_dir = create_sample_documents()
        processor = DocumentProcessor()
        
        documents = processor.load_documents_from_directory(
            sample_docs_dir,
            file_types=['txt', 'md']
        )
        
        split_docs = processor.smart_text_splitting(documents)
        print(f"Processed {len(split_docs)} document chunks")
        
    except Exception as e:
        print(f"Error in document processing: {e}")

if __name__ == "__main__":
    main()
```

## Chapter 8: Vector Stores and Embeddings

Vector stores enable semantic search and retrieval in LangChain applications.

### 8.1 Vector Store Implementation

```python
# ai/langchain/examples/08_vector_stores.py

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS, Pinecone
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
import os
from typing import List, Dict, Any
import numpy as np

class VectorStoreDemo:
    """Comprehensive vector store demonstrations"""
    
    def __init__(self):
        self.openai_embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.1)
    
    def demonstrate_chroma(self, documents: List[Document]):
        """Demonstrate Chroma vector store"""
        print("=== Chroma Vector Store Demo ===")
        
        # Create Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.openai_embeddings,
            persist_directory="./chroma_db"
        )
        
        # Persist the database
        vectorstore.persist()
        
        # Test similarity search
        query = "How to set up Python environment?"
        docs = vectorstore.similarity_search(query, k=3)
        
        print(f"Query: {query}")
        print(f"Found {len(docs)} similar documents:")
        for i, doc in enumerate(docs):
            print(f"{i+1}. {doc.metadata.get('source', 'Unknown')}")
            print(f"   Content: {doc.page_content[:100]}...")
        
        return vectorstore

def main():
    """Run vector store demonstrations"""
    print("LangChain Vector Stores Demo")
    print("=" * 40)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="Python is a programming language used for web development, data analysis, and AI.",
            metadata={"source": "python_intro.txt", "topic": "programming"}
        ),
        Document(
            page_content="To set up a Python environment, use virtual environments with pip or conda.",
            metadata={"source": "setup_guide.txt", "topic": "setup"}
        ),
        Document(
            page_content="LangChain is a framework for developing applications with language models.",
            metadata={"source": "langchain_intro.txt", "topic": "ai"}
        )
    ]
    
    demo = VectorStoreDemo()
    demo.demonstrate_chroma(sample_docs)

if __name__ == "__main__":
    main()
```

## Chapter 9: Agents and Tools

Agents in LangChain can reason about and use tools to accomplish complex tasks.

### 9.1 Creating Custom Tools

```python
# ai/langchain/examples/09_agents_tools.py

from langchain.agents import Tool, AgentType, initialize_agent
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from pydantic import BaseModel
import os
import requests
from typing import Optional, Type, List, Dict, Any

class CalculatorTool(BaseTool):
    """Custom calculator tool"""
    name = "calculator"
    description = "Useful for mathematical calculations"
    
    def _run(self, query: str) -> str:
        """Execute the calculator"""
        try:
            # Simple eval for demo (in production, use a safer parser)
            result = eval(query)
            return f"The answer is {result}"
        except Exception as e:
            return f"Error in calculation: {e}"
    
    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)

class WebSearchTool(BaseTool):
    """Simple web search tool (placeholder)"""
    name = "web_search"
    description = "Search the web for current information"
    
    def _run(self, query: str) -> str:
        """Simulate web search"""
        return f"Search results for '{query}': [This is a simulated search result]"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

def main():
    """Run agent demonstrations"""
    print("LangChain Agents and Tools Demo")
    print("=" * 40)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize LLM
    llm = ChatOpenAI(temperature=0.1)
    
    # Create tools
    tools = [
        CalculatorTool(),
        WebSearchTool(),
        Tool(
            name="string_length",
            description="Get the length of a string",
            func=lambda x: f"The string '{x}' has {len(x)} characters"
        )
    ]
    
    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Test queries
    queries = [
        "Calculate 25 * 47 + 123",
        "What's the length of 'LangChain'?",
        "Search for information about Python programming"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        try:
            response = agent.run(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

## Chapter 10: Advanced Patterns and Optimization

### 10.1 Performance Optimization

```python
# ai/langchain/examples/10_optimization.py

from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler, FileCallbackHandler
import time
import os

class PerformanceOptimizer:
    """Demonstrate performance optimization techniques"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.1)
    
    def demonstrate_caching(self):
        """Show the impact of caching"""
        print("=== Caching Demonstration ===")
        
        # Without cache
        print("Without cache:")
        start_time = time.time()
        response1 = self.llm.predict("What is the capital of France?")
        time1 = time.time() - start_time
        print(f"First call took: {time1:.2f} seconds")
        print(f"Response: {response1[:100]}...")
        
        # With cache
        set_llm_cache(InMemoryCache())
        print("\nWith cache:")
        start_time = time.time()
        response2 = self.llm.predict("What is the capital of France?")
        time2 = time.time() - start_time
        print(f"Cached call took: {time2:.2f} seconds")
        print(f"Response: {response2[:100]}...")
        
        print(f"\nSpeedup: {time1/time2:.2f}x faster with cache")

def main():
    """Run optimization demonstrations"""
    print("LangChain Performance Optimization Demo")
    print("=" * 45)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    optimizer = PerformanceOptimizer()
    optimizer.demonstrate_caching()

if __name__ == "__main__":
    main()
```

---

## Chapter 11: Real-World Applications

This chapter demonstrates complete, production-ready applications using LangChain to solve real business problems.

### 11.1 Customer Support Chatbot

A comprehensive customer support system with knowledge base integration, conversation memory, and escalation handling.

```python
# ai/langchain/examples/11_customer_support.py

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import logging

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings  
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.base import BaseCallbackHandler

@dataclass
class CustomerProfile:
    customer_id: str
    name: str
    tier: str  # "premium", "standard", "basic"
    previous_issues: List[str]
    account_status: str

@dataclass 
class SupportTicket:
    ticket_id: str
    customer_id: str
    category: str
    priority: str
    description: str
    created_at: datetime
    status: str = "open"

class SupportMetricsCallback(BaseCallbackHandler):
    """Callback to track support chatbot metrics"""
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "resolved_queries": 0,
            "escalated_queries": 0,
            "avg_response_time": 0.0,
            "knowledge_base_hits": 0
        }
        self.start_time = None
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        self.start_time = datetime.now()
        self.metrics["total_queries"] += 1
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        if self.start_time:
            response_time = (datetime.now() - self.start_time).total_seconds()
            current_avg = self.metrics["avg_response_time"]
            total = self.metrics["total_queries"]
            self.metrics["avg_response_time"] = ((current_avg * (total - 1)) + response_time) / total

class CustomerSupportChatbot:
    """Production-ready customer support chatbot with knowledge base integration"""
    
    def __init__(self, knowledge_base_path: str, persist_directory: str = "support_vectordb"):
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo-16k"
        )
        self.embeddings = OpenAIEmbeddings()
        self.persist_directory = persist_directory
        self.metrics_callback = SupportMetricsCallback()
        
        # Load knowledge base
        self.vectorstore = self._load_knowledge_base(knowledge_base_path)
        
        # Setup conversation chain
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Remember last 10 exchanges
        )
        
        # Custom support prompt
        support_prompt = PromptTemplate(
            template="""You are a helpful customer support representative. Use the following context from our knowledge base to answer the customer's question. 

Customer Tier: {customer_tier}
Previous Issues: {previous_issues}

Knowledge Base Context:
{context}

Chat History:
{chat_history}

Customer Question: {question}

Instructions:
- Be professional, empathetic, and helpful
- Reference specific knowledge base information when applicable
- For premium customers, offer additional assistance
- If you cannot resolve the issue, suggest escalation
- Always ask follow-up questions to better understand the issue

Support Response:""",
            input_variables=["context", "question", "chat_history", "customer_tier", "previous_issues"]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": support_prompt},
            callbacks=[self.metrics_callback],
            return_source_documents=True
        )
        
        # Intent classification
        self.intent_classifier = self._setup_intent_classifier()
        
        # Escalation rules
        self.escalation_keywords = [
            "refund", "cancel subscription", "billing dispute", 
            "data breach", "legal", "complaint", "manager", "supervisor"
        ]
    
    def _load_knowledge_base(self, knowledge_base_path: str) -> Chroma:
        """Load or create knowledge base vector store"""
        
        if os.path.exists(self.persist_directory):
            # Load existing vector store
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            # Create new vector store from knowledge base
            with open(knowledge_base_path, 'r') as file:
                knowledge_content = file.read()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            
            texts = text_splitter.split_text(knowledge_content)
            documents = [Document(page_content=text) for text in texts]
            
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            vectorstore.persist()
            return vectorstore
    
    def _setup_intent_classifier(self):
        """Setup intent classification for routing queries"""
        intent_prompt = PromptTemplate(
            template="""Classify the customer support intent from this message:

Message: {message}

Intent categories:
- technical_issue: Technical problems with product
- billing_inquiry: Questions about billing or payments  
- account_management: Account settings, profile changes
- product_information: Questions about features, pricing
- complaint: Complaints or negative feedback
- compliment: Positive feedback or compliments
- escalation_request: Explicit request to speak to manager/supervisor

Return only the intent category:""",
            input_variables=["message"]
        )
        
        return intent_prompt | self.llm
    
    def classify_intent(self, message: str) -> str:
        """Classify customer message intent"""
        try:
            result = self.intent_classifier.invoke({"message": message})
            return result.content.strip().lower()
        except Exception as e:
            logging.error(f"Intent classification failed: {e}")
            return "general_inquiry"
    
    def should_escalate(self, message: str, intent: str) -> bool:
        """Determine if query should be escalated to human agent"""
        
        # Check for explicit escalation keywords
        message_lower = message.lower()
        if any(keyword in message_lower for keyword in self.escalation_keywords):
            return True
        
        # Check intent-based escalation
        escalation_intents = ["complaint", "escalation_request", "billing_dispute"]
        if intent in escalation_intents:
            return True
        
        # Check conversation length (too many back-and-forth might need human)
        if len(self.memory.chat_memory.messages) > 10:
            return True
        
        return False
    
    def handle_escalation(self, customer_profile: CustomerProfile, message: str) -> Dict[str, Any]:
        """Handle escalation to human agent"""
        ticket = SupportTicket(
            ticket_id=f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            customer_id=customer_profile.customer_id,
            category="escalation",
            priority="high" if customer_profile.tier == "premium" else "medium",
            description=f"Escalated from chatbot: {message}",
            created_at=datetime.now()
        )
        
        self.metrics_callback.metrics["escalated_queries"] += 1
        
        return {
            "escalated": True,
            "ticket": ticket,
            "message": f"I've created ticket {ticket.ticket_id} and connected you with our human support team. They'll be with you shortly."
        }
    
    def chat(self, message: str, customer_profile: CustomerProfile) -> Dict[str, Any]:
        """Main chat interface"""
        
        # Classify intent
        intent = self.classify_intent(message)
        
        # Check for escalation
        if self.should_escalate(message, intent):
            return self.handle_escalation(customer_profile, message)
        
        try:
            # Prepare context for the chain
            context = {
                "question": message,
                "customer_tier": customer_profile.tier,
                "previous_issues": ", ".join(customer_profile.previous_issues) if customer_profile.previous_issues else "None"
            }
            
            # Get response from chain
            result = self.qa_chain(context)
            
            # Track knowledge base usage
            if result.get("source_documents"):
                self.metrics_callback.metrics["knowledge_base_hits"] += 1
            
            # Check if response seems to resolve the issue
            if self._is_resolution_response(result["answer"]):
                self.metrics_callback.metrics["resolved_queries"] += 1
            
            return {
                "escalated": False,
                "response": result["answer"],
                "sources": [doc.page_content[:200] + "..." for doc in result.get("source_documents", [])],
                "intent": intent,
                "confidence": self._calculate_confidence(result)
            }
            
        except Exception as e:
            logging.error(f"Chat processing failed: {e}")
            return {
                "escalated": False,
                "response": "I'm experiencing technical difficulties. Let me connect you with a human agent.",
                "error": True
            }
    
    def _is_resolution_response(self, response: str) -> bool:
        """Simple heuristic to determine if response likely resolves the issue"""
        resolution_indicators = [
            "should resolve", "fix this", "solution is", "try this",
            "this will help", "follow these steps"
        ]
        return any(indicator in response.lower() for indicator in resolution_indicators)
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score for the response"""
        score = 0.5  # Base confidence
        
        # Boost confidence if we have source documents
        if result.get("source_documents"):
            score += 0.2
        
        # Boost confidence based on number of sources
        num_sources = len(result.get("source_documents", []))
        score += min(0.2, num_sources * 0.05)
        
        # Response length heuristic (longer responses often more helpful)
        response_length = len(result.get("answer", ""))
        if response_length > 100:
            score += 0.1
        
        return min(1.0, score)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get chatbot performance metrics"""
        return {
            **self.metrics_callback.metrics,
            "resolution_rate": (
                self.metrics_callback.metrics["resolved_queries"] / 
                max(1, self.metrics_callback.metrics["total_queries"])
            ),
            "escalation_rate": (
                self.metrics_callback.metrics["escalated_queries"] / 
                max(1, self.metrics_callback.metrics["total_queries"])
            )
        }
    
    def reset_conversation(self):
        """Reset conversation memory"""
        self.memory.clear()

def create_sample_knowledge_base():
    """Create sample knowledge base for demonstration"""
    knowledge_base = """
# Customer Support Knowledge Base

## Account Management

### How to reset password
1. Go to login page and click "Forgot Password"
2. Enter your email address
3. Check your email for reset link
4. Follow the link and create new password
5. Password must be at least 8 characters with numbers and letters

### How to update billing information  
1. Log into your account
2. Navigate to Account Settings > Billing
3. Click "Update Payment Method"
4. Enter new credit card or bank account details
5. Save changes - you'll see confirmation

## Technical Issues

### App won't load or crashes
First try these basic troubleshooting steps:
1. Close and reopen the app
2. Restart your device
3. Check your internet connection
4. Update to latest app version
5. Clear app cache (Android) or reinstall (iOS)

If problem persists, it may be a server issue. Check our status page or contact support.

### Slow performance
Performance issues can be caused by:
- Poor internet connection
- Device storage running low  
- Too many apps running in background
- Outdated app version

Try closing other apps and freeing up storage space.

## Billing and Subscriptions

### Subscription tiers
- Basic: $9.99/month - Core features
- Premium: $19.99/month - Advanced features + priority support  
- Enterprise: $49.99/month - All features + dedicated account manager

### Refund policy
- Full refund within 30 days of purchase
- Prorated refunds for annual subscriptions
- No refunds for partial months
- Enterprise customers have custom refund terms

### Billing cycle and charges
- Subscriptions renew automatically each month/year
- Charges appear 1-3 business days before renewal date
- Failed payments result in 7-day grace period
- Account suspended after grace period expires

## Product Features

### Data export
Premium and Enterprise customers can export their data:
1. Go to Account Settings > Data Export
2. Select data types to include
3. Choose export format (CSV, JSON, PDF)  
4. Click "Generate Export" - you'll get email when ready
5. Download link expires after 48 hours

### Integration capabilities
Our API supports integration with:
- CRM systems (Salesforce, HubSpot)
- Analytics platforms (Google Analytics, Mixpanel)
- Communication tools (Slack, Teams)
- Custom applications via REST API

Enterprise customers get dedicated integration support.
"""
    
    with open("support_knowledge_base.txt", "w") as f:
        f.write(knowledge_base)
    
    return "support_knowledge_base.txt"

def simulate_customer_conversations():
    """Simulate realistic customer support conversations"""
    
    # Create sample knowledge base
    kb_path = create_sample_knowledge_base()
    
    # Initialize chatbot
    chatbot = CustomerSupportChatbot(kb_path)
    
    # Sample customer profiles
    customers = [
        CustomerProfile(
            customer_id="CUST-001",
            name="Alice Johnson", 
            tier="premium",
            previous_issues=["billing inquiry", "password reset"],
            account_status="active"
        ),
        CustomerProfile(
            customer_id="CUST-002", 
            name="Bob Smith",
            tier="basic",
            previous_issues=[],
            account_status="active"
        ),
        CustomerProfile(
            customer_id="CUST-003",
            name="Carol Williams",
            tier="enterprise", 
            previous_issues=["integration support", "data export"],
            account_status="active"
        )
    ]
    
    # Sample conversations
    conversations = [
        {
            "customer": customers[0],
            "messages": [
                "Hi, I'm having trouble logging into my account",
                "I tried the forgot password but didn't receive an email",
                "Yes, I checked my spam folder too"
            ]
        },
        {
            "customer": customers[1], 
            "messages": [
                "The app keeps crashing on my phone",
                "I'm using an iPhone 12 with iOS 15",
                "I tried restarting but it didn't help"
            ]
        },
        {
            "customer": customers[2],
            "messages": [
                "I need help setting up the Salesforce integration", 
                "We're an enterprise customer and need this working ASAP",
                "Can I speak to our dedicated account manager?"
            ]
        }
    ]
    
    print("Customer Support Chatbot Demonstration")
    print("=" * 50)
    
    for i, conversation in enumerate(conversations, 1):
        print(f"\n--- Conversation {i}: {conversation['customer'].name} ({conversation['customer'].tier}) ---")
        
        # Reset conversation for each customer
        chatbot.reset_conversation()
        
        for message in conversation["messages"]:
            print(f"\nCustomer: {message}")
            
            response = chatbot.chat(message, conversation["customer"])
            
            if response["escalated"]:
                print(f"System: {response['message']}")
                print(f"Ticket ID: {response['ticket'].ticket_id}")
                break
            else:
                print(f"Chatbot: {response['response']}")
                if response.get("sources"):
                    print("📚 Knowledge base sources referenced:")
                    for j, source in enumerate(response["sources"][:2], 1):
                        print(f"   {j}. {source}")
        
        print(f"\nConversation completed for {conversation['customer'].name}")
    
    # Show metrics
    print(f"\n--- Chatbot Performance Metrics ---")
    metrics = chatbot.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")

def main():
    """Run customer support chatbot demonstration"""
    print("Setting up Customer Support Chatbot...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        simulate_customer_conversations()
        print("\n✅ Customer support demonstration completed!")
        
    except Exception as e:
        print(f"❌ Error in customer support demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

### 11.2 Document Analysis and Q&A System

An intelligent document analysis system for processing legal documents, contracts, and research papers.

```python
# ai/langchain/examples/11_document_analyzer.py

import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import PyPDF2
import docx
from io import BytesIO

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.document_loaders import UnstructuredFileLoader

@dataclass
class DocumentMetadata:
    filename: str
    file_type: str
    pages: int
    word_count: int
    upload_date: datetime
    processed: bool = False

@dataclass 
class AnalysisResult:
    document_id: str
    summary: str
    key_points: List[str]
    entities: List[Dict[str, Any]]
    topics: List[str]
    sentiment: str
    confidence_score: float

class DocumentProcessor:
    """Handles document loading and preprocessing"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt', '.md']
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
    
    def load_document(self, file_path: str) -> Tuple[str, DocumentMetadata]:
        """Load document and extract metadata"""
        
        file_extension = os.path.splitext(file_path.lower())[1]
        
        if file_extension == '.pdf':
            return self._load_pdf(file_path)
        elif file_extension == '.docx':
            return self._load_docx(file_path)
        elif file_extension in ['.txt', '.md']:
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_pdf(self, file_path: str) -> Tuple[str, DocumentMetadata]:
        """Load PDF document"""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            metadata = DocumentMetadata(
                filename=os.path.basename(file_path),
                file_type="pdf",
                pages=len(reader.pages),
                word_count=len(text.split()),
                upload_date=datetime.now()
            )
            
            return text, metadata
    
    def _load_docx(self, file_path: str) -> Tuple[str, DocumentMetadata]:
        """Load Word document"""
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        metadata = DocumentMetadata(
            filename=os.path.basename(file_path),
            file_type="docx", 
            pages=1,  # Word doesn't have clear page breaks
            word_count=len(text.split()),
            upload_date=datetime.now()
        )
        
        return text, metadata
    
    def _load_text(self, file_path: str) -> Tuple[str, DocumentMetadata]:
        """Load text document"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            
            metadata = DocumentMetadata(
                filename=os.path.basename(file_path),
                file_type=os.path.splitext(file_path)[1][1:],
                pages=1,
                word_count=len(text.split()),
                upload_date=datetime.now()
            )
            
            return text, metadata
    
    def chunk_document(self, text: str, metadata: DocumentMetadata) -> List[Document]:
        """Split document into chunks for processing"""
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "filename": metadata.filename,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "file_type": metadata.file_type,
                    "word_count": len(chunk.split())
                }
            )
            documents.append(doc)
        
        return documents

class DocumentAnalyzer:
    """Advanced document analysis and Q&A system"""
    
    def __init__(self, persist_directory: str = "document_vectordb"):
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo-16k"
        )
        self.embeddings = OpenAIEmbeddings()
        self.persist_directory = persist_directory
        self.processor = DocumentProcessor()
        
        # Initialize vector store
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        # Setup analysis chains
        self._setup_analysis_chains()
        
        # Document registry
        self.documents_registry = {}
    
    def _setup_analysis_chains(self):
        """Setup specialized analysis chains"""
        
        # Summarization chain
        summarize_prompt = PromptTemplate(
            template="""Write a comprehensive summary of the following document:

{text}

Provide a summary that includes:
1. Main purpose/objective of the document
2. Key arguments or findings
3. Important details and supporting evidence
4. Conclusions or recommendations

Summary:""",
            input_variables=["text"]
        )
        
        self.summarize_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=summarize_prompt,
            combine_prompt=summarize_prompt
        )
        
        # Entity extraction chain
        entity_prompt = PromptTemplate(
            template="""Extract key entities from the following text:

{text}

Identify and categorize entities:
- PERSON: Names of people
- ORGANIZATION: Company names, institutions
- LOCATION: Places, addresses, countries
- DATE: Dates and time periods  
- MONEY: Monetary amounts
- LEGAL: Legal terms, case names, statutes
- TECHNICAL: Technical terms, product names

Return as JSON format:
{{"entities": [{{"text": "entity", "category": "CATEGORY", "context": "surrounding context"}}]}}

Entities:""",
            input_variables=["text"]
        )
        
        self.entity_chain = entity_prompt | self.llm
        
        # Topic extraction
        topic_prompt = PromptTemplate(
            template="""Analyze the following document and identify the main topics:

{text}

Extract 5-10 key topics that best describe the content. Topics should be:
- Specific and descriptive
- Relevant to the main content
- Useful for categorization

Return as a comma-separated list:

Topics:""",
            input_variables=["text"]
        )
        
        self.topic_chain = topic_prompt | self.llm
        
        # Q&A chain
        qa_prompt = PromptTemplate(
            template="""Use the following context to answer the question. If you cannot answer based on the context provided, say so explicitly.

Context: {context}

Question: {question}

Provide a detailed answer with references to specific parts of the document when possible:

Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
    
    def add_document(self, file_path: str) -> str:
        """Add document to the analysis system"""
        
        # Load and process document
        text, metadata = self.processor.load_document(file_path)
        chunks = self.processor.chunk_document(text, metadata)
        
        # Generate document ID
        doc_id = f"doc_{len(self.documents_registry) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Add metadata to chunks
        for chunk in chunks:
            chunk.metadata["document_id"] = doc_id
        
        # Add to vector store
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
        
        # Register document
        metadata.processed = True
        self.documents_registry[doc_id] = {
            "metadata": metadata,
            "text": text,
            "chunks": len(chunks)
        }
        
        return doc_id
    
    def analyze_document(self, doc_id: str) -> AnalysisResult:
        """Perform comprehensive document analysis"""
        
        if doc_id not in self.documents_registry:
            raise ValueError(f"Document {doc_id} not found")
        
        document_data = self.documents_registry[doc_id]
        text = document_data["text"]
        
        try:
            # Generate summary
            summary_result = self.summarize_chain.invoke({"input_documents": [Document(page_content=text)]})
            summary = summary_result["output_text"]
            
            # Extract entities
            entity_result = self.entity_chain.invoke({"text": text[:4000]})  # Limit for token constraints
            try:
                entities_data = json.loads(entity_result.content)
                entities = entities_data.get("entities", [])
            except json.JSONDecodeError:
                entities = []
            
            # Extract topics
            topic_result = self.topic_chain.invoke({"text": text[:4000]})
            topics = [topic.strip() for topic in topic_result.content.split(",")]
            
            # Simple sentiment analysis
            sentiment = self._analyze_sentiment(text[:2000])
            
            # Calculate confidence based on document length and processing success
            confidence = self._calculate_confidence(len(text), summary, entities, topics)
            
            # Extract key points from summary
            key_points = self._extract_key_points(summary)
            
            return AnalysisResult(
                document_id=doc_id,
                summary=summary,
                key_points=key_points,
                entities=entities,
                topics=topics,
                sentiment=sentiment,
                confidence_score=confidence
            )
            
        except Exception as e:
            print(f"Analysis failed for {doc_id}: {e}")
            return AnalysisResult(
                document_id=doc_id,
                summary="Analysis failed",
                key_points=[],
                entities=[],
                topics=[],
                sentiment="unknown",
                confidence_score=0.0
            )
    
    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis"""
        sentiment_prompt = PromptTemplate(
            template="""Analyze the overall sentiment/tone of this document:

{text}

Classify as one of: positive, negative, neutral, formal, technical, legal

Sentiment:""",
            input_variables=["text"]
        )
        
        try:
            result = (sentiment_prompt | self.llm).invoke({"text": text})
            return result.content.strip().lower()
        except:
            return "neutral"
    
    def _extract_key_points(self, summary: str) -> List[str]:
        """Extract key points from summary"""
        key_points_prompt = PromptTemplate(
            template="""From the following summary, extract 5-7 key points as bullet points:

{summary}

Key Points:""",
            input_variables=["summary"]
        )
        
        try:
            result = (key_points_prompt | self.llm).invoke({"summary": summary})
            points = [point.strip("- ").strip() for point in result.content.split("\n") if point.strip()]
            return [point for point in points if len(point) > 10][:7]
        except:
            return []
    
    def _calculate_confidence(self, text_length: int, summary: str, entities: List, topics: List) -> float:
        """Calculate confidence score for analysis"""
        confidence = 0.5  # Base confidence
        
        # Text length factor
        if text_length > 1000:
            confidence += 0.1
        if text_length > 5000:
            confidence += 0.1
        
        # Analysis completeness
        if len(summary) > 100:
            confidence += 0.1
        if len(entities) > 0:
            confidence += 0.1
        if len(topics) > 2:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def ask_question(self, question: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Ask questions about documents"""
        
        if doc_id:
            # Filter retriever for specific document
            filtered_retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 5,
                    "filter": {"document_id": doc_id}
                }
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff", 
                retriever=filtered_retriever,
                return_source_documents=True
            )
            
            result = qa_chain({"query": question})
        else:
            # Search across all documents
            result = self.qa_chain({"query": question})
        
        return {
            "question": question,
            "answer": result["result"],
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "filename": doc.metadata.get("filename", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0)
                }
                for doc in result["source_documents"]
            ],
            "confidence": len(result["source_documents"]) / 5.0  # Simple confidence metric
        }
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of all processed documents"""
        return [
            {
                "doc_id": doc_id,
                "filename": data["metadata"].filename,
                "file_type": data["metadata"].file_type,
                "pages": data["metadata"].pages,
                "word_count": data["metadata"].word_count,
                "upload_date": data["metadata"].upload_date.isoformat(),
                "chunks": data["chunks"]
            }
            for doc_id, data in self.documents_registry.items()
        ]
    
    def compare_documents(self, doc_id1: str, doc_id2: str) -> Dict[str, Any]:
        """Compare two documents"""
        
        if doc_id1 not in self.documents_registry or doc_id2 not in self.documents_registry:
            raise ValueError("One or both documents not found")
        
        doc1_data = self.documents_registry[doc_id1]
        doc2_data = self.documents_registry[doc_id2]
        
        comparison_prompt = PromptTemplate(
            template="""Compare these two documents and provide a detailed analysis:

Document 1 ({filename1}):
{text1}

Document 2 ({filename2}):
{text2}

Provide comparison covering:
1. Similarities in content and themes
2. Key differences and contrasts  
3. Writing style and tone differences
4. Unique aspects of each document
5. Overall relationship between documents

Comparison Analysis:""",
            input_variables=["filename1", "text1", "filename2", "text2"]
        )
        
        # Limit text for API constraints
        text1 = doc1_data["text"][:3000]
        text2 = doc2_data["text"][:3000]
        
        try:
            result = (comparison_prompt | self.llm).invoke({
                "filename1": doc1_data["metadata"].filename,
                "text1": text1,
                "filename2": doc2_data["metadata"].filename,
                "text2": text2
            })
            
            return {
                "doc1": {
                    "id": doc_id1,
                    "filename": doc1_data["metadata"].filename
                },
                "doc2": {
                    "id": doc_id2,
                    "filename": doc2_data["metadata"].filename
                },
                "comparison": result.content
            }
            
        except Exception as e:
            return {
                "error": f"Comparison failed: {e}"
            }

def create_sample_documents():
    """Create sample documents for testing"""
    
    # Sample business proposal
    proposal = """
# Business Proposal: AI-Powered Customer Analytics Platform

## Executive Summary

Our company proposes to develop an AI-powered customer analytics platform that will revolutionize how businesses understand and engage with their customers. The platform will leverage machine learning algorithms to provide real-time insights, predictive analytics, and personalized recommendations.

## Market Opportunity

The global customer analytics market is projected to reach $24.2 billion by 2025, growing at a CAGR of 15.3%. Current solutions in the market lack the sophisticated AI capabilities and real-time processing power that modern businesses require.

## Proposed Solution

### Core Features
1. Real-time customer behavior tracking
2. Predictive churn analysis
3. Personalized product recommendations
4. Automated marketing campaign optimization
5. Customer lifetime value prediction

### Technical Architecture
- Cloud-native architecture on AWS/Azure
- Machine learning pipeline using Python/TensorFlow
- Real-time data processing with Apache Kafka
- RESTful APIs for third-party integrations
- Advanced visualization dashboard

## Financial Projections

Year 1: $2.5M revenue, $1.8M expenses
Year 2: $6.2M revenue, $3.1M expenses  
Year 3: $12.8M revenue, $5.2M expenses

Break-even expected in Month 18.

## Team Requirements

- 5 Software Engineers
- 2 Data Scientists
- 2 Product Managers
- 1 UI/UX Designer
- 3 Sales Representatives

## Implementation Timeline

Phase 1 (Months 1-6): MVP development
Phase 2 (Months 7-12): Beta testing and refinement
Phase 3 (Months 13-18): Market launch and scaling

## Conclusion

This AI-powered platform represents a significant opportunity to capture market share in the rapidly growing customer analytics space. We request $5M in Series A funding to bring this vision to reality.
"""
    
    # Sample legal contract
    contract = """
# SOFTWARE DEVELOPMENT AGREEMENT

This Software Development Agreement ("Agreement") is entered into on January 15, 2024, between TechCorp Inc., a Delaware corporation ("Client"), and DevSolutions LLC, a California limited liability company ("Developer").

## 1. SCOPE OF WORK

Developer agrees to develop a custom customer relationship management (CRM) software application according to the specifications outlined in Exhibit A, which is incorporated herein by reference.

### 1.1 Deliverables
- Web-based CRM application
- Mobile companion app (iOS and Android)
- API documentation
- User training materials
- 90 days of post-launch support

### 1.2 Timeline
Development shall commence on February 1, 2024, with final delivery expected by August 31, 2024.

## 2. COMPENSATION

Client agrees to pay Developer a total fee of $485,000 according to the following schedule:
- 25% ($121,250) upon signing this Agreement
- 25% ($121,250) upon completion of design phase
- 25% ($121,250) upon completion of development phase  
- 25% ($121,250) upon final delivery and acceptance

## 3. INTELLECTUAL PROPERTY

All software code, documentation, and related materials developed under this Agreement shall be the exclusive property of Client. Developer retains no rights to the delivered work product.

## 4. CONFIDENTIALITY

Both parties acknowledge that they may have access to confidential information and agree to maintain strict confidentiality for a period of 5 years following termination of this Agreement.

## 5. WARRANTIES AND LIABILITY

Developer warrants that the delivered software will be free from material defects for 90 days. Developer's total liability shall not exceed the total compensation paid under this Agreement.

## 6. TERMINATION

Either party may terminate this Agreement with 30 days written notice. Upon termination, Client shall pay for all work completed to date.

## 7. GOVERNING LAW

This Agreement shall be governed by the laws of the State of California.

IN WITNESS WHEREOF, the parties have executed this Agreement on the date first written above.

TECHCORP INC.                    DEVSOLUTIONS LLC

_____________________          _____________________
John Smith, CEO                 Sarah Johnson, Managing Director
Date: _______________          Date: _______________
"""
    
    # Save sample documents
    with open("business_proposal.txt", "w") as f:
        f.write(proposal)
    
    with open("software_contract.txt", "w") as f:
        f.write(contract)
    
    return ["business_proposal.txt", "software_contract.txt"]

def demonstrate_document_analysis():
    """Demonstrate the document analysis system"""
    
    print("Document Analysis System Demonstration")
    print("=" * 50)
    
    # Create sample documents
    sample_docs = create_sample_documents()
    
    # Initialize analyzer
    analyzer = DocumentAnalyzer()
    
    # Add documents
    doc_ids = []
    for doc_path in sample_docs:
        print(f"\n📄 Adding document: {doc_path}")
        doc_id = analyzer.add_document(doc_path)
        doc_ids.append(doc_id)
        print(f"   Document ID: {doc_id}")
    
    # Show document list
    print(f"\n📋 Processed Documents:")
    docs = analyzer.get_document_list()
    for doc in docs:
        print(f"   {doc['filename']}: {doc['word_count']} words, {doc['chunks']} chunks")
    
    # Analyze each document
    for doc_id in doc_ids:
        print(f"\n🔍 Analyzing document: {doc_id}")
        analysis = analyzer.analyze_document(doc_id)
        
        print(f"   Summary: {analysis.summary[:200]}...")
        print(f"   Key Points:")
        for point in analysis.key_points[:3]:
            print(f"     • {point}")
        print(f"   Topics: {', '.join(analysis.topics[:5])}")
        print(f"   Sentiment: {analysis.sentiment}")
        print(f"   Confidence: {analysis.confidence_score:.2%}")
        
        if analysis.entities:
            print(f"   Entities found: {len(analysis.entities)}")
            for entity in analysis.entities[:3]:
                print(f"     • {entity.get('text', '')} ({entity.get('category', '')})")
    
    # Demonstrate Q&A
    questions = [
        "What is the total funding amount requested?",
        "What are the payment terms in the contract?",
        "What is the projected revenue for Year 2?",
        "Who are the parties in the software development agreement?"
    ]
    
    print(f"\n❓ Question & Answer Demonstration:")
    for question in questions:
        print(f"\nQ: {question}")
        result = analyzer.ask_question(question)
        print(f"A: {result['answer']}")
        if result['sources']:
            print(f"   Source: {result['sources'][0]['filename']}")
    
    # Document comparison
    if len(doc_ids) >= 2:
        print(f"\n🔄 Comparing documents...")
        comparison = analyzer.compare_documents(doc_ids[0], doc_ids[1])
        print(f"Comparison between {comparison['doc1']['filename']} and {comparison['doc2']['filename']}:")
        print(f"{comparison['comparison'][:400]}...")

def main():
    """Run document analysis demonstration"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        demonstrate_document_analysis()
        print("\n✅ Document analysis demonstration completed!")
        
    except Exception as e:
        print(f"❌ Error in document analysis demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

### 11.3 Automated Content Generation System

A comprehensive content creation system for blogs, marketing materials, and social media content.

```python
# ai/langchain/examples/11_content_generator.py

import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import random

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field

class ContentType(Enum):
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL_NEWSLETTER = "email_newsletter"
    PRODUCT_DESCRIPTION = "product_description"
    PRESS_RELEASE = "press_release"
    AD_COPY = "ad_copy"

class Platform(Enum):
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"

@dataclass
class ContentBrief:
    topic: str
    content_type: ContentType
    target_audience: str
    tone: str  # professional, casual, humorous, technical, etc.
    keywords: List[str]
    word_count: int
    platform: Optional[Platform] = None
    call_to_action: Optional[str] = None
    deadline: Optional[datetime] = None

class GeneratedContent(BaseModel):
    title: str = Field(description="Content title/headline")
    content: str = Field(description="Main content body")
    summary: str = Field(description="Brief summary/excerpt")
    hashtags: List[str] = Field(description="Relevant hashtags")
    seo_keywords: List[str] = Field(description="SEO-optimized keywords")
    estimated_reading_time: int = Field(description="Reading time in minutes")

class SocialMediaPost(BaseModel):
    content: str = Field(description="Post content")
    hashtags: List[str] = Field(description="Relevant hashtags")
    best_posting_time: str = Field(description="Optimal posting time")
    engagement_prediction: str = Field(description="Expected engagement level")

class EmailNewsletter(BaseModel):
    subject_line: str = Field(description="Email subject line")
    preview_text: str = Field(description="Email preview text")
    content_sections: List[Dict[str, str]] = Field(description="Newsletter sections")
    call_to_action: str = Field(description="Main CTA")

class ContentOptimizer:
    """Optimizes content for specific platforms and purposes"""
    
    def __init__(self):
        self.platform_limits = {
            Platform.TWITTER: {"char_limit": 280, "hashtag_limit": 3},
            Platform.LINKEDIN: {"char_limit": 1300, "hashtag_limit": 5},
            Platform.FACEBOOK: {"char_limit": 500, "hashtag_limit": 5},
            Platform.INSTAGRAM: {"char_limit": 2200, "hashtag_limit": 10},
        }
        
        self.tone_guidelines = {
            "professional": "formal, authoritative, industry-specific terminology",
            "casual": "conversational, friendly, accessible language",
            "humorous": "light-hearted, witty, engaging",
            "technical": "precise, detailed, expert-level",
            "inspirational": "motivating, uplifting, action-oriented"
        }
    
    def optimize_for_platform(self, content: str, platform: Platform) -> str:
        """Optimize content for specific social media platform"""
        limits = self.platform_limits.get(platform, {})
        char_limit = limits.get("char_limit", 1000)
        
        if len(content) <= char_limit:
            return content
        
        # Truncate intelligently at sentence boundaries
        sentences = content.split('. ')
        optimized = ""
        
        for sentence in sentences:
            if len(optimized + sentence + ". ") <= char_limit - 10:  # Leave room for ellipsis
                optimized += sentence + ". "
            else:
                optimized += "..."
                break
        
        return optimized.strip()
    
    def generate_hashtags(self, content: str, keywords: List[str], platform: Platform) -> List[str]:
        """Generate platform-optimized hashtags"""
        hashtag_limit = self.platform_limits.get(platform, {}).get("hashtag_limit", 5)
        
        # Start with keywords
        hashtags = [f"#{keyword.replace(' ', '').lower()}" for keyword in keywords]
        
        # Add common industry hashtags based on content analysis
        content_lower = content.lower()
        
        if "ai" in content_lower or "artificial intelligence" in content_lower:
            hashtags.extend(["#AI", "#ArtificialIntelligence", "#MachineLearning"])
        if "business" in content_lower:
            hashtags.extend(["#Business", "#Entrepreneur", "#Growth"])
        if "tech" in content_lower or "technology" in content_lower:
            hashtags.extend(["#Technology", "#Innovation", "#Digital"])
        
        # Remove duplicates and limit
        hashtags = list(set(hashtags))[:hashtag_limit]
        
        return hashtags

class ContentGenerator:
    """Advanced content generation system with optimization"""
    
    def __init__(self, model: str = "gpt-3.5-turbo-16k"):
        self.llm = ChatOpenAI(temperature=0.7, model=model)
        self.optimizer = ContentOptimizer()
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup specialized content generation chains"""
        
        # Blog post chain
        blog_prompt = ChatPromptTemplate.from_template("""
You are an expert content writer. Create a comprehensive blog post based on the following brief:

Topic: {topic}
Target Audience: {target_audience}
Tone: {tone}
Keywords to include: {keywords}
Word count target: {word_count}

Write a blog post that includes:
1. Engaging title
2. Introduction that hooks the reader
3. Well-structured body with subheadings
4. Practical insights and examples
5. Strong conclusion with call-to-action

Blog Post:""")
        
        self.blog_chain = LLMChain(llm=self.llm, prompt=blog_prompt, output_key="blog_content")
        
        # Social media chain
        social_prompt = ChatPromptTemplate.from_template("""
Create engaging social media content for {platform}:

Topic: {topic}
Target Audience: {target_audience}
Tone: {tone}
Character limit: {char_limit}
Keywords: {keywords}

Requirements:
- Engaging and shareable
- Include relevant hashtags
- Strong hook in first line
- Clear value proposition
- Appropriate for {platform} audience

Social Media Post:""")
        
        self.social_chain = LLMChain(llm=self.llm, prompt=social_prompt, output_key="social_content")
        
        # Email newsletter chain
        email_prompt = ChatPromptTemplate.from_template("""
Create an email newsletter based on:

Topic: {topic}
Target Audience: {target_audience}
Tone: {tone}
Keywords: {keywords}

Include:
1. Compelling subject line
2. Preview text
3. Newsletter sections with headers
4. Call-to-action
5. Personal touch

Email Newsletter:""")
        
        self.email_chain = LLMChain(llm=self.llm, prompt=email_prompt, output_key="email_content")
        
        # SEO optimization chain
        seo_prompt = ChatPromptTemplate.from_template("""
Optimize the following content for SEO:

Content: {content}
Primary Keywords: {keywords}

Provide:
1. SEO-optimized title variations
2. Meta description
3. Additional keywords to target
4. Content improvements for better ranking

SEO Optimization:""")
        
        self.seo_chain = LLMChain(llm=self.llm, prompt=seo_prompt, output_key="seo_optimization")
    
    def generate_blog_post(self, brief: ContentBrief) -> GeneratedContent:
        """Generate comprehensive blog post"""
        
        # Generate main content
        result = self.blog_chain.invoke({
            "topic": brief.topic,
            "target_audience": brief.target_audience,
            "tone": brief.tone,
            "keywords": ", ".join(brief.keywords),
            "word_count": brief.word_count
        })
        
        content = result["blog_content"]
        
        # Extract title (assume first line is title)
        lines = content.split('\n')
        title = lines[0].strip('#').strip()
        body_content = '\n'.join(lines[1:]).strip()
        
        # Generate SEO optimization
        seo_result = self.seo_chain.invoke({
            "content": content,
            "keywords": ", ".join(brief.keywords)
        })
        
        # Create summary (first paragraph)
        paragraphs = body_content.split('\n\n')
        summary = paragraphs[0][:200] + "..." if len(paragraphs[0]) > 200 else paragraphs[0]
        
        # Generate hashtags
        hashtags = self.optimizer.generate_hashtags(content, brief.keywords, Platform.LINKEDIN)
        
        # Estimate reading time (average 200 words per minute)
        word_count = len(content.split())
        reading_time = max(1, word_count // 200)
        
        return GeneratedContent(
            title=title,
            content=body_content,
            summary=summary,
            hashtags=hashtags,
            seo_keywords=brief.keywords,
            estimated_reading_time=reading_time
        )
    
    def generate_social_media_post(self, brief: ContentBrief) -> SocialMediaPost:
        """Generate optimized social media post"""
        
        if not brief.platform:
            brief.platform = Platform.LINKEDIN
        
        char_limit = self.optimizer.platform_limits.get(brief.platform, {}).get("char_limit", 500)
        
        result = self.social_chain.invoke({
            "topic": brief.topic,
            "target_audience": brief.target_audience,
            "tone": brief.tone,
            "platform": brief.platform.value,
            "char_limit": char_limit,
            "keywords": ", ".join(brief.keywords)
        })
        
        content = result["social_content"]
        
        # Optimize for platform
        optimized_content = self.optimizer.optimize_for_platform(content, brief.platform)
        
        # Generate hashtags
        hashtags = self.optimizer.generate_hashtags(content, brief.keywords, brief.platform)
        
        # Determine best posting time (simplified)
        posting_times = {
            Platform.TWITTER: "1-3 PM weekdays",
            Platform.LINKEDIN: "7-9 AM weekdays", 
            Platform.FACEBOOK: "9 AM - 3 PM weekdays",
            Platform.INSTAGRAM: "6-9 AM weekdays"
        }
        
        best_time = posting_times.get(brief.platform, "Business hours")
        
        # Predict engagement (simplified heuristic)
        engagement_score = len([word for word in optimized_content.lower().split() 
                               if word in ["question", "tip", "secret", "free", "new", "amazing"]])
        engagement_levels = ["Low", "Medium", "High", "Very High"]
        engagement_prediction = engagement_levels[min(engagement_score, 3)]
        
        return SocialMediaPost(
            content=optimized_content,
            hashtags=hashtags,
            best_posting_time=best_time,
            engagement_prediction=engagement_prediction
        )
    
    def generate_email_newsletter(self, brief: ContentBrief) -> EmailNewsletter:
        """Generate email newsletter"""
        
        result = self.email_chain.invoke({
            "topic": brief.topic,
            "target_audience": brief.target_audience,
            "tone": brief.tone,
            "keywords": ", ".join(brief.keywords)
        })
        
        content = result["email_content"]
        
        # Parse the generated content (simplified parsing)
        lines = content.split('\n')
        
        subject_line = "Newsletter Update"
        preview_text = ""
        sections = []
        call_to_action = brief.call_to_action or "Learn More"
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("Subject:"):
                subject_line = line.replace("Subject:", "").strip()
            elif line.startswith("Preview:"):
                preview_text = line.replace("Preview:", "").strip()
            elif line.startswith("#") and len(line) > 1:
                # Section header
                header = line.strip("#").strip()
                # Look for content in next few lines
                section_content = ""
                for j in range(i+1, min(i+5, len(lines))):
                    if not lines[j].startswith("#") and lines[j].strip():
                        section_content += lines[j].strip() + " "
                
                if section_content:
                    sections.append({"header": header, "content": section_content.strip()})
        
        # Ensure we have at least some sections
        if not sections:
            sections = [
                {"header": "Main Content", "content": content[:500]},
                {"header": "Key Takeaways", "content": "Important insights from this newsletter."}
            ]
        
        return EmailNewsletter(
            subject_line=subject_line,
            preview_text=preview_text or content[:100],
            content_sections=sections,
            call_to_action=call_to_action
        )
    
    def generate_content_series(self, main_topic: str, num_pieces: int = 5) -> List[GeneratedContent]:
        """Generate a series of related content pieces"""
        
        # Generate subtopics
        subtopic_prompt = ChatPromptTemplate.from_template("""
        Break down the topic "{main_topic}" into {num_pieces} related subtopics that would make great individual content pieces.
        
        Each subtopic should be:
        - Specific and focused
        - Valuable to the audience
        - SEO-friendly
        
        Return as a numbered list:""")
        
        subtopics_result = (subtopic_prompt | self.llm).invoke({
            "main_topic": main_topic,
            "num_pieces": num_pieces
        })
        
        # Parse subtopics
        subtopics = []
        for line in subtopics_result.content.split('\n'):
            if line.strip() and (line[0].isdigit() or line.startswith('-')):
                subtopic = line.strip()
                # Remove numbering
                for i, char in enumerate(subtopic):
                    if char.isalpha():
                        subtopic = subtopic[i:].strip()
                        break
                subtopics.append(subtopic)
        
        # Generate content for each subtopic
        content_series = []
        for subtopic in subtopics[:num_pieces]:
            brief = ContentBrief(
                topic=subtopic,
                content_type=ContentType.BLOG_POST,
                target_audience="professionals interested in " + main_topic,
                tone="professional",
                keywords=[main_topic.lower(), subtopic.lower()],
                word_count=800
            )
            
            content = self.generate_blog_post(brief)
            content_series.append(content)
        
        return content_series
    
    def batch_generate_content(self, briefs: List[ContentBrief]) -> List[Dict[str, Any]]:
        """Generate multiple content pieces efficiently"""
        
        results = []
        
        for i, brief in enumerate(briefs):
            print(f"Generating content {i+1}/{len(briefs)}: {brief.topic}")
            
            try:
                if brief.content_type == ContentType.BLOG_POST:
                    content = self.generate_blog_post(brief)
                    result = {
                        "type": "blog_post",
                        "brief": brief,
                        "content": content,
                        "success": True
                    }
                
                elif brief.content_type == ContentType.SOCIAL_MEDIA:
                    content = self.generate_social_media_post(brief)
                    result = {
                        "type": "social_media",
                        "brief": brief,
                        "content": content,
                        "success": True
                    }
                
                elif brief.content_type == ContentType.EMAIL_NEWSLETTER:
                    content = self.generate_email_newsletter(brief)
                    result = {
                        "type": "email_newsletter",
                        "brief": brief,
                        "content": content,
                        "success": True
                    }
                
                else:
                    result = {
                        "type": brief.content_type.value,
                        "brief": brief,
                        "content": None,
                        "success": False,
                        "error": "Content type not implemented"
                    }
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "type": brief.content_type.value,
                    "brief": brief,
                    "content": None,
                    "success": False,
                    "error": str(e)
                })
        
        return results

def demonstrate_content_generation():
    """Demonstrate the content generation system"""
    
    print("Automated Content Generation System")
    print("=" * 50)
    
    generator = ContentGenerator()
    
    # Demo 1: Blog Post Generation
    print("\n📝 Generating Blog Post...")
    blog_brief = ContentBrief(
        topic="The Future of Artificial Intelligence in Business",
        content_type=ContentType.BLOG_POST,
        target_audience="business executives and entrepreneurs", 
        tone="professional",
        keywords=["artificial intelligence", "business transformation", "automation", "AI strategy"],
        word_count=1200,
        call_to_action="Contact us to learn how AI can transform your business"
    )
    
    blog_post = generator.generate_blog_post(blog_brief)
    print(f"Title: {blog_post.title}")
    print(f"Content: {blog_post.content[:300]}...")
    print(f"Summary: {blog_post.summary}")
    print(f"Reading Time: {blog_post.estimated_reading_time} minutes")
    print(f"SEO Keywords: {', '.join(blog_post.seo_keywords)}")
    print(f"Hashtags: {' '.join(blog_post.hashtags)}")
    
    # Demo 2: Social Media Posts
    print("\n📱 Generating Social Media Posts...")
    
    platforms = [Platform.TWITTER, Platform.LINKEDIN, Platform.INSTAGRAM]
    
    for platform in platforms:
        social_brief = ContentBrief(
            topic="5 AI Tools Every Business Should Know About",
            content_type=ContentType.SOCIAL_MEDIA,
            target_audience="small business owners",
            tone="casual",
            keywords=["AI tools", "business productivity", "automation"],
            word_count=0,  # Not applicable for social media
            platform=platform
        )
        
        social_post = generator.generate_social_media_post(social_brief)
        print(f"\n{platform.value.upper()} Post:")
        print(f"Content: {social_post.content}")
        print(f"Hashtags: {' '.join(social_post.hashtags)}")
        print(f"Best Time: {social_post.best_posting_time}")
        print(f"Engagement Prediction: {social_post.engagement_prediction}")
    
    # Demo 3: Email Newsletter
    print("\n📧 Generating Email Newsletter...")
    email_brief = ContentBrief(
        topic="Weekly AI & Business Update",
        content_type=ContentType.EMAIL_NEWSLETTER,
        target_audience="business professionals interested in AI",
        tone="professional",
        keywords=["AI news", "business insights", "technology trends"],
        word_count=800,
        call_to_action="Read Full Article"
    )
    
    newsletter = generator.generate_email_newsletter(email_brief)
    print(f"Subject: {newsletter.subject_line}")
    print(f"Preview: {newsletter.preview_text}")
    print(f"Sections: {len(newsletter.content_sections)}")
    for i, section in enumerate(newsletter.content_sections[:2]):
        print(f"  Section {i+1}: {section['header']}")
        print(f"  Content: {section['content'][:100]}...")
    print(f"CTA: {newsletter.call_to_action}")
    
    # Demo 4: Content Series Generation
    print("\n📚 Generating Content Series...")
    series = generator.generate_content_series("Machine Learning for Beginners", 3)
    print(f"Generated {len(series)} pieces in the series:")
    for i, content in enumerate(series, 1):
        print(f"  {i}. {content.title}")
        print(f"     {content.summary[:100]}...")
    
    # Demo 5: Batch Content Generation
    print("\n🔄 Batch Content Generation...")
    batch_briefs = [
        ContentBrief(
            topic=f"AI Trend #{i+1}: {trend}",
            content_type=ContentType.BLOG_POST,
            target_audience="tech enthusiasts",
            tone="casual",
            keywords=["AI", "technology", "trends"],
            word_count=600
        )
        for i, trend in enumerate(["Natural Language Processing", "Computer Vision", "Robotics"])
    ]
    
    batch_results = generator.batch_generate_content(batch_briefs)
    successful_generations = [r for r in batch_results if r["success"]]
    print(f"Successfully generated {len(successful_generations)}/{len(batch_results)} content pieces")
    
    for result in successful_generations:
        content = result["content"]
        print(f"  ✅ {content.title}")

def main():
    """Run content generation demonstration"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        demonstrate_content_generation()
        print("\n✅ Content generation demonstration completed!")
        
    except Exception as e:
        print(f"❌ Error in content generation demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

---

## Chapter 12: Testing Strategies

### 12.1 Comprehensive Testing Framework

```python
# ai/langchain/examples/11_testing.py

import pytest
import os
from unittest.mock import Mock, patch
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class TestLangChainComponents:
    """Comprehensive test suite for LangChain applications"""
    
    def setup_method(self):
        """Set up test environment"""
        self.test_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.test_prompt = PromptTemplate(
            template="Test prompt: {input}",
            input_variables=["input"]
        )
    
    def test_chain_creation(self):
        """Test basic chain creation"""
        chain = LLMChain(llm=self.test_llm, prompt=self.test_prompt)
        assert chain is not None
        assert chain.llm == self.test_llm
        assert chain.prompt == self.test_prompt
    
    @patch('openai.ChatCompletion.create')
    def test_mocked_response(self, mock_openai):
        """Test with mocked OpenAI response"""
        mock_response = {
            'choices': [{'message': {'content': 'Mocked response'}}]
        }
        mock_openai.return_value = mock_response
        
        chain = LLMChain(llm=self.test_llm, prompt=self.test_prompt)
        result = chain.run(input="test input")
        
        assert "Mocked" in result

def main():
    """Run tests"""
    print("Running LangChain Tests")
    print("=" * 30)
    
    # Run pytest programmatically
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    main()
```

## Chapter 13: Production Deployment

### 12.1 Deployment Best Practices

```python
# ai/langchain/examples/12_production.py

from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import logging
import os
from typing import Dict, Any

class ProductionLangChainApp:
    """Production-ready LangChain application"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        self.setup_llm()
    
    def setup_logging(self):
        """Configure production logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('langchain_app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_llm(self):
        """Initialize LLM with production settings"""
        self.llm = ChatOpenAI(
            temperature=self.config.get('temperature', 0.1),
            model=self.config.get('model', 'gpt-3.5-turbo'),
            max_retries=self.config.get('max_retries', 3),
            request_timeout=self.config.get('timeout', 30)
        )
    
    def process_request(self, user_input: str) -> Dict[str, Any]:
        """Process user request with monitoring"""
        self.logger.info(f"Processing request: {user_input[:50]}...")
        
        try:
            with get_openai_callback() as cb:
                response = self.llm.predict(user_input)
                
                return {
                    'response': response,
                    'tokens_used': cb.total_tokens,
                    'cost': cb.total_cost,
                    'success': True
                }
        
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return {
                'response': 'I apologize, but I encountered an error processing your request.',
                'error': str(e),
                'success': False
            }

def main():
    """Run production app demo"""
    print("Production LangChain Application Demo")
    print("=" * 40)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    config = {
        'temperature': 0.1,
        'model': 'gpt-3.5-turbo',
        'max_retries': 3,
        'timeout': 30
    }
    
    app = ProductionLangChainApp(config)
    
    # Test request
    result = app.process_request("What are best practices for Python development?")
    
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Response: {result['response'][:200]}...")
        print(f"Tokens used: {result['tokens_used']}")
        print(f"Cost: ${result['cost']:.4f}")

if __name__ == "__main__":
    main()
```

## Chapter 14: Debugging and Troubleshooting

### 13.1 Common Issues and Solutions

```python
# ai/langchain/examples/13_debugging.py

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import StdOutCallbackHandler
import logging
import os

class DebuggingHelper:
    """Tools and techniques for debugging LangChain applications"""
    
    def __init__(self):
        self.setup_logging()
        self.llm = ChatOpenAI(temperature=0.1, verbose=True)
    
    def setup_logging(self):
        """Configure debug logging"""
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
    
    def debug_prompt_issues(self):
        """Debug common prompt formatting issues"""
        print("=== Debugging Prompt Issues ===")
        
        # Common issue: Wrong variable names
        try:
            bad_prompt = PromptTemplate(
                template="Tell me about {topic}",
                input_variables=["subject"]  # Mismatch!
            )
            chain = LLMChain(llm=self.llm, prompt=bad_prompt)
            result = chain.run(topic="Python")  # This will fail
        except Exception as e:
            print(f"Error caught: {e}")
            print("Solution: Ensure template variables match input_variables")
        
        # Correct version
        good_prompt = PromptTemplate(
            template="Tell me about {topic}",
            input_variables=["topic"]
        )
        chain = LLMChain(llm=self.llm, prompt=good_prompt)
        result = chain.run(topic="Python")
        print(f"Success: {result[:100]}...")
    
    def debug_api_issues(self):
        """Debug API-related issues"""
        print("\n=== Debugging API Issues ===")
        
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found")
            return
        
        if api_key.startswith("sk-"):
            print("✅ API key format looks correct")
        else:
            print("⚠️ API key format may be incorrect")
        
        # Test API connection
        try:
            test_llm = ChatOpenAI(temperature=0, max_tokens=10)
            response = test_llm.predict("Test")
            print("✅ API connection successful")
        except Exception as e:
            print(f"❌ API connection failed: {e}")

def main():
    """Run debugging demonstrations"""
    print("LangChain Debugging Guide")
    print("=" * 30)
    
    debugger = DebuggingHelper()
    debugger.debug_prompt_issues()
    debugger.debug_api_issues()

if __name__ == "__main__":
    main()
```

---

**🎉 Congratulations! You now have a comprehensive LangChain guide covering:**

- ✅ Installation and Setup
- ✅ Core Concepts and Architecture  
- ✅ LLM Integration
- ✅ Prompt Engineering
- ✅ Chains and Complex Workflows
- ✅ Memory Management
- ✅ Document Processing
- ✅ Vector Stores and Embeddings
- ✅ Agents and Tools
- ✅ Performance Optimization
- ✅ Testing Strategies
- ✅ Production Deployment
- ✅ Debugging and Troubleshooting

This guide provides a complete learning path from beginner to advanced LangChain development with practical examples, real-world patterns, and production-ready code.

## Chapter 15: LangSmith - Observability and Debugging

LangSmith is a powerful platform for debugging, testing, evaluating, and monitoring LangChain applications in production.

### 14.1 LangSmith Setup and Configuration

```python
# ai/langchain/examples/14_langsmith.py

import os
from langchain import LangSmith
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import LangSmithCallbackHandler
from langchain.schema import LLMResult
from typing import Dict, Any, List
import json
from datetime import datetime

class LangSmithIntegration:
    """Comprehensive LangSmith integration for LangChain applications"""
    
    def __init__(self):
        self.setup_langsmith()
        self.setup_langchain_components()
    
    def setup_langsmith(self):
        """Configure LangSmith environment"""
        # Set up LangSmith environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        
        # API key should be set in environment
        if not os.getenv("LANGCHAIN_API_KEY"):
            print("⚠️ LANGCHAIN_API_KEY not found. Set it to enable LangSmith tracing.")
            print("Get your API key from: https://smith.langchain.com/")
        
        # Optional: Set project name
        os.environ["LANGCHAIN_PROJECT"] = "comprehensive-langchain-demo"
        
        print("✅ LangSmith configuration completed")
    
    def setup_langchain_components(self):
        """Initialize LangChain components with LangSmith integration"""
        # Initialize LLM with LangSmith tracing
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo",
            callbacks=[LangSmithCallbackHandler()]
        )
        
        # Create a prompt template
        self.prompt_template = PromptTemplate(
            template="""You are a helpful AI assistant. 
            
Context: {context}
User Question: {question}

Please provide a clear and helpful answer:""",
            input_variables=["context", "question"]
        )
        
        # Create chain with LangSmith tracing
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            callbacks=[LangSmithCallbackHandler()],
            verbose=True
        )
        
        print("✅ LangChain components initialized with LangSmith tracing")

### 14.2 Advanced Tracing and Monitoring

```python
class LangSmithTracer:
    """Advanced tracing capabilities with LangSmith"""
    
    def __init__(self, project_name: str = "langchain-advanced-tracing"):
        self.project_name = project_name
        self.setup_tracing()
    
    def setup_tracing(self):
        """Configure advanced tracing"""
        os.environ["LANGCHAIN_PROJECT"] = self.project_name
        
        # Enable detailed tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        
        print(f"📊 Advanced tracing enabled for project: {self.project_name}")
    
    def trace_chain_execution(self, chain, inputs: Dict[str, Any], 
                            metadata: Dict[str, Any] = None):
        """Execute chain with detailed tracing"""
        
        # Add execution metadata
        execution_metadata = {
            "timestamp": datetime.now().isoformat(),
            "execution_id": f"exec_{int(datetime.now().timestamp())}",
            "input_tokens": len(str(inputs).split()),
            **(metadata or {})
        }
        
        print(f"🔍 Starting traced execution: {execution_metadata['execution_id']}")
        
        # Execute with tracing
        try:
            # Create callback with metadata
            callback = LangSmithCallbackHandler(
                project_name=self.project_name
            )
            
            # Run chain with callback
            result = chain.run(
                callbacks=[callback],
                **inputs
            )
            
            print(f"✅ Traced execution completed successfully")
            print(f"📊 Check LangSmith dashboard: https://smith.langchain.com/")
            
            return {
                "result": result,
                "metadata": execution_metadata,
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Traced execution failed: {e}")
            return {
                "error": str(e),
                "metadata": execution_metadata,
                "success": False
            }
    
    def batch_trace_execution(self, chain, batch_inputs: List[Dict[str, Any]]):
        """Execute multiple inputs with batch tracing"""
        
        print(f"📦 Starting batch execution of {len(batch_inputs)} inputs")
        
        batch_results = []
        
        for i, inputs in enumerate(batch_inputs):
            print(f"\n--- Batch Item {i+1}/{len(batch_inputs)} ---")
            
            result = self.trace_chain_execution(
                chain=chain,
                inputs=inputs,
                metadata={
                    "batch_index": i,
                    "batch_size": len(batch_inputs)
                }
            )
            
            batch_results.append(result)
        
        # Summary
        successful = sum(1 for r in batch_results if r["success"])
        print(f"\n📊 Batch Summary:")
        print(f"   Total: {len(batch_results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(batch_results) - successful}")
        
        return batch_results

### 14.3 Custom Evaluations and Metrics

```python
from langchain.evaluation import load_evaluator
from langchain.schema import Document

class LangSmithEvaluator:
    """Custom evaluation metrics for LangSmith"""
    
    def __init__(self):
        self.setup_evaluators()
    
    def setup_evaluators(self):
        """Initialize various evaluators"""
        self.evaluators = {
            "qa": load_evaluator("qa"),
            "criteria": load_evaluator("criteria", criteria="helpfulness"),
            "string_distance": load_evaluator("string_distance")
        }
        
        print("🎯 Evaluators initialized")
    
    def evaluate_qa_chain(self, chain, test_cases: List[Dict[str, str]]):
        """Evaluate Q&A chain performance"""
        
        print("📊 Evaluating Q&A Chain Performance")
        print("=" * 40)
        
        evaluation_results = []
        
        for i, test_case in enumerate(test_cases):
            question = test_case["question"]
            expected_answer = test_case.get("expected_answer", "")
            context = test_case.get("context", "")
            
            print(f"\nTest Case {i+1}: {question[:50]}...")
            
            # Get chain response
            try:
                actual_answer = chain.run(
                    question=question,
                    context=context
                )
                
                # Evaluate response
                if expected_answer:
                    # Compare with expected answer
                    string_eval = self.evaluators["string_distance"].evaluate_strings(
                        prediction=actual_answer,
                        reference=expected_answer
                    )
                    
                    # Criteria evaluation
                    criteria_eval = self.evaluators["criteria"].evaluate_strings(
                        prediction=actual_answer,
                        input=question
                    )
                    
                    result = {
                        "question": question,
                        "actual_answer": actual_answer,
                        "expected_answer": expected_answer,
                        "string_distance": string_eval["score"],
                        "helpfulness_score": criteria_eval["score"],
                        "success": True
                    }
                else:
                    # Just criteria evaluation
                    criteria_eval = self.evaluators["criteria"].evaluate_strings(
                        prediction=actual_answer,
                        input=question
                    )
                    
                    result = {
                        "question": question,
                        "actual_answer": actual_answer,
                        "helpfulness_score": criteria_eval["score"],
                        "success": True
                    }
                
                evaluation_results.append(result)
                
                print(f"✅ Helpfulness Score: {result['helpfulness_score']:.2f}")
                if 'string_distance' in result:
                    print(f"📏 String Distance: {result['string_distance']:.2f}")
                
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                evaluation_results.append({
                    "question": question,
                    "error": str(e),
                    "success": False
                })
        
        # Summary statistics
        successful_evals = [r for r in evaluation_results if r["success"]]
        
        if successful_evals:
            avg_helpfulness = sum(r["helpfulness_score"] for r in successful_evals) / len(successful_evals)
            
            print(f"\n📈 Evaluation Summary:")
            print(f"   Total Test Cases: {len(test_cases)}")
            print(f"   Successful: {len(successful_evals)}")
            print(f"   Average Helpfulness: {avg_helpfulness:.2f}")
            
            if any('string_distance' in r for r in successful_evals):
                distances = [r["string_distance"] for r in successful_evals if "string_distance" in r]
                avg_distance = sum(distances) / len(distances)
                print(f"   Average String Distance: {avg_distance:.2f}")
        
        return evaluation_results

### 14.4 Dataset Management and Testing

```python
class LangSmithDatasetManager:
    """Manage datasets and testing with LangSmith"""
    
    def __init__(self, project_name: str = "langchain-datasets"):
        self.project_name = project_name
        self.test_datasets = {}
    
    def create_test_dataset(self, name: str, test_cases: List[Dict[str, Any]]):
        """Create a test dataset for evaluation"""
        
        dataset = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "test_cases": test_cases,
            "metadata": {
                "size": len(test_cases),
                "project": self.project_name
            }
        }
        
        self.test_datasets[name] = dataset
        
        print(f"📁 Created dataset '{name}' with {len(test_cases)} test cases")
        return dataset
    
    def run_dataset_evaluation(self, dataset_name: str, chain):
        """Run evaluation on a specific dataset"""
        
        if dataset_name not in self.test_datasets:
            print(f"❌ Dataset '{dataset_name}' not found")
            return None
        
        dataset = self.test_datasets[dataset_name]
        test_cases = dataset["test_cases"]
        
        print(f"🧪 Running evaluation on dataset: {dataset_name}")
        print(f"📊 Test cases: {len(test_cases)}")
        
        # Initialize evaluator
        evaluator = LangSmithEvaluator()
        
        # Run evaluation
        results = evaluator.evaluate_qa_chain(chain, test_cases)
        
        # Store results in dataset
        dataset["last_evaluation"] = {
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        return results
    
    def create_sample_datasets(self):
        """Create sample datasets for demonstration"""
        
        # Customer service dataset
        customer_service_cases = [
            {
                "question": "How do I reset my password?",
                "context": "User account management system",
                "expected_answer": "Go to login page, click 'Forgot Password', enter your email"
            },
            {
                "question": "What are your business hours?",
                "context": "Customer service information",
                "expected_answer": "Monday-Friday 9AM-6PM EST"
            },
            {
                "question": "How do I cancel my subscription?",
                "context": "Subscription management",
                "expected_answer": "Go to Account Settings, click Subscription, then Cancel"
            }
        ]
        
        # Technical support dataset
        tech_support_cases = [
            {
                "question": "My API calls are failing with 401 error",
                "context": "API authentication troubleshooting",
                "expected_answer": "Check your API key validity and authentication headers"
            },
            {
                "question": "How do I increase my rate limits?",
                "context": "API rate limiting",
                "expected_answer": "Contact support or upgrade your plan for higher limits"
            }
        ]
        
        # Create datasets
        self.create_test_dataset("customer_service", customer_service_cases)
        self.create_test_dataset("tech_support", tech_support_cases)
        
        print("✅ Sample datasets created")

def main():
    """Run comprehensive LangSmith demonstration"""
    print("LangSmith Integration Comprehensive Demo")
    print("=" * 45)
    
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["LANGCHAIN_API_KEY"]
    
    for var in required_vars:
        if not os.getenv(var):
            print(f"❌ {var} is required but not found")
            return
    
    for var in optional_vars:
        if not os.getenv(var):
            print(f"⚠️ {var} not found - some features may be limited")
    
    try:
        # 1. Basic LangSmith integration
        print("\n🔧 Setting up LangSmith integration...")
        langsmith_integration = LangSmithIntegration()
        
        # 2. Test basic chain execution with tracing
        print("\n📊 Testing traced chain execution...")
        tracer = LangSmithTracer("demo-tracing")
        
        test_inputs = {
            "context": "LangChain is a framework for building LLM applications",
            "question": "What is LangChain used for?"
        }
        
        result = tracer.trace_chain_execution(
            chain=langsmith_integration.chain,
            inputs=test_inputs,
            metadata={"demo": "basic_tracing"}
        )
        
        if result["success"]:
            print(f"Response: {result['result'][:200]}...")
        
        # 3. Batch execution demo
        print("\n📦 Testing batch execution...")
        batch_inputs = [
            {
                "context": "Python programming language",
                "question": "What are Python's main advantages?"
            },
            {
                "context": "Machine learning with Python", 
                "question": "Which Python libraries are used for ML?"
            },
            {
                "context": "Web development with Python",
                "question": "What frameworks are popular for Python web development?"
            }
        ]
        
        batch_results = tracer.batch_trace_execution(
            chain=langsmith_integration.chain,
            inputs=batch_inputs
        )
        
        # 4. Dataset evaluation demo
        print("\n🧪 Testing dataset evaluation...")
        dataset_manager = LangSmithDatasetManager()
        dataset_manager.create_sample_datasets()
        
        # Run evaluation on customer service dataset
        evaluation_results = dataset_manager.run_dataset_evaluation(
            "customer_service",
            langsmith_integration.chain
        )
        
        print("\n✅ LangSmith integration demonstration completed!")
        print("🔗 Check your LangSmith dashboard at: https://smith.langchain.com/")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

### 14.5 Production Monitoring with LangSmith

```python
# ai/langchain/examples/14_langsmith_production.py

import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import LangSmithCallbackHandler
from langchain.schema import BaseMessage
import json

class ProductionLangSmithMonitor:
    """Production monitoring and alerting with LangSmith"""
    
    def __init__(self, project_name: str = "production-monitoring"):
        self.project_name = project_name
        self.setup_monitoring()
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency": 0.0,
            "error_rate": 0.0
        }
    
    def setup_monitoring(self):
        """Configure production monitoring"""
        os.environ["LANGCHAIN_PROJECT"] = self.project_name
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        
        # Setup monitoring-specific configurations
        os.environ["LANGCHAIN_TAGS"] = "production,monitoring"
        
        print(f"🔍 Production monitoring enabled for: {self.project_name}")
    
    def monitor_chain_execution(self, chain, inputs: Dict[str, Any], 
                              user_id: Optional[str] = None,
                              session_id: Optional[str] = None) -> Dict[str, Any]:
        """Monitor single chain execution with detailed metrics"""
        
        start_time = time.time()
        execution_id = f"exec_{int(start_time * 1000)}"
        
        # Prepare metadata
        metadata = {
            "execution_id": execution_id,
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "environment": "production"
        }
        
        self.metrics["total_requests"] += 1
        
        try:
            # Create callback with production metadata
            callback = LangSmithCallbackHandler(
                project_name=self.project_name,
                tags=["production", "monitored"]
            )
            
            # Execute chain
            result = chain.run(
                callbacks=[callback],
                **inputs
            )
            
            # Calculate metrics
            end_time = time.time()
            latency = end_time - start_time
            
            # Update success metrics
            self.metrics["successful_requests"] += 1
            self._update_latency_metric(latency)
            
            execution_result = {
                "result": result,
                "metadata": metadata,
                "performance": {
                    "latency_seconds": latency,
                    "success": True
                },
                "success": True
            }
            
            # Log for monitoring
            self._log_execution(execution_result)
            
            return execution_result
            
        except Exception as e:
            # Handle errors
            end_time = time.time()
            latency = end_time - start_time
            
            self.metrics["failed_requests"] += 1
            self._update_error_rate()
            
            error_result = {
                "error": str(e),
                "error_type": type(e).__name__,
                "metadata": metadata,
                "performance": {
                    "latency_seconds": latency,
                    "success": False
                },
                "success": False
            }
            
            # Log error for monitoring
            self._log_execution(error_result)
            
            return error_result
    
    def _update_latency_metric(self, new_latency: float):
        """Update average latency metric"""
        successful_requests = self.metrics["successful_requests"]
        current_avg = self.metrics["average_latency"]
        
        # Calculate new average
        self.metrics["average_latency"] = (
            (current_avg * (successful_requests - 1) + new_latency) / successful_requests
        )
    
    def _update_error_rate(self):
        """Update error rate metric"""
        total = self.metrics["total_requests"]
        failed = self.metrics["failed_requests"]
        self.metrics["error_rate"] = (failed / total) * 100 if total > 0 else 0
    
    def _log_execution(self, execution_result: Dict[str, Any]):
        """Log execution for monitoring (in production, send to logging system)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if execution_result["success"]:
            print(f"[{timestamp}] ✅ SUCCESS - Latency: {execution_result['performance']['latency_seconds']:.3f}s")
        else:
            print(f"[{timestamp}] ❌ ERROR - {execution_result['error_type']}: {execution_result['error']}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        return {
            **self.metrics,
            "last_updated": datetime.now().isoformat(),
            "uptime_minutes": time.time() / 60  # Simplified uptime
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        metrics = self.get_metrics_summary()
        
        # Define health thresholds
        error_rate_threshold = 10.0  # 10% error rate
        latency_threshold = 5.0      # 5 seconds average latency
        
        health_status = {
            "healthy": True,
            "issues": [],
            "metrics": metrics
        }
        
        # Check error rate
        if metrics["error_rate"] > error_rate_threshold:
            health_status["healthy"] = False
            health_status["issues"].append(f"High error rate: {metrics['error_rate']:.1f}%")
        
        # Check latency
        if metrics["average_latency"] > latency_threshold:
            health_status["healthy"] = False
            health_status["issues"].append(f"High latency: {metrics['average_latency']:.2f}s")
        
        return health_status

class LangSmithAlertManager:
    """Alert management for production monitoring"""
    
    def __init__(self, monitor: ProductionLangSmithMonitor):
        self.monitor = monitor
        self.alert_thresholds = {
            "error_rate": 15.0,    # 15% error rate
            "latency": 10.0,       # 10 seconds
            "failed_requests": 50   # 50 failed requests
        }
        self.last_alert_times = {}
        self.alert_cooldown = 300  # 5 minutes between same alert types
    
    def check_and_send_alerts(self):
        """Check metrics and send alerts if needed"""
        metrics = self.monitor.get_metrics_summary()
        current_time = time.time()
        
        alerts_sent = []
        
        # Check error rate
        if metrics["error_rate"] > self.alert_thresholds["error_rate"]:
            if self._should_send_alert("error_rate", current_time):
                alert = self._send_error_rate_alert(metrics["error_rate"])
                alerts_sent.append(alert)
        
        # Check latency
        if metrics["average_latency"] > self.alert_thresholds["latency"]:
            if self._should_send_alert("latency", current_time):
                alert = self._send_latency_alert(metrics["average_latency"])
                alerts_sent.append(alert)
        
        # Check failed requests
        if metrics["failed_requests"] > self.alert_thresholds["failed_requests"]:
            if self._should_send_alert("failed_requests", current_time):
                alert = self._send_failed_requests_alert(metrics["failed_requests"])
                alerts_sent.append(alert)
        
        return alerts_sent
    
    def _should_send_alert(self, alert_type: str, current_time: float) -> bool:
        """Check if enough time has passed to send alert again"""
        last_time = self.last_alert_times.get(alert_type, 0)
        return (current_time - last_time) > self.alert_cooldown
    
    def _send_error_rate_alert(self, error_rate: float) -> Dict[str, Any]:
        """Send error rate alert"""
        alert = {
            "type": "error_rate",
            "severity": "high" if error_rate > 25 else "medium",
            "message": f"High error rate detected: {error_rate:.1f}%",
            "timestamp": datetime.now().isoformat(),
            "action_required": "Check LangSmith logs for error patterns"
        }
        
        # In production, send to alerting system (PagerDuty, Slack, etc.)
        print(f"🚨 ALERT: {alert['message']}")
        
        self.last_alert_times["error_rate"] = time.time()
        return alert
    
    def _send_latency_alert(self, latency: float) -> Dict[str, Any]:
        """Send latency alert"""
        alert = {
            "type": "latency",
            "severity": "medium",
            "message": f"High average latency: {latency:.2f}s",
            "timestamp": datetime.now().isoformat(),
            "action_required": "Check system performance and LLM response times"
        }
        
        print(f"⚠️ ALERT: {alert['message']}")
        
        self.last_alert_times["latency"] = time.time()
        return alert
    
    def _send_failed_requests_alert(self, failed_count: int) -> Dict[str, Any]:
        """Send failed requests alert"""
        alert = {
            "type": "failed_requests",
            "severity": "high",
            "message": f"High number of failed requests: {failed_count}",
            "timestamp": datetime.now().isoformat(),
            "action_required": "Investigate request failures in LangSmith traces"
        }
        
        print(f"🚨 ALERT: {alert['message']}")
        
        self.last_alert_times["failed_requests"] = time.time()
        return alert

def simulate_production_traffic(monitor: ProductionLangSmithMonitor, 
                              chain, num_requests: int = 50):
    """Simulate production traffic for demonstration"""
    
    print(f"🚦 Simulating {num_requests} production requests...")
    
    # Sample requests that might come in production
    sample_requests = [
        {
            "context": "Customer support for e-commerce platform",
            "question": "How do I track my order?"
        },
        {
            "context": "Technical documentation",
            "question": "What are the API rate limits?"
        },
        {
            "context": "User account management",
            "question": "How do I update my billing information?"
        },
        {
            "context": "Product information",
            "question": "What are the system requirements?"
        },
        {
            "context": "Troubleshooting guide",
            "question": "Why is my login not working?"
        }
    ]
    
    results = []
    
    for i in range(num_requests):
        # Pick random request
        import random
        request_data = random.choice(sample_requests)
        
        # Add some variability
        user_id = f"user_{random.randint(1000, 9999)}"
        session_id = f"session_{random.randint(100000, 999999)}"
        
        # Sometimes simulate errors by using invalid input
        if random.random() < 0.1:  # 10% error rate
            request_data = {"invalid": "input"}  # This will cause an error
        
        # Monitor execution
        result = monitor.monitor_chain_execution(
            chain=chain,
            inputs=request_data,
            user_id=user_id,
            session_id=session_id
        )
        
        results.append(result)
        
        # Small delay to simulate real traffic
        time.sleep(0.1)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{num_requests} requests...")
    
    return results

def main():
    """Run production monitoring demonstration"""
    print("LangSmith Production Monitoring Demo")
    print("=" * 40)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required")
        return
    
    try:
        # Setup production monitoring
        monitor = ProductionLangSmithMonitor("production-demo")
        alert_manager = LangSmithAlertManager(monitor)
        
        # Create a simple chain for testing
        llm = ChatOpenAI(temperature=0.1)
        prompt = PromptTemplate(
            template="Context: {context}\nQuestion: {question}\nAnswer:",
            input_variables=["context", "question"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Simulate production traffic
        print("\n📊 Starting production traffic simulation...")
        traffic_results = simulate_production_traffic(monitor, chain, 30)
        
        # Check metrics
        print("\n📈 Current Metrics:")
        metrics = monitor.get_metrics_summary()
        for key, value in metrics.items():
            if key == "last_updated":
                continue
            print(f"   {key}: {value}")
        
        # Health check
        print("\n🏥 Health Check:")
        health = monitor.health_check()
        if health["healthy"]:
            print("   ✅ System is healthy")
        else:
            print("   ⚠️ Issues detected:")
            for issue in health["issues"]:
                print(f"      - {issue}")
        
        # Check for alerts
        print("\n🚨 Checking for alerts...")
        alerts = alert_manager.check_and_send_alerts()
        if alerts:
            print(f"   Sent {len(alerts)} alerts")
        else:
            print("   No alerts triggered")
        
        print("\n✅ Production monitoring demonstration completed!")
        print("🔗 View detailed traces at: https://smith.langchain.com/")
        
    except Exception as e:
        print(f"❌ Production monitoring demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

---

**🎯 LangSmith Chapter Summary:**

This comprehensive LangSmith integration covers:

✅ **Setup & Configuration**
- Environment setup and API key management
- Project configuration and tracing enablement

✅ **Advanced Tracing**
- Detailed execution tracing with metadata
- Batch execution monitoring
- Custom trace annotations

✅ **Evaluation & Testing**
- Custom evaluator implementations
- Dataset management and testing
- Performance metrics and scoring

✅ **Production Monitoring**
- Real-time monitoring and alerting
- Health checks and performance tracking
- Error rate and latency monitoring

✅ **Alert Management**
- Configurable alert thresholds
- Cooldown mechanisms
- Multi-channel alerting support

This addition makes the guide even more comprehensive by covering observability and production monitoring - essential aspects of deploying LangChain applications at scale.