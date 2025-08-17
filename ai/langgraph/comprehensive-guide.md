# LangGraph Complete Guide - From Zero to Production

LangGraph is a revolutionary framework for building stateful, multi-agent AI applications with cyclical workflows. This comprehensive guide takes you from absolute beginner to production deployment, covering every aspect with hands-on examples and real-world projects.

## Table of Contents

1. [Installation and Setup](#1-installation-and-setup)
2. [Development Environment](#2-development-environment) 
3. [Quick Start](#3-quick-start)
4. [Core Concepts](#4-core-concepts)
5. [Basic Graphs](#5-basic-graphs)
6. [State Management](#6-state-management)
7. [Node Development](#7-node-development)
8. [Edge Configuration](#8-edge-configuration)
9. [Memory and Checkpointing](#9-memory-and-checkpointing)
10. [Multi-Agent Systems](#10-multi-agent-systems)
11. [Advanced Patterns](#11-advanced-patterns)
12. [Real-World Project Structure](#12-real-world-project-structure)
13. [Testing Strategies](#13-testing-strategies)
14. [Performance Optimization](#14-performance-optimization)
15. [Production Deployment](#15-production-deployment)
16. [Monitoring and Observability](#16-monitoring-and-observability)
17. [Troubleshooting](#17-troubleshooting)
18. [Migration Patterns](#18-migration-patterns)
19. [Enterprise Integration](#19-enterprise-integration)

---

## 1. Installation and Setup

### Prerequisites

Before installing LangGraph, ensure you have:

```bash
# Check Python version (3.8+ required)
python --version
# Python 3.8.0 or higher

# Check pip version
pip --version
```

### Basic Installation

```bash
# Core LangGraph installation
pip install langgraph

# Verify installation
python -c "import langgraph; print(langgraph.__version__)"
```

### Complete Installation with Dependencies

```bash
# LangGraph with all optional dependencies
pip install "langgraph[all]"

# Individual optional dependencies
pip install "langgraph[async]"      # Async support
pip install "langgraph[postgres]"   # PostgreSQL checkpointing
pip install "langgraph[redis]"      # Redis checkpointing
pip install "langgraph[sqlite]"     # SQLite checkpointing

# For development and testing
pip install "langgraph[dev]"        # Development tools
pip install "langgraph[test]"       # Testing utilities
```

### LangChain Integration

```bash
# Core LangChain (required)
pip install langchain
pip install langchain-core
pip install langchain-community

# LLM providers (choose as needed)
pip install langchain-openai        # OpenAI GPT models
pip install langchain-anthropic     # Claude models
pip install langchain-google-genai  # Google Gemini
pip install langchain-ollama        # Local Ollama models
```

### Environment Variables Setup

Create a `.env` file in your project root:

```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your_project_name

# Database connections (for persistent memory)
DATABASE_URL=postgresql://user:password@localhost:5432/langgraph_db
REDIS_URL=redis://localhost:6379/0

# Application settings
LOG_LEVEL=INFO
DEBUG=false
```

### Verification Script

Create `verify_installation.py`:

```python
#!/usr/bin/env python3
"""
LangGraph installation verification script
Run this to ensure everything is properly installed
"""

import sys
from typing import Dict, List, Any

def check_core_installation():
    """Check core LangGraph installation"""
    try:
        import langgraph
        print(f"✅ LangGraph installed: {langgraph.__version__}")
        return True
    except ImportError as e:
        print(f"❌ LangGraph not installed: {e}")
        return False

def check_langchain_integration():
    """Check LangChain integration"""
    try:
        from langchain_core.messages import BaseMessage, HumanMessage
        from langgraph.graph import StateGraph, START, END
        print("✅ LangChain integration working")
        return True
    except ImportError as e:
        print(f"❌ LangChain integration failed: {e}")
        return False

def check_optional_dependencies():
    """Check optional dependencies"""
    results = {}
    
    # Async support
    try:
        import asyncio
        import aiohttp
        results['async'] = True
        print("✅ Async support available")
    except ImportError:
        results['async'] = False
        print("⚠️  Async support not available")
    
    # Database support
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        results['postgres'] = True
        print("✅ PostgreSQL support available")
    except ImportError:
        results['postgres'] = False
        print("⚠️  PostgreSQL support not available")
    
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        results['sqlite'] = True
        print("✅ SQLite support available")
    except ImportError:
        results['sqlite'] = False
        print("⚠️  SQLite support not available")
    
    return results

def check_llm_providers():
    """Check LLM provider availability"""
    providers = {}
    
    # OpenAI
    try:
        from langchain_openai import ChatOpenAI
        providers['openai'] = True
        print("✅ OpenAI provider available")
    except ImportError:
        providers['openai'] = False
        print("⚠️  OpenAI provider not available")
    
    # Anthropic
    try:
        from langchain_anthropic import ChatAnthropic
        providers['anthropic'] = True
        print("✅ Anthropic provider available")
    except ImportError:
        providers['anthropic'] = False
        print("⚠️  Anthropic provider not available")
    
    return providers

def test_basic_functionality():
    """Test basic LangGraph functionality"""
    try:
        from typing import TypedDict
        from langgraph.graph import StateGraph, START, END
        
        class TestState(TypedDict):
            message: str
        
        def test_node(state: TestState) -> TestState:
            return {"message": "Hello from LangGraph!"}
        
        # Build simple graph
        graph = StateGraph(TestState)
        graph.add_node("test", test_node)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)
        
        # Compile and test
        workflow = graph.compile()
        result = workflow.invoke({"message": ""})
        
        if result["message"] == "Hello from LangGraph!":
            print("✅ Basic functionality test passed")
            return True
        else:
            print(f"❌ Basic functionality test failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Basic functionality test error: {e}")
        return False

def main():
    """Main verification function"""
    print("🔍 Verifying LangGraph installation...\n")
    
    checks = [
        ("Core Installation", check_core_installation),
        ("LangChain Integration", check_langchain_integration),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n📋 {name}:")
        if not check_func():
            all_passed = False
    
    print(f"\n📋 Optional Dependencies:")
    check_optional_dependencies()
    
    print(f"\n📋 LLM Providers:")
    check_llm_providers()
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 LangGraph installation verified successfully!")
        print("You're ready to start building with LangGraph.")
    else:
        print("⚠️  Some issues found. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Run the verification:

```bash
python verify_installation.py
```

---

## 2. Development Environment

### Project Structure Setup

Create a new LangGraph project with proper structure:

```bash
# Create project directory
mkdir my_langgraph_project
cd my_langgraph_project

# Initialize git repository
git init

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Create project structure
mkdir -p {src,tests,config,data,logs,docs,scripts}
mkdir -p src/{agents,graphs,nodes,utils,models}
mkdir -p tests/{unit,integration,e2e}
```

### Complete Project Structure

```
my_langgraph_project/
├── .env                      # Environment variables
├── .gitignore               # Git ignore file
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── pyproject.toml          # Modern Python packaging
├── pytest.ini             # Pytest configuration
├── docker-compose.yml      # Docker setup
├── Dockerfile              # Container definition
│
├── src/
│   ├── __init__.py
│   ├── main.py             # Main application entry
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py     # Configuration management
│   │   └── logging.py      # Logging setup
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── state.py        # State definitions
│   │   └── schemas.py      # Data schemas
│   │
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── base.py         # Base node classes
│   │   ├── processing/     # Processing nodes
│   │   ├── analysis/       # Analysis nodes
│   │   └── output/         # Output nodes
│   │
│   ├── graphs/
│   │   ├── __init__.py
│   │   ├── base.py         # Base graph classes
│   │   ├── simple/         # Simple workflows
│   │   └── complex/        # Complex workflows
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py   # Base agent class
│   │   ├── research/       # Research agents
│   │   └── analysis/       # Analysis agents
│   │
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py      # Utility functions
│       ├── validation.py   # Data validation
│       └── monitoring.py   # Monitoring utilities
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Pytest fixtures
│   ├── unit/
│   │   ├── test_nodes.py
│   │   ├── test_graphs.py
│   │   └── test_agents.py
│   ├── integration/
│   │   ├── test_workflows.py
│   │   └── test_memory.py
│   └── e2e/
│       └── test_complete_flows.py
│
├── config/
│   ├── development.yaml    # Dev configuration
│   ├── production.yaml     # Prod configuration
│   └── test.yaml          # Test configuration
│
├── data/
│   ├── inputs/            # Input data
│   ├── outputs/           # Output data
│   └── samples/           # Sample data
│
├── logs/                  # Application logs
├── docs/                  # Documentation
└── scripts/
    ├── setup.sh          # Setup script
    ├── test.sh           # Test runner
    └── deploy.sh         # Deployment script
```

### Configuration Files

#### `requirements.txt`
```txt
# Core dependencies
langgraph>=0.0.40
langchain>=0.1.0
langchain-core>=0.1.0
langchain-community>=0.0.20

# LLM providers (choose as needed)
langchain-openai>=0.0.5
langchain-anthropic>=0.1.0

# Database and storage
psycopg2-binary>=2.9.0
redis>=4.5.0
sqlalchemy>=2.0.0

# Async support
aiohttp>=3.8.0
asyncio-mqtt>=0.13.0

# Development and testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
black>=23.0.0
isort>=5.12.0
mypy>=1.0.0
flake8>=6.0.0

# Monitoring and logging
structlog>=23.1.0
prometheus-client>=0.16.0

# Environment management
python-dotenv>=1.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Utilities
click>=8.1.0
rich>=13.0.0
httpx>=0.24.0
```

#### `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "my-langgraph-project"
version = "0.1.0"
description = "A comprehensive LangGraph application"
authors = [{name = "Your Name", email = "your.email@example.com"}]
dependencies = [
    "langgraph>=0.0.40",
    "langchain>=0.1.0",
    "langchain-core>=0.1.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0", 
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
    "flake8>=6.0.0",
]
llm = [
    "langchain-openai>=0.0.5",
    "langchain-anthropic>=0.1.0",
]
database = [
    "psycopg2-binary>=2.9.0",
    "redis>=4.5.0",
    "sqlalchemy>=2.0.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
src_paths = ["src", "tests"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=html --cov-report=term"
```

#### `src/config/settings.py`
```python
"""
Configuration management for LangGraph application
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    langchain_api_key: Optional[str] = Field(default=None, description="LangChain API key")
    
    # LangChain settings
    langchain_tracing_v2: bool = Field(default=True, description="Enable LangChain tracing")
    langchain_project: str = Field(default="langgraph-project", description="LangChain project name")
    
    # Database settings
    database_url: Optional[str] = Field(default=None, description="Database connection URL")
    redis_url: Optional[str] = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    
    # Application settings
    max_workers: int = Field(default=4, description="Maximum worker threads")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # Memory settings
    memory_backend: str = Field(default="memory", description="Memory backend type")
    checkpoint_ttl: int = Field(default=86400, description="Checkpoint TTL in seconds")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class DevelopmentSettings(Settings):
    """Development environment settings"""
    environment: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"

class ProductionSettings(Settings):
    """Production environment settings"""
    environment: str = "production"
    debug: bool = False
    log_level: str = "WARNING"
    
    # Stricter validation for production
    openai_api_key: str = Field(..., description="OpenAI API key (required in prod)")

class TestSettings(Settings):
    """Test environment settings"""
    environment: str = "test"
    debug: bool = True
    log_level: str = "DEBUG"
    database_url: str = "sqlite:///:memory:"

def get_settings() -> Settings:
    """Get settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()

# Global settings instance
settings = get_settings()
```

### IDE Configuration

#### VS Code Settings (`.vscode/settings.json`)
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true,
        "htmlcov": true
    }
}
```

---

## 3. Quick Start

Now let's build your first LangGraph application step by step.

### Step 1: Your First Simple Graph

Create `src/examples/hello_world.py`:

```python
"""
Hello World - Your first LangGraph application
This example shows the absolute basics of creating a graph
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Step 1: Define your state
class HelloState(TypedDict):
    """State that flows through our graph"""
    message: str
    counter: int

# Step 2: Create node functions
def greeting_node(state: HelloState) -> HelloState:
    """A simple node that updates the message"""
    print(f"🤖 Greeting node received: {state}")
    
    return {
        "message": "Hello from LangGraph!",
        "counter": state.get("counter", 0) + 1
    }

def farewell_node(state: HelloState) -> HelloState:
    """Another node that says farewell"""
    print(f"👋 Farewell node received: {state}")
    
    return {
        "message": f"{state['message']} Goodbye!",
        "counter": state["counter"] + 1
    }

# Step 3: Build the graph
def create_hello_graph():
    """Create and return a simple graph"""
    
    # Initialize the graph with our state type
    graph = StateGraph(HelloState)
    
    # Add nodes to the graph
    graph.add_node("greet", greeting_node)
    graph.add_node("farewell", farewell_node)
    
    # Define the flow: START -> greet -> farewell -> END
    graph.add_edge(START, "greet")
    graph.add_edge("greet", "farewell")
    graph.add_edge("farewell", END)
    
    # Compile the graph into an executable workflow
    return graph.compile()

# Step 4: Execute the graph
def main():
    """Main function to run our first graph"""
    print("🚀 Running your first LangGraph application!\n")
    
    # Create the graph
    workflow = create_hello_graph()
    
    # Define initial state
    initial_state = {
        "message": "",
        "counter": 0
    }
    
    print(f"📨 Initial state: {initial_state}")
    
    # Run the workflow
    result = workflow.invoke(initial_state)
    
    print(f"✅ Final result: {result}")
    print(f"📊 Message: {result['message']}")
    print(f"📊 Counter: {result['counter']}")

if __name__ == "__main__":
    main()
```

Run your first graph:

```bash
cd my_langgraph_project
python -m src.examples.hello_world
```

**Expected Output:**
```
🚀 Running your first LangGraph application!

📨 Initial state: {'message': '', 'counter': 0}
🤖 Greeting node received: {'message': '', 'counter': 0}
👋 Farewell node received: {'message': 'Hello from LangGraph!', 'counter': 1}
✅ Final result: {'message': 'Hello from LangGraph! Goodbye!', 'counter': 2}
📊 Message: Hello from LangGraph! Goodbye!
📊 Counter: 2
```

**🎉 Congratulations!** You just created and executed your first LangGraph application!

### Step 2: Understanding What Happened

Let's break down what happened in your first graph:

1. **State Definition**: We defined `HelloState` as a TypedDict to specify what data flows through our graph
2. **Node Functions**: We created simple Python functions that take state and return updated state
3. **Graph Construction**: We built a graph by adding nodes and connecting them with edges
4. **Execution**: We compiled the graph and ran it with initial data

### Step 3: Adding Conditional Logic

Now let's create a more interesting example with decision making:

Create `src/examples/conditional_flow.py`:

```python
"""
Conditional Flow - Adding decision logic to graphs
This example shows how to create branching logic
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class ConditionalState(TypedDict):
    """State with user input and processing path"""
    user_input: str
    processing_path: str
    result: str
    confidence: float

def analyze_input(state: ConditionalState) -> ConditionalState:
    """Analyze user input and determine confidence"""
    user_input = state["user_input"].lower().strip()
    
    # Simple sentiment analysis
    positive_words = ["good", "great", "excellent", "amazing", "love"]
    negative_words = ["bad", "terrible", "awful", "hate", "horrible"]
    
    positive_count = sum(1 for word in positive_words if word in user_input)
    negative_count = sum(1 for word in negative_words if word in user_input)
    
    # Calculate confidence based on word matches
    total_matches = positive_count + negative_count
    confidence = min(1.0, total_matches * 0.3) if total_matches > 0 else 0.1
    
    return {
        **state,
        "confidence": confidence,
        "processing_path": "analyzed"
    }

def high_confidence_processor(state: ConditionalState) -> ConditionalState:
    """Process high-confidence inputs"""
    return {
        **state,
        "result": f"High confidence processing: '{state['user_input']}'",
        "processing_path": "high_confidence"
    }

def low_confidence_processor(state: ConditionalState) -> ConditionalState:
    """Process low-confidence inputs"""
    return {
        **state,
        "result": f"Low confidence processing: '{state['user_input']}'",
        "processing_path": "low_confidence"
    }

def route_by_confidence(state: ConditionalState) -> Literal["high", "low"]:
    """Routing function based on confidence level"""
    return "high" if state["confidence"] > 0.5 else "low"

def create_conditional_graph():
    """Create a graph with conditional routing"""
    graph = StateGraph(ConditionalState)
    
    # Add all nodes
    graph.add_node("analyze", analyze_input)
    graph.add_node("high_confidence", high_confidence_processor)
    graph.add_node("low_confidence", low_confidence_processor)
    
    # Linear flow to analyzer
    graph.add_edge(START, "analyze")
    
    # Conditional routing based on confidence
    graph.add_conditional_edge(
        "analyze",                    # Source node
        route_by_confidence,          # Routing function
        {
            "high": "high_confidence", # If function returns "high"
            "low": "low_confidence"    # If function returns "low"
        }
    )
    
    # Both processors go to END
    graph.add_edge("high_confidence", END)
    graph.add_edge("low_confidence", END)
    
    return graph.compile()

def main():
    """Test conditional routing with different inputs"""
    workflow = create_conditional_graph()
    
    test_inputs = [
        "I love this amazing product!",      # High confidence positive
        "This is terrible and awful",        # High confidence negative  
        "I think this is okay maybe",        # Low confidence neutral
        "Hello there",                       # Low confidence generic
    ]
    
    for user_input in test_inputs:
        print(f"\n🔍 Testing: '{user_input}'")
        
        result = workflow.invoke({
            "user_input": user_input,
            "processing_path": "",
            "result": "",
            "confidence": 0.0
        })
        
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Path: {result['processing_path']}")
        print(f"   Result: {result['result']}")

if __name__ == "__main__":
    main()
```

Run the conditional flow example:

```bash
python -m src.examples.conditional_flow
```

### Step 4: Your First AI-Powered Graph

Now let's integrate with an actual language model:

Create `src/examples/ai_powered.py`:

```python
"""
AI-Powered Graph - Integration with language models
This example shows how to integrate LLMs with LangGraph
"""

import os
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI  # or your preferred LLM
from langgraph.graph import StateGraph, START, END

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class AIState(TypedDict):
    """State for AI-powered graph"""
    messages: List[BaseMessage]
    user_query: str
    ai_response: str
    processing_steps: List[str]

def prepare_messages(state: AIState) -> AIState:
    """Prepare messages for LLM processing"""
    user_query = state["user_query"]
    
    messages = [
        SystemMessage(content="You are a helpful AI assistant. Provide clear, concise answers."),
        HumanMessage(content=user_query)
    ]
    
    steps = state["processing_steps"] + ["Messages prepared"]
    
    return {
        **state,
        "messages": messages,
        "processing_steps": steps
    }

def llm_processing(state: AIState) -> AIState:
    """Process messages with LLM"""
    try:
        # Initialize LLM (ensure you have OPENAI_API_KEY set)
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=150
        )
        
        # Get AI response
        response = llm.invoke(state["messages"])
        
        # Update state
        messages = state["messages"] + [response]
        steps = state["processing_steps"] + ["LLM processing completed"]
        
        return {
            **state,
            "messages": messages,
            "ai_response": response.content,
            "processing_steps": steps
        }
        
    except Exception as e:
        # Fallback response if LLM fails
        fallback_response = f"I apologize, but I encountered an issue: {str(e)}"
        
        return {
            **state,
            "ai_response": fallback_response,
            "processing_steps": state["processing_steps"] + [f"Error: {str(e)}"]
        }

def post_process(state: AIState) -> AIState:
    """Post-process the AI response"""
    response = state["ai_response"]
    
    # Simple post-processing: add emoji and format
    if "?" in state["user_query"]:
        processed_response = f"💡 {response}"
    elif any(word in state["user_query"].lower() for word in ["help", "how", "what"]):
        processed_response = f"🤝 {response}"
    else:
        processed_response = f"💬 {response}"
    
    steps = state["processing_steps"] + ["Post-processing completed"]
    
    return {
        **state,
        "ai_response": processed_response,
        "processing_steps": steps
    }

def create_ai_graph():
    """Create AI-powered graph"""
    graph = StateGraph(AIState)
    
    # Add nodes
    graph.add_node("prepare", prepare_messages)
    graph.add_node("llm", llm_processing)
    graph.add_node("post_process", post_process)
    
    # Create linear flow
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "llm")
    graph.add_edge("llm", "post_process")
    graph.add_edge("post_process", END)
    
    return graph.compile()

def main():
    """Test AI-powered graph"""
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY in your .env file")
        print("   For this demo, we'll use a mock response")
        # You can modify llm_processing to use a mock response for testing
        return
    
    workflow = create_ai_graph()
    
    test_queries = [
        "What is machine learning?",
        "How do I bake a chocolate cake?",
        "Help me understand Python decorators"
    ]
    
    for query in test_queries:
        print(f"\n🤔 User Query: {query}")
        
        result = workflow.invoke({
            "user_query": query,
            "messages": [],
            "ai_response": "",
            "processing_steps": []
        })
        
        print(f"🤖 AI Response: {result['ai_response']}")
        print(f"📝 Processing Steps: {' → '.join(result['processing_steps'])}")

if __name__ == "__main__":
    main()
```

This example demonstrates:
- Integration with language models
- Error handling for API failures
- Message management with LangChain
- Step-by-step processing tracking

---

## 4. Core Concepts

Now that you've built your first graphs, let's dive deep into the fundamental concepts that make LangGraph powerful.

### Understanding State

State is the backbone of LangGraph applications. It's the data that flows through your graph and gets modified at each step.

#### State Design Principles

```python
"""
State Design Best Practices
"""

from typing import TypedDict, List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# 1. Simple State (Good for beginners)
class SimpleState(TypedDict):
    """Basic state structure"""
    input: str
    output: str

# 2. Structured State (Recommended)
class ProcessingStatus(str, Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"

class StructuredState(TypedDict):
    """Well-structured state with clear organization"""
    
    # Input data
    user_input: str
    request_id: str
    timestamp: datetime
    
    # Processing data
    current_step: str
    processing_status: ProcessingStatus
    progress: float  # 0.0 to 1.0
    
    # Results
    intermediate_results: List[Dict[str, Any]]
    final_result: Optional[str]
    
    # Metadata
    processing_time: Optional[float]
    error_message: Optional[str]
    debug_info: Dict[str, Any]

# 3. Hierarchical State (For complex applications)
@dataclass
class UserContext:
    user_id: str
    preferences: Dict[str, Any]
    session_id: str

@dataclass  
class ProcessingContext:
    model_name: str
    temperature: float
    max_tokens: int

class HierarchicalState(TypedDict):
    """Complex state with nested structures"""
    
    # Core data
    query: str
    response: str
    
    # Context objects
    user_context: UserContext
    processing_context: ProcessingContext
    
    # Collections
    conversation_history: List[Dict[str, str]]
    processing_log: List[str]
    
    # State management
    checkpoint_data: Optional[Dict[str, Any]]
```

#### State Evolution Patterns

```python
"""
How state evolves through a graph
"""

def demonstrate_state_evolution():
    """Show how state changes through graph execution"""
    
    # Initial state
    initial_state = {
        "user_input": "What is quantum computing?",
        "current_step": "start",
        "processing_status": ProcessingStatus.PENDING,
        "progress": 0.0,
        "intermediate_results": [],
        "final_result": None,
        "processing_time": None,
        "error_message": None,
        "debug_info": {}
    }
    
    print("📊 State Evolution:")
    print(f"Initial: {initial_state['current_step']} - {initial_state['processing_status']}")
    
    # After input validation node
    after_validation = {
        **initial_state,
        "current_step": "validation",
        "processing_status": ProcessingStatus.IN_PROGRESS,
        "progress": 0.2,
        "intermediate_results": [{"step": "validation", "valid": True}],
        "debug_info": {"validation_time": 0.05}
    }
    
    print(f"After validation: {after_validation['current_step']} - Progress: {after_validation['progress']}")
    
    # After processing node
    after_processing = {
        **after_validation,
        "current_step": "processing",
        "progress": 0.8,
        "intermediate_results": after_validation["intermediate_results"] + [
            {"step": "processing", "result": "LLM response generated"}
        ]
    }
    
    print(f"After processing: {after_processing['current_step']} - Progress: {after_processing['progress']}")
    
    # Final state
    final_state = {
        **after_processing,
        "current_step": "completed",
        "processing_status": ProcessingStatus.COMPLETED,
        "progress": 1.0,
        "final_result": "Quantum computing is...",
        "processing_time": 2.3
    }
    
    print(f"Final: {final_state['current_step']} - {final_state['processing_status']}")

# Run the demonstration
demonstrate_state_evolution()
```

### Graph Architecture Deep Dive

Understanding how graphs work internally helps you design better applications.

#### Graph Components

```python
"""
Deep dive into graph components
"""

from langgraph.graph import StateGraph, START, END
import time
from typing import TypedDict

class AnalysisState(TypedDict):
    content: str
    analysis_type: str
    results: dict
    metadata: dict

class GraphInternalsDemo:
    """Demonstrate internal graph mechanics"""
    
    def __init__(self):
        self.execution_log = []
    
    def log_execution(self, node_name: str, state: AnalysisState) -> None:
        """Log node execution for analysis"""
        self.execution_log.append({
            "timestamp": time.time(),
            "node": node_name,
            "state_keys": list(state.keys()),
            "content_length": len(state.get("content", ""))
        })
    
    def preprocessing_node(self, state: AnalysisState) -> AnalysisState:
        """Preprocessing with detailed logging"""
        self.log_execution("preprocessing", state)
        
        processed_content = state["content"].strip().lower()
        
        return {
            **state,
            "content": processed_content,
            "metadata": {
                **state.get("metadata", {}),
                "preprocessing_done": True,
                "original_length": len(state["content"])
            }
        }
    
    def analysis_node(self, state: AnalysisState) -> AnalysisState:
        """Analysis with performance tracking"""
        start_time = time.time()
        self.log_execution("analysis", state)
        
        # Simulate analysis
        analysis_results = {
            "word_count": len(state["content"].split()),
            "character_count": len(state["content"]),
            "analysis_timestamp": start_time
        }
        
        processing_time = time.time() - start_time
        
        return {
            **state,
            "results": analysis_results,
            "metadata": {
                **state["metadata"],
                "analysis_time": processing_time,
                "analysis_done": True
            }
        }
    
    def postprocessing_node(self, state: AnalysisState) -> AnalysisState:
        """Post-processing with final metrics"""
        self.log_execution("postprocessing", state)
        
        # Enhance results with summary
        enhanced_results = {
            **state["results"],
            "summary": f"Analyzed {state['results']['word_count']} words in {state['metadata']['analysis_time']:.3f}s"
        }
        
        return {
            **state,
            "results": enhanced_results,
            "metadata": {
                **state["metadata"],
                "postprocessing_done": True,
                "total_nodes": len(self.execution_log)
            }
        }
    
    def create_instrumented_graph(self):
        """Create a graph with detailed instrumentation"""
        graph = StateGraph(AnalysisState)
        
        # Add nodes with instrumentation
        graph.add_node("preprocess", self.preprocessing_node)
        graph.add_node("analyze", self.analysis_node) 
        graph.add_node("postprocess", self.postprocessing_node)
        
        # Linear flow
        graph.add_edge(START, "preprocess")
        graph.add_edge("preprocess", "analyze")
        graph.add_edge("analyze", "postprocess")
        graph.add_edge("postprocess", END)
        
        return graph.compile()
    
    def analyze_execution(self):
        """Analyze the execution log"""
        print("\n🔍 Graph Execution Analysis:")
        print("-" * 40)
        
        for i, log_entry in enumerate(self.execution_log):
            print(f"{i+1}. Node: {log_entry['node']}")
            print(f"   Timestamp: {log_entry['timestamp']:.3f}")
            print(f"   State keys: {log_entry['state_keys']}")
            print(f"   Content length: {log_entry['content_length']}")
            
            if i > 0:
                duration = log_entry['timestamp'] - self.execution_log[i-1]['timestamp']
                print(f"   Duration from previous: {duration:.3f}s")
            print()

def main():
    """Demonstrate graph internals"""
    demo = GraphInternalsDemo()
    workflow = demo.create_instrumented_graph()
    
    # Test with sample content
    result = workflow.invoke({
        "content": "  This is SAMPLE Content for Analysis!  ",
        "analysis_type": "text_analysis",
        "results": {},
        "metadata": {}
    })
    
    print("📊 Final Result:")
    print(f"Content: {result['content']}")
    print(f"Results: {result['results']}")
    print(f"Metadata: {result['metadata']}")
    
    # Analyze execution
    demo.analyze_execution()

if __name__ == "__main__":
    main()
```

This foundational understanding prepares you for building more complex applications. Next, we'll explore different types of graphs and when to use each pattern.

---

## 5. Basic Graphs

Let's systematically learn different graph patterns, starting from simple to complex.

### Chapter 1: Linear Graphs (Sequential Processing)

Linear graphs are the simplest pattern - data flows sequentially through nodes without branching.

#### When to Use Linear Graphs
- Data validation pipelines
- Document processing workflows
- Simple transformation chains
- ETL (Extract, Transform, Load) operations

#### Complete Linear Graph Example

Create `src/graphs/linear_example.py`:

```python
"""
Linear Graph Example - Data Processing Pipeline
This demonstrates a complete data processing pipeline using linear graphs
"""

import json
import time
from typing import TypedDict, List, Dict, Any
from datetime import datetime
from langgraph.graph import StateGraph, START, END

class DataPipelineState(TypedDict):
    """State for data processing pipeline"""
    raw_data: List[Dict[str, Any]]
    cleaned_data: List[Dict[str, Any]]
    transformed_data: List[Dict[str, Any]]
    aggregated_data: Dict[str, Any]
    validation_results: Dict[str, Any]
    processing_metadata: Dict[str, Any]

class DataProcessor:
    """Data processing utilities"""
    
    @staticmethod
    def validate_record(record: Dict[str, Any]) -> bool:
        """Validate individual data record"""
        required_fields = ['id', 'timestamp', 'value']
        return all(field in record for field in required_fields)
    
    @staticmethod
    def clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean individual data record"""
        cleaned = record.copy()
        
        # Convert timestamp to standard format
        if isinstance(cleaned.get('timestamp'), str):
            try:
                cleaned['timestamp'] = datetime.fromisoformat(cleaned['timestamp'])
            except ValueError:
                cleaned['timestamp'] = datetime.now()
        
        # Ensure value is numeric
        try:
            cleaned['value'] = float(cleaned['value'])
        except (ValueError, TypeError):
            cleaned['value'] = 0.0
        
        # Add processing timestamp
        cleaned['processed_at'] = datetime.now()
        
        return cleaned

def input_validation_node(state: DataPipelineState) -> DataPipelineState:
    """Validate input data structure and content"""
    start_time = time.time()
    raw_data = state["raw_data"]
    
    validation_results = {
        "total_records": len(raw_data),
        "valid_records": 0,
        "invalid_records": 0,
        "validation_errors": []
    }
    
    for i, record in enumerate(raw_data):
        if DataProcessor.validate_record(record):
            validation_results["valid_records"] += 1
        else:
            validation_results["invalid_records"] += 1
            validation_results["validation_errors"].append({
                "record_index": i,
                "record": record,
                "error": "Missing required fields"
            })
    
    processing_time = time.time() - start_time
    
    metadata = state.get("processing_metadata", {})
    metadata["validation_time"] = processing_time
    metadata["validation_completed"] = datetime.now().isoformat()
    
    print(f"✅ Validation: {validation_results['valid_records']}/{validation_results['total_records']} records valid")
    
    return {
        **state,
        "validation_results": validation_results,
        "processing_metadata": metadata
    }

def data_cleaning_node(state: DataPipelineState) -> DataPipelineState:
    """Clean and normalize data"""
    start_time = time.time()
    raw_data = state["raw_data"]
    
    cleaned_data = []
    for record in raw_data:
        if DataProcessor.validate_record(record):
            cleaned_record = DataProcessor.clean_record(record)
            cleaned_data.append(cleaned_record)
    
    processing_time = time.time() - start_time
    
    metadata = state["processing_metadata"]
    metadata["cleaning_time"] = processing_time
    metadata["cleaning_completed"] = datetime.now().isoformat()
    metadata["records_cleaned"] = len(cleaned_data)
    
    print(f"🧹 Cleaned: {len(cleaned_data)} records processed")
    
    return {
        **state,
        "cleaned_data": cleaned_data,
        "processing_metadata": metadata
    }

def data_transformation_node(state: DataPipelineState) -> DataPipelineState:
    """Transform cleaned data"""
    start_time = time.time()
    cleaned_data = state["cleaned_data"]
    
    transformed_data = []
    
    for record in cleaned_data:
        # Apply transformations
        transformed_record = {
            **record,
            "value_squared": record["value"] ** 2,
            "value_normalized": record["value"] / 100.0,
            "category": "high" if record["value"] > 50 else "low",
            "transformation_applied": True
        }
        transformed_data.append(transformed_record)
    
    processing_time = time.time() - start_time
    
    metadata = state["processing_metadata"]
    metadata["transformation_time"] = processing_time
    metadata["transformation_completed"] = datetime.now().isoformat()
    
    print(f"🔄 Transformed: {len(transformed_data)} records transformed")
    
    return {
        **state,
        "transformed_data": transformed_data,
        "processing_metadata": metadata
    }

def data_aggregation_node(state: DataPipelineState) -> DataPipelineState:
    """Aggregate transformed data"""
    start_time = time.time()
    transformed_data = state["transformed_data"]
    
    if not transformed_data:
        aggregated_data = {"error": "No data to aggregate"}
    else:
        values = [record["value"] for record in transformed_data]
        high_category_count = len([r for r in transformed_data if r["category"] == "high"])
        low_category_count = len([r for r in transformed_data if r["category"] == "low"])
        
        aggregated_data = {
            "total_records": len(transformed_data),
            "average_value": sum(values) / len(values),
            "max_value": max(values),
            "min_value": min(values),
            "high_category_count": high_category_count,
            "low_category_count": low_category_count,
            "high_category_percentage": (high_category_count / len(transformed_data)) * 100,
            "processing_summary": {
                "pipeline_completed": True,
                "total_processing_steps": 4,
                "aggregation_timestamp": datetime.now().isoformat()
            }
        }
    
    processing_time = time.time() - start_time
    
    metadata = state["processing_metadata"]
    metadata["aggregation_time"] = processing_time
    metadata["aggregation_completed"] = datetime.now().isoformat()
    metadata["total_pipeline_time"] = sum([
        metadata.get("validation_time", 0),
        metadata.get("cleaning_time", 0),
        metadata.get("transformation_time", 0),
        metadata.get("aggregation_time", 0)
    ])
    
    print(f"📊 Aggregated: {aggregated_data.get('total_records', 0)} records summarized")
    
    return {
        **state,
        "aggregated_data": aggregated_data,
        "processing_metadata": metadata
    }

def create_data_pipeline():
    """Create linear data processing pipeline"""
    graph = StateGraph(DataPipelineState)
    
    # Add processing nodes in sequence
    graph.add_node("validate", input_validation_node)
    graph.add_node("clean", data_cleaning_node)
    graph.add_node("transform", data_transformation_node)
    graph.add_node("aggregate", data_aggregation_node)
    
    # Create linear flow
    graph.add_edge(START, "validate")
    graph.add_edge("validate", "clean")
    graph.add_edge("clean", "transform")
    graph.add_edge("transform", "aggregate")
    graph.add_edge("aggregate", END)
    
    return graph.compile()

def generate_sample_data() -> List[Dict[str, Any]]:
    """Generate sample data for testing"""
    import random
    
    sample_data = []
    for i in range(20):
        record = {
            "id": f"record_{i:03d}",
            "timestamp": datetime.now().isoformat(),
            "value": random.randint(1, 100),
            "source": f"sensor_{random.randint(1, 5)}"
        }
        
        # Introduce some invalid records for testing
        if i % 7 == 0:  # Every 7th record missing timestamp
            del record["timestamp"]
        elif i % 11 == 0:  # Every 11th record has invalid value
            record["value"] = "invalid_value"
        
        sample_data.append(record)
    
    return sample_data

def main():
    """Test the linear data pipeline"""
    print("🚀 Starting Linear Data Processing Pipeline\n")
    
    # Create pipeline
    pipeline = create_data_pipeline()
    
    # Generate test data
    test_data = generate_sample_data()
    print(f"📥 Generated {len(test_data)} sample records")
    
    # Initialize state
    initial_state: DataPipelineState = {
        "raw_data": test_data,
        "cleaned_data": [],
        "transformed_data": [],
        "aggregated_data": {},
        "validation_results": {},
        "processing_metadata": {"pipeline_started": datetime.now().isoformat()}
    }
    
    # Execute pipeline
    print("\n🔄 Processing through pipeline...")
    result = pipeline.invoke(initial_state)
    
    # Display results
    print(f"\n📋 Pipeline Results:")
    print(f"   Validation: {result['validation_results']}")
    print(f"   Final aggregation: {result['aggregated_data']}")
    print(f"   Processing metadata: {result['processing_metadata']}")
    
    # Save results to file
    output_file = "data/outputs/pipeline_results.json"
    import os
    os.makedirs("data/outputs", exist_ok=True)
    
    with open(output_file, 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        serializable_result = json.loads(json.dumps(result, default=str))
        json.dump(serializable_result, f, indent=2)
    
    print(f"💾 Results saved to {output_file}")

if __name__ == "__main__":
    main()
```

#### Hands-on Exercise: Build Your Own Linear Pipeline

**Exercise 1**: Create a document processing pipeline

Create `src/exercises/document_pipeline.py`:

```python
"""
Exercise: Document Processing Pipeline
Build a linear pipeline that processes text documents
"""

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
import re
from collections import Counter

class DocumentState(TypedDict):
    """Your state definition here"""
    # TODO: Define the state structure
    pass

def load_document_node(state: DocumentState) -> DocumentState:
    """Load and validate document"""
    # TODO: Implement document loading
    pass

def preprocess_text_node(state: DocumentState) -> DocumentState:
    """Clean and preprocess text"""
    # TODO: Implement text preprocessing
    # - Remove special characters
    # - Convert to lowercase
    # - Remove extra whitespace
    pass

def analyze_text_node(state: DocumentState) -> DocumentState:
    """Analyze text content"""
    # TODO: Implement text analysis
    # - Count words, sentences, paragraphs
    # - Extract key phrases
    # - Calculate readability metrics
    pass

def generate_summary_node(state: DocumentState) -> DocumentState:
    """Generate document summary"""
    # TODO: Implement summary generation
    # - Create word frequency analysis
    # - Generate key statistics
    # - Create final summary
    pass

def create_document_pipeline():
    """Create your document processing pipeline"""
    # TODO: Build the graph
    # Hint: Use the same pattern as the data pipeline example
    pass

# Test your implementation
if __name__ == "__main__":
    # TODO: Test with sample text
    pass
```

**Solution** (Try the exercise first!):

<details>
<summary>Click to see solution</summary>

```python
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
import re
from collections import Counter

class DocumentState(TypedDict):
    raw_text: str
    cleaned_text: str
    analysis_results: Dict[str, any]
    summary: Dict[str, any]
    processing_steps: List[str]

def load_document_node(state: DocumentState) -> DocumentState:
    """Load and validate document"""
    raw_text = state["raw_text"]
    steps = state["processing_steps"] + ["Document loaded"]
    
    if not raw_text.strip():
        raise ValueError("Empty document provided")
    
    return {**state, "processing_steps": steps}

def preprocess_text_node(state: DocumentState) -> DocumentState:
    """Clean and preprocess text"""
    text = state["raw_text"]
    
    # Remove extra whitespace and normalize
    cleaned = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep basic punctuation
    cleaned = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', cleaned)
    
    steps = state["processing_steps"] + ["Text preprocessed"]
    
    return {
        **state,
        "cleaned_text": cleaned,
        "processing_steps": steps
    }

def analyze_text_node(state: DocumentState) -> DocumentState:
    """Analyze text content"""
    text = state["cleaned_text"]
    
    # Basic analysis
    words = text.lower().split()
    sentences = re.split(r'[.!?]+', text)
    paragraphs = text.split('\n\n')
    
    word_freq = Counter(words)
    
    analysis_results = {
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "paragraph_count": len([p for p in paragraphs if p.strip()]),
        "unique_words": len(word_freq),
        "most_common_words": word_freq.most_common(10),
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }
    
    steps = state["processing_steps"] + ["Text analyzed"]
    
    return {
        **state,
        "analysis_results": analysis_results,
        "processing_steps": steps
    }

def generate_summary_node(state: DocumentState) -> DocumentState:
    """Generate document summary"""
    analysis = state["analysis_results"]
    
    summary = {
        "overview": f"Document contains {analysis['word_count']} words in {analysis['sentence_count']} sentences",
        "readability": "Simple" if analysis['average_word_length'] < 5 else "Complex",
        "top_keywords": [word for word, _ in analysis['most_common_words'][:5]],
        "statistics": {
            "lexical_diversity": analysis['unique_words'] / analysis['word_count'] if analysis['word_count'] > 0 else 0,
            "words_per_sentence": analysis['word_count'] / analysis['sentence_count'] if analysis['sentence_count'] > 0 else 0
        }
    }
    
    steps = state["processing_steps"] + ["Summary generated"]
    
    return {
        **state,
        "summary": summary,
        "processing_steps": steps
    }

def create_document_pipeline():
    """Create document processing pipeline"""
    graph = StateGraph(DocumentState)
    
    graph.add_node("load", load_document_node)
    graph.add_node("preprocess", preprocess_text_node)
    graph.add_node("analyze", analyze_text_node)
    graph.add_node("summarize", generate_summary_node)
    
    graph.add_edge(START, "load")
    graph.add_edge("load", "preprocess")
    graph.add_edge("preprocess", "analyze")
    graph.add_edge("analyze", "summarize")
    graph.add_edge("summarize", END)
    
    return graph.compile()
```

</details>

### Chapter 2: Branching Graphs (Conditional Logic)

Branching graphs introduce decision-making into your workflows. They route execution based on conditions in your state.

#### When to Use Branching Graphs
- Content classification systems
- User intent routing
- Error handling with fallbacks
- Multi-path processing based on data characteristics

#### Complete Branching Example

Create `src/graphs/branching_example.py`:

```python
"""
Branching Graph Example - Intelligent Content Router
This demonstrates conditional routing based on content analysis
"""

import re
from typing import TypedDict, Literal, List, Dict, Any
from datetime import datetime
from langgraph.graph import StateGraph, START, END

class ContentState(TypedDict):
    """State for content routing system"""
    content: str
    content_type: str
    confidence: float
    processing_path: str
    analysis_results: Dict[str, Any]
    final_output: str
    routing_history: List[str]

class ContentAnalyzer:
    """Utility class for content analysis"""
    
    @staticmethod
    def detect_programming_code(text: str) -> float:
        """Detect if content contains programming code"""
        code_indicators = [
            r'def\s+\w+\(', r'class\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import',
            r'\w+\s*=\s*\w+\(', r'if\s+\w+\s*[=!<>]', r'for\s+\w+\s+in\s+',
            r'\{.*\}', r'\[.*\]', r'//.*|/\*.*\*/', r'#.*'
        ]
        
        matches = sum(1 for pattern in code_indicators if re.search(pattern, text))
        return min(1.0, matches * 0.2)
    
    @staticmethod
    def detect_question(text: str) -> float:
        """Detect if content is a question"""
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'who', 'which']
        text_lower = text.lower()
        
        # Direct question mark
        if '?' in text:
            return 0.9
        
        # Question words at the beginning
        words = text_lower.split()
        if words and words[0] in question_indicators:
            return 0.8
        
        # Question words anywhere
        question_word_count = sum(1 for word in question_indicators if word in text_lower)
        return min(0.7, question_word_count * 0.2)
    
    @staticmethod
    def detect_instruction(text: str) -> float:
        """Detect if content contains instructions"""
        instruction_indicators = [
            'please', 'create', 'build', 'make', 'generate', 'write',
            'implement', 'develop', 'design', 'help', 'show', 'explain'
        ]
        text_lower = text.lower()
        
        matches = sum(1 for word in instruction_indicators if word in text_lower)
        
        # Imperative sentences (starting with verb)
        imperative_patterns = [r'^(create|build|make|write|implement|show|explain)']
        imperative_match = any(re.search(pattern, text_lower) for pattern in imperative_patterns)
        
        base_score = min(0.8, matches * 0.15)
        return base_score + (0.2 if imperative_match else 0)

def content_analysis_node(state: ContentState) -> ContentState:
    """Analyze content to determine type and routing"""
    content = state["content"]
    
    # Run all detection algorithms
    code_confidence = ContentAnalyzer.detect_programming_code(content)
    question_confidence = ContentAnalyzer.detect_question(content)
    instruction_confidence = ContentAnalyzer.detect_instruction(content)
    
    # Determine primary content type
    confidences = {
        "code": code_confidence,
        "question": question_confidence,
        "instruction": instruction_confidence
    }
    
    content_type = max(confidences, key=confidences.get)
    confidence = confidences[content_type]
    
    # Default to general if confidence is too low
    if confidence < 0.3:
        content_type = "general"
        confidence = 0.5
    
    analysis_results = {
        "detected_types": confidences,
        "primary_type": content_type,
        "confidence": confidence,
        "content_length": len(content),
        "word_count": len(content.split()),
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    routing_history = state["routing_history"] + [f"Analyzed as {content_type} (conf: {confidence:.2f})"]
    
    print(f"🔍 Analysis: Detected '{content_type}' with {confidence:.1%} confidence")
    
    return {
        **state,
        "content_type": content_type,
        "confidence": confidence,
        "analysis_results": analysis_results,
        "routing_history": routing_history
    }

def code_processing_node(state: ContentState) -> ContentState:
    """Process programming code content"""
    content = state["content"]
    
    # Analyze code characteristics
    lines = content.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Detect language hints
    language_hints = {
        'python': ['def ', 'import ', 'class ', 'if __name__'],
        'javascript': ['function ', 'const ', 'let ', 'var '],
        'java': ['public class', 'private ', 'public static'],
        'sql': ['SELECT', 'FROM', 'WHERE', 'INSERT']
    }
    
    detected_language = "unknown"
    for lang, hints in language_hints.items():
        if any(hint in content for hint in hints):
            detected_language = lang
            break
    
    output = f"""
🔧 Code Analysis Results:
- Detected Language: {detected_language}
- Total Lines: {len(lines)}
- Code Lines: {len(non_empty_lines)}
- Complexity: {'High' if len(non_empty_lines) > 20 else 'Medium' if len(non_empty_lines) > 10 else 'Low'}

Code processed successfully through specialized code handler.
"""
    
    routing_history = state["routing_history"] + ["Processed via code handler"]
    
    return {
        **state,
        "processing_path": "code_processing",
        "final_output": output.strip(),
        "routing_history": routing_history
    }

def question_processing_node(state: ContentState) -> ContentState:
    """Process question content"""
    content = state["content"]
    
    # Analyze question characteristics
    question_types = {
        'what': 'definition/explanation',
        'how': 'process/method',
        'why': 'reason/cause',
        'when': 'time/timing',
        'where': 'location/place',
        'who': 'person/entity',
        'which': 'choice/selection'
    }
    
    content_lower = content.lower()
    detected_question_type = "general"
    
    for word, qtype in question_types.items():
        if word in content_lower:
            detected_question_type = qtype
            break
    
    output = f"""
❓ Question Analysis Results:
- Question Type: {detected_question_type}
- Length: {len(content.split())} words
- Complexity: {'Complex' if len(content.split()) > 15 else 'Simple'}

This question has been categorized and is ready for specialized Q&A processing.
Recommended response strategy: Provide structured answer matching the question type.
"""
    
    routing_history = state["routing_history"] + ["Processed via question handler"]
    
    return {
        **state,
        "processing_path": "question_processing",
        "final_output": output.strip(),
        "routing_history": routing_history
    }

def instruction_processing_node(state: ContentState) -> ContentState:
    """Process instruction/command content"""
    content = state["content"]
    
    # Extract action verbs and objects
    action_words = ['create', 'build', 'make', 'write', 'implement', 'develop', 'design', 'generate']
    detected_actions = [word for word in action_words if word in content.lower()]
    
    # Estimate complexity based on content
    complexity_indicators = len(content.split())
    complexity = "High" if complexity_indicators > 30 else "Medium" if complexity_indicators > 15 else "Low"
    
    output = f"""
⚡ Instruction Analysis Results:
- Detected Actions: {', '.join(detected_actions) if detected_actions else 'General'}
- Estimated Complexity: {complexity}
- Word Count: {complexity_indicators}
- Priority: {'High' if any(word in content.lower() for word in ['urgent', 'asap', 'immediately']) else 'Normal'}

This instruction has been analyzed and prepared for task execution.
Recommended approach: Break down into actionable steps and execute systematically.
"""
    
    routing_history = state["routing_history"] + ["Processed via instruction handler"]
    
    return {
        **state,
        "processing_path": "instruction_processing", 
        "final_output": output.strip(),
        "routing_history": routing_history
    }

def general_processing_node(state: ContentState) -> ContentState:
    """Process general content"""
    content = state["content"]
    
    # Basic content analysis
    word_count = len(content.split())
    char_count = len(content)
    
    # Sentiment approximation
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
    
    positive_count = sum(1 for word in positive_words if word in content.lower())
    negative_count = sum(1 for word in negative_words if word in content.lower())
    
    sentiment = "Positive" if positive_count > negative_count else "Negative" if negative_count > positive_count else "Neutral"
    
    output = f"""
📝 General Content Analysis:
- Word Count: {word_count}
- Character Count: {char_count}
- Estimated Sentiment: {sentiment}
- Reading Level: {'Advanced' if word_count > 50 else 'Intermediate' if word_count > 20 else 'Basic'}

This content has been processed through the general content handler.
No specific content type detected, using default processing approach.
"""
    
    routing_history = state["routing_history"] + ["Processed via general handler"]
    
    return {
        **state,
        "processing_path": "general_processing",
        "final_output": output.strip(),
        "routing_history": routing_history
    }

def route_by_content_type(state: ContentState) -> Literal["code", "question", "instruction", "general"]:
    """Routing function based on detected content type"""
    content_type = state["content_type"]
    confidence = state["confidence"]
    
    # Add confidence-based routing logic
    if confidence < 0.4:
        print(f"⚠️  Low confidence ({confidence:.1%}), routing to general handler")
        return "general"
    
    print(f"🎯 Routing to {content_type} handler (confidence: {confidence:.1%})")
    return content_type

def create_content_router():
    """Create intelligent content routing graph"""
    graph = StateGraph(ContentState)
    
    # Add analysis node
    graph.add_node("analyze", content_analysis_node)
    
    # Add specialized processing nodes
    graph.add_node("code", code_processing_node)
    graph.add_node("question", question_processing_node)
    graph.add_node("instruction", instruction_processing_node)
    graph.add_node("general", general_processing_node)
    
    # Create flow
    graph.add_edge(START, "analyze")
    
    # Conditional routing from analysis
    graph.add_conditional_edge(
        "analyze",
        route_by_content_type,
        {
            "code": "code",
            "question": "question", 
            "instruction": "instruction",
            "general": "general"
        }
    )
    
    # All processors end the flow
    for processor in ["code", "question", "instruction", "general"]:
        graph.add_edge(processor, END)
    
    return graph.compile()

def main():
    """Test the content routing system"""
    print("🚀 Testing Intelligent Content Router\n")
    
    # Create router
    router = create_content_router()
    
    # Test cases
    test_contents = [
        # Code example
        """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
        """,
        
        # Question example
        "What is the best way to implement a binary search algorithm in Python?",
        
        # Instruction example
        "Please create a REST API endpoint that handles user authentication with JWT tokens.",
        
        # General example
        "I really enjoyed the movie last night. The cinematography was excellent and the story was engaging."
    ]
    
    for i, content in enumerate(test_contents, 1):
        print(f"🔍 Test Case {i}:")
        print(f"Content: {content[:100]}{'...' if len(content) > 100 else ''}")
        print("-" * 50)
        
        # Initialize state
        initial_state: ContentState = {
            "content": content.strip(),
            "content_type": "",
            "confidence": 0.0,
            "processing_path": "",
            "analysis_results": {},
            "final_output": "",
            "routing_history": []
        }
        
        # Process through router
        result = router.invoke(initial_state)
        
        # Display results
        print(f"📊 Results:")
        print(f"   Content Type: {result['content_type']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Processing Path: {result['processing_path']}")
        print(f"   Routing History: {' → '.join(result['routing_history'])}")
        print(f"\n📋 Final Output:")
        print(result['final_output'])
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
```

#### Advanced Branching: Multi-Level Decision Trees

For complex scenarios, you can create multi-level decision trees:

Create `src/graphs/multi_level_routing.py`:

```python
"""
Multi-Level Routing Example
Demonstrates complex decision trees with multiple routing levels
"""

from typing import TypedDict, Literal, Dict, Any
from langgraph.graph import StateGraph, START, END

class MultiLevelState(TypedDict):
    user_input: str
    user_type: str  # "premium", "standard", "free"
    complexity: str  # "high", "medium", "low"
    urgency: str   # "urgent", "normal", "low"
    final_handler: str
    processing_metadata: Dict[str, Any]

def classify_user_type(state: MultiLevelState) -> MultiLevelState:
    """First level: Classify user type"""
    # Simulate user type detection logic
    input_text = state["user_input"].lower()
    
    if "premium" in input_text or "priority" in input_text:
        user_type = "premium"
    elif "basic" in input_text or "free" in input_text:
        user_type = "free"
    else:
        user_type = "standard"
    
    return {**state, "user_type": user_type}

def analyze_complexity(state: MultiLevelState) -> MultiLevelState:
    """Second level: Analyze request complexity"""
    input_text = state["user_input"]
    word_count = len(input_text.split())
    
    # Complexity based on length and keywords
    complex_keywords = ["integration", "architecture", "optimization", "scalability"]
    
    if word_count > 50 or any(keyword in input_text.lower() for keyword in complex_keywords):
        complexity = "high"
    elif word_count > 20:
        complexity = "medium"
    else:
        complexity = "low"
    
    return {**state, "complexity": complexity}

def assess_urgency(state: MultiLevelState) -> MultiLevelState:
    """Third level: Assess urgency"""
    input_text = state["user_input"].lower()
    
    urgent_indicators = ["urgent", "asap", "immediately", "critical", "emergency"]
    low_indicators = ["whenever", "no rush", "eventually", "later"]
    
    if any(indicator in input_text for indicator in urgent_indicators):
        urgency = "urgent"
    elif any(indicator in input_text for indicator in low_indicators):
        urgency = "low"
    else:
        urgency = "normal"
    
    return {**state, "urgency": urgency}

# First level routing
def route_by_user_type(state: MultiLevelState) -> Literal["premium_flow", "standard_flow", "free_flow"]:
    user_type = state["user_type"]
    return f"{user_type}_flow"

# Second level routing (for premium users)
def route_premium_by_complexity(state: MultiLevelState) -> Literal["premium_high", "premium_medium", "premium_low"]:
    complexity = state["complexity"]
    return f"premium_{complexity}"

# Second level routing (for standard users)
def route_standard_by_urgency(state: MultiLevelState) -> Literal["standard_urgent", "standard_normal"]:
    urgency = state["urgency"]
    if urgency == "urgent":
        return "standard_urgent"
    else:
        return "standard_normal"

# Handler nodes for different paths
def premium_high_handler(state: MultiLevelState) -> MultiLevelState:
    return {**state, "final_handler": "Premium High-Complexity Handler", 
            "processing_metadata": {"sla": "2 hours", "dedicated_support": True}}

def premium_medium_handler(state: MultiLevelState) -> MultiLevelState:
    return {**state, "final_handler": "Premium Medium-Complexity Handler",
            "processing_metadata": {"sla": "4 hours", "dedicated_support": True}}

def premium_low_handler(state: MultiLevelState) -> MultiLevelState:
    return {**state, "final_handler": "Premium Quick Handler",
            "processing_metadata": {"sla": "1 hour", "dedicated_support": True}}

def standard_urgent_handler(state: MultiLevelState) -> MultiLevelState:
    return {**state, "final_handler": "Standard Urgent Handler",
            "processing_metadata": {"sla": "24 hours", "escalation_available": True}}

def standard_normal_handler(state: MultiLevelState) -> MultiLevelState:
    return {**state, "final_handler": "Standard Normal Handler",
            "processing_metadata": {"sla": "72 hours", "self_service": True}}

def free_handler(state: MultiLevelState) -> MultiLevelState:
    return {**state, "final_handler": "Free Tier Handler",
            "processing_metadata": {"sla": "7 days", "community_support": True}}

def create_multi_level_router():
    """Create multi-level routing graph"""
    graph = StateGraph(MultiLevelState)
    
    # Level 0: Initial analysis
    graph.add_node("classify_user", classify_user_type)
    graph.add_node("analyze_complexity", analyze_complexity)
    graph.add_node("assess_urgency", assess_urgency)
    
    # Level 1: User type routing
    graph.add_node("premium_flow", lambda state: state)  # Passthrough node
    graph.add_node("standard_flow", lambda state: state)  # Passthrough node
    graph.add_node("free_flow", free_handler)
    
    # Level 2: Premium complexity routing
    graph.add_node("premium_high", premium_high_handler)
    graph.add_node("premium_medium", premium_medium_handler)
    graph.add_node("premium_low", premium_low_handler)
    
    # Level 2: Standard urgency routing  
    graph.add_node("standard_urgent", standard_urgent_handler)
    graph.add_node("standard_normal", standard_normal_handler)
    
    # Create flow structure
    graph.add_edge(START, "classify_user")
    graph.add_edge("classify_user", "analyze_complexity")
    graph.add_edge("analyze_complexity", "assess_urgency")
    
    # Level 1 routing
    graph.add_conditional_edge(
        "assess_urgency",
        route_by_user_type,
        {
            "premium_flow": "premium_flow",
            "standard_flow": "standard_flow", 
            "free_flow": "free_flow"
        }
    )
    
    # Level 2 routing for premium
    graph.add_conditional_edge(
        "premium_flow",
        route_premium_by_complexity,
        {
            "premium_high": "premium_high",
            "premium_medium": "premium_medium",
            "premium_low": "premium_low"
        }
    )
    
    # Level 2 routing for standard
    graph.add_conditional_edge(
        "standard_flow", 
        route_standard_by_urgency,
        {
            "standard_urgent": "standard_urgent",
            "standard_normal": "standard_normal"
        }
    )
    
    # All final handlers go to END
    final_handlers = ["premium_high", "premium_medium", "premium_low", 
                     "standard_urgent", "standard_normal", "free_flow"]
    for handler in final_handlers:
        graph.add_edge(handler, END)
    
    return graph.compile()
```

These branching examples show how to build sophisticated routing logic that can handle complex decision-making scenarios.

---

## 6. State Management

State management is the backbone of LangGraph applications. Understanding how to design, manage, and evolve state effectively is crucial for building robust applications.

### 6.1 State Design Principles

#### Understanding State Types

```python
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# Basic State - Simple key-value pairs
class BasicState(TypedDict):
    counter: int
    message: str
    processed: bool

# Message State - For conversation-based applications
class ChatState(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    user_id: str
    context: Dict[str, Any]

# Complex State - Multi-dimensional state
class ProcessingState(TypedDict):
    # Input data
    input_data: Dict[str, Any]
    
    # Processing stages
    validated: bool
    processed_data: Dict[str, Any]
    
    # Metadata
    processing_time: float
    error_log: List[str]
    
    # Results
    results: Dict[str, Any]
    confidence_score: float
```

#### State Evolution Patterns

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict
import time

class DataProcessingState(TypedDict):
    raw_data: Dict[str, Any]
    validated_data: Dict[str, Any] 
    enriched_data: Dict[str, Any]
    final_result: Dict[str, Any]
    stage: str
    errors: List[str]

def validate_data(state: DataProcessingState) -> DataProcessingState:
    """Validate input data and update state"""
    raw_data = state["raw_data"]
    errors = []
    validated_data = {}
    
    # Validation logic
    if not raw_data:
        errors.append("No input data provided")
    else:
        # Validate required fields
        required_fields = ["id", "name", "type"]
        for field in required_fields:
            if field not in raw_data:
                errors.append(f"Missing required field: {field}")
            else:
                validated_data[field] = raw_data[field]
    
    return {
        **state,
        "validated_data": validated_data,
        "stage": "validated",
        "errors": errors
    }

def enrich_data(state: DataProcessingState) -> DataProcessingState:
    """Enrich validated data with additional information"""
    validated_data = state["validated_data"]
    
    # Enrichment logic
    enriched_data = {
        **validated_data,
        "processed_at": time.time(),
        "version": "1.0",
        "metadata": {
            "processing_pipeline": "standard",
            "enrichment_rules": ["timestamp", "version"]
        }
    }
    
    return {
        **state,
        "enriched_data": enriched_data,
        "stage": "enriched"
    }

def finalize_processing(state: DataProcessingState) -> DataProcessingState:
    """Finalize processing and generate results"""
    enriched_data = state["enriched_data"]
    
    final_result = {
        "processed_item": enriched_data,
        "processing_summary": {
            "total_stages": 3,
            "final_stage": state["stage"],
            "error_count": len(state["errors"])
        }
    }
    
    return {
        **state,
        "final_result": final_result,
        "stage": "completed"
    }

# Build the state management graph
def create_state_management_example():
    graph = StateGraph(DataProcessingState)
    
    # Add nodes
    graph.add_node("validate", validate_data)
    graph.add_node("enrich", enrich_data)
    graph.add_node("finalize", finalize_processing)
    
    # Define flow
    graph.add_edge("validate", "enrich")
    graph.add_edge("enrich", "finalize")
    graph.add_edge("finalize", END)
    
    graph.set_entry_point("validate")
    
    return graph.compile()

# Usage example
app = create_state_management_example()

# Test with sample data
initial_state = {
    "raw_data": {"id": "123", "name": "Test Item", "type": "sample"},
    "validated_data": {},
    "enriched_data": {},
    "final_result": {},
    "stage": "initial",
    "errors": []
}

result = app.invoke(initial_state)
print("Final state:", result)
```

### 6.2 Advanced State Patterns

#### State Branching and Merging

```python
from typing import TypedDict, Union, Literal
from langgraph.graph import StateGraph, END

class BranchingState(TypedDict):
    input_value: int
    path_taken: str
    processing_result: Dict[str, Any]
    final_output: str

def route_decision(state: BranchingState) -> Literal["high_value", "low_value", "invalid"]:
    """Determine processing path based on input value"""
    value = state["input_value"]
    
    if value < 0:
        return "invalid"
    elif value > 100:
        return "high_value" 
    else:
        return "low_value"

def process_high_value(state: BranchingState) -> BranchingState:
    """Process high-value inputs with complex logic"""
    return {
        **state,
        "path_taken": "high_value",
        "processing_result": {
            "type": "premium",
            "discount": 0.15,
            "priority": "high"
        }
    }

def process_low_value(state: BranchingState) -> BranchingState:
    """Process low-value inputs with standard logic"""
    return {
        **state,
        "path_taken": "low_value", 
        "processing_result": {
            "type": "standard",
            "discount": 0.05,
            "priority": "normal"
        }
    }

def handle_invalid(state: BranchingState) -> BranchingState:
    """Handle invalid inputs"""
    return {
        **state,
        "path_taken": "invalid",
        "processing_result": {
            "type": "error",
            "message": "Invalid input value",
            "requires_review": True
        }
    }

def merge_results(state: BranchingState) -> BranchingState:
    """Merge results from different paths"""
    result = state["processing_result"]
    path = state["path_taken"]
    
    final_output = f"Processed via {path} path: {result}"
    
    return {
        **state,
        "final_output": final_output
    }

# Create branching state graph
def create_branching_example():
    graph = StateGraph(BranchingState)
    
    # Add processing nodes
    graph.add_node("high_value", process_high_value)
    graph.add_node("low_value", process_low_value)
    graph.add_node("invalid", handle_invalid)
    graph.add_node("merge", merge_results)
    
    # Add conditional routing
    graph.add_conditional_edges(
        "route", 
        route_decision,
        {
            "high_value": "high_value",
            "low_value": "low_value", 
            "invalid": "invalid"
        }
    )
    
    # All paths converge to merge
    graph.add_edge("high_value", "merge")
    graph.add_edge("low_value", "merge")
    graph.add_edge("invalid", "merge")
    graph.add_edge("merge", END)
    
    graph.set_entry_point("route")
    
    return graph.compile()
```

#### State Persistence and Recovery

```python
import pickle
import json
from pathlib import Path
from typing import TypedDict, Optional

class PersistentState(TypedDict):
    session_id: str
    data: Dict[str, Any]
    checkpoint: int
    last_updated: float

class StateManager:
    """Advanced state management with persistence"""
    
    def __init__(self, storage_path: str = "state_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def save_state(self, state: PersistentState) -> None:
        """Save state to persistent storage"""
        session_id = state["session_id"]
        filepath = self.storage_path / f"session_{session_id}.json"
        
        # Convert state to JSON-serializable format
        serializable_state = {
            "session_id": state["session_id"],
            "data": state["data"],
            "checkpoint": state["checkpoint"],
            "last_updated": state["last_updated"]
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_state, f, indent=2)
    
    def load_state(self, session_id: str) -> Optional[PersistentState]:
        """Load state from persistent storage"""
        filepath = self.storage_path / f"session_{session_id}.json"
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            return PersistentState(**data)
    
    def create_checkpoint(self, state: PersistentState) -> PersistentState:
        """Create a checkpoint in the processing"""
        checkpointed_state = {
            **state,
            "checkpoint": state["checkpoint"] + 1,
            "last_updated": time.time()
        }
        
        self.save_state(checkpointed_state)
        return checkpointed_state

# Usage with LangGraph
def stateful_processing_node(state: PersistentState) -> PersistentState:
    """Node that processes and checkpoints state"""
    state_manager = StateManager()
    
    # Process data
    processed_data = {
        **state["data"],
        "processed": True,
        "processing_timestamp": time.time()
    }
    
    # Update state
    updated_state = {
        **state,
        "data": processed_data
    }
    
    # Create checkpoint
    return state_manager.create_checkpoint(updated_state)
```

---

## 7. Node Development

Nodes are the building blocks of LangGraph applications. This section covers everything from basic node creation to advanced node patterns and optimization.

### 7.1 Basic Node Patterns

#### Simple Function Nodes

```python
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END

class ProcessingState(TypedDict):
    input_text: str
    processed_text: str
    word_count: int
    metadata: Dict[str, Any]

def text_preprocessor(state: ProcessingState) -> ProcessingState:
    """Clean and preprocess text input"""
    raw_text = state["input_text"]
    
    # Basic text cleaning
    cleaned_text = raw_text.strip().lower()
    cleaned_text = ' '.join(cleaned_text.split())  # Normalize whitespace
    
    return {
        **state,
        "processed_text": cleaned_text,
        "metadata": {
            **state.get("metadata", {}),
            "preprocessing_applied": True,
            "original_length": len(raw_text),
            "cleaned_length": len(cleaned_text)
        }
    }

def word_counter(state: ProcessingState) -> ProcessingState:
    """Count words in processed text"""
    text = state["processed_text"]
    word_count = len(text.split()) if text else 0
    
    return {
        **state,
        "word_count": word_count,
        "metadata": {
            **state["metadata"],
            "word_counting_completed": True
        }
    }

def text_analyzer(state: ProcessingState) -> ProcessingState:
    """Analyze text characteristics"""
    text = state["processed_text"]
    
    analysis = {
        "character_count": len(text),
        "sentence_count": len([s for s in text.split('.') if s.strip()]),
        "average_word_length": sum(len(word) for word in text.split()) / len(text.split()) if text else 0
    }
    
    return {
        **state,
        "metadata": {
            **state["metadata"],
            "analysis": analysis,
            "analysis_completed": True
        }
    }
```

#### Class-Based Nodes

```python
from abc import ABC, abstractmethod
import logging
from datetime import datetime

class BaseProcessor(ABC):
    """Base class for all processors"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"processor.{name}")
    
    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state"""
        pass
    
    def log_processing(self, state: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Log processing information"""
        self.logger.info(f"Processed state in {self.name}: {len(str(state))} -> {len(str(result))} characters")

class TextProcessor(BaseProcessor):
    """Advanced text processing node"""
    
    def __init__(self, processing_rules: Dict[str, bool] = None):
        super().__init__("text_processor")
        self.rules = processing_rules or {
            "lowercase": True,
            "remove_punctuation": False,
            "remove_numbers": False,
            "normalize_whitespace": True
        }
    
    def process(self, state: ProcessingState) -> ProcessingState:
        """Process text according to configured rules"""
        text = state["input_text"]
        
        if self.rules["lowercase"]:
            text = text.lower()
        
        if self.rules["normalize_whitespace"]:
            text = ' '.join(text.split())
        
        if self.rules["remove_punctuation"]:
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        if self.rules["remove_numbers"]:
            text = ''.join(char for char in text if not char.isdigit())
        
        result = {
            **state,
            "processed_text": text,
            "metadata": {
                **state.get("metadata", {}),
                "processing_rules": self.rules,
                "processed_at": datetime.now().isoformat(),
                "processor": self.name
            }
        }
        
        self.log_processing(state, result)
        return result

class ValidationProcessor(BaseProcessor):
    """Input validation node"""
    
    def __init__(self, validation_rules: Dict[str, Any]):
        super().__init__("validator")
        self.validation_rules = validation_rules
    
    def process(self, state: ProcessingState) -> ProcessingState:
        """Validate input according to rules"""
        text = state["input_text"]
        errors = []
        
        # Length validation
        if "min_length" in self.validation_rules:
            if len(text) < self.validation_rules["min_length"]:
                errors.append(f"Text too short (min: {self.validation_rules['min_length']})")
        
        if "max_length" in self.validation_rules:
            if len(text) > self.validation_rules["max_length"]:
                errors.append(f"Text too long (max: {self.validation_rules['max_length']})")
        
        # Content validation
        if "required_words" in self.validation_rules:
            required_words = self.validation_rules["required_words"]
            missing_words = [word for word in required_words if word not in text.lower()]
            if missing_words:
                errors.append(f"Missing required words: {missing_words}")
        
        result = {
            **state,
            "metadata": {
                **state.get("metadata", {}),
                "validation_errors": errors,
                "is_valid": len(errors) == 0,
                "validated_at": datetime.now().isoformat()
            }
        }
        
        self.log_processing(state, result)
        return result

# Usage in graph
def create_advanced_processing_graph():
    # Initialize processors
    validator = ValidationProcessor({
        "min_length": 10,
        "max_length": 1000,
        "required_words": ["important"]
    })
    
    processor = TextProcessor({
        "lowercase": True,
        "normalize_whitespace": True,
        "remove_punctuation": False
    })
    
    # Create graph
    graph = StateGraph(ProcessingState)
    
    # Add nodes using class methods
    graph.add_node("validate", validator.process)
    graph.add_node("process", processor.process)
    graph.add_node("count_words", word_counter)
    graph.add_node("analyze", text_analyzer)
    
    # Define flow
    graph.add_edge("validate", "process")
    graph.add_edge("process", "count_words")
    graph.add_edge("count_words", "analyze")
    graph.add_edge("analyze", END)
    
    graph.set_entry_point("validate")
    
    return graph.compile()
```

### 7.2 Advanced Node Patterns

#### Conditional Processing Nodes

```python
from typing import Literal, Union
import random

class ConditionalState(TypedDict):
    input_data: Dict[str, Any]
    processing_path: str
    results: Dict[str, Any]
    confidence_score: float

def intelligent_router(state: ConditionalState) -> Literal["fast_path", "detailed_path", "expert_path"]:
    """Route based on input complexity and requirements"""
    data = state["input_data"]
    
    # Analyze input complexity
    complexity_score = len(str(data))  # Simple complexity measure
    
    if complexity_score < 100:
        return "fast_path"
    elif complexity_score < 500:
        return "detailed_path"
    else:
        return "expert_path"

def fast_processor(state: ConditionalState) -> ConditionalState:
    """Quick processing for simple inputs"""
    results = {
        "processing_type": "fast",
        "processing_time": 0.1,
        "accuracy": 0.85,
        "details": "Quick processing applied"
    }
    
    return {
        **state,
        "processing_path": "fast",
        "results": results,
        "confidence_score": 0.85
    }

def detailed_processor(state: ConditionalState) -> ConditionalState:
    """Detailed processing for medium complexity"""
    results = {
        "processing_type": "detailed",
        "processing_time": 0.5,
        "accuracy": 0.92,
        "details": "Comprehensive analysis performed",
        "additional_metrics": {
            "depth_analysis": True,
            "cross_validation": True
        }
    }
    
    return {
        **state,
        "processing_path": "detailed",
        "results": results,
        "confidence_score": 0.92
    }

def expert_processor(state: ConditionalState) -> ConditionalState:
    """Expert-level processing for complex inputs"""
    results = {
        "processing_type": "expert",
        "processing_time": 1.2,
        "accuracy": 0.97,
        "details": "Expert analysis with multiple validation layers",
        "expert_features": {
            "deep_analysis": True,
            "multi_model_validation": True,
            "uncertainty_quantification": True,
            "explanability_metrics": True
        }
    }
    
    return {
        **state,
        "processing_path": "expert",
        "results": results,
        "confidence_score": 0.97
    }

# Create conditional processing graph
def create_conditional_processing_graph():
    graph = StateGraph(ConditionalState)
    
    # Add processing nodes
    graph.add_node("fast_path", fast_processor)
    graph.add_node("detailed_path", detailed_processor) 
    graph.add_node("expert_path", expert_processor)
    
    # Add conditional routing
    graph.add_conditional_edges(
        "route",
        intelligent_router,
        {
            "fast_path": "fast_path",
            "detailed_path": "detailed_path",
            "expert_path": "expert_path"
        }
    )
    
    # All paths end
    graph.add_edge("fast_path", END)
    graph.add_edge("detailed_path", END)
    graph.add_edge("expert_path", END)
    
    graph.set_entry_point("route")
    
    return graph.compile()
```

---

## 8. Edge Configuration

Edges define the flow of execution between nodes in your LangGraph application. This section covers everything from simple connections to complex conditional routing patterns.

### 8.1 Basic Edge Types

#### Simple Direct Edges

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class SimpleState(TypedDict):
    message: str
    step_count: int

def step_one(state: SimpleState) -> SimpleState:
    return {
        **state,
        "message": f"Step 1: {state['message']}",
        "step_count": state["step_count"] + 1
    }

def step_two(state: SimpleState) -> SimpleState:
    return {
        **state,
        "message": f"Step 2: {state['message']}",
        "step_count": state["step_count"] + 1
    }

# Create graph with direct edges
graph = StateGraph(SimpleState)
graph.add_node("step1", step_one)
graph.add_node("step2", step_two)

# Simple sequential flow
graph.add_edge(START, "step1")
graph.add_edge("step1", "step2")
graph.add_edge("step2", END)

app = graph.compile()
```

#### Fan-out Edges (One-to-Many)

```python
from typing import TypedDict, List

class ParallelState(TypedDict):
    input_data: str
    process_a_result: str
    process_b_result: str
    process_c_result: str
    results: List[str]

def data_splitter(state: ParallelState) -> ParallelState:
    """Split input for parallel processing"""
    return {
        **state,
        "input_data": state["input_data"]
    }

def process_a(state: ParallelState) -> ParallelState:
    """Process A - uppercase transformation"""
    result = state["input_data"].upper()
    return {
        **state,
        "process_a_result": f"A: {result}"
    }

def process_b(state: ParallelState) -> ParallelState:
    """Process B - length analysis"""
    length = len(state["input_data"])
    return {
        **state,
        "process_b_result": f"B: Length={length}"
    }

def process_c(state: ParallelState) -> ParallelState:
    """Process C - word count"""
    words = len(state["input_data"].split())
    return {
        **state,
        "process_c_result": f"C: Words={words}"
    }

def result_aggregator(state: ParallelState) -> ParallelState:
    """Aggregate all results"""
    results = [
        state["process_a_result"],
        state["process_b_result"], 
        state["process_c_result"]
    ]
    return {
        **state,
        "results": results
    }

# Create fan-out graph
graph = StateGraph(ParallelState)

# Add all nodes
graph.add_node("splitter", data_splitter)
graph.add_node("process_a", process_a)
graph.add_node("process_b", process_b) 
graph.add_node("process_c", process_c)
graph.add_node("aggregator", result_aggregator)

# Fan-out pattern: one node to multiple nodes
graph.add_edge(START, "splitter")
graph.add_edge("splitter", "process_a")
graph.add_edge("splitter", "process_b")
graph.add_edge("splitter", "process_c")

# Fan-in pattern: multiple nodes to one node
graph.add_edge("process_a", "aggregator")
graph.add_edge("process_b", "aggregator")
graph.add_edge("process_c", "aggregator")
graph.add_edge("aggregator", END)

app = graph.compile()
```

### 8.2 Conditional Edges

#### Basic Conditional Routing

```python
from typing import TypedDict, Literal

class RoutingState(TypedDict):
    user_input: str
    intent: str
    confidence: float
    response: str

def intent_classifier(state: RoutingState) -> RoutingState:
    """Classify user intent"""
    text = state["user_input"].lower()
    
    if "weather" in text or "temperature" in text:
        intent = "weather"
        confidence = 0.9
    elif "news" in text or "current events" in text:
        intent = "news"
        confidence = 0.85
    elif "help" in text or "support" in text:
        intent = "support"
        confidence = 0.8
    else:
        intent = "general"
        confidence = 0.3
    
    return {
        **state,
        "intent": intent,
        "confidence": confidence
    }

def weather_handler(state: RoutingState) -> RoutingState:
    """Handle weather queries"""
    return {
        **state,
        "response": f"Weather info for: {state['user_input']}"
    }

def news_handler(state: RoutingState) -> RoutingState:
    """Handle news queries"""
    return {
        **state,
        "response": f"Latest news about: {state['user_input']}"
    }

def support_handler(state: RoutingState) -> RoutingState:
    """Handle support queries"""
    return {
        **state,
        "response": f"Support assistance for: {state['user_input']}"
    }

def general_handler(state: RoutingState) -> RoutingState:
    """Handle general queries"""
    return {
        **state,
        "response": f"General response to: {state['user_input']}"
    }

def route_by_intent(state: RoutingState) -> Literal["weather", "news", "support", "general"]:
    """Route based on classified intent"""
    return state["intent"]

# Create conditional routing graph
graph = StateGraph(RoutingState)

# Add nodes
graph.add_node("classifier", intent_classifier)
graph.add_node("weather", weather_handler)
graph.add_node("news", news_handler)
graph.add_node("support", support_handler)
graph.add_node("general", general_handler)

# Start with classification
graph.add_edge(START, "classifier")

# Conditional routing based on intent
graph.add_conditional_edge(
    "classifier",           # Source node
    route_by_intent,        # Routing function
    {
        "weather": "weather",    # Intent -> node mapping
        "news": "news",
        "support": "support",
        "general": "general"
    }
)

# All handlers go to END
graph.add_edge("weather", END)
graph.add_edge("news", END)
graph.add_edge("support", END)
graph.add_edge("general", END)

app = graph.compile()
```

#### Multi-Condition Routing

```python
from typing import TypedDict, Literal, Union

class ComplexRoutingState(TypedDict):
    user_input: str
    user_tier: str  # "premium", "standard", "basic"
    processing_complexity: str  # "simple", "moderate", "complex"
    route_taken: str
    result: str

def analyze_complexity(state: ComplexRoutingState) -> ComplexRoutingState:
    """Analyze processing complexity needed"""
    text = state["user_input"]
    
    if len(text.split()) > 100:
        complexity = "complex"
    elif len(text.split()) > 20:
        complexity = "moderate"  
    else:
        complexity = "simple"
    
    return {
        **state,
        "processing_complexity": complexity
    }

def simple_processor(state: ComplexRoutingState) -> ComplexRoutingState:
    """Handle simple processing"""
    return {
        **state,
        "result": f"Simple processing: {state['user_input'][:50]}...",
        "route_taken": "simple"
    }

def moderate_processor(state: ComplexRoutingState) -> ComplexRoutingState:
    """Handle moderate processing"""
    return {
        **state,
        "result": f"Moderate processing: {len(state['user_input'])} chars analyzed",
        "route_taken": "moderate"
    }

def complex_processor(state: ComplexRoutingState) -> ComplexRoutingState:
    """Handle complex processing"""
    return {
        **state,
        "result": f"Complex processing: {state['user_input'].count('.')} sentences processed",
        "route_taken": "complex"
    }

def premium_complex_processor(state: ComplexRoutingState) -> ComplexRoutingState:
    """Handle premium complex processing with advanced features"""
    word_count = len(state['user_input'].split())
    sentence_count = state['user_input'].count('.')
    return {
        **state,
        "result": f"Premium complex: {word_count} words, {sentence_count} sentences with AI enhancement",
        "route_taken": "premium_complex"
    }

def route_by_complexity_and_tier(state: ComplexRoutingState) -> Literal[
    "simple", "moderate", "complex", "premium_complex"
]:
    """Route based on both complexity and user tier"""
    complexity = state["processing_complexity"]
    tier = state["user_tier"]
    
    # Premium users get special treatment for complex tasks
    if complexity == "complex" and tier == "premium":
        return "premium_complex"
    elif complexity == "complex":
        return "complex"
    elif complexity == "moderate":
        return "moderate"
    else:
        return "simple"

# Create multi-condition routing graph
graph = StateGraph(ComplexRoutingState)

# Add nodes
graph.add_node("analyzer", analyze_complexity)
graph.add_node("simple", simple_processor)
graph.add_node("moderate", moderate_processor)
graph.add_node("complex", complex_processor)
graph.add_node("premium_complex", premium_complex_processor)

# Flow
graph.add_edge(START, "analyzer")
graph.add_conditional_edge(
    "analyzer",
    route_by_complexity_and_tier,
    {
        "simple": "simple",
        "moderate": "moderate", 
        "complex": "complex",
        "premium_complex": "premium_complex"
    }
)

# All processors end the flow
graph.add_edge("simple", END)
graph.add_edge("moderate", END)
graph.add_edge("complex", END)
graph.add_edge("premium_complex", END)

app = graph.compile()
```

### 8.3 Advanced Edge Patterns

#### Dynamic Edge Creation

```python
from typing import TypedDict, List, Dict, Any

class DynamicRoutingState(TypedDict):
    tasks: List[Dict[str, Any]]
    completed_tasks: List[str]
    current_task: Dict[str, Any]
    results: Dict[str, Any]

def task_dispatcher(state: DynamicRoutingState) -> DynamicRoutingState:
    """Dispatch tasks dynamically"""
    if not state["tasks"]:
        return state
        
    # Get next task
    next_task = state["tasks"][0]
    remaining_tasks = state["tasks"][1:]
    
    return {
        **state,
        "current_task": next_task,
        "tasks": remaining_tasks
    }

def task_processor_a(state: DynamicRoutingState) -> DynamicRoutingState:
    """Process type A tasks"""
    task = state["current_task"]
    result = f"Processed A: {task['data']}"
    
    results = {**state["results"]}
    results[task["id"]] = result
    
    completed = state["completed_tasks"] + [task["id"]]
    
    return {
        **state,
        "results": results,
        "completed_tasks": completed
    }

def task_processor_b(state: DynamicRoutingState) -> DynamicRoutingState:
    """Process type B tasks"""
    task = state["current_task"]
    result = f"Processed B: {task['data']}"
    
    results = {**state["results"]}
    results[task["id"]] = result
    
    completed = state["completed_tasks"] + [task["id"]]
    
    return {
        **state,
        "results": results,
        "completed_tasks": completed
    }

def route_by_task_type(state: DynamicRoutingState) -> str:
    """Route based on task type"""
    if not state["current_task"]:
        return "end"
    
    task_type = state["current_task"].get("type", "unknown")
    
    if task_type == "type_a":
        return "processor_a"
    elif task_type == "type_b": 
        return "processor_b"
    else:
        return "end"

def should_continue(state: DynamicRoutingState) -> str:
    """Decide whether to continue processing or end"""
    if state["tasks"]:
        return "dispatcher"  # More tasks to process
    else:
        return "end"  # No more tasks

# Create dynamic routing graph  
graph = StateGraph(DynamicRoutingState)

# Add nodes
graph.add_node("dispatcher", task_dispatcher)
graph.add_node("processor_a", task_processor_a)
graph.add_node("processor_b", task_processor_b)

# Entry point
graph.add_edge(START, "dispatcher")

# Dynamic routing from dispatcher
graph.add_conditional_edge(
    "dispatcher",
    route_by_task_type,
    {
        "processor_a": "processor_a",
        "processor_b": "processor_b",
        "end": END
    }
)

# Continue or end after processing
graph.add_conditional_edge(
    "processor_a",
    should_continue,
    {
        "dispatcher": "dispatcher",
        "end": END
    }
)

graph.add_conditional_edge(
    "processor_b", 
    should_continue,
    {
        "dispatcher": "dispatcher",
        "end": END
    }
)

app = graph.compile()
```

#### Loop and Retry Patterns

```python
from typing import TypedDict, Optional
import random
import time

class RetryState(TypedDict):
    data: str
    attempt_count: int
    max_attempts: int
    success: bool
    error_message: Optional[str]
    result: Optional[str]

def unreliable_processor(state: RetryState) -> RetryState:
    """Simulates an unreliable processing node"""
    attempt = state["attempt_count"] + 1
    
    # Simulate occasional failures (30% failure rate)
    if random.random() < 0.3:
        return {
            **state,
            "attempt_count": attempt,
            "success": False,
            "error_message": f"Processing failed on attempt {attempt}"
        }
    
    # Success case
    return {
        **state,
        "attempt_count": attempt,
        "success": True,
        "error_message": None,
        "result": f"Successfully processed: {state['data']}"
    }

def error_handler(state: RetryState) -> RetryState:
    """Handle processing errors"""
    return {
        **state,
        "error_message": f"Max attempts ({state['max_attempts']}) exceeded"
    }

def should_retry(state: RetryState) -> str:
    """Decide whether to retry, succeed, or fail"""
    if state["success"]:
        return "success"
    elif state["attempt_count"] >= state["max_attempts"]:
        return "max_attempts_reached"
    else:
        return "retry"

# Create retry pattern graph
graph = StateGraph(RetryState)

# Add nodes
graph.add_node("processor", unreliable_processor)
graph.add_node("error_handler", error_handler)

# Start processing
graph.add_edge(START, "processor")

# Conditional retry logic
graph.add_conditional_edge(
    "processor",
    should_retry,
    {
        "success": END,                    # Success - end flow
        "retry": "processor",              # Retry - go back to processor
        "max_attempts_reached": "error_handler"  # Failed - handle error
    }
)

# Error handler ends the flow
graph.add_edge("error_handler", END)

app = graph.compile()
```

### 8.4 Edge Testing and Debugging

#### Testing Edge Conditions

```python
import pytest
from typing import TypedDict

class TestState(TypedDict):
    value: int
    category: str
    result: str

def categorizer(state: TestState) -> TestState:
    """Categorize values"""
    value = state["value"]
    
    if value < 0:
        category = "negative"
    elif value == 0:
        category = "zero"  
    elif value <= 10:
        category = "small_positive"
    else:
        category = "large_positive"
    
    return {
        **state,
        "category": category
    }

def route_by_category(state: TestState) -> str:
    """Route based on category"""
    return state["category"]

def negative_handler(state: TestState) -> TestState:
    return {**state, "result": "Handled negative"}

def zero_handler(state: TestState) -> TestState:
    return {**state, "result": "Handled zero"}

def small_positive_handler(state: TestState) -> TestState:
    return {**state, "result": "Handled small positive"}

def large_positive_handler(state: TestState) -> TestState:
    return {**state, "result": "Handled large positive"}

# Test routing function directly
@pytest.mark.unit
class TestEdgeRouting:
    """Test edge routing logic"""
    
    def test_negative_routing(self):
        """Test routing for negative values"""
        state = {"value": -5, "category": "", "result": ""}
        categorized = categorizer(state)
        route = route_by_category(categorized)
        assert route == "negative"
    
    def test_zero_routing(self):
        """Test routing for zero value"""
        state = {"value": 0, "category": "", "result": ""}
        categorized = categorizer(state)
        route = route_by_category(categorized)
        assert route == "zero"
    
    def test_small_positive_routing(self):
        """Test routing for small positive values"""
        state = {"value": 5, "category": "", "result": ""}
        categorized = categorizer(state)
        route = route_by_category(categorized)
        assert route == "small_positive"
    
    def test_large_positive_routing(self):
        """Test routing for large positive values"""
        state = {"value": 15, "category": "", "result": ""}
        categorized = categorizer(state)
        route = route_by_category(categorized)
        assert route == "large_positive"

# Test complete graph execution
@pytest.mark.integration 
def test_complete_conditional_flow():
    """Test complete conditional flow execution"""
    # Create test graph
    graph = StateGraph(TestState)
    
    graph.add_node("categorizer", categorizer)
    graph.add_node("negative", negative_handler)
    graph.add_node("zero", zero_handler)
    graph.add_node("small_positive", small_positive_handler)
    graph.add_node("large_positive", large_positive_handler)
    
    graph.add_edge(START, "categorizer")
    graph.add_conditional_edge(
        "categorizer",
        route_by_category,
        {
            "negative": "negative",
            "zero": "zero",
            "small_positive": "small_positive",
            "large_positive": "large_positive"
        }
    )
    
    for handler in ["negative", "zero", "small_positive", "large_positive"]:
        graph.add_edge(handler, END)
    
    app = graph.compile()
    
    # Test different routing scenarios
    test_cases = [
        ({"value": -3, "category": "", "result": ""}, "Handled negative"),
        ({"value": 0, "category": "", "result": ""}, "Handled zero"), 
        ({"value": 7, "category": "", "result": ""}, "Handled small positive"),
        ({"value": 20, "category": "", "result": ""}, "Handled large positive"),
    ]
    
    for input_state, expected_result in test_cases:
        result = app.invoke(input_state)
        assert result["result"] == expected_result
```

#### Edge Performance Monitoring

```python
import time
import logging
from functools import wraps
from typing import TypedDict, Callable, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoredState(TypedDict):
    data: str
    processing_path: list
    timing_info: dict

def monitor_edge_performance(func: Callable) -> Callable:
    """Decorator to monitor edge routing performance"""
    @wraps(func)
    def wrapper(state: MonitoredState) -> str:
        start_time = time.time()
        
        try:
            result = func(state)
            execution_time = time.time() - start_time
            
            # Log routing decision
            logger.info(f"Edge routing: {func.__name__} -> {result} ({execution_time:.4f}s)")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Edge routing error in {func.__name__}: {e} ({execution_time:.4f}s)")
            raise
    
    return wrapper

@monitor_edge_performance
def monitored_router(state: MonitoredState) -> str:
    """Example router with performance monitoring"""
    # Simulate routing logic
    time.sleep(0.01)  # Simulate processing time
    
    if len(state["data"]) > 100:
        return "complex_processor"
    else:
        return "simple_processor"

def simple_processor(state: MonitoredState) -> MonitoredState:
    """Simple processing node"""
    return {
        **state,
        "processing_path": state["processing_path"] + ["simple"],
        "timing_info": {
            **state["timing_info"],
            "simple_processor": time.time()
        }
    }

def complex_processor(state: MonitoredState) -> MonitoredState:
    """Complex processing node"""
    return {
        **state,
        "processing_path": state["processing_path"] + ["complex"],
        "timing_info": {
            **state["timing_info"],
            "complex_processor": time.time()
        }
    }
```

### 8.5 Best Practices for Edge Configuration

#### Edge Design Principles

1. **Clear Routing Logic**: Make routing conditions explicit and testable
2. **Fail-Safe Defaults**: Always provide fallback routes for unexpected conditions
3. **Performance Awareness**: Monitor edge routing performance in complex graphs
4. **State Preservation**: Ensure edges don't lose critical state information
5. **Debugging Support**: Include logging and monitoring for routing decisions

#### Common Edge Patterns Summary

```python
"""
Edge Pattern Reference Guide

1. Sequential Flow:
   A → B → C → END
   
2. Fan-Out (Parallel):
   A → B
   A → C  
   A → D
   
3. Fan-In (Aggregation):
   A → D
   B → D
   C → D
   
4. Conditional Routing:
   A → router() → {B, C, D}
   
5. Loop/Retry:
   A → condition() → {A, B, END}
   
6. Multi-Path Conditional:
   A → complex_router() → {B, C, D, E}
   
Use these patterns as building blocks for more complex graphs.
"""
```

---

## 9. Memory and Checkpointing

Memory and checkpointing enable LangGraph applications to persist state across executions, handle interruptions gracefully, and support long-running processes. This section covers everything from basic persistence to advanced recovery patterns.

### 9.1 Understanding Memory Systems

#### Memory Concepts in LangGraph

```python
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from datetime import datetime

class ConversationState(TypedDict):
    messages: List[Dict[str, str]]
    user_id: str
    session_id: str
    context: Dict[str, Any]
    last_updated: str

def add_message_node(state: ConversationState) -> ConversationState:
    """Add a new message to the conversation"""
    # This would typically receive a new message from input
    new_message = {
        "role": "assistant", 
        "content": "This is a response from the system",
        "timestamp": datetime.now().isoformat()
    }
    
    messages = state["messages"] + [new_message]
    
    return {
        **state,
        "messages": messages,
        "last_updated": datetime.now().isoformat()
    }

def update_context_node(state: ConversationState) -> ConversationState:
    """Update conversation context"""
    # Extract context from recent messages
    recent_messages = state["messages"][-5:]  # Last 5 messages
    
    context_update = {
        "message_count": len(state["messages"]),
        "last_message_time": datetime.now().isoformat(),
        "conversation_length": sum(len(msg["content"]) for msg in recent_messages)
    }
    
    return {
        **state,
        "context": {**state["context"], **context_update}
    }

# Create graph with in-memory checkpointer
memory_saver = MemorySaver()

graph = StateGraph(ConversationState)
graph.add_node("add_message", add_message_node)
graph.add_node("update_context", update_context_node)

graph.add_edge(START, "add_message")
graph.add_edge("add_message", "update_context") 
graph.add_edge("update_context", END)

# Compile with memory
app = graph.compile(checkpointer=memory_saver)
```

#### Persistent Memory with SQLite

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Create persistent memory saver
def create_persistent_memory_app():
    """Create app with SQLite-backed memory"""
    
    # Setup SQLite connection
    connection = sqlite3.connect("conversation_memory.db", check_same_thread=False)
    
    # Create SQLite checkpointer
    sqlite_saver = SqliteSaver(connection)
    
    # Use same graph structure as before
    graph = StateGraph(ConversationState)
    graph.add_node("add_message", add_message_node)
    graph.add_node("update_context", update_context_node)
    
    graph.add_edge(START, "add_message")
    graph.add_edge("add_message", "update_context")
    graph.add_edge("update_context", END)
    
    # Compile with persistent memory
    return graph.compile(checkpointer=sqlite_saver)

# Usage with thread-based conversations
def run_persistent_conversation():
    """Example of running persistent conversation"""
    app = create_persistent_memory_app()
    
    # Configuration for this conversation thread
    config = {
        "configurable": {
            "thread_id": "user_123_session_456"
        }
    }
    
    # Initial state
    initial_state = {
        "messages": [
            {"role": "user", "content": "Hello, I want to learn about AI", "timestamp": datetime.now().isoformat()}
        ],
        "user_id": "user_123",
        "session_id": "session_456", 
        "context": {"topic": "ai_learning"},
        "last_updated": datetime.now().isoformat()
    }
    
    # Run conversation - state will be persisted
    result = app.invoke(initial_state, config=config)
    
    print(f"Conversation has {len(result['messages'])} messages")
    print(f"Last updated: {result['last_updated']}")
    
    return result

# Later, resume the same conversation
def resume_conversation():
    """Resume conversation from persisted state"""
    app = create_persistent_memory_app()
    
    # Same config to access the same thread
    config = {
        "configurable": {
            "thread_id": "user_123_session_456"
        }
    }
    
    # Get current state (will load from SQLite)
    current_state = app.get_state(config)
    print(f"Resumed conversation with {len(current_state.values['messages'])} existing messages")
    
    # Continue conversation
    new_message_state = {
        **current_state.values,
        "messages": current_state.values["messages"] + [
            {"role": "user", "content": "Can you tell me more?", "timestamp": datetime.now().isoformat()}
        ]
    }
    
    result = app.invoke(new_message_state, config=config)
    return result
```

### 9.2 Advanced Checkpointing Patterns

#### Custom Memory Implementations

```python
from typing import Any, Dict, Optional
from langgraph.checkpoint.base import Checkpointer, Checkpoint, CheckpointConfig
import json
import hashlib
from pathlib import Path

class FileSystemCheckpointer(Checkpointer):
    """Custom file-based checkpointer"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True, parents=True)
    
    def _get_checkpoint_path(self, config: CheckpointConfig) -> Path:
        """Generate checkpoint file path"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        # Create safe filename from thread_id
        safe_name = hashlib.md5(thread_id.encode()).hexdigest()
        return self.base_path / f"checkpoint_{safe_name}.json"
    
    def put(self, config: CheckpointConfig, checkpoint: Checkpoint) -> None:
        """Save checkpoint to file system"""
        checkpoint_path = self._get_checkpoint_path(config)
        
        checkpoint_data = {
            "id": checkpoint["id"],
            "values": checkpoint["values"],
            "next": checkpoint.get("next", []),
            "timestamp": datetime.now().isoformat(),
            "config": config
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
    
    def get(self, config: CheckpointConfig) -> Optional[Checkpoint]:
        """Load checkpoint from file system"""
        checkpoint_path = self._get_checkpoint_path(config)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            
            return {
                "id": data["id"],
                "values": data["values"],
                "next": data.get("next", [])
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def list(self, config: CheckpointConfig) -> List[Checkpoint]:
        """List all checkpoints (simplified implementation)"""
        checkpoint = self.get(config)
        return [checkpoint] if checkpoint else []

# Usage with custom checkpointer
def create_app_with_file_checkpointer():
    """Create app with file-based checkpointer"""
    file_checkpointer = FileSystemCheckpointer("./checkpoints")
    
    graph = StateGraph(ConversationState)
    graph.add_node("add_message", add_message_node)
    graph.add_node("update_context", update_context_node)
    
    graph.add_edge(START, "add_message")
    graph.add_edge("add_message", "update_context")
    graph.add_edge("update_context", END)
    
    return graph.compile(checkpointer=file_checkpointer)
```

#### Distributed Memory Systems

```python
import redis
import pickle
from typing import Any, Dict, Optional, List

class RedisCheckpointer(Checkpointer):
    """Redis-based distributed checkpointer"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "langgraph"):
        self.redis_client = redis.from_url(redis_url)
        self.prefix = prefix
    
    def _get_key(self, config: CheckpointConfig) -> str:
        """Generate Redis key"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        return f"{self.prefix}:checkpoint:{thread_id}"
    
    def put(self, config: CheckpointConfig, checkpoint: Checkpoint) -> None:
        """Save checkpoint to Redis"""
        key = self._get_key(config)
        
        # Serialize checkpoint
        checkpoint_data = {
            "id": checkpoint["id"],
            "values": checkpoint["values"],
            "next": checkpoint.get("next", []),
            "timestamp": datetime.now().isoformat()
        }
        
        serialized = pickle.dumps(checkpoint_data)
        
        # Store with expiration (24 hours)
        self.redis_client.setex(key, 86400, serialized)
    
    def get(self, config: CheckpointConfig) -> Optional[Checkpoint]:
        """Load checkpoint from Redis"""
        key = self._get_key(config)
        
        try:
            serialized = self.redis_client.get(key)
            if not serialized:
                return None
            
            data = pickle.loads(serialized)
            
            return {
                "id": data["id"],
                "values": data["values"], 
                "next": data.get("next", [])
            }
        except (pickle.PickleError, redis.RedisError) as e:
            print(f"Error loading from Redis: {e}")
            return None
    
    def list(self, config: CheckpointConfig) -> List[Checkpoint]:
        """List checkpoints (simplified)"""
        checkpoint = self.get(config)
        return [checkpoint] if checkpoint else []

# High-availability memory setup
def create_ha_memory_app():
    """Create app with high-availability memory"""
    
    # Primary Redis instance
    primary_checkpointer = RedisCheckpointer("redis://primary:6379", "primary")
    
    # Could implement failover logic here
    
    graph = StateGraph(ConversationState)
    graph.add_node("add_message", add_message_node)
    graph.add_node("update_context", update_context_node)
    
    graph.add_edge(START, "add_message")
    graph.add_edge("add_message", "update_context")
    graph.add_edge("update_context", END)
    
    return graph.compile(checkpointer=primary_checkpointer)
```

### 9.3 State Recovery and Interruption Handling

#### Graceful Interruption Handling

```python
from typing import TypedDict, List, Dict, Any
import time
import signal
import sys
from langgraph.graph import StateGraph, START, END, interrupt

class InterruptibleState(TypedDict):
    tasks: List[str]
    completed_tasks: List[str]
    current_task: str
    progress: Dict[str, Any]
    interrupted: bool

def long_running_processor(state: InterruptibleState) -> InterruptibleState:
    """Simulate long-running process that can be interrupted"""
    
    if not state["tasks"]:
        return {**state, "current_task": ""}
    
    current_task = state["tasks"][0]
    remaining_tasks = state["tasks"][1:]
    
    # Update state to show current task
    updated_state = {
        **state,
        "current_task": current_task,
        "tasks": remaining_tasks
    }
    
    # Simulate work with checkpoints
    for i in range(5):  # 5 steps of work
        time.sleep(1)  # Simulate processing time
        
        # Check for interruption signal
        if updated_state.get("interrupted", False):
            # Save progress and interrupt
            updated_state["progress"][current_task] = f"Step {i+1}/5"
            interrupt(f"Processing interrupted at step {i+1} of task: {current_task}")
        
        # Update progress
        updated_state["progress"][current_task] = f"Step {i+1}/5"
    
    # Task completed
    completed = updated_state["completed_tasks"] + [current_task]
    updated_state["completed_tasks"] = completed
    updated_state["current_task"] = ""
    
    return updated_state

def should_continue_processing(state: InterruptibleState) -> str:
    """Decide whether to continue or stop"""
    if state["tasks"]:
        return "continue"
    else:
        return "complete"

# Create interruptible graph
def create_interruptible_app():
    """Create app that can handle interruptions gracefully"""
    
    graph = StateGraph(InterruptibleState)
    graph.add_node("processor", long_running_processor)
    
    graph.add_edge(START, "processor")
    graph.add_conditional_edge(
        "processor",
        should_continue_processing,
        {
            "continue": "processor",  # Loop back for more tasks
            "complete": END
        }
    )
    
    # Use SQLite for persistence
    conn = sqlite3.connect("interruption_recovery.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    return graph.compile(checkpointer=checkpointer)

# Interruption recovery example
def handle_interruption_recovery():
    """Example of handling and recovering from interruptions"""
    app = create_interruptible_app()
    
    config = {"configurable": {"thread_id": "long_process_123"}}
    
    initial_state = {
        "tasks": ["task_1", "task_2", "task_3", "task_4"],
        "completed_tasks": [],
        "current_task": "",
        "progress": {},
        "interrupted": False
    }
    
    try:
        # Start processing
        result = app.invoke(initial_state, config=config)
        print(f"Completed all tasks: {result['completed_tasks']}")
        
    except Exception as e:
        print(f"Process was interrupted: {e}")
        
        # Later, resume from checkpoint
        print("Resuming from checkpoint...")
        
        # Get current state
        current_state = app.get_state(config)
        
        if current_state:
            print(f"Resuming with {len(current_state.values['tasks'])} remaining tasks")
            print(f"Progress: {current_state.values['progress']}")
            
            # Resume processing
            resumed_result = app.invoke(current_state.values, config=config)
            print(f"Finally completed: {resumed_result['completed_tasks']}")
```

### 9.4 Memory Optimization and Management

#### Memory Cleanup and Optimization

```python
from datetime import datetime, timedelta
import sqlite3
from typing import List, Dict, Any

class OptimizedMemoryManager:
    """Manager for memory optimization and cleanup"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
    
    def cleanup_old_checkpoints(self, max_age_days: int = 7) -> int:
        """Clean up checkpoints older than max_age_days"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM checkpoints WHERE created_at < ?", 
            (cutoff_date.isoformat(),)
        )
        
        deleted_count = cursor.rowcount
        self.conn.commit()
        
        print(f"Cleaned up {deleted_count} old checkpoints")
        return deleted_count
    
    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        cursor = self.conn.cursor()
        
        # Total checkpoints
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        total_checkpoints = cursor.fetchone()[0]
        
        # Database size
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        db_size = cursor.fetchone()[0]
        
        # Active threads
        cursor.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
        active_threads = cursor.fetchone()[0]
        
        return {
            "total_checkpoints": total_checkpoints,
            "database_size_bytes": db_size,
            "active_threads": active_threads,
            "size_mb": round(db_size / (1024 * 1024), 2)
        }
    
    def compress_checkpoint_history(self, thread_id: str, keep_last_n: int = 10) -> None:
        """Keep only the last N checkpoints for a thread"""
        cursor = self.conn.cursor()
        
        # Get checkpoint IDs to delete (all but the last keep_last_n)
        cursor.execute("""
            SELECT checkpoint_id FROM checkpoints 
            WHERE thread_id = ? 
            ORDER BY created_at DESC 
            LIMIT -1 OFFSET ?
        """, (thread_id, keep_last_n))
        
        old_checkpoints = cursor.fetchall()
        
        if old_checkpoints:
            checkpoint_ids = [cp[0] for cp in old_checkpoints]
            placeholders = ','.join(['?' for _ in checkpoint_ids])
            
            cursor.execute(
                f"DELETE FROM checkpoints WHERE checkpoint_id IN ({placeholders})",
                checkpoint_ids
            )
            
            self.conn.commit()
            print(f"Compressed {len(checkpoint_ids)} old checkpoints for thread {thread_id}")

# Memory-efficient state design
class MemoryEfficientState(TypedDict):
    """State designed for memory efficiency"""
    
    # Core data - always keep
    essential_data: Dict[str, Any]
    
    # Temporary data - can be cleared
    temp_processing_data: Dict[str, Any]
    
    # Cached data - can be recomputed
    cached_results: Dict[str, Any]
    
    # Metadata for memory management
    last_accessed: str
    memory_level: str  # "minimal", "standard", "full"

def memory_cleanup_node(state: MemoryEfficientState) -> MemoryEfficientState:
    """Clean up memory based on access patterns"""
    
    memory_level = state.get("memory_level", "standard")
    
    if memory_level == "minimal":
        # Keep only essential data
        return {
            "essential_data": state["essential_data"],
            "temp_processing_data": {},
            "cached_results": {},
            "last_accessed": datetime.now().isoformat(),
            "memory_level": "minimal"
        }
    elif memory_level == "standard":
        # Keep essential + some cached data
        return {
            **state,
            "temp_processing_data": {},  # Clear temp data
            "last_accessed": datetime.now().isoformat()
        }
    else:
        # Keep everything (full mode)
        return {
            **state,
            "last_accessed": datetime.now().isoformat()
        }
```

### 9.5 Testing Memory and Checkpointing

#### Memory System Testing

```python
import pytest
import tempfile
import shutil
from pathlib import Path

class TestMemorySystem:
    """Test memory and checkpointing functionality"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        yield str(db_path)
        shutil.rmtree(temp_dir)
    
    def test_basic_checkpointing(self, temp_db_path):
        """Test basic checkpoint save and restore"""
        
        # Create app with SQLite checkpointer
        conn = sqlite3.connect(temp_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        
        graph = StateGraph(ConversationState)
        graph.add_node("add_message", add_message_node)
        graph.add_edge(START, "add_message")
        graph.add_edge("add_message", END)
        
        app = graph.compile(checkpointer=checkpointer)
        
        # Test configuration
        config = {"configurable": {"thread_id": "test_123"}}
        
        # Initial state
        initial_state = {
            "messages": [{"role": "user", "content": "Hello"}],
            "user_id": "test_user",
            "session_id": "test_session",
            "context": {},
            "last_updated": datetime.now().isoformat()
        }
        
        # Run and checkpoint
        result = app.invoke(initial_state, config=config)
        
        # Verify checkpoint exists
        saved_state = app.get_state(config)
        assert saved_state is not None
        assert len(saved_state.values["messages"]) >= 1
    
    def test_checkpoint_recovery(self, temp_db_path):
        """Test recovery from checkpoint"""
        
        conn = sqlite3.connect(temp_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        
        # Create and run app first time
        app1 = create_test_app(checkpointer)
        config = {"configurable": {"thread_id": "recovery_test"}}
        
        initial_state = {
            "messages": [{"role": "user", "content": "First message"}],
            "user_id": "test_user",
            "session_id": "test_session", 
            "context": {"step": 1},
            "last_updated": datetime.now().isoformat()
        }
        
        result1 = app1.invoke(initial_state, config=config)
        
        # Create new app instance (simulates restart)
        app2 = create_test_app(checkpointer)
        
        # Get state from checkpoint
        recovered_state = app2.get_state(config)
        
        assert recovered_state is not None
        assert recovered_state.values["context"]["step"] == 1
        assert len(recovered_state.values["messages"]) > 0
    
    def test_memory_cleanup(self, temp_db_path):
        """Test memory cleanup functionality"""
        
        memory_manager = OptimizedMemoryManager(temp_db_path)
        
        # Create some test checkpoints
        conn = sqlite3.connect(temp_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        
        app = create_test_app(checkpointer)
        
        # Create multiple checkpoint versions
        for i in range(5):
            config = {"configurable": {"thread_id": f"cleanup_test_{i}"}}
            state = {
                "messages": [{"role": "user", "content": f"Message {i}"}],
                "user_id": "test_user",
                "session_id": "test_session",
                "context": {"iteration": i},
                "last_updated": datetime.now().isoformat()
            }
            app.invoke(state, config=config)
        
        # Check initial stats
        stats_before = memory_manager.get_memory_usage_stats()
        assert stats_before["total_checkpoints"] >= 5
        
        # Test cleanup (won't delete much since they're recent)
        deleted = memory_manager.cleanup_old_checkpoints(max_age_days=0)
        
        # Verify cleanup worked
        stats_after = memory_manager.get_memory_usage_stats()
        assert stats_after["total_checkpoints"] <= stats_before["total_checkpoints"]

def create_test_app(checkpointer):
    """Helper to create test app"""
    graph = StateGraph(ConversationState)
    graph.add_node("add_message", add_message_node)
    graph.add_node("update_context", update_context_node)
    
    graph.add_edge(START, "add_message")
    graph.add_edge("add_message", "update_context")
    graph.add_edge("update_context", END)
    
    return graph.compile(checkpointer=checkpointer)
```

### 9.6 Best Practices for Memory Management

#### Memory Design Principles

1. **Selective Persistence**: Only checkpoint essential state data
2. **Cleanup Strategies**: Implement regular cleanup of old checkpoints
3. **Recovery Planning**: Design clear recovery paths for interrupted processes
4. **Performance Monitoring**: Monitor memory usage and checkpoint performance
5. **Data Lifecycle**: Define clear policies for data retention and archival

#### Common Memory Patterns

```python
"""
Memory Pattern Reference Guide

1. Session-Based Memory:
   - Use thread_id for user sessions
   - Automatic cleanup after session timeout
   
2. Process-Based Memory:
   - Checkpoint at major process milestones
   - Enable resumption from interruptions
   
3. Distributed Memory:
   - Use Redis/external systems for multi-instance apps
   - Implement failover and backup strategies
   
4. Hierarchical Memory:
   - Hot: In-memory for active sessions
   - Warm: SQLite for recent sessions  
   - Cold: Archive for historical data

Choose patterns based on your application's scale and reliability needs.
"""
```

---

## 10. Multi-Agent Systems

Multi-agent systems in LangGraph enable complex workflows where multiple specialized agents collaborate to solve problems. This section covers agent design patterns, coordination mechanisms, and advanced multi-agent architectures.

### 10.1 Basic Multi-Agent Architecture

#### Simple Agent Coordination

```python
from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from datetime import datetime

class MultiAgentState(TypedDict):
    user_query: str
    research_results: Dict[str, Any]
    analysis_results: Dict[str, Any]
    final_report: str
    agent_logs: List[Dict[str, Any]]
    current_agent: str

def research_agent(state: MultiAgentState) -> MultiAgentState:
    """Research agent - gathers information"""
    query = state["user_query"]
    
    # Simulate research process
    research_data = {
        "sources_found": 5,
        "key_facts": [
            f"Fact 1 about {query}",
            f"Fact 2 about {query}",
            f"Fact 3 about {query}"
        ],
        "confidence": 0.85,
        "research_time": datetime.now().isoformat()
    }
    
    # Log agent activity
    log_entry = {
        "agent": "research_agent",
        "action": "research_completed", 
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }
    
    return {
        **state,
        "research_results": research_data,
        "agent_logs": state["agent_logs"] + [log_entry],
        "current_agent": "research_agent"
    }

def analysis_agent(state: MultiAgentState) -> MultiAgentState:
    """Analysis agent - processes research data"""
    research = state["research_results"]
    
    # Analyze the research data
    analysis = {
        "summary": f"Analysis of {len(research['key_facts'])} key facts",
        "confidence_score": research["confidence"],
        "recommendations": [
            "Recommendation 1 based on research",
            "Recommendation 2 based on analysis",
        ],
        "analysis_time": datetime.now().isoformat()
    }
    
    # Log agent activity
    log_entry = {
        "agent": "analysis_agent",
        "action": "analysis_completed",
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }
    
    return {
        **state,
        "analysis_results": analysis,
        "agent_logs": state["agent_logs"] + [log_entry],
        "current_agent": "analysis_agent"
    }

def report_agent(state: MultiAgentState) -> MultiAgentState:
    """Report agent - creates final output"""
    research = state["research_results"]
    analysis = state["analysis_results"]
    
    # Generate comprehensive report
    report = f"""
    Research & Analysis Report
    =========================
    
    Query: {state['user_query']}
    
    Research Summary:
    - Sources found: {research['sources_found']}
    - Key findings: {len(research['key_facts'])} facts
    - Research confidence: {research['confidence']:.2f}
    
    Analysis Summary:
    - {analysis['summary']}
    - Confidence score: {analysis['confidence_score']:.2f}
    - Recommendations: {len(analysis['recommendations'])}
    
    Final Recommendations:
    {chr(10).join(f"- {rec}" for rec in analysis['recommendations'])}
    
    Report generated at: {datetime.now().isoformat()}
    """
    
    # Log agent activity
    log_entry = {
        "agent": "report_agent",
        "action": "report_generated",
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }
    
    return {
        **state,
        "final_report": report,
        "agent_logs": state["agent_logs"] + [log_entry],
        "current_agent": "report_agent"
    }

# Create multi-agent workflow
def create_basic_multi_agent_system():
    """Create basic multi-agent system"""
    
    graph = StateGraph(MultiAgentState)
    
    # Add agent nodes
    graph.add_node("researcher", research_agent)
    graph.add_node("analyzer", analysis_agent) 
    graph.add_node("reporter", report_agent)
    
    # Sequential workflow
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "analyzer")
    graph.add_edge("analyzer", "reporter")
    graph.add_edge("reporter", END)
    
    return graph.compile()

# Usage example
def run_multi_agent_research():
    """Example of running multi-agent research"""
    app = create_basic_multi_agent_system()
    
    initial_state = {
        "user_query": "What are the benefits of renewable energy?",
        "research_results": {},
        "analysis_results": {},
        "final_report": "",
        "agent_logs": [],
        "current_agent": ""
    }
    
    result = app.invoke(initial_state)
    
    print("=== AGENT EXECUTION LOG ===")
    for log in result["agent_logs"]:
        print(f"{log['timestamp']}: {log['agent']} - {log['action']} ({log['status']})")
    
    print("\n=== FINAL REPORT ===")
    print(result["final_report"])
    
    return result
```

### 10.2 Advanced Agent Coordination Patterns

#### Supervisor-Worker Pattern

```python
from typing import TypedDict, List, Dict, Any, Literal

class SupervisorState(TypedDict):
    user_request: str
    task_assignments: List[Dict[str, Any]]
    worker_results: Dict[str, Dict[str, Any]]
    supervisor_decision: str
    final_output: str
    coordination_log: List[str]

def supervisor_agent(state: SupervisorState) -> SupervisorState:
    """Supervisor agent - coordinates and assigns tasks"""
    request = state["user_request"]
    
    # Analyze request and create task assignments
    if "data analysis" in request.lower():
        assignments = [
            {"worker": "data_worker", "task": "process_data", "priority": 1},
            {"worker": "viz_worker", "task": "create_visualizations", "priority": 2}
        ]
        decision = "data_analysis_workflow"
    elif "content creation" in request.lower():
        assignments = [
            {"worker": "research_worker", "task": "gather_information", "priority": 1},
            {"worker": "writing_worker", "task": "create_content", "priority": 2}
        ]
        decision = "content_creation_workflow"
    else:
        assignments = [
            {"worker": "general_worker", "task": "general_processing", "priority": 1}
        ]
        decision = "general_workflow"
    
    coordination_log = state["coordination_log"] + [
        f"Supervisor assigned {len(assignments)} tasks for {decision}"
    ]
    
    return {
        **state,
        "task_assignments": assignments,
        "supervisor_decision": decision,
        "coordination_log": coordination_log
    }

def data_worker_agent(state: SupervisorState) -> SupervisorState:
    """Specialized data processing worker"""
    
    # Find relevant task assignment
    task = next((t for t in state["task_assignments"] if t["worker"] == "data_worker"), None)
    if not task:
        return state
    
    # Perform data processing work
    result = {
        "task_completed": task["task"],
        "data_processed": True,
        "records_processed": 1500,
        "processing_time": "2.3s",
        "status": "completed"
    }
    
    # Update results
    worker_results = {**state["worker_results"]}
    worker_results["data_worker"] = result
    
    coordination_log = state["coordination_log"] + [
        "Data worker completed data processing task"
    ]
    
    return {
        **state,
        "worker_results": worker_results,
        "coordination_log": coordination_log
    }

def viz_worker_agent(state: SupervisorState) -> SupervisorState:
    """Specialized visualization worker"""
    
    task = next((t for t in state["task_assignments"] if t["worker"] == "viz_worker"), None)
    if not task:
        return state
    
    # Check if data worker completed (dependency)
    if "data_worker" not in state["worker_results"]:
        return state  # Wait for dependency
    
    # Create visualizations
    result = {
        "task_completed": task["task"],
        "charts_created": ["bar_chart", "line_chart", "pie_chart"],
        "visualization_count": 3,
        "status": "completed"
    }
    
    worker_results = {**state["worker_results"]}
    worker_results["viz_worker"] = result
    
    coordination_log = state["coordination_log"] + [
        "Visualization worker completed chart creation"
    ]
    
    return {
        **state,
        "worker_results": worker_results,
        "coordination_log": coordination_log
    }

def final_coordinator(state: SupervisorState) -> SupervisorState:
    """Final coordinator - combines all worker results"""
    
    # Combine results from all workers
    output_parts = []
    
    for worker, result in state["worker_results"].items():
        output_parts.append(f"{worker}: {result['task_completed']} - {result['status']}")
    
    final_output = f"""
    Task Coordination Results
    ========================
    
    Original Request: {state['user_request']}
    Supervisor Decision: {state['supervisor_decision']}
    
    Worker Results:
    {chr(10).join(f"- {part}" for part in output_parts)}
    
    Coordination Summary:
    {chr(10).join(f"- {log}" for log in state['coordination_log'])}
    """
    
    return {
        **state,
        "final_output": final_output
    }

def route_to_workers(state: SupervisorState) -> List[str]:
    """Route to appropriate workers based on assignments"""
    workers = [task["worker"] for task in state["task_assignments"]]
    return workers

# Create supervisor-worker system
def create_supervisor_worker_system():
    """Create supervisor-worker multi-agent system"""
    
    graph = StateGraph(SupervisorState)
    
    # Add agents
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("data_worker", data_worker_agent)
    graph.add_node("viz_worker", viz_worker_agent)
    graph.add_node("coordinator", final_coordinator)
    
    # Workflow
    graph.add_edge(START, "supervisor")
    
    # Dynamic routing to workers
    graph.add_conditional_edge(
        "supervisor",
        lambda state: "data_analysis" if state["supervisor_decision"] == "data_analysis_workflow" else "general",
        {
            "data_analysis": "data_worker",
            "general": "coordinator"
        }
    )
    
    graph.add_edge("data_worker", "viz_worker")
    graph.add_edge("viz_worker", "coordinator")
    graph.add_edge("coordinator", END)
    
    return graph.compile()
```

### 10.3 Agent Communication and Coordination

#### Message-Based Agent Communication

```python
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str
    message_type: str
    content: Dict[str, Any]
    timestamp: str
    priority: int = 1  # 1=high, 2=medium, 3=low

class CommunicatingAgentState(TypedDict):
    user_task: str
    agent_messages: List[AgentMessage]
    agent_states: Dict[str, Dict[str, Any]]
    task_progress: Dict[str, str]
    final_result: str

def message_broker(state: CommunicatingAgentState) -> CommunicatingAgentState:
    """Central message broker for agent communication"""
    
    # Sort messages by priority and timestamp
    messages = state["agent_messages"]
    unprocessed_messages = [msg for msg in messages if msg not in state.get("processed_messages", [])]
    
    # Process highest priority messages first
    unprocessed_messages.sort(key=lambda x: (x.priority, x.timestamp))
    
    # Log message routing
    routing_log = []
    for msg in unprocessed_messages[:5]:  # Process top 5 messages
        routing_log.append(f"Routing {msg.message_type} from {msg.from_agent} to {msg.to_agent}")
    
    return {
        **state,
        "routing_log": routing_log
    }

def planning_agent(state: CommunicatingAgentState) -> CommunicatingAgentState:
    """Planning agent - creates task plans and coordinates"""
    
    task = state["user_task"]
    
    # Create task plan
    plan = {
        "task_breakdown": [
            "research_phase",
            "analysis_phase", 
            "implementation_phase",
            "review_phase"
        ],
        "agent_assignments": {
            "research_phase": "research_agent",
            "analysis_phase": "analysis_agent",
            "implementation_phase": "implementation_agent",
            "review_phase": "review_agent"
        },
        "dependencies": {
            "analysis_phase": ["research_phase"],
            "implementation_phase": ["analysis_phase"],
            "review_phase": ["implementation_phase"]
        }
    }
    
    # Send planning messages to other agents
    planning_messages = [
        AgentMessage(
            from_agent="planning_agent",
            to_agent="research_agent",
            message_type="task_assignment",
            content={"phase": "research_phase", "task": task},
            timestamp=datetime.now().isoformat(),
            priority=1
        ),
        AgentMessage(
            from_agent="planning_agent",
            to_agent="coordination_agent",
            message_type="plan_update",
            content={"plan": plan},
            timestamp=datetime.now().isoformat(),
            priority=1
        )
    ]
    
    # Update agent state
    agent_states = {**state["agent_states"]}
    agent_states["planning_agent"] = {
        "status": "plan_created",
        "plan": plan,
        "last_updated": datetime.now().isoformat()
    }
    
    return {
        **state,
        "agent_messages": state["agent_messages"] + planning_messages,
        "agent_states": agent_states,
        "task_progress": {**state["task_progress"], "planning": "completed"}
    }

def research_agent_communicating(state: CommunicatingAgentState) -> CommunicatingAgentState:
    """Research agent with communication capabilities"""
    
    # Check for messages addressed to this agent
    my_messages = [msg for msg in state["agent_messages"] 
                   if msg.to_agent == "research_agent" and msg.message_type == "task_assignment"]
    
    if not my_messages:
        return state  # No work assigned yet
    
    latest_assignment = my_messages[-1]  # Get latest assignment
    
    # Perform research
    research_results = {
        "sources_found": 8,
        "research_summary": f"Research completed for: {latest_assignment.content['task']}",
        "confidence": 0.9,
        "completion_time": datetime.now().isoformat()
    }
    
    # Send results to analysis agent
    result_message = AgentMessage(
        from_agent="research_agent",
        to_agent="analysis_agent",
        message_type="research_results",
        content=research_results,
        timestamp=datetime.now().isoformat(),
        priority=1
    )
    
    # Send status update to planning agent
    status_message = AgentMessage(
        from_agent="research_agent",
        to_agent="planning_agent", 
        message_type="status_update",
        content={"phase": "research_phase", "status": "completed"},
        timestamp=datetime.now().isoformat(),
        priority=2
    )
    
    # Update own state
    agent_states = {**state["agent_states"]}
    agent_states["research_agent"] = {
        "status": "research_completed",
        "results": research_results,
        "last_updated": datetime.now().isoformat()
    }
    
    return {
        **state,
        "agent_messages": state["agent_messages"] + [result_message, status_message],
        "agent_states": agent_states,
        "task_progress": {**state["task_progress"], "research": "completed"}
    }

def analysis_agent_communicating(state: CommunicatingAgentState) -> CommunicatingAgentState:
    """Analysis agent with communication capabilities"""
    
    # Check for research results
    research_messages = [msg for msg in state["agent_messages"]
                        if msg.to_agent == "analysis_agent" and msg.message_type == "research_results"]
    
    if not research_messages:
        return state  # Waiting for research results
    
    latest_research = research_messages[-1]
    research_data = latest_research.content
    
    # Perform analysis
    analysis_results = {
        "analysis_type": "comprehensive",
        "insights": [
            "Key insight 1 from research",
            "Key insight 2 from research",
            "Key insight 3 from research"
        ],
        "recommendations": [
            "Recommendation based on analysis",
            "Strategic suggestion from insights"
        ],
        "confidence": research_data["confidence"] * 0.95,  # Slightly lower due to analysis uncertainty
        "completion_time": datetime.now().isoformat()
    }
    
    # Send results to implementation agent
    analysis_message = AgentMessage(
        from_agent="analysis_agent",
        to_agent="implementation_agent",
        message_type="analysis_results",
        content=analysis_results,
        timestamp=datetime.now().isoformat(),
        priority=1
    )
    
    # Update state
    agent_states = {**state["agent_states"]}
    agent_states["analysis_agent"] = {
        "status": "analysis_completed",
        "results": analysis_results,
        "last_updated": datetime.now().isoformat()
    }
    
    return {
        **state,
        "agent_messages": state["agent_messages"] + [analysis_message],
        "agent_states": agent_states,
        "task_progress": {**state["task_progress"], "analysis": "completed"}
    }
```

### 10.4 Specialized Agent Patterns

#### Expert Agent Network

```python
from typing import TypedDict, List, Dict, Any, Set
from enum import Enum

class ExpertiseArea(Enum):
    DATA_SCIENCE = "data_science"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USER_EXPERIENCE = "user_experience"
    ARCHITECTURE = "architecture"

class ExpertNetworkState(TypedDict):
    problem_statement: str
    required_expertise: List[str]
    expert_opinions: Dict[str, Dict[str, Any]]
    consensus_report: Dict[str, Any]
    confidence_scores: Dict[str, float]
    expert_recommendations: List[Dict[str, Any]]

class ExpertAgent:
    """Base class for expert agents"""
    
    def __init__(self, expertise_area: ExpertiseArea, confidence_threshold: float = 0.7):
        self.expertise = expertise_area
        self.confidence_threshold = confidence_threshold
        self.name = f"{expertise_area.value}_expert"
    
    def can_handle_problem(self, problem_keywords: Set[str]) -> bool:
        """Determine if this expert can handle the problem"""
        expertise_keywords = {
            ExpertiseArea.DATA_SCIENCE: {"data", "analysis", "machine learning", "statistics"},
            ExpertiseArea.SECURITY: {"security", "vulnerability", "encryption", "authentication"},
            ExpertiseArea.PERFORMANCE: {"performance", "optimization", "speed", "efficiency"},
            ExpertiseArea.USER_EXPERIENCE: {"ux", "ui", "user", "interface", "usability"},
            ExpertiseArea.ARCHITECTURE: {"architecture", "design", "scalability", "system"}
        }
        
        return bool(problem_keywords.intersection(expertise_keywords.get(self.expertise, set())))
    
    def provide_expert_opinion(self, problem: str) -> Dict[str, Any]:
        """Provide expert opinion on the problem"""
        # This would be implemented differently for each expert
        return {
            "expert": self.name,
            "opinion": f"Expert opinion from {self.expertise.value}",
            "confidence": 0.8,
            "recommendations": [],
            "concerns": []
        }

def data_science_expert_agent(state: ExpertNetworkState) -> ExpertNetworkState:
    """Data science expert agent"""
    problem = state["problem_statement"]
    
    # Analyze problem from data science perspective
    if any(keyword in problem.lower() for keyword in ["data", "analysis", "model", "prediction"]):
        opinion = {
            "expert": "data_science_expert",
            "analysis": {
                "data_requirements": "Large dataset needed for accurate modeling",
                "recommended_approaches": [
                    "Feature engineering and selection",
                    "Cross-validation for model evaluation",
                    "Ensemble methods for improved accuracy"
                ],
                "potential_challenges": [
                    "Data quality and completeness",
                    "Overfitting with small datasets",
                    "Feature drift over time"
                ],
                "confidence": 0.85
            },
            "recommendations": [
                "Implement robust data validation pipeline",
                "Use A/B testing for model deployment",
                "Monitor model performance continuously"
            ]
        }
        
        # Update expert opinions
        expert_opinions = {**state["expert_opinions"]}
        expert_opinions["data_science_expert"] = opinion
        
        confidence_scores = {**state["confidence_scores"]}
        confidence_scores["data_science_expert"] = opinion["analysis"]["confidence"]
        
        return {
            **state,
            "expert_opinions": expert_opinions,
            "confidence_scores": confidence_scores
        }
    
    return state  # This expert doesn't handle this type of problem

def security_expert_agent(state: ExpertNetworkState) -> ExpertNetworkState:
    """Security expert agent"""
    problem = state["problem_statement"]
    
    if any(keyword in problem.lower() for keyword in ["security", "auth", "vulnerability", "attack"]):
        opinion = {
            "expert": "security_expert",
            "analysis": {
                "security_assessment": "High priority security considerations identified",
                "threat_vectors": [
                    "Authentication bypass",
                    "Data injection attacks",
                    "Privilege escalation"
                ],
                "compliance_requirements": [
                    "GDPR data protection",
                    "SOC 2 compliance",
                    "Industry-specific regulations"
                ],
                "confidence": 0.92
            },
            "recommendations": [
                "Implement multi-factor authentication",
                "Regular security audits and penetration testing",
                "Zero-trust architecture principles"
            ]
        }
        
        expert_opinions = {**state["expert_opinions"]}
        expert_opinions["security_expert"] = opinion
        
        confidence_scores = {**state["confidence_scores"]}
        confidence_scores["security_expert"] = opinion["analysis"]["confidence"]
        
        return {
            **state,
            "expert_opinions": expert_opinions,
            "confidence_scores": confidence_scores
        }
    
    return state

def consensus_coordinator_agent(state: ExpertNetworkState) -> ExpertNetworkState:
    """Coordinator that builds consensus from expert opinions"""
    
    expert_opinions = state["expert_opinions"]
    
    if len(expert_opinions) < 2:
        return state  # Wait for more expert opinions
    
    # Analyze expert consensus
    all_recommendations = []
    all_concerns = []
    confidence_sum = 0
    expert_count = 0
    
    for expert_name, opinion in expert_opinions.items():
        if "recommendations" in opinion:
            all_recommendations.extend(opinion["recommendations"])
        if "analysis" in opinion:
            analysis = opinion["analysis"]
            if "potential_challenges" in analysis:
                all_concerns.extend(analysis["potential_challenges"])
            if "threat_vectors" in analysis:
                all_concerns.extend(analysis["threat_vectors"])
        
        # Get confidence score
        confidence = state["confidence_scores"].get(expert_name, 0)
        confidence_sum += confidence
        expert_count += 1
    
    # Build consensus report
    consensus = {
        "participating_experts": list(expert_opinions.keys()),
        "average_confidence": confidence_sum / expert_count if expert_count > 0 else 0,
        "consolidated_recommendations": list(set(all_recommendations)),  # Remove duplicates
        "identified_concerns": list(set(all_concerns)),
        "consensus_level": "high" if confidence_sum / expert_count > 0.8 else "medium",
        "report_generated_at": datetime.now().isoformat()
    }
    
    # Create prioritized recommendations
    expert_recommendations = []
    for i, rec in enumerate(consensus["consolidated_recommendations"][:5]):  # Top 5
        expert_recommendations.append({
            "recommendation": rec,
            "priority": i + 1,
            "supporting_experts": len([e for e in expert_opinions.values() 
                                    if "recommendations" in e and rec in e["recommendations"]])
        })
    
    return {
        **state,
        "consensus_report": consensus,
        "expert_recommendations": expert_recommendations
    }

# Create expert network system
def create_expert_network_system():
    """Create expert network multi-agent system"""
    
    graph = StateGraph(ExpertNetworkState)
    
    # Add expert agents
    graph.add_node("data_science_expert", data_science_expert_agent)
    graph.add_node("security_expert", security_expert_agent)
    graph.add_node("consensus_coordinator", consensus_coordinator_agent)
    
    # Parallel expert consultation
    graph.add_edge(START, "data_science_expert")
    graph.add_edge(START, "security_expert")
    
    # Both experts feed into consensus coordinator
    graph.add_edge("data_science_expert", "consensus_coordinator")
    graph.add_edge("security_expert", "consensus_coordinator")
    graph.add_edge("consensus_coordinator", END)
    
    return graph.compile()
```

### 10.5 Testing Multi-Agent Systems

#### Multi-Agent System Testing

```python
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

class TestMultiAgentSystem:
    """Test multi-agent system functionality"""
    
    def test_basic_agent_coordination(self):
        """Test basic multi-agent workflow"""
        app = create_basic_multi_agent_system()
        
        initial_state = {
            "user_query": "Test query for agents",
            "research_results": {},
            "analysis_results": {},
            "final_report": "",
            "agent_logs": [],
            "current_agent": ""
        }
        
        result = app.invoke(initial_state)
        
        # Verify all agents executed
        agent_names = [log["agent"] for log in result["agent_logs"]]
        assert "research_agent" in agent_names
        assert "analysis_agent" in agent_names
        assert "report_agent" in agent_names
        
        # Verify final report was generated
        assert result["final_report"].strip() != ""
        assert "Research & Analysis Report" in result["final_report"]
    
    def test_supervisor_worker_coordination(self):
        """Test supervisor-worker pattern"""
        app = create_supervisor_worker_system()
        
        initial_state = {
            "user_request": "data analysis task",
            "task_assignments": [],
            "worker_results": {},
            "supervisor_decision": "",
            "final_output": "",
            "coordination_log": []
        }
        
        result = app.invoke(initial_state)
        
        # Verify supervisor made decision
        assert result["supervisor_decision"] == "data_analysis_workflow"
        
        # Verify workers were assigned tasks
        assert len(result["task_assignments"]) > 0
        
        # Verify coordination occurred
        assert len(result["coordination_log"]) > 0
    
    def test_expert_network_consensus(self):
        """Test expert network consensus building"""
        app = create_expert_network_system()
        
        initial_state = {
            "problem_statement": "We need to improve data security and analysis capabilities",
            "required_expertise": ["data_science", "security"],
            "expert_opinions": {},
            "consensus_report": {},
            "confidence_scores": {},
            "expert_recommendations": []
        }
        
        result = app.invoke(initial_state)
        
        # Verify experts provided opinions
        assert len(result["expert_opinions"]) >= 2
        assert "data_science_expert" in result["expert_opinions"]
        assert "security_expert" in result["expert_opinions"]
        
        # Verify consensus was built
        assert "consensus_report" in result
        assert len(result["expert_recommendations"]) > 0
    
    @patch('datetime.datetime')
    def test_agent_communication_timing(self, mock_datetime):
        """Test agent communication with controlled timing"""
        fixed_time = datetime(2024, 1, 15, 10, 0, 0)
        mock_datetime.now.return_value = fixed_time
        
        # Test message creation
        message = AgentMessage(
            from_agent="test_agent_1",
            to_agent="test_agent_2",
            message_type="test_message",
            content={"data": "test"},
            timestamp=fixed_time.isoformat(),
            priority=1
        )
        
        assert message.timestamp == fixed_time.isoformat()
        assert message.from_agent == "test_agent_1"
        assert message.priority == 1
    
    def test_agent_message_priority_ordering(self):
        """Test that messages are processed by priority"""
        messages = [
            AgentMessage("agent1", "agent2", "low_priority", {}, "2024-01-01T10:00:00", priority=3),
            AgentMessage("agent1", "agent2", "high_priority", {}, "2024-01-01T10:01:00", priority=1),
            AgentMessage("agent1", "agent2", "medium_priority", {}, "2024-01-01T10:02:00", priority=2)
        ]
        
        # Sort by priority (lower number = higher priority)
        sorted_messages = sorted(messages, key=lambda x: x.priority)
        
        assert sorted_messages[0].message_type == "high_priority"
        assert sorted_messages[1].message_type == "medium_priority"
        assert sorted_messages[2].message_type == "low_priority"

# Performance testing for multi-agent systems
class TestMultiAgentPerformance:
    """Test multi-agent system performance"""
    
    def test_agent_execution_time(self):
        """Test that agents execute within reasonable time"""
        import time
        
        app = create_basic_multi_agent_system()
        
        initial_state = {
            "user_query": "Performance test query",
            "research_results": {},
            "analysis_results": {},
            "final_report": "",
            "agent_logs": [],
            "current_agent": ""
        }
        
        start_time = time.time()
        result = app.invoke(initial_state)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert execution_time < 5.0  # 5 seconds max
        
        # Verify all agents completed
        assert len(result["agent_logs"]) == 3  # 3 agents should have executed
    
    def test_large_message_volume_handling(self):
        """Test system with large volume of messages"""
        
        # Create many messages
        messages = []
        for i in range(100):
            messages.append(AgentMessage(
                f"agent_{i % 10}",  # 10 different agents
                f"target_{(i + 1) % 10}",
                f"message_{i}",
                {"data": f"payload_{i}"},
                f"2024-01-01T10:{i:02d}:00",
                priority=i % 3 + 1  # Vary priorities
            ))
        
        state = {
            "user_task": "Large volume test",
            "agent_messages": messages,
            "agent_states": {},
            "task_progress": {},
            "final_result": ""
        }
        
        # Test message broker can handle volume
        result = message_broker(state)
        
        # Should have routing log entries
        assert len(result["routing_log"]) > 0
```

### 10.6 Best Practices for Multi-Agent Systems

#### Multi-Agent Design Principles

1. **Clear Agent Responsibilities**: Each agent should have well-defined roles and capabilities
2. **Efficient Communication**: Use message passing and shared state appropriately
3. **Fault Tolerance**: Design agents to handle failures of other agents gracefully
4. **Scalability**: Consider how the system scales with more agents
5. **Coordination Overhead**: Balance coordination benefits with communication costs

#### Common Multi-Agent Patterns

```python
"""
Multi-Agent Pattern Reference Guide

1. Sequential Chain:
   Agent A → Agent B → Agent C → Result
   
2. Supervisor-Worker:
   Supervisor → {Worker 1, Worker 2, Worker 3} → Coordinator
   
3. Expert Network:
   Problem → {Expert 1, Expert 2, Expert N} → Consensus
   
4. Bidding System:
   Task → Agents Bid → Winner Selected → Execution
   
5. Hierarchical:
   Manager → Team Leads → Individual Agents
   
6. Peer-to-Peer:
   Agents communicate directly without central coordination

Choose patterns based on your coordination needs, scalability requirements, and fault tolerance needs.
"""
```

---

## 11. Advanced Patterns

This section explores sophisticated LangGraph patterns for complex applications, including streaming, batching, dynamic graph construction, and advanced flow control mechanisms.

### 11.1 Streaming and Real-time Processing

#### Streaming Graph Execution

```python
from typing import TypedDict, AsyncIterator, Dict, Any, List
import asyncio
from langgraph.graph import StateGraph, START, END

class StreamingState(TypedDict):
    input_stream: List[str]
    processed_items: List[Dict[str, Any]]
    current_item: str
    stream_position: int
    batch_results: List[str]

async def stream_processor_node(state: StreamingState) -> StreamingState:
    """Process items from stream one at a time"""
    
    if state["stream_position"] >= len(state["input_stream"]):
        return state  # No more items to process
    
    current_item = state["input_stream"][state["stream_position"]]
    
    # Simulate async processing
    await asyncio.sleep(0.1)
    
    processed_item = {
        "original": current_item,
        "processed": current_item.upper(),
        "timestamp": asyncio.get_event_loop().time(),
        "position": state["stream_position"]
    }
    
    return {
        **state,
        "current_item": current_item,
        "processed_items": state["processed_items"] + [processed_item],
        "stream_position": state["stream_position"] + 1
    }

def should_continue_streaming(state: StreamingState) -> str:
    """Determine if more streaming is needed"""
    if state["stream_position"] < len(state["input_stream"]):
        return "continue"
    else:
        return "complete"

async def create_streaming_graph():
    """Create streaming processing graph"""
    
    graph = StateGraph(StreamingState)
    
    graph.add_node("stream_processor", stream_processor_node)
    
    graph.add_edge(START, "stream_processor")
    graph.add_conditional_edge(
        "stream_processor",
        should_continue_streaming,
        {
            "continue": "stream_processor",  # Loop back for more items
            "complete": END
        }
    )
    
    return graph.compile()

# Real-time streaming example
async def process_real_time_stream():
    """Example of real-time stream processing"""
    
    app = await create_streaming_graph()
    
    # Simulate real-time data stream
    stream_data = [f"item_{i}" for i in range(10)]
    
    initial_state = {
        "input_stream": stream_data,
        "processed_items": [],
        "current_item": "",
        "stream_position": 0,
        "batch_results": []
    }
    
    # Process stream
    result = await app.ainvoke(initial_state)
    
    print(f"Processed {len(result['processed_items'])} items from stream")
    for item in result['processed_items']:
        print(f"  {item['original']} -> {item['processed']} at {item['timestamp']:.2f}")
    
    return result

# Generator-based streaming
async def streaming_generator(items: List[str]) -> AsyncIterator[Dict[str, Any]]:
    """Generate streaming results"""
    for i, item in enumerate(items):
        await asyncio.sleep(0.05)  # Simulate processing time
        yield {
            "item": item,
            "processed": item.upper(),
            "index": i,
            "timestamp": asyncio.get_event_loop().time()
        }

async def consume_streaming_results():
    """Example of consuming streaming results"""
    items = [f"stream_item_{i}" for i in range(5)]
    
    print("Processing stream:")
    async for result in streaming_generator(items):
        print(f"  Received: {result['item']} -> {result['processed']}")
```

#### Batch Processing Patterns

```python
from typing import TypedDict, List, Dict, Any
from datetime import datetime
import asyncio

class BatchProcessingState(TypedDict):
    input_data: List[str]
    batch_size: int
    current_batch: List[str]
    processed_batches: List[Dict[str, Any]]
    batch_index: int
    total_items: int

def batch_creator_node(state: BatchProcessingState) -> BatchProcessingState:
    """Create batches from input data"""
    
    start_idx = state["batch_index"] * state["batch_size"]
    end_idx = min(start_idx + state["batch_size"], len(state["input_data"]))
    
    if start_idx >= len(state["input_data"]):
        return {**state, "current_batch": []}  # No more batches
    
    current_batch = state["input_data"][start_idx:end_idx]
    
    return {
        **state,
        "current_batch": current_batch
    }

async def batch_processor_node(state: BatchProcessingState) -> BatchProcessingState:
    """Process a batch of items"""
    
    if not state["current_batch"]:
        return state
    
    batch = state["current_batch"]
    
    # Process batch items in parallel
    async def process_item(item: str) -> Dict[str, Any]:
        await asyncio.sleep(0.01)  # Simulate async work
        return {
            "original": item,
            "processed": item.upper(),
            "length": len(item)
        }
    
    # Process all items in batch concurrently
    batch_tasks = [process_item(item) for item in batch]
    processed_items = await asyncio.gather(*batch_tasks)
    
    batch_result = {
        "batch_index": state["batch_index"],
        "batch_size": len(batch),
        "items": processed_items,
        "processing_time": datetime.now().isoformat(),
        "total_length": sum(item["length"] for item in processed_items)
    }
    
    return {
        **state,
        "processed_batches": state["processed_batches"] + [batch_result],
        "batch_index": state["batch_index"] + 1,
        "current_batch": []
    }

def has_more_batches(state: BatchProcessingState) -> str:
    """Check if there are more batches to process"""
    total_batches = (len(state["input_data"]) + state["batch_size"] - 1) // state["batch_size"]
    
    if state["batch_index"] < total_batches:
        return "more_batches"
    else:
        return "complete"

async def create_batch_processing_graph():
    """Create batch processing graph"""
    
    graph = StateGraph(BatchProcessingState)
    
    graph.add_node("batch_creator", batch_creator_node)
    graph.add_node("batch_processor", batch_processor_node)
    
    graph.add_edge(START, "batch_creator")
    graph.add_edge("batch_creator", "batch_processor")
    
    graph.add_conditional_edge(
        "batch_processor",
        has_more_batches,
        {
            "more_batches": "batch_creator",  # Create next batch
            "complete": END
        }
    )
    
    return graph.compile()

# Usage example
async def run_batch_processing():
    """Example of batch processing"""
    
    app = await create_batch_processing_graph()
    
    # Large dataset to process in batches
    large_dataset = [f"data_item_{i:04d}" for i in range(100)]
    
    initial_state = {
        "input_data": large_dataset,
        "batch_size": 10,
        "current_batch": [],
        "processed_batches": [],
        "batch_index": 0,
        "total_items": len(large_dataset)
    }
    
    result = await app.ainvoke(initial_state)
    
    print(f"Processed {len(result['processed_batches'])} batches")
    for batch_result in result['processed_batches']:
        print(f"  Batch {batch_result['batch_index']}: {batch_result['batch_size']} items, "
              f"total length: {batch_result['total_length']}")
    
    return result
```

### 11.2 Dynamic Graph Construction

#### Runtime Graph Modification

```python
from typing import TypedDict, Dict, Any, List, Callable
from langgraph.graph import StateGraph, START, END

class DynamicGraphState(TypedDict):
    graph_config: Dict[str, Any]
    available_nodes: Dict[str, Callable]
    execution_plan: List[str]
    current_step: int
    step_results: Dict[str, Any]
    graph_modified: bool

def graph_planner_node(state: DynamicGraphState) -> DynamicGraphState:
    """Plan graph structure based on configuration"""
    
    config = state["graph_config"]
    task_type = config.get("task_type", "default")
    complexity = config.get("complexity", "medium")
    
    # Determine execution plan based on configuration
    if task_type == "data_processing":
        if complexity == "simple":
            plan = ["data_loader", "simple_processor", "output_formatter"]
        elif complexity == "complex":
            plan = ["data_loader", "validator", "complex_processor", "analyzer", "output_formatter"]
        else:  # medium
            plan = ["data_loader", "processor", "output_formatter"]
    elif task_type == "analysis":
        plan = ["data_collector", "statistical_analyzer", "report_generator"]
    else:
        plan = ["default_processor"]
    
    return {
        **state,
        "execution_plan": plan,
        "current_step": 0
    }

def dynamic_executor_node(state: DynamicGraphState) -> DynamicGraphState:
    """Execute current step in the dynamic plan"""
    
    plan = state["execution_plan"]
    step_idx = state["current_step"]
    
    if step_idx >= len(plan):
        return state  # No more steps
    
    current_node_name = plan[step_idx]
    current_node_func = state["available_nodes"].get(current_node_name)
    
    if not current_node_func:
        # Node not available, skip or handle error
        result = {"error": f"Node {current_node_name} not available"}
    else:
        # Execute the node function
        result = current_node_func(state)
    
    # Store step result
    step_results = {**state["step_results"]}
    step_results[current_node_name] = result
    
    return {
        **state,
        "step_results": step_results,
        "current_step": step_idx + 1
    }

def has_more_steps(state: DynamicGraphState) -> str:
    """Check if there are more steps to execute"""
    if state["current_step"] < len(state["execution_plan"]):
        return "continue"
    else:
        return "complete"

# Available node functions
def data_loader_func(state: DynamicGraphState) -> Dict[str, Any]:
    """Load data step"""
    return {
        "step": "data_loading",
        "data_loaded": True,
        "records_count": 1000,
        "status": "completed"
    }

def simple_processor_func(state: DynamicGraphState) -> Dict[str, Any]:
    """Simple processing step"""
    return {
        "step": "simple_processing",
        "processing_applied": "basic_transformation",
        "status": "completed"
    }

def complex_processor_func(state: DynamicGraphState) -> Dict[str, Any]:
    """Complex processing step"""
    return {
        "step": "complex_processing", 
        "processing_applied": "advanced_transformation",
        "algorithms_used": ["algorithm_a", "algorithm_b"],
        "status": "completed"
    }

def validator_func(state: DynamicGraphState) -> Dict[str, Any]:
    """Data validation step"""
    return {
        "step": "validation",
        "validation_passed": True,
        "errors_found": 0,
        "status": "completed"
    }

def analyzer_func(state: DynamicGraphState) -> Dict[str, Any]:
    """Analysis step"""
    return {
        "step": "analysis",
        "insights_generated": 5,
        "patterns_found": ["pattern_1", "pattern_2"],
        "status": "completed"
    }

def output_formatter_func(state: DynamicGraphState) -> Dict[str, Any]:
    """Output formatting step"""
    return {
        "step": "output_formatting",
        "format": "json",
        "output_ready": True,
        "status": "completed"
    }

def create_dynamic_graph():
    """Create dynamic graph that can modify itself at runtime"""
    
    available_nodes = {
        "data_loader": data_loader_func,
        "simple_processor": simple_processor_func,
        "complex_processor": complex_processor_func,
        "validator": validator_func,
        "analyzer": analyzer_func,
        "output_formatter": output_formatter_func,
        "processor": simple_processor_func,  # Alias
        "data_collector": data_loader_func,  # Alias
        "statistical_analyzer": analyzer_func,  # Alias
        "report_generator": output_formatter_func,  # Alias
        "default_processor": simple_processor_func  # Alias
    }
    
    graph = StateGraph(DynamicGraphState)
    
    graph.add_node("planner", graph_planner_node)
    graph.add_node("executor", dynamic_executor_node)
    
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    
    graph.add_conditional_edge(
        "executor",
        has_more_steps,
        {
            "continue": "executor",  # Loop back for next step
            "complete": END
        }
    )
    
    return graph.compile(), available_nodes

# Usage examples
def run_dynamic_graph_examples():
    """Examples of dynamic graph usage"""
    
    app, available_nodes = create_dynamic_graph()
    
    # Example 1: Simple data processing
    simple_config = {
        "task_type": "data_processing",
        "complexity": "simple"
    }
    
    simple_state = {
        "graph_config": simple_config,
        "available_nodes": available_nodes,
        "execution_plan": [],
        "current_step": 0,
        "step_results": {},
        "graph_modified": False
    }
    
    result1 = app.invoke(simple_state)
    print("Simple processing plan:", result1["execution_plan"])
    print("Steps executed:", len(result1["step_results"]))
    
    # Example 2: Complex data processing
    complex_config = {
        "task_type": "data_processing",
        "complexity": "complex"
    }
    
    complex_state = {
        "graph_config": complex_config,
        "available_nodes": available_nodes,
        "execution_plan": [],
        "current_step": 0,
        "step_results": {},
        "graph_modified": False
    }
    
    result2 = app.invoke(complex_state)
    print("Complex processing plan:", result2["execution_plan"])
    print("Steps executed:", len(result2["step_results"]))
    
    # Example 3: Analysis task
    analysis_config = {
        "task_type": "analysis",
        "complexity": "medium"
    }
    
    analysis_state = {
        "graph_config": analysis_config,
        "available_nodes": available_nodes,
        "execution_plan": [],
        "current_step": 0,
        "step_results": {},
        "graph_modified": False
    }
    
    result3 = app.invoke(analysis_state)
    print("Analysis plan:", result3["execution_plan"])
    print("Steps executed:", len(result3["step_results"]))
    
    return [result1, result2, result3]
```

### 11.3 Advanced Flow Control

#### Circuit Breaker Pattern

```python
from typing import TypedDict, Dict, Any
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, not allowing calls
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreakerState(TypedDict):
    service_name: str
    circuit_state: str
    failure_count: int
    failure_threshold: int
    success_count: int
    last_failure_time: float
    timeout_duration: float
    half_open_max_calls: int
    half_open_calls: int
    service_response: Dict[str, Any]

def circuit_breaker_node(state: CircuitBreakerState) -> CircuitBreakerState:
    """Circuit breaker logic for service calls"""
    
    current_time = time.time()
    circuit_state = state["circuit_state"]
    
    # Check if circuit should transition from OPEN to HALF_OPEN
    if (circuit_state == CircuitState.OPEN.value and 
        current_time - state["last_failure_time"] > state["timeout_duration"]):
        circuit_state = CircuitState.HALF_OPEN.value
        return {
            **state,
            "circuit_state": circuit_state,
            "half_open_calls": 0
        }
    
    # If circuit is OPEN, don't make the call
    if circuit_state == CircuitState.OPEN.value:
        return {
            **state,
            "service_response": {
                "status": "circuit_open",
                "message": "Service calls blocked due to circuit breaker",
                "error": True
            }
        }
    
    return state

def service_call_node(state: CircuitBreakerState) -> CircuitBreakerState:
    """Make service call with circuit breaker protection"""
    
    if state["circuit_state"] == CircuitState.OPEN.value:
        return state  # Already handled by circuit breaker
    
    # Simulate service call
    import random
    call_successful = random.random() > 0.3  # 70% success rate
    
    if call_successful:
        # Success - update circuit breaker state
        success_count = state["success_count"] + 1
        
        # If we're in HALF_OPEN state, check if we should close the circuit
        if state["circuit_state"] == CircuitState.HALF_OPEN.value:
            if success_count >= 2:  # Require 2 successes to close circuit
                circuit_state = CircuitState.CLOSED.value
                failure_count = 0
            else:
                circuit_state = state["circuit_state"]
                failure_count = state["failure_count"]
        else:
            circuit_state = CircuitState.CLOSED.value
            failure_count = max(0, state["failure_count"] - 1)  # Reduce failure count on success
        
        return {
            **state,
            "circuit_state": circuit_state,
            "failure_count": failure_count,
            "success_count": success_count,
            "service_response": {
                "status": "success",
                "data": f"Service call successful for {state['service_name']}",
                "timestamp": time.time(),
                "error": False
            }
        }
    
    else:
        # Failure - update circuit breaker state
        failure_count = state["failure_count"] + 1
        last_failure_time = time.time()
        
        # Check if we should open the circuit
        if failure_count >= state["failure_threshold"]:
            circuit_state = CircuitState.OPEN.value
        elif state["circuit_state"] == CircuitState.HALF_OPEN.value:
            circuit_state = CircuitState.OPEN.value  # Go back to OPEN on failure
        else:
            circuit_state = state["circuit_state"]
        
        return {
            **state,
            "circuit_state": circuit_state,
            "failure_count": failure_count,
            "last_failure_time": last_failure_time,
            "service_response": {
                "status": "failure",
                "message": f"Service call failed for {state['service_name']}",
                "error": True,
                "failure_count": failure_count
            }
        }

def create_circuit_breaker_graph():
    """Create circuit breaker protected service call graph"""
    
    graph = StateGraph(CircuitBreakerState)
    
    graph.add_node("circuit_check", circuit_breaker_node)
    graph.add_node("service_call", service_call_node)
    
    graph.add_edge(START, "circuit_check")
    graph.add_edge("circuit_check", "service_call")
    graph.add_edge("service_call", END)
    
    return graph.compile()

# Usage example
def test_circuit_breaker():
    """Test circuit breaker pattern"""
    
    app = create_circuit_breaker_graph()
    
    initial_state = {
        "service_name": "external_api",
        "circuit_state": CircuitState.CLOSED.value,
        "failure_count": 0,
        "failure_threshold": 3,
        "success_count": 0,
        "last_failure_time": 0.0,
        "timeout_duration": 5.0,  # 5 seconds
        "half_open_max_calls": 2,
        "half_open_calls": 0,
        "service_response": {}
    }
    
    # Make multiple calls to test circuit breaker behavior
    for i in range(10):
        result = app.invoke(initial_state)
        
        print(f"Call {i+1}: Circuit={result['circuit_state']}, "
              f"Failures={result['failure_count']}, "
              f"Response={result['service_response']['status']}")
        
        # Update state for next call
        initial_state = {
            **result,
            "service_response": {}  # Reset response for next call
        }
        
        time.sleep(0.1)  # Small delay between calls
    
    return result
```

#### Retry with Exponential Backoff

```python
import asyncio
import random
from typing import TypedDict, Dict, Any

class RetryState(TypedDict):
    operation_name: str
    attempt_count: int
    max_attempts: int
    base_delay: float
    max_delay: float
    backoff_multiplier: float
    jitter: bool
    last_error: str
    operation_result: Dict[str, Any]
    success: bool

async def retryable_operation_node(state: RetryState) -> RetryState:
    """Perform operation that might fail and need retry"""
    
    attempt = state["attempt_count"] + 1
    
    # Simulate operation that might fail
    # Increasing success probability with each attempt
    success_probability = min(0.2 + (attempt * 0.2), 0.9)
    operation_successful = random.random() < success_probability
    
    if operation_successful:
        return {
            **state,
            "attempt_count": attempt,
            "success": True,
            "operation_result": {
                "status": "success",
                "data": f"Operation {state['operation_name']} succeeded on attempt {attempt}",
                "attempt_count": attempt
            },
            "last_error": ""
        }
    else:
        error_msg = f"Operation {state['operation_name']} failed on attempt {attempt}"
        return {
            **state,
            "attempt_count": attempt,
            "success": False,
            "last_error": error_msg,
            "operation_result": {
                "status": "failure",
                "error": error_msg,
                "attempt_count": attempt
            }
        }

async def retry_delay_node(state: RetryState) -> RetryState:
    """Calculate and apply retry delay with exponential backoff"""
    
    if state["success"] or state["attempt_count"] >= state["max_attempts"]:
        return state  # No delay needed
    
    # Calculate delay with exponential backoff
    delay = min(
        state["base_delay"] * (state["backoff_multiplier"] ** (state["attempt_count"] - 1)),
        state["max_delay"]
    )
    
    # Add jitter if enabled
    if state["jitter"]:
        jitter_range = delay * 0.1  # 10% jitter
        delay += random.uniform(-jitter_range, jitter_range)
    
    # Apply delay
    await asyncio.sleep(delay)
    
    return {
        **state,
        "operation_result": {
            **state["operation_result"],
            "delay_applied": delay,
            "next_attempt_in": delay
        }
    }

def should_retry(state: RetryState) -> str:
    """Determine if operation should be retried"""
    
    if state["success"]:
        return "success"
    elif state["attempt_count"] >= state["max_attempts"]:
        return "max_attempts_reached"
    else:
        return "retry"

async def create_retry_graph():
    """Create retry graph with exponential backoff"""
    
    graph = StateGraph(RetryState)
    
    graph.add_node("operation", retryable_operation_node)
    graph.add_node("delay", retry_delay_node)
    
    graph.add_edge(START, "operation")
    graph.add_edge("operation", "delay")
    
    graph.add_conditional_edge(
        "delay",
        should_retry,
        {
            "success": END,
            "retry": "operation",  # Try again
            "max_attempts_reached": END
        }
    )
    
    return graph.compile()

# Usage example
async def test_retry_pattern():
    """Test retry pattern with exponential backoff"""
    
    app = await create_retry_graph()
    
    initial_state = {
        "operation_name": "critical_api_call",
        "attempt_count": 0,
        "max_attempts": 5,
        "base_delay": 1.0,  # Start with 1 second
        "max_delay": 16.0,  # Cap at 16 seconds
        "backoff_multiplier": 2.0,  # Double delay each time
        "jitter": True,
        "last_error": "",
        "operation_result": {},
        "success": False
    }
    
    print("Starting retry operation...")
    start_time = asyncio.get_event_loop().time()
    
    result = await app.ainvoke(initial_state)
    
    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time
    
    print(f"Operation completed in {total_time:.2f} seconds")
    print(f"Success: {result['success']}")
    print(f"Attempts: {result['attempt_count']}")
    print(f"Final result: {result['operation_result']}")
    
    return result
```

### 11.4 Advanced State Transformation Patterns

#### State Aggregation and Reduction

```python
from typing import TypedDict, List, Dict, Any, Callable
from functools import reduce

class AggregationState(TypedDict):
    data_sources: List[Dict[str, Any]]
    aggregation_rules: Dict[str, str]
    intermediate_results: Dict[str, Any]
    final_aggregation: Dict[str, Any]
    processing_metadata: Dict[str, Any]

def data_collector_node(state: AggregationState) -> AggregationState:
    """Collect data from multiple sources"""
    
    # Simulate collecting from multiple data sources
    collected_data = []
    
    for i, source in enumerate(state["data_sources"]):
        source_data = {
            "source_id": source.get("id", f"source_{i}"),
            "data": source.get("data", []),
            "timestamp": source.get("timestamp", time.time()),
            "metadata": source.get("metadata", {})
        }
        collected_data.append(source_data)
    
    return {
        **state,
        "intermediate_results": {
            **state["intermediate_results"],
            "collected_data": collected_data,
            "collection_completed": True
        }
    }

def data_transformer_node(state: AggregationState) -> AggregationState:
    """Transform data before aggregation"""
    
    collected_data = state["intermediate_results"].get("collected_data", [])
    
    transformed_data = []
    for source_data in collected_data:
        # Apply transformations based on source type
        source_type = source_data.get("metadata", {}).get("type", "default")
        
        if source_type == "numeric":
            transformed = [float(x) for x in source_data["data"] if isinstance(x, (int, float, str)) and str(x).replace('.','').isdigit()]
        elif source_type == "text":
            transformed = [str(x).lower().strip() for x in source_data["data"]]
        else:
            transformed = source_data["data"]
        
        transformed_data.append({
            **source_data,
            "transformed_data": transformed
        })
    
    return {
        **state,
        "intermediate_results": {
            **state["intermediate_results"],
            "transformed_data": transformed_data,
            "transformation_completed": True
        }
    }

def aggregator_node(state: AggregationState) -> AggregationState:
    """Aggregate transformed data according to rules"""
    
    transformed_data = state["intermediate_results"].get("transformed_data", [])
    aggregation_rules = state["aggregation_rules"]
    
    aggregations = {}
    
    for rule_name, rule_type in aggregation_rules.items():
        if rule_type == "sum":
            # Sum all numeric data
            all_numeric = []
            for source in transformed_data:
                if isinstance(source.get("transformed_data"), list):
                    all_numeric.extend([x for x in source["transformed_data"] if isinstance(x, (int, float))])
            aggregations[rule_name] = sum(all_numeric)
            
        elif rule_type == "count":
            # Count all items
            total_count = 0
            for source in transformed_data:
                if isinstance(source.get("transformed_data"), list):
                    total_count += len(source["transformed_data"])
            aggregations[rule_name] = total_count
            
        elif rule_type == "average":
            # Average of all numeric data
            all_numeric = []
            for source in transformed_data:
                if isinstance(source.get("transformed_data"), list):
                    all_numeric.extend([x for x in source["transformed_data"] if isinstance(x, (int, float))])
            aggregations[rule_name] = sum(all_numeric) / len(all_numeric) if all_numeric else 0
            
        elif rule_type == "concat":
            # Concatenate all text data
            all_text = []
            for source in transformed_data:
                if isinstance(source.get("transformed_data"), list):
                    all_text.extend([str(x) for x in source["transformed_data"]])
            aggregations[rule_name] = " ".join(all_text)
    
    # Metadata about aggregation process
    processing_metadata = {
        "sources_processed": len(transformed_data),
        "rules_applied": len(aggregation_rules),
        "aggregation_timestamp": time.time(),
        "data_points_processed": sum(len(s.get("transformed_data", [])) for s in transformed_data)
    }
    
    return {
        **state,
        "final_aggregation": aggregations,
        "processing_metadata": processing_metadata
    }

def create_aggregation_graph():
    """Create data aggregation graph"""
    
    graph = StateGraph(AggregationState)
    
    graph.add_node("collector", data_collector_node)
    graph.add_node("transformer", data_transformer_node)
    graph.add_node("aggregator", aggregator_node)
    
    graph.add_edge(START, "collector")
    graph.add_edge("collector", "transformer")
    graph.add_edge("transformer", "aggregator")
    graph.add_edge("aggregator", END)
    
    return graph.compile()

# Usage example
def test_aggregation_pattern():
    """Test data aggregation pattern"""
    
    app = create_aggregation_graph()
    
    # Sample data sources
    data_sources = [
        {
            "id": "source_1",
            "data": [10, 20, 30, 40],
            "metadata": {"type": "numeric", "source": "database_a"}
        },
        {
            "id": "source_2", 
            "data": [5, 15, 25],
            "metadata": {"type": "numeric", "source": "database_b"}
        },
        {
            "id": "source_3",
            "data": ["hello", "world", "test"],
            "metadata": {"type": "text", "source": "text_api"}
        }
    ]
    
    # Aggregation rules
    aggregation_rules = {
        "total_sum": "sum",
        "item_count": "count", 
        "average_value": "average",
        "combined_text": "concat"
    }
    
    initial_state = {
        "data_sources": data_sources,
        "aggregation_rules": aggregation_rules,
        "intermediate_results": {},
        "final_aggregation": {},
        "processing_metadata": {}
    }
    
    result = app.invoke(initial_state)
    
    print("Aggregation Results:")
    for rule, value in result["final_aggregation"].items():
        print(f"  {rule}: {value}")
    
    print("\nProcessing Metadata:")
    for key, value in result["processing_metadata"].items():
        print(f"  {key}: {value}")
    
    return result
```

### 11.5 Testing Advanced Patterns

#### Advanced Pattern Testing

```python
import pytest
import asyncio
from unittest.mock import patch, AsyncMock

class TestAdvancedPatterns:
    """Test advanced LangGraph patterns"""
    
    @pytest.mark.asyncio
    async def test_streaming_processing(self):
        """Test streaming graph execution"""
        
        app = await create_streaming_graph()
        
        test_stream = ["item1", "item2", "item3"]
        initial_state = {
            "input_stream": test_stream,
            "processed_items": [],
            "current_item": "",
            "stream_position": 0,
            "batch_results": []
        }
        
        result = await app.ainvoke(initial_state)
        
        # Verify all items were processed
        assert len(result["processed_items"]) == len(test_stream)
        assert result["stream_position"] == len(test_stream)
        
        # Verify processing occurred
        for i, item in enumerate(result["processed_items"]):
            assert item["original"] == f"item{i+1}"
            assert item["processed"] == f"ITEM{i+1}"
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing pattern"""
        
        app = await create_batch_processing_graph()
        
        # Test data that should create multiple batches
        test_data = [f"item_{i}" for i in range(15)]
        batch_size = 5
        
        initial_state = {
            "input_data": test_data,
            "batch_size": batch_size,
            "current_batch": [],
            "processed_batches": [],
            "batch_index": 0,
            "total_items": len(test_data)
        }
        
        result = await app.ainvoke(initial_state)
        
        # Should create 3 batches (15 items / 5 per batch)
        assert len(result["processed_batches"]) == 3
        
        # Check batch sizes
        assert result["processed_batches"][0]["batch_size"] == 5
        assert result["processed_batches"][1]["batch_size"] == 5
        assert result["processed_batches"][2]["batch_size"] == 5
        
        # Verify all items processed
        total_processed = sum(batch["batch_size"] for batch in result["processed_batches"])
        assert total_processed == len(test_data)
    
    def test_dynamic_graph_construction(self):
        """Test dynamic graph construction"""
        
        app, available_nodes = create_dynamic_graph()
        
        # Test different configurations
        configs = [
            {"task_type": "data_processing", "complexity": "simple"},
            {"task_type": "data_processing", "complexity": "complex"},
            {"task_type": "analysis", "complexity": "medium"}
        ]
        
        for config in configs:
            state = {
                "graph_config": config,
                "available_nodes": available_nodes,
                "execution_plan": [],
                "current_step": 0,
                "step_results": {},
                "graph_modified": False
            }
            
            result = app.invoke(state)
            
            # Verify plan was created
            assert len(result["execution_plan"]) > 0
            
            # Verify all steps executed
            assert len(result["step_results"]) == len(result["execution_plan"])
            
            # Verify specific plans for known configurations
            if config["task_type"] == "data_processing" and config["complexity"] == "simple":
                expected_plan = ["data_loader", "simple_processor", "output_formatter"]
                assert result["execution_plan"] == expected_plan
    
    def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern"""
        
        app = create_circuit_breaker_graph()
        
        initial_state = {
            "service_name": "test_service",
            "circuit_state": "closed",
            "failure_count": 0,
            "failure_threshold": 3,
            "success_count": 0,
            "last_failure_time": 0.0,
            "timeout_duration": 1.0,
            "half_open_max_calls": 2,
            "half_open_calls": 0,
            "service_response": {}
        }
        
        # Mock the service call to always fail
        with patch('random.random', return_value=0.9):  # Force failure (> 0.3)
            
            # Make calls until circuit opens
            state = initial_state.copy()
            for i in range(5):
                result = app.invoke(state)
                state = {**result, "service_response": {}}  # Reset response
                
                if result["circuit_state"] == "open":
                    break
            
            # Circuit should be open after threshold failures
            assert result["circuit_state"] == "open"
            assert result["failure_count"] >= initial_state["failure_threshold"]
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff(self):
        """Test retry pattern with exponential backoff"""
        
        app = await create_retry_graph()
        
        initial_state = {
            "operation_name": "test_operation",
            "attempt_count": 0,
            "max_attempts": 3,
            "base_delay": 0.1,  # Short delay for testing
            "max_delay": 1.0,
            "backoff_multiplier": 2.0,
            "jitter": False,  # Disable for predictable testing
            "last_error": "",
            "operation_result": {},
            "success": False
        }
        
        # Mock operation to fail first 2 attempts, succeed on 3rd
        call_count = 0
        def mock_random():
            nonlocal call_count
            call_count += 1
            return 0.1 if call_count >= 3 else 0.9  # Success on 3rd call
        
        with patch('random.random', side_effect=mock_random):
            start_time = asyncio.get_event_loop().time()
            result = await app.ainvoke(initial_state)
            end_time = asyncio.get_event_loop().time()
            
            # Should eventually succeed
            assert result["success"] == True
            assert result["attempt_count"] == 3
            
            # Should have taken some time due to delays
            assert end_time - start_time >= 0.3  # At least sum of delays (0.1 + 0.2)
    
    def test_data_aggregation_pattern(self):
        """Test data aggregation pattern"""
        
        app = create_aggregation_graph()
        
        data_sources = [
            {"id": "source1", "data": [1, 2, 3], "metadata": {"type": "numeric"}},
            {"id": "source2", "data": [4, 5], "metadata": {"type": "numeric"}},
            {"id": "source3", "data": ["a", "b"], "metadata": {"type": "text"}}
        ]
        
        aggregation_rules = {
            "total": "sum",
            "count": "count",
            "average": "average"
        }
        
        initial_state = {
            "data_sources": data_sources,
            "aggregation_rules": aggregation_rules,
            "intermediate_results": {},
            "final_aggregation": {},
            "processing_metadata": {}
        }
        
        result = app.invoke(initial_state)
        
        # Verify aggregations
        assert result["final_aggregation"]["total"] == 15  # 1+2+3+4+5
        assert result["final_aggregation"]["count"] == 7   # 5 numbers + 2 text items
        assert result["final_aggregation"]["average"] == 3.0  # (1+2+3+4+5)/5
        
        # Verify metadata
        assert result["processing_metadata"]["sources_processed"] == 3
        assert result["processing_metadata"]["rules_applied"] == 3

# Performance benchmarks
class TestAdvancedPatternPerformance:
    """Performance tests for advanced patterns"""
    
    @pytest.mark.asyncio
    async def test_streaming_performance(self):
        """Test streaming performance with large dataset"""
        
        app = await create_streaming_graph()
        
        # Large dataset
        large_stream = [f"item_{i}" for i in range(100)]
        
        initial_state = {
            "input_stream": large_stream,
            "processed_items": [],
            "current_item": "",
            "stream_position": 0,
            "batch_results": []
        }
        
        start_time = asyncio.get_event_loop().time()
        result = await app.ainvoke(initial_state)
        end_time = asyncio.get_event_loop().time()
        
        processing_time = end_time - start_time
        
        # Should process all items
        assert len(result["processed_items"]) == 100
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert processing_time < 20.0  # 20 seconds max for 100 items
        
        # Calculate throughput
        throughput = len(large_stream) / processing_time
        print(f"Streaming throughput: {throughput:.2f} items/second")
        
        assert throughput > 5.0  # At least 5 items per second
```

### 11.6 Best Practices for Advanced Patterns

#### Advanced Pattern Guidelines

```python
"""
Advanced LangGraph Pattern Best Practices

1. **Streaming Patterns**:
   - Use async/await for I/O bound streaming operations
   - Implement backpressure mechanisms for high-volume streams
   - Consider memory usage with large streams
   - Add monitoring and metrics for stream processing

2. **Batch Processing**:
   - Choose optimal batch sizes based on memory and processing constraints
   - Implement parallel processing within batches when possible
   - Handle partial batch failures gracefully
   - Monitor batch processing times and throughput

3. **Dynamic Graphs**:
   - Validate dynamic configurations before graph construction
   - Cache compiled graphs when possible to avoid overhead
   - Implement fallback strategies for missing nodes
   - Log graph construction decisions for debugging

4. **Circuit Breakers**:
   - Set appropriate failure thresholds based on service SLAs
   - Implement proper timeout and recovery mechanisms
   - Monitor circuit breaker state changes
   - Provide meaningful error messages when circuits are open

5. **Retry Patterns**:
   - Use exponential backoff to avoid overwhelming failing services
   - Add jitter to prevent thundering herd problems
   - Set maximum retry limits to prevent infinite loops
   - Log retry attempts for monitoring and debugging

6. **State Transformation**:
   - Keep transformation logic pure and testable
   - Validate data types and formats during transformation
   - Implement efficient aggregation algorithms for large datasets
   - Consider memory usage during aggregation operations

7. **Performance Considerations**:
   - Profile advanced patterns under realistic load conditions
   - Monitor memory usage, especially with streaming and aggregation
   - Implement appropriate caching strategies
   - Use async patterns for I/O bound operations
"""
```

---

## 12. Real-World Project Structure

This section provides comprehensive guidance on structuring LangGraph projects for production environments, covering project organization, configuration management, deployment strategies, and scalability patterns.

### 12.1 Production Project Organization

#### Enterprise-Grade Directory Structure

```
langgraph-application/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── deploy.yml
│       └── security-scan.yml
├── scripts/
│   ├── setup.sh
│   ├── run-tests.sh
│   ├── build.sh
│   └── deploy.sh
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── logging.py
│   │   └── database.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── graphs/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── processing_graph.py
│   │   │   ├── analysis_graph.py
│   │   │   └── multi_agent_graph.py
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── processing_nodes.py
│   │   │   ├── analysis_nodes.py
│   │   │   ├── agent_nodes.py
│   │   │   └── external_service_nodes.py
│   │   ├── state/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── processing_state.py
│   │   │   ├── analysis_state.py
│   │   │   └── agent_state.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── validators.py
│   │       ├── transformers.py
│   │       ├── helpers.py
│   │       └── exceptions.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── cache.py
│   │   ├── external_apis.py
│   │   ├── file_storage.py
│   │   └── messaging.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py
│   │   │   ├── processing.py
│   │   │   ├── analysis.py
│   │   │   └── admin.py
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── logging.py
│   │   │   ├── rate_limiting.py
│   │   │   └── error_handling.py
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── requests.py
│   │       ├── responses.py
│   │       └── schemas.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── commands/
│   │   │   ├── __init__.py
│   │   │   ├── process.py
│   │   │   ├── analyze.py
│   │   │   └── admin.py
│   │   └── utils.py
│   └── monitoring/
│       ├── __init__.py
│       ├── metrics.py
│       ├── health_checks.py
│       ├── tracing.py
│       └── alerting.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── test_graphs.py
│   │   │   ├── test_nodes.py
│   │   │   └── test_state.py
│   │   ├── services/
│   │   │   └── test_services.py
│   │   └── api/
│   │       └── test_routes.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_graph_flows.py
│   │   ├── test_api_integration.py
│   │   └── test_database_integration.py
│   ├── e2e/
│   │   ├── __init__.py
│   │   ├── test_complete_workflows.py
│   │   └── test_user_scenarios.py
│   ├── performance/
│   │   ├── __init__.py
│   │   ├── test_load.py
│   │   ├── test_stress.py
│   │   └── test_benchmarks.py
│   └── fixtures/
│       ├── __init__.py
│       ├── sample_data.py
│       └── mock_services.py
├── docs/
│   ├── README.md
│   ├── api/
│   │   ├── openapi.yaml
│   │   └── endpoints.md
│   ├── deployment/
│   │   ├── local.md
│   │   ├── docker.md
│   │   ├── kubernetes.md
│   │   └── cloud.md
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── graphs.md
│   │   ├── nodes.md
│   │   └── state.md
│   └── guides/
│       ├── getting-started.md
│       ├── configuration.md
│       ├── monitoring.md
│       └── troubleshooting.md
├── config/
│   ├── development.yaml
│   ├── testing.yaml
│   ├── staging.yaml
│   └── production.yaml
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile.prod
│   │   ├── Dockerfile.dev
│   │   └── docker-compose.prod.yml
│   ├── kubernetes/
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── ingress.yaml
│   │   ├── configmap.yaml
│   │   └── secrets.yaml
│   ├── terraform/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── modules/
│   │       ├── vpc/
│   │       ├── database/
│   │       └── application/
│   └── ansible/
│       ├── playbook.yml
│       ├── inventory/
│       └── roles/
├── monitoring/
│   ├── prometheus/
│   │   └── rules.yml
│   ├── grafana/
│   │   └── dashboards/
│   └── alerts/
│       └── alertmanager.yml
└── data/
    ├── migrations/
    ├── seeds/
    └── samples/
```

#### Core Module Implementation

```python
# src/core/graphs/base.py
from abc import ABC, abstractmethod
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph
from langgraph.checkpoint.base import Checkpointer

class BaseGraph(ABC):
    """Base class for all LangGraph implementations"""
    
    def __init__(self, config: Dict[str, Any], checkpointer: Optional[Checkpointer] = None):
        self.config = config
        self.checkpointer = checkpointer
        self._graph = None
        self._compiled_graph = None
    
    @abstractmethod
    def define_state(self) -> TypedDict:
        """Define the state schema for this graph"""
        pass
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Build the graph structure"""
        pass
    
    def compile(self):
        """Compile the graph for execution"""
        if self._compiled_graph is None:
            self._graph = self.build_graph()
            self._compiled_graph = self._graph.compile(checkpointer=self.checkpointer)
        return self._compiled_graph
    
    def invoke(self, initial_state: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Execute the graph with given initial state"""
        compiled_graph = self.compile()
        return compiled_graph.invoke(initial_state, config=config)
    
    async def ainvoke(self, initial_state: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Async execution of the graph"""
        compiled_graph = self.compile()
        return await compiled_graph.ainvoke(initial_state, config=config)
```

```python
# src/core/nodes/base.py
from abc import ABC, abstractmethod
from typing import TypedDict, Dict, Any
from dataclasses import dataclass
import logging
from datetime import datetime

@dataclass
class NodeConfig:
    """Configuration for nodes"""
    name: str
    enabled: bool = True
    timeout_seconds: int = 30
    retry_count: int = 3
    log_level: str = "INFO"
    
class BaseNode(ABC):
    """Base class for all nodes"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.logger = logging.getLogger(f"node.{config.name}")
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
    
    @abstractmethod
    def execute(self, state: TypedDict) -> TypedDict:
        """Main execution logic for the node"""
        pass
    
    def pre_execute(self, state: TypedDict) -> Dict[str, Any]:
        """Pre-execution hook"""
        return {
            "node_name": self.config.name,
            "execution_start": datetime.now().isoformat(),
            "input_keys": list(state.keys())
        }
    
    def post_execute(self, state: TypedDict, result: TypedDict, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Post-execution hook"""
        metadata.update({
            "execution_end": datetime.now().isoformat(),
            "output_keys": list(result.keys()),
            "success": True
        })
        return metadata
    
    def __call__(self, state: TypedDict) -> TypedDict:
        """Main entry point with error handling and logging"""
        if not self.config.enabled:
            self.logger.info(f"Node {self.config.name} is disabled, skipping")
            return state
        
        metadata = self.pre_execute(state)
        
        try:
            self.logger.info(f"Executing node: {self.config.name}")
            result = self.execute(state)
            
            metadata = self.post_execute(state, result, metadata)
            self.logger.info(f"Node {self.config.name} executed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Node {self.config.name} failed: {str(e)}")
            metadata.update({
                "execution_end": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            
            # Add error information to state
            error_state = {
                **state,
                "node_errors": state.get("node_errors", []) + [{
                    "node": self.config.name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }]
            }
            
            raise Exception(f"Node {self.config.name} failed: {str(e)}") from e
```

### 12.2 Configuration Management

#### Environment-Based Configuration

```python
# src/config/settings.py
from pydantic import BaseSettings, Field
from typing import Optional, Dict, Any
import os

class DatabaseConfig(BaseSettings):
    """Database configuration"""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="langgraph_app")
    username: str = Field(default="postgres")
    password: str = Field(default="")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    
    class Config:
        env_prefix = "DB_"

class RedisConfig(BaseSettings):
    """Redis configuration"""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = Field(default=None)
    max_connections: int = Field(default=10)
    
    class Config:
        env_prefix = "REDIS_"

class APIConfig(BaseSettings):
    """API configuration"""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    debug: bool = Field(default=False)
    cors_origins: list = Field(default=["*"])
    rate_limit: str = Field(default="100/minute")
    
    class Config:
        env_prefix = "API_"

class LangGraphConfig(BaseSettings):
    """LangGraph specific configuration"""
    checkpoint_backend: str = Field(default="sqlite")
    checkpoint_path: str = Field(default="./data/checkpoints.db")
    memory_limit_mb: int = Field(default=1024)
    execution_timeout: int = Field(default=300)
    
    class Config:
        env_prefix = "LANGGRAPH_"

class MonitoringConfig(BaseSettings):
    """Monitoring configuration"""
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    tracing_enabled: bool = Field(default=False)
    tracing_endpoint: Optional[str] = Field(default=None)
    log_level: str = Field(default="INFO")
    
    class Config:
        env_prefix = "MONITORING_"

class Settings(BaseSettings):
    """Main application settings"""
    
    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    # Service configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    api: APIConfig = APIConfig()
    langgraph: LangGraphConfig = LangGraphConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    # External services
    external_service_timeout: int = Field(default=30)
    external_service_retries: int = Field(default=3)
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    @classmethod
    def load_from_file(cls, config_file: str):
        """Load settings from YAML config file"""
        import yaml
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)

# Global settings instance
settings = Settings()
```

#### Configuration Validation and Loading

```python
# src/config/loader.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from .settings import Settings

class ConfigLoader:
    """Configuration loader with environment-specific overrides"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.environment = os.getenv("ENVIRONMENT", "development")
    
    def load_config(self) -> Settings:
        """Load configuration with environment overrides"""
        
        # Load base configuration
        base_config = self._load_yaml_config("base.yaml")
        
        # Load environment-specific configuration
        env_config_file = f"{self.environment}.yaml"
        env_config = self._load_yaml_config(env_config_file)
        
        # Merge configurations
        merged_config = self._merge_configs(base_config, env_config)
        
        # Create settings instance
        settings = Settings(**merged_config)
        
        # Validate configuration
        self._validate_config(settings)
        
        return settings
    
    def _load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, settings: Settings):
        """Validate configuration settings"""
        
        # Validate database configuration
        if settings.environment == "production":
            assert settings.database.password, "Database password required in production"
            assert not settings.api.debug, "Debug mode should be disabled in production"
        
        # Validate file paths
        if settings.langgraph.checkpoint_backend == "sqlite":
            checkpoint_dir = Path(settings.langgraph.checkpoint_path).parent
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate memory limits
        assert settings.langgraph.memory_limit_mb > 0, "Memory limit must be positive"
        assert settings.langgraph.execution_timeout > 0, "Execution timeout must be positive"

# Usage
def get_settings() -> Settings:
    """Get application settings"""
    loader = ConfigLoader()
    return loader.load_config()
```

### 12.3 Service Layer Architecture

#### Database Service

```python
# src/services/database.py
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import logging

from ..config.settings import Settings

logger = logging.getLogger(__name__)

class DatabaseService:
    """Database service for managing connections and sessions"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = None
        self.SessionLocal = None
        self._initialize()
    
    def _initialize(self):
        """Initialize database connection"""
        db_config = self.settings.database
        
        database_url = (
            f"postgresql://{db_config.username}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.name}"
        )
        
        self.engine = create_engine(
            database_url,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_pre_ping=True,
            echo=self.settings.debug
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info("Database service initialized")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

# Models
Base = declarative_base()

# Usage in nodes
class DatabaseNode(BaseNode):
    """Base class for nodes that need database access"""
    
    def __init__(self, config: NodeConfig, db_service: DatabaseService):
        super().__init__(config)
        self.db_service = db_service
    
    def execute_with_db(self, state: TypedDict, db_operation):
        """Execute database operation with proper session management"""
        with self.db_service.get_session() as session:
            return db_operation(session, state)
```

#### External API Service

```python
# src/services/external_apis.py
import httpx
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from ..config.settings import Settings

logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Standardized API response"""
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    status_code: int
    response_time: float

class ExternalAPIService:
    """Service for managing external API calls"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.AsyncClient(
            timeout=settings.external_service_timeout,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=100)
        )
        self._rate_limits = {}
    
    async def call_api(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        retries: Optional[int] = None
    ) -> APIResponse:
        """Make API call with retry logic and rate limiting"""
        
        retries = retries or self.settings.external_service_retries
        start_time = datetime.now()
        
        for attempt in range(retries + 1):
            try:
                # Check rate limits
                if not self._check_rate_limit(url):
                    await asyncio.sleep(1)
                    continue
                
                # Make the request
                response = await self.client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data
                )
                
                response_time = (datetime.now() - start_time).total_seconds()
                
                if response.is_success:
                    return APIResponse(
                        success=True,
                        data=response.json() if response.content else None,
                        error=None,
                        status_code=response.status_code,
                        response_time=response_time
                    )
                else:
                    logger.warning(f"API call failed with status {response.status_code}: {url}")
                    
                    if attempt == retries:
                        return APIResponse(
                            success=False,
                            data=None,
                            error=f"HTTP {response.status_code}: {response.text}",
                            status_code=response.status_code,
                            response_time=response_time
                        )
            
            except Exception as e:
                logger.error(f"API call attempt {attempt + 1} failed: {e}")
                
                if attempt == retries:
                    response_time = (datetime.now() - start_time).total_seconds()
                    return APIResponse(
                        success=False,
                        data=None,
                        error=str(e),
                        status_code=0,
                        response_time=response_time
                    )
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        # Should never reach here
        return APIResponse(
            success=False,
            data=None,
            error="Unexpected error",
            status_code=0,
            response_time=0
        )
    
    def _check_rate_limit(self, url: str) -> bool:
        """Simple rate limiting implementation"""
        now = datetime.now()
        
        if url not in self._rate_limits:
            self._rate_limits[url] = []
        
        # Remove old entries (older than 1 minute)
        self._rate_limits[url] = [
            timestamp for timestamp in self._rate_limits[url]
            if now - timestamp < timedelta(minutes=1)
        ]
        
        # Check if under limit (60 requests per minute)
        if len(self._rate_limits[url]) < 60:
            self._rate_limits[url].append(now)
            return True
        
        return False
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
```

### 12.4 API Layer Implementation

#### FastAPI Application Structure

```python
# src/api/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from ..config.settings import get_settings
from ..services.database import DatabaseService
from ..services.external_apis import ExternalAPIService
from .routes import health, processing, analysis
from .middleware.error_handling import ErrorHandlingMiddleware
from .middleware.logging import LoggingMiddleware
from ..monitoring.metrics import setup_metrics

settings = get_settings()
logger = logging.getLogger(__name__)

# Global services
db_service = None
api_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global db_service, api_service
    
    # Startup
    logger.info("Starting LangGraph API application")
    
    # Initialize services
    db_service = DatabaseService(settings)
    api_service = ExternalAPIService(settings)
    
    # Setup monitoring
    if settings.monitoring.metrics_enabled:
        setup_metrics(app)
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    if api_service:
        await api_service.close()
    
    logger.info("Application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="LangGraph Application API",
    description="Production-ready LangGraph application",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(processing.router, prefix="/api/v1/processing", tags=["processing"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

# Dependency injection
def get_db_service() -> DatabaseService:
    return db_service

def get_api_service() -> ExternalAPIService:
    return api_service
```

#### API Route Implementation

```python
# src/api/routes/processing.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

from ...services.database import DatabaseService
from ...services.external_apis import ExternalAPIService
from ...core.graphs.processing_graph import ProcessingGraph
from ...config.settings import get_settings
from ..models.requests import ProcessingRequest
from ..models.responses import ProcessingResponse, TaskStatusResponse
from ..main import get_db_service, get_api_service

router = APIRouter()
settings = get_settings()

# In-memory task storage (in production, use Redis or database)
task_storage = {}

@router.post("/submit", response_model=ProcessingResponse)
async def submit_processing_task(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
    db_service: DatabaseService = Depends(get_db_service),
    api_service: ExternalAPIService = Depends(get_api_service)
):
    """Submit a processing task for execution"""
    
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Store task info
        task_storage[task_id] = {
            "status": "submitted",
            "created_at": datetime.now().isoformat(),
            "request": request.dict(),
            "result": None,
            "error": None
        }
        
        # Add background task
        background_tasks.add_task(
            execute_processing_task,
            task_id,
            request,
            db_service,
            api_service
        )
        
        return ProcessingResponse(
            task_id=task_id,
            status="submitted",
            message="Task submitted for processing"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a processing task"""
    
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = task_storage[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        created_at=task_info["created_at"],
        completed_at=task_info.get("completed_at"),
        result=task_info.get("result"),
        error=task_info.get("error")
    )

async def execute_processing_task(
    task_id: str,
    request: ProcessingRequest,
    db_service: DatabaseService,
    api_service: ExternalAPIService
):
    """Execute processing task in background"""
    
    try:
        # Update task status
        task_storage[task_id]["status"] = "processing"
        
        # Create processing graph
        processing_graph = ProcessingGraph(
            config=settings.langgraph.__dict__,
            db_service=db_service,
            api_service=api_service
        )
        
        # Execute graph
        initial_state = {
            "task_id": task_id,
            "input_data": request.data,
            "processing_config": request.config or {},
            "result": None,
            "errors": []
        }
        
        result = await processing_graph.ainvoke(initial_state)
        
        # Update task with result
        task_storage[task_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "result": result
        })
        
    except Exception as e:
        # Update task with error
        task_storage[task_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })

@router.get("/tasks", response_model=list[TaskStatusResponse])
async def list_tasks(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None
):
    """List processing tasks with pagination and filtering"""
    
    tasks = list(task_storage.items())
    
    # Filter by status if provided
    if status:
        tasks = [(task_id, info) for task_id, info in tasks if info["status"] == status]
    
    # Apply pagination
    paginated_tasks = tasks[offset:offset + limit]
    
    return [
        TaskStatusResponse(
            task_id=task_id,
            status=info["status"],
            created_at=info["created_at"],
            completed_at=info.get("completed_at"),
            result=info.get("result"),
            error=info.get("error")
        )
        for task_id, info in paginated_tasks
    ]
```

### 12.5 Testing Strategy Implementation

#### Test Configuration

```python
# tests/conftest.py
import pytest
import asyncio
from unittest.mock import Mock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from src.config.settings import Settings
from src.services.database import DatabaseService, Base
from src.services.external_apis import ExternalAPIService
from src.api.main import app

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_settings():
    """Test settings configuration"""
    return Settings(
        environment="testing",
        debug=True,
        database={
            "host": "localhost",
            "port": 5432,
            "name": "test_langgraph",
            "username": "test_user",
            "password": "test_pass"
        },
        langgraph={
            "checkpoint_backend": "memory",
            "memory_limit_mb": 256,
            "execution_timeout": 30
        }
    )

@pytest.fixture
def test_db_service(test_settings):
    """Test database service with in-memory SQLite"""
    # Override with in-memory database for testing
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    # Mock the database service
    db_service = Mock(spec=DatabaseService)
    db_service.settings = test_settings
    db_service.engine = engine
    db_service.SessionLocal = sessionmaker(bind=engine)
    
    return db_service

@pytest.fixture
def mock_api_service():
    """Mock external API service"""
    api_service = Mock(spec=ExternalAPIService)
    
    # Configure mock responses
    api_service.call_api.return_value = {
        "success": True,
        "data": {"result": "test_response"},
        "error": None,
        "status_code": 200,
        "response_time": 0.1
    }
    
    return api_service

@pytest.fixture
def test_client(test_db_service, mock_api_service):
    """Test client with mocked dependencies"""
    
    # Override dependencies
    app.dependency_overrides[get_db_service] = lambda: test_db_service
    app.dependency_overrides[get_api_service] = lambda: mock_api_service
    
    with TestClient(app) as client:
        yield client
    
    # Clean up
    app.dependency_overrides.clear()

@pytest.fixture
def sample_processing_data():
    """Sample data for processing tests"""
    return {
        "data": {
            "text": "This is a sample text for processing",
            "metadata": {"source": "test", "priority": "high"}
        },
        "config": {
            "processing_type": "standard",
            "timeout": 30
        }
    }
```

Great progress! I've successfully added a comprehensive Real-World Project Structure section that covers:

- **Enterprise-Grade Directory Structure**: Complete production-ready project organization
- **Core Module Implementation**: Base classes for graphs and nodes with proper abstraction
- **Configuration Management**: Environment-based config with validation and loading
- **Service Layer Architecture**: Database and external API services with proper error handling
- **API Layer Implementation**: FastAPI with proper middleware, routing, and dependency injection
- **Testing Strategy**: Comprehensive test setup with fixtures and mocking

This provides developers with a solid foundation for building production-ready LangGraph applications with proper separation of concerns, scalability, and maintainability.

---

## 13. Testing Strategies

Testing is crucial for building reliable LangGraph applications. This section covers comprehensive testing approaches from unit tests to end-to-end testing, performance testing, and continuous integration strategies.

### 13.1 Test Structure Overview

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_nodes.py       # Test individual nodes
│   ├── test_routing.py     # Test routing functions
│   ├── test_state.py       # Test state management
│   └── test_utils.py       # Test utility functions
├── integration/             # Integration tests
│   ├── test_graph_flows.py # Test complete graph flows
│   ├── test_memory.py      # Test memory/checkpointing
│   └── test_llm_integration.py # Test LLM integrations
├── e2e/                    # End-to-end tests
│   ├── test_complete_workflows.py
│   └── test_performance.py
├── fixtures/               # Test fixtures and data
│   ├── sample_data.py
│   └── mock_responses.py
└── conftest.py            # Pytest configuration
```

### Setting Up Testing Framework

Create `tests/conftest.py`:

```python
"""
Pytest configuration and fixtures for LangGraph testing
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, patch
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Test fixtures

@pytest.fixture
def sample_state():
    """Basic state fixture for testing"""
    return {
        "input": "test input",
        "output": "",
        "processing_steps": [],
        "metadata": {}
    }

@pytest.fixture
def memory_saver():
    """Memory saver fixture for testing persistence"""
    return MemorySaver()

@pytest.fixture
def mock_llm():
    """Mock LLM for testing without API calls"""
    mock = Mock()
    mock.invoke.return_value = Mock(content="Mocked response")
    return mock

@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {"configurable": {"thread_id": "test-thread-123"}}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Helper functions for testing

class GraphTestHelper:
    """Helper class for testing graphs"""
    
    @staticmethod
    def create_simple_test_graph(node_func, state_class):
        """Create a simple graph for testing"""
        graph = StateGraph(state_class)
        graph.add_node("test_node", node_func)
        graph.add_edge("__start__", "test_node")
        graph.add_edge("test_node", "__end__")
        return graph.compile()
    
    @staticmethod
    def assert_state_contains(result_state: Dict[str, Any], expected_keys: List[str]):
        """Assert that state contains expected keys"""
        for key in expected_keys:
            assert key in result_state, f"Key '{key}' not found in state"
    
    @staticmethod
    def assert_state_values(result_state: Dict[str, Any], expected_values: Dict[str, Any]):
        """Assert that state contains expected values"""
        for key, expected_value in expected_values.items():
            assert key in result_state, f"Key '{key}' not found in state"
            assert result_state[key] == expected_value, f"Expected {expected_value}, got {result_state[key]}"

@pytest.fixture
def graph_helper():
    """Graph testing helper fixture"""
    return GraphTestHelper()

# Markers for different test categories

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
```

### Unit Testing: Testing Individual Nodes

Create `tests/unit/test_nodes.py`:

```python
"""
Unit tests for individual LangGraph nodes
"""

import pytest
from typing import TypedDict, List
from unittest.mock import Mock, patch
from datetime import datetime

# Import your nodes (adjust imports based on your structure)
# from src.nodes.processing import data_processing_node, validation_node
# from src.nodes.analysis import content_analysis_node

class TestState(TypedDict):
    """Test state for unit testing"""
    input: str
    output: str
    metadata: dict
    processing_steps: List[str]

# Example node for testing
def sample_processing_node(state: TestState) -> TestState:
    """Sample node for testing purposes"""
    input_text = state["input"]
    
    # Simple processing logic
    processed = input_text.upper().strip()
    
    metadata = state.get("metadata", {})
    metadata["processed_at"] = datetime.now().isoformat()
    metadata["processing_applied"] = True
    
    steps = state.get("processing_steps", [])
    steps.append("sample_processing_completed")
    
    return {
        **state,
        "output": processed,
        "metadata": metadata,
        "processing_steps": steps
    }

@pytest.mark.unit
class TestSampleProcessingNode:
    """Test cases for sample processing node"""
    
    def test_basic_processing(self):
        """Test basic node processing"""
        # Arrange
        input_state = {
            "input": "  hello world  ",
            "output": "",
            "metadata": {},
            "processing_steps": []
        }
        
        # Act
        result = sample_processing_node(input_state)
        
        # Assert
        assert result["output"] == "HELLO WORLD"
        assert result["input"] == "  hello world  "  # Input should be unchanged
        assert "processed_at" in result["metadata"]
        assert result["metadata"]["processing_applied"] == True
        assert "sample_processing_completed" in result["processing_steps"]
    
    def test_empty_input(self):
        """Test node behavior with empty input"""
        # Arrange
        input_state = {
            "input": "",
            "output": "",
            "metadata": {},
            "processing_steps": []
        }
        
        # Act
        result = sample_processing_node(input_state)
        
        # Assert
        assert result["output"] == ""
        assert result["metadata"]["processing_applied"] == True
    
    def test_preserves_existing_metadata(self):
        """Test that node preserves existing metadata"""
        # Arrange
        existing_metadata = {"previous_step": "validation", "user_id": "123"}
        input_state = {
            "input": "test",
            "output": "",
            "metadata": existing_metadata,
            "processing_steps": ["previous_step"]
        }
        
        # Act
        result = sample_processing_node(input_state)
        
        # Assert
        assert result["metadata"]["previous_step"] == "validation"
        assert result["metadata"]["user_id"] == "123"
        assert result["metadata"]["processing_applied"] == True
        assert len(result["processing_steps"]) == 2
    
    def test_node_performance(self):
        """Test node performance with large input"""
        # Arrange
        large_input = "test data " * 1000
        input_state = {
            "input": large_input,
            "output": "",
            "metadata": {},
            "processing_steps": []
        }
        
        # Act
        import time
        start_time = time.time()
        result = sample_processing_node(input_state)
        end_time = time.time()
        
        # Assert
        processing_time = end_time - start_time
        assert processing_time < 1.0  # Should complete within 1 second
        assert len(result["output"]) == len(large_input)

# Parametrized testing for different scenarios
@pytest.mark.unit
@pytest.mark.parametrize("input_text,expected_output", [
    ("hello", "HELLO"),
    ("  world  ", "WORLD"),
    ("Hello World", "HELLO WORLD"),
    ("", ""),
    ("123", "123"),
    ("Hello123World", "HELLO123WORLD")
])
def test_processing_node_variations(input_text, expected_output):
    """Test processing node with various inputs"""
    input_state = {
        "input": input_text,
        "output": "",
        "metadata": {},
        "processing_steps": []
    }
    
    result = sample_processing_node(input_state)
    assert result["output"] == expected_output

# Testing error handling
@pytest.mark.unit
def test_node_error_handling():
    """Test node behavior with invalid state"""
    # Test with missing required fields
    with pytest.raises(KeyError):
        invalid_state = {"wrong_key": "value"}
        sample_processing_node(invalid_state)

# Mock testing for external dependencies
@pytest.mark.unit
@patch('datetime.datetime')
def test_node_with_mocked_datetime(mock_datetime):
    """Test node with mocked external dependencies"""
    # Arrange
    fixed_time = datetime(2024, 1, 15, 10, 0, 0)
    mock_datetime.now.return_value = fixed_time
    
    input_state = {
        "input": "test",
        "output": "",
        "metadata": {},
        "processing_steps": []
    }
    
    # Act
    result = sample_processing_node(input_state)
    
    # Assert
    assert result["metadata"]["processed_at"] == fixed_time.isoformat()
```

### Integration Testing: Testing Complete Flows

Create `tests/integration/test_graph_flows.py`:

```python
"""
Integration tests for complete graph flows
"""

import pytest
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class IntegrationTestState(TypedDict):
    """State for integration testing"""
    user_input: str
    processed_input: str
    analysis_results: Dict[str, Any]
    final_output: str
    processing_history: List[str]

# Create test nodes
def input_processor(state: IntegrationTestState) -> IntegrationTestState:
    """Process input text"""
    processed = state["user_input"].strip().lower()
    history = state["processing_history"] + ["input_processed"]
    
    return {
        **state,
        "processed_input": processed,
        "processing_history": history
    }

def content_analyzer(state: IntegrationTestState) -> IntegrationTestState:
    """Analyze processed content"""
    content = state["processed_input"]
    
    analysis = {
        "word_count": len(content.split()),
        "char_count": len(content),
        "contains_question": "?" in content,
        "analysis_timestamp": "2024-01-15T10:00:00"
    }
    
    history = state["processing_history"] + ["content_analyzed"]
    
    return {
        **state,
        "analysis_results": analysis,
        "processing_history": history
    }

def output_generator(state: IntegrationTestState) -> IntegrationTestState:
    """Generate final output"""
    analysis = state["analysis_results"]
    
    output = f"Analysis complete: {analysis['word_count']} words processed"
    if analysis["contains_question"]:
        output += " (Question detected)"
    
    history = state["processing_history"] + ["output_generated"]
    
    return {
        **state,
        "final_output": output,
        "processing_history": history
    }

@pytest.fixture
def integration_graph():
    """Create integration test graph"""
    graph = StateGraph(IntegrationTestState)
    
    # Add nodes
    graph.add_node("process", input_processor)
    graph.add_node("analyze", content_analyzer)
    graph.add_node("generate", output_generator)
    
    # Create flow
    graph.add_edge(START, "process")
    graph.add_edge("process", "analyze")
    graph.add_edge("analyze", "generate")
    graph.add_edge("generate", END)
    
    return graph.compile()

@pytest.mark.integration
class TestCompleteGraphFlow:
    """Integration tests for complete graph execution"""
    
    def test_full_pipeline_execution(self, integration_graph):
        """Test complete pipeline from start to finish"""
        # Arrange
        initial_state = {
            "user_input": "  What is machine learning?  ",
            "processed_input": "",
            "analysis_results": {},
            "final_output": "",
            "processing_history": []
        }
        
        # Act
        result = integration_graph.invoke(initial_state)
        
        # Assert
        assert result["processed_input"] == "what is machine learning?"
        assert result["analysis_results"]["word_count"] == 4
        assert result["analysis_results"]["contains_question"] == True
        assert "Question detected" in result["final_output"]
        assert len(result["processing_history"]) == 3
        assert result["processing_history"] == [
            "input_processed", "content_analyzed", "output_generated"
        ]
    
    def test_pipeline_with_statement(self, integration_graph):
        """Test pipeline with statement (no question)"""
        # Arrange
        initial_state = {
            "user_input": "Machine learning is fascinating",
            "processed_input": "",
            "analysis_results": {},
            "final_output": "",
            "processing_history": []
        }
        
        # Act
        result = integration_graph.invoke(initial_state)
        
        # Assert
        assert result["analysis_results"]["contains_question"] == False
        assert "Question detected" not in result["final_output"]
        assert result["analysis_results"]["word_count"] == 4
    
    def test_pipeline_with_empty_input(self, integration_graph):
        """Test pipeline behavior with empty input"""
        # Arrange
        initial_state = {
            "user_input": "",
            "processed_input": "",
            "analysis_results": {},
            "final_output": "",
            "processing_history": []
        }
        
        # Act
        result = integration_graph.invoke(initial_state)
        
        # Assert
        assert result["processed_input"] == ""
        assert result["analysis_results"]["word_count"] == 0
        assert "0 words processed" in result["final_output"]
    
    def test_state_persistence_between_nodes(self, integration_graph):
        """Test that state is properly maintained between nodes"""
        initial_state = {
            "user_input": "Test input for persistence",
            "processed_input": "",
            "analysis_results": {},
            "final_output": "",
            "processing_history": []
        }
        
        result = integration_graph.invoke(initial_state)
        
        # Verify state evolution
        assert result["user_input"] == "Test input for persistence"  # Original preserved
        assert result["processed_input"] == "test input for persistence"  # Processed version
        assert isinstance(result["analysis_results"], dict)  # Analysis added
        assert result["final_output"] != ""  # Output generated
        assert len(result["processing_history"]) == 3  # All steps recorded

# Testing with memory/checkpointing
@pytest.mark.integration
def test_graph_with_memory(integration_graph, memory_saver):
    """Test graph execution with memory persistence"""
    # Compile graph with memory
    graph_with_memory = integration_graph.__class__(integration_graph.nodes)
    # Note: This is simplified - in practice you'd need to properly set up memory
    
    config = {"configurable": {"thread_id": "test-session-1"}}
    
    initial_state = {
        "user_input": "Test with memory",
        "processed_input": "",
        "analysis_results": {},
        "final_output": "",
        "processing_history": []
    }
    
    # First execution
    result1 = integration_graph.invoke(initial_state)
    
    # Verify results
    assert result1["final_output"] != ""
    assert len(result1["processing_history"]) == 3

# Stress testing
@pytest.mark.integration
@pytest.mark.slow
def test_graph_performance_with_large_input(integration_graph):
    """Test graph performance with large inputs"""
    import time
    
    # Create large input
    large_input = " ".join(["word"] * 1000)
    
    initial_state = {
        "user_input": large_input,
        "processed_input": "",
        "analysis_results": {},
        "final_output": "",
        "processing_history": []
    }
    
    # Measure execution time
    start_time = time.time()
    result = integration_graph.invoke(initial_state)
    execution_time = time.time() - start_time
    
    # Assert performance criteria
    assert execution_time < 5.0  # Should complete within 5 seconds
    assert result["analysis_results"]["word_count"] == 1000
    assert result["final_output"] != ""

# Error handling in integration
@pytest.mark.integration
def test_graph_error_propagation():
    """Test how errors propagate through the graph"""
    
    def failing_node(state):
        """Node that always fails"""
        raise ValueError("Simulated processing error")
    
    # Create graph with failing node
    error_graph = StateGraph(IntegrationTestState)
    error_graph.add_node("fail", failing_node)
    error_graph.add_edge(START, "fail")
    error_graph.add_edge("fail", END)
    
    compiled_graph = error_graph.compile()
    
    initial_state = {
        "user_input": "test",
        "processed_input": "",
        "analysis_results": {},
        "final_output": "",
        "processing_history": []
    }
    
    # Should raise the error from the failing node
    with pytest.raises(ValueError, match="Simulated processing error"):
        compiled_graph.invoke(initial_state)
```

### End-to-End Testing

Create `tests/e2e/test_complete_workflows.py`:

```python
"""
End-to-end tests for complete LangGraph workflows
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

# Import your complete applications
# from src.examples.ai_powered import create_ai_graph
# from src.graphs.branching_example import create_content_router

@pytest.mark.e2e
class TestCompleteWorkflows:
    """End-to-end tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_ai_powered_workflow_end_to_end(self, mock_llm):
        """Test complete AI-powered workflow"""
        # This would test your actual AI-powered graph
        # with proper mocking of external services
        
        # Mock the LLM response
        mock_llm.invoke.return_value.content = "This is a mocked AI response about machine learning."
        
        # Test data
        test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain deep learning concepts"
        ]
        
        for query in test_queries:
            # Simulate running your AI workflow
            result = {
                "user_query": query,
                "ai_response": f"💡 {mock_llm.invoke.return_value.content}",
                "processing_steps": ["Messages prepared", "LLM processing completed", "Post-processing completed"]
            }
            
            # Assertions
            assert result["user_query"] == query
            assert "machine learning" in result["ai_response"].lower()
            assert len(result["processing_steps"]) == 3
    
    def test_content_router_end_to_end(self):
        """Test complete content routing workflow"""
        # Test different content types through the router
        test_cases = [
            {
                "content": "def hello(): print('world')",
                "expected_type": "code",
                "expected_confidence_range": (0.4, 1.0)
            },
            {
                "content": "What is the capital of France?",
                "expected_type": "question", 
                "expected_confidence_range": (0.8, 1.0)
            },
            {
                "content": "Please create a web application",
                "expected_type": "instruction",
                "expected_confidence_range": (0.3, 1.0)
            },
            {
                "content": "I enjoyed the movie yesterday",
                "expected_type": "general",
                "expected_confidence_range": (0.1, 1.0)
            }
        ]
        
        for test_case in test_cases:
            # Simulate content router processing
            # This would use your actual content router
            result = {
                "content_type": test_case["expected_type"],
                "confidence": 0.8,  # Simulated
                "processing_path": f"{test_case['expected_type']}_processing",
                "final_output": f"Processed as {test_case['expected_type']}"
            }
            
            # Assertions
            assert result["content_type"] == test_case["expected_type"]
            assert test_case["expected_confidence_range"][0] <= result["confidence"] <= test_case["expected_confidence_range"][1]
            assert result["processing_path"].startswith(test_case["expected_type"])
    
    def test_data_pipeline_end_to_end(self):
        """Test complete data processing pipeline"""
        # Generate test data
        test_data = [
            {"id": "1", "timestamp": "2024-01-01T10:00:00", "value": 100},
            {"id": "2", "timestamp": "2024-01-01T11:00:00", "value": 200},
            {"id": "3", "value": 300},  # Missing timestamp
            {"id": "4", "timestamp": "2024-01-01T13:00:00", "value": "invalid"},  # Invalid value
        ]
        
        # Simulate pipeline execution
        expected_result = {
            "validation_results": {
                "total_records": 4,
                "valid_records": 2,
                "invalid_records": 2
            },
            "aggregated_data": {
                "total_records": 2,  # Only valid records processed
                "average_value": 150.0,
                "max_value": 200,
                "min_value": 100
            }
        }
        
        # Assertions
        assert expected_result["validation_results"]["valid_records"] == 2
        assert expected_result["aggregated_data"]["average_value"] == 150.0
    
    def test_workflow_with_file_io(self):
        """Test workflow with file input/output"""
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "input.txt"
            output_file = Path(temp_dir) / "output.json"
            
            # Create test input file
            test_content = "This is test content for file processing workflow."
            input_file.write_text(test_content)
            
            # Simulate file processing workflow
            # Read input
            content = input_file.read_text()
            
            # Process (simplified)
            result = {
                "input_file": str(input_file),
                "content_length": len(content),
                "word_count": len(content.split()),
                "processing_status": "completed"
            }
            
            # Write output
            output_file.write_text(json.dumps(result, indent=2))
            
            # Verify file operations
            assert input_file.exists()
            assert output_file.exists()
            
            # Verify output content
            output_data = json.loads(output_file.read_text())
            assert output_data["word_count"] == 9
            assert output_data["processing_status"] == "completed"
    
    @pytest.mark.slow
    def test_concurrent_workflow_execution(self):
        """Test multiple workflows running concurrently"""
        import threading
        import time
        
        results = []
        
        def run_workflow(workflow_id):
            """Simulate running a workflow"""
            # Simulate processing time
            time.sleep(0.1)
            
            result = {
                "workflow_id": workflow_id,
                "status": "completed",
                "execution_time": 0.1,
                "timestamp": time.time()
            }
            results.append(result)
        
        # Start multiple workflows concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_workflow, args=(f"workflow_{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all workflows to complete
        for thread in threads:
            thread.join(timeout=5)
        
        # Verify all workflows completed
        assert len(results) == 5
        assert all(r["status"] == "completed" for r in results)
        
        # Verify concurrent execution (should complete faster than sequential)
        total_time = max(r["timestamp"] for r in results) - min(r["timestamp"] for r in results)
        assert total_time < 0.5  # Should complete much faster than 5 * 0.1 = 0.5s

# Performance testing
@pytest.mark.e2e
@pytest.mark.slow
def test_workflow_performance_benchmarks():
    """Test workflow performance benchmarks"""
    import time
    
    # Define performance benchmarks
    benchmarks = {
        "simple_workflow": {"max_time": 1.0, "iterations": 10},
        "complex_workflow": {"max_time": 5.0, "iterations": 5},
        "data_processing": {"max_time": 2.0, "iterations": 3}
    }
    
    for workflow_name, benchmark in benchmarks.items():
        times = []
        
        for i in range(benchmark["iterations"]):
            start_time = time.time()
            
            # Simulate workflow execution
            time.sleep(0.1)  # Simulated processing
            
            end_time = time.time()
            execution_time = end_time - start_time
            times.append(execution_time)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Assert performance requirements
        assert avg_time < benchmark["max_time"], f"{workflow_name} average time {avg_time:.2f}s exceeds limit {benchmark['max_time']}s"
        assert max_time < benchmark["max_time"] * 1.5, f"{workflow_name} max time {max_time:.2f}s exceeds tolerance"
        
        print(f"✅ {workflow_name}: avg={avg_time:.2f}s, max={max_time:.2f}s")

# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

### Performance and Load Testing

Create `tests/performance/test_load_testing.py`:

```python
"""
Load testing for LangGraph applications
"""

import pytest
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

class PerformanceMetrics:
    """Collect and analyze performance metrics"""
    
    def __init__(self):
        self.execution_times: List[float] = []
        self.success_count = 0
        self.error_count = 0
        self.start_time = None
        self.end_time = None
    
    def add_execution_time(self, duration: float, success: bool = True):
        """Add execution time measurement"""
        self.execution_times.append(duration)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def start_measurement(self):
        """Start overall measurement"""
        self.start_time = time.time()
    
    def end_measurement(self):
        """End overall measurement"""
        self.end_time = time.time()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.execution_times:
            return {"error": "No execution times recorded"}
        
        return {
            "total_executions": len(self.execution_times),
            "successful_executions": self.success_count,
            "failed_executions": self.error_count,
            "success_rate": self.success_count / len(self.execution_times) * 100,
            "average_time": statistics.mean(self.execution_times),
            "median_time": statistics.median(self.execution_times),
            "min_time": min(self.execution_times),
            "max_time": max(self.execution_times),
            "std_deviation": statistics.stdev(self.execution_times) if len(self.execution_times) > 1 else 0,
            "total_duration": self.end_time - self.start_time if self.start_time and self.end_time else 0,
            "throughput_per_second": len(self.execution_times) / (self.end_time - self.start_time) if self.start_time and self.end_time else 0
        }

def simulate_graph_execution(execution_id: int, complexity: str = "simple") -> Dict[str, Any]:
    """Simulate graph execution for load testing"""
    start_time = time.time()
    
    try:
        # Simulate different complexity levels
        if complexity == "simple":
            time.sleep(0.01)  # 10ms processing
        elif complexity == "medium":
            time.sleep(0.05)  # 50ms processing
        elif complexity == "complex":
            time.sleep(0.1)   # 100ms processing
        else:
            time.sleep(0.02)  # Default
        
        # Simulate occasional errors
        if execution_id % 50 == 0:  # 2% error rate
            raise Exception("Simulated processing error")
        
        end_time = time.time()
        
        return {
            "execution_id": execution_id,
            "success": True,
            "duration": end_time - start_time,
            "result": f"Processed execution {execution_id}"
        }
    
    except Exception as e:
        end_time = time.time()
        return {
            "execution_id": execution_id,
            "success": False,
            "duration": end_time - start_time,
            "error": str(e)
        }

@pytest.mark.slow
@pytest.mark.performance
class TestLoadTesting:
    """Load testing for LangGraph applications"""
    
    def test_single_thread_performance(self):
        """Test performance with single-threaded execution"""
        metrics = PerformanceMetrics()
        metrics.start_measurement()
        
        # Execute 100 operations sequentially
        for i in range(100):
            result = simulate_graph_execution(i, "simple")
            metrics.add_execution_time(result["duration"], result["success"])
        
        metrics.end_measurement()
        stats = metrics.get_statistics()
        
        # Performance assertions
        assert stats["success_rate"] >= 95.0  # At least 95% success rate
        assert stats["average_time"] < 0.02   # Average under 20ms
        assert stats["max_time"] < 0.1        # No execution over 100ms
        
        print(f"Single-thread stats: {stats}")
    
    def test_concurrent_execution_performance(self):
        """Test performance with concurrent execution"""
        metrics = PerformanceMetrics()
        metrics.start_measurement()
        
        # Execute with thread pool
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit 200 tasks
            futures = [
                executor.submit(simulate_graph_execution, i, "simple")
                for i in range(200)
            ]
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                metrics.add_execution_time(result["duration"], result["success"])
        
        metrics.end_measurement()
        stats = metrics.get_statistics()
        
        # Performance assertions for concurrent execution
        assert stats["success_rate"] >= 95.0
        assert stats["throughput_per_second"] > 50  # At least 50 ops/second
        assert stats["total_duration"] < 10        # Complete within 10 seconds
        
        print(f"Concurrent stats: {stats}")
    
    def test_stress_testing_with_high_load(self):
        """Test system behavior under high load"""
        metrics = PerformanceMetrics()
        metrics.start_measurement()
        
        # High load test with more threads and operations
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(simulate_graph_execution, i, "medium")
                for i in range(500)  # 500 operations
            ]
            
            for future in as_completed(futures):
                result = future.result()
                metrics.add_execution_time(result["duration"], result["success"])
        
        metrics.end_measurement()
        stats = metrics.get_statistics()
        
        # Stress test assertions (more lenient)
        assert stats["success_rate"] >= 90.0      # At least 90% under stress
        assert stats["average_time"] < 0.2        # Average under 200ms
        assert stats["throughput_per_second"] > 20 # At least 20 ops/second
        
        print(f"Stress test stats: {stats}")
    
    def test_memory_usage_under_load(self):
        """Test memory usage during load testing"""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run load test
        metrics = PerformanceMetrics()
        
        for batch in range(5):  # 5 batches of 100 operations
            batch_metrics = PerformanceMetrics()
            batch_metrics.start_measurement()
            
            for i in range(100):
                result = simulate_graph_execution(batch * 100 + i, "simple")
                batch_metrics.add_execution_time(result["duration"], result["success"])
                metrics.add_execution_time(result["duration"], result["success"])
            
            batch_metrics.end_measurement()
            
            # Check memory after each batch
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            
            # Memory should not increase excessively
            memory_increase_mb = memory_increase / (1024 * 1024)
            assert memory_increase_mb < 100, f"Memory increased by {memory_increase_mb:.2f}MB after batch {batch}"
            
            # Force garbage collection
            gc.collect()
        
        final_memory = process.memory_info().rss
        total_memory_increase = (final_memory - initial_memory) / (1024 * 1024)
        
        print(f"Total memory increase: {total_memory_increase:.2f}MB")
        assert total_memory_increase < 50  # Should not increase by more than 50MB
    
    def test_performance_degradation_over_time(self):
        """Test for performance degradation over extended periods"""
        all_metrics = []
        
        # Run multiple batches and track performance over time
        for batch in range(10):
            batch_metrics = PerformanceMetrics()
            batch_metrics.start_measurement()
            
            # Each batch has 50 operations
            for i in range(50):
                result = simulate_graph_execution(batch * 50 + i, "simple")
                batch_metrics.add_execution_time(result["duration"], result["success"])
            
            batch_metrics.end_measurement()
            stats = batch_metrics.get_statistics()
            all_metrics.append(stats)
            
            print(f"Batch {batch + 1}: avg_time={stats['average_time']:.4f}s, throughput={stats['throughput_per_second']:.1f}/s")
        
        # Check for performance degradation
        first_half_avg = statistics.mean([m["average_time"] for m in all_metrics[:5]])
        second_half_avg = statistics.mean([m["average_time"] for m in all_metrics[5:]])
        
        performance_degradation = (second_half_avg - first_half_avg) / first_half_avg * 100
        
        # Performance should not degrade by more than 20%
        assert performance_degradation < 20, f"Performance degraded by {performance_degradation:.1f}%"
        
        print(f"Performance change over time: {performance_degradation:.1f}%")

# Benchmark testing
@pytest.mark.benchmark
def test_benchmark_different_complexities():
    """Benchmark different workflow complexities"""
    complexities = ["simple", "medium", "complex"]
    results = {}
    
    for complexity in complexities:
        metrics = PerformanceMetrics()
        metrics.start_measurement()
        
        # Run 100 operations for each complexity
        for i in range(100):
            result = simulate_graph_execution(i, complexity)
            metrics.add_execution_time(result["duration"], result["success"])
        
        metrics.end_measurement()
        results[complexity] = metrics.get_statistics()
    
    # Print benchmark results
    print("\n📊 Complexity Benchmark Results:")
    print("-" * 50)
    for complexity, stats in results.items():
        print(f"{complexity.upper():>8}: avg={stats['average_time']:.4f}s, "
              f"throughput={stats['throughput_per_second']:.1f}/s")
    
    # Verify expected performance relationships
    assert results["simple"]["average_time"] < results["medium"]["average_time"]
    assert results["medium"]["average_time"] < results["complex"]["average_time"]

if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance"])
```

### Running Tests

Create a comprehensive test runner script `scripts/run_tests.sh`:

```bash
#!/bin/bash

# LangGraph Test Runner
# Runs comprehensive test suite with different categories

set -e

echo "🧪 LangGraph Comprehensive Test Suite"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
PYTEST_ARGS="--tb=short --strict-markers -v"

# Function to run test category
run_test_category() {
    local category=$1
    local description=$2
    local marker=$3
    local extra_args=${4:-""}
    
    echo -e "\n${BLUE}🔍 Running $description${NC}"
    echo "----------------------------------------"
    
    if pytest $PYTEST_ARGS -m "$marker" $extra_args; then
        echo -e "${GREEN}✅ $description passed${NC}"
    else
        echo -e "${RED}❌ $description failed${NC}"
        exit 1
    fi
}

# Parse command line arguments
QUICK_MODE=false
INCLUDE_SLOW=false
COVERAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --include-slow)
            INCLUDE_SLOW=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick        Run only fast tests"
            echo "  --include-slow Include slow tests"
            echo "  --coverage     Generate coverage report"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Add coverage arguments if requested
if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=src --cov-report=html --cov-report=term-missing"
fi

echo "Test Configuration:"
echo "- Quick mode: $QUICK_MODE"
echo "- Include slow tests: $INCLUDE_SLOW"
echo "- Coverage: $COVERAGE"
echo ""

# 1. Unit Tests
run_test_category "unit" "Unit Tests" "unit"

# 2. Integration Tests  
run_test_category "integration" "Integration Tests" "integration"

# 3. End-to-End Tests (skip in quick mode)
if [ "$QUICK_MODE" = false ]; then
    run_test_category "e2e" "End-to-End Tests" "e2e"
fi

# 4. Performance Tests (only if explicitly requested)
if [ "$INCLUDE_SLOW" = true ]; then
    run_test_category "performance" "Performance Tests" "performance"
fi

# Test Summary
echo -e "\n${GREEN}🎉 All tests passed successfully!${NC}"

# Coverage Report
if [ "$COVERAGE" = true ]; then
    echo -e "\n${BLUE}📊 Coverage Report:${NC}"
    echo "HTML coverage report generated at: htmlcov/index.html"
fi

echo -e "\n${YELLOW}💡 Test Tips:${NC}"
echo "- Use 'pytest tests/unit' to run only unit tests"
echo "- Use 'pytest -k test_name' to run specific tests"
echo "- Use 'pytest --lf' to run only last failed tests"
echo "- Use 'pytest --pdb' to drop into debugger on failures"
```

Make it executable:
```bash
chmod +x scripts/run_tests.sh
```

---

## 14. Performance Optimization

Performance is critical for production LangGraph applications. This section covers profiling, optimization techniques, scaling patterns, and best practices for high-performance graph execution.

### 14.1 Performance Profiling and Monitoring

#### Built-in Performance Monitoring

```python
import time
import asyncio
from typing import TypedDict, Dict, Any, List
from dataclasses import dataclass
from contextlib import asynccontextmanager
import psutil
import threading
from datetime import datetime

@dataclass
class NodePerformanceMetrics:
    """Performance metrics for a single node execution"""
    node_name: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    input_size_bytes: int
    output_size_bytes: int
    timestamp: str

@dataclass
class GraphPerformanceMetrics:
    """Overall graph performance metrics"""
    graph_name: str
    total_execution_time_ms: float
    node_metrics: List[NodePerformanceMetrics]
    memory_peak_mb: float
    memory_start_mb: float
    memory_end_mb: float
    total_nodes_executed: int
    
class PerformanceProfiler:
    """Performance profiler for LangGraph applications"""
    
    def __init__(self):
        self.node_metrics: List[NodePerformanceMetrics] = []
        self.graph_start_time: float = 0
        self.memory_start: float = 0
        self.memory_peak: float = 0
        
    def start_graph_profiling(self):
        """Start profiling a graph execution"""
        self.graph_start_time = time.time()
        self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_peak = self.memory_start
        self.node_metrics.clear()
    
    @asynccontextmanager
    async def profile_node(self, node_name: str, state: Dict[str, Any]):
        """Profile a single node execution"""
        import sys
        
        # Measure input size
        input_size = sys.getsizeof(str(state))
        
        # Start metrics
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Monitor CPU usage in background
        cpu_samples = []
        cpu_monitoring = True
        
        def monitor_cpu():
            while cpu_monitoring:
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
        
        cpu_thread = threading.Thread(target=monitor_cpu)
        cpu_thread.start()
        
        try:
            yield
            
            # Stop monitoring
            cpu_monitoring = False
            cpu_thread.join(timeout=1)
            
            # Calculate metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            execution_time = (end_time - start_time) * 1000  # ms
            
            # Update peak memory
            self.memory_peak = max(self.memory_peak, end_memory)
            
            # Calculate average CPU usage
            avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
            
            # Estimate output size (simplified)
            output_size = sys.getsizeof(str(state))
            
            # Record metrics
            metrics = NodePerformanceMetrics(
                node_name=node_name,
                execution_time_ms=execution_time,
                memory_usage_mb=end_memory - start_memory,
                cpu_usage_percent=avg_cpu,
                input_size_bytes=input_size,
                output_size_bytes=output_size,
                timestamp=datetime.now().isoformat()
            )
            
            self.node_metrics.append(metrics)
            
        except Exception:
            cpu_monitoring = False
            cpu_thread.join(timeout=1)
            raise
    
    def get_graph_metrics(self, graph_name: str) -> GraphPerformanceMetrics:
        """Get complete graph performance metrics"""
        total_time = (time.time() - self.graph_start_time) * 1000  # ms
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return GraphPerformanceMetrics(
            graph_name=graph_name,
            total_execution_time_ms=total_time,
            node_metrics=self.node_metrics.copy(),
            memory_peak_mb=self.memory_peak,
            memory_start_mb=self.memory_start,
            memory_end_mb=current_memory,
            total_nodes_executed=len(self.node_metrics)
        )
    
    def print_performance_report(self, metrics: GraphPerformanceMetrics):
        """Print a formatted performance report"""
        print(f"\n=== Performance Report: {metrics.graph_name} ===")
        print(f"Total Execution Time: {metrics.total_execution_time_ms:.2f} ms")
        print(f"Memory Usage: {metrics.memory_start_mb:.1f} → {metrics.memory_end_mb:.1f} MB (Peak: {metrics.memory_peak_mb:.1f} MB)")
        print(f"Nodes Executed: {metrics.total_nodes_executed}")
        
        print(f"\n=== Node Performance ===")
        for node in sorted(metrics.node_metrics, key=lambda x: x.execution_time_ms, reverse=True):
            print(f"{node.node_name:20} | {node.execution_time_ms:8.2f} ms | {node.memory_usage_mb:+6.1f} MB | CPU: {node.cpu_usage_percent:5.1f}%")

# Usage with existing graphs
class ProfiledNode(BaseNode):
    """Base node with built-in performance profiling"""
    
    def __init__(self, config: NodeConfig, profiler: PerformanceProfiler):
        super().__init__(config)
        self.profiler = profiler
    
    async def __call__(self, state: TypedDict) -> TypedDict:
        """Execute node with profiling"""
        async with self.profiler.profile_node(self.config.name, state):
            return await super().__call__(state)

# Example usage
async def run_profiled_graph():
    """Example of running a graph with performance profiling"""
    profiler = PerformanceProfiler()
    profiler.start_graph_profiling()
    
    # Your graph execution here
    # ... 
    
    # Get and display metrics
    metrics = profiler.get_graph_metrics("my_graph")
    profiler.print_performance_report(metrics)
    
    return metrics
```

#### Advanced Profiling with py-spy and Memory Profiler

```python
import subprocess
import os
import signal
from pathlib import Path
from typing import Optional
import tempfile

class AdvancedProfiler:
    """Advanced profiling using external tools"""
    
    def __init__(self, output_dir: str = "profiling_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._py_spy_pid: Optional[int] = None
    
    def start_py_spy_profiling(self, duration_seconds: int = 60):
        """Start py-spy CPU profiling"""
        output_file = self.output_dir / f"cpu_profile_{int(time.time())}.svg"
        
        cmd = [
            "py-spy", "record",
            "-o", str(output_file),
            "-d", str(duration_seconds),
            "-p", str(os.getpid())
        ]
        
        try:
            # Start py-spy in background
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._py_spy_pid = process.pid
            print(f"Started py-spy profiling (PID: {process.pid}), output: {output_file}")
            return output_file
        except FileNotFoundError:
            print("py-spy not found. Install with: pip install py-spy")
            return None
    
    def stop_py_spy_profiling(self):
        """Stop py-spy profiling"""
        if self._py_spy_pid:
            try:
                os.kill(self._py_spy_pid, signal.SIGTERM)
                print("Stopped py-spy profiling")
            except ProcessLookupError:
                print("py-spy process already terminated")
            finally:
                self._py_spy_pid = None
    
    def profile_memory_usage(self, func, *args, **kwargs):
        """Profile memory usage of a function"""
        try:
            from memory_profiler import profile as memory_profile
            
            # Create a temporary file for memory profile output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                temp_file = f.name
            
            # Run function with memory profiling
            @memory_profile(stream=open(temp_file, 'w+'))
            def profiled_func():
                return func(*args, **kwargs)
            
            result = profiled_func()
            
            # Read and display results
            with open(temp_file, 'r') as f:
                profile_output = f.read()
                print("Memory Usage Profile:")
                print(profile_output)
            
            # Clean up
            os.unlink(temp_file)
            
            return result
            
        except ImportError:
            print("memory_profiler not found. Install with: pip install memory-profiler")
            return func(*args, **kwargs)
    
    def create_flame_graph(self, perf_data_file: str):
        """Create flame graph from perf data"""
        try:
            flame_graph_file = self.output_dir / "flamegraph.svg"
            
            cmd = [
                "flamegraph.pl",
                perf_data_file,
                ">", str(flame_graph_file)
            ]
            
            subprocess.run(" ".join(cmd), shell=True, check=True)
            print(f"Flame graph created: {flame_graph_file}")
            return flame_graph_file
            
        except subprocess.CalledProcessError:
            print("Failed to create flame graph. Ensure flamegraph.pl is installed.")
            return None

# Example usage
async def profile_graph_execution():
    """Example of comprehensive graph profiling"""
    profiler = AdvancedProfiler()
    
    # Start CPU profiling
    py_spy_output = profiler.start_py_spy_profiling(duration_seconds=30)
    
    try:
        # Run your graph with memory profiling
        def run_graph():
            # Your graph execution code here
            time.sleep(1)  # Simulate work
            return "Graph execution complete"
        
        result = profiler.profile_memory_usage(run_graph)
        
        print(f"Graph execution result: {result}")
        
    finally:
        # Stop CPU profiling
        profiler.stop_py_spy_profiling()
        
        if py_spy_output:
            print(f"CPU profile saved to: {py_spy_output}")
```

### 14.2 Optimization Techniques

#### Node-Level Optimizations

```python
import asyncio
import concurrent.futures
from typing import TypedDict, List, Dict, Any
from functools import lru_cache, wraps
import time

class OptimizedState(TypedDict):
    data: List[str]
    processed_data: List[Dict[str, Any]]
    cache_hits: int
    cache_misses: int

# 1. Caching expensive operations
class CachedProcessingNode(BaseNode):
    """Node with built-in caching for expensive operations"""
    
    def __init__(self, config: NodeConfig):
        super().__init__(config)
        self.cache: Dict[str, Any] = {}
        self.cache_stats = {"hits": 0, "misses": 0}
    
    @lru_cache(maxsize=1000)
    def expensive_computation(self, data: str) -> Dict[str, Any]:
        """Example expensive computation with caching"""
        # Simulate expensive operation
        time.sleep(0.1)
        
        return {
            "processed": data.upper(),
            "word_count": len(data.split()),
            "char_count": len(data),
            "timestamp": time.time()
        }
    
    def execute(self, state: OptimizedState) -> OptimizedState:
        """Execute with caching"""
        processed_data = []
        
        for item in state["data"]:
            # Check cache first
            cache_key = f"processed_{hash(item)}"
            
            if cache_key in self.cache:
                result = self.cache[cache_key]
                self.cache_stats["hits"] += 1
            else:
                result = self.expensive_computation(item)
                self.cache[cache_key] = result
                self.cache_stats["misses"] += 1
            
            processed_data.append(result)
        
        return {
            **state,
            "processed_data": processed_data,
            "cache_hits": state.get("cache_hits", 0) + self.cache_stats["hits"],
            "cache_misses": state.get("cache_misses", 0) + self.cache_stats["misses"]
        }

# 2. Parallel processing within nodes
class ParallelProcessingNode(BaseNode):
    """Node that processes data in parallel"""
    
    def __init__(self, config: NodeConfig, max_workers: int = 4):
        super().__init__(config)
        self.max_workers = max_workers
    
    def process_single_item(self, item: str) -> Dict[str, Any]:
        """Process a single item"""
        # Simulate some processing
        time.sleep(0.05)
        return {
            "item": item,
            "length": len(item),
            "processed_at": time.time()
        }
    
    def execute(self, state: OptimizedState) -> OptimizedState:
        """Execute with parallel processing"""
        
        # Use ThreadPoolExecutor for I/O-bound tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self.process_single_item, item): item
                for item in state["data"]
            }
            
            # Collect results as they complete
            processed_data = []
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    result = future.result()
                    processed_data.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process item: {e}")
        
        return {
            **state,
            "processed_data": processed_data
        }

# 3. Async processing with batching
class AsyncBatchProcessingNode(BaseNode):
    """Node that processes data asynchronously in batches"""
    
    def __init__(self, config: NodeConfig, batch_size: int = 10):
        super().__init__(config)
        self.batch_size = batch_size
    
    async def process_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of items asynchronously"""
        tasks = [self.process_item_async(item) for item in batch]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_item_async(self, item: str) -> Dict[str, Any]:
        """Process a single item asynchronously"""
        # Simulate async I/O operation
        await asyncio.sleep(0.01)
        return {
            "item": item,
            "processed": item.upper(),
            "timestamp": time.time()
        }
    
    async def execute(self, state: OptimizedState) -> OptimizedState:
        """Execute with async batch processing"""
        data = state["data"]
        all_results = []
        
        # Process data in batches
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_results = await self.process_batch(batch)
            
            # Filter out exceptions and add successful results
            successful_results = [
                result for result in batch_results 
                if not isinstance(result, Exception)
            ]
            all_results.extend(successful_results)
        
        return {
            **state,
            "processed_data": all_results
        }

# 4. Memory-efficient streaming processing
class StreamingProcessingNode(BaseNode):
    """Node that processes data in a memory-efficient streaming manner"""
    
    def __init__(self, config: NodeConfig, chunk_size: int = 100):
        super().__init__(config)
        self.chunk_size = chunk_size
    
    def process_chunk(self, chunk: List[str]) -> List[Dict[str, Any]]:
        """Process a chunk of data"""
        return [
            {
                "item": item,
                "processed": item.strip().lower(),
                "chunk_id": id(chunk)
            }
            for item in chunk
        ]
    
    def execute(self, state: OptimizedState) -> OptimizedState:
        """Execute with streaming processing"""
        data = state["data"]
        
        # Generator for memory-efficient processing
        def process_stream():
            for i in range(0, len(data), self.chunk_size):
                chunk = data[i:i + self.chunk_size]
                yield from self.process_chunk(chunk)
        
        # Process stream and collect results
        processed_data = list(process_stream())
        
        return {
            **state,
            "processed_data": processed_data
        }
```

#### Graph-Level Optimizations

```python
from typing import TypedDict, Set
from langgraph.graph import StateGraph, START, END

class OptimizedGraphBuilder:
    """Builder for creating optimized graphs"""
    
    def __init__(self):
        self.graph = None
        self.node_dependencies: Dict[str, Set[str]] = {}
        self.node_priorities: Dict[str, int] = {}
    
    def analyze_dependencies(self, state_schema: TypedDict) -> Dict[str, Set[str]]:
        """Analyze node dependencies based on state keys"""
        # This is a simplified example - in practice, you'd analyze
        # which state keys each node reads vs writes
        return {
            "data_loader": set(),
            "preprocessor": {"data_loader"},
            "processor_a": {"preprocessor"},
            "processor_b": {"preprocessor"},
            "aggregator": {"processor_a", "processor_b"}
        }
    
    def identify_parallel_nodes(self) -> List[Set[str]]:
        """Identify nodes that can run in parallel"""
        # Find nodes with no dependencies between them
        parallel_groups = []
        
        # Simplified logic - group nodes by dependency level
        level_0 = {node for node, deps in self.node_dependencies.items() if not deps}
        level_1 = {node for node, deps in self.node_dependencies.items() 
                  if deps and deps.issubset(level_0)}
        
        if len(level_1) > 1:
            parallel_groups.append(level_1)
        
        return parallel_groups
    
    def optimize_graph_structure(self, original_graph: StateGraph) -> StateGraph:
        """Optimize graph structure for better performance"""
        
        # Analyze the graph
        parallel_groups = self.identify_parallel_nodes()
        
        # Create optimized graph
        optimized_graph = StateGraph(OptimizedState)
        
        # Add nodes (this would be more sophisticated in practice)
        optimized_graph.add_node("data_loader", CachedProcessingNode(NodeConfig("data_loader")))
        optimized_graph.add_node("parallel_processor", ParallelProcessingNode(NodeConfig("parallel_processor")))
        optimized_graph.add_node("stream_processor", StreamingProcessingNode(NodeConfig("stream_processor")))
        
        # Add optimized edges
        optimized_graph.add_edge(START, "data_loader")
        optimized_graph.add_edge("data_loader", "parallel_processor")
        optimized_graph.add_edge("parallel_processor", "stream_processor")
        optimized_graph.add_edge("stream_processor", END)
        
        return optimized_graph

# Performance testing utilities
class PerformanceBenchmark:
    """Utility for benchmarking graph performance"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    async def benchmark_graph(
        self, 
        graph_factory: callable, 
        test_data: List[Dict[str, Any]], 
        iterations: int = 5
    ) -> Dict[str, Any]:
        """Benchmark a graph with different configurations"""
        
        results = []
        
        for i in range(iterations):
            graph = graph_factory()
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Run graph
            for data in test_data:
                await graph.ainvoke(data)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            results.append({
                "iteration": i + 1,
                "execution_time": end_time - start_time,
                "memory_used": end_memory - start_memory,
                "data_points": len(test_data)
            })
        
        # Calculate statistics
        execution_times = [r["execution_time"] for r in results]
        memory_usage = [r["memory_used"] for r in results]
        
        return {
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "avg_memory_usage": sum(memory_usage) / len(memory_usage),
            "throughput": len(test_data) / (sum(execution_times) / len(execution_times)),
            "detailed_results": results
        }
    
    def compare_optimizations(self, configurations: Dict[str, callable]) -> None:
        """Compare different optimization configurations"""
        
        print("Performance Comparison Results")
        print("=" * 50)
        
        for name, factory in configurations.items():
            test_data = [{"data": [f"item_{i}" for i in range(100)]} for _ in range(10)]
            
            benchmark_result = asyncio.run(
                self.benchmark_graph(factory, test_data, iterations=3)
            )
            
            print(f"\n{name}:")
            print(f"  Avg Execution Time: {benchmark_result['avg_execution_time']:.3f}s")
            print(f"  Avg Memory Usage: {benchmark_result['avg_memory_usage']:.1f} MB")
            print(f"  Throughput: {benchmark_result['throughput']:.1f} items/sec")

# Example usage
def run_performance_benchmarks():
    """Example of running performance benchmarks"""
    
    benchmark = PerformanceBenchmark()
    
    # Define different configurations to test
    configurations = {
        "Basic Graph": lambda: create_basic_graph(),
        "Cached Graph": lambda: create_cached_graph(),
        "Parallel Graph": lambda: create_parallel_graph(),
        "Optimized Graph": lambda: create_optimized_graph()
    }
    
    benchmark.compare_optimizations(configurations)

def create_basic_graph():
    """Create basic unoptimized graph"""
    graph = StateGraph(OptimizedState)
    graph.add_node("processor", BaseNode(NodeConfig("processor")))
    graph.add_edge(START, "processor")
    graph.add_edge("processor", END)
    return graph.compile()

def create_cached_graph():
    """Create graph with caching"""
    graph = StateGraph(OptimizedState)
    graph.add_node("cached_processor", CachedProcessingNode(NodeConfig("cached_processor")))
    graph.add_edge(START, "cached_processor")
    graph.add_edge("cached_processor", END)
    return graph.compile()

def create_parallel_graph():
    """Create graph with parallel processing"""
    graph = StateGraph(OptimizedState)
    graph.add_node("parallel_processor", ParallelProcessingNode(NodeConfig("parallel_processor")))
    graph.add_edge(START, "parallel_processor")
    graph.add_edge("parallel_processor", END)
    return graph.compile()

def create_optimized_graph():
    """Create fully optimized graph"""
    optimizer = OptimizedGraphBuilder()
    return optimizer.optimize_graph_structure(StateGraph(OptimizedState))
```

### 14.3 Scaling Patterns

#### Horizontal Scaling with Multiple Workers

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
import queue
import threading
import time

class GraphWorkerPool:
    """Pool of workers for processing graph tasks"""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or mp.cpu_count()
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers: List[threading.Thread] = []
        self.shutdown_flag = threading.Event()
    
    def start_workers(self, graph_factory: callable):
        """Start worker threads"""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(graph_factory, i),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        print(f"Started {self.num_workers} workers")
    
    def _worker_loop(self, graph_factory: callable, worker_id: int):
        """Main loop for worker threads"""
        graph = graph_factory()
        
        while not self.shutdown_flag.is_set():
            try:
                # Get task with timeout
                task = self.task_queue.get(timeout=1)
                
                if task is None:  # Poison pill
                    break
                
                task_id, state = task
                
                try:
                    # Process task
                    start_time = time.time()
                    result = graph.invoke(state)
                    execution_time = time.time() - start_time
                    
                    # Put result
                    self.result_queue.put({
                        "task_id": task_id,
                        "result": result,
                        "worker_id": worker_id,
                        "execution_time": execution_time,
                        "status": "success"
                    })
                    
                except Exception as e:
                    # Put error result
                    self.result_queue.put({
                        "task_id": task_id,
                        "result": None,
                        "worker_id": worker_id,
                        "error": str(e),
                        "status": "error"
                    })
                
                finally:
                    self.task_queue.task_done()
                    
            except queue.Empty:
                continue
    
    def submit_task(self, task_id: str, state: Dict[str, Any]):
        """Submit a task for processing"""
        self.task_queue.put((task_id, state))
    
    def get_results(self, timeout: float = None) -> List[Dict[str, Any]]:
        """Get all available results"""
        results = []
        
        while True:
            try:
                result = self.result_queue.get(timeout=timeout or 0.1)
                results.append(result)
                self.result_queue.task_done()
            except queue.Empty:
                break
        
        return results
    
    def shutdown(self, timeout: float = 10):
        """Shutdown worker pool"""
        print("Shutting down worker pool...")
        
        # Signal shutdown
        self.shutdown_flag.set()
        
        # Add poison pills
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
        
        print("Worker pool shutdown complete")

# Distributed processing with process pools
class DistributedGraphProcessor:
    """Distributed graph processing using multiprocessing"""
    
    def __init__(self, num_processes: int = None):
        self.num_processes = num_processes or mp.cpu_count()
    
    def process_batch(self, graph_factory: callable, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of tasks across multiple processes"""
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Create a graph in each process
            futures = []
            
            for i, task in enumerate(tasks):
                future = executor.submit(self._process_single_task, graph_factory, task, i)
                futures.append(future)
            
            # Collect results as they complete
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "error": str(e),
                        "status": "failed"
                    })
            
            return results
    
    @staticmethod
    def _process_single_task(graph_factory: callable, task: Dict[str, Any], task_id: int) -> Dict[str, Any]:
        """Process a single task in a separate process"""
        try:
            graph = graph_factory()
            start_time = time.time()
            result = graph.invoke(task)
            execution_time = time.time() - start_time
            
            return {
                "task_id": task_id,
                "result": result,
                "execution_time": execution_time,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "task_id": task_id,
                "error": str(e),
                "status": "failed"
            }

# Example usage
def demonstrate_scaling():
    """Demonstrate different scaling approaches"""
    
    def create_test_graph():
        """Create a test graph for scaling demonstration"""
        graph = StateGraph(OptimizedState)
        graph.add_node("processor", ParallelProcessingNode(NodeConfig("processor")))
        graph.add_edge(START, "processor")
        graph.add_edge("processor", END)
        return graph.compile()
    
    # Test data
    test_tasks = [
        {"data": [f"task_{i}_item_{j}" for j in range(10)]}
        for i in range(20)
    ]
    
    print("Testing scaling approaches...")
    
    # 1. Single-threaded baseline
    start_time = time.time()
    graph = create_test_graph()
    single_results = []
    for task in test_tasks:
        result = graph.invoke(task)
        single_results.append(result)
    single_time = time.time() - start_time
    
    print(f"Single-threaded: {single_time:.2f}s for {len(test_tasks)} tasks")
    
    # 2. Thread pool scaling
    start_time = time.time()
    worker_pool = GraphWorkerPool(num_workers=4)
    worker_pool.start_workers(create_test_graph)
    
    # Submit tasks
    for i, task in enumerate(test_tasks):
        worker_pool.submit_task(f"task_{i}", task)
    
    # Wait for completion and collect results
    worker_pool.task_queue.join()
    thread_results = worker_pool.get_results(timeout=1)
    worker_pool.shutdown()
    thread_time = time.time() - start_time
    
    print(f"Thread pool (4 workers): {thread_time:.2f}s for {len(test_tasks)} tasks")
    
    # 3. Process pool scaling
    start_time = time.time()
    distributed_processor = DistributedGraphProcessor(num_processes=4)
    process_results = distributed_processor.process_batch(create_test_graph, test_tasks)
    process_time = time.time() - start_time
    
    print(f"Process pool (4 processes): {process_time:.2f}s for {len(test_tasks)} tasks")
    
    # Performance summary
    print("\nScaling Performance Summary:")
    print(f"Single-threaded baseline: {single_time:.2f}s (1.00x)")
    print(f"Thread pool speedup: {single_time/thread_time:.2f}x")
    print(f"Process pool speedup: {single_time/process_time:.2f}x")
```

### 14.4 Best Practices for Performance

#### Performance Guidelines

```python
"""
Performance Optimization Best Practices for LangGraph

1. **Node-Level Optimizations**:
   - Cache expensive computations using @lru_cache or custom caching
   - Use parallel processing for independent operations
   - Implement async processing for I/O-bound operations
   - Process data in batches to reduce overhead
   - Use generators for memory-efficient streaming

2. **Graph-Level Optimizations**:
   - Minimize state size by removing unnecessary data
   - Use conditional edges to skip unnecessary nodes
   - Parallelize independent node execution
   - Optimize the order of node execution
   - Use checkpoints strategically to enable recovery without performance penalty

3. **Memory Management**:
   - Monitor memory usage and identify leaks
   - Use weak references where appropriate
   - Clear large objects from state when no longer needed
   - Consider using memory-mapped files for large datasets
   - Profile memory allocation patterns

4. **Scaling Strategies**:
   - Use thread pools for I/O-bound parallel processing
   - Use process pools for CPU-bound parallel processing
   - Implement horizontal scaling with worker queues
   - Consider distributed processing for very large workloads
   - Monitor resource utilization and scale accordingly

5. **Profiling and Monitoring**:
   - Profile regularly during development
   - Set up continuous performance monitoring in production
   - Use flame graphs to identify CPU bottlenecks
   - Monitor memory usage patterns over time
   - Track key performance metrics (latency, throughput, error rates)

6. **Database and I/O Optimizations**:
   - Use connection pooling for database operations
   - Implement query optimization and indexing
   - Use async I/O operations where possible
   - Cache frequently accessed data
   - Batch database operations when possible

7. **LLM-Specific Optimizations**:
   - Reuse LLM clients across node executions
   - Implement response caching for repeated queries
   - Use streaming for long-form content generation
   - Optimize prompt templates for efficiency
   - Monitor token usage and costs
"""

# Performance testing framework
class PerformanceTestSuite:
    """Comprehensive performance testing framework"""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
    
    def run_all_tests(self, graph_factory: callable) -> Dict[str, Any]:
        """Run complete performance test suite"""
        
        print("Running Performance Test Suite...")
        print("=" * 50)
        
        # 1. Latency tests
        latency_results = self.test_latency(graph_factory)
        
        # 2. Throughput tests
        throughput_results = self.test_throughput(graph_factory)
        
        # 3. Memory usage tests
        memory_results = self.test_memory_usage(graph_factory)
        
        # 4. Concurrency tests
        concurrency_results = self.test_concurrency(graph_factory)
        
        # 5. Stress tests
        stress_results = self.test_stress_limits(graph_factory)
        
        # Compile results
        results = {
            "latency": latency_results,
            "throughput": throughput_results,
            "memory": memory_results,
            "concurrency": concurrency_results,
            "stress": stress_results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.print_performance_summary(results)
        return results
    
    def test_latency(self, graph_factory: callable, iterations: int = 100) -> Dict[str, float]:
        """Test execution latency"""
        graph = graph_factory()
        test_state = {"data": ["test_item"] * 10}
        
        latencies = []
        for _ in range(iterations):
            start_time = time.time()
            graph.invoke(test_state)
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
        
        return {
            "mean_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))],
            "p99_latency_ms": sorted(latencies)[int(0.99 * len(latencies))]
        }
    
    def test_throughput(self, graph_factory: callable, duration_seconds: int = 30) -> Dict[str, float]:
        """Test processing throughput"""
        graph = graph_factory()
        test_state = {"data": ["item"] * 5}
        
        start_time = time.time()
        completed_tasks = 0
        
        while (time.time() - start_time) < duration_seconds:
            graph.invoke(test_state)
            completed_tasks += 1
        
        actual_duration = time.time() - start_time
        throughput = completed_tasks / actual_duration
        
        return {
            "throughput_tasks_per_second": throughput,
            "total_tasks_completed": completed_tasks,
            "test_duration_seconds": actual_duration
        }
    
    def test_memory_usage(self, graph_factory: callable) -> Dict[str, float]:
        """Test memory usage patterns"""
        import psutil
        
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        graph = graph_factory()
        after_creation_memory = process.memory_info().rss / 1024 / 1024
        
        # Run multiple iterations
        test_state = {"data": ["item"] * 100}
        peak_memory = after_creation_memory
        
        for _ in range(10):
            graph.invoke(test_state)
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        return {
            "baseline_memory_mb": baseline_memory,
            "after_creation_memory_mb": after_creation_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": final_memory - baseline_memory
        }
    
    def test_concurrency(self, graph_factory: callable, num_threads: int = 10) -> Dict[str, Any]:
        """Test concurrent execution"""
        import threading
        
        results = []
        errors = []
        
        def worker():
            try:
                graph = graph_factory()
                test_state = {"data": ["concurrent_item"] * 5}
                
                start_time = time.time()
                result = graph.invoke(test_state)
                execution_time = time.time() - start_time
                
                results.append(execution_time)
            except Exception as e:
                errors.append(str(e))
        
        # Start threads
        threads = []
        start_time = time.time()
        
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        return {
            "num_threads": num_threads,
            "successful_executions": len(results),
            "failed_executions": len(errors),
            "total_time_seconds": total_time,
            "average_execution_time": sum(results) / len(results) if results else 0,
            "errors": errors[:5]  # First 5 errors only
        }
    
    def test_stress_limits(self, graph_factory: callable) -> Dict[str, Any]:
        """Test system limits under stress"""
        graph = graph_factory()
        
        # Test with increasing load
        stress_results = []
        
        for load_size in [10, 50, 100, 500, 1000]:
            test_state = {"data": ["stress_item"] * load_size}
            
            try:
                start_time = time.time()
                result = graph.invoke(test_state)
                execution_time = time.time() - start_time
                
                stress_results.append({
                    "load_size": load_size,
                    "execution_time": execution_time,
                    "success": True
                })
                
            except Exception as e:
                stress_results.append({
                    "load_size": load_size,
                    "error": str(e),
                    "success": False
                })
                break  # Stop at first failure
        
        return {
            "max_successful_load": max([r["load_size"] for r in stress_results if r["success"]], default=0),
            "stress_test_results": stress_results
        }
    
    def print_performance_summary(self, results: Dict[str, Any]):
        """Print formatted performance summary"""
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        
        # Latency results
        latency = results["latency"]
        print(f"\n📊 LATENCY METRICS:")
        print(f"   Mean: {latency['mean_latency_ms']:.2f} ms")
        print(f"   P95:  {latency['p95_latency_ms']:.2f} ms")
        print(f"   P99:  {latency['p99_latency_ms']:.2f} ms")
        
        # Throughput results
        throughput = results["throughput"]
        print(f"\n🚀 THROUGHPUT METRICS:")
        print(f"   Rate: {throughput['throughput_tasks_per_second']:.2f} tasks/sec")
        print(f"   Completed: {throughput['total_tasks_completed']} tasks")
        
        # Memory results
        memory = results["memory"]
        print(f"\n💾 MEMORY USAGE:")
        print(f"   Peak: {memory['peak_memory_mb']:.1f} MB")
        print(f"   Increase: {memory['memory_increase_mb']:.1f} MB")
        
        # Concurrency results
        concurrency = results["concurrency"]
        print(f"\n🔄 CONCURRENCY TEST:")
        print(f"   Success Rate: {concurrency['successful_executions']}/{concurrency['num_threads']}")
        print(f"   Avg Time: {concurrency['average_execution_time']:.3f} sec")
        
        # Stress results
        stress = results["stress"]
        print(f"\n⚡ STRESS TEST:")
        print(f"   Max Load: {stress['max_successful_load']} items")
        
        print("\n" + "=" * 60)

# Example usage
if __name__ == "__main__":
    def create_optimized_test_graph():
        graph = StateGraph(OptimizedState)
        graph.add_node("cached_node", CachedProcessingNode(NodeConfig("cached")))
        graph.add_node("parallel_node", ParallelProcessingNode(NodeConfig("parallel")))
        graph.add_edge(START, "cached_node")
        graph.add_edge("cached_node", "parallel_node")
        graph.add_edge("parallel_node", END)
        return graph.compile()
    
    # Run performance tests
    test_suite = PerformanceTestSuite()
    results = test_suite.run_all_tests(create_optimized_test_graph)
    
    # Save results to file
    import json
    with open(f"performance_results_{int(time.time())}.json", "w") as f:
        json.dump(results, f, indent=2)
```

---

## 15. Production Deployment Examples

### 13.1 Docker Containerization

#### Basic Dockerfile for LangGraph Application

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "src.api.main"]
```

#### Docker Compose for Complete Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  langgraph-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://user:password@postgres:5432/langgraph
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=langgraph
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
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
      - ./ssl:/etc/ssl/certs
    depends_on:
      - langgraph-app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 13.2 Kubernetes Deployment

#### Application Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-app
  labels:
    app: langgraph-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langgraph-app
  template:
    metadata:
      labels:
        app: langgraph-app
    spec:
      containers:
      - name: langgraph-app
        image: your-registry/langgraph-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: langgraph-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: langgraph-config
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: langgraph-config
      - name: logs-volume
        emptyDir: {}
```

#### Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: langgraph-service
spec:
  selector:
    app: langgraph-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: langgraph-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: langgraph-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: langgraph-service
            port:
              number: 80
```

#### ConfigMap and Secrets

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: langgraph-config
data:
  redis-url: "redis://redis-service:6379"
  log-level: "INFO"
  environment: "production"

---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: langgraph-secrets
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  api-key: <base64-encoded-api-key>
```

### 13.3 AWS Lambda Deployment

#### Lambda Function Handler

```python
# src/lambda/handler.py
import json
import logging
from typing import Dict, Any
from src.agents.customer_support import CustomerSupportAgent
from src.utils.config import get_config

logger = logging.getLogger(__name__)

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    AWS Lambda handler for LangGraph applications
    """
    try:
        # Initialize configuration
        config = get_config()
        
        # Extract request data
        body = json.loads(event.get('body', '{}'))
        
        # Initialize agent
        agent = CustomerSupportAgent(config=config)
        
        # Process request
        result = agent.process_request(
            query=body.get('query', ''),
            user_id=body.get('user_id', ''),
            session_id=body.get('session_id', '')
        )
        
        # Return response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({
                'success': True,
                'data': result,
                'request_id': context.aws_request_id
            })
        }
        
    except Exception as e:
        logger.error(f"Lambda execution error: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': str(e),
                'request_id': context.aws_request_id
            })
        }

# Cold start optimization
agent_instance = None

def optimized_lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Optimized Lambda handler with cold start mitigation
    """
    global agent_instance
    
    try:
        # Reuse agent instance across invocations
        if agent_instance is None:
            config = get_config()
            agent_instance = CustomerSupportAgent(config=config)
        
        body = json.loads(event.get('body', '{}'))
        
        result = agent_instance.process_request(
            query=body.get('query', ''),
            user_id=body.get('user_id', ''),
            session_id=body.get('session_id', '')
        )
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'data': result,
                'request_id': context.aws_request_id
            })
        }
        
    except Exception as e:
        logger.error(f"Optimized lambda error: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }
```

#### Serverless Framework Configuration

```yaml
# serverless.yml
service: langgraph-app

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  stage: ${opt:stage, 'dev'}
  environment:
    STAGE: ${self:provider.stage}
    LOG_LEVEL: INFO
  iam:
    role:
      statements:
        - Effect: Allow
          Action:
            - dynamodb:Query
            - dynamodb:Scan
            - dynamodb:GetItem
            - dynamodb:PutItem
            - dynamodb:UpdateItem
            - dynamodb:DeleteItem
          Resource: "arn:aws:dynamodb:${self:provider.region}:*:table/*"

functions:
  processQuery:
    handler: src.lambda.handler.optimized_lambda_handler
    timeout: 30
    memory: 1024
    events:
      - http:
          path: /process
          method: post
          cors: true
    environment:
      DATABASE_URL: ${env:DATABASE_URL}

  healthCheck:
    handler: src.lambda.health.health_check
    events:
      - http:
          path: /health
          method: get

plugins:
  - serverless-python-requirements
  - serverless-plugin-warmup

custom:
  pythonRequirements:
    dockerizePip: true
    layer: true
  warmup:
    enabled: true
    events:
      - schedule: rate(5 minutes)

resources:
  Resources:
    LangGraphTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: langgraph-${self:provider.stage}
        BillingMode: PAY_PER_REQUEST
        AttributeDefinitions:
          - AttributeName: pk
            AttributeType: S
          - AttributeName: sk
            AttributeType: S
        KeySchema:
          - AttributeName: pk
            KeyType: HASH
          - AttributeName: sk
            KeyType: RANGE
```

### 13.4 FastAPI Production Setup

#### Production API Server

```python
# src/api/main.py
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.agents.customer_support import CustomerSupportAgent
from src.utils.config import get_config
from src.utils.monitoring import setup_monitoring
from src.api.models import QueryRequest, QueryResponse
from src.api.auth import get_current_user
from src.api.rate_limiter import RateLimiter

# Global variables for shared resources
agent_pool = None
config = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global agent_pool, config
    
    # Startup
    logging.info("Starting LangGraph API server...")
    config = get_config()
    
    # Initialize agent pool for better concurrency
    from src.agents.pool import AgentPool
    agent_pool = AgentPool(
        agent_class=CustomerSupportAgent,
        pool_size=config.agent_pool_size,
        config=config
    )
    
    # Setup monitoring
    setup_monitoring()
    
    yield
    
    # Shutdown
    logging.info("Shutting down LangGraph API server...")
    if agent_pool:
        await agent_pool.close()

# Create FastAPI app
app = FastAPI(
    title="LangGraph API",
    description="Production LangGraph Application API",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Setup Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Rate limiting
rate_limiter = RateLimiter(
    calls=100,
    period=60,
    storage_backend="redis"
)

@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        # Check if agent pool is ready
        if agent_pool and agent_pool.is_healthy():
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {e}")

@app.post("/process", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user)
):
    """Process query using LangGraph agent"""
    try:
        # Apply rate limiting
        await rate_limiter.check_rate_limit(
            key=f"user:{current_user.id}",
            request=request
        )
        
        # Get agent from pool
        async with agent_pool.get_agent() as agent:
            # Process query
            result = await agent.aprocess_request(
                query=request.query,
                user_id=current_user.id,
                session_id=request.session_id,
                context=request.context
            )
            
            # Log metrics in background
            background_tasks.add_task(
                log_query_metrics,
                user_id=current_user.id,
                query_type=request.query_type,
                processing_time=time.time() - request.timestamp
            )
            
            return QueryResponse(
                success=True,
                data=result,
                processing_time=time.time() - request.timestamp
            )
            
    except Exception as e:
        logging.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def log_query_metrics(user_id: str, query_type: str, processing_time: float):
    """Log query metrics for monitoring"""
    # Implement your metrics logging here
    pass

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        access_log=True
    )
```

### 13.5 Monitoring and Observability

#### Prometheus Metrics

```python
# src/utils/monitoring.py
import time
import logging
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from functools import wraps

# Metrics registry
registry = CollectorRegistry()

# Define metrics
query_counter = Counter(
    'langgraph_queries_total',
    'Total number of queries processed',
    ['query_type', 'status'],
    registry=registry
)

query_duration = Histogram(
    'langgraph_query_duration_seconds',
    'Query processing duration',
    ['query_type'],
    registry=registry
)

active_connections = Gauge(
    'langgraph_active_connections',
    'Number of active connections',
    registry=registry
)

agent_pool_size = Gauge(
    'langgraph_agent_pool_size',
    'Current agent pool size',
    registry=registry
)

def monitor_query(query_type: str = "default"):
    """Decorator to monitor query execution"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                query_counter.labels(
                    query_type=query_type,
                    status='success'
                ).inc()
                return result
                
            except Exception as e:
                query_counter.labels(
                    query_type=query_type,
                    status='error'
                ).inc()
                raise
                
            finally:
                duration = time.time() - start_time
                query_duration.labels(query_type=query_type).observe(duration)
                
        return wrapper
    return decorator

def setup_monitoring():
    """Setup monitoring and logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Log startup metrics
    logging.info("Monitoring setup complete")
```

#### Application Performance Monitoring

```python
# src/utils/apm.py
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

class APMSetup:
    """Application Performance Monitoring setup"""
    
    def __init__(self, service_name: str = "langgraph-api"):
        self.service_name = service_name
        self.logger = structlog.get_logger()
        
    def setup_tracing(self, jaeger_endpoint: str = "http://localhost:14268/api/traces"):
        """Setup distributed tracing with Jaeger"""
        resource = Resource(attributes={
            SERVICE_NAME: self.service_name
        })
        
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(
            JaegerExporter(endpoint=jaeger_endpoint)
        )
        provider.add_span_processor(processor)
        
        trace.set_tracer_provider(provider)
        
        self.logger.info("Tracing setup complete", service=self.service_name)
        
    def get_tracer(self, name: str = None):
        """Get tracer instance"""
        return trace.get_tracer(name or self.service_name)

# Usage example
apm = APMSetup("langgraph-api")
apm.setup_tracing()
tracer = apm.get_tracer()

def trace_execution(operation_name: str):
    """Decorator for tracing function execution"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(operation_name) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    raise
        return wrapper
    return decorator
```

### 13.6 CI/CD Pipeline

#### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy LangGraph Application

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
          
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        
    - name: Run linting
      run: |
        flake8 src tests
        black --check src tests
        
    - name: Run type checking
      run: mypy src
      
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:test@localhost:5432/test
        REDIS_URL: redis://localhost:6379
      run: |
        pytest tests/ --cov=src --cov-report=xml
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        
  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    permissions:
      contents: read
      packages: write
      
    steps:
    - name: Checkout
      uses: actions/checkout@v4
      
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=sha
          
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        
  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to Kubernetes
      env:
        KUBE_CONFIG: ${{ secrets.KUBE_CONFIG }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        echo "$KUBE_CONFIG" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        
        # Update deployment with new image
        kubectl set image deployment/langgraph-app \
          langgraph-app=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:sha-${{ github.sha }}
          
        # Wait for rollout to complete
        kubectl rollout status deployment/langgraph-app --timeout=300s
        
        # Verify deployment
        kubectl get pods -l app=langgraph-app
```

---

## 16. Monitoring and Observability

Comprehensive monitoring and observability are essential for production LangGraph applications. This section covers metrics collection, logging strategies, distributed tracing, alerting, and performance monitoring.

### 16.1 Metrics Collection and Monitoring

#### Prometheus Integration

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
from typing import TypedDict, Dict, Any
import time
from datetime import datetime
from functools import wraps

# Define Prometheus metrics
REGISTRY = CollectorRegistry()

# Node execution metrics
NODE_EXECUTIONS = Counter(
    'langgraph_node_executions_total',
    'Total number of node executions',
    ['node_name', 'status'],
    registry=REGISTRY
)

NODE_DURATION = Histogram(
    'langgraph_node_duration_seconds',
    'Time spent executing nodes',
    ['node_name'],
    registry=REGISTRY
)

# Graph execution metrics
GRAPH_EXECUTIONS = Counter(
    'langgraph_graph_executions_total',
    'Total number of graph executions',
    ['graph_name', 'status'],
    registry=REGISTRY
)

GRAPH_DURATION = Histogram(
    'langgraph_graph_duration_seconds',
    'Time spent executing graphs',
    ['graph_name'],
    registry=REGISTRY
)

# System metrics
ACTIVE_GRAPHS = Gauge(
    'langgraph_active_graphs',
    'Number of currently executing graphs',
    registry=REGISTRY
)

MEMORY_USAGE = Gauge(
    'langgraph_memory_usage_bytes',
    'Memory usage of the application',
    registry=REGISTRY
)

class MetricsCollector:
    """Collects and exports metrics for LangGraph applications"""
    
    def __init__(self, metrics_port: int = 8000):
        self.metrics_port = metrics_port
        self._active_graphs = 0
    
    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        start_http_server(self.metrics_port, registry=REGISTRY)
        print(f"Metrics server started on port {self.metrics_port}")
    
    def record_node_execution(self, node_name: str, duration: float, status: str):
        """Record node execution metrics"""
        NODE_EXECUTIONS.labels(node_name=node_name, status=status).inc()
        NODE_DURATION.labels(node_name=node_name).observe(duration)
    
    def record_graph_execution(self, graph_name: str, duration: float, status: str):
        """Record graph execution metrics"""
        GRAPH_EXECUTIONS.labels(graph_name=graph_name, status=status).inc()
        GRAPH_DURATION.labels(graph_name=graph_name).observe(duration)
    
    def increment_active_graphs(self):
        """Increment active graph counter"""
        self._active_graphs += 1
        ACTIVE_GRAPHS.set(self._active_graphs)
    
    def decrement_active_graphs(self):
        """Decrement active graph counter"""
        self._active_graphs = max(0, self._active_graphs - 1)
        ACTIVE_GRAPHS.set(self._active_graphs)
    
    def update_memory_usage(self):
        """Update memory usage metric"""
        import psutil
        memory_bytes = psutil.Process().memory_info().rss
        MEMORY_USAGE.set(memory_bytes)

# Decorator for automatic metrics collection
def collect_metrics(metrics_collector: MetricsCollector):
    """Decorator for collecting metrics on graph/node execution"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            # Increment active counter for graphs
            if hasattr(func, '__name__') and 'graph' in func.__name__.lower():
                metrics_collector.increment_active_graphs()
            
            try:
                result = wrapper(*args, **kwargs)
                return result
                
            except Exception as e:
                status = "error"
                raise
                
            finally:
                duration = time.time() - start_time
                
                # Record metrics based on function type
                if hasattr(func, '__name__'):
                    if 'node' in func.__name__.lower():
                        metrics_collector.record_node_execution(
                            func.__name__, duration, status
                        )
                    elif 'graph' in func.__name__.lower():
                        metrics_collector.record_graph_execution(
                            func.__name__, duration, status
                        )
                        metrics_collector.decrement_active_graphs()
                
                # Update system metrics
                metrics_collector.update_memory_usage()
        
        return wrapper
    return decorator

# Monitored node base class
class MonitoredNode(BaseNode):
    """Base node class with built-in monitoring"""
    
    def __init__(self, config: NodeConfig, metrics_collector: MetricsCollector):
        super().__init__(config)
        self.metrics_collector = metrics_collector
    
    def __call__(self, state: TypedDict) -> TypedDict:
        """Execute node with metrics collection"""
        start_time = time.time()
        status = "success"
        
        try:
            result = super().__call__(state)
            return result
            
        except Exception as e:
            status = "error"
            raise
            
        finally:
            duration = time.time() - start_time
            self.metrics_collector.record_node_execution(
                self.config.name, duration, status
            )

# Example usage
def setup_monitoring():
    """Set up monitoring for LangGraph application"""
    metrics_collector = MetricsCollector(metrics_port=8000)
    metrics_collector.start_metrics_server()
    
    return metrics_collector
```

#### Custom Metrics Dashboard

```python
import json
import sqlite3
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

@dataclass
class MetricPoint:
    """A single metric data point"""
    metric_name: str
    value: float
    labels: Dict[str, str]
    timestamp: datetime

class MetricsStore:
    """Storage backend for custom metrics"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize metrics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                labels TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_name_time 
            ON metrics(metric_name, timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def store_metric(self, metric: MetricPoint):
        """Store a metric point"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO metrics (metric_name, value, labels, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            metric.metric_name,
            metric.value,
            json.dumps(metric.labels),
            metric.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_metrics(
        self, 
        metric_name: str, 
        start_time: datetime, 
        end_time: datetime,
        labels: Dict[str, str] = None
    ) -> List[MetricPoint]:
        """Retrieve metrics within time range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT metric_name, value, labels, timestamp
            FROM metrics
            WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
        """
        params = [metric_name, start_time.isoformat(), end_time.isoformat()]
        
        if labels:
            # Simple label filtering (in production, use proper JSON queries)
            query += " AND labels LIKE ?"
            params.append(f"%{list(labels.items())[0][0]}%")
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        metrics = []
        for row in rows:
            metrics.append(MetricPoint(
                metric_name=row[0],
                value=row[1],
                labels=json.loads(row[2]),
                timestamp=datetime.fromisoformat(row[3])
            ))
        
        return metrics
    
    def get_metric_summary(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = self.get_metrics(metric_name, start_time, end_time)
        
        if not metrics:
            return {"count": 0, "min": 0, "max": 0, "avg": 0}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }

class DashboardGenerator:
    """Generate monitoring dashboards"""
    
    def __init__(self, metrics_store: MetricsStore):
        self.metrics_store = metrics_store
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate complete dashboard data"""
        
        # Get summaries for key metrics
        node_executions = self.metrics_store.get_metric_summary("node_executions")
        graph_executions = self.metrics_store.get_metric_summary("graph_executions")
        response_times = self.metrics_store.get_metric_summary("response_time_ms")
        error_rates = self.metrics_store.get_metric_summary("error_rate")
        
        # Get recent performance data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        recent_response_times = self.metrics_store.get_metrics(
            "response_time_ms", start_time, end_time
        )
        
        return {
            "overview": {
                "node_executions": node_executions,
                "graph_executions": graph_executions,
                "response_times": response_times,
                "error_rates": error_rates
            },
            "recent_performance": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "value": m.value,
                    "labels": m.labels
                }
                for m in recent_response_times
            ],
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard"""
        dashboard_data = self.generate_dashboard_data()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LangGraph Monitoring Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric-card {{ 
                    border: 1px solid #ccc; 
                    padding: 15px; 
                    margin: 10px; 
                    display: inline-block; 
                    min-width: 200px; 
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ color: #666; }}
            </style>
        </head>
        <body>
            <h1>LangGraph Monitoring Dashboard</h1>
            <p>Generated at: {dashboard_data['generated_at']}</p>
            
            <h2>Overview (Last 24 Hours)</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Node Executions</div>
                    <div class="metric-value">{dashboard_data['overview']['node_executions']['count']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Graph Executions</div>
                    <div class="metric-value">{dashboard_data['overview']['graph_executions']['count']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Response Time</div>
                    <div class="metric-value">{dashboard_data['overview']['response_times']['avg']:.2f}ms</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Error Rate</div>
                    <div class="metric-value">{dashboard_data['overview']['error_rates']['avg']:.2f}%</div>
                </div>
            </div>
            
            <h2>Recent Performance (Last Hour)</h2>
            <div id="performance-chart">
                <!-- In a real implementation, you'd add a charting library here -->
                <p>Recent performance data points: {len(dashboard_data['recent_performance'])}</p>
            </div>
        </body>
        </html>
        """
        
        return html
```

### 16.2 Structured Logging

#### Comprehensive Logging Strategy

```python
import logging
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import traceback
from enum import Enum

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class StructuredLogger:
    """Structured logger for LangGraph applications"""
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add structured JSON handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
    
    def _log_structured(
        self, 
        level: LogLevel, 
        message: str, 
        **context: Any
    ):
        """Log structured message with context"""
        extra = {
            'structured_context': context,
            'logger_name': self.name,
            'timestamp': datetime.now().isoformat()
        }
        
        getattr(self.logger, level.value.lower())(message, extra=extra)
    
    def info(self, message: str, **context):
        """Log info message"""
        self._log_structured(LogLevel.INFO, message, **context)
    
    def debug(self, message: str, **context):
        """Log debug message"""
        self._log_structured(LogLevel.DEBUG, message, **context)
    
    def warning(self, message: str, **context):
        """Log warning message"""
        self._log_structured(LogLevel.WARNING, message, **context)
    
    def error(self, message: str, error: Optional[Exception] = None, **context):
        """Log error message with optional exception details"""
        if error:
            context.update({
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc()
            })
        self._log_structured(LogLevel.ERROR, message, **context)
    
    def critical(self, message: str, **context):
        """Log critical message"""
        self._log_structured(LogLevel.CRITICAL, message, **context)

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logs"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add structured context if available
        if hasattr(record, 'structured_context'):
            log_data['context'] = record.structured_context
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)

class GraphExecutionLogger:
    """Specialized logger for graph execution tracking"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    def log_graph_start(self, graph_name: str, initial_state: Dict[str, Any], config: Dict[str, Any] = None):
        """Log graph execution start"""
        self.logger.info(
            "Graph execution started",
            graph_name=graph_name,
            state_keys=list(initial_state.keys()),
            config=config or {},
            event_type="graph_start"
        )
    
    def log_node_execution(
        self, 
        graph_name: str, 
        node_name: str, 
        execution_time_ms: float,
        input_keys: List[str], 
        output_keys: List[str],
        status: str = "success",
        error: str = None
    ):
        """Log node execution details"""
        self.logger.info(
            f"Node '{node_name}' executed",
            graph_name=graph_name,
            node_name=node_name,
            execution_time_ms=execution_time_ms,
            input_keys=input_keys,
            output_keys=output_keys,
            status=status,
            error=error,
            event_type="node_execution"
        )
    
    def log_graph_completion(
        self, 
        graph_name: str, 
        total_execution_time_ms: float,
        nodes_executed: int,
        final_state_keys: List[str],
        status: str = "success"
    ):
        """Log graph execution completion"""
        self.logger.info(
            "Graph execution completed",
            graph_name=graph_name,
            total_execution_time_ms=total_execution_time_ms,
            nodes_executed=nodes_executed,
            final_state_keys=final_state_keys,
            status=status,
            event_type="graph_completion"
        )
    
    def log_graph_error(self, graph_name: str, error: Exception, context: Dict[str, Any] = None):
        """Log graph execution error"""
        self.logger.error(
            f"Graph execution failed: {graph_name}",
            error=error,
            graph_name=graph_name,
            context=context or {},
            event_type="graph_error"
        )

# Enhanced node with logging
class LoggingNode(BaseNode):
    """Node with comprehensive logging"""
    
    def __init__(self, config: NodeConfig, logger: StructuredLogger):
        super().__init__(config)
        self.logger = logger
        self.execution_logger = GraphExecutionLogger(logger)
    
    def __call__(self, state: TypedDict) -> TypedDict:
        """Execute node with comprehensive logging"""
        start_time = time.time()
        input_keys = list(state.keys())
        
        try:
            self.logger.debug(
                f"Starting node execution: {self.config.name}",
                node_name=self.config.name,
                input_keys=input_keys,
                state_size_bytes=sys.getsizeof(str(state))
            )
            
            result = self.execute(state)
            
            execution_time_ms = (time.time() - start_time) * 1000
            output_keys = list(result.keys())
            
            self.execution_logger.log_node_execution(
                graph_name="unknown",  # Would need to be passed in
                node_name=self.config.name,
                execution_time_ms=execution_time_ms,
                input_keys=input_keys,
                output_keys=output_keys,
                status="success"
            )
            
            return result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            self.execution_logger.log_node_execution(
                graph_name="unknown",
                node_name=self.config.name,
                execution_time_ms=execution_time_ms,
                input_keys=input_keys,
                output_keys=[],
                status="error",
                error=str(e)
            )
            
            raise
```

### 16.3 Distributed Tracing

#### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from typing import TypedDict, Dict, Any, Optional
import time
from functools import wraps

class TracingConfig:
    """Configuration for distributed tracing"""
    
    def __init__(
        self,
        service_name: str = "langgraph-app",
        jaeger_endpoint: str = "http://localhost:14268/api/traces",
        enable_tracing: bool = True
    ):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.enable_tracing = enable_tracing

class DistributedTracer:
    """Distributed tracing for LangGraph applications"""
    
    def __init__(self, config: TracingConfig):
        self.config = config
        self.tracer = None
        
        if config.enable_tracing:
            self._setup_tracing()
    
    def _setup_tracing(self):
        """Set up OpenTelemetry tracing"""
        # Create resource
        resource = Resource.create({
            "service.name": self.config.service_name,
            "service.version": "1.0.0"
        })
        
        # Set tracer provider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        
        # Create Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
    
    def trace_graph_execution(self, graph_name: str):
        """Decorator for tracing graph execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.config.enable_tracing:
                    return func(*args, **kwargs)
                
                with self.tracer.start_as_current_span(
                    f"graph.{graph_name}",
                    attributes={
                        "graph.name": graph_name,
                        "operation.type": "graph_execution"
                    }
                ) as span:
                    try:
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        
                        # Add success attributes
                        span.set_attribute("execution.duration_ms", (time.time() - start_time) * 1000)
                        span.set_attribute("execution.status", "success")
                        span.set_status(Status(StatusCode.OK))
                        
                        return result
                        
                    except Exception as e:
                        # Record error
                        span.record_exception(e)
                        span.set_attribute("execution.status", "error")
                        span.set_attribute("error.message", str(e))
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        
                        raise
            
            return wrapper
        return decorator
    
    def trace_node_execution(self, node_name: str):
        """Decorator for tracing node execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.config.enable_tracing:
                    return func(*args, **kwargs)
                
                with self.tracer.start_as_current_span(
                    f"node.{node_name}",
                    attributes={
                        "node.name": node_name,
                        "operation.type": "node_execution"
                    }
                ) as span:
                    try:
                        # Extract state information if available
                        if args and isinstance(args[0], dict):
                            state = args[0]
                            span.set_attribute("state.keys_count", len(state.keys()))
                            span.set_attribute("state.size_bytes", sys.getsizeof(str(state)))
                        
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        
                        # Add success attributes
                        execution_time = (time.time() - start_time) * 1000
                        span.set_attribute("execution.duration_ms", execution_time)
                        span.set_attribute("execution.status", "success")
                        
                        if isinstance(result, dict):
                            span.set_attribute("result.keys_count", len(result.keys()))
                        
                        span.set_status(Status(StatusCode.OK))
                        return result
                        
                    except Exception as e:
                        span.record_exception(e)
                        span.set_attribute("execution.status", "error")
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            
            return wrapper
        return decorator
    
    def create_custom_span(self, span_name: str, attributes: Dict[str, Any] = None):
        """Create a custom span for manual tracing"""
        if not self.config.enable_tracing:
            return trace.NoOpTracer().start_span("noop")
        
        return self.tracer.start_as_current_span(
            span_name,
            attributes=attributes or {}
        )

# Traced node implementation
class TracedNode(BaseNode):
    """Node with distributed tracing"""
    
    def __init__(self, config: NodeConfig, tracer: DistributedTracer):
        super().__init__(config)
        self.tracer = tracer
    
    def __call__(self, state: TypedDict) -> TypedDict:
        """Execute node with tracing"""
        with self.tracer.create_custom_span(
            f"node.{self.config.name}",
            attributes={
                "node.name": self.config.name,
                "node.enabled": self.config.enabled,
                "node.timeout_seconds": self.config.timeout_seconds
            }
        ) as span:
            try:
                result = self.execute(state)
                span.set_attribute("execution.success", True)
                return result
            except Exception as e:
                span.set_attribute("execution.success", False)
                span.record_exception(e)
                raise

# Example usage
def setup_tracing_example():
    """Example of setting up distributed tracing"""
    
    # Configure tracing
    tracing_config = TracingConfig(
        service_name="my-langgraph-app",
        jaeger_endpoint="http://localhost:14268/api/traces",
        enable_tracing=True
    )
    
    # Create tracer
    tracer = DistributedTracer(tracing_config)
    
    # Use tracing decorators
    @tracer.trace_graph_execution("processing_graph")
    def execute_graph(state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute graph with tracing"""
        
        # Create traced nodes
        node1 = TracedNode(NodeConfig("preprocessing"), tracer)
        node2 = TracedNode(NodeConfig("processing"), tracer)
        
        # Execute with tracing
        intermediate = node1(state)
        result = node2(intermediate)
        
        return result
    
    return execute_graph
```

### 16.4 Alerting and Notification

#### Alert Management System

```python
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import json
import time
import threading
from datetime import datetime, timedelta

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"

@dataclass
class Alert:
    """Alert definition"""
    name: str
    description: str
    severity: AlertSeverity
    metric_name: str
    threshold: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    duration_minutes: int  # How long condition must be true
    channels: List[AlertChannel]
    enabled: bool = True

@dataclass
class AlertNotification:
    """Alert notification instance"""
    alert: Alert
    current_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    message: str = ""

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, metrics_store: MetricsStore):
        self.metrics_store = metrics_store
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, AlertNotification] = {}
        self.notification_handlers: Dict[AlertChannel, Callable] = {}
        self.running = False
        self.check_thread = None
    
    def register_alert(self, alert: Alert):
        """Register a new alert"""
        self.alerts[alert.name] = alert
        self.logger.info(f"Registered alert: {alert.name}")
    
    def register_notification_handler(self, channel: AlertChannel, handler: Callable):
        """Register notification handler for a channel"""
        self.notification_handlers[channel] = handler
    
    def start_monitoring(self, check_interval_seconds: int = 60):
        """Start alert monitoring"""
        self.running = True
        self.check_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(check_interval_seconds,),
            daemon=True
        )
        self.check_thread.start()
        print(f"Alert monitoring started (checking every {check_interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop alert monitoring"""
        self.running = False
        if self.check_thread:
            self.check_thread.join()
        print("Alert monitoring stopped")
    
    def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop"""
        while self.running:
            try:
                self._check_alerts()
                time.sleep(check_interval)
            except Exception as e:
                print(f"Error in alert monitoring: {e}")
    
    def _check_alerts(self):
        """Check all registered alerts"""
        current_time = datetime.now()
        
        for alert_name, alert in self.alerts.items():
            if not alert.enabled:
                continue
            
            try:
                # Get recent metrics for this alert
                start_time = current_time - timedelta(minutes=alert.duration_minutes)
                metrics = self.metrics_store.get_metrics(
                    alert.metric_name, start_time, current_time
                )
                
                if not metrics:
                    continue
                
                # Check if alert condition is met
                current_value = metrics[-1].value  # Latest value
                condition_met = self._evaluate_condition(
                    current_value, alert.threshold, alert.comparison
                )
                
                # Handle alert state
                if condition_met and alert_name not in self.active_alerts:
                    # New alert
                    notification = AlertNotification(
                        alert=alert,
                        current_value=current_value,
                        triggered_at=current_time,
                        message=f"Alert '{alert.name}' triggered: {alert.metric_name} is {current_value} (threshold: {alert.threshold})"
                    )
                    
                    self.active_alerts[alert_name] = notification
                    self._send_notifications(notification)
                
                elif not condition_met and alert_name in self.active_alerts:
                    # Alert resolved
                    notification = self.active_alerts[alert_name]
                    notification.resolved_at = current_time
                    notification.message = f"Alert '{alert.name}' resolved: {alert.metric_name} is {current_value}"
                    
                    self._send_notifications(notification)
                    del self.active_alerts[alert_name]
                
            except Exception as e:
                print(f"Error checking alert {alert_name}: {e}")
    
    def _evaluate_condition(self, current_value: float, threshold: float, comparison: str) -> bool:
        """Evaluate alert condition"""
        if comparison == "gt":
            return current_value > threshold
        elif comparison == "lt":
            return current_value < threshold
        elif comparison == "eq":
            return current_value == threshold
        elif comparison == "gte":
            return current_value >= threshold
        elif comparison == "lte":
            return current_value <= threshold
        else:
            return False
    
    def _send_notifications(self, notification: AlertNotification):
        """Send notifications through configured channels"""
        for channel in notification.alert.channels:
            handler = self.notification_handlers.get(channel)
            if handler:
                try:
                    handler(notification)
                except Exception as e:
                    print(f"Failed to send notification via {channel.value}: {e}")

# Notification handlers
class EmailNotificationHandler:
    """Email notification handler"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    def send_alert(self, notification: AlertNotification, recipients: List[str]):
        """Send alert via email"""
        
        subject = f"LangGraph Alert: {notification.alert.name}"
        
        body = f"""
        Alert: {notification.alert.name}
        Severity: {notification.alert.severity.value.upper()}
        Description: {notification.alert.description}
        
        Metric: {notification.alert.metric_name}
        Current Value: {notification.current_value}
        Threshold: {notification.alert.threshold}
        
        Triggered At: {notification.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}
        
        Message: {notification.message}
        """
        
        if notification.resolved_at:
            body += f"\nResolved At: {notification.resolved_at.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Create email
        msg = MimeMultipart()
        msg['From'] = self.username
        msg['Subject'] = subject
        msg.attach(MimeText(body, 'plain'))
        
        # Send to all recipients
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            
            for recipient in recipients:
                msg['To'] = recipient
                server.sendmail(self.username, recipient, msg.as_string())

class SlackNotificationHandler:
    """Slack notification handler"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_alert(self, notification: AlertNotification):
        """Send alert to Slack"""
        import requests
        
        color = {
            AlertSeverity.LOW: "#36a64f",
            AlertSeverity.MEDIUM: "#ff9500", 
            AlertSeverity.HIGH: "#ff6b6b",
            AlertSeverity.CRITICAL: "#ff0000"
        }.get(notification.alert.severity, "#36a64f")
        
        status_emoji = "🚨" if not notification.resolved_at else "✅"
        status_text = "TRIGGERED" if not notification.resolved_at else "RESOLVED"
        
        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"{status_emoji} LangGraph Alert: {notification.alert.name}",
                    "text": f"*Status:* {status_text}\n*Severity:* {notification.alert.severity.value.upper()}",
                    "fields": [
                        {
                            "title": "Metric",
                            "value": notification.alert.metric_name,
                            "short": True
                        },
                        {
                            "title": "Current Value",
                            "value": str(notification.current_value),
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": str(notification.alert.threshold),
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": (notification.resolved_at or notification.triggered_at).strftime('%Y-%m-%d %H:%M:%S'),
                            "short": True
                        }
                    ],
                    "footer": "LangGraph Monitoring",
                    "ts": int(time.time())
                }
            ]
        }
        
        response = requests.post(self.webhook_url, json=payload)
        response.raise_for_status()

# Example alert setup
def setup_alerts_example():
    """Example alert configuration"""
    
    # Create metrics store and alert manager
    metrics_store = MetricsStore()
    alert_manager = AlertManager(metrics_store)
    
    # Configure notification handlers
    email_handler = EmailNotificationHandler(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username="alerts@mycompany.com",
        password="password"
    )
    
    slack_handler = SlackNotificationHandler(
        webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    )
    
    # Register handlers
    alert_manager.register_notification_handler(
        AlertChannel.EMAIL, 
        lambda notif: email_handler.send_alert(notif, ["team@mycompany.com"])
    )
    
    alert_manager.register_notification_handler(
        AlertChannel.SLACK,
        slack_handler.send_alert
    )
    
    # Define alerts
    alerts = [
        Alert(
            name="high_response_time",
            description="Response time is higher than normal",
            severity=AlertSeverity.HIGH,
            metric_name="response_time_ms",
            threshold=5000,  # 5 seconds
            comparison="gt",
            duration_minutes=5,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        ),
        Alert(
            name="high_error_rate",
            description="Error rate is above acceptable threshold",
            severity=AlertSeverity.CRITICAL,
            metric_name="error_rate",
            threshold=5.0,  # 5%
            comparison="gt", 
            duration_minutes=2,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        ),
        Alert(
            name="low_throughput",
            description="Graph execution throughput is below normal",
            severity=AlertSeverity.MEDIUM,
            metric_name="graphs_per_minute",
            threshold=10,
            comparison="lt",
            duration_minutes=10,
            channels=[AlertChannel.SLACK]
        )
    ]
    
    # Register alerts
    for alert in alerts:
        alert_manager.register_alert(alert)
    
    # Start monitoring
    alert_manager.start_monitoring(check_interval_seconds=30)
    
    return alert_manager
```

### 16.5 Best Practices for Monitoring

#### Monitoring Strategy Guidelines

```python
"""
Monitoring and Observability Best Practices for LangGraph

1. **Metrics Strategy**:
   - Track key business metrics (throughput, latency, error rates)
   - Monitor system metrics (CPU, memory, disk usage)
   - Implement custom metrics for domain-specific concerns
   - Use appropriate metric types (counters, gauges, histograms)
   - Set up dashboards for different audiences (ops, dev, business)

2. **Logging Strategy**:
   - Use structured logging (JSON format)
   - Include correlation IDs for distributed tracing
   - Log at appropriate levels (DEBUG for development, INFO+ for production)
   - Implement log aggregation and searchability
   - Include context information in all log messages

3. **Tracing Strategy**:
   - Implement distributed tracing for multi-service architectures
   - Trace critical execution paths
   - Include relevant metadata in spans
   - Use trace sampling for high-volume applications
   - Correlate traces with logs and metrics

4. **Alerting Strategy**:
   - Alert on symptoms, not causes
   - Use tiered alert severity levels
   - Implement alert fatigue prevention
   - Include runbook links in alert notifications
   - Test alert delivery mechanisms regularly

5. **Performance Monitoring**:
   - Monitor end-to-end request latency
   - Track resource utilization trends
   - Set up synthetic monitoring for critical paths
   - Implement SLA/SLO monitoring
   - Monitor third-party dependencies

6. **Security Monitoring**:
   - Monitor authentication failures
   - Track unusual access patterns
   - Monitor for data exfiltration attempts
   - Log security-relevant events
   - Implement anomaly detection

7. **Operational Guidelines**:
   - Implement health checks at multiple levels
   - Monitor deployment and rollback processes
   - Track configuration changes
   - Monitor batch job execution
   - Implement capacity planning metrics
"""

# Complete monitoring setup example
class ComprehensiveMonitoring:
    """Complete monitoring solution for LangGraph applications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = None
        self.logger = None
        self.tracer = None
        self.alert_manager = None
        self.dashboard_generator = None
    
    def initialize_monitoring(self):
        """Initialize all monitoring components"""
        
        # Set up metrics collection
        self.metrics_collector = MetricsCollector(
            metrics_port=self.config.get('metrics_port', 8000)
        )
        self.metrics_collector.start_metrics_server()
        
        # Set up structured logging
        self.logger = StructuredLogger(
            name=self.config.get('service_name', 'langgraph-app'),
            level=LogLevel[self.config.get('log_level', 'INFO')]
        )
        
        # Set up distributed tracing
        if self.config.get('enable_tracing', False):
            tracing_config = TracingConfig(
                service_name=self.config.get('service_name', 'langgraph-app'),
                jaeger_endpoint=self.config.get('jaeger_endpoint'),
                enable_tracing=True
            )
            self.tracer = DistributedTracer(tracing_config)
        
        # Set up metrics storage and alerting
        metrics_store = MetricsStore(self.config.get('metrics_db', 'metrics.db'))
        self.alert_manager = AlertManager(metrics_store)
        
        # Set up dashboard
        self.dashboard_generator = DashboardGenerator(metrics_store)
        
        print("Comprehensive monitoring initialized")
    
    def create_monitored_node(self, config: NodeConfig) -> BaseNode:
        """Create a fully monitored node"""
        
        # Start with base monitoring capabilities
        if self.tracer:
            node = TracedNode(config, self.tracer)
        else:
            node = LoggingNode(config, self.logger)
        
        # Add metrics collection
        if self.metrics_collector:
            original_call = node.__call__
            
            @collect_metrics(self.metrics_collector)
            def monitored_call(state):
                return original_call(state)
            
            node.__call__ = monitored_call
        
        return node
    
    def get_monitoring_middleware(self):
        """Get middleware for request monitoring"""
        def middleware(request, call_next):
            start_time = time.time()
            
            # Log request start
            self.logger.info(
                "Request started",
                method=request.method,
                url=str(request.url),
                client_ip=request.client.host
            )
            
            try:
                response = call_next(request)
                duration = time.time() - start_time
                
                # Log successful response
                self.logger.info(
                    "Request completed",
                    method=request.method,
                    url=str(request.url),
                    status_code=response.status_code,
                    duration_ms=duration * 1000
                )
                
                # Record metrics
                if self.metrics_collector:
                    self.metrics_collector.record_request(
                        method=request.method,
                        status_code=response.status_code,
                        duration=duration
                    )
                
                return response
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log error
                self.logger.error(
                    "Request failed",
                    method=request.method,
                    url=str(request.url),
                    error=e,
                    duration_ms=duration * 1000
                )
                
                raise
        
        return middleware
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "service": self.config.get('service_name', 'langgraph-app'),
            "version": self.config.get('version', '1.0.0'),
            "status": "healthy",
            "checks": {}
        }
        
        # Check metrics collection
        try:
            self.metrics_collector.update_memory_usage()
            report["checks"]["metrics"] = {"status": "healthy", "message": "Metrics collection active"}
        except Exception as e:
            report["checks"]["metrics"] = {"status": "unhealthy", "message": str(e)}
            report["status"] = "degraded"
        
        # Check logging
        try:
            self.logger.debug("Health check test log")
            report["checks"]["logging"] = {"status": "healthy", "message": "Logging system active"}
        except Exception as e:
            report["checks"]["logging"] = {"status": "unhealthy", "message": str(e)}
            report["status"] = "degraded"
        
        # Check alerting
        try:
            active_alerts = len(self.alert_manager.active_alerts) if self.alert_manager else 0
            report["checks"]["alerting"] = {
                "status": "healthy", 
                "message": f"{active_alerts} active alerts",
                "active_alerts": active_alerts
            }
        except Exception as e:
            report["checks"]["alerting"] = {"status": "unhealthy", "message": str(e)}
            report["status"] = "degraded"
        
        return report

# Usage example
def setup_comprehensive_monitoring():
    """Set up comprehensive monitoring for LangGraph application"""
    
    config = {
        'service_name': 'my-langgraph-app',
        'metrics_port': 8000,
        'log_level': 'INFO',
        'enable_tracing': True,
        'jaeger_endpoint': 'http://localhost:14268/api/traces',
        'metrics_db': 'metrics.db',
        'version': '1.0.0'
    }
    
    monitoring = ComprehensiveMonitoring(config)
    monitoring.initialize_monitoring()
    
    return monitoring
```

---

## 17. Troubleshooting

### 14.1 Common Issues and Solutions

#### Graph Construction Issues

```python
# src/debug/graph_validator.py
import logging
from typing import Dict, List, Any
from langgraph.graph import StateGraph

class GraphValidator:
    """Validator for LangGraph graph construction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_graph(self, graph: StateGraph) -> Dict[str, Any]:
        """
        Comprehensive graph validation
        """
        issues = []
        warnings = []
        
        # 1. Check for unreachable nodes
        unreachable = self._find_unreachable_nodes(graph)
        if unreachable:
            issues.append({
                "type": "unreachable_nodes",
                "nodes": unreachable,
                "message": f"Found unreachable nodes: {unreachable}"
            })
        
        # 2. Check for infinite loops
        loops = self._detect_potential_loops(graph)
        if loops:
            warnings.append({
                "type": "potential_loops",
                "loops": loops,
                "message": f"Potential infinite loops detected: {loops}"
            })
        
        # 3. Check for missing conditional logic
        missing_conditions = self._check_conditional_edges(graph)
        if missing_conditions:
            issues.append({
                "type": "missing_conditions",
                "edges": missing_conditions,
                "message": "Conditional edges without proper conditions"
            })
        
        # 4. Validate state schema consistency
        state_issues = self._validate_state_schema(graph)
        if state_issues:
            issues.extend(state_issues)
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "summary": self._generate_summary(issues, warnings)
        }
    
    def _find_unreachable_nodes(self, graph: StateGraph) -> List[str]:
        """Find nodes that cannot be reached from START"""
        # Implementation for finding unreachable nodes
        reachable = set()
        to_visit = ["__start__"]
        
        while to_visit:
            current = to_visit.pop()
            if current in reachable:
                continue
                
            reachable.add(current)
            
            # Get edges from current node
            edges = graph.edges.get(current, [])
            to_visit.extend(edges)
        
        # Find all nodes in graph
        all_nodes = set(graph.nodes.keys())
        all_nodes.add("__start__")
        all_nodes.add("__end__")
        
        return list(all_nodes - reachable)
    
    def _detect_potential_loops(self, graph: StateGraph) -> List[List[str]]:
        """Detect potential infinite loops using DFS"""
        loops = []
        visited = set()
        path = []
        
        def dfs(node: str):
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                loops.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            # Visit neighbors
            for neighbor in graph.edges.get(node, []):
                dfs(neighbor)
            
            path.pop()
        
        # Start DFS from each node
        for node in graph.nodes.keys():
            if node not in visited:
                dfs(node)
        
        return loops
    
    def _check_conditional_edges(self, graph: StateGraph) -> List[str]:
        """Check for conditional edges without proper conditions"""
        missing = []
        
        for node, edges in graph.edges.items():
            if len(edges) > 1:  # Multiple edges = conditional
                # Check if node has conditional logic
                if node not in graph.conditional_edge_conditions:
                    missing.append(node)
        
        return missing
    
    def _validate_state_schema(self, graph: StateGraph) -> List[Dict]:
        """Validate state schema consistency across nodes"""
        issues = []
        
        # Check if all nodes handle required state fields
        required_fields = getattr(graph.schema, '__annotations__', {})
        
        for node_name, node_func in graph.nodes.items():
            # Analyze node function signature and state usage
            node_issues = self._analyze_node_state_usage(
                node_name, node_func, required_fields
            )
            issues.extend(node_issues)
        
        return issues
    
    def _analyze_node_state_usage(self, node_name: str, node_func: callable, 
                                required_fields: Dict) -> List[Dict]:
        """Analyze how a node uses state"""
        import inspect
        
        issues = []
        
        try:
            # Get function source code
            source = inspect.getsource(node_func)
            
            # Check for common state access patterns
            for field in required_fields:
                if f"state['{field}']" not in source and f'state["{field}"]' not in source:
                    issues.append({
                        "type": "unused_state_field",
                        "node": node_name,
                        "field": field,
                        "message": f"Node {node_name} doesn't use required field {field}"
                    })
        
        except (OSError, TypeError):
            # Cannot inspect source (built-in function, etc.)
            pass
        
        return issues
    
    def _generate_summary(self, issues: List, warnings: List) -> str:
        """Generate validation summary"""
        if not issues and not warnings:
            return "✅ Graph validation passed with no issues"
        
        summary = []
        if issues:
            summary.append(f"❌ {len(issues)} issues found")
        if warnings:
            summary.append(f"⚠️ {len(warnings)} warnings")
            
        return " | ".join(summary)

# Usage example
def debug_graph_construction():
    """Debug graph construction issues"""
    from src.agents.customer_support import CustomerSupportAgent
    
    # Create agent and get graph
    agent = CustomerSupportAgent()
    graph = agent.create_graph()
    
    # Validate graph
    validator = GraphValidator()
    validation_result = validator.validate_graph(graph)
    
    print("Graph Validation Results:")
    print("=" * 50)
    print(f"Valid: {validation_result['valid']}")
    print(f"Summary: {validation_result['summary']}")
    
    if validation_result['issues']:
        print("\n🔴 Issues:")
        for issue in validation_result['issues']:
            print(f"  - {issue['type']}: {issue['message']}")
    
    if validation_result['warnings']:
        print("\n🟡 Warnings:")
        for warning in validation_result['warnings']:
            print(f"  - {warning['type']}: {warning['message']}")

if __name__ == "__main__":
    debug_graph_construction()
```

#### State Management Debugging

```python
# src/debug/state_debugger.py
import json
import logging
from typing import Dict, Any, List
from copy import deepcopy

class StateDebugger:
    """Debugger for LangGraph state management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.state_history: List[Dict[str, Any]] = []
        self.checkpoint_history: List[Dict[str, Any]] = []
        
    def capture_state_transition(self, node_name: str, before_state: Dict, 
                               after_state: Dict, metadata: Dict = None):
        """Capture state transitions for debugging"""
        transition = {
            "timestamp": self._get_timestamp(),
            "node": node_name,
            "before": deepcopy(before_state),
            "after": deepcopy(after_state),
            "changes": self._compute_state_diff(before_state, after_state),
            "metadata": metadata or {}
        }
        
        self.state_history.append(transition)
        
        self.logger.debug(f"State transition in {node_name}: {transition['changes']}")
    
    def _compute_state_diff(self, before: Dict, after: Dict) -> Dict:
        """Compute differences between states"""
        changes = {
            "added": {},
            "modified": {},
            "deleted": []
        }
        
        # Find added and modified keys
        for key, value in after.items():
            if key not in before:
                changes["added"][key] = value
            elif before[key] != value:
                changes["modified"][key] = {
                    "old": before[key],
                    "new": value
                }
        
        # Find deleted keys
        for key in before:
            if key not in after:
                changes["deleted"].append(key)
        
        return changes
    
    def print_state_history(self):
        """Print formatted state history"""
        print("\n" + "=" * 60)
        print("STATE TRANSITION HISTORY")
        print("=" * 60)
        
        for i, transition in enumerate(self.state_history):
            print(f"\n[{i+1}] Node: {transition['node']} | {transition['timestamp']}")
            
            if transition['changes']['added']:
                print("  ➕ Added:")
                for key, value in transition['changes']['added'].items():
                    print(f"    {key}: {self._format_value(value)}")
            
            if transition['changes']['modified']:
                print("  🔄 Modified:")
                for key, change in transition['changes']['modified'].items():
                    print(f"    {key}: {self._format_value(change['old'])} → {self._format_value(change['new'])}")
            
            if transition['changes']['deleted']:
                print("  ❌ Deleted:")
                for key in transition['changes']['deleted']:
                    print(f"    {key}")
    
    def _format_value(self, value: Any) -> str:
        """Format value for display"""
        if isinstance(value, str) and len(value) > 50:
            return f'"{value[:47]}..."'
        elif isinstance(value, (list, dict)):
            return f"{type(value).__name__}(len={len(value)})"
        else:
            return str(value)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    def validate_state_schema(self, state: Dict, expected_schema: Dict) -> List[str]:
        """Validate state against expected schema"""
        errors = []
        
        # Check required fields
        for field, field_type in expected_schema.items():
            if field not in state:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(state[field], field_type):
                errors.append(f"Field {field} should be {field_type.__name__}, got {type(state[field]).__name__}")
        
        return errors

# State debugging decorator
def debug_state_transitions(debugger: StateDebugger):
    """Decorator to automatically capture state transitions"""
    def decorator(node_func):
        def wrapper(state: Dict, **kwargs) -> Dict:
            before_state = deepcopy(state)
            
            try:
                result = node_func(state, **kwargs)
                after_state = result if result is not None else state
                
                debugger.capture_state_transition(
                    node_name=node_func.__name__,
                    before_state=before_state,
                    after_state=after_state,
                    metadata={"success": True}
                )
                
                return after_state
                
            except Exception as e:
                debugger.capture_state_transition(
                    node_name=node_func.__name__,
                    before_state=before_state,
                    after_state=before_state,  # State unchanged on error
                    metadata={"success": False, "error": str(e)}
                )
                raise
                
        return wrapper
    return decorator
```

#### Performance Debugging

```python
# src/debug/performance_profiler.py
import time
import cProfile
import pstats
from functools import wraps
from typing import Dict, List, Any
from io import StringIO

class PerformanceProfiler:
    """Performance profiler for LangGraph applications"""
    
    def __init__(self):
        self.execution_times: Dict[str, List[float]] = {}
        self.memory_usage: Dict[str, List[float]] = {}
        self.call_counts: Dict[str, int] = {}
        
    def profile_execution_time(self, func_name: str = None):
        """Decorator to profile execution time"""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    if name not in self.execution_times:
                        self.execution_times[name] = []
                    
                    self.execution_times[name].append(execution_time)
                    self.call_counts[name] = self.call_counts.get(name, 0) + 1
                    
            return wrapper
        return decorator
    
    def profile_memory_usage(self, func_name: str = None):
        """Decorator to profile memory usage"""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    import psutil
                    import os
                    
                    process = psutil.Process(os.getpid())
                    start_memory = process.memory_info().rss / 1024 / 1024  # MB
                    
                    result = func(*args, **kwargs)
                    
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = end_memory - start_memory
                    
                    if name not in self.memory_usage:
                        self.memory_usage[name] = []
                    
                    self.memory_usage[name].append(memory_delta)
                    
                    return result
                    
                except ImportError:
                    # psutil not available, skip memory profiling
                    return func(*args, **kwargs)
                    
            return wrapper
        return decorator
    
    def detailed_profile(self, func, *args, **kwargs):
        """Run detailed cProfile analysis"""
        profiler = cProfile.Profile()
        
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Capture stats
        stats_buffer = StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('cumulative').print_stats(20)
        
        profile_output = stats_buffer.getvalue()
        
        return result, profile_output
    
    def print_performance_report(self):
        """Print comprehensive performance report"""
        print("\n" + "=" * 70)
        print("PERFORMANCE REPORT")
        print("=" * 70)
        
        # Execution Time Report
        if self.execution_times:
            print("\n📊 EXECUTION TIME ANALYSIS")
            print("-" * 40)
            
            for func_name, times in self.execution_times.items():
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                total_time = sum(times)
                call_count = self.call_counts.get(func_name, len(times))
                
                print(f"\n{func_name}:")
                print(f"  Calls: {call_count}")
                print(f"  Total: {total_time:.4f}s")
                print(f"  Average: {avg_time:.4f}s")
                print(f"  Min: {min_time:.4f}s")
                print(f"  Max: {max_time:.4f}s")
        
        # Memory Usage Report
        if self.memory_usage:
            print("\n🧠 MEMORY USAGE ANALYSIS")
            print("-" * 40)
            
            for func_name, usage in self.memory_usage.items():
                avg_usage = sum(usage) / len(usage)
                max_usage = max(usage) if usage else 0
                
                print(f"\n{func_name}:")
                print(f"  Average Memory Delta: {avg_usage:.2f} MB")
                print(f"  Max Memory Delta: {max_usage:.2f} MB")
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Analyze execution times
        for func_name, times in self.execution_times.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            call_count = self.call_counts.get(func_name, len(times))
            
            # Flag functions that take longer than 1 second on average
            # or consume more than 10% of total execution time
            if avg_time > 1.0 or total_time > sum(sum(t) for t in self.execution_times.values()) * 0.1:
                bottlenecks.append({
                    "function": func_name,
                    "issue": "slow_execution",
                    "avg_time": avg_time,
                    "total_time": total_time,
                    "call_count": call_count
                })
        
        return bottlenecks

# Usage example for debugging
def debug_agent_performance():
    """Debug agent performance issues"""
    profiler = PerformanceProfiler()
    debugger = StateDebugger()
    
    # Apply decorators to agent methods
    from src.agents.customer_support import CustomerSupportAgent
    
    agent = CustomerSupportAgent()
    
    # Profile key methods
    agent.process_request = profiler.profile_execution_time("process_request")(agent.process_request)
    agent.analyze_query = profiler.profile_execution_time("analyze_query")(agent.analyze_query)
    
    # Test with sample data
    sample_requests = [
        "How can I reset my password?",
        "What's your return policy?",
        "I need help with my order",
    ]
    
    print("Running performance tests...")
    
    for i, request in enumerate(sample_requests):
        print(f"Processing request {i+1}/{len(sample_requests)}")
        
        try:
            result = agent.process_request(request)
            print(f"✅ Request processed successfully")
        except Exception as e:
            print(f"❌ Error processing request: {e}")
    
    # Print reports
    profiler.print_performance_report()
    debugger.print_state_history()
    
    # Identify bottlenecks
    bottlenecks = profiler.identify_bottlenecks()
    if bottlenecks:
        print("\n🚨 PERFORMANCE BOTTLENECKS DETECTED")
        print("-" * 40)
        for bottleneck in bottlenecks:
            print(f"Function: {bottleneck['function']}")
            print(f"Issue: {bottleneck['issue']}")
            print(f"Average time: {bottleneck['avg_time']:.4f}s")
            print(f"Total time: {bottleneck['total_time']:.4f}s")
            print(f"Call count: {bottleneck['call_count']}")
            print()
```

### 14.2 Debugging Tools and Techniques

#### Interactive Debug Console

```python
# src/debug/debug_console.py
import cmd
import json
import pprint
from typing import Dict, Any

class LangGraphDebugConsole(cmd.Cmd):
    """Interactive debug console for LangGraph applications"""
    
    intro = """
    🐛 LangGraph Debug Console
    ========================
    
    Available commands:
    - inspect <node_name>     : Inspect a node
    - trace <execution_id>    : Show execution trace
    - state                   : Show current state
    - graph                   : Show graph structure
    - run <node_name>         : Run a specific node
    - reset                   : Reset debug session
    - help                    : Show this help
    - quit                    : Exit debug console
    """
    
    prompt = "(langgraph-debug) "
    
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.current_state = {}
        self.execution_traces = []
        
    def do_inspect(self, node_name: str):
        """Inspect a node: inspect <node_name>"""
        if not node_name:
            print("Usage: inspect <node_name>")
            return
            
        try:
            node_func = self.agent.graph.nodes.get(node_name)
            if not node_func:
                print(f"❌ Node '{node_name}' not found")
                return
                
            print(f"\n📋 Node: {node_name}")
            print("-" * 40)
            
            # Show function signature
            import inspect
            sig = inspect.signature(node_func)
            print(f"Signature: {node_name}{sig}")
            
            # Show docstring
            if node_func.__doc__:
                print(f"Description: {node_func.__doc__.strip()}")
            
            # Show source code
            try:
                source = inspect.getsource(node_func)
                print(f"\nSource code:")
                print(source)
            except (OSError, TypeError):
                print("Source code not available")
                
        except Exception as e:
            print(f"❌ Error inspecting node: {e}")
    
    def do_state(self, line):
        """Show current state"""
        if not self.current_state:
            print("No current state available")
        else:
            print("\n📊 Current State:")
            print("-" * 40)
            pprint.pprint(self.current_state, width=100)
    
    def do_graph(self, line):
        """Show graph structure"""
        print("\n🕸️ Graph Structure:")
        print("-" * 40)
        
        print("Nodes:")
        for node_name in self.agent.graph.nodes.keys():
            print(f"  - {node_name}")
        
        print("\nEdges:")
        for source, targets in self.agent.graph.edges.items():
            if isinstance(targets, list):
                for target in targets:
                    print(f"  {source} → {target}")
            else:
                print(f"  {source} → {targets}")
    
    def do_run(self, node_name: str):
        """Run a specific node: run <node_name>"""
        if not node_name:
            print("Usage: run <node_name>")
            return
            
        try:
            node_func = self.agent.graph.nodes.get(node_name)
            if not node_func:
                print(f"❌ Node '{node_name}' not found")
                return
            
            print(f"\n🏃 Running node: {node_name}")
            
            if not self.current_state:
                # Initialize with default state
                self.current_state = self.agent.get_initial_state()
                print("Initialized with default state")
            
            # Run the node
            result = node_func(self.current_state)
            
            if result:
                self.current_state.update(result)
                print("✅ Node executed successfully")
                print(f"State updated with: {list(result.keys())}")
            else:
                print("✅ Node executed (no state changes)")
                
        except Exception as e:
            print(f"❌ Error running node: {e}")
            import traceback
            traceback.print_exc()
    
    def do_trace(self, execution_id: str):
        """Show execution trace: trace <execution_id>"""
        print(f"\n🔍 Execution Trace: {execution_id or 'latest'}")
        print("-" * 40)
        
        # Show trace information
        if self.execution_traces:
            latest_trace = self.execution_traces[-1]
            for step in latest_trace:
                print(f"Step {step['step']}: {step['node']} ({step['duration']:.3f}s)")
                if step.get('error'):
                    print(f"  ❌ Error: {step['error']}")
        else:
            print("No execution traces available")
    
    def do_reset(self, line):
        """Reset debug session"""
        self.current_state = {}
        self.execution_traces = []
        print("🔄 Debug session reset")
    
    def do_quit(self, line):
        """Exit debug console"""
        print("👋 Goodbye!")
        return True
    
    def do_EOF(self, line):
        """Handle Ctrl+D"""
        return self.do_quit(line)

# Debugging utility functions
def debug_node_execution(agent, node_name: str, state: Dict[str, Any]):
    """Debug execution of a specific node"""
    print(f"\n🔍 Debugging node: {node_name}")
    print("=" * 50)
    
    try:
        # Get node function
        node_func = agent.graph.nodes.get(node_name)
        if not node_func:
            print(f"❌ Node '{node_name}' not found")
            return
        
        print("📋 Input State:")
        pprint.pprint(state, width=100)
        
        print(f"\n🏃 Executing {node_name}...")
        start_time = time.time()
        
        result = node_func(state.copy())
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"⏱️ Execution time: {execution_time:.4f}s")
        
        if result:
            print("\n📋 Output:")
            pprint.pprint(result, width=100)
        else:
            print("\n📋 No output returned")
            
    except Exception as e:
        print(f"❌ Error executing node: {e}")
        import traceback
        traceback.print_exc()

def start_debug_session(agent):
    """Start interactive debug session"""
    console = LangGraphDebugConsole(agent)
    console.cmdloop()

if __name__ == "__main__":
    # Example usage
    from src.agents.customer_support import CustomerSupportAgent
    
    agent = CustomerSupportAgent()
    start_debug_session(agent)
```

### 14.3 Troubleshooting Common Issues

#### Issue Resolution Guide

```python
# src/debug/troubleshoot.py
from typing import Dict, List, Any, Callable
import logging

class TroubleshootingGuide:
    """Common issues and solutions for LangGraph applications"""
    
    def __init__(self):
        self.solutions = {
            "graph_construction": self._troubleshoot_graph_construction,
            "state_management": self._troubleshoot_state_management,
            "performance": self._troubleshoot_performance,
            "memory_issues": self._troubleshoot_memory_issues,
            "llm_integration": self._troubleshoot_llm_integration,
        }
    
    def diagnose_issue(self, symptoms: List[str]) -> Dict[str, Any]:
        """Diagnose issues based on symptoms"""
        recommendations = []
        
        # Graph construction issues
        if any(symptom in ["unreachable_nodes", "infinite_loop", "missing_edges"] 
               for symptom in symptoms):
            recommendations.extend(self.solutions["graph_construction"]())
        
        # State management issues
        if any(symptom in ["state_corruption", "missing_state_fields", "type_errors"] 
               for symptom in symptoms):
            recommendations.extend(self.solutions["state_management"]())
        
        # Performance issues
        if any(symptom in ["slow_execution", "high_memory", "timeout"] 
               for symptom in symptoms):
            recommendations.extend(self.solutions["performance"]())
        
        return {
            "symptoms": symptoms,
            "recommendations": recommendations,
            "diagnostic_steps": self._get_diagnostic_steps(symptoms)
        }
    
    def _troubleshoot_graph_construction(self) -> List[Dict[str, Any]]:
        """Solutions for graph construction issues"""
        return [
            {
                "issue": "Unreachable Nodes",
                "solution": "Ensure all nodes are connected to the graph flow",
                "code_example": """
# ✅ Correct: All nodes connected
workflow.add_node("start", start_node)
workflow.add_node("process", process_node)
workflow.add_node("end", end_node)

workflow.add_edge("start", "process")
workflow.add_edge("process", "end")

# ❌ Incorrect: 'orphan' node not connected
workflow.add_node("orphan", orphan_node)  # This node is unreachable
                """,
                "prevention": "Use the GraphValidator to check for unreachable nodes"
            },
            {
                "issue": "Infinite Loops",
                "solution": "Add proper exit conditions in conditional edges",
                "code_example": """
def should_continue(state: State) -> str:
    # ✅ Correct: Clear exit condition
    if state["attempts"] > 3:
        return "end"
    elif state["success"]:
        return "end"
    else:
        return "retry"

# ❌ Incorrect: No exit condition
def should_continue(state: State) -> str:
    return "retry"  # Always loops back!
                """,
                "prevention": "Always include termination conditions in loops"
            }
        ]
    
    def _troubleshoot_state_management(self) -> List[Dict[str, Any]]:
        """Solutions for state management issues"""
        return [
            {
                "issue": "State Field Missing",
                "solution": "Initialize all required fields in the state schema",
                "code_example": """
# ✅ Correct: Complete state initialization
class State(TypedDict):
    input: str
    output: str
    steps: List[str]
    metadata: Dict[str, Any]

def initialize_state(input_text: str) -> State:
    return {
        "input": input_text,
        "output": "",
        "steps": [],
        "metadata": {}
    }
                """,
                "prevention": "Use type hints and validate state schema"
            },
            {
                "issue": "State Type Errors",
                "solution": "Ensure consistent data types across nodes",
                "code_example": """
# ✅ Correct: Consistent types
def process_node(state: State) -> State:
    state["steps"].append("processed")  # List operation
    return state

# ❌ Incorrect: Type mismatch
def bad_node(state: State) -> State:
    state["steps"] = "processed"  # Changes list to string!
    return state
                """,
                "prevention": "Use mypy or similar type checking tools"
            }
        ]
    
    def _troubleshoot_performance(self) -> List[Dict[str, Any]]:
        """Solutions for performance issues"""
        return [
            {
                "issue": "Slow LLM Calls",
                "solution": "Implement caching and async processing",
                "code_example": """
import asyncio
from functools import lru_cache

# ✅ Use caching for repeated queries
@lru_cache(maxsize=100)
def cached_llm_call(prompt: str) -> str:
    return llm.invoke(prompt)

# ✅ Use async for concurrent calls
async def async_process_nodes(state: State) -> State:
    tasks = [
        process_node_1(state),
        process_node_2(state),
        process_node_3(state)
    ]
    results = await asyncio.gather(*tasks)
    return combine_results(results)
                """,
                "prevention": "Profile code regularly and optimize bottlenecks"
            }
        ]
    
    def _troubleshoot_memory_issues(self) -> List[Dict[str, Any]]:
        """Solutions for memory issues"""
        return [
            {
                "issue": "Memory Leaks",
                "solution": "Clear large objects from state when no longer needed",
                "code_example": """
def cleanup_node(state: State) -> State:
    # ✅ Clear large data structures
    if "large_data" in state:
        del state["large_data"]
    
    # ✅ Keep only essential information
    state["summary"] = summarize_large_data(state.get("raw_data", ""))
    if "raw_data" in state:
        del state["raw_data"]
    
    return state
                """,
                "prevention": "Monitor memory usage and clean up regularly"
            }
        ]
    
    def _troubleshoot_llm_integration(self) -> List[Dict[str, Any]]:
        """Solutions for LLM integration issues"""
        return [
            {
                "issue": "API Rate Limits",
                "solution": "Implement exponential backoff and request queuing",
                "code_example": """
import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def safe_llm_call(prompt: str) -> str:
    return llm.invoke(prompt)
                """,
                "prevention": "Monitor API usage and implement proper error handling"
            }
        ]
    
    def _get_diagnostic_steps(self, symptoms: List[str]) -> List[str]:
        """Get diagnostic steps for symptoms"""
        steps = []
        
        if "slow_execution" in symptoms:
            steps.extend([
                "1. Profile code execution times",
                "2. Check for blocking I/O operations",
                "3. Analyze LLM call frequency",
                "4. Review state management efficiency"
            ])
        
        if "state_corruption" in symptoms:
            steps.extend([
                "1. Enable state debugging",
                "2. Check for concurrent state modifications",
                "3. Validate state schema consistency",
                "4. Review node return values"
            ])
        
        return steps

# Command-line troubleshooting tool
def run_troubleshooter():
    """Interactive troubleshooting session"""
    guide = TroubleshootingGuide()
    
    print("🔧 LangGraph Troubleshooting Assistant")
    print("=" * 40)
    
    print("\nWhat symptoms are you experiencing? (comma-separated)")
    print("Options: unreachable_nodes, infinite_loop, state_corruption, slow_execution, high_memory, timeout, missing_state_fields, type_errors")
    
    symptoms_input = input("Symptoms: ").strip()
    symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
    
    if not symptoms:
        print("No symptoms provided. Exiting.")
        return
    
    diagnosis = guide.diagnose_issue(symptoms)
    
    print(f"\n🔍 DIAGNOSIS")
    print("-" * 40)
    print(f"Symptoms: {', '.join(diagnosis['symptoms'])}")
    
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 40)
    for i, rec in enumerate(diagnosis['recommendations'], 1):
        print(f"{i}. Issue: {rec['issue']}")
        print(f"   Solution: {rec['solution']}")
        if 'code_example' in rec:
            print(f"   Example:")
            print(rec['code_example'])
        print()
    
    if diagnosis['diagnostic_steps']:
        print(f"\n🔍 DIAGNOSTIC STEPS")
        print("-" * 40)
        for step in diagnosis['diagnostic_steps']:
            print(step)

if __name__ == "__main__":
    run_troubleshooter()
```

### 14.4 Production Monitoring and Alerting

```python
# src/monitoring/alerts.py
import logging
import smtplib
from email.mime.text import MIMEText
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    level: AlertLevel
    title: str
    message: str
    component: str
    metadata: Dict[str, Any] = None

class AlertManager:
    """Manage alerts and notifications for LangGraph applications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.alert_handlers = {
            AlertLevel.INFO: [],
            AlertLevel.WARNING: [self._log_alert],
            AlertLevel.ERROR: [self._log_alert, self._send_email],
            AlertLevel.CRITICAL: [self._log_alert, self._send_email, self._send_slack]
        }
    
    def send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        handlers = self.alert_handlers.get(alert.level, [])
        
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Failed to send alert via {handler.__name__}: {e}")
    
    def _log_alert(self, alert: Alert):
        """Log alert to application logs"""
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }[alert.level]
        
        self.logger.log(
            log_level,
            f"[{alert.component}] {alert.title}: {alert.message}",
            extra=alert.metadata or {}
        )
    
    def _send_email(self, alert: Alert):
        """Send alert via email"""
        if not self.config.get('email_alerts_enabled'):
            return
        
        try:
            msg = MIMEText(f"""
Alert Level: {alert.level.value.upper()}
Component: {alert.component}
Title: {alert.title}
Message: {alert.message}

Metadata:
{alert.metadata or 'None'}
            """)
            
            msg['Subject'] = f"[LangGraph] {alert.level.value.upper()}: {alert.title}"
            msg['From'] = self.config['email']['from']
            msg['To'] = ', '.join(self.config['email']['recipients'])
            
            with smtplib.SMTP(self.config['email']['smtp_host']) as server:
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def _send_slack(self, alert: Alert):
        """Send alert to Slack"""
        if not self.config.get('slack_alerts_enabled'):
            return
            
        try:
            import requests
            
            color_map = {
                AlertLevel.INFO: "good",
                AlertLevel.WARNING: "warning", 
                AlertLevel.ERROR: "danger",
                AlertLevel.CRITICAL: "danger"
            }
            
            payload = {
                "text": f"LangGraph Alert: {alert.title}",
                "attachments": [{
                    "color": color_map[alert.level],
                    "fields": [
                        {"title": "Level", "value": alert.level.value.upper(), "short": True},
                        {"title": "Component", "value": alert.component, "short": True},
                        {"title": "Message", "value": alert.message, "short": False}
                    ]
                }]
            }
            
            response = requests.post(
                self.config['slack']['webhook_url'],
                json=payload
            )
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")

# Health check monitors
class HealthCheckMonitor:
    """Monitor application health and send alerts"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)
    
    def check_graph_execution_health(self, execution_times: List[float]):
        """Check if graph execution is healthy"""
        if not execution_times:
            return
        
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        
        # Alert if average time > 10 seconds
        if avg_time > 10.0:
            self.alert_manager.send_alert(Alert(
                level=AlertLevel.WARNING,
                title="Slow Graph Execution",
                message=f"Average execution time is {avg_time:.2f}s (threshold: 10s)",
                component="graph_execution",
                metadata={"avg_time": avg_time, "max_time": max_time}
            ))
        
        # Alert if max time > 30 seconds
        if max_time > 30.0:
            self.alert_manager.send_alert(Alert(
                level=AlertLevel.ERROR,
                title="Very Slow Graph Execution",
                message=f"Maximum execution time is {max_time:.2f}s (threshold: 30s)",
                component="graph_execution",
                metadata={"avg_time": avg_time, "max_time": max_time}
            ))
    
    def check_memory_usage(self, memory_usage_mb: float):
        """Check memory usage levels"""
        if memory_usage_mb > 1000:  # 1GB
            level = AlertLevel.CRITICAL if memory_usage_mb > 2000 else AlertLevel.WARNING
            
            self.alert_manager.send_alert(Alert(
                level=level,
                title="High Memory Usage",
                message=f"Memory usage is {memory_usage_mb:.2f} MB",
                component="memory",
                metadata={"memory_usage_mb": memory_usage_mb}
            ))
    
    def check_error_rate(self, error_count: int, total_requests: int):
        """Check error rate"""
        if total_requests == 0:
            return
        
        error_rate = error_count / total_requests
        
        if error_rate > 0.1:  # 10% error rate
            level = AlertLevel.CRITICAL if error_rate > 0.25 else AlertLevel.WARNING
            
            self.alert_manager.send_alert(Alert(
                level=level,
                title="High Error Rate",
                message=f"Error rate is {error_rate:.1%} ({error_count}/{total_requests})",
                component="error_handling",
                metadata={
                    "error_count": error_count,
                    "total_requests": total_requests,
                    "error_rate": error_rate
                }
            ))

# Example usage
def setup_production_monitoring():
    """Setup production monitoring and alerting"""
    config = {
        'email_alerts_enabled': True,
        'slack_alerts_enabled': True,
        'email': {
            'smtp_host': 'smtp.gmail.com',
            'from': 'alerts@yourcompany.com',
            'recipients': ['team@yourcompany.com']
        },
        'slack': {
            'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        }
    }
    
    alert_manager = AlertManager(config)
    health_monitor = HealthCheckMonitor(alert_manager)
    
    return alert_manager, health_monitor
```

---

## 18. Migration Patterns

### 18.1 Legacy System Migration

#### From Sequential to Graph-Based Processing

```python
# src/migration/legacy_adapter.py
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import logging

class LegacySystemAdapter(ABC):
    """Base adapter for migrating legacy systems to LangGraph"""
    
    @abstractmethod
    def extract_workflow_steps(self) -> List[Dict[str, Any]]:
        """Extract workflow steps from legacy system"""
        pass
    
    @abstractmethod
    def map_data_formats(self, legacy_data: Any) -> Dict[str, Any]:
        """Map legacy data formats to LangGraph state"""
        pass

class SequentialProcessorAdapter(LegacySystemAdapter):
    """Adapter for sequential processing systems"""
    
    def __init__(self, legacy_processor):
        self.legacy_processor = legacy_processor
        self.logger = logging.getLogger(__name__)
        
    def extract_workflow_steps(self) -> List[Dict[str, Any]]:
        """Extract steps from sequential processor"""
        steps = []
        
        # Map legacy processor methods to graph nodes
        for method_name in dir(self.legacy_processor):
            if method_name.startswith('process_'):
                steps.append({
                    'name': method_name,
                    'type': 'processing',
                    'function': getattr(self.legacy_processor, method_name),
                    'dependencies': self._extract_dependencies(method_name)
                })
        
        return steps
    
    def map_data_formats(self, legacy_data: Any) -> Dict[str, Any]:
        """Map legacy data to graph state"""
        if hasattr(legacy_data, '__dict__'):
            # Object to dict conversion
            return {
                'input_data': legacy_data.__dict__,
                'processing_stage': 'initial',
                'results': [],
                'metadata': {
                    'source': 'legacy_system',
                    'original_type': type(legacy_data).__name__
                }
            }
        else:
            # Simple data types
            return {
                'input_data': legacy_data,
                'processing_stage': 'initial',
                'results': [],
                'metadata': {'source': 'legacy_system'}
            }
    
    def _extract_dependencies(self, method_name: str) -> List[str]:
        """Extract method dependencies (simplified)"""
        # In real implementation, use AST parsing or inspection
        dependency_map = {
            'process_input': [],
            'process_validation': ['process_input'],
            'process_transformation': ['process_validation'],
            'process_output': ['process_transformation']
        }
        return dependency_map.get(method_name, [])

class MigrationManager:
    """Manages the migration process"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.migration_state = {}
    
    def create_parallel_system(self, adapter: LegacySystemAdapter) -> StateGraph:
        """Create parallel LangGraph system"""
        
        # Extract workflow from legacy system
        steps = adapter.extract_workflow_steps()
        
        # Create graph
        graph = StateGraph(Dict[str, Any])
        
        # Add nodes for each step
        for step in steps:
            node_func = self._create_node_function(step, adapter)
            graph.add_node(step['name'], node_func)
        
        # Add edges based on dependencies
        for step in steps:
            if not step['dependencies']:
                graph.add_edge(START, step['name'])
            else:
                for dep in step['dependencies']:
                    graph.add_edge(dep, step['name'])
        
        # Add final edge
        final_nodes = [s['name'] for s in steps if not any(
            s['name'] in other['dependencies'] for other in steps
        )]
        
        for final_node in final_nodes:
            graph.add_edge(final_node, END)
        
        return graph.compile(checkpointer=MemorySaver())
    
    def _create_node_function(self, step: Dict[str, Any], adapter: LegacySystemAdapter):
        """Create node function from legacy step"""
        
        def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Map state to legacy format
                legacy_input = self._state_to_legacy_format(state, step)
                
                # Execute legacy function
                result = step['function'](legacy_input)
                
                # Map result back to state
                updated_state = adapter.map_data_formats(result)
                
                # Merge with existing state
                return {
                    **state,
                    'results': state.get('results', []) + [result],
                    'processing_stage': step['name'],
                    'last_updated': step['name']
                }
                
            except Exception as e:
                self.logger.error(f"Error in {step['name']}: {e}")
                return {
                    **state,
                    'error': str(e),
                    'failed_step': step['name']
                }
        
        return node_function
    
    def _state_to_legacy_format(self, state: Dict[str, Any], step: Dict[str, Any]) -> Any:
        """Convert graph state to legacy format"""
        # This would be customized based on legacy system requirements
        if 'input_data' in state:
            return state['input_data']
        return state

# Example migration implementation
class LegacyOrderProcessor:
    """Example legacy order processing system"""
    
    def process_input(self, order_data):
        return {'order_id': order_data.get('id'), 'status': 'received'}
    
    def process_validation(self, order):
        if order.get('order_id'):
            order['status'] = 'validated'
        else:
            order['status'] = 'invalid'
        return order
    
    def process_transformation(self, order):
        order['transformed_at'] = 'now'
        order['status'] = 'processed'
        return order
    
    def process_output(self, order):
        order['status'] = 'completed'
        return order

def migrate_legacy_system():
    """Example migration process"""
    # Legacy system
    legacy_processor = LegacyOrderProcessor()
    
    # Create adapter
    adapter = SequentialProcessorAdapter(legacy_processor)
    
    # Create migration manager
    migration_manager = MigrationManager()
    
    # Create parallel graph system
    graph = migration_manager.create_parallel_system(adapter)
    
    return graph, legacy_processor
```

### 18.2 Data Migration Strategies

#### State Schema Evolution

```python
# src/migration/schema_migration.py
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
import logging

class MigrationVersion(Enum):
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"

@dataclass
class SchemaMigration:
    from_version: MigrationVersion
    to_version: MigrationVersion
    migration_function: callable
    rollback_function: Optional[callable] = None

class StateSchemaManager:
    """Manages state schema migrations"""
    
    def __init__(self):
        self.migrations: List[SchemaMigration] = []
        self.logger = logging.getLogger(__name__)
        self._register_migrations()
    
    def _register_migrations(self):
        """Register all available migrations"""
        
        # V1.0 -> V1.1: Add metadata field
        self.migrations.append(SchemaMigration(
            from_version=MigrationVersion.V1_0,
            to_version=MigrationVersion.V1_1,
            migration_function=self._migrate_v1_0_to_v1_1,
            rollback_function=self._rollback_v1_1_to_v1_0
        ))
        
        # V1.1 -> V2.0: Restructure data format
        self.migrations.append(SchemaMigration(
            from_version=MigrationVersion.V1_1,
            to_version=MigrationVersion.V2_0,
            migration_function=self._migrate_v1_1_to_v2_0,
            rollback_function=self._rollback_v2_0_to_v1_1
        ))
    
    def migrate_state(self, state: Dict[str, Any], 
                     from_version: MigrationVersion, 
                     to_version: MigrationVersion) -> Dict[str, Any]:
        """Migrate state from one version to another"""
        
        current_state = state.copy()
        current_version = from_version
        
        # Find migration path
        migration_path = self._find_migration_path(from_version, to_version)
        
        if not migration_path:
            raise ValueError(f"No migration path from {from_version} to {to_version}")
        
        # Apply migrations in sequence
        for migration in migration_path:
            try:
                current_state = migration.migration_function(current_state)
                current_version = migration.to_version
                self.logger.info(f"Migrated state from {migration.from_version} to {migration.to_version}")
            except Exception as e:
                self.logger.error(f"Migration failed: {e}")
                raise
        
        # Add version metadata
        current_state['_schema_version'] = to_version.value
        return current_state
    
    def _find_migration_path(self, from_version: MigrationVersion, 
                           to_version: MigrationVersion) -> List[SchemaMigration]:
        """Find path of migrations to apply"""
        path = []
        current_version = from_version
        
        while current_version != to_version:
            next_migration = None
            
            # Find next migration in path
            for migration in self.migrations:
                if migration.from_version == current_version:
                    next_migration = migration
                    break
            
            if not next_migration:
                return []  # No path found
            
            path.append(next_migration)
            current_version = next_migration.to_version
        
        return path
    
    def _migrate_v1_0_to_v1_1(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from v1.0 to v1.1 - add metadata"""
        return {
            **state,
            'metadata': {
                'created_at': 'migration_time',
                'migration_applied': 'v1.0_to_v1.1'
            }
        }
    
    def _rollback_v1_1_to_v1_0(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback from v1.1 to v1.0 - remove metadata"""
        state_copy = state.copy()
        if 'metadata' in state_copy:
            del state_copy['metadata']
        return state_copy
    
    def _migrate_v1_1_to_v2_0(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from v1.1 to v2.0 - restructure data"""
        return {
            'data': {
                'input': state.get('input_data', {}),
                'output': state.get('output_data', {}),
                'intermediate': state.get('results', [])
            },
            'execution': {
                'stage': state.get('processing_stage', 'initial'),
                'errors': state.get('errors', [])
            },
            'metadata': state.get('metadata', {})
        }
    
    def _rollback_v2_0_to_v1_1(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback from v2.0 to v1.1 - flatten structure"""
        data = state.get('data', {})
        execution = state.get('execution', {})
        
        return {
            'input_data': data.get('input', {}),
            'output_data': data.get('output', {}),
            'results': data.get('intermediate', []),
            'processing_stage': execution.get('stage', 'initial'),
            'errors': execution.get('errors', []),
            'metadata': state.get('metadata', {})
        }

# Migration-aware graph wrapper
class MigrationAwareGraph:
    """Graph wrapper that handles state migrations"""
    
    def __init__(self, graph, target_version: MigrationVersion = MigrationVersion.V2_0):
        self.graph = graph
        self.target_version = target_version
        self.schema_manager = StateSchemaManager()
        self.logger = logging.getLogger(__name__)
    
    def invoke(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Invoke graph with automatic migration"""
        
        # Detect input version
        input_version = self._detect_version(input_data)
        
        # Migrate input if needed
        if input_version != self.target_version:
            input_data = self.schema_manager.migrate_state(
                input_data, input_version, self.target_version
            )
        
        # Execute graph
        result = self.graph.invoke(input_data, **kwargs)
        
        return result
    
    def _detect_version(self, state: Dict[str, Any]) -> MigrationVersion:
        """Detect state schema version"""
        if '_schema_version' in state:
            return MigrationVersion(state['_schema_version'])
        
        # Heuristic detection based on structure
        if 'data' in state and 'execution' in state:
            return MigrationVersion.V2_0
        elif 'metadata' in state:
            return MigrationVersion.V1_1
        else:
            return MigrationVersion.V1_0
```

### 18.3 Gradual Migration Strategy

#### Blue-Green Deployment with Feature Flags

```python
# src/migration/gradual_migration.py
from typing import Dict, Any, Optional, Callable
from enum import Enum
import random
import logging
from dataclasses import dataclass

class MigrationPhase(Enum):
    PREPARATION = "preparation"
    CANARY = "canary"
    ROLLING = "rolling"  
    COMPLETE = "complete"

@dataclass
class MigrationConfig:
    phase: MigrationPhase
    traffic_percentage: float  # 0.0 to 1.0
    feature_flags: Dict[str, bool]
    rollback_threshold: float  # Error rate threshold for rollback

class FeatureFlagManager:
    """Manages feature flags for gradual migration"""
    
    def __init__(self):
        self.flags = {}
        self.logger = logging.getLogger(__name__)
    
    def set_flag(self, flag_name: str, enabled: bool, percentage: float = 1.0):
        """Set feature flag"""
        self.flags[flag_name] = {
            'enabled': enabled,
            'percentage': percentage
        }
    
    def is_enabled(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """Check if feature flag is enabled"""
        if flag_name not in self.flags:
            return False
        
        flag = self.flags[flag_name]
        
        if not flag['enabled']:
            return False
        
        # Percentage-based rollout
        if user_id:
            # Consistent hash-based distribution
            hash_value = hash(f"{flag_name}:{user_id}") % 100
            return (hash_value / 100) < flag['percentage']
        else:
            # Random distribution
            return random.random() < flag['percentage']

class GradualMigrationManager:
    """Manages gradual migration from legacy to new system"""
    
    def __init__(self, legacy_system, new_system):
        self.legacy_system = legacy_system
        self.new_system = new_system
        self.feature_flags = FeatureFlagManager()
        self.config = MigrationConfig(
            phase=MigrationPhase.PREPARATION,
            traffic_percentage=0.0,
            feature_flags={},
            rollback_threshold=0.05  # 5% error rate
        )
        self.metrics = {
            'legacy_requests': 0,
            'new_requests': 0,
            'legacy_errors': 0,
            'new_errors': 0,
            'migration_errors': 0
        }
        self.logger = logging.getLogger(__name__)
    
    def process_request(self, request_data: Dict[str, Any], 
                       user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process request with migration logic"""
        
        try:
            # Decide which system to use
            use_new_system = self._should_use_new_system(user_id)
            
            if use_new_system:
                return self._process_with_new_system(request_data, user_id)
            else:
                return self._process_with_legacy_system(request_data, user_id)
                
        except Exception as e:
            self.logger.error(f"Migration error: {e}")
            self.metrics['migration_errors'] += 1
            
            # Fallback to legacy system
            return self._process_with_legacy_system(request_data, user_id)
    
    def _should_use_new_system(self, user_id: Optional[str]) -> bool:
        """Determine if request should use new system"""
        
        # Check migration phase
        if self.config.phase == MigrationPhase.PREPARATION:
            return False
        
        # Check feature flag
        if not self.feature_flags.is_enabled("use_new_system", user_id):
            return False
        
        # Check traffic percentage
        if user_id:
            hash_value = hash(user_id) % 100
            return (hash_value / 100) < self.config.traffic_percentage
        else:
            return random.random() < self.config.traffic_percentage
    
    def _process_with_legacy_system(self, request_data: Dict[str, Any], 
                                  user_id: Optional[str]) -> Dict[str, Any]:
        """Process request with legacy system"""
        try:
            self.metrics['legacy_requests'] += 1
            result = self.legacy_system.process(request_data)
            
            # Shadow testing - also run new system for comparison
            if self.feature_flags.is_enabled("shadow_testing", user_id):
                self._shadow_test_new_system(request_data)
            
            return {
                **result,
                'processed_by': 'legacy_system',
                'migration_metadata': {
                    'phase': self.config.phase.value,
                    'system_used': 'legacy'
                }
            }
            
        except Exception as e:
            self.metrics['legacy_errors'] += 1
            raise
    
    def _process_with_new_system(self, request_data: Dict[str, Any], 
                               user_id: Optional[str]) -> Dict[str, Any]:
        """Process request with new system"""
        try:
            self.metrics['new_requests'] += 1
            
            # Invoke LangGraph system
            result = self.new_system.invoke(request_data)
            
            return {
                **result,
                'processed_by': 'new_system',
                'migration_metadata': {
                    'phase': self.config.phase.value,
                    'system_used': 'new'
                }
            }
            
        except Exception as e:
            self.metrics['new_errors'] += 1
            
            # Fallback to legacy if enabled
            if self.feature_flags.is_enabled("fallback_on_error", user_id):
                self.logger.warning(f"New system failed, falling back to legacy: {e}")
                return self._process_with_legacy_system(request_data, user_id)
            else:
                raise
    
    def _shadow_test_new_system(self, request_data: Dict[str, Any]):
        """Run new system in shadow mode for testing"""
        try:
            # Run new system without affecting response
            result = self.new_system.invoke(request_data)
            
            # Log results for comparison
            self.logger.info(f"Shadow test successful: {result}")
            
        except Exception as e:
            self.logger.error(f"Shadow test failed: {e}")
    
    def advance_migration_phase(self):
        """Advance to next migration phase"""
        current_error_rate = self._calculate_error_rate()
        
        if current_error_rate > self.config.rollback_threshold:
            self.logger.warning(f"Error rate {current_error_rate} exceeds threshold, not advancing")
            return False
        
        if self.config.phase == MigrationPhase.PREPARATION:
            # Start canary deployment
            self.config.phase = MigrationPhase.CANARY
            self.config.traffic_percentage = 0.05  # 5%
            self.feature_flags.set_flag("use_new_system", True, 0.05)
            self.feature_flags.set_flag("shadow_testing", True, 0.1)
            
        elif self.config.phase == MigrationPhase.CANARY:
            # Start rolling deployment
            self.config.phase = MigrationPhase.ROLLING
            self.config.traffic_percentage = 0.5  # 50%
            self.feature_flags.set_flag("use_new_system", True, 0.5)
            
        elif self.config.phase == MigrationPhase.ROLLING:
            # Complete migration
            self.config.phase = MigrationPhase.COMPLETE
            self.config.traffic_percentage = 1.0  # 100%
            self.feature_flags.set_flag("use_new_system", True, 1.0)
            
        self.logger.info(f"Advanced to phase: {self.config.phase}")
        return True
    
    def rollback_migration(self):
        """Rollback migration to previous phase"""
        if self.config.phase == MigrationPhase.COMPLETE:
            self.config.phase = MigrationPhase.ROLLING
            self.config.traffic_percentage = 0.5
            self.feature_flags.set_flag("use_new_system", True, 0.5)
            
        elif self.config.phase == MigrationPhase.ROLLING:
            self.config.phase = MigrationPhase.CANARY
            self.config.traffic_percentage = 0.05
            self.feature_flags.set_flag("use_new_system", True, 0.05)
            
        elif self.config.phase == MigrationPhase.CANARY:
            self.config.phase = MigrationPhase.PREPARATION
            self.config.traffic_percentage = 0.0
            self.feature_flags.set_flag("use_new_system", False)
        
        self.logger.info(f"Rolled back to phase: {self.config.phase}")
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate for new system"""
        if self.metrics['new_requests'] == 0:
            return 0.0
        
        return self.metrics['new_errors'] / self.metrics['new_requests']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get migration metrics"""
        return {
            **self.metrics,
            'error_rates': {
                'legacy': self.metrics['legacy_errors'] / max(1, self.metrics['legacy_requests']),
                'new': self.metrics['new_errors'] / max(1, self.metrics['new_requests']),
            },
            'traffic_split': {
                'legacy_percentage': 1 - self.config.traffic_percentage,
                'new_percentage': self.config.traffic_percentage
            },
            'phase': self.config.phase.value
        }

# Example usage
def setup_gradual_migration(legacy_system, langgraph_system):
    """Setup gradual migration"""
    migration_manager = GradualMigrationManager(legacy_system, langgraph_system)
    
    # Start with shadow testing
    migration_manager.feature_flags.set_flag("shadow_testing", True, 0.1)
    migration_manager.feature_flags.set_flag("fallback_on_error", True)
    
    return migration_manager
```

---

## 19. Enterprise Integration

### 19.1 SSO and Authentication Integration

#### SAML/OAuth Integration

```python
# src/enterprise/auth_integration.py
from typing import Dict, Any, Optional, List
import jwt
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

@dataclass
class UserContext:
    user_id: str
    roles: List[str]
    permissions: List[str]
    org_id: str
    session_id: str
    expires_at: datetime

class EnterpriseAuthManager:
    """Enterprise authentication and authorization manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.jwt_secret = config.get('jwt_secret', 'your-secret-key')
        self.saml_config = config.get('saml', {})
        self.oauth_config = config.get('oauth', {})
    
    def validate_jwt_token(self, token: str) -> Optional[UserContext]:
        """Validate JWT token and extract user context"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            return UserContext(
                user_id=payload['sub'],
                roles=payload.get('roles', []),
                permissions=payload.get('permissions', []),
                org_id=payload.get('org_id', ''),
                session_id=payload.get('session_id', ''),
                expires_at=datetime.fromtimestamp(payload['exp'])
            )
        except jwt.InvalidTokenError as e:
            self.logger.error(f"Invalid JWT token: {e}")
            return None
    
    def validate_saml_response(self, saml_response: str) -> Optional[UserContext]:
        """Validate SAML response and extract user context"""
        try:
            # In a real implementation, use proper SAML library like python3-saml
            # This is a simplified example
            user_data = self._parse_saml_response(saml_response)
            
            return UserContext(
                user_id=user_data['user_id'],
                roles=user_data.get('roles', []),
                permissions=self._map_roles_to_permissions(user_data.get('roles', [])),
                org_id=user_data.get('org_id', ''),
                session_id=user_data.get('session_id', ''),
                expires_at=datetime.now() + timedelta(hours=8)
            )
        except Exception as e:
            self.logger.error(f"SAML validation failed: {e}")
            return None
    
    def validate_oauth_token(self, access_token: str) -> Optional[UserContext]:
        """Validate OAuth access token"""
        try:
            # Validate with OAuth provider
            response = requests.get(
                f"{self.oauth_config['userinfo_endpoint']}",
                headers={'Authorization': f'Bearer {access_token}'},
                timeout=10
            )
            
            if response.status_code == 200:
                user_data = response.json()
                
                return UserContext(
                    user_id=user_data['sub'],
                    roles=user_data.get('roles', []),
                    permissions=self._map_roles_to_permissions(user_data.get('roles', [])),
                    org_id=user_data.get('organization', ''),
                    session_id=access_token[:16],  # Use part of token as session ID
                    expires_at=datetime.now() + timedelta(seconds=user_data.get('exp', 3600))
                )
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"OAuth validation failed: {e}")
            return None
    
    def check_permission(self, user_context: UserContext, required_permission: str) -> bool:
        """Check if user has required permission"""
        return required_permission in user_context.permissions
    
    def _parse_saml_response(self, saml_response: str) -> Dict[str, Any]:
        """Parse SAML response - simplified implementation"""
        # In reality, use proper SAML parsing library
        return {
            'user_id': 'user123',
            'roles': ['data_analyst', 'report_viewer'],
            'org_id': 'org456'
        }
    
    def _map_roles_to_permissions(self, roles: List[str]) -> List[str]:
        """Map roles to permissions"""
        role_permission_map = {
            'admin': ['read', 'write', 'delete', 'manage_users'],
            'data_analyst': ['read', 'write', 'analyze'],
            'report_viewer': ['read'],
            'manager': ['read', 'write', 'approve']
        }
        
        permissions = set()
        for role in roles:
            permissions.update(role_permission_map.get(role, []))
        
        return list(permissions)

class AuthenticatedGraphWrapper:
    """Wrapper that adds authentication to LangGraph execution"""
    
    def __init__(self, graph: StateGraph, auth_manager: EnterpriseAuthManager):
        self.graph = graph
        self.auth_manager = auth_manager
        self.logger = logging.getLogger(__name__)
    
    def invoke(self, input_data: Dict[str, Any], auth_token: str, 
               required_permission: str = 'read', **kwargs) -> Dict[str, Any]:
        """Invoke graph with authentication"""
        
        # Validate authentication
        user_context = self._authenticate(auth_token)
        if not user_context:
            raise PermissionError("Authentication failed")
        
        # Check permissions
        if not self.auth_manager.check_permission(user_context, required_permission):
            raise PermissionError(f"Permission '{required_permission}' required")
        
        # Add user context to state
        authenticated_input = {
            **input_data,
            'user_context': {
                'user_id': user_context.user_id,
                'roles': user_context.roles,
                'permissions': user_context.permissions,
                'org_id': user_context.org_id
            }
        }
        
        # Execute graph
        try:
            result = self.graph.invoke(authenticated_input, **kwargs)
            
            # Add audit trail
            self._log_execution(user_context, input_data, result)
            
            return result
            
        except Exception as e:
            self._log_error(user_context, input_data, str(e))
            raise
    
    def _authenticate(self, auth_token: str) -> Optional[UserContext]:
        """Authenticate user with various methods"""
        
        # Try JWT first
        if auth_token.startswith('eyJ'):  # JWT tokens start with this
            return self.auth_manager.validate_jwt_token(auth_token)
        
        # Try OAuth
        if auth_token.startswith('Bearer '):
            token = auth_token[7:]  # Remove 'Bearer ' prefix
            return self.auth_manager.validate_oauth_token(token)
        
        # Try SAML (if it's a SAML response)
        if '<saml' in auth_token.lower():
            return self.auth_manager.validate_saml_response(auth_token)
        
        return None
    
    def _log_execution(self, user_context: UserContext, 
                      input_data: Dict[str, Any], result: Dict[str, Any]):
        """Log successful execution for audit"""
        self.logger.info(f"Graph executed by {user_context.user_id} from {user_context.org_id}")
    
    def _log_error(self, user_context: UserContext, 
                   input_data: Dict[str, Any], error: str):
        """Log execution error for audit"""
        self.logger.error(f"Graph execution failed for {user_context.user_id}: {error}")
```

### 19.2 Enterprise Data Integration

#### Database and API Integration

```python
# src/enterprise/data_integration.py
from typing import Dict, Any, List, Optional, AsyncGenerator
import asyncio
import aiohttp
import asyncpg
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
import pandas as pd
from dataclasses import dataclass
import logging

@dataclass
class DataSource:
    name: str
    type: str  # 'postgresql', 'mysql', 'api', 'file'
    connection_string: str
    config: Dict[str, Any]

class EnterpriseDataManager:
    """Manages enterprise data sources and integration"""
    
    def __init__(self, data_sources: List[DataSource]):
        self.data_sources = {ds.name: ds for ds in data_sources}
        self.connections = {}
        self.logger = logging.getLogger(__name__)
    
    async def initialize_connections(self):
        """Initialize all data source connections"""
        for name, source in self.data_sources.items():
            try:
                if source.type == 'postgresql':
                    self.connections[name] = await asyncpg.create_pool(source.connection_string)
                elif source.type == 'mysql':
                    # MySQL async connection setup
                    pass
                elif source.type == 'api':
                    # HTTP client session
                    self.connections[name] = aiohttp.ClientSession()
                
                self.logger.info(f"Connected to {name} data source")
            except Exception as e:
                self.logger.error(f"Failed to connect to {name}: {e}")
    
    async def query_data(self, source_name: str, query: str, 
                        params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute query on specified data source"""
        
        if source_name not in self.data_sources:
            raise ValueError(f"Unknown data source: {source_name}")
        
        source = self.data_sources[source_name]
        connection = self.connections.get(source_name)
        
        if not connection:
            raise ConnectionError(f"No connection to {source_name}")
        
        try:
            if source.type == 'postgresql':
                async with connection.acquire() as conn:
                    result = await conn.fetch(query, *(params.values() if params else []))
                    return [dict(row) for row in result]
            
            elif source.type == 'api':
                return await self._query_api(source, query, params)
            
        except Exception as e:
            self.logger.error(f"Query failed for {source_name}: {e}")
            raise
    
    async def _query_api(self, source: DataSource, endpoint: str, 
                        params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query REST API"""
        session = self.connections[source.name]
        
        url = f"{source.config['base_url']}/{endpoint.lstrip('/')}"
        headers = source.config.get('headers', {})
        
        async with session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data if isinstance(data, list) else [data]
            else:
                raise Exception(f"API request failed: {response.status}")
    
    async def stream_data(self, source_name: str, query: str, 
                         batch_size: int = 1000) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """Stream data in batches for large datasets"""
        
        source = self.data_sources[source_name]
        connection = self.connections.get(source_name)
        
        if source.type == 'postgresql':
            async with connection.acquire() as conn:
                async with conn.transaction():
                    cursor = await conn.cursor(query)
                    
                    while True:
                        batch = await cursor.fetch(batch_size)
                        if not batch:
                            break
                        
                        yield [dict(row) for row in batch]
    
    async def execute_transaction(self, source_name: str, operations: List[Dict[str, Any]]) -> bool:
        """Execute multiple operations in a transaction"""
        
        source = self.data_sources[source_name]
        connection = self.connections.get(source_name)
        
        if source.type == 'postgresql':
            async with connection.acquire() as conn:
                async with conn.transaction():
                    try:
                        for op in operations:
                            if op['type'] == 'query':
                                await conn.execute(op['sql'], *op.get('params', []))
                            elif op['type'] == 'bulk_insert':
                                await conn.executemany(op['sql'], op['data'])
                        return True
                    except Exception as e:
                        self.logger.error(f"Transaction failed: {e}")
                        raise
    
    async def close_connections(self):
        """Close all connections"""
        for name, connection in self.connections.items():
            try:
                if hasattr(connection, 'close'):
                    if asyncio.iscoroutinefunction(connection.close):
                        await connection.close()
                    else:
                        connection.close()
                self.logger.info(f"Closed connection to {name}")
            except Exception as e:
                self.logger.error(f"Error closing {name}: {e}")

# Data integration nodes for LangGraph
class DataIntegrationNodes:
    """LangGraph nodes for enterprise data integration"""
    
    def __init__(self, data_manager: EnterpriseDataManager):
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
    
    async def fetch_customer_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch customer data from CRM system"""
        customer_id = state.get('customer_id')
        
        if not customer_id:
            return {**state, 'error': 'Customer ID required'}
        
        try:
            query = "SELECT * FROM customers WHERE id = $1"
            customer_data = await self.data_manager.query_data(
                'crm_db', query, {'customer_id': customer_id}
            )
            
            return {
                **state,
                'customer_data': customer_data[0] if customer_data else None
            }
        except Exception as e:
            return {**state, 'error': f'Failed to fetch customer data: {e}'}
    
    async def fetch_financial_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch financial data from financial systems"""
        customer_id = state.get('customer_id')
        
        try:
            # Query multiple financial data sources
            tasks = [
                self.data_manager.query_data(
                    'financial_db',
                    "SELECT * FROM transactions WHERE customer_id = $1 ORDER BY date DESC LIMIT 100",
                    {'customer_id': customer_id}
                ),
                self.data_manager.query_data(
                    'credit_api',
                    f"credit-scores/{customer_id}"
                )
            ]
            
            transactions, credit_scores = await asyncio.gather(*tasks)
            
            return {
                **state,
                'financial_data': {
                    'transactions': transactions,
                    'credit_scores': credit_scores
                }
            }
        except Exception as e:
            return {**state, 'error': f'Failed to fetch financial data: {e}'}
    
    async def enrich_with_external_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich data with external sources"""
        try:
            customer_data = state.get('customer_data', {})
            company_name = customer_data.get('company')
            
            if company_name:
                # Fetch company data from external API
                company_info = await self.data_manager.query_data(
                    'external_api',
                    f"companies/search?name={company_name}"
                )
                
                return {
                    **state,
                    'enriched_data': {
                        **customer_data,
                        'company_info': company_info[0] if company_info else None
                    }
                }
            else:
                return {**state, 'enriched_data': customer_data}
                
        except Exception as e:
            return {**state, 'error': f'Data enrichment failed: {e}'}
    
    async def aggregate_insights(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate data from multiple sources into insights"""
        try:
            customer_data = state.get('customer_data', {})
            financial_data = state.get('financial_data', {})
            enriched_data = state.get('enriched_data', {})
            
            # Calculate insights
            transactions = financial_data.get('transactions', [])
            avg_transaction_value = sum(t.get('amount', 0) for t in transactions) / len(transactions) if transactions else 0
            
            insights = {
                'customer_profile': {
                    'id': customer_data.get('id'),
                    'name': customer_data.get('name'),
                    'segment': customer_data.get('segment', 'standard')
                },
                'financial_summary': {
                    'avg_transaction_value': avg_transaction_value,
                    'transaction_count': len(transactions),
                    'credit_score': financial_data.get('credit_scores', {}).get('score', 0)
                },
                'risk_assessment': self._calculate_risk_score(customer_data, financial_data),
                'recommendations': self._generate_recommendations(customer_data, financial_data)
            }
            
            return {**state, 'insights': insights}
            
        except Exception as e:
            return {**state, 'error': f'Insight aggregation failed: {e}'}
    
    def _calculate_risk_score(self, customer_data: Dict, financial_data: Dict) -> float:
        """Calculate customer risk score"""
        base_score = 0.5
        
        # Adjust based on credit score
        credit_score = financial_data.get('credit_scores', {}).get('score', 600)
        if credit_score > 750:
            base_score -= 0.2
        elif credit_score < 600:
            base_score += 0.3
        
        # Adjust based on transaction history
        transactions = financial_data.get('transactions', [])
        if len(transactions) > 50:
            base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _generate_recommendations(self, customer_data: Dict, financial_data: Dict) -> List[str]:
        """Generate customer recommendations"""
        recommendations = []
        
        credit_score = financial_data.get('credit_scores', {}).get('score', 600)
        if credit_score > 750:
            recommendations.append("Eligible for premium financial products")
        
        transactions = financial_data.get('transactions', [])
        if len(transactions) > 100:
            recommendations.append("High-value customer - prioritize for retention")
        
        return recommendations

# Example enterprise graph setup
async def create_enterprise_data_graph():
    """Create enterprise data processing graph"""
    
    # Setup data sources
    data_sources = [
        DataSource(
            name='crm_db',
            type='postgresql',
            connection_string='postgresql://user:pass@localhost/crm',
            config={}
        ),
        DataSource(
            name='financial_db',
            type='postgresql',
            connection_string='postgresql://user:pass@localhost/finance',
            config={}
        ),
        DataSource(
            name='external_api',
            type='api',
            connection_string='',
            config={
                'base_url': 'https://api.external-provider.com',
                'headers': {'Authorization': 'Bearer your-api-key'}
            }
        )
    ]
    
    # Initialize data manager
    data_manager = EnterpriseDataManager(data_sources)
    await data_manager.initialize_connections()
    
    # Create nodes
    nodes = DataIntegrationNodes(data_manager)
    
    # Build graph
    graph = StateGraph(Dict[str, Any])
    graph.add_node("fetch_customer", nodes.fetch_customer_data)
    graph.add_node("fetch_financial", nodes.fetch_financial_data)
    graph.add_node("enrich_data", nodes.enrich_with_external_data)
    graph.add_node("aggregate_insights", nodes.aggregate_insights)
    
    # Add edges
    graph.add_edge("fetch_customer", "fetch_financial")
    graph.add_edge("fetch_financial", "enrich_data")
    graph.add_edge("enrich_data", "aggregate_insights")
    
    return graph.compile(checkpointer=MemorySaver()), data_manager
```

### 19.3 Compliance and Governance

#### GDPR, SOX, and Audit Trail Implementation

```python
# src/enterprise/compliance.py
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import logging
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class ProcessingPurpose(Enum):
    CUSTOMER_SERVICE = "customer_service"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    COMPLIANCE = "compliance"
    OPERATIONS = "operations"

@dataclass
class DataProcessingRecord:
    data_subject_id: str
    processing_purpose: ProcessingPurpose
    data_classification: DataClassification
    data_elements: List[str]
    legal_basis: str
    retention_period: timedelta
    consent_given: bool = False
    consent_timestamp: Optional[datetime] = None
    processing_timestamp: datetime = field(default_factory=datetime.now)

class GDPRComplianceManager:
    """GDPR compliance management for enterprise LangGraph applications"""
    
    def __init__(self, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.logger = logging.getLogger(__name__)
        self._create_tables()
    
    def _create_tables(self):
        """Create compliance tracking tables"""
        Base = declarative_base()
        
        class DataProcessingLog(Base):
            __tablename__ = "data_processing_log"
            
            id = Column(String, primary_key=True)
            data_subject_id = Column(String, nullable=False)
            processing_purpose = Column(String, nullable=False)
            data_classification = Column(String, nullable=False)
            data_elements = Column(Text, nullable=False)  # JSON array
            legal_basis = Column(String, nullable=False)
            consent_given = Column(Boolean, default=False)
            consent_timestamp = Column(DateTime)
            processing_timestamp = Column(DateTime, default=datetime.now)
            retention_until = Column(DateTime, nullable=False)
        
        Base.metadata.create_all(bind=self.engine)
    
    def record_data_processing(self, record: DataProcessingRecord) -> str:
        """Record data processing activity for GDPR compliance"""
        
        record_id = hashlib.sha256(
            f"{record.data_subject_id}:{record.processing_timestamp.isoformat()}".encode()
        ).hexdigest()
        
        db = self.SessionLocal()
        try:
            processing_log = {
                'id': record_id,
                'data_subject_id': record.data_subject_id,
                'processing_purpose': record.processing_purpose.value,
                'data_classification': record.data_classification.value,
                'data_elements': json.dumps(record.data_elements),
                'legal_basis': record.legal_basis,
                'consent_given': record.consent_given,
                'consent_timestamp': record.consent_timestamp,
                'processing_timestamp': record.processing_timestamp,
                'retention_until': record.processing_timestamp + record.retention_period
            }
            
            # Insert into database (using raw SQL for simplicity)
            db.execute(
                "INSERT INTO data_processing_log VALUES (:id, :data_subject_id, :processing_purpose, :data_classification, :data_elements, :legal_basis, :consent_given, :consent_timestamp, :processing_timestamp, :retention_until)",
                processing_log
            )
            db.commit()
            
            self.logger.info(f"Recorded data processing: {record_id}")
            return record_id
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Failed to record data processing: {e}")
            raise
        finally:
            db.close()
    
    def check_data_retention(self) -> List[str]:
        """Check for data that should be deleted per retention policies"""
        db = self.SessionLocal()
        try:
            result = db.execute(
                "SELECT id, data_subject_id FROM data_processing_log WHERE retention_until < :now",
                {'now': datetime.now()}
            )
            
            expired_records = [{'id': row[0], 'data_subject_id': row[1]} for row in result]
            
            if expired_records:
                self.logger.warning(f"Found {len(expired_records)} records past retention period")
            
            return expired_records
            
        finally:
            db.close()
    
    def handle_data_subject_request(self, data_subject_id: str, request_type: str) -> Dict[str, Any]:
        """Handle GDPR data subject requests (access, portability, deletion)"""
        db = self.SessionLocal()
        try:
            if request_type == 'access':
                # Right to access
                result = db.execute(
                    "SELECT * FROM data_processing_log WHERE data_subject_id = :subject_id",
                    {'subject_id': data_subject_id}
                )
                records = [dict(row) for row in result]
                
                return {
                    'request_type': 'access',
                    'data_subject_id': data_subject_id,
                    'records': records,
                    'total_records': len(records)
                }
            
            elif request_type == 'deletion':
                # Right to be forgotten
                result = db.execute(
                    "DELETE FROM data_processing_log WHERE data_subject_id = :subject_id",
                    {'subject_id': data_subject_id}
                )
                deleted_count = result.rowcount
                db.commit()
                
                self.logger.info(f"Deleted {deleted_count} records for data subject {data_subject_id}")
                
                return {
                    'request_type': 'deletion',
                    'data_subject_id': data_subject_id,
                    'deleted_records': deleted_count
                }
            
            elif request_type == 'portability':
                # Data portability
                result = db.execute(
                    "SELECT data_elements, processing_timestamp FROM data_processing_log WHERE data_subject_id = :subject_id",
                    {'subject_id': data_subject_id}
                )
                
                portable_data = {}
                for row in result:
                    data_elements = json.loads(row[0])
                    timestamp = row[1].isoformat()
                    portable_data[timestamp] = data_elements
                
                return {
                    'request_type': 'portability',
                    'data_subject_id': data_subject_id,
                    'portable_data': portable_data
                }
                
        finally:
            db.close()

class SOXComplianceManager:
    """SOX compliance for financial data processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.audit_trail = []
    
    def create_control_checkpoint(self, control_id: str, control_description: str, 
                                state: Dict[str, Any]) -> Dict[str, Any]:
        """Create SOX control checkpoint"""
        
        checkpoint = {
            'control_id': control_id,
            'description': control_description,
            'timestamp': datetime.now().isoformat(),
            'state_hash': self._hash_state(state),
            'data_elements': list(state.keys()),
            'validation_status': 'passed'
        }
        
        self.audit_trail.append(checkpoint)
        
        # Add control metadata to state
        return {
            **state,
            'sox_controls': state.get('sox_controls', []) + [checkpoint]
        }
    
    def validate_financial_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate financial data per SOX requirements"""
        
        financial_data = state.get('financial_data', {})
        validation_errors = []
        
        # Check data completeness
        required_fields = ['amount', 'date', 'account_id', 'transaction_type']
        for field in required_fields:
            if field not in financial_data:
                validation_errors.append(f"Missing required field: {field}")
        
        # Check data quality
        amount = financial_data.get('amount', 0)
        if not isinstance(amount, (int, float)) or amount < 0:
            validation_errors.append("Invalid amount value")
        
        # Record validation
        control_state = self.create_control_checkpoint(
            'SOX-001',
            'Financial Data Validation',
            state
        )
        
        if validation_errors:
            control_state['validation_errors'] = validation_errors
            control_state['sox_controls'][-1]['validation_status'] = 'failed'
        
        return control_state
    
    def segregation_of_duties_check(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure segregation of duties compliance"""
        
        user_context = state.get('user_context', {})
        operation_type = state.get('operation_type', 'read')
        
        user_roles = user_context.get('roles', [])
        
        # Check for conflicting roles
        conflicting_combinations = [
            (['approver', 'preparer'], 'Cannot approve and prepare same transaction'),
            (['auditor', 'processor'], 'Cannot audit and process same data')
        ]
        
        segregation_violations = []
        for conflicting_roles, message in conflicting_combinations:
            if all(role in user_roles for role in conflicting_roles):
                segregation_violations.append(message)
        
        control_state = self.create_control_checkpoint(
            'SOX-002',
            'Segregation of Duties Check',
            state
        )
        
        if segregation_violations:
            control_state['segregation_violations'] = segregation_violations
            control_state['sox_controls'][-1]['validation_status'] = 'failed'
        
        return control_state
    
    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Create hash of state for audit purposes"""
        state_json = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(state_json.encode()).hexdigest()
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate SOX audit report"""
        
        total_controls = len(self.audit_trail)
        passed_controls = sum(1 for c in self.audit_trail if c['validation_status'] == 'passed')
        failed_controls = total_controls - passed_controls
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'total_controls_executed': total_controls,
            'controls_passed': passed_controls,
            'controls_failed': failed_controls,
            'compliance_rate': passed_controls / total_controls if total_controls > 0 else 0,
            'audit_trail': self.audit_trail
        }

class ComplianceAwareGraphWrapper:
    """Graph wrapper that ensures compliance with enterprise regulations"""
    
    def __init__(self, graph, gdpr_manager: GDPRComplianceManager, 
                 sox_manager: SOXComplianceManager):
        self.graph = graph
        self.gdpr_manager = gdpr_manager
        self.sox_manager = sox_manager
        self.logger = logging.getLogger(__name__)
    
    def invoke(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Invoke graph with compliance checks"""
        
        # Pre-processing compliance checks
        compliant_input = self._apply_compliance_checks(input_data)
        
        # Record GDPR processing
        if 'user_context' in compliant_input:
            self._record_gdpr_processing(compliant_input)
        
        # Execute graph
        result = self.graph.invoke(compliant_input, **kwargs)
        
        # Post-processing compliance
        compliant_result = self._finalize_compliance(result)
        
        return compliant_result
    
    def _apply_compliance_checks(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply pre-processing compliance checks"""
        
        # Apply SOX controls if financial data present
        if 'financial_data' in state:
            state = self.sox_manager.validate_financial_data(state)
            state = self.sox_manager.segregation_of_duties_check(state)
        
        # Check data classification
        state = self._classify_data(state)
        
        return state
    
    def _record_gdpr_processing(self, state: Dict[str, Any]):
        """Record GDPR data processing activity"""
        
        user_context = state.get('user_context', {})
        data_subject_id = user_context.get('user_id', 'unknown')
        
        # Determine processing purpose
        processing_purpose = ProcessingPurpose.ANALYTICS  # Default
        if 'customer_service' in str(state).lower():
            processing_purpose = ProcessingPurpose.CUSTOMER_SERVICE
        elif 'marketing' in str(state).lower():
            processing_purpose = ProcessingPurpose.MARKETING
        
        # Create processing record
        record = DataProcessingRecord(
            data_subject_id=data_subject_id,
            processing_purpose=processing_purpose,
            data_classification=DataClassification.INTERNAL,
            data_elements=list(state.keys()),
            legal_basis="Legitimate interest",
            retention_period=timedelta(days=365),
            consent_given=True,  # Assume consent for this example
            consent_timestamp=datetime.now()
        )
        
        self.gdpr_manager.record_data_processing(record)
    
    def _classify_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classify data elements according to sensitivity"""
        
        sensitive_patterns = ['ssn', 'social_security', 'credit_card', 'password', 'tax_id']
        confidential_patterns = ['financial', 'salary', 'revenue', 'profit']
        
        data_classification = DataClassification.PUBLIC
        
        state_str = json.dumps(state, default=str).lower()
        
        if any(pattern in state_str for pattern in sensitive_patterns):
            data_classification = DataClassification.RESTRICTED
        elif any(pattern in state_str for pattern in confidential_patterns):
            data_classification = DataClassification.CONFIDENTIAL
        elif 'internal' in state_str:
            data_classification = DataClassification.INTERNAL
        
        return {
            **state,
            'data_classification': data_classification.value
        }
    
    def _finalize_compliance(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize compliance checks and reporting"""
        
        # Generate compliance summary
        compliance_summary = {
            'gdpr_compliant': True,  # Simplified
            'sox_compliant': all(
                c['validation_status'] == 'passed' 
                for c in result.get('sox_controls', [])
            ),
            'data_classification': result.get('data_classification', 'public'),
            'audit_trail_id': hashlib.sha256(
                json.dumps(result, sort_keys=True, default=str).encode()
            ).hexdigest()
        }
        
        return {
            **result,
            'compliance_summary': compliance_summary
        }

# Example enterprise compliance setup
def setup_enterprise_compliance_graph(base_graph):
    """Setup enterprise graph with full compliance"""
    
    # Initialize compliance managers
    gdpr_manager = GDPRComplianceManager("sqlite:///compliance.db")
    sox_manager = SOXComplianceManager()
    
    # Wrap graph with compliance
    compliant_graph = ComplianceAwareGraphWrapper(base_graph, gdpr_manager, sox_manager)
    
    return compliant_graph, gdpr_manager, sox_manager
```

---

## Conclusion

This comprehensive LangGraph guide provides everything you need to master LangGraph development from basic concepts to production deployment. The guide follows a progressive learning approach with:

✅ **Complete Installation & Setup** - Step-by-step installation with verification
✅ **Progressive Learning Path** - From basic concepts to advanced patterns  
✅ **Real Project Structure** - Complete directory organization and configuration
✅ **Comprehensive Testing** - Unit, integration, e2e, and performance testing
✅ **Production Deployment** - Docker, Kubernetes, Lambda, and FastAPI examples
✅ **Debugging & Troubleshooting** - Tools and techniques for problem resolution
✅ **Monitoring & Observability** - Complete monitoring and alerting setup

### Key Takeaways:

1. **Graph Design**: Start simple and gradually add complexity
2. **State Management**: Use TypedDict for type safety and clear state schemas  
3. **Testing Strategy**: Implement comprehensive testing at all levels
4. **Performance**: Profile regularly and optimize bottlenecks
5. **Production**: Plan for monitoring, alerting, and scalability from day one

### Next Steps:

1. Follow the installation guide to set up your development environment
2. Work through the progressive examples in order
3. Implement the testing strategies for your specific use case
4. Deploy using the production examples that match your infrastructure
5. Set up monitoring and alerting for production systems

This guide serves as your complete reference for building robust, scalable LangGraph applications. Keep it handy as you develop and deploy your own LangGraph projects!

---

*Happy coding with LangGraph! 🎉*
2. **Integration Testing** - Component interactions  
3. **End-to-End Testing** - Complete workflows
4. **Performance Testing** - Load and stress testing
5. **Benchmark Testing** - Performance comparisons

You can run tests with:
```bash
# Quick tests only
./scripts/run_tests.sh --quick

# All tests including slow ones
./scripts/run_tests.sh --include-slow

# With coverage report
./scripts/run_tests.sh --coverage
```