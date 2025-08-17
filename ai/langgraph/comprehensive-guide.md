# LangGraph Complete Guide - From Zero to Production

LangGraph is a revolutionary framework for building stateful, multi-agent AI applications with cyclical workflows. This comprehensive guide takes you from absolute beginner to production deployment, covering every aspect with hands-on examples and real-world projects.

## Table of Contents

- [Installation and Setup](#installation-and-setup)
- [Development Environment](#development-environment) 
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Basic Graphs](#basic-graphs)
- [State Management](#state-management)
- [Node Development](#node-development)
- [Edge Configuration](#edge-configuration)
- [Memory and Checkpointing](#memory-and-checkpointing)
- [Multi-Agent Systems](#multi-agent-systems)
- [Advanced Patterns](#advanced-patterns)
- [Real-World Project Structure](#real-world-project-structure)
- [Testing Strategies](#testing-strategies)
- [Performance Optimization](#performance-optimization)
- [Production Deployment](#production-deployment)
- [Monitoring and Observability](#monitoring-and-observability)
- [Troubleshooting](#troubleshooting)
- [Migration Patterns](#migration-patterns)
- [Enterprise Integration](#enterprise-integration)

---

## Installation and Setup

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

## Development Environment

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

## Quick Start

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

## Core Concepts

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

## Basic Graphs

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

## Testing Strategies

Testing is crucial for building reliable LangGraph applications. Let's explore comprehensive testing approaches from unit tests to end-to-end testing.

### Test Structure Overview

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

## Chapter 13: Production Deployment Examples

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

## Chapter 14: Debugging and Troubleshooting

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