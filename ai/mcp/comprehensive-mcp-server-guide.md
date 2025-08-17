# Comprehensive MCP Server Development Guide

## Table of Contents

1. [Introduction to MCP](#chapter-1-introduction-to-mcp)
2. [MCP Fundamentals and Architecture](#chapter-2-mcp-fundamentals-and-architecture)
3. [Setting Up Development Environment](#chapter-3-setting-up-development-environment)
4. [Basic MCP Server Implementation](#chapter-4-basic-mcp-server-implementation)
5. [Resources and Resource Management](#chapter-5-resources-and-resource-management)
6. [Tools and Tool Implementation](#chapter-6-tools-and-tool-implementation)
7. [Prompts and Prompt Templates](#chapter-7-prompts-and-prompt-templates)
8. [Advanced Server Patterns](#chapter-8-advanced-server-patterns)
9. [Error Handling and Validation](#chapter-9-error-handling-and-validation)
10. [Testing MCP Servers](#chapter-10-testing-mcp-servers)
11. [Security and Authentication](#chapter-11-security-and-authentication)
12. [Performance Optimization](#chapter-12-performance-optimization)
13. [Deployment Strategies](#chapter-13-deployment-strategies)
14. [Real-World MCP Server Projects](#chapter-14-real-world-mcp-server-projects)
15. [Debugging and Troubleshooting](#chapter-15-debugging-and-troubleshooting)

---

## Chapter 1: Introduction to MCP

The Model Context Protocol (MCP) is a revolutionary open protocol that enables AI models to securely connect to external data sources and systems. It provides a standardized way to expose tools, resources, and prompts to AI models.

### 1.1 What is MCP?

MCP allows AI models to:
- Access external data sources (databases, APIs, files)
- Execute tools and functions
- Use dynamic prompt templates
- Maintain security boundaries
- Enable composable AI workflows

### 1.2 Key Benefits

- **Standardization**: Universal protocol for AI-system integration
- **Security**: Built-in authentication and authorization
- **Scalability**: Distributed architecture supports multiple servers
- **Flexibility**: Supports various resource types and tools
- **Interoperability**: Works with different AI models and platforms

### 1.3 MCP Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Model      │◄──►│   MCP Client     │◄──►│   MCP Server    │
│   (Claude, etc.)│    │   (Claude Code)  │    │   (Your Code)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   Resources     │
                                               │   Tools         │
                                               │   Prompts       │
                                               └─────────────────┘
```

### 1.4 Core Concepts

- **Server**: Exposes resources, tools, and prompts to clients
- **Client**: Consumes server capabilities (typically AI models)
- **Resources**: Data sources (files, databases, APIs)
- **Tools**: Executable functions and operations
- **Prompts**: Dynamic template systems

---

## Chapter 2: MCP Fundamentals and Architecture

### 2.1 Protocol Specification

MCP is built on JSON-RPC 2.0 and supports multiple transport layers:
- Standard I/O (stdio)
- Server-Sent Events (SSE)
- WebSockets
- HTTP

### 2.2 Message Flow

```python
# ai/mcp/examples/01_basic_concepts.py

from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass
from enum import Enum

class MessageType(Enum):
    """MCP message types"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"

@dataclass
class MCPMessage:
    """Base MCP message structure"""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class MCPCapabilities:
    """MCP server capabilities"""
    
    def __init__(self):
        self.resources = True
        self.tools = True
        self.prompts = True
        self.roots = True
        self.sampling = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resources": {"subscribe": True, "listChanged": True} if self.resources else None,
            "tools": {"listChanged": True} if self.tools else None,
            "prompts": {"listChanged": True} if self.prompts else None,
            "roots": {"listChanged": True} if self.roots else None,
            "sampling": {} if self.sampling else None
        }

def create_initialize_response(server_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create MCP initialize response"""
    capabilities = MCPCapabilities()
    
    return {
        "jsonrpc": "2.0",
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": capabilities.to_dict(),
            "serverInfo": server_info
        }
    }

# Example usage
if __name__ == "__main__":
    print("MCP Fundamentals Demo")
    print("=" * 30)
    
    # Server info
    server_info = {
        "name": "demo-mcp-server",
        "version": "1.0.0"
    }
    
    # Create initialize response
    init_response = create_initialize_response(server_info)
    print("Initialize Response:")
    print(json.dumps(init_response, indent=2))
```

### 2.3 Resource System

Resources in MCP represent data sources that can be read by AI models:

```python
# ai/mcp/examples/02_resource_concepts.py

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse
import mimetypes

@dataclass
class ResourceReference:
    """Reference to an MCP resource"""
    uri: str
    name: Optional[str] = None
    description: Optional[str] = None
    mimeType: Optional[str] = None

@dataclass
class ResourceContent:
    """Content of an MCP resource"""
    uri: str
    mimeType: str
    text: Optional[str] = None
    blob: Optional[bytes] = None

class ResourceManager:
    """Manages MCP resources"""
    
    def __init__(self):
        self.resources: Dict[str, ResourceReference] = {}
        self.content_cache: Dict[str, ResourceContent] = {}
    
    def register_resource(self, uri: str, name: str = None, 
                         description: str = None, mime_type: str = None) -> ResourceReference:
        """Register a new resource"""
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(uri)
            mime_type = mime_type or "text/plain"
        
        resource = ResourceReference(
            uri=uri,
            name=name or self._extract_name_from_uri(uri),
            description=description,
            mimeType=mime_type
        )
        
        self.resources[uri] = resource
        return resource
    
    def _extract_name_from_uri(self, uri: str) -> str:
        """Extract name from URI"""
        parsed = urlparse(uri)
        if parsed.path:
            return parsed.path.split('/')[-1]
        return uri
    
    def list_resources(self) -> List[ResourceReference]:
        """List all registered resources"""
        return list(self.resources.values())
    
    def get_resource_content(self, uri: str) -> Optional[ResourceContent]:
        """Get resource content (placeholder for actual implementation)"""
        if uri in self.content_cache:
            return self.content_cache[uri]
        
        # In real implementation, this would fetch actual content
        resource = self.resources.get(uri)
        if not resource:
            return None
        
        # Mock content
        content = ResourceContent(
            uri=uri,
            mimeType=resource.mimeType,
            text=f"Mock content for resource: {uri}"
        )
        
        self.content_cache[uri] = content
        return content

# Example usage
if __name__ == "__main__":
    print("Resource Management Demo")
    print("=" * 30)
    
    manager = ResourceManager()
    
    # Register resources
    manager.register_resource(
        uri="file:///project/readme.md",
        name="Project README",
        description="Main project documentation"
    )
    
    manager.register_resource(
        uri="https://api.example.com/data",
        name="API Data",
        description="External API data source",
        mime_type="application/json"
    )
    
    # List resources
    resources = manager.list_resources()
    print(f"Registered {len(resources)} resources:")
    for resource in resources:
        print(f"  - {resource.name}: {resource.uri}")
        print(f"    Type: {resource.mimeType}")
        print(f"    Description: {resource.description}")
    
    # Get content
    content = manager.get_resource_content("file:///project/readme.md")
    if content:
        print(f"\nContent for {content.uri}:")
        print(f"MIME Type: {content.mimeType}")
        print(f"Text: {content.text}")
```

---

## Chapter 3: Setting Up Development Environment

### 3.1 Installation and Dependencies

```bash
# ai/mcp/setup.sh

#!/bin/bash
echo "Setting up MCP Server Development Environment"

# Create project directory
mkdir -p mcp-server-project
cd mcp-server-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -e .

# Install development dependencies
pip install pytest pytest-asyncio black isort mypy

# Create project structure
mkdir -p src/mcp_server
mkdir -p tests
mkdir -p examples
mkdir -p docs

echo "✅ MCP Server development environment ready!"
```

### 3.2 Project Structure

```
mcp-server-project/
├── src/
│   └── mcp_server/
│       ├── __init__.py
│       ├── server.py
│       ├── resources.py
│       ├── tools.py
│       └── prompts.py
├── tests/
│   ├── test_server.py
│   ├── test_resources.py
│   └── test_tools.py
├── examples/
│   ├── basic_server.py
│   └── advanced_server.py
├── pyproject.toml
├── README.md
└── .gitignore
```

### 3.3 Package Configuration

```toml
# ai/mcp/pyproject.toml

[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mcp-server"
version = "0.1.0"
description = "Comprehensive MCP Server Implementation"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.8"

dependencies = [
    "pydantic>=2.0.0",
    "asyncio-mqtt>=0.11.0",
    "websockets>=11.0.0",
    "aiofiles>=23.0.0",
    "httpx>=0.24.0",
    "typing-extensions>=4.5.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.4.0",
    "coverage>=7.2.0"
]

[project.scripts]
mcp-server = "mcp_server.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.8"
strict = true
```

---

## Chapter 4: Basic MCP Server Implementation

### 4.1 Core Server Framework

```python
# ai/mcp/examples/04_basic_server.py

import asyncio
import json
import sys
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPError(Exception):
    """Base MCP error"""
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)

class MCPServer:
    """Basic MCP Server implementation"""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.initialized = False
        self.capabilities = {
            "resources": {"subscribe": True, "listChanged": True},
            "tools": {"listChanged": True},
            "prompts": {"listChanged": True}
        }
        
        # Message handlers
        self.handlers = {
            "initialize": self.handle_initialize,
            "initialized": self.handle_initialized,
            "resources/list": self.handle_list_resources,
            "resources/read": self.handle_read_resource,
            "tools/list": self.handle_list_tools,
            "tools/call": self.handle_call_tool,
            "prompts/list": self.handle_list_prompts,
            "prompts/get": self.handle_get_prompt
        }
        
        logger.info(f"MCP Server '{name}' v{version} created")
    
    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming MCP message"""
        try:
            method = message.get("method")
            params = message.get("params", {})
            msg_id = message.get("id")
            
            if method in self.handlers:
                logger.info(f"Handling method: {method}")
                result = await self.handlers[method](params)
                
                if msg_id is not None:  # Request (not notification)
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": result
                    }
                return None
            else:
                raise MCPError(-32601, f"Method not found: {method}")
                
        except MCPError as e:
            logger.error(f"MCP Error: {e.message}")
            if msg_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": e.code,
                        "message": e.message,
                        "data": e.data
                    }
                }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if msg_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32603,
                        "message": "Internal error",
                        "data": str(e)
                    }
                }
        
        return None
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request"""
        client_info = params.get("clientInfo", {})
        protocol_version = params.get("protocolVersion", "2024-11-05")
        
        logger.info(f"Initialize from client: {client_info.get('name', 'unknown')}")
        
        return {
            "protocolVersion": protocol_version,
            "capabilities": self.capabilities,
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }
    
    async def handle_initialized(self, params: Dict[str, Any]) -> None:
        """Handle initialized notification"""
        self.initialized = True
        logger.info("Server initialized successfully")
    
    async def handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list resources request"""
        # Override in subclass to provide actual resources
        return {"resources": []}
    
    async def handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle read resource request"""
        uri = params.get("uri")
        if not uri:
            raise MCPError(-32602, "Missing required parameter: uri")
        
        # Override in subclass to provide actual resource content
        return {
            "contents": [{
                "uri": uri,
                "mimeType": "text/plain",
                "text": f"Mock content for {uri}"
            }]
        }
    
    async def handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list tools request"""
        # Override in subclass to provide actual tools
        return {"tools": []}
    
    async def handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call tool request"""
        name = params.get("name")
        if not name:
            raise MCPError(-32602, "Missing required parameter: name")
        
        # Override in subclass to provide actual tool execution
        return {
            "content": [{
                "type": "text",
                "text": f"Mock result for tool: {name}"
            }]
        }
    
    async def handle_list_prompts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list prompts request"""
        # Override in subclass to provide actual prompts
        return {"prompts": []}
    
    async def handle_get_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get prompt request"""
        name = params.get("name")
        if not name:
            raise MCPError(-32602, "Missing required parameter: name")
        
        # Override in subclass to provide actual prompt content
        return {
            "description": f"Mock prompt: {name}",
            "messages": [{
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"This is a mock prompt for {name}"
                }
            }]
        }

class StdioMCPServer(MCPServer):
    """MCP Server with stdio transport"""
    
    async def run(self):
        """Run the server with stdio transport"""
        logger.info("Starting MCP Server with stdio transport")
        
        try:
            while True:
                # Read from stdin
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON message
                    message = json.loads(line)
                    
                    # Handle message
                    response = await self.handle_message(message)
                    
                    # Send response if any
                    if response:
                        print(json.dumps(response), flush=True)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        }
                    }
                    print(json.dumps(error_response), flush=True)
                    
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            logger.info("MCP Server stopped")

# Example implementation
class DemoMCPServer(StdioMCPServer):
    """Demo MCP Server with sample resources and tools"""
    
    def __init__(self):
        super().__init__("demo-server", "1.0.0")
        self.demo_resources = [
            {
                "uri": "demo://readme",
                "name": "Demo README",
                "description": "Demo server documentation",
                "mimeType": "text/markdown"
            },
            {
                "uri": "demo://config", 
                "name": "Demo Config",
                "description": "Server configuration",
                "mimeType": "application/json"
            }
        ]
        
        self.demo_tools = [
            {
                "name": "echo",
                "description": "Echo back the input text",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to echo"
                        }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "calculate", 
                "description": "Perform basic arithmetic calculations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]
    
    async def handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List demo resources"""
        return {"resources": self.demo_resources}
    
    async def handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read demo resource content"""
        uri = params.get("uri")
        
        if uri == "demo://readme":
            content = """# Demo MCP Server

This is a demonstration MCP server that showcases basic functionality:

- Resources: Access to demo documentation and configuration
- Tools: Echo and basic calculator functions
- Prompts: Sample prompt templates

## Usage

The server responds to standard MCP protocol messages and provides
mock implementations for testing and development purposes.
"""
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": content
                }]
            }
        
        elif uri == "demo://config":
            config = {
                "server": "demo-mcp-server",
                "version": "1.0.0",
                "features": ["resources", "tools", "prompts"],
                "debug": True
            }
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(config, indent=2)
                }]
            }
        
        else:
            raise MCPError(-32602, f"Resource not found: {uri}")
    
    async def handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List demo tools"""
        return {"tools": self.demo_tools}
    
    async def handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute demo tools"""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if name == "echo":
            text = arguments.get("text", "")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Echo: {text}"
                }]
            }
        
        elif name == "calculate":
            expression = arguments.get("expression", "")
            try:
                # Simple eval for demo - in production use safe math parser
                result = eval(expression)
                return {
                    "content": [{
                        "type": "text", 
                        "text": f"{expression} = {result}"
                    }]
                }
            except Exception as e:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Error calculating {expression}: {e}"
                    }],
                    "isError": True
                }
        
        else:
            raise MCPError(-32602, f"Tool not found: {name}")

async def main():
    """Main server entry point"""
    server = DemoMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### 4.2 Testing the Basic Server

```python
# ai/mcp/examples/04_test_basic_server.py

import asyncio
import json
from typing import Dict, Any

async def test_mcp_server():
    """Test basic MCP server functionality"""
    from basic_server import DemoMCPServer
    
    server = DemoMCPServer()
    
    print("Testing MCP Server")
    print("=" * 30)
    
    # Test initialize
    init_message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    response = await server.handle_message(init_message)
    print("Initialize Response:")
    print(json.dumps(response, indent=2))
    
    # Test initialized notification
    init_notify = {
        "jsonrpc": "2.0", 
        "method": "initialized",
        "params": {}
    }
    
    await server.handle_message(init_notify)
    print(f"\nServer initialized: {server.initialized}")
    
    # Test list resources
    list_resources = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "resources/list",
        "params": {}
    }
    
    response = await server.handle_message(list_resources)
    print("\nList Resources Response:")
    print(json.dumps(response, indent=2))
    
    # Test read resource
    read_resource = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "resources/read",
        "params": {
            "uri": "demo://readme"
        }
    }
    
    response = await server.handle_message(read_resource)
    print("\nRead Resource Response:")
    print(json.dumps(response, indent=2))
    
    # Test list tools
    list_tools = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/list",
        "params": {}
    }
    
    response = await server.handle_message(list_tools)
    print("\nList Tools Response:")
    print(json.dumps(response, indent=2))
    
    # Test call tool
    call_tool = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "echo",
            "arguments": {
                "text": "Hello, MCP!"
            }
        }
    }
    
    response = await server.handle_message(call_tool)
    print("\nCall Tool Response:")
    print(json.dumps(response, indent=2))
    
    # Test calculator tool
    calc_tool = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "calculate",
            "arguments": {
                "expression": "2 + 2 * 3"
            }
        }
    }
    
    response = await server.handle_message(calc_tool)
    print("\nCalculator Tool Response:")
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
```

---

## Chapter 5: Resources and Resource Management

Resources in MCP represent data that can be read by AI models. This chapter covers comprehensive resource management patterns.

### 5.1 Advanced Resource Types

```python
# ai/mcp/examples/05_advanced_resources.py

import asyncio
import aiofiles
import json
import sqlite3
import httpx
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
import mimetypes
import base64

@dataclass
class ResourceSubscription:
    """Resource subscription for change notifications"""
    uri: str
    subscriber_id: str

class AdvancedResourceManager:
    """Advanced resource management with multiple backends"""
    
    def __init__(self):
        self.subscriptions: Dict[str, List[ResourceSubscription]] = {}
        self.resource_handlers = {
            "file": self._handle_file_resource,
            "http": self._handle_http_resource,
            "https": self._handle_http_resource,
            "sqlite": self._handle_sqlite_resource,
            "memory": self._handle_memory_resource
        }
        self.memory_store: Dict[str, Any] = {}
        
    async def list_resources(self, cursor: Optional[str] = None) -> Dict[str, Any]:
        """List all available resources with pagination"""
        resources = []
        
        # File resources
        file_resources = await self._discover_file_resources()
        resources.extend(file_resources)
        
        # Memory resources
        memory_resources = self._list_memory_resources()
        resources.extend(memory_resources)
        
        # HTTP resources (predefined)
        http_resources = self._list_http_resources()
        resources.extend(http_resources)
        
        # Apply pagination
        start_idx = 0
        if cursor:
            try:
                start_idx = int(cursor)
            except ValueError:
                start_idx = 0
        
        page_size = 50
        paginated_resources = resources[start_idx:start_idx + page_size]
        
        result = {"resources": paginated_resources}
        
        if len(resources) > start_idx + page_size:
            result["nextCursor"] = str(start_idx + page_size)
        
        return result
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read resource content based on URI scheme"""
        parsed = urlparse(uri)
        scheme = parsed.scheme
        
        if scheme not in self.resource_handlers:
            raise ValueError(f"Unsupported URI scheme: {scheme}")
        
        return await self.resource_handlers[scheme](uri)
    
    async def _handle_file_resource(self, uri: str) -> Dict[str, Any]:
        """Handle file:// resources"""
        parsed = urlparse(uri)
        file_path = Path(parsed.path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "text/plain"
        
        content = {"uri": uri, "mimeType": mime_type}
        
        if mime_type.startswith("text/") or mime_type == "application/json":
            # Read as text
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                text_content = await f.read()
                content["text"] = text_content
        else:
            # Read as binary and encode as base64
            async with aiofiles.open(file_path, 'rb') as f:
                binary_content = await f.read()
                content["blob"] = base64.b64encode(binary_content).decode('utf-8')
        
        return {"contents": [content]}
    
    async def _handle_http_resource(self, uri: str) -> Dict[str, Any]:
        """Handle http:// and https:// resources"""
        async with httpx.AsyncClient() as client:
            response = await client.get(uri)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "text/plain")
            main_type = content_type.split(";")[0].strip()
            
            content = {
                "uri": uri,
                "mimeType": main_type
            }
            
            if main_type.startswith("text/") or main_type == "application/json":
                content["text"] = response.text
            else:
                content["blob"] = base64.b64encode(response.content).decode('utf-8')
            
            return {"contents": [content]}
    
    async def _handle_sqlite_resource(self, uri: str) -> Dict[str, Any]:
        """Handle sqlite:// resources"""
        # Parse sqlite://path/to/db.sqlite?query=SELECT * FROM table
        parsed = urlparse(uri)
        db_path = parsed.path
        query_params = dict(param.split('=', 1) for param in parsed.query.split('&') if '=' in param)
        
        sql_query = query_params.get('query', 'SELECT name FROM sqlite_master WHERE type="table"')
        
        def execute_query():
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        
        results = await asyncio.to_thread(execute_query)
        
        return {
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(results, indent=2)
            }]
        }
    
    async def _handle_memory_resource(self, uri: str) -> Dict[str, Any]:
        """Handle memory:// resources"""
        # memory://key
        parsed = urlparse(uri)
        key = parsed.path.lstrip('/')
        
        if key not in self.memory_store:
            raise KeyError(f"Memory resource not found: {key}")
        
        data = self.memory_store[key]
        
        return {
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(data, indent=2)
            }]
        }
    
    async def _discover_file_resources(self) -> List[Dict[str, Any]]:
        """Discover file resources in current directory"""
        resources = []
        current_dir = Path(".")
        
        for file_path in current_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                mime_type, _ = mimetypes.guess_type(str(file_path))
                mime_type = mime_type or "application/octet-stream"
                
                resources.append({
                    "uri": file_path.as_uri(),
                    "name": file_path.name,
                    "description": f"File: {file_path}",
                    "mimeType": mime_type
                })
        
        return resources
    
    def _list_memory_resources(self) -> List[Dict[str, Any]]:
        """List memory resources"""
        resources = []
        for key in self.memory_store:
            resources.append({
                "uri": f"memory://{key}",
                "name": f"Memory: {key}",
                "description": f"In-memory data for {key}",
                "mimeType": "application/json"
            })
        return resources
    
    def _list_http_resources(self) -> List[Dict[str, Any]]:
        """List predefined HTTP resources"""
        return [
            {
                "uri": "https://api.github.com/repos/microsoft/vscode/releases/latest",
                "name": "VS Code Latest Release",
                "description": "Latest VS Code release information",
                "mimeType": "application/json"
            },
            {
                "uri": "https://httpbin.org/json",
                "name": "HTTPBin JSON",
                "description": "Sample JSON data from HTTPBin",
                "mimeType": "application/json"
            }
        ]
    
    def set_memory_resource(self, key: str, data: Any):
        """Set memory resource"""
        self.memory_store[key] = data
        # Notify subscribers
        self._notify_resource_changed(f"memory://{key}")
    
    def subscribe_to_resource(self, uri: str, subscriber_id: str):
        """Subscribe to resource changes"""
        if uri not in self.subscriptions:
            self.subscriptions[uri] = []
        
        subscription = ResourceSubscription(uri, subscriber_id)
        self.subscriptions[uri].append(subscription)
    
    def unsubscribe_from_resource(self, uri: str, subscriber_id: str):
        """Unsubscribe from resource changes"""
        if uri in self.subscriptions:
            self.subscriptions[uri] = [
                sub for sub in self.subscriptions[uri] 
                if sub.subscriber_id != subscriber_id
            ]
    
    def _notify_resource_changed(self, uri: str):
        """Notify subscribers of resource changes"""
        if uri in self.subscriptions:
            for subscription in self.subscriptions[uri]:
                # In real implementation, send notification to subscriber
                print(f"Resource changed notification for {uri} to {subscription.subscriber_id}")

class ResourceMCPServer:
    """MCP Server focused on resource management"""
    
    def __init__(self):
        self.resource_manager = AdvancedResourceManager()
        
        # Setup some sample memory resources
        self.resource_manager.set_memory_resource("config", {
            "server_name": "Resource MCP Server",
            "version": "1.0.0",
            "supported_schemes": ["file", "http", "https", "sqlite", "memory"]
        })
        
        self.resource_manager.set_memory_resource("stats", {
            "resources_served": 0,
            "active_subscriptions": 0,
            "uptime_seconds": 0
        })
    
    async def handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list resources request"""
        cursor = params.get("cursor")
        return await self.resource_manager.list_resources(cursor)
    
    async def handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle read resource request"""
        uri = params.get("uri")
        if not uri:
            raise ValueError("Missing required parameter: uri")
        
        try:
            result = await self.resource_manager.read_resource(uri)
            
            # Update stats
            stats = self.resource_manager.memory_store.get("stats", {})
            stats["resources_served"] = stats.get("resources_served", 0) + 1
            self.resource_manager.set_memory_resource("stats", stats)
            
            return result
            
        except Exception as e:
            raise ValueError(f"Failed to read resource {uri}: {e}")
    
    async def handle_subscribe_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource subscription"""
        uri = params.get("uri")
        subscriber_id = params.get("subscriberId", "default")
        
        if not uri:
            raise ValueError("Missing required parameter: uri")
        
        self.resource_manager.subscribe_to_resource(uri, subscriber_id)
        
        # Update stats
        stats = self.resource_manager.memory_store.get("stats", {})
        stats["active_subscriptions"] = len(self.resource_manager.subscriptions)
        self.resource_manager.set_memory_resource("stats", stats)
        
        return {"subscribed": True}

async def demo_resource_management():
    """Demonstrate advanced resource management"""
    print("Advanced Resource Management Demo")
    print("=" * 40)
    
    server = ResourceMCPServer()
    
    # List resources
    print("1. Listing resources...")
    resources_result = await server.handle_list_resources({})
    resources = resources_result["resources"]
    print(f"Found {len(resources)} resources:")
    
    for resource in resources[:5]:  # Show first 5
        print(f"  - {resource['name']}: {resource['uri']}")
        print(f"    Type: {resource['mimeType']}")
    
    if len(resources) > 5:
        print(f"  ... and {len(resources) - 5} more")
    
    # Read memory resource
    print("\n2. Reading memory resource...")
    try:
        config_result = await server.handle_read_resource({"uri": "memory://config"})
        config_content = config_result["contents"][0]
        print(f"Config content type: {config_content['mimeType']}")
        print("Config data:")
        print(config_content["text"])
    except Exception as e:
        print(f"Error reading config: {e}")
    
    # Read HTTP resource
    print("\n3. Reading HTTP resource...")
    try:
        http_result = await server.handle_read_resource({
            "uri": "https://httpbin.org/json"
        })
        http_content = http_result["contents"][0]
        print(f"HTTP content type: {http_content['mimeType']}")
        print("HTTP response preview:")
        print(http_content["text"][:200] + "..." if len(http_content["text"]) > 200 else http_content["text"])
    except Exception as e:
        print(f"Error reading HTTP resource: {e}")
    
    # Subscribe to resource
    print("\n4. Subscribing to resource changes...")
    try:
        sub_result = await server.handle_subscribe_resource({
            "uri": "memory://stats",
            "subscriberId": "demo-client"
        })
        print(f"Subscription result: {sub_result}")
    except Exception as e:
        print(f"Error subscribing: {e}")
    
    # Update memory resource to trigger change notification
    print("\n5. Updating memory resource...")
    server.resource_manager.set_memory_resource("stats", {
        "resources_served": 10,
        "active_subscriptions": 1,
        "uptime_seconds": 300,
        "last_updated": "2024-01-01T00:05:00Z"
    })
    
    # Read updated stats
    print("\n6. Reading updated stats...")
    try:
        stats_result = await server.handle_read_resource({"uri": "memory://stats"})
        stats_content = stats_result["contents"][0]
        print("Updated stats:")
        print(stats_content["text"])
    except Exception as e:
        print(f"Error reading stats: {e}")

if __name__ == "__main__":
    asyncio.run(demo_resource_management())
```

### 5.2 Resource Caching and Performance

```python
# ai/mcp/examples/05_resource_caching.py

import asyncio
import time
import hashlib
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class CacheEntry:
    """Cache entry for resource content"""
    content: Dict[str, Any]
    timestamp: float
    etag: Optional[str] = None
    max_age: Optional[int] = None

class ResourceCache:
    """Intelligent resource caching system"""
    
    def __init__(self, max_size: int = 100, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_times: Dict[str, float] = {}
    
    def _compute_cache_key(self, uri: str, params: Dict[str, Any] = None) -> str:
        """Compute cache key for resource"""
        key_data = f"{uri}"
        if params:
            key_data += json.dumps(params, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, uri: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get resource from cache if valid"""
        cache_key = self._compute_cache_key(uri, params)
        
        if cache_key not in self.cache:
            return None
        
        entry = self.cache[cache_key]
        current_time = time.time()
        
        # Check TTL
        age = current_time - entry.timestamp
        max_age = entry.max_age or self.default_ttl
        
        if age > max_age:
            # Entry expired
            del self.cache[cache_key]
            if cache_key in self.access_times:
                del self.access_times[cache_key]
            return None
        
        # Update access time
        self.access_times[cache_key] = current_time
        return entry.content
    
    def put(self, uri: str, content: Dict[str, Any], 
            params: Dict[str, Any] = None, max_age: int = None):
        """Put resource in cache"""
        cache_key = self._compute_cache_key(uri, params)
        current_time = time.time()
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
            self._evict_lru()
        
        # Create cache entry
        entry = CacheEntry(
            content=content,
            timestamp=current_time,
            max_age=max_age
        )
        
        self.cache[cache_key] = entry
        self.access_times[cache_key] = current_time
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), 
                     key=lambda k: self.access_times[k])
        
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def invalidate(self, uri: str, params: Dict[str, Any] = None):
        """Invalidate cached resource"""
        cache_key = self._compute_cache_key(uri, params)
        
        if cache_key in self.cache:
            del self.cache[cache_key]
        if cache_key in self.access_times:
            del self.access_times[cache_key]
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        
        # Count expired entries
        expired_count = 0
        for entry in self.cache.values():
            age = current_time - entry.timestamp
            max_age = entry.max_age or self.default_ttl
            if age > max_age:
                expired_count += 1
        
        return {
            "total_entries": len(self.cache),
            "expired_entries": expired_count,
            "cache_hit_ratio": getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1),
            "memory_usage_mb": self._estimate_memory_usage() / (1024 * 1024)
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        total_size = 0
        for entry in self.cache.values():
            # Rough estimation
            content_str = json.dumps(entry.content)
            total_size += len(content_str.encode('utf-8'))
        return total_size

class CachedResourceManager(AdvancedResourceManager):
    """Resource manager with caching support"""
    
    def __init__(self, cache_size: int = 100, cache_ttl: int = 300):
        super().__init__()
        self.cache = ResourceCache(cache_size, cache_ttl)
        self._hit_count = 0
        self._miss_count = 0
    
    async def read_resource(self, uri: str, use_cache: bool = True) -> Dict[str, Any]:
        """Read resource with caching support"""
        # Check cache first
        if use_cache:
            cached_content = self.cache.get(uri)
            if cached_content is not None:
                self._hit_count += 1
                return cached_content
        
        # Cache miss - fetch from source
        self._miss_count += 1
        content = await super().read_resource(uri)
        
        # Cache the result
        if use_cache:
            # Determine cache TTL based on resource type
            ttl = self._get_cache_ttl(uri)
            self.cache.put(uri, content, max_age=ttl)
        
        return content
    
    def _get_cache_ttl(self, uri: str) -> int:
        """Get appropriate cache TTL for resource type"""
        if uri.startswith("file://"):
            # File resources - shorter TTL as they might change
            return 60  # 1 minute
        elif uri.startswith("http"):
            # HTTP resources - medium TTL
            return 300  # 5 minutes
        elif uri.startswith("memory://"):
            # Memory resources - very short TTL
            return 10  # 10 seconds
        elif uri.startswith("sqlite://"):
            # Database resources - short TTL
            return 30  # 30 seconds
        else:
            # Default TTL
            return 300
    
    def invalidate_resource_cache(self, uri: str):
        """Invalidate specific resource cache"""
        self.cache.invalidate(uri)
        # Also notify subscribers that resource changed
        self._notify_resource_changed(uri)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        cache_stats = self.cache.get_stats()
        cache_stats.update({
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_ratio": self._hit_count / max(self._hit_count + self._miss_count, 1)
        })
        return cache_stats

async def demo_resource_caching():
    """Demonstrate resource caching functionality"""
    print("Resource Caching Demo")
    print("=" * 30)
    
    # Create cached resource manager
    manager = CachedResourceManager(cache_size=50, cache_ttl=60)
    
    # Set up some memory resources for testing
    manager.set_memory_resource("test-data", {
        "items": [1, 2, 3, 4, 5],
        "timestamp": time.time()
    })
    
    # Test cache miss (first read)
    print("1. First read (cache miss)...")
    start_time = time.time()
    result1 = await manager.read_resource("memory://test-data")
    miss_time = time.time() - start_time
    print(f"   Time: {miss_time:.4f}s")
    print(f"   Content preview: {result1['contents'][0]['text'][:50]}...")
    
    # Test cache hit (second read)
    print("\n2. Second read (cache hit)...")
    start_time = time.time()
    result2 = await manager.read_resource("memory://test-data")
    hit_time = time.time() - start_time
    print(f"   Time: {hit_time:.4f}s")
    print(f"   Speedup: {miss_time/hit_time:.2f}x")
    
    # Show cache statistics
    print("\n3. Cache statistics...")
    stats = manager.get_cache_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # Test cache invalidation
    print("\n4. Testing cache invalidation...")
    manager.invalidate_resource_cache("memory://test-data")
    
    # Update the resource
    manager.set_memory_resource("test-data", {
        "items": [6, 7, 8, 9, 10],
        "timestamp": time.time(),
        "updated": True
    })
    
    # Read again (should be cache miss due to invalidation)
    print("\n5. Read after invalidation (cache miss)...")
    result3 = await manager.read_resource("memory://test-data")
    print(f"   Content preview: {result3['contents'][0]['text'][:100]}...")
    
    # Final cache statistics
    print("\n6. Final cache statistics...")
    stats = manager.get_cache_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(demo_resource_caching())
```

---

## Chapter 6: Tools and Tool Implementation

Tools in MCP allow AI models to execute functions and perform actions. This chapter covers comprehensive tool implementation patterns.

### 6.1 Advanced Tool Architecture

```python
# ai/mcp/examples/06_advanced_tools.py

import asyncio
import json
import subprocess
import os
import tempfile
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import inspect
import functools

class ToolInputType(Enum):
    """Tool input parameter types"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"

@dataclass
class ToolParameter:
    """Tool input parameter definition"""
    name: str
    type: ToolInputType
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[List[Any]] = None
    pattern: Optional[str] = None

@dataclass
class ToolDefinition:
    """Complete tool definition"""
    name: str
    description: str
    parameters: List[ToolParameter]
    category: str = "general"
    dangerous: bool = False
    requires_confirmation: bool = False

class BaseTool(ABC):
    """Base class for all MCP tools"""
    
    def __init__(self, name: str, description: str, category: str = "general"):
        self.name = name
        self.description = description
        self.category = category
        self.dangerous = False
        self.requires_confirmation = False
    
    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given arguments"""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters"""
        pass
    
    def get_definition(self) -> ToolDefinition:
        """Get tool definition"""
        schema = self.get_schema()
        parameters = []
        
        if "properties" in schema:
            required_fields = schema.get("required", [])
            for name, prop in schema["properties"].items():
                param_type = ToolInputType(prop.get("type", "string"))
                parameters.append(ToolParameter(
                    name=name,
                    type=param_type,
                    description=prop.get("description", ""),
                    required=name in required_fields,
                    default=prop.get("default"),
                    enum_values=prop.get("enum")
                ))
        
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=parameters,
            category=self.category,
            dangerous=self.dangerous,
            requires_confirmation=self.requires_confirmation
        )

class StringTool(BaseTool):
    """Tools for string manipulation"""
    
    def __init__(self):
        super().__init__(
            name="string_tools",
            description="Various string manipulation operations",
            category="text"
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute string operations"""
        operation = arguments.get("operation")
        text = arguments.get("text", "")
        
        if operation == "uppercase":
            result = text.upper()
        elif operation == "lowercase":
            result = text.lower()
        elif operation == "title":
            result = text.title()
        elif operation == "reverse":
            result = text[::-1]
        elif operation == "length":
            result = str(len(text))
        elif operation == "word_count":
            result = str(len(text.split()))
        elif operation == "remove_spaces":
            result = text.replace(" ", "")
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Unknown operation: {operation}"
                }],
                "isError": True
            }
        
        return {
            "content": [{
                "type": "text",
                "text": result
            }]
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema for string tools"""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "String operation to perform",
                    "enum": ["uppercase", "lowercase", "title", "reverse", "length", "word_count", "remove_spaces"]
                },
                "text": {
                    "type": "string",
                    "description": "Text to process"
                }
            },
            "required": ["operation", "text"]
        }

class FileSystemTool(BaseTool):
    """Tools for file system operations"""
    
    def __init__(self, allowed_paths: List[str] = None):
        super().__init__(
            name="filesystem",
            description="File system operations (read, write, list)",
            category="system"
        )
        self.allowed_paths = allowed_paths or [os.getcwd()]
        self.dangerous = True
        self.requires_confirmation = True
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file system operations"""
        operation = arguments.get("operation")
        path = arguments.get("path", "")
        
        # Security check
        if not self._is_path_allowed(path):
            return {
                "content": [{
                    "type": "text",
                    "text": f"Access denied to path: {path}"
                }],
                "isError": True
            }
        
        try:
            if operation == "list":
                items = os.listdir(path)
                result = "\n".join(items)
            elif operation == "read":
                with open(path, 'r', encoding='utf-8') as f:
                    result = f.read()
            elif operation == "write":
                content = arguments.get("content", "")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                result = f"Successfully wrote {len(content)} characters to {path}"
            elif operation == "exists":
                result = "Yes" if os.path.exists(path) else "No"
            elif operation == "size":
                if os.path.exists(path):
                    result = str(os.path.getsize(path))
                else:
                    result = "File does not exist"
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Unknown operation: {operation}"
                    }],
                    "isError": True
                }
            
            return {
                "content": [{
                    "type": "text",
                    "text": result
                }]
            }
            
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: {str(e)}"
                }],
                "isError": True
            }
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is in allowed paths"""
        abs_path = os.path.abspath(path)
        return any(abs_path.startswith(os.path.abspath(allowed)) 
                  for allowed in self.allowed_paths)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema for file system tools"""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "File system operation to perform",
                    "enum": ["list", "read", "write", "exists", "size"]
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write operation)"
                }
            },
            "required": ["operation", "path"]
        }

class CalculatorTool(BaseTool):
    """Advanced calculator tool"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations including advanced functions",
            category="math"
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute mathematical calculations"""
        expression = arguments.get("expression", "")
        mode = arguments.get("mode", "basic")
        
        try:
            if mode == "basic":
                # Safe evaluation for basic math
                result = self._safe_eval(expression)
            elif mode == "advanced":
                # Advanced math with numpy/scipy if available
                result = self._advanced_eval(expression)
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Unknown mode: {mode}"
                    }],
                    "isError": True
                }
            
            return {
                "content": [{
                    "type": "text",
                    "text": f"{expression} = {result}"
                }]
            }
            
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Calculation error: {str(e)}"
                }],
                "isError": True
            }
    
    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate basic mathematical expressions"""
        # Whitelist of allowed operations
        allowed_names = {
            k: v for k, v in __builtins__.items()
            if k in ["abs", "min", "max", "sum", "round", "pow"]
        }
        
        # Add math operations
        allowed_names.update({
            "__builtins__": {},
            "sin": lambda x: __import__("math").sin(x),
            "cos": lambda x: __import__("math").cos(x),
            "tan": lambda x: __import__("math").tan(x),
            "sqrt": lambda x: __import__("math").sqrt(x),
            "log": lambda x: __import__("math").log(x),
            "pi": __import__("math").pi,
            "e": __import__("math").e
        })
        
        return eval(expression, {"__builtins__": {}}, allowed_names)
    
    def _advanced_eval(self, expression: str) -> float:
        """Advanced mathematical evaluation"""
        try:
            import numpy as np
            import math
            
            # Create safe namespace with numpy and math functions
            safe_dict = {
                "__builtins__": {},
                "np": np,
                "math": math,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "sqrt": math.sqrt,
                "log": math.log,
                "exp": math.exp,
                "pi": math.pi,
                "e": math.e
            }
            
            return eval(expression, safe_dict)
        except ImportError:
            # Fallback to basic evaluation
            return self._safe_eval(expression)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema for calculator"""
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                },
                "mode": {
                    "type": "string",
                    "description": "Calculation mode",
                    "enum": ["basic", "advanced"],
                    "default": "basic"
                }
            },
            "required": ["expression"]
        }

class SystemCommandTool(BaseTool):
    """Execute system commands (dangerous tool)"""
    
    def __init__(self, allowed_commands: List[str] = None):
        super().__init__(
            name="system_command",
            description="Execute system commands",
            category="system"
        )
        self.allowed_commands = allowed_commands or ["ls", "pwd", "date", "whoami"]
        self.dangerous = True
        self.requires_confirmation = True
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute system command"""
        command = arguments.get("command", "")
        args = arguments.get("args", [])
        timeout = arguments.get("timeout", 30)
        
        # Security check
        if not self._is_command_allowed(command):
            return {
                "content": [{
                    "type": "text",
                    "text": f"Command not allowed: {command}"
                }],
                "isError": True
            }
        
        try:
            # Construct full command
            full_command = [command] + args
            
            # Execute with timeout
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            
            return {
                "content": [{
                    "type": "text",
                    "text": output
                }]
            }
            
        except subprocess.TimeoutExpired:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Command timed out after {timeout} seconds"
                }],
                "isError": True
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Command execution error: {str(e)}"
                }],
                "isError": True
            }
    
    def _is_command_allowed(self, command: str) -> bool:
        """Check if command is in allowed list"""
        return command in self.allowed_commands
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema for system command"""
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "System command to execute",
                    "enum": self.allowed_commands
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command arguments"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Command timeout in seconds",
                    "default": 30
                }
            },
            "required": ["command"]
        }

class ToolManager:
    """Manages all available tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_categories: Dict[str, List[str]] = {}
    
    def register_tool(self, tool: BaseTool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        
        # Update categories
        if tool.category not in self.tool_categories:
            self.tool_categories[tool.category] = []
        self.tool_categories[tool.category].append(tool.name)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self, category: str = None) -> List[Dict[str, Any]]:
        """List all tools or tools in specific category"""
        tools_list = []
        
        for tool in self.tools.values():
            if category is None or tool.category == category:
                schema = tool.get_schema()
                tool_def = {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": schema,
                    "category": tool.category,
                    "dangerous": tool.dangerous,
                    "requiresConfirmation": tool.requires_confirmation
                }
                tools_list.append(tool_def)
        
        return tools_list
    
    def get_categories(self) -> List[str]:
        """Get all tool categories"""
        return list(self.tool_categories.keys())
    
    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name"""
        tool = self.get_tool(name)
        if not tool:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Tool not found: {name}"
                }],
                "isError": True
            }
        
        return await tool.execute(arguments)

async def demo_advanced_tools():
    """Demonstrate advanced tool functionality"""
    print("Advanced Tools Demo")
    print("=" * 30)
    
    # Create tool manager
    manager = ToolManager()
    
    # Register tools
    manager.register_tool(StringTool())
    manager.register_tool(CalculatorTool())
    manager.register_tool(FileSystemTool(allowed_paths=[os.getcwd()]))
    manager.register_tool(SystemCommandTool())
    
    # List all tools
    print("1. Available tools:")
    tools = manager.list_tools()
    for tool in tools:
        status = "⚠️ DANGEROUS" if tool["dangerous"] else "✅ SAFE"
        print(f"   - {tool['name']}: {tool['description']} [{status}]")
        print(f"     Category: {tool['category']}")
    
    # Test string tool
    print("\n2. Testing string tool...")
    result = await manager.execute_tool("string_tools", {
        "operation": "uppercase",
        "text": "hello world"
    })
    print(f"   Result: {result['content'][0]['text']}")
    
    # Test calculator
    print("\n3. Testing calculator...")
    result = await manager.execute_tool("calculator", {
        "expression": "sqrt(16) + 2 * 3",
        "mode": "basic"
    })
    print(f"   Result: {result['content'][0]['text']}")
    
    # Test file system (list current directory)
    print("\n4. Testing file system tool...")
    result = await manager.execute_tool("filesystem", {
        "operation": "list",
        "path": "."
    })
    print(f"   Files in current directory:")
    for line in result['content'][0]['text'].split('\n')[:5]:  # Show first 5
        print(f"     {line}")
    
    # Test system command
    print("\n5. Testing system command...")
    result = await manager.execute_tool("system_command", {
        "command": "date"
    })
    print(f"   System date: {result['content'][0]['text'].strip()}")
    
    # Show tool categories
    print("\n6. Tool categories:")
    categories = manager.get_categories()
    for category in categories:
        tools_in_category = [t['name'] for t in manager.list_tools(category)]
        print(f"   {category}: {', '.join(tools_in_category)}")

if __name__ == "__main__":
    asyncio.run(demo_advanced_tools())
```

### 6.2 Tool Decorators and Auto-Registration

```python
# ai/mcp/examples/06_tool_decorators.py

import asyncio
import inspect
from typing import Dict, Any, List, Callable, get_type_hints, Union
from functools import wraps
from dataclasses import dataclass
import json

# Tool registry for auto-registration
TOOL_REGISTRY: Dict[str, Callable] = {}

def mcp_tool(name: str = None, description: str = "", category: str = "general", 
             dangerous: bool = False, requires_confirmation: bool = False):
    """Decorator to automatically register MCP tools"""
    
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        
        # Store tool metadata
        func._mcp_tool_name = tool_name
        func._mcp_tool_description = description or func.__doc__ or ""
        func._mcp_tool_category = category
        func._mcp_tool_dangerous = dangerous
        func._mcp_tool_requires_confirmation = requires_confirmation
        
        # Generate schema from function signature and type hints
        func._mcp_tool_schema = _generate_schema_from_function(func)
        
        # Register the tool
        TOOL_REGISTRY[tool_name] = func
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        # Copy metadata to wrapper
        wrapper._mcp_tool_name = tool_name
        wrapper._mcp_tool_description = func._mcp_tool_description
        wrapper._mcp_tool_category = category
        wrapper._mcp_tool_dangerous = dangerous
        wrapper._mcp_tool_requires_confirmation = requires_confirmation
        wrapper._mcp_tool_schema = func._mcp_tool_schema
        
        return wrapper
    
    return decorator

def _generate_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Generate JSON schema from function signature"""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        
        # Get type hint
        param_type = type_hints.get(param_name, str)
        
        # Convert Python type to JSON schema type
        schema_type, additional_props = _python_type_to_json_schema(param_type)
        
        prop = {
            "type": schema_type,
            **additional_props
        }
        
        # Add description from docstring if available
        if func.__doc__:
            # Simple docstring parsing - in practice you'd use a proper parser
            prop["description"] = f"Parameter {param_name}"
        
        # Check if parameter has default value
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
        else:
            prop["default"] = param.default
        
        properties[param_name] = prop
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }

def _python_type_to_json_schema(python_type) -> tuple[str, Dict[str, Any]]:
    """Convert Python type to JSON schema type"""
    additional_props = {}
    
    # Handle Union types (Optional)
    if hasattr(python_type, '__origin__') and python_type.__origin__ is Union:
        # For Optional types, use the non-None type
        non_none_types = [t for t in python_type.__args__ if t is not type(None)]
        if non_none_types:
            python_type = non_none_types[0]
    
    # Handle List types
    if hasattr(python_type, '__origin__') and python_type.__origin__ is list:
        item_type = python_type.__args__[0] if python_type.__args__ else str
        item_schema_type, _ = _python_type_to_json_schema(item_type)
        additional_props["items"] = {"type": item_schema_type}
        return "array", additional_props
    
    # Handle Dict types
    if hasattr(python_type, '__origin__') and python_type.__origin__ is dict:
        return "object", additional_props
    
    # Basic type mapping
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }
    
    return type_mapping.get(python_type, "string"), additional_props

# Example tools using decorators

@mcp_tool(
    name="text_analyzer",
    description="Analyze text for various metrics",
    category="text"
)
async def analyze_text(text: str, include_stats: bool = True, 
                      include_sentiment: bool = False) -> Dict[str, Any]:
    """Analyze text and return metrics"""
    
    # Basic text analysis
    words = text.split()
    sentences = text.split('.')
    
    result = {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "paragraph_count": len(text.split('\n\n')),
    }
    
    if include_stats:
        if words:
            avg_word_length = sum(len(word.strip('.,!?;:"()')) for word in words) / len(words)
            result["average_word_length"] = round(avg_word_length, 2)
        
        if sentences:
            avg_sentence_length = len(words) / len([s for s in sentences if s.strip()])
            result["average_sentence_length"] = round(avg_sentence_length, 2)
    
    if include_sentiment:
        # Simple sentiment analysis (in practice, use proper NLP library)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        result["sentiment"] = {
            "classification": sentiment,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count
        }
    
    return {
        "content": [{
            "type": "text",
            "text": json.dumps(result, indent=2)
        }]
    }

@mcp_tool(
    name="json_formatter",
    description="Format and validate JSON data",
    category="data"
)
def format_json(json_string: str, indent: int = 2, 
               sort_keys: bool = False) -> Dict[str, Any]:
    """Format JSON string with proper indentation"""
    
    try:
        # Parse JSON
        data = json.loads(json_string)
        
        # Format with specified options
        formatted = json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
        
        return {
            "content": [{
                "type": "text",
                "text": formatted
            }]
        }
        
    except json.JSONDecodeError as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Invalid JSON: {str(e)}"
            }],
            "isError": True
        }

@mcp_tool(
    name="base64_converter",
    description="Encode and decode base64 strings",
    category="encoding"
)
def base64_convert(text: str, operation: str = "encode") -> Dict[str, Any]:
    """Convert text to/from base64 encoding"""
    
    import base64
    
    try:
        if operation == "encode":
            encoded = base64.b64encode(text.encode('utf-8')).decode('utf-8')
            result = encoded
        elif operation == "decode":
            decoded = base64.b64decode(text.encode('utf-8')).decode('utf-8')
            result = decoded
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Unknown operation: {operation}. Use 'encode' or 'decode'"
                }],
                "isError": True
            }
        
        return {
            "content": [{
                "type": "text",
                "text": result
            }]
        }
        
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Conversion error: {str(e)}"
            }],
            "isError": True
        }

@mcp_tool(
    name="hash_generator",
    description="Generate various hash values for text",
    category="crypto",
    dangerous=False
)
def generate_hash(text: str, algorithm: str = "sha256") -> Dict[str, Any]:
    """Generate hash for given text"""
    
    import hashlib
    
    try:
        # Get hash function
        if algorithm == "md5":
            hasher = hashlib.md5()
        elif algorithm == "sha1":
            hasher = hashlib.sha1()
        elif algorithm == "sha256":
            hasher = hashlib.sha256()
        elif algorithm == "sha512":
            hasher = hashlib.sha512()
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Unsupported algorithm: {algorithm}"
                }],
                "isError": True
            }
        
        # Generate hash
        hasher.update(text.encode('utf-8'))
        hash_value = hasher.hexdigest()
        
        return {
            "content": [{
                "type": "text",
                "text": f"{algorithm.upper()}: {hash_value}"
            }]
        }
        
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Hash generation error: {str(e)}"
            }],
            "isError": True
        }

class DecoratorBasedToolManager:
    """Tool manager that uses decorator-registered tools"""
    
    def __init__(self):
        self.tools = TOOL_REGISTRY.copy()
    
    def list_tools(self, category: str = None) -> List[Dict[str, Any]]:
        """List all registered tools"""
        tools_list = []
        
        for tool_func in self.tools.values():
            if category is None or tool_func._mcp_tool_category == category:
                tool_def = {
                    "name": tool_func._mcp_tool_name,
                    "description": tool_func._mcp_tool_description,
                    "inputSchema": tool_func._mcp_tool_schema,
                    "category": tool_func._mcp_tool_category,
                    "dangerous": tool_func._mcp_tool_dangerous,
                    "requiresConfirmation": tool_func._mcp_tool_requires_confirmation
                }
                tools_list.append(tool_def)
        
        return tools_list
    
    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered tool"""
        if name not in self.tools:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Tool not found: {name}"
                }],
                "isError": True
            }
        
        tool_func = self.tools[name]
        
        try:
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**arguments)
            else:
                result = tool_func(**arguments)
            
            return result
            
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Tool execution error: {str(e)}"
                }],
                "isError": True
            }

async def demo_decorator_tools():
    """Demonstrate decorator-based tools"""
    print("Decorator-Based Tools Demo")
    print("=" * 35)
    
    manager = DecoratorBasedToolManager()
    
    # List all tools
    print("1. Auto-registered tools:")
    tools = manager.list_tools()
    for tool in tools:
        print(f"   - {tool['name']}: {tool['description']}")
        print(f"     Category: {tool['category']}")
        print(f"     Schema: {json.dumps(tool['inputSchema'], indent=6)}")
        print()
    
    # Test text analyzer
    print("2. Testing text analyzer...")
    result = await manager.execute_tool("text_analyzer", {
        "text": "This is a wonderful example of text analysis. It works great!",
        "include_stats": True,
        "include_sentiment": True
    })
    print("   Result:")
    print("   " + result['content'][0]['text'].replace('\n', '\n   '))
    
    # Test JSON formatter
    print("\n3. Testing JSON formatter...")
    messy_json = '{"name":"John","age":30,"city":"New York","hobbies":["reading","swimming"]}'
    result = await manager.execute_tool("json_formatter", {
        "json_string": messy_json,
        "indent": 2,
        "sort_keys": True
    })
    print("   Formatted JSON:")
    print("   " + result['content'][0]['text'].replace('\n', '\n   '))
    
    # Test base64 converter
    print("\n4. Testing base64 converter...")
    result = await manager.execute_tool("base64_converter", {
        "text": "Hello, MCP World!",
        "operation": "encode"
    })
    print(f"   Encoded: {result['content'][0]['text']}")
    
    # Test hash generator
    print("\n5. Testing hash generator...")
    result = await manager.execute_tool("hash_generator", {
        "text": "Hello, MCP World!",
        "algorithm": "sha256"
    })
    print(f"   Hash: {result['content'][0]['text']}")

if __name__ == "__main__":
    asyncio.run(demo_decorator_tools())
```

---

## Chapter 7: Prompts and Prompt Templates

MCP prompts provide dynamic template systems that can be customized with arguments and used by AI models.

### 7.1 Advanced Prompt System

```python
# ai/mcp/examples/07_advanced_prompts.py

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from string import Template
import re
from datetime import datetime

@dataclass
class PromptArgument:
    """Argument definition for prompts"""
    name: str
    description: str
    required: bool = True
    default: Any = None

@dataclass
class PromptMessage:
    """Individual message in a prompt"""
    role: str  # "user", "assistant", "system"
    content: Union[str, Dict[str, Any]]

@dataclass
class PromptTemplate:
    """Complete prompt template definition"""
    name: str
    description: str
    arguments: List[PromptArgument] = field(default_factory=list)
    messages: List[PromptMessage] = field(default_factory=list)
    category: str = "general"
    version: str = "1.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)

class BasePromptProcessor(ABC):
    """Base class for prompt processing"""
    
    @abstractmethod
    def process(self, template: str, arguments: Dict[str, Any]) -> str:
        """Process template with arguments"""
        pass

class SimpleTemplateProcessor(BasePromptProcessor):
    """Simple string template processor"""
    
    def process(self, template: str, arguments: Dict[str, Any]) -> str:
        """Process using Python string templates"""
        try:
            tmpl = Template(template)
            return tmpl.safe_substitute(**arguments)
        except Exception as e:
            raise ValueError(f"Template processing error: {e}")

class AdvancedTemplateProcessor(BasePromptProcessor):
    """Advanced template processor with conditionals and loops"""
    
    def process(self, template: str, arguments: Dict[str, Any]) -> str:
        """Process with advanced template features"""
        result = template
        
        # Process conditionals: {{if condition}}content{{endif}}
        result = self._process_conditionals(result, arguments)
        
        # Process loops: {{for item in items}}content{{endfor}}
        result = self._process_loops(result, arguments)
        
        # Process variables: {{variable}}
        result = self._process_variables(result, arguments)
        
        return result
    
    def _process_conditionals(self, template: str, arguments: Dict[str, Any]) -> str:
        """Process conditional blocks"""
        pattern = r'\{\{if\s+(\w+)\}\}(.*?)\{\{endif\}\}'
        
        def replace_conditional(match):
            condition = match.group(1)
            content = match.group(2)
            
            # Simple boolean check
            if arguments.get(condition, False):
                return content
            return ""
        
        return re.sub(pattern, replace_conditional, template, flags=re.DOTALL)
    
    def _process_loops(self, template: str, arguments: Dict[str, Any]) -> str:
        """Process loop blocks"""
        pattern = r'\{\{for\s+(\w+)\s+in\s+(\w+)\}\}(.*?)\{\{endfor\}\}'
        
        def replace_loop(match):
            item_var = match.group(1)
            list_var = match.group(2)
            content = match.group(3)
            
            items = arguments.get(list_var, [])
            if not isinstance(items, list):
                return ""
            
            result = ""
            for item in items:
                # Create new context with item variable
                item_content = content.replace(f"{{{{{item_var}}}}}", str(item))
                result += item_content
            
            return result
        
        return re.sub(pattern, replace_loop, template, flags=re.DOTALL)
    
    def _process_variables(self, template: str, arguments: Dict[str, Any]) -> str:
        """Process variable substitutions"""
        pattern = r'\{\{(\w+)\}\}'
        
        def replace_var(match):
            var_name = match.group(1)
            return str(arguments.get(var_name, f"{{{{{var_name}}}}}"))
        
        return re.sub(pattern, replace_var, template)

class PromptManager:
    """Manages prompt templates"""
    
    def __init__(self):
        self.prompts: Dict[str, PromptTemplate] = {}
        self.processor = AdvancedTemplateProcessor()
        self._load_default_prompts()
    
    def _load_default_prompts(self):
        """Load default prompt templates"""
        
        # Code review prompt
        code_review_prompt = PromptTemplate(
            name="code_review",
            description="Review code for quality, bugs, and improvements",
            arguments=[
                PromptArgument("code", "Code to review", required=True),
                PromptArgument("language", "Programming language", required=False, default="python"),
                PromptArgument("focus", "Review focus areas", required=False, default="general")
            ],
            messages=[
                PromptMessage(
                    role="system",
                    content="You are an expert code reviewer. Analyze the provided code for:\n- Bugs and potential issues\n- Code quality and best practices\n- Performance optimizations\n- Security concerns\n- Maintainability improvements"
                ),
                PromptMessage(
                    role="user",
                    content="""Please review this {{language}} code:

```{{language}}
{{code}}
```

{{if focus}}Focus areas: {{focus}}{{endif}}

Provide detailed feedback with specific suggestions for improvement."""
                )
            ],
            category="development",
            tags=["code", "review", "development"]
        )
        
        # Data analysis prompt
        data_analysis_prompt = PromptTemplate(
            name="data_analysis",
            description="Analyze data and provide insights",
            arguments=[
                PromptArgument("data", "Data to analyze", required=True),
                PromptArgument("questions", "Specific questions to answer", required=False),
                PromptArgument("format", "Data format", required=False, default="csv")
            ],
            messages=[
                PromptMessage(
                    role="system",
                    content="You are a data analyst. Analyze the provided data and extract meaningful insights, patterns, and trends."
                ),
                PromptMessage(
                    role="user",
                    content="""Please analyze this {{format}} data:

{{data}}

{{if questions}}Specific questions to address:
{{for question in questions}}
- {{question}}
{{endfor}}
{{endif}}

Provide a comprehensive analysis including:
1. Data overview and quality assessment
2. Key findings and patterns
3. Statistical insights
4. Recommendations based on the analysis"""
                )
            ],
            category="analysis",
            tags=["data", "analysis", "insights"]
        )
        
        # Writing assistant prompt
        writing_prompt = PromptTemplate(
            name="writing_assistant",
            description="Help with writing and editing tasks",
            arguments=[
                PromptArgument("task", "Writing task type", required=True),
                PromptArgument("content", "Content to work with", required=False),
                PromptArgument("style", "Writing style", required=False, default="professional"),
                PromptArgument("length", "Target length", required=False)
            ],
            messages=[
                PromptMessage(
                    role="system",
                    content="You are a professional writing assistant. Help with various writing tasks including creation, editing, and improvement of written content."
                ),
                PromptMessage(
                    role="user",
                    content="""Writing task: {{task}}

{{if content}}Original content:
{{content}}
{{endif}}

Style: {{style}}
{{if length}}Target length: {{length}}{{endif}}

Please help with this writing task, ensuring the result is clear, engaging, and appropriate for the intended audience."""
                )
            ],
            category="writing",
            tags=["writing", "editing", "communication"]
        )
        
        # Register prompts
        self.register_prompt(code_review_prompt)
        self.register_prompt(data_analysis_prompt)
        self.register_prompt(writing_prompt)
    
    def register_prompt(self, prompt: PromptTemplate):
        """Register a new prompt template"""
        self.prompts[prompt.name] = prompt
    
    def list_prompts(self, category: str = None) -> List[Dict[str, Any]]:
        """List all prompts or prompts in specific category"""
        prompts_list = []
        
        for prompt in self.prompts.values():
            if category is None or prompt.category == category:
                # Convert arguments to schema format
                arguments_schema = self._arguments_to_schema(prompt.arguments)
                
                prompt_def = {
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": arguments_schema,
                    "category": prompt.category,
                    "version": prompt.version,
                    "tags": prompt.tags
                }
                
                if prompt.author:
                    prompt_def["author"] = prompt.author
                
                prompts_list.append(prompt_def)
        
        return prompts_list
    
    def _arguments_to_schema(self, arguments: List[PromptArgument]) -> List[Dict[str, Any]]:
        """Convert arguments to schema format"""
        schema_args = []
        
        for arg in arguments:
            arg_def = {
                "name": arg.name,
                "description": arg.description,
                "required": arg.required
            }
            
            if arg.default is not None:
                arg_def["default"] = arg.default
            
            schema_args.append(arg_def)
        
        return schema_args
    
    def get_prompt(self, name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get processed prompt with arguments"""
        if name not in self.prompts:
            raise ValueError(f"Prompt not found: {name}")
        
        prompt = self.prompts[name]
        args = arguments or {}
        
        # Validate required arguments
        for arg in prompt.arguments:
            if arg.required and arg.name not in args:
                if arg.default is not None:
                    args[arg.name] = arg.default
                else:
                    raise ValueError(f"Required argument missing: {arg.name}")
        
        # Process messages
        processed_messages = []
        for message in prompt.messages:
            if isinstance(message.content, str):
                # Process template
                processed_content = self.processor.process(message.content, args)
            else:
                # Already structured content
                processed_content = message.content
            
            processed_messages.append({
                "role": message.role,
                "content": {
                    "type": "text",
                    "text": processed_content
                }
            })
        
        return {
            "description": prompt.description,
            "messages": processed_messages
        }

class PromptMCPServer:
    """MCP Server focused on prompt management"""
    
    def __init__(self):
        self.prompt_manager = PromptManager()
    
    async def handle_list_prompts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list prompts request"""
        cursor = params.get("cursor")
        category = params.get("category")
        
        prompts = self.prompt_manager.list_prompts(category)
        
        # Apply pagination
        start_idx = 0
        if cursor:
            try:
                start_idx = int(cursor)
            except ValueError:
                start_idx = 0
        
        page_size = 20
        paginated_prompts = prompts[start_idx:start_idx + page_size]
        
        result = {"prompts": paginated_prompts}
        
        if len(prompts) > start_idx + page_size:
            result["nextCursor"] = str(start_idx + page_size)
        
        return result
    
    async def handle_get_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get prompt request"""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not name:
            raise ValueError("Missing required parameter: name")
        
        try:
            return self.prompt_manager.get_prompt(name, arguments)
        except Exception as e:
            raise ValueError(f"Failed to get prompt {name}: {e}")

async def demo_advanced_prompts():
    """Demonstrate advanced prompt functionality"""
    print("Advanced Prompts Demo")
    print("=" * 30)
    
    server = PromptMCPServer()
    
    # List all prompts
    print("1. Available prompts:")
    prompts_result = await server.handle_list_prompts({})
    prompts = prompts_result["prompts"]
    
    for prompt in prompts:
        print(f"   - {prompt['name']}: {prompt['description']}")
        print(f"     Category: {prompt['category']}")
        print(f"     Arguments: {len(prompt['arguments'])}")
        print(f"     Tags: {', '.join(prompt['tags'])}")
        print()
    
    # Test code review prompt
    print("2. Testing code review prompt...")
    code_sample = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# This could be optimized
result = fibonacci(35)
print(result)
'''
    
    try:
        prompt_result = await server.handle_get_prompt({
            "name": "code_review",
            "arguments": {
                "code": code_sample,
                "language": "python",
                "focus": "performance optimization"
            }
        })
        
        print("   Generated prompt:")
        for i, message in enumerate(prompt_result["messages"]):
            role = message["role"]
            content = message["content"]["text"]
            print(f"   Message {i+1} ({role}):")
            print("   " + content.replace('\n', '\n   ')[:300] + "...")
            print()
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test data analysis prompt
    print("3. Testing data analysis prompt...")
    sample_data = """Name,Age,Department,Salary
John,30,Engineering,75000
Jane,25,Marketing,65000
Bob,35,Engineering,80000
Alice,28,Marketing,60000"""
    
    try:
        prompt_result = await server.handle_get_prompt({
            "name": "data_analysis",
            "arguments": {
                "data": sample_data,
                "format": "csv",
                "questions": ["What is the average salary by department?", "Are there any salary outliers?"]
            }
        })
        
        print("   Generated prompt:")
        user_message = prompt_result["messages"][1]["content"]["text"]
        print("   " + user_message.replace('\n', '\n   ')[:400] + "...")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test writing assistant
    print("\n4. Testing writing assistant prompt...")
    try:
        prompt_result = await server.handle_get_prompt({
            "name": "writing_assistant",
            "arguments": {
                "task": "email",
                "content": "Need to write professional email about project delay",
                "style": "diplomatic",
                "length": "brief"
            }
        })
        
        print("   Generated prompt:")
        user_message = prompt_result["messages"][1]["content"]["text"]
        print("   " + user_message.replace('\n', '\n   ')[:300] + "...")
        
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    asyncio.run(demo_advanced_prompts())
```

---

## Chapter 8: Advanced Server Patterns

### 8.1 Plugin Architecture

```python
# ai/mcp/examples/08_plugin_architecture.py

import asyncio
import importlib
import inspect
import os
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
from pathlib import Path
import json

class MCPPlugin(ABC):
    """Base class for MCP server plugins"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass
    
    @abstractmethod
    def get_resources(self) -> List[Dict[str, Any]]:
        """Get resources provided by this plugin"""
        pass
    
    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get tools provided by this plugin"""
        pass
    
    @abstractmethod
    def get_prompts(self) -> List[Dict[str, Any]]:
        """Get prompts provided by this plugin"""
        pass

class PluginManager:
    """Manages MCP server plugins"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.plugins: Dict[str, MCPPlugin] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        
    def load_plugin_config(self, config_path: str):
        """Load plugin configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.plugin_configs.update(config.get('plugins', {}))
    
    async def discover_plugins(self):
        """Discover and load plugins from plugins directory"""
        if not self.plugins_dir.exists():
            return
        
        for plugin_file in self.plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            
            try:
                await self._load_plugin_from_file(plugin_file)
            except Exception as e:
                print(f"Failed to load plugin {plugin_file}: {e}")
    
    async def _load_plugin_from_file(self, plugin_file: Path):
        """Load plugin from Python file"""
        module_name = plugin_file.stem
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find plugin classes
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, MCPPlugin) and 
                obj != MCPPlugin):
                
                plugin_instance = obj()
                config = self.plugin_configs.get(plugin_instance.name, {})
                
                if await plugin_instance.initialize(config):
                    self.plugins[plugin_instance.name] = plugin_instance
                    print(f"Loaded plugin: {plugin_instance.name} v{plugin_instance.version}")
    
    async def unload_plugin(self, name: str):
        """Unload a plugin"""
        if name in self.plugins:
            await self.plugins[name].cleanup()
            del self.plugins[name]
    
    def get_all_resources(self) -> List[Dict[str, Any]]:
        """Get resources from all plugins"""
        resources = []
        for plugin in self.plugins.values():
            plugin_resources = plugin.get_resources()
            for resource in plugin_resources:
                resource['plugin'] = plugin.name
            resources.extend(plugin_resources)
        return resources
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get tools from all plugins"""
        tools = []
        for plugin in self.plugins.values():
            plugin_tools = plugin.get_tools()
            for tool in plugin_tools:
                tool['plugin'] = plugin.name
            tools.extend(plugin_tools)
        return tools
    
    def get_all_prompts(self) -> List[Dict[str, Any]]:
        """Get prompts from all plugins"""
        prompts = []
        for plugin in self.plugins.values():
            plugin_prompts = plugin.get_prompts()
            for prompt in plugin_prompts:
                prompt['plugin'] = plugin.name
            prompts.extend(plugin_prompts)
        return prompts

# Example plugin implementation
class FileSystemPlugin(MCPPlugin):
    """Plugin for file system operations"""
    
    @property
    def name(self) -> str:
        return "filesystem"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize file system plugin"""
        self.allowed_paths = config.get('allowed_paths', [os.getcwd()])
        self.readonly = config.get('readonly', False)
        return True
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass
    
    def get_resources(self) -> List[Dict[str, Any]]:
        """Get file system resources"""
        resources = []
        for path in self.allowed_paths:
            path_obj = Path(path)
            if path_obj.exists():
                resources.append({
                    "uri": path_obj.as_uri(),
                    "name": f"Directory: {path_obj.name}",
                    "description": f"File system access to {path}",
                    "mimeType": "inode/directory"
                })
        return resources
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get file system tools"""
        tools = [
            {
                "name": "fs_list",
                "description": "List files and directories",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "fs_read",
                "description": "Read file content",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"]
                }
            }
        ]
        
        if not self.readonly:
            tools.append({
                "name": "fs_write",
                "description": "Write file content",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "File content"}
                    },
                    "required": ["path", "content"]
                }
            })
        
        return tools
    
    def get_prompts(self) -> List[Dict[str, Any]]:
        """Get file system prompts"""
        return [
            {
                "name": "file_analyzer",
                "description": "Analyze file structure and content",
                "arguments": [
                    {"name": "path", "description": "Path to analyze", "required": True}
                ]
            }
        ]

class WebPlugin(MCPPlugin):
    """Plugin for web operations"""
    
    @property
    def name(self) -> str:
        return "web"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize web plugin"""
        self.allowed_domains = config.get('allowed_domains', [])
        self.timeout = config.get('timeout', 30)
        return True
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass
    
    def get_resources(self) -> List[Dict[str, Any]]:
        """Get web resources"""
        resources = []
        for domain in self.allowed_domains:
            resources.append({
                "uri": f"https://{domain}",
                "name": f"Website: {domain}",
                "description": f"Web content from {domain}",
                "mimeType": "text/html"
            })
        return resources
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get web tools"""
        return [
            {
                "name": "web_fetch",
                "description": "Fetch content from URL",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "timeout": {"type": "integer", "description": "Request timeout"}
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "web_search",
                "description": "Search the web",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Number of results"}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    def get_prompts(self) -> List[Dict[str, Any]]:
        """Get web prompts"""
        return [
            {
                "name": "web_summarizer",
                "description": "Summarize web page content",
                "arguments": [
                    {"name": "url", "description": "URL to summarize", "required": True},
                    {"name": "focus", "description": "Focus areas", "required": False}
                ]
            }
        ]
```

### 8.2 Middleware System

```python
# ai/mcp/examples/08_middleware.py

import asyncio
import time
from typing import Dict, Any, List, Callable, Optional, Awaitable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import logging

@dataclass
class MCPRequest:
    """MCP request context"""
    method: str
    params: Dict[str, Any]
    message_id: Optional[str]
    timestamp: float
    client_info: Optional[Dict[str, Any]] = None

@dataclass
class MCPResponse:
    """MCP response context"""
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None

class MCPMiddleware(ABC):
    """Base class for MCP middleware"""
    
    @abstractmethod
    async def process_request(self, request: MCPRequest) -> Optional[MCPRequest]:
        """Process incoming request. Return None to block request."""
        pass
    
    @abstractmethod
    async def process_response(self, request: MCPRequest, 
                             response: MCPResponse) -> MCPResponse:
        """Process outgoing response"""
        pass

class LoggingMiddleware(MCPMiddleware):
    """Middleware for request/response logging"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    async def process_request(self, request: MCPRequest) -> Optional[MCPRequest]:
        """Log incoming request"""
        self.logger.info(
            f"Request: {request.method} "
            f"[ID: {request.message_id}] "
            f"[Params: {len(request.params)} items]"
        )
        return request
    
    async def process_response(self, request: MCPRequest, 
                             response: MCPResponse) -> MCPResponse:
        """Log outgoing response"""
        status = "SUCCESS" if response.result else "ERROR"
        exec_time = response.execution_time or 0
        
        self.logger.info(
            f"Response: {request.method} "
            f"[ID: {request.message_id}] "
            f"[Status: {status}] "
            f"[Time: {exec_time:.3f}s]"
        )
        
        if response.error:
            self.logger.error(f"Error: {response.error}")
        
        return response

class RateLimitingMiddleware(MCPMiddleware):
    """Middleware for rate limiting"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_counts: Dict[str, List[float]] = {}
    
    async def process_request(self, request: MCPRequest) -> Optional[MCPRequest]:
        """Check rate limits"""
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        # Clean old requests
        if client_id in self.request_counts:
            cutoff_time = current_time - self.window_seconds
            self.request_counts[client_id] = [
                t for t in self.request_counts[client_id] if t > cutoff_time
            ]
        else:
            self.request_counts[client_id] = []
        
        # Check rate limit
        if len(self.request_counts[client_id]) >= self.max_requests:
            raise ValueError(f"Rate limit exceeded for client {client_id}")
        
        # Record request
        self.request_counts[client_id].append(current_time)
        return request
    
    async def process_response(self, request: MCPRequest, 
                             response: MCPResponse) -> MCPResponse:
        """Pass through response"""
        return response
    
    def _get_client_id(self, request: MCPRequest) -> str:
        """Get client identifier"""
        if request.client_info:
            return request.client_info.get('name', 'unknown')
        return 'unknown'

class AuthenticationMiddleware(MCPMiddleware):
    """Middleware for authentication"""
    
    def __init__(self, api_keys: List[str]):
        self.api_keys = set(api_keys)
        self.public_methods = {'initialize', 'initialized'}
    
    async def process_request(self, request: MCPRequest) -> Optional[MCPRequest]:
        """Check authentication"""
        if request.method in self.public_methods:
            return request
        
        # Check for API key in params
        api_key = request.params.get('api_key')
        if not api_key or api_key not in self.api_keys:
            raise ValueError("Invalid or missing API key")
        
        # Remove API key from params to avoid passing to handlers
        request.params = {k: v for k, v in request.params.items() if k != 'api_key'}
        return request
    
    async def process_response(self, request: MCPRequest, 
                             response: MCPResponse) -> MCPResponse:
        """Pass through response"""
        return response

class CachingMiddleware(MCPMiddleware):
    """Middleware for response caching"""
    
    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, tuple[Dict[str, Any], float]] = {}
        self.cacheable_methods = {'resources/list', 'tools/list', 'prompts/list'}
    
    async def process_request(self, request: MCPRequest) -> Optional[MCPRequest]:
        """Check cache for response"""
        if request.method not in self.cacheable_methods:
            return request
        
        cache_key = self._get_cache_key(request)
        current_time = time.time()
        
        if cache_key in self.cache:
            cached_response, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                # Return cached response by modifying the request
                request.cached_response = cached_response
        
        return request
    
    async def process_response(self, request: MCPRequest, 
                             response: MCPResponse) -> MCPResponse:
        """Cache successful responses"""
        if (request.method in self.cacheable_methods and 
            response.result and not response.error):
            
            cache_key = self._get_cache_key(request)
            self.cache[cache_key] = (response.result, time.time())
        
        return response
    
    def _get_cache_key(self, request: MCPRequest) -> str:
        """Generate cache key"""
        params_str = json.dumps(request.params, sort_keys=True)
        return f"{request.method}:{hash(params_str)}"

class MetricsMiddleware(MCPMiddleware):
    """Middleware for collecting metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'method_counts': {},
            'error_counts': {}
        }
    
    async def process_request(self, request: MCPRequest) -> Optional[MCPRequest]:
        """Track request metrics"""
        self.metrics['total_requests'] += 1
        
        method = request.method
        if method not in self.metrics['method_counts']:
            self.metrics['method_counts'][method] = 0
        self.metrics['method_counts'][method] += 1
        
        return request
    
    async def process_response(self, request: MCPRequest, 
                             response: MCPResponse) -> MCPResponse:
        """Track response metrics"""
        if response.error:
            self.metrics['failed_requests'] += 1
            error_code = response.error.get('code', 'unknown')
            if error_code not in self.metrics['error_counts']:
                self.metrics['error_counts'][error_code] = 0
            self.metrics['error_counts'][error_code] += 1
        else:
            self.metrics['successful_requests'] += 1
        
        # Update average response time
        if response.execution_time:
            total_requests = self.metrics['total_requests']
            current_avg = self.metrics['average_response_time']
            new_avg = ((current_avg * (total_requests - 1)) + response.execution_time) / total_requests
            self.metrics['average_response_time'] = new_avg
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()

class MiddlewareStack:
    """Manages middleware stack for MCP server"""
    
    def __init__(self):
        self.middlewares: List[MCPMiddleware] = []
    
    def add_middleware(self, middleware: MCPMiddleware):
        """Add middleware to the stack"""
        self.middlewares.append(middleware)
    
    async def process_request(self, method: str, params: Dict[str, Any], 
                            message_id: str = None, 
                            client_info: Dict[str, Any] = None) -> MCPRequest:
        """Process request through middleware stack"""
        request = MCPRequest(
            method=method,
            params=params,
            message_id=message_id,
            timestamp=time.time(),
            client_info=client_info
        )
        
        for middleware in self.middlewares:
            request = await middleware.process_request(request)
            if request is None:
                raise ValueError("Request blocked by middleware")
        
        return request
    
    async def process_response(self, request: MCPRequest, 
                             result: Dict[str, Any] = None,
                             error: Dict[str, Any] = None) -> MCPResponse:
        """Process response through middleware stack"""
        execution_time = time.time() - request.timestamp
        
        response = MCPResponse(
            result=result,
            error=error,
            execution_time=execution_time
        )
        
        # Process through middleware in reverse order
        for middleware in reversed(self.middlewares):
            response = await middleware.process_response(request, response)
        
        return response

async def demo_middleware_system():
    """Demonstrate middleware system"""
    print("MCP Middleware System Demo")
    print("=" * 35)
    
    # Create middleware stack
    middleware_stack = MiddlewareStack()
    
    # Add middlewares
    middleware_stack.add_middleware(LoggingMiddleware())
    middleware_stack.add_middleware(RateLimitingMiddleware(max_requests=5, window_seconds=60))
    middleware_stack.add_middleware(AuthenticationMiddleware(['secret-key-123']))
    middleware_stack.add_middleware(CachingMiddleware(cache_ttl=30))
    metrics_middleware = MetricsMiddleware()
    middleware_stack.add_middleware(metrics_middleware)
    
    # Test requests
    test_requests = [
        ("initialize", {"protocolVersion": "2024-11-05"}, "1"),
        ("tools/list", {"api_key": "secret-key-123"}, "2"),
        ("tools/list", {"api_key": "secret-key-123"}, "3"),  # Should be cached
        ("resources/list", {"api_key": "secret-key-123"}, "4"),
        ("invalid/method", {"api_key": "wrong-key"}, "5"),  # Should fail auth
    ]
    
    for method, params, msg_id in test_requests:
        print(f"\nProcessing: {method}")
        try:
            # Process request
            request = await middleware_stack.process_request(
                method, params, msg_id, {"name": "test-client"}
            )
            
            # Simulate handler response
            if hasattr(request, 'cached_response'):
                result = request.cached_response
                print("  -> Using cached response")
            elif method == "initialize":
                result = {"protocolVersion": "2024-11-05", "capabilities": {}}
            elif method == "tools/list":
                result = {"tools": [{"name": "demo-tool", "description": "Demo"}]}
            elif method == "resources/list":
                result = {"resources": [{"uri": "demo://resource", "name": "Demo Resource"}]}
            else:
                result = None
            
            # Process response
            response = await middleware_stack.process_response(request, result=result)
            print(f"  -> Success: {response.execution_time:.3f}s")
            
        except Exception as e:
            # Process error response
            try:
                request = MCPRequest(method, params, msg_id, time.time())
                error = {"code": -32603, "message": str(e)}
                response = await middleware_stack.process_response(request, error=error)
                print(f"  -> Error: {e}")
            except:
                print(f"  -> Blocked: {e}")
    
    # Show metrics
    print(f"\nMetrics Summary:")
    metrics = metrics_middleware.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(demo_middleware_system())
```

---

## Chapter 9: Error Handling and Validation

### 9.1 Comprehensive Error System

```python
# ai/mcp/examples/09_error_handling.py

import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass
from enum import IntEnum
import traceback
import logging

class MCPErrorCode(IntEnum):
    """Standard MCP error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Custom error codes
    RESOURCE_NOT_FOUND = -32001
    TOOL_EXECUTION_ERROR = -32002
    PERMISSION_DENIED = -32003
    RATE_LIMITED = -32004
    VALIDATION_ERROR = -32005

@dataclass
class MCPError:
    """MCP error response"""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None

class MCPException(Exception):
    """Base MCP exception"""
    
    def __init__(self, code: MCPErrorCode, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to error response format"""
        error = {
            "code": self.code,
            "message": self.message
        }
        if self.data is not None:
            error["data"] = self.data
        return error

class ValidationException(MCPException):
    """Validation error"""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        data = {}
        if field:
            data["field"] = field
        if value is not None:
            data["value"] = value
        
        super().__init__(MCPErrorCode.VALIDATION_ERROR, message, data)

class ResourceNotFoundException(MCPException):
    """Resource not found error"""
    
    def __init__(self, uri: str):
        super().__init__(
            MCPErrorCode.RESOURCE_NOT_FOUND,
            f"Resource not found: {uri}",
            {"uri": uri}
        )

class ToolExecutionException(MCPException):
    """Tool execution error"""
    
    def __init__(self, tool_name: str, error_message: str, details: Dict[str, Any] = None):
        data = {"tool": tool_name, "details": details or {}}
        super().__init__(
            MCPErrorCode.TOOL_EXECUTION_ERROR,
            f"Tool execution failed: {error_message}",
            data
        )

class ParameterValidator:
    """Validates MCP request parameters"""
    
    @staticmethod
    def validate_required_params(params: Dict[str, Any], required: List[str]):
        """Validate required parameters are present"""
        missing = [param for param in required if param not in params]
        if missing:
            raise ValidationException(
                f"Missing required parameters: {', '.join(missing)}",
                field="required_params",
                value=missing
            )
    
    @staticmethod
    def validate_param_type(params: Dict[str, Any], param_name: str, 
                          expected_type: Type, required: bool = True):
        """Validate parameter type"""
        if param_name not in params:
            if required:
                raise ValidationException(
                    f"Missing required parameter: {param_name}",
                    field=param_name
                )
            return
        
        value = params[param_name]
        if not isinstance(value, expected_type):
            raise ValidationException(
                f"Parameter '{param_name}' must be of type {expected_type.__name__}",
                field=param_name,
                value=type(value).__name__
            )
    
    @staticmethod
    def validate_json_schema(params: Dict[str, Any], schema: Dict[str, Any]):
        """Validate parameters against JSON schema"""
        try:
            import jsonschema
            jsonschema.validate(params, schema)
        except ImportError:
            # Fallback to basic validation
            ParameterValidator._basic_schema_validation(params, schema)
        except Exception as e:
            raise ValidationException(f"Schema validation failed: {e}")
    
    @staticmethod
    def _basic_schema_validation(params: Dict[str, Any], schema: Dict[str, Any]):
        """Basic schema validation fallback"""
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            # Check required properties
            ParameterValidator.validate_required_params(params, required)
            
            # Check property types
            for prop_name, prop_schema in properties.items():
                if prop_name in params:
                    value = params[prop_name]
                    prop_type = prop_schema.get("type")
                    
                    if prop_type == "string" and not isinstance(value, str):
                        raise ValidationException(f"Property '{prop_name}' must be string")
                    elif prop_type == "integer" and not isinstance(value, int):
                        raise ValidationException(f"Property '{prop_name}' must be integer")
                    elif prop_type == "number" and not isinstance(value, (int, float)):
                        raise ValidationException(f"Property '{prop_name}' must be number")
                    elif prop_type == "boolean" and not isinstance(value, bool):
                        raise ValidationException(f"Property '{prop_name}' must be boolean")
                    elif prop_type == "array" and not isinstance(value, list):
                        raise ValidationException(f"Property '{prop_name}' must be array")

class ErrorHandler:
    """Handles and formats MCP errors"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
    
    def handle_exception(self, exc: Exception, request_id: str = None) -> Dict[str, Any]:
        """Handle exception and return error response"""
        if isinstance(exc, MCPException):
            error = exc.to_dict()
        else:
            # Map common exceptions to MCP errors
            error = self._map_exception_to_error(exc)
        
        # Log error
        self._log_error(exc, request_id)
        
        # Add debug information if enabled
        if self.debug:
            error["data"] = error.get("data", {})
            error["data"]["traceback"] = traceback.format_exc()
        
        response = {
            "jsonrpc": "2.0",
            "error": error
        }
        
        if request_id:
            response["id"] = request_id
        
        return response
    
    def _map_exception_to_error(self, exc: Exception) -> Dict[str, Any]:
        """Map standard exceptions to MCP errors"""
        if isinstance(exc, ValueError):
            return {
                "code": MCPErrorCode.INVALID_PARAMS,
                "message": str(exc)
            }
        elif isinstance(exc, FileNotFoundError):
            return {
                "code": MCPErrorCode.RESOURCE_NOT_FOUND,
                "message": str(exc)
            }
        elif isinstance(exc, PermissionError):
            return {
                "code": MCPErrorCode.PERMISSION_DENIED,
                "message": str(exc)
            }
        else:
            return {
                "code": MCPErrorCode.INTERNAL_ERROR,
                "message": "Internal server error"
            }
    
    def _log_error(self, exc: Exception, request_id: str = None):
        """Log error details"""
        error_msg = f"Error in request {request_id}: {exc}" if request_id else f"Error: {exc}"
        
        if isinstance(exc, MCPException):
            self.logger.error(error_msg)
        else:
            self.logger.exception(error_msg)

class RetryManager:
    """Manages retry logic for failed operations"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
    
    async def retry_async(self, operation, *args, **kwargs):
        """Retry async operation with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                # Calculate delay
                delay = self.base_delay * (self.backoff_factor ** attempt)
                await asyncio.sleep(delay)
        
        # Re-raise the last exception
        raise last_exception
    
    def retry_sync(self, operation, *args, **kwargs):
        """Retry sync operation with exponential backoff"""
        import time
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                # Calculate delay
                delay = self.base_delay * (self.backoff_factor ** attempt)
                time.sleep(delay)
        
        # Re-raise the last exception
        raise last_exception

class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
    
    async def call(self, operation, *args, **kwargs):
        """Execute operation with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
            else:
                raise MCPException(
                    MCPErrorCode.INTERNAL_ERROR,
                    "Circuit breaker is open"
                )
        
        try:
            result = await operation(*args, **kwargs)
            
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e

async def demo_error_handling():
    """Demonstrate error handling system"""
    print("MCP Error Handling Demo")
    print("=" * 30)
    
    error_handler = ErrorHandler(debug=True)
    validator = ParameterValidator()
    retry_manager = RetryManager(max_retries=2)
    
    # Test validation errors
    print("1. Testing parameter validation...")
    try:
        params = {"name": "test"}
        validator.validate_required_params(params, ["name", "description"])
    except ValidationException as e:
        error_response = error_handler.handle_exception(e, "req-001")
        print(f"   Validation Error: {json.dumps(error_response, indent=2)}")
    
    # Test type validation
    print("\n2. Testing type validation...")
    try:
        params = {"count": "not a number"}
        validator.validate_param_type(params, "count", int)
    except ValidationException as e:
        error_response = error_handler.handle_exception(e, "req-002")
        print(f"   Type Error: {json.dumps(error_response, indent=2)}")
    
    # Test schema validation
    print("\n3. Testing schema validation...")
    try:
        params = {"name": 123}  # Should be string
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        validator.validate_json_schema(params, schema)
    except ValidationException as e:
        error_response = error_handler.handle_exception(e, "req-003")
        print(f"   Schema Error: {json.dumps(error_response, indent=2)}")
    
    # Test custom exceptions
    print("\n4. Testing custom exceptions...")
    try:
        raise ResourceNotFoundException("demo://missing-resource")
    except MCPException as e:
        error_response = error_handler.handle_exception(e, "req-004")
        print(f"   Custom Error: {json.dumps(error_response, indent=2)}")
    
    # Test retry mechanism
    print("\n5. Testing retry mechanism...")
    
    attempt_count = 0
    async def failing_operation():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError(f"Attempt {attempt_count} failed")
        return f"Success on attempt {attempt_count}"
    
    try:
        result = await retry_manager.retry_async(failing_operation)
        print(f"   Retry Result: {result}")
    except Exception as e:
        print(f"   Retry Failed: {e}")
    
    # Test circuit breaker
    print("\n6. Testing circuit breaker...")
    circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)
    
    async def unreliable_service():
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise ValueError("Service unavailable")
        return "Service response"
    
    for i in range(5):
        try:
            result = await circuit_breaker.call(unreliable_service)
            print(f"   Call {i+1}: {result}")
        except Exception as e:
            print(f"   Call {i+1}: Failed - {e}")
        
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(demo_error_handling())
```

This comprehensive guide continues with detailed implementations of MCP server components, covering everything from basic concepts to advanced production patterns, making it easy for developers to build robust MCP servers.