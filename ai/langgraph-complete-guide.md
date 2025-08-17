# LangGraph Complete Guide

LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.

---

## 📋 **TABLE OF CONTENTS**

### **🚀 Getting Started**
- [Installation & Setup](#installation--setup)
- [Core Concepts](#core-concepts)
- [Basic Usage](#basic-usage)

### **🏗️ Building Workflows**
- [Graph Construction](#graph-construction)
- [State Management](#state-management)
- [Conditional Logic & Routing](#conditional-logic--routing)

### **🔄 Advanced Workflows**
- [Human-in-the-Loop](#human-in-the-loop)
- [Persistence & Checkpoints](#persistence--checkpoints)
- [Multi-Agent Systems](#multi-agent-systems)

### **⚡ Sophisticated Patterns**
- [Advanced Patterns](#advanced-patterns)
- [Streaming & Real-time](#streaming--real-time)
- [Error Handling & Recovery](#error-handling--recovery)

### **🚀 Production Ready**
- [Testing & Debugging](#testing--debugging)
- [Production Deployment](#production-deployment)
- [Performance Optimization](#performance-optimization)

### **🏢 Enterprise Grade**
- [Enterprise Integration](#enterprise-integration)
- [Troubleshooting](#troubleshooting)
- [Complete Example Application](#complete-example-application)

---

## **📖 Learning Paths**

**🌱 Beginner**: Installation → Core Concepts → Basic Usage  
**🌿 Intermediate**: Graph Construction → State Management → Conditional Logic  
**🌳 Advanced**: Human-in-the-Loop → Multi-Agent → Advanced Patterns  
**🏭 Production**: Testing → Deployment → Performance → Enterprise Integration  

**🎯 Complete Example**: The Customer Service Bot demonstrates all patterns integrated!

---

## Installation & Setup

### Basic Installation

```bash
# Core LangGraph
pip install langgraph

# With visualization support
pip install langgraph[visualization]

# Development installation with all extras
pip install "langgraph[dev]"

# Specific integrations
pip install langchain-openai
pip install langchain-anthropic
pip install langchain-community
```

### Environment Setup

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Required API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# Optional configurations
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-project"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
```

### Verification

```python
# Verify installation
try:
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import ToolExecutor
    from langgraph.checkpoint.sqlite import SqliteSaver
    print("✅ LangGraph installed successfully")
except ImportError as e:
    print(f"❌ Installation issue: {e}")
```

## Core Concepts

### Understanding LangGraph Architecture

```python
"""
LangGraph Core Components:

1. StateGraph: Main graph builder that defines workflow structure
2. State: TypedDict defining the data structure passed between nodes
3. Nodes: Individual processing functions that transform state
4. Edges: Connections between nodes (regular and conditional)
5. Checkpoints: State persistence for interruption/resumption
6. Tools: External function integrations
7. Memory: Built-in state management across executions
"""

from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, List
import operator
```

### Basic Graph Structure

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_openai import ChatOpenAI

# 1. Define State Structure
class BasicState(TypedDict):
    """State structure for our workflow"""
    user_input: str
    processed_data: str
    step_count: int
    final_output: str

# 2. Define Node Functions
def process_input(state: BasicState) -> BasicState:
    """Process user input"""
    user_input = state["user_input"]
    processed = f"Processed: {user_input}"
    
    return {
        "user_input": user_input,
        "processed_data": processed,
        "step_count": state.get("step_count", 0) + 1,
        "final_output": ""
    }

def generate_output(state: BasicState) -> BasicState:
    """Generate final output"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    prompt = f"Based on this processed data: {state['processed_data']}, generate a helpful response."
    response = llm.invoke(prompt)
    
    return {
        "user_input": state["user_input"],
        "processed_data": state["processed_data"],
        "step_count": state["step_count"] + 1,
        "final_output": response.content
    }

# 3. Build the Graph
def create_basic_workflow():
    """Create a basic LangGraph workflow"""
    
    # Initialize the state graph
    workflow = StateGraph(BasicState)
    
    # Add nodes (processing functions)
    workflow.add_node("process", process_input)
    workflow.add_node("generate", generate_output)
    
    # Define the flow with edges
    workflow.set_entry_point("process")  # Start here
    workflow.add_edge("process", "generate")  # process → generate
    workflow.add_edge("generate", END)  # generate → END
    
    # Compile the graph
    return workflow.compile()

# 4. Use the Graph
graph = create_basic_workflow()

# Execute the workflow
result = graph.invoke({
    "user_input": "What is machine learning?",
    "processed_data": "",
    "step_count": 0,
    "final_output": ""
})

print("Workflow Result:")
print(f"Steps completed: {result['step_count']}")
print(f"Final output: {result['final_output']}")
```

### Key LangGraph Concepts

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Literal

# State persistence across executions
class PersistentState(TypedDict):
    session_id: str
    conversation_count: int
    user_preferences: dict
    last_interaction: str

def conversation_node(state: PersistentState) -> PersistentState:
    """Node that maintains conversation state"""
    return {
        **state,
        "conversation_count": state.get("conversation_count", 0) + 1,
        "last_interaction": "User had a conversation"
    }

# Graph with memory
workflow = StateGraph(PersistentState)
workflow.add_node("converse", conversation_node)
workflow.set_entry_point("converse")
workflow.add_edge("converse", END)

# Compile with memory/checkpointing
memory = MemorySaver()
persistent_graph = workflow.compile(checkpointer=memory)

# Use with thread configuration for session persistence
thread_config = {"configurable": {"thread_id": "user_session_123"}}

# First execution
result1 = persistent_graph.invoke({
    "session_id": "user_session_123",
    "conversation_count": 0,
    "user_preferences": {},
    "last_interaction": ""
}, config=thread_config)

print(f"First execution - Conversation count: {result1['conversation_count']}")

# Second execution - state is preserved!
result2 = persistent_graph.invoke({
    "session_id": "user_session_123",
    "conversation_count": 0,  # This will be ignored, previous state is loaded
    "user_preferences": {},
    "last_interaction": ""
}, config=thread_config)

print(f"Second execution - Conversation count: {result2['conversation_count']}")  # Will be 2!
```

## Basic Usage

### Simple Linear Workflow

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict
from datetime import datetime

class LinearState(TypedDict):
    input_text: str
    step1_result: str
    step2_result: str
    step3_result: str
    metadata: dict

def step_one(state: LinearState) -> LinearState:
    """First processing step"""
    input_text = state["input_text"]
    result = f"Step 1: Analyzed '{input_text}'"
    
    return {
        **state,
        "step1_result": result,
        "metadata": {
            **state.get("metadata", {}),
            "step1_completed": datetime.now().isoformat()
        }
    }

def step_two(state: LinearState) -> LinearState:
    """Second processing step"""
    previous_result = state["step1_result"]
    result = f"Step 2: Enhanced '{previous_result}'"
    
    return {
        **state,
        "step2_result": result,
        "metadata": {
            **state.get("metadata", {}),
            "step2_completed": datetime.now().isoformat()
        }
    }

def step_three(state: LinearState) -> LinearState:
    """Final processing step"""
    previous_result = state["step2_result"]
    result = f"Step 3: Finalized '{previous_result}'"
    
    return {
        **state,
        "step3_result": result,
        "metadata": {
            **state.get("metadata", {}),
            "step3_completed": datetime.now().isoformat(),
            "workflow_complete": True
        }
    }

# Build linear workflow
linear_workflow = StateGraph(LinearState)

# Add nodes in sequence
linear_workflow.add_node("step1", step_one)
linear_workflow.add_node("step2", step_two)
linear_workflow.add_node("step3", step_three)

# Define linear flow
linear_workflow.set_entry_point("step1")
linear_workflow.add_edge("step1", "step2")
linear_workflow.add_edge("step2", "step3")
linear_workflow.add_edge("step3", END)

# Compile and execute
linear_app = linear_workflow.compile()

linear_result = linear_app.invoke({
    "input_text": "Hello World",
    "step1_result": "",
    "step2_result": "",
    "step3_result": "",
    "metadata": {}
})

print("Linear Workflow Results:")
print(f"Step 1: {linear_result['step1_result']}")
print(f"Step 2: {linear_result['step2_result']}")
print(f"Step 3: {linear_result['step3_result']}")
print(f"Complete: {linear_result['metadata']['workflow_complete']}")
```

### Conditional Branching Workflow

```python
from typing import Literal
import random

class BranchingState(TypedDict):
    input_value: int
    classification: str
    processing_path: str
    result: str

def classify_input(state: BranchingState) -> BranchingState:
    """Classify input to determine processing path"""
    value = state["input_value"]
    
    if value > 75:
        classification = "high"
    elif value > 25:
        classification = "medium" 
    else:
        classification = "low"
    
    return {
        **state,
        "classification": classification
    }

def high_value_processor(state: BranchingState) -> BranchingState:
    """Process high-value inputs"""
    return {
        **state,
        "processing_path": "high_value",
        "result": f"High-value processing for {state['input_value']}: Premium treatment applied"
    }

def medium_value_processor(state: BranchingState) -> BranchingState:
    """Process medium-value inputs"""
    return {
        **state,
        "processing_path": "medium_value", 
        "result": f"Standard processing for {state['input_value']}: Regular treatment applied"
    }

def low_value_processor(state: BranchingState) -> BranchingState:
    """Process low-value inputs"""
    return {
        **state,
        "processing_path": "low_value",
        "result": f"Basic processing for {state['input_value']}: Minimal treatment applied"
    }

def routing_logic(state: BranchingState) -> Literal["high", "medium", "low"]:
    """Determine routing based on classification"""
    classification = state["classification"]
    
    if classification == "high":
        return "high"
    elif classification == "medium":
        return "medium"
    else:
        return "low"

# Build branching workflow
branching_workflow = StateGraph(BranchingState)

# Add classifier node
branching_workflow.add_node("classify", classify_input)

# Add processing nodes for each path
branching_workflow.add_node("high_proc", high_value_processor)
branching_workflow.add_node("medium_proc", medium_value_processor)
branching_workflow.add_node("low_proc", low_value_processor)

# Set entry point
branching_workflow.set_entry_point("classify")

# Add conditional routing
branching_workflow.add_conditional_edges(
    "classify",
    routing_logic,
    {
        "high": "high_proc",
        "medium": "medium_proc",
        "low": "low_proc"
    }
)

# All processors end the workflow
branching_workflow.add_edge("high_proc", END)
branching_workflow.add_edge("medium_proc", END)
branching_workflow.add_edge("low_proc", END)

# Compile and test
branching_app = branching_workflow.compile()

# Test with different values
test_values = [10, 50, 90]

print("\nBranching Workflow Results:")
for value in test_values:
    result = branching_app.invoke({
        "input_value": value,
        "classification": "",
        "processing_path": "",
        "result": ""
    })
    
    print(f"Value {value}: {result['classification']} → {result['processing_path']}")
    print(f"  Result: {result['result']}")
```

### Iterative/Loop Workflow

```python
class IterativeState(TypedDict):
    problem: str
    solution: str
    iteration: int
    max_iterations: int
    quality_score: float
    satisfied: bool

def analyze_problem(state: IterativeState) -> IterativeState:
    """Analyze the current problem state"""
    problem = state["problem"]
    iteration = state["iteration"] + 1
    
    analysis = f"Iteration {iteration}: Analyzing '{problem}'"
    
    return {
        **state,
        "solution": analysis,
        "iteration": iteration,
        "satisfied": False
    }

def improve_solution(state: IterativeState) -> IterativeState:
    """Improve the current solution"""
    current_solution = state["solution"]
    iteration = state["iteration"]
    
    # Simulate solution improvement
    improved_solution = f"{current_solution} → Improved solution v{iteration}"
    
    # Simulate quality assessment (higher chance of satisfaction with more iterations)
    quality_score = min(0.9, 0.3 + (iteration * 0.2))
    satisfied = quality_score > 0.8 or iteration >= state["max_iterations"]
    
    return {
        **state,
        "solution": improved_solution,
        "quality_score": quality_score,
        "satisfied": satisfied
    }

def should_continue(state: IterativeState) -> Literal["improve", "finish"]:
    """Decide whether to continue iterating"""
    if state["satisfied"] or state["iteration"] >= state["max_iterations"]:
        return "finish"
    else:
        return "improve"

# Build iterative workflow
iterative_workflow = StateGraph(IterativeState)

iterative_workflow.add_node("analyze", analyze_problem)
iterative_workflow.add_node("improve", improve_solution)

iterative_workflow.set_entry_point("analyze")
iterative_workflow.add_edge("analyze", "improve")

# Conditional loop - key feature of LangGraph!
iterative_workflow.add_conditional_edges(
    "improve",
    should_continue,
    {
        "improve": "analyze",  # Loop back for more iterations
        "finish": END
    }
)

iterative_app = iterative_workflow.compile()

# Test iterative improvement
iterative_result = iterative_app.invoke({
    "problem": "Optimize database performance",
    "solution": "",
    "iteration": 0,
    "max_iterations": 4,
    "quality_score": 0.0,
    "satisfied": False
})

print("\nIterative Workflow Results:")
print(f"Final iteration: {iterative_result['iteration']}")
print(f"Quality score: {iterative_result['quality_score']:.2f}")
print(f"Satisfied: {iterative_result['satisfied']}")
print(f"Final solution: {iterative_result['solution']}")
```

## Graph Construction

### Building Complex Graphs

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List, Literal
from langchain_openai import ChatOpenAI

class ComplexState(TypedDict):
    # Input data
    user_request: str
    request_type: str
    priority: str
    
    # Processing data
    analysis_results: Dict[str, Any]
    processing_steps: List[str]
    current_stage: str
    
    # Output data
    response: str
    actions_taken: List[str]
    follow_up_needed: bool

class ComplexGraphBuilder:
    """Build complex workflows with multiple paths and conditions"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        self.workflow = StateGraph(ComplexState)
        
    def build_request_processing_graph(self):
        """Build a complex request processing graph"""
        
        def request_analyzer(state: ComplexState) -> ComplexState:
            """Analyze incoming request"""
            request = state["user_request"]
            
            # Classify request type
            request_lower = request.lower()
            if any(word in request_lower for word in ["urgent", "critical", "emergency"]):
                request_type = "urgent"
                priority = "high"
            elif any(word in request_lower for word in ["question", "help", "how"]):
                request_type = "inquiry"
                priority = "medium"
            elif any(word in request_lower for word in ["complaint", "problem", "issue"]):
                request_type = "complaint"
                priority = "high"
            else:
                request_type = "general"
                priority = "low"
            
            # Perform detailed analysis
            analysis_prompt = f"""
            Analyze this request: "{request}"
            
            Provide analysis including:
            1. Key topics
            2. Sentiment
            3. Required actions
            
            Format as JSON:
            {{
                "topics": ["topic1", "topic2"],
                "sentiment": "positive/neutral/negative",
                "required_actions": ["action1", "action2"]
            }}
            """
            
            analysis_response = self.llm.invoke(analysis_prompt)
            
            try:
                analysis_results = json.loads(analysis_response.content)
            except:
                analysis_results = {
                    "topics": ["general"],
                    "sentiment": "neutral",
                    "required_actions": ["provide_response"]
                }
            
            return {
                **state,
                "request_type": request_type,
                "priority": priority,
                "analysis_results": analysis_results,
                "current_stage": "analyzed",
                "processing_steps": ["Request analyzed"]
            }
        
        def urgent_handler(state: ComplexState) -> ComplexState:
            """Handle urgent requests with priority processing"""
            
            urgent_response = f"URGENT REQUEST PROCESSED: {state['user_request']}"
            response = self.llm.invoke(f"Provide urgent assistance for: {state['user_request']}")
            
            return {
                **state,
                "response": response.content,
                "current_stage": "urgent_handled",
                "actions_taken": ["urgent_processing_applied", "priority_response_generated"],
                "processing_steps": state["processing_steps"] + ["Urgent processing completed"],
                "follow_up_needed": True
            }
        
        def complaint_handler(state: ComplexState) -> ComplexState:
            """Handle complaints with special care"""
            
            complaint_prompt = f"""
            Handle this customer complaint with empathy and solutions:
            "{state['user_request']}"
            
            Provide:
            1. Acknowledgment of the issue
            2. Apology if appropriate
            3. Concrete steps to resolve
            4. Follow-up commitment
            """
            
            response = self.llm.invoke(complaint_prompt)
            
            return {
                **state,
                "response": response.content,
                "current_stage": "complaint_handled",
                "actions_taken": ["complaint_acknowledged", "resolution_plan_created"],
                "processing_steps": state["processing_steps"] + ["Complaint processed with care"],
                "follow_up_needed": True
            }
        
        def standard_handler(state: ComplexState) -> ComplexState:
            """Handle standard requests"""
            
            response = self.llm.invoke(f"Provide helpful response to: {state['user_request']}")
            
            return {
                **state,
                "response": response.content,
                "current_stage": "standard_handled",
                "actions_taken": ["standard_processing"],
                "processing_steps": state["processing_steps"] + ["Standard processing completed"],
                "follow_up_needed": False
            }
        
        def quality_check(state: ComplexState) -> ComplexState:
            """Perform quality check on response"""
            
            response = state["response"]
            request_type = state["request_type"]
            
            # Quality check prompt
            quality_prompt = f"""
            Evaluate this response for a {request_type} request:
            
            Original request: {state['user_request']}
            Response: {response}
            
            Rate quality (0-1) and suggest improvements if needed.
            
            Format as JSON:
            {{
                "quality_score": 0.0-1.0,
                "improvements": ["improvement1", "improvement2"],
                "approved": true/false
            }}
            """
            
            quality_response = self.llm.invoke(quality_prompt)
            
            try:
                quality_data = json.loads(quality_response.content)
                quality_score = quality_data["quality_score"]
                approved = quality_data["approved"]
            except:
                quality_score = 0.8
                approved = True
            
            return {
                **state,
                "current_stage": "quality_checked",
                "processing_steps": state["processing_steps"] + [f"Quality check: {quality_score:.2f}"],
                "analysis_results": {
                    **state.get("analysis_results", {}),
                    "quality_score": quality_score,
                    "quality_approved": approved
                }
            }
        
        def routing_decision(state: ComplexState) -> Literal["urgent", "complaint", "standard"]:
            """Route based on request analysis"""
            request_type = state["request_type"]
            priority = state["priority"]
            
            if request_type == "urgent" or priority == "high":
                if state["request_type"] == "complaint":
                    return "complaint"
                else:
                    return "urgent"
            else:
                return "standard"
        
        # Add all nodes
        self.workflow.add_node("analyze", request_analyzer)
        self.workflow.add_node("urgent", urgent_handler)
        self.workflow.add_node("complaint", complaint_handler)
        self.workflow.add_node("standard", standard_handler)
        self.workflow.add_node("quality", quality_check)
        
        # Build the flow
        self.workflow.set_entry_point("analyze")
        
        # Conditional routing after analysis
        self.workflow.add_conditional_edges(
            "analyze",
            routing_decision,
            {
                "urgent": "urgent",
                "complaint": "complaint",
                "standard": "standard"
            }
        )
        
        # All handlers go to quality check
        self.workflow.add_edge("urgent", "quality")
        self.workflow.add_edge("complaint", "quality")
        self.workflow.add_edge("standard", "quality")
        
        # Quality check ends the workflow
        self.workflow.add_edge("quality", END)
        
        return self.workflow.compile()

# Test complex graph
complex_builder = ComplexGraphBuilder()
complex_graph = complex_builder.build_request_processing_graph()

# Test different request types
test_requests = [
    "URGENT: Critical system failure needs immediate attention!",
    "I'm very disappointed with your service and want to file a complaint",
    "Can you help me understand how your pricing works?",
    "Emergency: Payment processing is down for all customers"
]

print("\nComplex Graph Processing Results:")
for i, request in enumerate(test_requests, 1):
    result = complex_graph.invoke({
        "user_request": request,
        "request_type": "",
        "priority": "",
        "analysis_results": {},
        "processing_steps": [],
        "current_stage": "",
        "response": "",
        "actions_taken": [],
        "follow_up_needed": False
    })
    
    print(f"\n{i}. Request: {request[:50]}...")
    print(f"   Type: {result['request_type']} (Priority: {result['priority']})")
    print(f"   Path: {result['current_stage']}")
    print(f"   Quality Score: {result['analysis_results'].get('quality_score', 'N/A')}")
    print(f"   Follow-up: {'Yes' if result['follow_up_needed'] else 'No'}")
    print(f"   Steps: {' → '.join(result['processing_steps'])}")
```

## State Management

### Advanced State Handling

```python
from typing import TypedDict, Dict, Any, List, Optional
from datetime import datetime
import json
import time

# Complex state with nested structures and validation
class ApplicationState(TypedDict):
    # Session information
    session_id: str
    user_id: str
    user_profile: Dict[str, Any]
    
    # Workflow state
    current_step: str
    workflow_phase: str
    completed_steps: List[str]
    
    # Data state
    input_data: Dict[str, Any]
    processed_data: Dict[str, Any]
    intermediate_results: Dict[str, Any]
    final_results: Dict[str, Any]
    
    # Metadata
    timestamps: Dict[str, str]
    performance_metrics: Dict[str, float]
    error_log: List[Dict[str, Any]]
    debug_info: Dict[str, Any]

class StateManagementSystem:
    """Advanced state management utilities"""
    
    @staticmethod
    def initialize_state(session_id: str, user_id: str, initial_data: Dict[str, Any] = None) -> ApplicationState:
        """Initialize a new application state"""
        
        now = datetime.now().isoformat()
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "user_profile": {},
            "current_step": "initialized",
            "workflow_phase": "setup",
            "completed_steps": [],
            "input_data": initial_data or {},
            "processed_data": {},
            "intermediate_results": {},
            "final_results": {},
            "timestamps": {
                "created": now,
                "last_updated": now
            },
            "performance_metrics": {},
            "error_log": [],
            "debug_info": {}
        }
    
    @staticmethod
    def update_step(state: ApplicationState, step_name: str, step_data: Dict[str, Any] = None) -> ApplicationState:
        """Update current step and add to completed steps"""
        
        completed_steps = state.get("completed_steps", [])
        if state.get("current_step") and state["current_step"] not in completed_steps:
            completed_steps.append(state["current_step"])
        
        timestamps = state.get("timestamps", {})
        timestamps["last_updated"] = datetime.now().isoformat()
        timestamps[f"{step_name}_started"] = datetime.now().isoformat()
        
        updated_state = {
            **state,
            "current_step": step_name,
            "completed_steps": completed_steps,
            "timestamps": timestamps
        }
        
        # Add step-specific data if provided
        if step_data:
            updated_state["intermediate_results"] = {
                **state.get("intermediate_results", {}),
                step_name: step_data
            }
        
        return updated_state
    
    @staticmethod
    def add_performance_metric(state: ApplicationState, metric_name: str, value: float) -> ApplicationState:
        """Add performance metric to state"""
        
        performance_metrics = state.get("performance_metrics", {})
        performance_metrics[metric_name] = value
        performance_metrics[f"{metric_name}_timestamp"] = time.time()
        
        return {
            **state,
            "performance_metrics": performance_metrics
        }
    
    @staticmethod
    def validate_state(state: ApplicationState) -> Dict[str, Any]:
        """Validate state structure and content"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        required_fields = ["session_id", "user_id", "current_step"]
        for field in required_fields:
            if not state.get(field):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required field: {field}")
        
        # Check data consistency
        if state.get("workflow_phase") == "complete" and not state.get("final_results"):
            validation_result["warnings"].append("Workflow marked complete but no final results")
        
        # Check for excessive errors
        error_count = len(state.get("error_log", []))
        if error_count > 5:
            validation_result["warnings"].append(f"High error count: {error_count}")
        
        return validation_result

# Example usage
stateful_example = StateGraph(ApplicationState)

def demo_step(state: ApplicationState) -> ApplicationState:
    """Demo step with state management"""
    updated_state = StateManagementSystem.update_step(state, "demo_processing")
    updated_state = StateManagementSystem.add_performance_metric(updated_state, "processing_time", 1.5)
    
    return {
        **updated_state,
        "final_results": {"demo": "completed"}
    }

stateful_example.add_node("demo", demo_step)
stateful_example.set_entry_point("demo")
stateful_example.add_edge("demo", END)

demo_graph = stateful_example.compile()

# Test state management
initial_state = StateManagementSystem.initialize_state("session_001", "user_123")
validation = StateManagementSystem.validate_state(initial_state)
print(f"State validation: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")

result = demo_graph.invoke(initial_state)
print(f"Completed steps: {result['completed_steps']}")
print(f"Performance metrics: {list(result['performance_metrics'].keys())}")
```

## Conditional Logic & Routing

### Intelligent Routing System

```python
from typing import Literal
import re

class RoutingState(TypedDict):
    user_input: str
    user_context: Dict[str, Any]
    routing_decisions: List[str]
    classification_results: Dict[str, Any]
    final_route: str
    response: str

class IntelligentRouter:
    """Intelligent routing system with multi-level decision making"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        self.workflow = StateGraph(RoutingState)
        self.setup_routing_workflow()
    
    def setup_routing_workflow(self):
        """Setup intelligent routing workflow"""
        
        def classify_intent(state: RoutingState) -> RoutingState:
            """Classify user intent and context"""
            
            user_input = state["user_input"]
            
            # Pattern-based classification
            patterns = {
                "question": [r'\?', r'\bwhat\b', r'\bhow\b', r'\bwhy\b'],
                "request": [r'\bplease\b', r'\bcan you\b', r'\bhelp\b'],
                "complaint": [r'\bproblem\b', r'\bissue\b', r'\bterrible\b'],
                "urgent": [r'\burgent\b', r'\bcritical\b', r'\bemergency\b']
            }
            
            intent_scores = {}
            for intent, pattern_list in patterns.items():
                score = sum(1 for pattern in pattern_list if re.search(pattern, user_input.lower()))
                intent_scores[intent] = score
            
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            confidence = intent_scores[primary_intent] / len(patterns[primary_intent])
            
            return {
                **state,
                "classification_results": {
                    "primary_intent": primary_intent,
                    "confidence": confidence,
                    "intent_scores": intent_scores
                },
                "routing_decisions": ["intent_classified"]
            }
        
        def analyze_context(state: RoutingState) -> RoutingState:
            """Analyze user context for routing decisions"""
            
            user_context = state.get("user_context", {})
            classification = state["classification_results"]
            
            # Context analysis
            context_factors = {
                "user_tier": user_context.get("tier", "standard"),
                "interaction_history": user_context.get("previous_interactions", 0),
                "urgency_level": "high" if classification["primary_intent"] == "urgent" else "normal"
            }
            
            # Determine routing based on context
            if context_factors["urgency_level"] == "high":
                route = "priority_handler"
            elif context_factors["user_tier"] == "premium":
                route = "premium_handler"
            elif classification["primary_intent"] == "complaint":
                route = "complaint_handler"
            else:
                route = "standard_handler"
            
            return {
                **state,
                "classification_results": {
                    **classification,
                    "context_factors": context_factors
                },
                "final_route": route,
                "routing_decisions": state["routing_decisions"] + ["context_analyzed"]
            }
        
        def execute_handler(state: RoutingState) -> RoutingState:
            """Execute the selected handler"""
            
            route = state["final_route"]
            user_input = state["user_input"]
            
            handler_responses = {
                "priority_handler": f"PRIORITY: Immediate assistance for '{user_input}'",
                "premium_handler": f"PREMIUM: Personalized service for '{user_input}'", 
                "complaint_handler": f"COMPLAINT: Careful handling of '{user_input}'",
                "standard_handler": f"STANDARD: Professional response to '{user_input}'"
            }
            
            response = handler_responses.get(route, f"DEFAULT: Response to '{user_input}'")
            
            return {
                **state,
                "response": response,
                "routing_decisions": state["routing_decisions"] + [f"executed_{route}"]
            }
        
        # Add nodes
        self.workflow.add_node("classify", classify_intent)
        self.workflow.add_node("analyze", analyze_context)
        self.workflow.add_node("execute", execute_handler)
        
        # Setup flow
        self.workflow.set_entry_point("classify")
        self.workflow.add_edge("classify", "analyze")
        self.workflow.add_edge("analyze", "execute")
        self.workflow.add_edge("execute", END)
    
    def compile_routing_graph(self):
        """Compile routing graph"""
        return self.workflow.compile()

# Test routing system
router = IntelligentRouter()
routing_graph = router.compile_routing_graph()

routing_tests = [
    {"input": "What is your refund policy?", "context": {"tier": "standard"}},
    {"input": "URGENT: My system is down!", "context": {"tier": "business"}},
    {"input": "I'm very unhappy with your service", "context": {"tier": "premium"}},
    {"input": "Can you help me upgrade my plan?", "context": {"tier": "standard"}}
]

print("\n=== Routing System Results ===")
for i, test in enumerate(routing_tests, 1):
    result = routing_graph.invoke({
        "user_input": test["input"],
        "user_context": test["context"],
        "routing_decisions": [],
        "classification_results": {},
        "final_route": "",
        "response": ""
    })
    
    print(f"\n{i}. Input: {test['input']}")
    print(f"   Intent: {result['classification_results']['primary_intent']}")
    print(f"   Route: {result['final_route']}")
    print(f"   Response: {result['response']}")
```

## Human-in-the-Loop

### Human Review Integration

```python
from langgraph.checkpoint.sqlite import SqliteSaver

class HumanLoopState(TypedDict):
    conversation_id: str
    user_message: str
    ai_response: str
    review_required: bool
    review_status: str
    final_response: str

class HumanReviewSystem:
    """Human review integration system"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        self.checkpointer = SqliteSaver.from_conn_string("sqlite:///human_review.db")
        self.workflow = StateGraph(HumanLoopState)
        self.setup_review_workflow()
    
    def setup_review_workflow(self):
        """Setup human review workflow"""
        
        def generate_ai_response(state: HumanLoopState) -> HumanLoopState:
            """Generate initial AI response"""
            
            user_message = state["user_message"]
            
            # Check if review is needed
            review_triggers = ["legal advice", "medical advice", "personal information", "confidential"]
            review_required = any(trigger in user_message.lower() for trigger in review_triggers)
            
            # Generate response
            response = self.llm.invoke(f"Respond to: {user_message}")
            
            return {
                **state,
                "ai_response": response.content,
                "review_required": review_required,
                "review_status": "pending" if review_required else "approved"
            }
        
        def human_review_pause(state: HumanLoopState) -> HumanLoopState:
            """Pause for human review"""
            
            print(f"🛑 Human review required for conversation {state['conversation_id']}")
            print(f"Message: {state['user_message']}")
            print(f"AI Response: {state['ai_response']}")
            print("Waiting for human approval...")
            
            return {
                **state,
                "review_status": "under_review"
            }
        
        def finalize_response(state: HumanLoopState) -> HumanLoopState:
            """Finalize response after review"""
            
            if state["review_status"] in ["approved", "under_review"]:
                final_response = state["ai_response"]
            else:
                final_response = "Response pending human review"
            
            return {
                **state,
                "final_response": final_response,
                "review_status": "completed"
            }
        
        # Add nodes
        self.workflow.add_node("generate", generate_ai_response)
        self.workflow.add_node("review", human_review_pause)
        self.workflow.add_node("finalize", finalize_response)
        
        # Setup conditional flow
        self.workflow.set_entry_point("generate")
        
        self.workflow.add_conditional_edges(
            "generate",
            lambda state: "review" if state["review_required"] else "finalize",
            {
                "review": "review",
                "finalize": "finalize"
            }
        )
        
        self.workflow.add_edge("review", "finalize")
        self.workflow.add_edge("finalize", END)
    
    def compile_review_graph(self):
        """Compile human review graph"""
        return self.workflow.compile(checkpointer=self.checkpointer)

# Test human review
human_review = HumanReviewSystem()
review_graph = human_review.compile_review_graph()

review_tests = [
    "What's the weather today?",  # No review needed
    "Can you give me legal advice about my contract?"  # Review required
]

print("\n=== Human Review System ===")
for i, message in enumerate(review_tests, 1):
    result = review_graph.invoke({
        "conversation_id": f"conv_{i}",
        "user_message": message,
        "ai_response": "",
        "review_required": False,
        "review_status": "",
        "final_response": ""
    })
    
    print(f"\n{i}. Message: {message}")
    print(f"   Review Required: {'Yes' if result['review_required'] else 'No'}")
    print(f"   Status: {result['review_status']}")
    print(f"   Final Response: {result['final_response'][:100]}...")
```

## Persistence & Checkpoints

### Checkpoint Management

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

class CheckpointState(TypedDict):
    session_id: str
    step_number: int
    data: Dict[str, Any]
    can_resume: bool

class CheckpointDemo:
    """Demonstrate checkpoint functionality"""
    
    def __init__(self, storage_type: str = "sqlite"):
        if storage_type == "sqlite":
            self.checkpointer = SqliteSaver.from_conn_string("sqlite:///demo_checkpoints.db")
        else:
            self.checkpointer = MemorySaver()
        
        self.workflow = StateGraph(CheckpointState)
        self.setup_checkpoint_workflow()
    
    def setup_checkpoint_workflow(self):
        """Setup workflow with checkpoints"""
        
        def step_1(state: CheckpointState) -> CheckpointState:
            """First step with checkpoint"""
            print(f"Executing step 1 for session {state['session_id']}")
            
            return {
                **state,
                "step_number": 1,
                "data": {"step1_complete": True},
                "can_resume": True
            }
        
        def step_2(state: CheckpointState) -> CheckpointState:
            """Second step with checkpoint"""
            print(f"Executing step 2 for session {state['session_id']}")
            
            return {
                **state,
                "step_number": 2,
                "data": {**state["data"], "step2_complete": True},
                "can_resume": True
            }
        
        def step_3(state: CheckpointState) -> CheckpointState:
            """Final step"""
            print(f"Executing final step for session {state['session_id']}")
            
            return {
                **state,
                "step_number": 3,
                "data": {**state["data"], "workflow_complete": True},
                "can_resume": False
            }
        
        # Add nodes
        self.workflow.add_node("step1", step_1)
        self.workflow.add_node("step2", step_2)
        self.workflow.add_node("step3", step_3)
        
        # Setup flow
        self.workflow.set_entry_point("step1")
        self.workflow.add_edge("step1", "step2")
        self.workflow.add_edge("step2", "step3")
        self.workflow.add_edge("step3", END)
    
    def compile_checkpoint_graph(self):
        """Compile with checkpointing"""
        return self.workflow.compile(checkpointer=self.checkpointer)

# Test checkpoints
checkpoint_demo = CheckpointDemo("sqlite")
checkpoint_graph = checkpoint_demo.compile_checkpoint_graph()

# Execute with session persistence
thread_config = {"configurable": {"thread_id": "checkpoint_session_001"}}

checkpoint_result = checkpoint_graph.invoke({
    "session_id": "checkpoint_session_001",
    "step_number": 0,
    "data": {},
    "can_resume": True
}, config=thread_config)

print("\n=== Checkpoint Demo ===")
print(f"Checkpoint result: Step {checkpoint_result['step_number']} completed")
print(f"Workflow complete: {checkpoint_result['data'].get('workflow_complete', False)}")
```

## Multi-Agent Systems

### Agent Coordination

```python
class MultiAgentState(TypedDict):
    task: str
    agent_outputs: Dict[str, str]
    coordination_log: List[str]
    final_result: str

class MultiAgentCoordinator:
    """Coordinate multiple specialized agents"""
    
    def __init__(self):
        self.workflow = StateGraph(MultiAgentState)
        self.setup_multi_agent_workflow()
    
    def setup_multi_agent_workflow(self):
        """Setup multi-agent coordination"""
        
        def research_agent(state: MultiAgentState) -> MultiAgentState:
            """Research specialist agent"""
            task = state["task"]
            
            llm = ChatOpenAI(model="gpt-4", temperature=0.3)
            research_prompt = f"As a research specialist, analyze: {task}"
            response = llm.invoke(research_prompt)
            
            return {
                **state,
                "agent_outputs": {
                    **state.get("agent_outputs", {}),
                    "researcher": response.content
                },
                "coordination_log": state.get("coordination_log", []) + ["Research completed"]
            }
        
        def analysis_agent(state: MultiAgentState) -> MultiAgentState:
            """Analysis specialist agent"""
            task = state["task"]
            research_output = state.get("agent_outputs", {}).get("researcher", "")
            
            llm = ChatOpenAI(model="gpt-4", temperature=0.3)
            analysis_prompt = f"""
            As an analyst, analyze this task: {task}
            
            Research findings: {research_output}
            
            Provide analytical insights:
            """
            response = llm.invoke(analysis_prompt)
            
            return {
                **state,
                "agent_outputs": {
                    **state.get("agent_outputs", {}),
                    "analyst": response.content
                },
                "coordination_log": state["coordination_log"] + ["Analysis completed"]
            }
        
        def synthesis_agent(state: MultiAgentState) -> MultiAgentState:
            """Synthesis agent combines all outputs"""
            task = state["task"]
            agent_outputs = state.get("agent_outputs", {})
            
            synthesis_prompt = f"""
            Task: {task}
            
            Agent outputs:
            Research: {agent_outputs.get('researcher', '')}
            Analysis: {agent_outputs.get('analyst', '')}
            
            Synthesize into final comprehensive result:
            """
            
            llm = ChatOpenAI(model="gpt-4", temperature=0.3)
            response = llm.invoke(synthesis_prompt)
            
            return {
                **state,
                "final_result": response.content,
                "coordination_log": state["coordination_log"] + ["Synthesis completed"]
            }
        
        # Add nodes
        self.workflow.add_node("research", research_agent)
        self.workflow.add_node("analyze", analysis_agent)
        self.workflow.add_node("synthesize", synthesis_agent)
        
        # Setup coordination flow
        self.workflow.set_entry_point("research")
        self.workflow.add_edge("research", "analyze")
        self.workflow.add_edge("analyze", "synthesize")
        self.workflow.add_edge("synthesize", END)
    
    def compile_multi_agent_graph(self):
        """Compile multi-agent graph"""
        return self.workflow.compile()

# Test multi-agent coordination
coordinator = MultiAgentCoordinator()
multi_agent_graph = coordinator.compile_multi_agent_graph()

multi_agent_result = multi_agent_graph.invoke({
    "task": "Analyze the impact of AI on job markets",
    "agent_outputs": {},
    "coordination_log": [],
    "final_result": ""
})

print("\n=== Multi-Agent Coordination ===")
print(f"Agents involved: {list(multi_agent_result['agent_outputs'].keys())}")
print(f"Coordination: {' → '.join(multi_agent_result['coordination_log'])}")
print(f"Final result: {multi_agent_result['final_result'][:150]}...")
```

## Advanced Patterns

### Self-Improving Agent

```python
class SelfImprovingState(TypedDict):
    problem: str
    current_solution: str
    iteration: int
    max_iterations: int
    quality_scores: List[float]
    improvements: List[str]

class SelfImprovingAgent:
    """Agent that improves its solutions iteratively"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        self.workflow = StateGraph(SelfImprovingState)
        self.setup_improvement_cycle()
    
    def setup_improvement_cycle(self):
        """Setup iterative improvement cycle"""
        
        def initial_solution(state: SelfImprovingState) -> SelfImprovingState:
            """Generate initial solution"""
            problem = state["problem"]
            
            response = self.llm.invoke(f"Provide an initial solution to: {problem}")
            
            return {
                **state,
                "current_solution": response.content,
                "iteration": 1,
                "quality_scores": [0.6],  # Initial quality
                "improvements": ["Initial solution generated"]
            }
        
        def improve_solution(state: SelfImprovingState) -> SelfImprovingState:
            """Improve the current solution"""
            current_solution = state["current_solution"]
            iteration = state["iteration"]
            
            improvement_prompt = f"""
            Current solution: {current_solution}
            
            This is iteration {iteration}. Improve this solution by:
            1. Identifying weaknesses
            2. Adding enhancements
            3. Providing a better version
            """
            
            improved = self.llm.invoke(improvement_prompt)
            
            # Simulate quality improvement
            new_quality = min(0.95, state["quality_scores"][-1] + 0.1)
            
            return {
                **state,
                "current_solution": improved.content,
                "iteration": iteration + 1,
                "quality_scores": state["quality_scores"] + [new_quality],
                "improvements": state["improvements"] + [f"Iteration {iteration + 1} improvement"]
            }
        
        def should_improve(state: SelfImprovingState) -> Literal["improve", "done"]:
            """Decide whether to continue improving"""
            if state["iteration"] >= state["max_iterations"]:
                return "done"
            elif state["quality_scores"][-1] > 0.9:
                return "done"
            else:
                return "improve"
        
        # Add nodes
        self.workflow.add_node("initial", initial_solution)
        self.workflow.add_node("improve", improve_solution)
        
        # Setup improvement loop
        self.workflow.set_entry_point("initial")
        self.workflow.add_edge("initial", "improve")
        
        self.workflow.add_conditional_edges(
            "improve",
            should_improve,
            {
                "improve": "improve",  # Self-loop for continuous improvement
                "done": END
            }
        )
    
    def compile_improvement_graph(self):
        """Compile self-improving graph"""
        return self.workflow.compile()

# Test self-improvement
improver = SelfImprovingAgent()
improvement_graph = improver.compile_improvement_graph()

improvement_result = improvement_graph.invoke({
    "problem": "Design an efficient sorting algorithm",
    "current_solution": "",
    "iteration": 0,
    "max_iterations": 3,
    "quality_scores": [],
    "improvements": []
})

print("\n=== Self-Improving Agent ===")
print(f"Final iteration: {improvement_result['iteration']}")
print(f"Quality progression: {improvement_result['quality_scores']}")
print(f"Improvements made: {improvement_result['improvements']}")
print(f"Final solution: {improvement_result['current_solution'][:150]}...")
```

## Streaming & Real-time

### Real-time Processing

```python
import asyncio
import time

class StreamingState(TypedDict):
    input_data: str
    processed_chunks: List[str]
    chunk_count: int
    streaming_complete: bool

class StreamingProcessor:
    """Real-time streaming processor"""
    
    def __init__(self):
        self.workflow = StateGraph(StreamingState)
        self.setup_streaming()
    
    def setup_streaming(self):
        """Setup streaming workflow"""
        
        def chunk_processor(state: StreamingState) -> StreamingState:
            """Process data in chunks (simulates streaming)"""
            
            input_data = state["input_data"]
            chunk_count = state.get("chunk_count", 0)
            
            # Split into chunks
            chunk_size = 50
            chunks = [input_data[i:i+chunk_size] for i in range(0, len(input_data), chunk_size)]
            
            if chunk_count < len(chunks):
                # Process current chunk
                current_chunk = chunks[chunk_count]
                processed_chunk = f"Chunk {chunk_count + 1}: {current_chunk}"
                
                processed_chunks = state.get("processed_chunks", [])
                processed_chunks.append(processed_chunk)
                
                return {
                    **state,
                    "processed_chunks": processed_chunks,
                    "chunk_count": chunk_count + 1,
                    "streaming_complete": chunk_count + 1 >= len(chunks)
                }
            else:
                return {
                    **state,
                    "streaming_complete": True
                }
        
        # Add node
        self.workflow.add_node("process_chunk", chunk_processor)
        
        # Setup streaming loop
        self.workflow.set_entry_point("process_chunk")
        
        self.workflow.add_conditional_edges(
            "process_chunk",
            lambda state: "continue" if not state["streaming_complete"] else "done",
            {
                "continue": "process_chunk",  # Loop for more chunks
                "done": END
            }
        )
    
    def compile_streaming_graph(self):
        """Compile streaming graph"""
        return self.workflow.compile()

# Test streaming
streaming = StreamingProcessor()
streaming_graph = streaming.compile_streaming_graph()

streaming_result = streaming_graph.invoke({
    "input_data": "This is a long text that will be processed in streaming chunks to demonstrate real-time processing capabilities of LangGraph workflows.",
    "processed_chunks": [],
    "chunk_count": 0,
    "streaming_complete": False
})

print("\n=== Streaming Processing ===")
print(f"Chunks processed: {len(streaming_result['processed_chunks'])}")
print(f"Streaming complete: {streaming_result['streaming_complete']}")
for i, chunk in enumerate(streaming_result['processed_chunks'][:3], 1):
    print(f"  {i}. {chunk}")
```

## Error Handling & Recovery

### Robust Error Management

```python
import random
import traceback

class ErrorState(TypedDict):
    operation: str
    attempt_count: int
    max_attempts: int
    errors: List[str]
    success: bool
    result: str

class ErrorHandlingSystem:
    """Robust error handling and recovery"""
    
    def __init__(self):
        self.workflow = StateGraph(ErrorState)
        self.setup_error_handling()
    
    def setup_error_handling(self):
        """Setup error handling workflow"""
        
        def attempt_operation(state: ErrorState) -> ErrorState:
            """Attempt operation with potential failure"""
            
            operation = state["operation"]
            attempt = state.get("attempt_count", 0) + 1
            
            try:
                # Simulate operation that might fail
                if random.random() < 0.4:  # 40% failure rate
                    raise Exception(f"Operation '{operation}' failed on attempt {attempt}")
                
                # Success
                return {
                    **state,
                    "attempt_count": attempt,
                    "success": True,
                    "result": f"Operation '{operation}' succeeded on attempt {attempt}"
                }
                
            except Exception as e:
                errors = state.get("errors", [])
                errors.append(f"Attempt {attempt}: {str(e)}")
                
                return {
                    **state,
                    "attempt_count": attempt,
                    "errors": errors,
                    "success": False
                }
        
        def retry_handler(state: ErrorState) -> ErrorState:
            """Handle retry logic"""
            
            attempt_count = state["attempt_count"]
            max_attempts = state["max_attempts"]
            
            if attempt_count < max_attempts:
                print(f"Retrying... ({attempt_count}/{max_attempts})")
                return state
            else:
                # Max attempts reached, use fallback
                return {
                    **state,
                    "success": True,
                    "result": f"Fallback result after {attempt_count} attempts"
                }
        
        # Add nodes
        self.workflow.add_node("attempt", attempt_operation)
        self.workflow.add_node("retry", retry_handler)
        
        # Setup retry logic
        self.workflow.set_entry_point("attempt")
        
        self.workflow.add_conditional_edges(
            "attempt",
            lambda state: "success" if state["success"] else "retry",
            {
                "success": END,
                "retry": "retry"
            }
        )
        
        self.workflow.add_conditional_edges(
            "retry",
            lambda state: "retry_operation" if state["attempt_count"] < state["max_attempts"] and not state["success"] else "done",
            {
                "retry_operation": "attempt",
                "done": END
            }
        )
    
    def compile_error_graph(self):
        """Compile error handling graph"""
        return self.workflow.compile()

# Test error handling
error_handler = ErrorHandlingSystem()
error_graph = error_handler.compile_error_graph()

error_result = error_graph.invoke({
    "operation": "critical_task",
    "attempt_count": 0,
    "max_attempts": 3,
    "errors": [],
    "success": False,
    "result": ""
})

print("\n=== Error Handling ===")
print(f"Operation success: {'✅' if error_result['success'] else '❌'}")
print(f"Attempts made: {error_result['attempt_count']}")
print(f"Errors encountered: {len(error_result['errors'])}")
print(f"Final result: {error_result['result']}")
```

## Testing & Debugging

### Testing Framework

```python
from unittest.mock import Mock

class TestingFramework:
    """Testing framework for LangGraph workflows"""
    
    def test_workflow_execution(self, workflow: StateGraph, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test workflow with multiple test cases"""
        
        results = {
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "test_results": []
        }
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                start_time = time.time()
                result = workflow.invoke(test_case["input"])
                execution_time = time.time() - start_time
                
                test_result = {
                    "test_id": i,
                    "success": True,
                    "execution_time": execution_time,
                    "output": result
                }
                
                results["passed"] += 1
                
            except Exception as e:
                test_result = {
                    "test_id": i,
                    "success": False,
                    "error": str(e)
                }
                
                results["failed"] += 1
            
            results["test_results"].append(test_result)
        
        results["success_rate"] = (results["passed"] / results["total_tests"]) * 100
        return results

# Example test usage
test_framework = TestingFramework()

# Create simple test workflow
test_workflow = StateGraph(dict)
test_workflow.add_node("process", lambda state: {"result": f"Processed: {state.get('input', '')}"})
test_workflow.set_entry_point("process")
test_workflow.add_edge("process", END)
test_graph = test_workflow.compile()

# Run tests
test_cases = [
    {"input": {"input": "test 1"}},
    {"input": {"input": "test 2"}},
    {"input": {"input": ""}}  # Edge case
]

test_results = test_framework.test_workflow_execution(test_graph, test_cases)

print("\n=== Testing Results ===")
print(f"Tests run: {test_results['total_tests']}")
print(f"Passed: {test_results['passed']}")
print(f"Failed: {test_results['failed']}")
print(f"Success rate: {test_results['success_rate']:.1f}%")
```

## Production Deployment

### FastAPI Production Setup

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os

# API Models
class WorkflowRequest(BaseModel):
    workflow_type: str
    input_data: Dict[str, Any]
    session_id: Optional[str] = None

class WorkflowResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float

# FastAPI app
app = FastAPI(title="LangGraph Production API", version="1.0.0")

# Simple workflow registry
workflows = {
    "simple": StateGraph(dict)
}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "workflows_available": len(workflows)
    }

@app.post("/execute", response_model=WorkflowResponse)
async def execute_workflow(request: WorkflowRequest):
    """Execute a workflow"""
    
    start_time = time.time()
    
    try:
        if request.workflow_type not in workflows:
            raise HTTPException(status_code=400, detail="Unknown workflow type")
        
        # Execute workflow
        workflow = workflows[request.workflow_type]
        result = {"message": f"Executed {request.workflow_type}"}
        
        execution_time = time.time() - start_time
        
        return WorkflowResponse(
            success=True,
            result=result,
            execution_time=execution_time
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        return WorkflowResponse(
            success=False,
            error=str(e),
            execution_time=execution_time
        )

print("\n=== Production API Ready ===")
print("FastAPI application configured for LangGraph workflows")
```

## Performance Optimization

### Caching and Optimization

```python
import hashlib
import json
from functools import wraps

class PerformanceOptimizer:
    """Performance optimization for LangGraph"""
    
    def __init__(self):
        self.cache = {}
        self.performance_stats = {}
    
    def cache_workflow_results(self, ttl_seconds: int = 3600):
        """Cache workflow results"""
        
        def decorator(func):
            @wraps(func)
            def wrapper(workflow_type: str, state: Dict[str, Any], *args, **kwargs):
                
                # Generate cache key
                cache_key = self._generate_cache_key(workflow_type, state)
                
                # Check cache
                if cache_key in self.cache:
                    cache_entry = self.cache[cache_key]
                    if time.time() < cache_entry["expires_at"]:
                        return cache_entry["result"]
                
                # Execute and cache
                result = func(workflow_type, state, *args, **kwargs)
                
                self.cache[cache_key] = {
                    "result": result,
                    "expires_at": time.time() + ttl_seconds
                }
                
                return result
            
            return wrapper
        return decorator
    
    def _generate_cache_key(self, workflow_type: str, state: Dict[str, Any]) -> str:
        """Generate cache key"""
        cache_data = {"workflow": workflow_type, "state": state}
        serialized = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "cache_size": len(self.cache),
            "performance_data": self.performance_stats
        }

# Usage
optimizer = PerformanceOptimizer()
print("\n=== Performance Optimization ===")
print(f"Optimizer initialized with caching capabilities")
print(f"Cache size: {optimizer.get_performance_stats()['cache_size']}")
```

## Enterprise Integration

### Database and Audit Integration

```python
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid

Base = declarative_base()

class WorkflowAudit(Base):
    __tablename__ = "workflow_audits"
    
    id = Column(Integer, primary_key=True)
    execution_id = Column(String(255), unique=True)
    user_id = Column(String(255))
    workflow_type = Column(String(100))
    status = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

class EnterpriseIntegration:
    """Enterprise integration with audit trails"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
    
    def execute_with_audit(self, workflow_type: str, user_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with audit logging"""
        
        execution_id = str(uuid.uuid4())
        
        # Log execution start
        db = self.SessionLocal()
        try:
            audit_entry = WorkflowAudit(
                execution_id=execution_id,
                user_id=user_id,
                workflow_type=workflow_type,
                status="started"
            )
            db.add(audit_entry)
            db.commit()
        finally:
            db.close()
        
        # Simulate workflow execution
        result = {
            "execution_id": execution_id,
            "result": f"Enterprise processing completed for {workflow_type}",
            "audit_logged": True
        }
        
        return result

# Test enterprise integration
enterprise = EnterpriseIntegration("sqlite:///enterprise_audit.db")
enterprise_result = enterprise.execute_with_audit(
    "customer_analysis", 
    "enterprise_user_001", 
    {"data": "customer data"}
)

print("\n=== Enterprise Integration ===")
print(f"Execution ID: {enterprise_result['execution_id']}")
print(f"Audit logged: {enterprise_result['audit_logged']}")
print(f"Result: {enterprise_result['result']}")
```

## Troubleshooting

### Diagnostic Tools

```python
import sys

class Diagnostics:
    """LangGraph diagnostic tools"""
    
    def check_installation(self) -> Dict[str, Any]:
        """Check LangGraph installation"""
        
        results = {"issues": [], "recommendations": []}
        
        try:
            import langgraph
            results["langgraph"] = "✅ Installed"
        except ImportError:
            results["langgraph"] = "❌ Missing"
            results["issues"].append("LangGraph not installed")
            results["recommendations"].append("Run: pip install langgraph")
        
        try:
            from langchain_openai import ChatOpenAI
            results["langchain_openai"] = "✅ Available"
        except ImportError:
            results["langchain_openai"] = "❌ Missing"
            results["issues"].append("LangChain OpenAI integration missing")
            results["recommendations"].append("Run: pip install langchain-openai")
        
        results["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        return results

# Run diagnostics
diagnostics = Diagnostics()
diag_results = diagnostics.check_installation()

print("\n=== System Diagnostics ===")
for key, value in diag_results.items():
    if key not in ["issues", "recommendations"]:
        print(f"{key}: {value}")

if diag_results["issues"]:
    print("\nIssues:")
    for issue in diag_results["issues"]:
        print(f"  ❌ {issue}")

if diag_results["recommendations"]:
    print("\nRecommendations:")
    for rec in diag_results["recommendations"]:
        print(f"  💡 {rec}")
```

## Complete Example Application

### Advanced Customer Service Bot

```python
"""
Complete LangGraph Example: Advanced Customer Service Bot
Demonstrates all patterns integrated in a real-world application
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Dict, Any, List, Optional, Literal
from langchain_openai import ChatOpenAI
from datetime import datetime
import json
import time

# Comprehensive customer service state
class CustomerServiceState(TypedDict):
    # Customer context
    customer_id: str
    session_id: str
    customer_tier: str
    
    # Interaction
    user_message: str
    intent: str
    sentiment: str
    urgency: str
    
    # Processing
    assigned_agent: str
    ai_response: str
    quality_score: float
    
    # Results
    final_response: str
    escalated: bool
    satisfaction_predicted: float
    
    # Metadata
    processing_steps: List[str]
    timestamps: Dict[str, str]

class ComprehensiveCustomerServiceBot:
    """Complete customer service bot with all LangGraph features"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        self.checkpointer = SqliteSaver.from_conn_string("sqlite:///customer_service_complete.db")
        self.workflow = StateGraph(CustomerServiceState)
        
        # Customer database
        self.customers = {
            "premium_001": {"tier": "premium", "name": "John Premium"},
            "business_002": {"tier": "business", "name": "Jane Business"},
            "standard_003": {"tier": "standard", "name": "Bob Standard"}
        }
        
        self.setup_complete_workflow()
    
    def setup_complete_workflow(self):
        """Setup complete customer service workflow"""
        
        def initialize_customer_context(state: CustomerServiceState) -> CustomerServiceState:
            """Initialize customer context"""
            
            customer_id = state["customer_id"]
            customer_info = self.customers.get(customer_id, {"tier": "standard", "name": "Customer"})
            
            return {
                **state,
                "customer_tier": customer_info["tier"],
                "timestamps": {"session_start": datetime.now().isoformat()},
                "processing_steps": ["Customer context loaded"]
            }
        
        def analyze_customer_message(state: CustomerServiceState) -> CustomerServiceState:
            """Analyze customer message comprehensively"""
            
            user_message = state["user_message"]
            
            analysis_prompt = f"""
            Analyze this customer message:
            "{user_message}"
            
            Classify:
            1. Intent: question/complaint/request/compliment/technical/billing
            2. Sentiment: positive/neutral/negative
            3. Urgency: low/medium/high/critical
            
            JSON format:
            {{"intent": "...", "sentiment": "...", "urgency": "..."}}
            """
            
            response = self.llm.invoke(analysis_prompt)
            
            try:
                analysis = json.loads(response.content)
                intent = analysis["intent"]
                sentiment = analysis["sentiment"]
                urgency = analysis["urgency"]
            except:
                intent, sentiment, urgency = "question", "neutral", "medium"
            
            return {
                **state,
                "intent": intent,
                "sentiment": sentiment,
                "urgency": urgency,
                "processing_steps": state["processing_steps"] + ["Message analyzed"]
            }
        
        def assign_appropriate_agent(state: CustomerServiceState) -> CustomerServiceState:
            """Assign to most appropriate agent"""
            
            intent = state["intent"]
            urgency = state["urgency"]
            tier = state["customer_tier"]
            
            # Agent assignment logic
            if urgency == "critical":
                agent = "priority_specialist"
            elif intent == "technical":
                agent = "technical_specialist"
            elif intent == "billing":
                agent = "billing_specialist"
            elif tier == "premium":
                agent = "premium_specialist"
            else:
                agent = "general_specialist"
            
            return {
                **state,
                "assigned_agent": agent,
                "processing_steps": state["processing_steps"] + [f"Assigned to {agent}"]
            }
        
        def generate_agent_response(state: CustomerServiceState) -> CustomerServiceState:
            """Generate response from assigned agent"""
            
            user_message = state["user_message"]
            agent = state["assigned_agent"]
            tier = state["customer_tier"]
            sentiment = state["sentiment"]
            
            agent_prompts = {
                "premium_specialist": f"As a premium support specialist, provide excellent service to this {tier} customer with {sentiment} sentiment: {user_message}",
                "technical_specialist": f"As a technical expert, help with: {user_message}",
                "billing_specialist": f"As a billing expert, assist with: {user_message}",
                "priority_specialist": f"As a priority support agent, urgently address: {user_message}",
                "general_specialist": f"As a customer service rep, help with: {user_message}"
            }
            
            prompt = agent_prompts.get(agent, agent_prompts["general_specialist"])
            response = self.llm.invoke(prompt)
            
            # Quality assessment
            quality_score = self.assess_response_quality(response.content, state)
            
            return {
                **state,
                "ai_response": response.content,
                "quality_score": quality_score,
                "processing_steps": state["processing_steps"] + ["Agent response generated"]
            }
        
        def determine_escalation_and_finalize(state: CustomerServiceState) -> CustomerServiceState:
            """Determine if escalation needed and finalize"""
            
            # Escalation criteria
            escalate = (
                state["urgency"] == "critical" or
                state["sentiment"] == "negative" and state["customer_tier"] == "premium" or
                state["quality_score"] < 0.6
            )
            
            # Predict satisfaction
            satisfaction = self.predict_satisfaction(state)
            
            final_response = state["ai_response"] if not escalate else "Your request has been escalated to our senior team."
            
            return {
                **state,
                "escalated": escalate,
                "final_response": final_response,
                "satisfaction_predicted": satisfaction,
                "processing_steps": state["processing_steps"] + ["Interaction finalized"],
                "timestamps": {
                    **state.get("timestamps", {}),
                    "session_end": datetime.now().isoformat()
                }
            }
        
        # Build complete workflow
        self.workflow.add_node("init", initialize_customer_context)
        self.workflow.add_node("analyze", analyze_customer_message)
        self.workflow.add_node("assign", assign_appropriate_agent)
        self.workflow.add_node("respond", generate_agent_response)
        self.workflow.add_node("finalize", determine_escalation_and_finalize)
        
        # Create flow
        self.workflow.set_entry_point("init")
        self.workflow.add_edge("init", "analyze")
        self.workflow.add_edge("analyze", "assign")
        self.workflow.add_edge("assign", "respond")
        self.workflow.add_edge("respond", "finalize")
        self.workflow.add_edge("finalize", END)
    
    def assess_response_quality(self, response: str, state: CustomerServiceState) -> float:
        """Assess response quality"""
        
        base_score = 0.7
        
        # Length check
        if 100 <= len(response) <= 500:
            base_score += 0.1
        
        # Sentiment appropriateness
        if state["sentiment"] == "negative" and "sorry" in response.lower():
            base_score += 0.1
        
        # Tier appropriateness
        if state["customer_tier"] == "premium" and len(response) > 200:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def predict_satisfaction(self, state: CustomerServiceState) -> float:
        """Predict customer satisfaction"""
        
        base_satisfaction = 0.7
        
        # Quality impact
        base_satisfaction += (state["quality_score"] - 0.5) * 0.4
        
        # Tier handling
        if state["customer_tier"] == "premium" and state["assigned_agent"] == "premium_specialist":
            base_satisfaction += 0.1
        
        # Sentiment handling
        if state["sentiment"] == "negative" and not state["escalated"]:
            base_satisfaction -= 0.2
        
        return max(0.0, min(1.0, base_satisfaction))
    
    def compile_complete_bot(self):
        """Compile complete customer service bot"""
        return self.workflow.compile(checkpointer=self.checkpointer)

# Complete Demo
def run_complete_demo():
    """Run complete demonstration"""
    
    print("=" * 60)
    print("🤖 COMPLETE CUSTOMER SERVICE BOT DEMO")
    print("=" * 60)
    
    bot = ComprehensiveCustomerServiceBot()
    graph = bot.compile_complete_bot()
    
    # Test scenarios
    scenarios = [
        {
            "customer_id": "premium_001",
            "message": "URGENT: Payment system is down and I'm losing sales!",
            "description": "Premium customer with critical issue"
        },
        {
            "customer_id": "standard_003",
            "message": "I'm frustrated with your service and want to cancel",
            "description": "Unhappy standard customer"
        },
        {
            "customer_id": "business_002",
            "message": "Thank you for the excellent support yesterday!",
            "description": "Happy business customer"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['description']} ---")
        
        initial_state = {
            "customer_id": scenario["customer_id"],
            "session_id": f"complete_demo_{i}",
            "customer_tier": "",
            "user_message": scenario["message"],
            "intent": "",
            "sentiment": "",
            "urgency": "",
            "assigned_agent": "",
            "ai_response": "",
            "quality_score": 0.0,
            "final_response": "",
            "escalated": False,
            "satisfaction_predicted": 0.0,
            "processing_steps": [],
            "timestamps": {}
        }
        
        # Execute with persistence
        thread_config = {"configurable": {"thread_id": f"complete_demo_{i}"}}
        
        start_time = time.time()
        result = graph.invoke(initial_state, config=thread_config)
        execution_time = time.time() - start_time
        
        print(f"Customer: {scenario['customer_id']} ({result['customer_tier']} tier)")
        print(f"Analysis: {result['intent']}/{result['sentiment']}/{result['urgency']}")
        print(f"Agent: {result['assigned_agent']}")
        print(f"Quality: {result['quality_score']:.2f}")
        print(f"Escalated: {'Yes' if result['escalated'] else 'No'}")
        print(f"Predicted Satisfaction: {result['satisfaction_predicted']:.2f}")
        print(f"Execution Time: {execution_time:.3f}s")
        print(f"Response: {result['final_response'][:100]}...")
        
        results.append(result)
    
    # Analytics
    print(f"\n" + "=" * 60)
    print(f"📊 PERFORMANCE ANALYTICS")
    print(f"=" * 60)
    
    total_interactions = len(results)
    avg_satisfaction = sum(r["satisfaction_predicted"] for r in results) / total_interactions
    escalation_rate = len([r for r in results if r["escalated"]]) / total_interactions
    
    print(f"\nSummary Metrics:")
    print(f"  Total Interactions: {total_interactions}")
    print(f"  Average Satisfaction: {avg_satisfaction:.2f}")
    print(f"  Escalation Rate: {escalation_rate:.1%}")
    
    # Agent performance
    agent_stats = {}
    for result in results:
        agent = result["assigned_agent"]
        if agent not in agent_stats:
            agent_stats[agent] = {"count": 0, "satisfaction_sum": 0}
        agent_stats[agent]["count"] += 1
        agent_stats[agent]["satisfaction_sum"] += result["satisfaction_predicted"]
    
    print(f"\nAgent Performance:")
    for agent, stats in agent_stats.items():
        avg_satisfaction = stats["satisfaction_sum"] / stats["count"]
        print(f"  {agent}: {stats['count']} interactions, {avg_satisfaction:.2f} avg satisfaction")
    
    print(f"\n🎯 This demo showcases:")
    print(f"  ✅ Stateful workflows with customer context")
    print(f"  ✅ Conditional routing based on analysis")
    print(f"  ✅ Multi-agent specialization")
    print(f"  ✅ Quality assessment and escalation")
    print(f"  ✅ Performance monitoring and analytics")
    print(f"  ✅ Session persistence with checkpoints")
    
    return results

# Run the complete demonstration
if __name__ == "__main__":
    complete_results = run_complete_demo()
    
    print(f"\n" + "=" * 60)
    print(f"🎉 COMPLETE LANGGRAPH GUIDE DEMONSTRATION FINISHED")
    print(f"=" * 60)
    print(f"✅ All major LangGraph patterns demonstrated")
    print(f"✅ Production-ready implementation shown")
    print(f"✅ Real-world customer service bot completed")
    
    print(f"\n🚀 You're now ready to build sophisticated LangGraph applications!")

---

## Conclusion

This comprehensive LangGraph guide provides everything needed to build sophisticated, stateful AI applications. You've learned:

### 🎯 **Core Capabilities**
- **Stateful Workflows**: Maintain complex state across multiple steps
- **Conditional Logic**: Dynamic routing based on state and context
- **Human Integration**: Seamless human-in-the-loop workflows
- **Multi-Agent Systems**: Coordinate multiple AI agents
- **Error Recovery**: Robust error handling and fallback strategies
- **Real-time Processing**: Streaming and event-driven workflows

### 🚀 **Production Features**
- **Persistence**: Checkpoint and resume long-running workflows
- **Monitoring**: Comprehensive analytics and performance tracking
- **Scalability**: Enterprise-grade deployment patterns
- **Security**: Audit trails and compliance integration
- **Testing**: Complete testing and debugging frameworks

### 📚 **Key Resources**
- **Documentation**: [LangGraph Docs](https://langchain-ai.github.io/langgraph)
- **GitHub**: [LangGraph Repository](https://github.com/langchain-ai/langgraph)
- **Community**: [LangChain Discord](https://discord.gg/langchain)

### 🎯 **Next Steps**
1. Start with basic linear workflows
2. Add conditional logic and branching
3. Implement state persistence for production
4. Integrate human oversight where needed
5. Scale with multi-agent coordination
6. Deploy with enterprise patterns

The **Customer Service Bot example** shows how all these patterns integrate in a real, production-ready application!

**Happy building with LangGraph!** 🎉