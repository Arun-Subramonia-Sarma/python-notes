# Comprehensive RAG (Retrieval-Augmented Generation) Guide

A complete guide to building production-ready RAG systems with embedding models, vector databases, and advanced retrieval strategies.

## Table of Contents

1. [Introduction to RAG](#chapter-1-introduction-to-rag)
2. [RAG Architecture and Components](#chapter-2-rag-architecture-and-components)
3. [Embedding Models Deep Dive](#chapter-3-embedding-models-deep-dive)
4. [Vector Databases Comparison](#chapter-4-vector-databases-comparison)
5. [Document Processing and Chunking Strategies](#chapter-5-document-processing-and-chunking-strategies)
6. [Retrieval Strategies and Optimization](#chapter-6-retrieval-strategies-and-optimization)
7. [Generation and Response Synthesis](#chapter-7-generation-and-response-synthesis)
8. [Advanced RAG Patterns](#chapter-8-advanced-rag-patterns)
9. [Evaluation and Monitoring](#chapter-9-evaluation-and-monitoring)
10. [Production Deployment and Scaling](#chapter-10-production-deployment-and-scaling)

---

## Chapter 1: Introduction to RAG

### 1.1 What is RAG?

Retrieval-Augmented Generation (RAG) is an AI framework that combines the strengths of pre-trained language models with external knowledge retrieval to generate more accurate, up-to-date, and contextually relevant responses.

```python
# Basic RAG workflow conceptual example
def basic_rag_pipeline(query: str, knowledge_base: list) -> str:
    """
    Simplified RAG pipeline demonstration
    """
    # Step 1: Retrieve relevant documents
    relevant_docs = retrieve_documents(query, knowledge_base)
    
    # Step 2: Augment query with retrieved context
    augmented_prompt = f"""
    Context: {' '.join(relevant_docs)}
    
    Question: {query}
    
    Answer based on the provided context:
    """
    
    # Step 3: Generate response
    response = generate_response(augmented_prompt)
    
    return response
```

### 1.2 Why RAG?

**Problems RAG Solves:**

1. **Knowledge Cutoff**: LLMs have training data cutoffs
2. **Hallucination**: LLMs can generate factually incorrect information
3. **Domain Specificity**: Generic models lack specialized knowledge
4. **Real-time Information**: Models can't access current data
5. **Transparency**: Difficult to trace information sources

**RAG Advantages:**

- **Up-to-date Information**: Access to current data
- **Source Attribution**: Traceable information sources
- **Domain Expertise**: Specialized knowledge integration
- **Reduced Hallucination**: Grounded in retrieved facts
- **Cost Efficiency**: No need to retrain large models

### 1.3 RAG vs Fine-tuning vs Prompt Engineering

```python
# Comparison of approaches
class AIApproachComparison:
    """Compare different AI enhancement approaches"""
    
    def __init__(self):
        self.approaches = {
            "rag": {
                "cost": "Low",
                "flexibility": "High",
                "real_time_updates": True,
                "domain_knowledge": "Excellent",
                "transparency": "High",
                "latency": "Medium"
            },
            "fine_tuning": {
                "cost": "High",
                "flexibility": "Low",
                "real_time_updates": False,
                "domain_knowledge": "Excellent",
                "transparency": "Low",
                "latency": "Low"
            },
            "prompt_engineering": {
                "cost": "Very Low",
                "flexibility": "High",
                "real_time_updates": True,
                "domain_knowledge": "Limited",
                "transparency": "High",
                "latency": "Low"
            }
        }
    
    def compare_approaches(self):
        """Display comparison matrix"""
        import pandas as pd
        df = pd.DataFrame(self.approaches).T
        return df

# Usage
comparison = AIApproachComparison()
print(comparison.compare_approaches())
```

### 1.4 RAG Use Cases

**Enterprise Applications:**
- Customer support knowledge bases
- Internal documentation systems
- Legal document analysis
- Medical literature search
- Financial report analysis

**Content Applications:**
- News and media aggregation
- Research assistance
- Educational content generation
- Technical documentation

**Real-world Example:**
```python
class CustomerSupportRAG:
    """Customer support RAG system example"""
    
    def __init__(self):
        self.knowledge_base = [
            "Product manual sections",
            "FAQ database",
            "Previous support tickets",
            "Troubleshooting guides",
            "Company policies"
        ]
    
    def handle_customer_query(self, query: str) -> dict:
        """Process customer support query using RAG"""
        
        # Retrieve relevant support documents
        relevant_docs = self.retrieve_support_docs(query)
        
        # Generate contextual response
        response = self.generate_support_response(query, relevant_docs)
        
        # Track for continuous improvement
        self.log_interaction(query, response, relevant_docs)
        
        return {
            "response": response,
            "sources": [doc["source"] for doc in relevant_docs],
            "confidence": self.calculate_confidence(relevant_docs)
        }
    
    def retrieve_support_docs(self, query: str) -> list:
        # Implementation would use vector similarity search
        pass
    
    def generate_support_response(self, query: str, docs: list) -> str:
        # Implementation would use LLM with augmented context
        pass
```

---

## Chapter 2: RAG Architecture and Components

### 2.1 Core RAG Architecture

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Document:
    """Document representation"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = datetime.now()

@dataclass
class RetrievalResult:
    """Retrieval result with scoring"""
    document: Document
    score: float
    relevance_explanation: Optional[str] = None

class RAGComponent(ABC):
    """Abstract base class for RAG components"""
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data"""
        pass

class DocumentProcessor(RAGComponent):
    """Document processing and chunking component"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process(self, raw_documents: List[str]) -> List[Document]:
        """Process raw documents into structured chunks"""
        processed_docs = []
        
        for i, doc in enumerate(raw_documents):
            chunks = self.chunk_document(doc)
            for j, chunk in enumerate(chunks):
                processed_docs.append(Document(
                    id=f"doc_{i}_chunk_{j}",
                    content=chunk,
                    metadata={
                        "source_doc": i,
                        "chunk_index": j,
                        "chunk_size": len(chunk)
                    }
                ))
        
        return processed_docs
    
    def chunk_document(self, document: str) -> List[str]:
        """Split document into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(document):
            end = start + self.chunk_size
            chunk = document[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
            
            if start >= len(document):
                break
        
        return chunks

class EmbeddingGenerator(RAGComponent):
    """Embedding generation component"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    
    def process(self, documents: List[Document]) -> List[Document]:
        """Generate embeddings for documents"""
        texts = [doc.content for doc in documents]
        embeddings = self.model.encode(texts)
        
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        return documents

class VectorStore(RAGComponent):
    """Vector storage and retrieval component"""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.index = None
    
    def process(self, documents: List[Document]) -> None:
        """Store documents in vector database"""
        self.documents.extend(documents)
        self._build_index()
    
    def _build_index(self):
        """Build vector index for fast similarity search"""
        if not self.documents:
            return
        
        embeddings = np.array([doc.embedding for doc in self.documents])
        # Simple implementation - production would use FAISS, Pinecone, etc.
        self.index = embeddings
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve similar documents"""
        if self.index is None:
            return []
        
        # Calculate cosine similarity
        similarities = np.dot(self.index, query_embedding) / (
            np.linalg.norm(self.index, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=similarities[idx]
            ))
        
        return results

class ResponseGenerator(RAGComponent):
    """Response generation component"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client"""
        # Implementation would initialize OpenAI, Anthropic, or other LLM client
        pass
    
    def process(self, query: str, retrieved_docs: List[RetrievalResult]) -> str:
        """Generate response using retrieved context"""
        
        # Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)
        
        # Create augmented prompt
        prompt = self._create_prompt(query, context)
        
        # Generate response
        response = self._generate_response(prompt)
        
        return response
    
    def _prepare_context(self, retrieved_docs: List[RetrievalResult]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        for i, result in enumerate(retrieved_docs, 1):
            context_parts.append(f"Source {i}:\n{result.document.content}\n")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create augmented prompt"""
        return f"""
        Based on the following context, please answer the question. If the answer cannot be found in the context, say so clearly.

        Context:
        {context}

        Question: {query}

        Answer:
        """
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using LLM"""
        # Implementation would call actual LLM API
        return "Generated response based on context"

class RAGPipeline:
    """Complete RAG pipeline orchestrator"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.response_generator = ResponseGenerator()
    
    def index_documents(self, raw_documents: List[str]) -> None:
        """Index documents into the vector store"""
        
        print("Processing documents...")
        processed_docs = self.document_processor.process(raw_documents)
        
        print("Generating embeddings...")
        embedded_docs = self.embedding_generator.process(processed_docs)
        
        print("Storing in vector database...")
        self.vector_store.process(embedded_docs)
        
        print(f"Indexed {len(embedded_docs)} document chunks")
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        
        # Generate query embedding
        query_embedding = self.embedding_generator.model.encode([question])[0]
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.retrieve(query_embedding, top_k)
        
        # Generate response
        response = self.response_generator.process(question, retrieved_docs)
        
        return {
            "question": question,
            "answer": response,
            "sources": [
                {
                    "content": result.document.content[:200] + "...",
                    "score": result.score,
                    "metadata": result.document.metadata
                }
                for result in retrieved_docs
            ]
        }

# Usage example
def demonstrate_rag_pipeline():
    """Demonstrate RAG pipeline usage"""
    
    # Sample documents
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "RAG combines retrieval and generation to produce more accurate and contextual responses.",
        "Vector databases store high-dimensional vectors for efficient similarity search operations."
    ]
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Index documents
    rag.index_documents(documents)
    
    # Query the system
    result = rag.query("What is RAG and how does it work?")
    
    print("Question:", result["question"])
    print("Answer:", result["answer"])
    print("Sources:")
    for source in result["sources"]:
        print(f"  Score: {source['score']:.3f} - {source['content']}")

if __name__ == "__main__":
    demonstrate_rag_pipeline()
```

### 2.2 RAG System Design Patterns

```python
from enum import Enum
from typing import Protocol

class RAGPattern(Enum):
    """Common RAG architectural patterns"""
    BASIC = "basic"              # Simple retrieve + generate
    HIERARCHICAL = "hierarchical"  # Multi-level retrieval
    ITERATIVE = "iterative"      # Multiple retrieval rounds
    HYBRID = "hybrid"            # Multiple retrieval strategies
    AGENTIC = "agentic"         # Agent-based RAG

class RetrievalStrategy(Protocol):
    """Protocol for retrieval strategies"""
    
    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Retrieve relevant documents"""
        ...

class BasicRetrievalStrategy:
    """Basic semantic similarity retrieval"""
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Perform semantic similarity search"""
        query_embedding = self.embedding_generator.model.encode([query])[0]
        return self.vector_store.retrieve(query_embedding, top_k)

class HybridRetrievalStrategy:
    """Combines multiple retrieval methods"""
    
    def __init__(self, 
                 semantic_retriever: BasicRetrievalStrategy,
                 keyword_retriever: 'KeywordRetrievalStrategy',
                 weights: tuple = (0.7, 0.3)):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.semantic_weight, self.keyword_weight = weights
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Combine semantic and keyword retrieval"""
        
        # Get results from both methods
        semantic_results = self.semantic_retriever.retrieve(query, top_k * 2)
        keyword_results = self.keyword_retriever.retrieve(query, top_k * 2)
        
        # Combine and re-rank results
        combined_scores = {}
        
        # Weight semantic scores
        for result in semantic_results:
            doc_id = result.document.id
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + \
                                    result.score * self.semantic_weight
        
        # Weight keyword scores
        for result in keyword_results:
            doc_id = result.document.id
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + \
                                    result.score * self.keyword_weight
        
        # Create result documents mapping
        all_docs = {r.document.id: r.document for r in semantic_results + keyword_results}
        
        # Sort by combined score and return top-k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            RetrievalResult(document=all_docs[doc_id], score=score)
            for doc_id, score in sorted_results[:top_k]
            if doc_id in all_docs
        ]

class AdvancedRAGPipeline:
    """Advanced RAG pipeline with configurable strategies"""
    
    def __init__(self, 
                 pattern: RAGPattern = RAGPattern.BASIC,
                 retrieval_strategy: RetrievalStrategy = None):
        self.pattern = pattern
        self.retrieval_strategy = retrieval_strategy
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.response_generator = ResponseGenerator()
        
        # Initialize default strategy if none provided
        if not self.retrieval_strategy:
            self.retrieval_strategy = BasicRetrievalStrategy(
                self.vector_store, 
                self.embedding_generator
            )
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Query using the configured pattern"""
        
        if self.pattern == RAGPattern.BASIC:
            return self._basic_query(question, **kwargs)
        elif self.pattern == RAGPattern.ITERATIVE:
            return self._iterative_query(question, **kwargs)
        elif self.pattern == RAGPattern.HIERARCHICAL:
            return self._hierarchical_query(question, **kwargs)
        else:
            return self._basic_query(question, **kwargs)
    
    def _basic_query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Basic RAG query"""
        retrieved_docs = self.retrieval_strategy.retrieve(question, top_k)
        response = self.response_generator.process(question, retrieved_docs)
        
        return {
            "question": question,
            "answer": response,
            "sources": retrieved_docs,
            "pattern": "basic"
        }
    
    def _iterative_query(self, question: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Iterative RAG with refinement"""
        all_sources = []
        current_query = question
        
        for iteration in range(max_iterations):
            # Retrieve documents
            retrieved_docs = self.retrieval_strategy.retrieve(current_query, top_k=3)
            all_sources.extend(retrieved_docs)
            
            # Generate intermediate response
            intermediate_response = self.response_generator.process(current_query, retrieved_docs)
            
            # Check if we need more information
            if self._is_complete_answer(intermediate_response):
                break
            
            # Refine query for next iteration
            current_query = self._refine_query(question, intermediate_response)
        
        # Generate final response using all sources
        final_response = self.response_generator.process(question, all_sources)
        
        return {
            "question": question,
            "answer": final_response,
            "sources": all_sources,
            "iterations": iteration + 1,
            "pattern": "iterative"
        }
    
    def _hierarchical_query(self, question: str) -> Dict[str, Any]:
        """Hierarchical RAG with multi-level retrieval"""
        
        # Level 1: High-level topic retrieval
        topic_docs = self.retrieval_strategy.retrieve(question, top_k=10)
        
        # Level 2: Detailed information retrieval within topics
        detailed_docs = []
        for doc in topic_docs[:5]:  # Limit to top 5 topics
            # Retrieve more specific information
            specific_query = f"{question} {doc.document.content[:100]}"
            specific_docs = self.retrieval_strategy.retrieve(specific_query, top_k=3)
            detailed_docs.extend(specific_docs)
        
        # Generate response using hierarchical context
        all_sources = topic_docs + detailed_docs
        response = self.response_generator.process(question, all_sources)
        
        return {
            "question": question,
            "answer": response,
            "sources": {
                "topic_level": topic_docs,
                "detailed_level": detailed_docs
            },
            "pattern": "hierarchical"
        }
    
    def _is_complete_answer(self, response: str) -> bool:
        """Determine if response is complete (simplified heuristic)"""
        incomplete_indicators = [
            "I need more information",
            "incomplete",
            "partial",
            "cannot find",
            "unclear"
        ]
        return not any(indicator in response.lower() for indicator in incomplete_indicators)
    
    def _refine_query(self, original_query: str, partial_response: str) -> str:
        """Refine query based on partial response"""
        return f"{original_query} focusing on aspects not covered in: {partial_response[:100]}"
```

---

## Chapter 3: Embedding Models Deep Dive

### 3.1 Types of Embedding Models

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

class EmbeddingModelType(Enum):
    """Types of embedding models"""
    SENTENCE_TRANSFORMER = "sentence_transformer"
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGING_FACE = "hugging_face"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    model_name: str
    model_type: EmbeddingModelType
    dimension: int
    max_tokens: int
    cost_per_1k_tokens: float
    context_window: int
    normalization: bool = True
    batch_size: int = 32

class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Load the embedding model"""
        pass
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings"""
        pass
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text"""
        return self.encode([text])[0]
    
    def batch_encode(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Encode texts in batches"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)

class SentenceTransformerModel(EmbeddingModel):
    """Sentence Transformer embedding model implementation"""
    
    def _load_model(self):
        """Load Sentence Transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.model_name)
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Sentence Transformers"""
        embeddings = self.model.encode(
            texts, 
            normalize_embeddings=self.config.normalization,
            batch_size=self.config.batch_size
        )
        return np.array(embeddings)

class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model implementation"""
    
    def _load_model(self):
        """Initialize OpenAI client"""
        try:
            import openai
            self.client = openai.OpenAI()
        except ImportError:
            raise ImportError("Install OpenAI library: pip install openai")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenAI embeddings"""
        response = self.client.embeddings.create(
            model=self.config.model_name,
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

class CohereEmbeddingModel(EmbeddingModel):
    """Cohere embedding model implementation"""
    
    def _load_model(self):
        """Initialize Cohere client"""
        try:
            import cohere
            self.client = cohere.Client()
        except ImportError:
            raise ImportError("Install Cohere library: pip install cohere")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Cohere embeddings"""
        response = self.client.embed(
            texts=texts,
            model=self.config.model_name
        )
        return np.array(response.embeddings)

class HuggingFaceEmbeddingModel(EmbeddingModel):
    """Hugging Face transformers embedding model"""
    
    def _load_model(self):
        """Load Hugging Face model"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Hugging Face transformers"""
        import torch
        
        embeddings = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True,
                max_length=self.config.max_tokens
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding or mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)

# Embedding model factory
class EmbeddingModelFactory:
    """Factory for creating embedding models"""
    
    # Predefined model configurations
    MODEL_CONFIGS = {
        # Sentence Transformers
        "all-MiniLM-L6-v2": EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            dimension=384,
            max_tokens=256,
            cost_per_1k_tokens=0.0,  # Free
            context_window=256
        ),
        "all-mpnet-base-v2": EmbeddingConfig(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            dimension=768,
            max_tokens=384,
            cost_per_1k_tokens=0.0,  # Free
            context_window=384
        ),
        
        # OpenAI
        "text-embedding-ada-002": EmbeddingConfig(
            model_name="text-embedding-ada-002",
            model_type=EmbeddingModelType.OPENAI,
            dimension=1536,
            max_tokens=8191,
            cost_per_1k_tokens=0.0001,
            context_window=8191
        ),
        "text-embedding-3-small": EmbeddingConfig(
            model_name="text-embedding-3-small",
            model_type=EmbeddingModelType.OPENAI,
            dimension=1536,
            max_tokens=8191,
            cost_per_1k_tokens=0.00002,
            context_window=8191
        ),
        "text-embedding-3-large": EmbeddingConfig(
            model_name="text-embedding-3-large",
            model_type=EmbeddingModelType.OPENAI,
            dimension=3072,
            max_tokens=8191,
            cost_per_1k_tokens=0.00013,
            context_window=8191
        ),
        
        # Cohere
        "embed-english-v3.0": EmbeddingConfig(
            model_name="embed-english-v3.0",
            model_type=EmbeddingModelType.COHERE,
            dimension=1024,
            max_tokens=512,
            cost_per_1k_tokens=0.0001,
            context_window=512
        ),
        "embed-multilingual-v3.0": EmbeddingConfig(
            model_name="embed-multilingual-v3.0",
            model_type=EmbeddingModelType.COHERE,
            dimension=1024,
            max_tokens=512,
            cost_per_1k_tokens=0.0001,
            context_window=512
        )
    }
    
    @classmethod
    def create_model(cls, model_name: str) -> EmbeddingModel:
        """Create embedding model by name"""
        if model_name not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = cls.MODEL_CONFIGS[model_name]
        
        if config.model_type == EmbeddingModelType.SENTENCE_TRANSFORMER:
            return SentenceTransformerModel(config)
        elif config.model_type == EmbeddingModelType.OPENAI:
            return OpenAIEmbeddingModel(config)
        elif config.model_type == EmbeddingModelType.COHERE:
            return CohereEmbeddingModel(config)
        elif config.model_type == EmbeddingModelType.HUGGING_FACE:
            return HuggingFaceEmbeddingModel(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available models with their specifications"""
        models_info = {}
        
        for name, config in cls.MODEL_CONFIGS.items():
            models_info[name] = {
                "type": config.model_type.value,
                "dimension": config.dimension,
                "max_tokens": config.max_tokens,
                "cost_per_1k_tokens": config.cost_per_1k_tokens,
                "context_window": config.context_window,
                "free": config.cost_per_1k_tokens == 0.0
            }
        
        return models_info
```

### 3.2 Specialized Embedding Models

```python
class SpecializedEmbeddingModels:
    """Collection of specialized embedding models for different domains"""
    
    @staticmethod
    def get_code_embedding_models() -> Dict[str, EmbeddingConfig]:
        """Embedding models specialized for code"""
        return {
            "code-search-ada-code-001": EmbeddingConfig(
                model_name="code-search-ada-code-001",
                model_type=EmbeddingModelType.OPENAI,
                dimension=1024,
                max_tokens=8191,
                cost_per_1k_tokens=0.0001,
                context_window=8191
            ),
            "codet5p-110m": EmbeddingConfig(
                model_name="Salesforce/codet5p-110m-embedding",
                model_type=EmbeddingModelType.HUGGING_FACE,
                dimension=768,
                max_tokens=512,
                cost_per_1k_tokens=0.0,
                context_window=512
            ),
            "unixcoder-base": EmbeddingConfig(
                model_name="microsoft/unixcoder-base",
                model_type=EmbeddingModelType.HUGGING_FACE,
                dimension=768,
                max_tokens=1024,
                cost_per_1k_tokens=0.0,
                context_window=1024
            )
        }
    
    @staticmethod
    def get_multilingual_models() -> Dict[str, EmbeddingConfig]:
        """Multilingual embedding models"""
        return {
            "multilingual-e5-large": EmbeddingConfig(
                model_name="intfloat/multilingual-e5-large",
                model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
                dimension=1024,
                max_tokens=512,
                cost_per_1k_tokens=0.0,
                context_window=512
            ),
            "paraphrase-multilingual-mpnet-base-v2": EmbeddingConfig(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
                dimension=768,
                max_tokens=256,
                cost_per_1k_tokens=0.0,
                context_window=256
            ),
            "labse": EmbeddingConfig(
                model_name="sentence-transformers/LaBSE",
                model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
                dimension=768,
                max_tokens=256,
                cost_per_1k_tokens=0.0,
                context_window=256
            )
        }
    
    @staticmethod
    def get_domain_specific_models() -> Dict[str, Dict[str, EmbeddingConfig]]:
        """Domain-specific embedding models"""
        return {
            "scientific": {
                "scibert": EmbeddingConfig(
                    model_name="allenai/scibert_scivocab_uncased",
                    model_type=EmbeddingModelType.HUGGING_FACE,
                    dimension=768,
                    max_tokens=512,
                    cost_per_1k_tokens=0.0,
                    context_window=512
                ),
                "biobert": EmbeddingConfig(
                    model_name="dmis-lab/biobert-base-cased-v1.1",
                    model_type=EmbeddingModelType.HUGGING_FACE,
                    dimension=768,
                    max_tokens=512,
                    cost_per_1k_tokens=0.0,
                    context_window=512
                )
            },
            "legal": {
                "legal-bert": EmbeddingConfig(
                    model_name="nlpaueb/legal-bert-base-uncased",
                    model_type=EmbeddingModelType.HUGGING_FACE,
                    dimension=768,
                    max_tokens=512,
                    cost_per_1k_tokens=0.0,
                    context_window=512
                )
            },
            "financial": {
                "finbert": EmbeddingConfig(
                    model_name="ProsusAI/finbert",
                    model_type=EmbeddingModelType.HUGGING_FACE,
                    dimension=768,
                    max_tokens=512,
                    cost_per_1k_tokens=0.0,
                    context_window=512
                )
            }
        }

class EmbeddingEvaluator:
    """Evaluate and compare embedding models"""
    
    def __init__(self, test_queries: List[str], ground_truth: List[List[str]]):
        """
        Initialize evaluator
        
        Args:
            test_queries: List of test queries
            ground_truth: List of relevant documents for each query
        """
        self.test_queries = test_queries
        self.ground_truth = ground_truth
    
    def evaluate_model(self, model: EmbeddingModel, corpus: List[str]) -> Dict[str, float]:
        """Evaluate embedding model performance"""
        
        # Encode corpus
        print("Encoding corpus...")
        corpus_embeddings = model.batch_encode(corpus)
        
        # Evaluate each query
        all_precisions = []
        all_recalls = []
        all_f1s = []
        
        for i, query in enumerate(self.test_queries):
            print(f"Evaluating query {i+1}/{len(self.test_queries)}")
            
            # Encode query
            query_embedding = model.encode_single(query)
            
            # Calculate similarities
            similarities = np.dot(corpus_embeddings, query_embedding) / (
                np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k results
            top_k = len(self.ground_truth[i])
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Calculate precision, recall, F1
            retrieved_docs = [corpus[idx] for idx in top_indices]
            relevant_docs = self.ground_truth[i]
            
            precision, recall, f1 = self._calculate_metrics(retrieved_docs, relevant_docs)
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
        
        return {
            "precision": np.mean(all_precisions),
            "recall": np.mean(all_recalls),
            "f1": np.mean(all_f1s),
            "model_name": model.config.model_name,
            "dimension": model.config.dimension
        }
    
    def _calculate_metrics(self, retrieved: List[str], relevant: List[str]) -> tuple:
        """Calculate precision, recall, and F1 score"""
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        
        true_positives = len(retrieved_set.intersection(relevant_set))
        
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(relevant_set) if relevant_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def compare_models(self, model_names: List[str], corpus: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare multiple embedding models"""
        results = {}
        
        for model_name in model_names:
            print(f"\nEvaluating {model_name}...")
            try:
                model = EmbeddingModelFactory.create_model(model_name)
                results[model_name] = self.evaluate_model(model, corpus)
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results

# Usage example
def demonstrate_embedding_models():
    """Demonstrate different embedding models"""
    
    # List available models
    print("Available embedding models:")
    available_models = EmbeddingModelFactory.list_available_models()
    
    for name, info in available_models.items():
        print(f"  {name}:")
        print(f"    Type: {info['type']}")
        print(f"    Dimensions: {info['dimension']}")
        print(f"    Cost per 1K tokens: ${info['cost_per_1k_tokens']}")
        print(f"    Free: {info['free']}")
        print()
    
    # Test different models
    test_texts = [
        "Python is a programming language",
        "Machine learning algorithms",
        "Natural language processing",
        "Deep learning neural networks"
    ]
    
    # Compare free models
    free_models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    
    for model_name in free_models:
        try:
            print(f"\nTesting {model_name}:")
            model = EmbeddingModelFactory.create_model(model_name)
            embeddings = model.encode(test_texts)
            
            print(f"  Shape: {embeddings.shape}")
            print(f"  Sample embedding (first 5 dims): {embeddings[0][:5]}")
            
            # Calculate similarity between first two texts
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            print(f"  Similarity between first two texts: {similarity:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    demonstrate_embedding_models()
```

---

## Chapter 4: Vector Databases Comparison

### 4.1 Vector Database Landscape

Vector databases are specialized systems designed to store, index, and query high-dimensional vectors efficiently. Here's a comprehensive comparison of the leading vector database solutions:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time
import uuid

class VectorDBType(Enum):
    """Types of vector databases"""
    OPEN_SOURCE = "open_source"
    CLOUD_MANAGED = "cloud_managed"
    EMBEDDED = "embedded"
    TRADITIONAL_WITH_VECTOR = "traditional_with_vector"

@dataclass
class VectorDBConfig:
    """Configuration for vector databases"""
    name: str
    db_type: VectorDBType
    max_dimensions: int
    supported_metrics: List[str]
    supports_filtering: bool
    supports_hybrid_search: bool
    horizontal_scaling: bool
    persistence: bool
    cost_model: str
    ease_of_deployment: str  # easy, medium, complex
    performance_tier: str    # high, medium, low

@dataclass
class SearchResult:
    """Vector search result"""
    id: str
    score: float
    metadata: Dict[str, Any]
    vector: Optional[np.ndarray] = None

class VectorDatabase(ABC):
    """Abstract base class for vector databases"""
    
    def __init__(self, config: VectorDBConfig, **kwargs):
        self.config = config
        self.client = None
        self.index_name = kwargs.get('index_name', 'default')
        self._initialize_client(**kwargs)
    
    @abstractmethod
    def _initialize_client(self, **kwargs):
        """Initialize database client"""
        pass
    
    @abstractmethod
    def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """Create vector index"""
        pass
    
    @abstractmethod
    def insert_vectors(self, vectors: List[np.ndarray], ids: List[str], 
                      metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Insert vectors into database"""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        pass

# Pinecone Implementation
class PineconeDB(VectorDatabase):
    """Pinecone vector database implementation"""
    
    def _initialize_client(self, **kwargs):
        """Initialize Pinecone client"""
        try:
            import pinecone
            api_key = kwargs.get('api_key')
            environment = kwargs.get('environment', 'us-west1-gcp')
            
            pinecone.init(api_key=api_key, environment=environment)
            self.pinecone = pinecone
            
        except ImportError:
            raise ImportError("Install Pinecone: pip install pinecone-client")
    
    def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """Create Pinecone index"""
        try:
            if self.index_name not in self.pinecone.list_indexes():
                self.pinecone.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    pod_type="p1.x1"  # Starter pod
                )
                
                # Wait for index to be ready
                while not self.pinecone.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
            
            self.index = self.pinecone.Index(self.index_name)
            return True
            
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            return False
    
    def insert_vectors(self, vectors: List[np.ndarray], ids: List[str], 
                      metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Insert vectors into Pinecone"""
        try:
            upsert_data = []
            
            for i, (vector, vector_id) in enumerate(zip(vectors, ids)):
                item = {
                    "id": vector_id,
                    "values": vector.tolist()
                }
                if metadata and i < len(metadata):
                    item["metadata"] = metadata[i]
                
                upsert_data.append(item)
            
            # Batch upsert
            batch_size = 100
            for i in range(0, len(upsert_data), batch_size):
                batch = upsert_data[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            return True
            
        except Exception as e:
            print(f"Error inserting vectors: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search Pinecone index"""
        try:
            query_params = {
                "vector": query_vector.tolist(),
                "top_k": top_k,
                "include_metadata": True
            }
            
            if filters:
                query_params["filter"] = filters
            
            response = self.index.query(**query_params)
            
            results = []
            for match in response['matches']:
                results.append(SearchResult(
                    id=match['id'],
                    score=match['score'],
                    metadata=match.get('metadata', {})
                ))
            
            return results
            
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Pinecone"""
        try:
            self.index.delete(ids=ids)
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats['total_vector_count'],
                "dimension": stats['dimension'],
                "index_fullness": stats['index_fullness']
            }
        except Exception as e:
            return {"error": str(e)}

# Weaviate Implementation
class WeaviateDB(VectorDatabase):
    """Weaviate vector database implementation"""
    
    def _initialize_client(self, **kwargs):
        """Initialize Weaviate client"""
        try:
            import weaviate
            
            url = kwargs.get('url', 'http://localhost:8080')
            
            self.client = weaviate.Client(
                url=url,
                auth_client_secret=kwargs.get('auth_config'),
                additional_headers=kwargs.get('headers', {})
            )
            
            self.class_name = kwargs.get('class_name', 'Document')
            
        except ImportError:
            raise ImportError("Install Weaviate: pip install weaviate-client")
    
    def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """Create Weaviate class (schema)"""
        try:
            # Check if class exists
            if self.client.schema.exists(self.class_name):
                return True
            
            # Create class schema
            class_schema = {
                "class": self.class_name,
                "description": "Document vectors for RAG system",
                "vectorizer": "none",  # We'll provide our own vectors
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Document content"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                        "description": "Document metadata"
                    }
                ]
            }
            
            self.client.schema.create_class(class_schema)
            return True
            
        except Exception as e:
            print(f"Error creating Weaviate class: {e}")
            return False
    
    def insert_vectors(self, vectors: List[np.ndarray], ids: List[str], 
                      metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Insert vectors into Weaviate"""
        try:
            with self.client.batch as batch:
                batch.batch_size = 100
                
                for i, (vector, vector_id) in enumerate(zip(vectors, ids)):
                    properties = {
                        "content": metadata[i].get('content', '') if metadata else '',
                        "metadata": metadata[i] if metadata and i < len(metadata) else {}
                    }
                    
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.class_name,
                        uuid=vector_id,
                        vector=vector.tolist()
                    )
            
            return True
            
        except Exception as e:
            print(f"Error inserting vectors: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search Weaviate index"""
        try:
            query_builder = (
                self.client.query
                .get(self.class_name, ["content", "metadata"])
                .with_near_vector({
                    "vector": query_vector.tolist()
                })
                .with_limit(top_k)
                .with_additional(["distance", "id"])
            )
            
            if filters:
                # Add where filters
                query_builder = query_builder.with_where(filters)
            
            result = query_builder.do()
            
            results = []
            for item in result['data']['Get'][self.class_name]:
                results.append(SearchResult(
                    id=item['_additional']['id'],
                    score=1 - item['_additional']['distance'],  # Convert distance to similarity
                    metadata={
                        'content': item.get('content', ''),
                        **item.get('metadata', {})
                    }
                ))
            
            return results
            
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Weaviate"""
        try:
            for vector_id in ids:
                self.client.data_object.delete(vector_id)
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Weaviate statistics"""
        try:
            result = (
                self.client.query
                .aggregate(self.class_name)
                .with_meta_count()
                .do()
            )
            
            return {
                "total_vectors": result['data']['Aggregate'][self.class_name][0]['meta']['count']
            }
        except Exception as e:
            return {"error": str(e)}

# Chroma Implementation
class ChromaDB(VectorDatabase):
    """Chroma vector database implementation"""
    
    def _initialize_client(self, **kwargs):
        """Initialize Chroma client"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            if kwargs.get('persistent', True):
                self.client = chromadb.PersistentClient(
                    path=kwargs.get('path', './chroma_db')
                )
            else:
                self.client = chromadb.Client()
            
            self.collection = None
            
        except ImportError:
            raise ImportError("Install Chroma: pip install chromadb")
    
    def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """Create Chroma collection"""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.index_name,
                metadata={"hnsw:space": metric}
            )
            return True
        except Exception as e:
            print(f"Error creating Chroma collection: {e}")
            return False
    
    def insert_vectors(self, vectors: List[np.ndarray], ids: List[str], 
                      metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Insert vectors into Chroma"""
        try:
            embeddings = [vector.tolist() for vector in vectors]
            documents = []
            metadatas = []
            
            for i, vector_id in enumerate(ids):
                # Extract document content from metadata
                if metadata and i < len(metadata):
                    documents.append(metadata[i].get('content', ''))
                    metadatas.append(metadata[i])
                else:
                    documents.append('')
                    metadatas.append({})
            
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            return True
            
        except Exception as e:
            print(f"Error inserting vectors: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search Chroma collection"""
        try:
            query_params = {
                "query_embeddings": [query_vector.tolist()],
                "n_results": top_k
            }
            
            if filters:
                query_params["where"] = filters
            
            results = self.collection.query(**query_params)
            
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, (id_, distance, metadata) in enumerate(zip(
                    results['ids'][0],
                    results['distances'][0],
                    results['metadatas'][0]
                )):
                    search_results.append(SearchResult(
                        id=id_,
                        score=1 - distance,  # Convert distance to similarity
                        metadata=metadata or {}
                    ))
            
            return search_results
            
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Chroma"""
        try:
            self.collection.delete(ids=ids)
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Chroma collection statistics"""
        try:
            return {
                "total_vectors": self.collection.count(),
                "collection_name": self.collection.name
            }
        except Exception as e:
            return {"error": str(e)}

# FAISS Implementation
class FAISSDB(VectorDatabase):
    """FAISS vector database implementation"""
    
    def _initialize_client(self, **kwargs):
        """Initialize FAISS index"""
        try:
            import faiss
            self.faiss = faiss
            self.index = None
            self.id_to_metadata = {}
            self.vector_ids = []
            self.dimension = None
            
        except ImportError:
            raise ImportError("Install FAISS: pip install faiss-cpu or faiss-gpu")
    
    def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """Create FAISS index"""
        try:
            self.dimension = dimension
            
            if metric == "cosine":
                # Normalize vectors for cosine similarity
                self.index = self.faiss.IndexFlatIP(dimension)
            elif metric == "euclidean" or metric == "l2":
                self.index = self.faiss.IndexFlatL2(dimension)
            else:
                # Default to L2
                self.index = self.faiss.IndexFlatL2(dimension)
            
            return True
            
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            return False
    
    def insert_vectors(self, vectors: List[np.ndarray], ids: List[str], 
                      metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Insert vectors into FAISS"""
        try:
            vectors_array = np.array(vectors).astype('float32')
            
            # Normalize vectors if using cosine similarity
            if isinstance(self.index, self.faiss.IndexFlatIP):
                self.faiss.normalize_L2(vectors_array)
            
            self.index.add(vectors_array)
            
            # Store metadata and IDs
            for i, vector_id in enumerate(ids):
                self.vector_ids.append(vector_id)
                if metadata and i < len(metadata):
                    self.id_to_metadata[vector_id] = metadata[i]
                else:
                    self.id_to_metadata[vector_id] = {}
            
            return True
            
        except Exception as e:
            print(f"Error inserting vectors: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search FAISS index"""
        try:
            query_array = query_vector.reshape(1, -1).astype('float32')
            
            # Normalize query vector if using cosine similarity
            if isinstance(self.index, self.faiss.IndexFlatIP):
                self.faiss.normalize_L2(query_array)
            
            scores, indices = self.index.search(query_array, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.vector_ids):
                    vector_id = self.vector_ids[idx]
                    metadata = self.id_to_metadata.get(vector_id, {})
                    
                    # Apply filters if specified
                    if filters:
                        match = True
                        for key, value in filters.items():
                            if key not in metadata or metadata[key] != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    results.append(SearchResult(
                        id=vector_id,
                        score=float(score),
                        metadata=metadata
                    ))
            
            return results
            
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from FAISS (rebuild index without deleted vectors)"""
        try:
            # FAISS doesn't support direct deletion, so we rebuild
            print("Note: FAISS requires index rebuilding for deletions")
            
            # Remove from metadata and ID tracking
            for vector_id in ids:
                if vector_id in self.id_to_metadata:
                    del self.id_to_metadata[vector_id]
                if vector_id in self.vector_ids:
                    self.vector_ids.remove(vector_id)
            
            return True
            
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics"""
        try:
            return {
                "total_vectors": self.index.ntotal,
                "dimension": self.dimension,
                "is_trained": self.index.is_trained
            }
        except Exception as e:
            return {"error": str(e)}

# Database Comparison and Factory
class VectorDatabaseFactory:
    """Factory for creating vector database instances"""
    
    # Database configurations
    DATABASE_CONFIGS = {
        "pinecone": VectorDBConfig(
            name="Pinecone",
            db_type=VectorDBType.CLOUD_MANAGED,
            max_dimensions=20000,
            supported_metrics=["cosine", "euclidean", "dotproduct"],
            supports_filtering=True,
            supports_hybrid_search=False,
            horizontal_scaling=True,
            persistence=True,
            cost_model="Usage-based",
            ease_of_deployment="easy",
            performance_tier="high"
        ),
        "weaviate": VectorDBConfig(
            name="Weaviate",
            db_type=VectorDBType.OPEN_SOURCE,
            max_dimensions=65536,
            supported_metrics=["cosine", "dot", "l2-squared", "hamming", "manhattan"],
            supports_filtering=True,
            supports_hybrid_search=True,
            horizontal_scaling=True,
            persistence=True,
            cost_model="Self-hosted or cloud",
            ease_of_deployment="medium",
            performance_tier="high"
        ),
        "chroma": VectorDBConfig(
            name="Chroma",
            db_type=VectorDBType.EMBEDDED,
            max_dimensions=None,  # No explicit limit
            supported_metrics=["cosine", "l2", "ip"],
            supports_filtering=True,
            supports_hybrid_search=False,
            horizontal_scaling=False,
            persistence=True,
            cost_model="Free",
            ease_of_deployment="easy",
            performance_tier="medium"
        ),
        "faiss": VectorDBConfig(
            name="FAISS",
            db_type=VectorDBType.EMBEDDED,
            max_dimensions=None,  # No explicit limit
            supported_metrics=["cosine", "l2", "ip"],
            supports_filtering=False,
            supports_hybrid_search=False,
            horizontal_scaling=False,
            persistence=False,  # Requires manual save/load
            cost_model="Free",
            ease_of_deployment="medium",
            performance_tier="high"
        )
    }
    
    @classmethod
    def create_database(cls, db_name: str, **kwargs) -> VectorDatabase:
        """Create vector database instance"""
        db_name = db_name.lower()
        
        if db_name == "pinecone":
            return PineconeDB(cls.DATABASE_CONFIGS[db_name], **kwargs)
        elif db_name == "weaviate":
            return WeaviateDB(cls.DATABASE_CONFIGS[db_name], **kwargs)
        elif db_name == "chroma":
            return ChromaDB(cls.DATABASE_CONFIGS[db_name], **kwargs)
        elif db_name == "faiss":
            return FAISSDB(cls.DATABASE_CONFIGS[db_name], **kwargs)
        else:
            raise ValueError(f"Unsupported database: {db_name}")
    
    @classmethod
    def get_database_comparison(cls) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive database comparison"""
        comparison = {}
        
        for name, config in cls.DATABASE_CONFIGS.items():
            comparison[name] = {
                "type": config.db_type.value,
                "max_dimensions": config.max_dimensions,
                "metrics": config.supported_metrics,
                "filtering": config.supports_filtering,
                "hybrid_search": config.supports_hybrid_search,
                "scaling": config.horizontal_scaling,
                "persistence": config.persistence,
                "cost": config.cost_model,
                "deployment": config.ease_of_deployment,
                "performance": config.performance_tier
            }
        
        return comparison
    
    @classmethod
    def recommend_database(cls, requirements: Dict[str, Any]) -> List[str]:
        """Recommend databases based on requirements"""
        recommendations = []
        
        for name, config in cls.DATABASE_CONFIGS.items():
            score = 0
            
            # Check requirements
            if requirements.get('cloud_managed') and config.db_type == VectorDBType.CLOUD_MANAGED:
                score += 2
            elif requirements.get('self_hosted') and config.db_type == VectorDBType.OPEN_SOURCE:
                score += 2
            elif requirements.get('embedded') and config.db_type == VectorDBType.EMBEDDED:
                score += 2
            
            if requirements.get('filtering') and config.supports_filtering:
                score += 1
            
            if requirements.get('hybrid_search') and config.supports_hybrid_search:
                score += 1
            
            if requirements.get('scaling') and config.horizontal_scaling:
                score += 1
            
            if requirements.get('free') and config.cost_model == "Free":
                score += 1
            
            if requirements.get('high_performance') and config.performance_tier == "high":
                score += 1
            
            recommendations.append((name, score))
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, score in recommendations if score > 0]
```

### 4.2 Vector Database Performance Benchmarking

```python
import time
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import numpy as np

class VectorDBBenchmark:
    """Comprehensive vector database benchmarking suite"""
    
    def __init__(self, databases: Dict[str, VectorDatabase]):
        self.databases = databases
        self.results = {}
    
    def generate_test_data(self, num_vectors: int, dimension: int) -> Tuple[List[np.ndarray], List[str], List[Dict]]:
        """Generate synthetic test data"""
        vectors = []
        ids = []
        metadata = []
        
        for i in range(num_vectors):
            # Generate random normalized vector
            vector = np.random.randn(dimension).astype('float32')
            vector = vector / np.linalg.norm(vector)  # Normalize
            
            vectors.append(vector)
            ids.append(f"vec_{i}")
            metadata.append({
                "content": f"Test document {i}",
                "category": random.choice(["A", "B", "C"]),
                "timestamp": time.time()
            })
        
        return vectors, ids, metadata
    
    def benchmark_insertion(self, num_vectors: int, dimension: int) -> Dict[str, float]:
        """Benchmark insertion performance"""
        print(f"Benchmarking insertion: {num_vectors} vectors, {dimension}D")
        
        vectors, ids, metadata = self.generate_test_data(num_vectors, dimension)
        insertion_times = {}
        
        for db_name, db in self.databases.items():
            print(f"  Testing {db_name}...")
            
            # Create index
            db.create_index(dimension)
            
            # Measure insertion time
            start_time = time.time()
            success = db.insert_vectors(vectors, ids, metadata)
            end_time = time.time()
            
            if success:
                insertion_times[db_name] = end_time - start_time
                print(f"    Inserted in {insertion_times[db_name]:.2f}s")
            else:
                insertion_times[db_name] = float('inf')
                print(f"    Failed to insert")
        
        return insertion_times
    
    def benchmark_search(self, num_queries: int, top_k: int = 10) -> Dict[str, Dict[str, float]]:
        """Benchmark search performance"""
        print(f"Benchmarking search: {num_queries} queries, top_k={top_k}")
        
        search_results = {}
        
        # Generate random query vectors
        dimension = 384  # Assume standard dimension
        query_vectors = []
        for _ in range(num_queries):
            vector = np.random.randn(dimension).astype('float32')
            vector = vector / np.linalg.norm(vector)
            query_vectors.append(vector)
        
        for db_name, db in self.databases.items():
            print(f"  Testing {db_name}...")
            
            search_times = []
            total_results = 0
            
            for query_vector in query_vectors:
                start_time = time.time()
                results = db.search(query_vector, top_k)
                end_time = time.time()
                
                search_times.append(end_time - start_time)
                total_results += len(results)
            
            avg_search_time = np.mean(search_times)
            p95_search_time = np.percentile(search_times, 95)
            avg_results = total_results / num_queries
            
            search_results[db_name] = {
                "avg_search_time": avg_search_time,
                "p95_search_time": p95_search_time,
                "avg_results_returned": avg_results
            }
            
            print(f"    Avg search time: {avg_search_time:.4f}s")
            print(f"    P95 search time: {p95_search_time:.4f}s")
        
        return search_results
    
    def benchmark_memory_usage(self) -> Dict[str, Dict[str, Any]]:
        """Benchmark memory usage (simplified)"""
        import psutil
        import os
        
        memory_results = {}
        
        for db_name, db in self.databases.items():
            process = psutil.Process(os.getpid())
            
            # Measure memory before
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Get database stats
            stats = db.get_stats()
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_results[db_name] = {
                "memory_usage_mb": memory_after - memory_before,
                "total_vectors": stats.get("total_vectors", 0),
                "memory_per_vector": (memory_after - memory_before) / max(stats.get("total_vectors", 1), 1)
            }
        
        return memory_results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        print("=== Vector Database Benchmark Suite ===\n")
        
        benchmark_results = {}
        
        # Test different vector sizes
        test_configs = [
            {"num_vectors": 1000, "dimension": 384},
            {"num_vectors": 10000, "dimension": 384},
            {"num_vectors": 1000, "dimension": 1536}
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\n--- Test {i+1}: {config['num_vectors']} vectors, {config['dimension']}D ---")
            
            # Benchmark insertion
            insertion_times = self.benchmark_insertion(
                config["num_vectors"], 
                config["dimension"]
            )
            
            # Benchmark search
            search_results = self.benchmark_search(100)  # 100 queries
            
            # Store results
            benchmark_results[f"test_{i+1}"] = {
                "config": config,
                "insertion_times": insertion_times,
                "search_results": search_results
            }
        
        # Memory usage
        print("\n--- Memory Usage ---")
        memory_results = self.benchmark_memory_usage()
        benchmark_results["memory_usage"] = memory_results
        
        return benchmark_results
    
    def plot_results(self, results: Dict[str, Any]):
        """Plot benchmark results"""
        try:
            import matplotlib.pyplot as plt
            
            # Plot insertion times
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Insertion times comparison
            db_names = list(self.databases.keys())
            insertion_times = []
            
            for test_name, test_data in results.items():
                if test_name.startswith("test_"):
                    insertion_data = test_data["insertion_times"]
                    for db_name in db_names:
                        if db_name in insertion_data:
                            insertion_times.append(insertion_data[db_name])
            
            if insertion_times:
                axes[0, 0].bar(db_names, insertion_times)
                axes[0, 0].set_title("Insertion Times")
                axes[0, 0].set_ylabel("Time (seconds)")
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Search times comparison
            search_times = []
            for db_name in db_names:
                if "test_1" in results and "search_results" in results["test_1"]:
                    search_data = results["test_1"]["search_results"]
                    if db_name in search_data:
                        search_times.append(search_data[db_name]["avg_search_time"])
                    else:
                        search_times.append(0)
            
            if search_times:
                axes[0, 1].bar(db_names, search_times)
                axes[0, 1].set_title("Average Search Times")
                axes[0, 1].set_ylabel("Time (seconds)")
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Memory usage
            if "memory_usage" in results:
                memory_data = results["memory_usage"]
                memory_values = [memory_data[db]["memory_usage_mb"] for db in db_names if db in memory_data]
                
                if memory_values:
                    axes[1, 0].bar(db_names, memory_values)
                    axes[1, 0].set_title("Memory Usage")
                    axes[1, 0].set_ylabel("Memory (MB)")
                    axes[1, 0].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('vector_db_benchmark.png')
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")

# Example usage and comparison
def demonstrate_vector_databases():
    """Demonstrate vector database comparison"""
    
    print("=== Vector Database Comparison ===")
    
    # Show database comparison table
    comparison = VectorDatabaseFactory.get_database_comparison()
    
    print("\nDatabase Comparison:")
    print("-" * 80)
    print(f"{'Database':<12} {'Type':<15} {'Filtering':<10} {'Scaling':<10} {'Cost':<15}")
    print("-" * 80)
    
    for db_name, info in comparison.items():
        print(f"{db_name:<12} {info['type']:<15} {str(info['filtering']):<10} {str(info['scaling']):<10} {info['cost']:<15}")
    
    # Database recommendations
    print("\n=== Database Recommendations ===")
    
    test_requirements = [
        {"name": "Startup MVP", "requirements": {"embedded": True, "free": True}},
        {"name": "Enterprise Scale", "requirements": {"cloud_managed": True, "scaling": True, "high_performance": True}},
        {"name": "Self-hosted", "requirements": {"self_hosted": True, "filtering": True, "hybrid_search": True}},
        {"name": "Research Project", "requirements": {"free": True, "high_performance": True}}
    ]
    
    for test in test_requirements:
        recommendations = VectorDatabaseFactory.recommend_database(test["requirements"])
        print(f"\n{test['name']}:")
        print(f"  Requirements: {test['requirements']}")
        print(f"  Recommended: {', '.join(recommendations[:3])}")
    
    # Benchmark (using only free/embedded databases for demo)
    print("\n=== Running Benchmarks (Free Databases Only) ===")
    
    try:
        # Initialize free databases
        databases = {
            "chroma": VectorDatabaseFactory.create_database("chroma", persistent=False),
            "faiss": VectorDatabaseFactory.create_database("faiss")
        }
        
        # Run benchmark
        benchmark = VectorDBBenchmark(databases)
        results = benchmark.run_comprehensive_benchmark()
        
        print("\nBenchmark completed successfully!")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        print("Make sure required libraries are installed:")
        print("  pip install chromadb faiss-cpu")

if __name__ == "__main__":
    demonstrate_vector_databases()
```

## Chapter 5: Document Processing and Chunking Strategies

Document processing and chunking are critical components of RAG systems that significantly impact retrieval accuracy and response quality. This chapter covers comprehensive strategies for preparing documents for vector storage.

### 5.1 Document Processing Pipeline

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from pathlib import Path
import logging

class DocumentType(Enum):
    """Supported document types"""
    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    DOCX = "docx"
    JSON = "json"
    CSV = "csv"
    XML = "xml"

@dataclass
class Document:
    """Document representation with metadata"""
    content: str
    doc_id: str
    doc_type: DocumentType
    source: str
    metadata: Dict[str, Any]
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    
    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        source_hash = hashlib.md5(self.source.encode()).hexdigest()[:8]
        return f"doc_{source_hash}_{content_hash}"
    
    def get_word_count(self) -> int:
        """Get word count of document"""
        return len(self.content.split())
    
    def get_char_count(self) -> int:
        """Get character count of document"""
        return len(self.content)

@dataclass
class DocumentChunk:
    """Document chunk with positioning and metadata"""
    content: str
    chunk_id: str
    doc_id: str
    chunk_index: int
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    overlap_with_previous: int = 0
    overlap_with_next: int = 0
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"{self.doc_id}_chunk_{self.chunk_index}"
    
    def get_word_count(self) -> int:
        """Get word count of chunk"""
        return len(self.content.split())
    
    def get_char_count(self) -> int:
        """Get character count of chunk"""
        return len(self.content)

class DocumentProcessor(ABC):
    """Abstract base class for document processors"""
    
    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """Check if processor can handle the file type"""
        pass
    
    @abstractmethod
    def process(self, file_path: str, **kwargs) -> Document:
        """Process document and return Document object"""
        pass

class TextDocumentProcessor(DocumentProcessor):
    """Processor for plain text documents"""
    
    def can_process(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in ['.txt', '.text']
    
    def process(self, file_path: str, encoding: str = 'utf-8', **kwargs) -> Document:
        """Process text document"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return Document(
                content=content,
                doc_id="",  # Will be auto-generated
                doc_type=DocumentType.TEXT,
                source=file_path,
                metadata={
                    "encoding": encoding,
                    "file_size": Path(file_path).stat().st_size,
                    "line_count": len(content.splitlines())
                }
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process text document {file_path}: {e}")

class PDFDocumentProcessor(DocumentProcessor):
    """Processor for PDF documents"""
    
    def can_process(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == '.pdf'
    
    def process(self, file_path: str, **kwargs) -> Document:
        """Process PDF document"""
        try:
            import PyPDF2
            
            content = ""
            metadata = {"pages": 0}
            
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                metadata["pages"] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    content += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            return Document(
                content=content.strip(),
                doc_id="",  # Will be auto-generated
                doc_type=DocumentType.PDF,
                source=file_path,
                metadata=metadata
            )
        except ImportError:
            raise RuntimeError("PyPDF2 not installed. Install with: pip install PyPDF2")
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF document {file_path}: {e}")

class HTMLDocumentProcessor(DocumentProcessor):
    """Processor for HTML documents"""
    
    def can_process(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in ['.html', '.htm']
    
    def process(self, file_path: str, **kwargs) -> Document:
        """Process HTML document"""
        try:
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            meta_description = soup.find('meta', attrs={'name': 'description'})
            description = meta_description.get('content', '') if meta_description else ""
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            content = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            return Document(
                content=content,
                doc_id="",  # Will be auto-generated
                doc_type=DocumentType.HTML,
                source=file_path,
                metadata={
                    "title": title_text,
                    "description": description,
                    "original_size": len(html_content),
                    "processed_size": len(content)
                }
            )
        except ImportError:
            raise RuntimeError("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
        except Exception as e:
            raise RuntimeError(f"Failed to process HTML document {file_path}: {e}")

class MarkdownDocumentProcessor(DocumentProcessor):
    """Processor for Markdown documents"""
    
    def can_process(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in ['.md', '.markdown']
    
    def process(self, file_path: str, preserve_structure: bool = True, **kwargs) -> Document:
        """Process Markdown document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = self._extract_frontmatter(content)
            
            if not preserve_structure:
                # Convert to plain text
                try:
                    import markdown
                    from bs4 import BeautifulSoup
                    
                    html = markdown.markdown(content)
                    soup = BeautifulSoup(html, 'html.parser')
                    content = soup.get_text()
                except ImportError:
                    # Fallback: simple regex-based conversion
                    content = self._simple_markdown_to_text(content)
            
            return Document(
                content=content,
                doc_id="",  # Will be auto-generated
                doc_type=DocumentType.MARKDOWN,
                source=file_path,
                metadata=metadata
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process Markdown document {file_path}: {e}")
    
    def _extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from markdown"""
        metadata = {}
        
        if content.startswith('---'):
            try:
                import yaml
                
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = parts[1].strip()
                    metadata = yaml.safe_load(frontmatter) or {}
                    # Remove frontmatter from content
                    content = parts[2].strip()
            except ImportError:
                pass
        
        return metadata
    
    def _simple_markdown_to_text(self, content: str) -> str:
        """Simple markdown to text conversion"""
        # Remove frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                content = parts[2].strip()
        
        # Remove markdown syntax
        content = re.sub(r'#+\s*', '', content)  # Headers
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Bold
        content = re.sub(r'\*(.*?)\*', r'\1', content)  # Italic
        content = re.sub(r'`([^`]+)`', r'\1', content)  # Inline code
        content = re.sub(r'```[\s\S]*?```', '', content)  # Code blocks
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)  # Links
        
        return content

class DocumentProcessorFactory:
    """Factory for creating appropriate document processors"""
    
    _processors = [
        TextDocumentProcessor(),
        PDFDocumentProcessor(),
        HTMLDocumentProcessor(),
        MarkdownDocumentProcessor(),
    ]
    
    @classmethod
    def get_processor(cls, file_path: str) -> Optional[DocumentProcessor]:
        """Get appropriate processor for file type"""
        for processor in cls._processors:
            if processor.can_process(file_path):
                return processor
        return None
    
    @classmethod
    def process_document(cls, file_path: str, **kwargs) -> Document:
        """Process document using appropriate processor"""
        processor = cls.get_processor(file_path)
        if not processor:
            raise ValueError(f"No processor available for file: {file_path}")
        
        return processor.process(file_path, **kwargs)
```

### 5.2 Chunking Strategies

```python
from abc import ABC, abstractmethod
import tiktoken
from typing import List, Dict, Any, Optional, Callable
import spacy
from dataclasses import dataclass

class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"
    SEMANTIC_BASED = "semantic_based"
    HIERARCHICAL = "hierarchical"
    SLIDING_WINDOW = "sliding_window"

@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies"""
    strategy: ChunkingStrategy
    chunk_size: int = 512  # characters or tokens
    overlap: int = 50  # characters or tokens
    use_tokens: bool = False  # Use tokens instead of characters
    min_chunk_size: int = 50  # Minimum chunk size
    max_chunk_size: int = 2000  # Maximum chunk size
    respect_sentence_boundaries: bool = True
    preserve_formatting: bool = False
    custom_separators: Optional[List[str]] = None

class DocumentChunker(ABC):
    """Abstract base class for document chunkers"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        if self.config.use_tokens:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    
    @abstractmethod
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document into smaller pieces"""
        pass
    
    def _count_units(self, text: str) -> int:
        """Count characters or tokens based on configuration"""
        if self.config.use_tokens:
            return len(self.tokenizer.encode(text))
        return len(text)
    
    def _create_chunk(self, content: str, doc_id: str, chunk_index: int, 
                      start_pos: int, end_pos: int, metadata: Dict[str, Any]) -> DocumentChunk:
        """Create a document chunk"""
        return DocumentChunk(
            content=content,
            chunk_id="",  # Will be auto-generated
            doc_id=doc_id,
            chunk_index=chunk_index,
            start_pos=start_pos,
            end_pos=end_pos,
            metadata=metadata
        )

class FixedSizeChunker(DocumentChunker):
    """Fixed-size chunking strategy"""
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document into fixed-size pieces"""
        content = document.content
        chunks = []
        
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap
        
        start = 0
        chunk_index = 0
        
        while start < len(content):
            # Calculate end position
            end = start + chunk_size
            
            if end >= len(content):
                end = len(content)
                chunk_content = content[start:end]
            else:
                # Respect sentence boundaries if configured
                if self.config.respect_sentence_boundaries:
                    chunk_content = content[start:end]
                    
                    # Find the last sentence ending
                    last_period = chunk_content.rfind('.')
                    last_exclamation = chunk_content.rfind('!')
                    last_question = chunk_content.rfind('?')
                    
                    last_sentence_end = max(last_period, last_exclamation, last_question)
                    
                    if last_sentence_end > len(chunk_content) * 0.5:  # Don't make chunks too small
                        end = start + last_sentence_end + 1
                        chunk_content = content[start:end]
                    else:
                        chunk_content = content[start:end]
                else:
                    chunk_content = content[start:end]
            
            # Create chunk
            if len(chunk_content.strip()) >= self.config.min_chunk_size:
                chunk = self._create_chunk(
                    content=chunk_content.strip(),
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    start_pos=start,
                    end_pos=end,
                    metadata={
                        **document.metadata,
                        "chunk_strategy": "fixed_size",
                        "chunk_size": len(chunk_content),
                        "word_count": len(chunk_content.split())
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position (with overlap)
            start = max(start + chunk_size - overlap, end)
            
            if start >= len(content):
                break
        
        return chunks

class SentenceBasedChunker(DocumentChunker):
    """Sentence-based chunking strategy"""
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document by sentences"""
        content = document.content
        doc = self.nlp(content)
        
        sentences = [sent.text for sent in doc.sents]
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence_size = self._count_units(sentence)
            
            # Check if adding this sentence exceeds chunk size
            if current_size + sentence_size > self.config.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_content = ' '.join(current_chunk)
                chunk = self._create_chunk(
                    content=chunk_content,
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_content),
                    metadata={
                        **document.metadata,
                        "chunk_strategy": "sentence_based",
                        "sentence_count": len(current_chunk),
                        "chunk_size": len(chunk_content),
                        "word_count": len(chunk_content.split())
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Handle overlap
                if self.config.overlap > 0 and len(current_chunk) > 1:
                    overlap_sentences = current_chunk[-self.config.overlap:]
                    current_chunk = overlap_sentences + [sentence]
                    current_size = sum(self._count_units(s) for s in current_chunk)
                    start_pos = start_pos + len(' '.join(current_chunk[:-len(overlap_sentences)-1])) + 1
                else:
                    current_chunk = [sentence]
                    current_size = sentence_size
                    start_pos = start_pos + len(chunk_content) + 1
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Handle remaining sentences
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            if len(chunk_content.strip()) >= self.config.min_chunk_size:
                chunk = self._create_chunk(
                    content=chunk_content,
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_content),
                    metadata={
                        **document.metadata,
                        "chunk_strategy": "sentence_based",
                        "sentence_count": len(current_chunk),
                        "chunk_size": len(chunk_content),
                        "word_count": len(chunk_content.split())
                    }
                )
                chunks.append(chunk)
        
        return chunks

class ParagraphBasedChunker(DocumentChunker):
    """Paragraph-based chunking strategy"""
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document by paragraphs"""
        content = document.content
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        start_pos = 0
        
        for paragraph in paragraphs:
            paragraph_size = self._count_units(paragraph)
            
            # If single paragraph exceeds max size, split it further
            if paragraph_size > self.config.max_chunk_size:
                # Use sentence-based chunking for large paragraphs
                temp_doc = Document(
                    content=paragraph,
                    doc_id=f"{document.doc_id}_temp",
                    doc_type=document.doc_type,
                    source=document.source,
                    metadata=document.metadata
                )
                
                sentence_chunker = SentenceBasedChunker(self.config)
                para_chunks = sentence_chunker.chunk(temp_doc)
                
                for para_chunk in para_chunks:
                    para_chunk.doc_id = document.doc_id
                    para_chunk.chunk_index = chunk_index
                    para_chunk.start_pos = start_pos + para_chunk.start_pos
                    para_chunk.end_pos = start_pos + para_chunk.end_pos
                    chunks.append(para_chunk)
                    chunk_index += 1
                
                start_pos += len(paragraph) + 2  # +2 for \n\n
                continue
            
            # Check if adding this paragraph exceeds chunk size
            if current_size + paragraph_size > self.config.chunk_size and current_chunk:
                # Create chunk from current paragraphs
                chunk_content = '\n\n'.join(current_chunk)
                chunk = self._create_chunk(
                    content=chunk_content,
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_content),
                    metadata={
                        **document.metadata,
                        "chunk_strategy": "paragraph_based",
                        "paragraph_count": len(current_chunk),
                        "chunk_size": len(chunk_content),
                        "word_count": len(chunk_content.split())
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Handle overlap
                if self.config.overlap > 0 and len(current_chunk) > 1:
                    overlap_paragraphs = current_chunk[-self.config.overlap:]
                    current_chunk = overlap_paragraphs + [paragraph]
                    current_size = sum(self._count_units(p) for p in current_chunk)
                    start_pos = start_pos + len('\n\n'.join(current_chunk[:-len(overlap_paragraphs)-1])) + 4
                else:
                    current_chunk = [paragraph]
                    current_size = paragraph_size
                    start_pos = start_pos + len(chunk_content) + 4
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size
        
        # Handle remaining paragraphs
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            if len(chunk_content.strip()) >= self.config.min_chunk_size:
                chunk = self._create_chunk(
                    content=chunk_content,
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_content),
                    metadata={
                        **document.metadata,
                        "chunk_strategy": "paragraph_based",
                        "paragraph_count": len(current_chunk),
                        "chunk_size": len(chunk_content),
                        "word_count": len(chunk_content.split())
                    }
                )
                chunks.append(chunk)
        
        return chunks

class SemanticChunker(DocumentChunker):
    """Semantic-based chunking using sentence embeddings"""
    
    def __init__(self, config: ChunkingConfig, embedding_model=None):
        super().__init__(config)
        self.embedding_model = embedding_model
        if not self.embedding_model:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                raise RuntimeError("sentence-transformers not installed. Install with: pip install sentence-transformers")
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document based on semantic similarity"""
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # First, split into sentences
            sentence_chunker = SentenceBasedChunker(self.config)
            sentence_chunks = sentence_chunker.chunk(document)
            
            if len(sentence_chunks) <= 1:
                return sentence_chunks
            
            sentences = [chunk.content for chunk in sentence_chunks]
            
            # Get sentence embeddings
            embeddings = self.embedding_model.encode(sentences)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find semantic boundaries (low similarity points)
            boundaries = [0]
            threshold = 0.5  # Similarity threshold for splitting
            
            for i in range(1, len(sentences)):
                if similarity_matrix[i-1][i] < threshold:
                    boundaries.append(i)
            
            boundaries.append(len(sentences))
            
            # Create semantic chunks
            chunks = []
            chunk_index = 0
            
            for i in range(len(boundaries) - 1):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]
                
                chunk_sentences = sentences[start_idx:end_idx]
                chunk_content = ' '.join(chunk_sentences)
                
                # Check if chunk size is within limits
                if self._count_units(chunk_content) > self.config.max_chunk_size:
                    # If too large, fall back to sentence-based chunking
                    for j in range(start_idx, end_idx):
                        if len(sentences[j].strip()) >= self.config.min_chunk_size:
                            chunk = self._create_chunk(
                                content=sentences[j],
                                doc_id=document.doc_id,
                                chunk_index=chunk_index,
                                start_pos=sentence_chunks[j].start_pos,
                                end_pos=sentence_chunks[j].end_pos,
                                metadata={
                                    **document.metadata,
                                    "chunk_strategy": "semantic_based",
                                    "semantic_boundary": True,
                                    "chunk_size": len(sentences[j]),
                                    "word_count": len(sentences[j].split())
                                }
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                else:
                    if len(chunk_content.strip()) >= self.config.min_chunk_size:
                        chunk = self._create_chunk(
                            content=chunk_content,
                            doc_id=document.doc_id,
                            chunk_index=chunk_index,
                            start_pos=sentence_chunks[start_idx].start_pos,
                            end_pos=sentence_chunks[end_idx-1].end_pos,
                            metadata={
                                **document.metadata,
                                "chunk_strategy": "semantic_based",
                                "semantic_boundary": True,
                                "sentence_count": len(chunk_sentences),
                                "chunk_size": len(chunk_content),
                                "word_count": len(chunk_content.split())
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
            
            return chunks
            
        except ImportError:
            raise RuntimeError("Required libraries not installed. Install with: pip install numpy scikit-learn")

class HierarchicalChunker(DocumentChunker):
    """Hierarchical chunking that preserves document structure"""
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document hierarchically"""
        content = document.content
        
        # Detect document structure (headers, sections, etc.)
        sections = self._detect_sections(content)
        
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_chunks = self._chunk_section(section, document, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        return chunks
    
    def _detect_sections(self, content: str) -> List[Dict[str, Any]]:
        """Detect document sections based on headers"""
        sections = []
        lines = content.split('\n')
        
        current_section = {
            "title": "",
            "level": 0,
            "content": [],
            "start_line": 0
        }
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect markdown headers
            if line.startswith('#'):
                # Save previous section
                if current_section["content"]:
                    sections.append(current_section)
                
                # Start new section
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                
                current_section = {
                    "title": title,
                    "level": level,
                    "content": [],
                    "start_line": i
                }
            else:
                current_section["content"].append(line)
        
        # Add final section
        if current_section["content"]:
            sections.append(current_section)
        
        return sections
    
    def _chunk_section(self, section: Dict[str, Any], document: Document, start_index: int) -> List[DocumentChunk]:
        """Chunk a document section"""
        section_content = '\n'.join(section["content"]).strip()
        
        if not section_content:
            return []
        
        # Create a temporary document for this section
        temp_doc = Document(
            content=section_content,
            doc_id=f"{document.doc_id}_section",
            doc_type=document.doc_type,
            source=document.source,
            metadata={
                **document.metadata,
                "section_title": section["title"],
                "section_level": section["level"]
            }
        )
        
        # Use paragraph-based chunking for sections
        paragraph_chunker = ParagraphBasedChunker(self.config)
        section_chunks = paragraph_chunker.chunk(temp_doc)
        
        # Update chunk metadata and indices
        for i, chunk in enumerate(section_chunks):
            chunk.doc_id = document.doc_id
            chunk.chunk_index = start_index + i
            chunk.metadata.update({
                "chunk_strategy": "hierarchical",
                "section_title": section["title"],
                "section_level": section["level"]
            })
        
        return section_chunks

class DocumentChunkerFactory:
    """Factory for creating document chunkers"""
    
    @staticmethod
    def create_chunker(strategy: ChunkingStrategy, config: ChunkingConfig, **kwargs) -> DocumentChunker:
        """Create appropriate chunker based on strategy"""
        if strategy == ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunker(config)
        elif strategy == ChunkingStrategy.SENTENCE_BASED:
            return SentenceBasedChunker(config)
        elif strategy == ChunkingStrategy.PARAGRAPH_BASED:
            return ParagraphBasedChunker(config)
        elif strategy == ChunkingStrategy.SEMANTIC_BASED:
            return SemanticChunker(config, kwargs.get('embedding_model'))
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            return HierarchicalChunker(config)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
```

### 5.3 Advanced Document Processing Pipeline

```python
from typing import List, Dict, Any, Optional, Callable
import asyncio
import concurrent.futures
from dataclasses import dataclass, field
import logging
from pathlib import Path

@dataclass
class ProcessingPipeline:
    """Complete document processing pipeline"""
    processors: List[DocumentProcessor] = field(default_factory=list)
    chunkers: List[DocumentChunker] = field(default_factory=list)
    validators: List[Callable[[Document], bool]] = field(default_factory=list)
    transformers: List[Callable[[Document], Document]] = field(default_factory=list)
    filters: List[Callable[[DocumentChunk], bool]] = field(default_factory=list)
    
    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
    
    def add_processor(self, processor: DocumentProcessor) -> 'ProcessingPipeline':
        """Add document processor to pipeline"""
        self.processors.append(processor)
        return self
    
    def add_chunker(self, chunker: DocumentChunker) -> 'ProcessingPipeline':
        """Add chunker to pipeline"""
        self.chunkers.append(chunker)
        return self
    
    def add_validator(self, validator: Callable[[Document], bool]) -> 'ProcessingPipeline':
        """Add document validator"""
        self.validators.append(validator)
        return self
    
    def add_transformer(self, transformer: Callable[[Document], Document]) -> 'ProcessingPipeline':
        """Add document transformer"""
        self.transformers.append(transformer)
        return self
    
    def add_filter(self, filter_func: Callable[[DocumentChunk], bool]) -> 'ProcessingPipeline':
        """Add chunk filter"""
        self.filters.append(filter_func)
        return self
    
    def process_file(self, file_path: str, **kwargs) -> List[DocumentChunk]:
        """Process a single file through the pipeline"""
        try:
            # Step 1: Process document
            document = DocumentProcessorFactory.process_document(file_path, **kwargs)
            self.logger.info(f"Processed document: {file_path}")
            
            # Step 2: Validate document
            for validator in self.validators:
                if not validator(document):
                    self.logger.warning(f"Document validation failed: {file_path}")
                    return []
            
            # Step 3: Transform document
            for transformer in self.transformers:
                document = transformer(document)
            
            # Step 4: Chunk document
            all_chunks = []
            for chunker in self.chunkers:
                chunks = chunker.chunk(document)
                all_chunks.extend(chunks)
            
            # If no chunkers specified, create default chunking
            if not self.chunkers:
                default_config = ChunkingConfig(
                    strategy=ChunkingStrategy.PARAGRAPH_BASED,
                    chunk_size=512,
                    overlap=50
                )
                default_chunker = DocumentChunkerFactory.create_chunker(
                    ChunkingStrategy.PARAGRAPH_BASED, 
                    default_config
                )
                all_chunks = default_chunker.chunk(document)
            
            # Step 5: Filter chunks
            filtered_chunks = []
            for chunk in all_chunks:
                passed_filters = True
                for filter_func in self.filters:
                    if not filter_func(chunk):
                        passed_filters = False
                        break
                
                if passed_filters:
                    filtered_chunks.append(chunk)
            
            self.logger.info(f"Created {len(filtered_chunks)} chunks from {file_path}")
            return filtered_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {e}")
            return []
    
    def process_directory(self, directory_path: str, recursive: bool = True, 
                         max_workers: int = 4, **kwargs) -> List[DocumentChunk]:
        """Process all files in a directory"""
        directory = Path(directory_path)
        
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Find all processable files
        pattern = "**/*" if recursive else "*"
        all_files = []
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                processor = DocumentProcessorFactory.get_processor(str(file_path))
                if processor:
                    all_files.append(str(file_path))
        
        self.logger.info(f"Found {len(all_files)} processable files")
        
        # Process files in parallel
        all_chunks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, file_path, **kwargs): file_path
                for file_path in all_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
        
        self.logger.info(f"Processed {len(all_files)} files, created {len(all_chunks)} chunks")
        return all_chunks
    
    async def process_files_async(self, file_paths: List[str], **kwargs) -> List[DocumentChunk]:
        """Process files asynchronously"""
        async def process_file_async(file_path: str) -> List[DocumentChunk]:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.process_file, file_path, **kwargs)
        
        tasks = [process_file_async(file_path) for file_path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_chunks = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Async processing error: {result}")
            else:
                all_chunks.extend(result)
        
        return all_chunks

# Common validators, transformers, and filters
class CommonProcessors:
    """Common processing functions"""
    
    @staticmethod
    def min_length_validator(min_length: int = 100) -> Callable[[Document], bool]:
        """Validator for minimum document length"""
        def validate(doc: Document) -> bool:
            return len(doc.content.strip()) >= min_length
        return validate
    
    @staticmethod
    def language_validator(expected_language: str = 'en') -> Callable[[Document], bool]:
        """Validator for document language"""
        def validate(doc: Document) -> bool:
            try:
                from langdetect import detect
                detected = detect(doc.content[:1000])  # Check first 1000 chars
                return detected == expected_language
            except:
                return True  # If detection fails, assume valid
        return validate
    
    @staticmethod
    def clean_text_transformer() -> Callable[[Document], Document]:
        """Transformer to clean document text"""
        def transform(doc: Document) -> Document:
            import re
            
            content = doc.content
            
            # Remove excessive whitespace
            content = re.sub(r'\s+', ' ', content)
            
            # Remove special characters (keep basic punctuation)
            content = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', content)
            
            # Remove empty lines
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            content = '\n'.join(lines)
            
            doc.content = content
            return doc
        
        return transform
    
    @staticmethod
    def normalize_unicode_transformer() -> Callable[[Document], Document]:
        """Transformer to normalize Unicode text"""
        def transform(doc: Document) -> Document:
            import unicodedata
            doc.content = unicodedata.normalize('NFKC', doc.content)
            return doc
        
        return transform
    
    @staticmethod
    def min_chunk_size_filter(min_size: int = 50) -> Callable[[DocumentChunk], bool]:
        """Filter for minimum chunk size"""
        def filter_func(chunk: DocumentChunk) -> bool:
            return len(chunk.content.strip()) >= min_size
        return filter_func
    
    @staticmethod
    def max_chunk_size_filter(max_size: int = 2000) -> Callable[[DocumentChunk], bool]:
        """Filter for maximum chunk size"""
        def filter_func(chunk: DocumentChunk) -> bool:
            return len(chunk.content.strip()) <= max_size
        return filter_func
    
    @staticmethod
    def word_count_filter(min_words: int = 10, max_words: int = 500) -> Callable[[DocumentChunk], bool]:
        """Filter for word count range"""
        def filter_func(chunk: DocumentChunk) -> bool:
            word_count = len(chunk.content.split())
            return min_words <= word_count <= max_words
        return filter_func

# Example usage
def create_comprehensive_pipeline() -> ProcessingPipeline:
    """Create a comprehensive document processing pipeline"""
    
    # Create chunking configurations
    paragraph_config = ChunkingConfig(
        strategy=ChunkingStrategy.PARAGRAPH_BASED,
        chunk_size=512,
        overlap=50,
        respect_sentence_boundaries=True
    )
    
    sentence_config = ChunkingConfig(
        strategy=ChunkingStrategy.SENTENCE_BASED,
        chunk_size=256,
        overlap=20,
        use_tokens=True
    )
    
    # Create chunkers
    paragraph_chunker = DocumentChunkerFactory.create_chunker(
        ChunkingStrategy.PARAGRAPH_BASED, 
        paragraph_config
    )
    
    sentence_chunker = DocumentChunkerFactory.create_chunker(
        ChunkingStrategy.SENTENCE_BASED, 
        sentence_config
    )
    
    # Build pipeline
    pipeline = ProcessingPipeline()
    
    # Add validators
    pipeline.add_validator(CommonProcessors.min_length_validator(50))
    pipeline.add_validator(CommonProcessors.language_validator('en'))
    
    # Add transformers
    pipeline.add_transformer(CommonProcessors.clean_text_transformer())
    pipeline.add_transformer(CommonProcessors.normalize_unicode_transformer())
    
    # Add chunkers
    pipeline.add_chunker(paragraph_chunker)
    
    # Add filters
    pipeline.add_filter(CommonProcessors.min_chunk_size_filter(30))
    pipeline.add_filter(CommonProcessors.max_chunk_size_filter(1000))
    pipeline.add_filter(CommonProcessors.word_count_filter(5, 200))
    
    return pipeline

def demonstrate_document_processing():
    """Demonstrate document processing capabilities"""
    
    print("=== Document Processing and Chunking Demo ===")
    
    # Create sample documents
    sample_files = {
        "sample.txt": "This is a simple text document. It contains multiple sentences. Each sentence provides some information. The document will be processed and chunked.",
        "sample.md": """# Sample Markdown Document

## Introduction

This is a sample markdown document with multiple sections.

## Main Content

### Subsection 1

Content of subsection 1 with detailed information.

### Subsection 2

Content of subsection 2 with more details.

## Conclusion

This concludes our sample document.
""",
        "sample.html": """<!DOCTYPE html>
<html>
<head>
    <title>Sample HTML Document</title>
    <meta name="description" content="A sample HTML document for testing">
</head>
<body>
    <h1>Main Heading</h1>
    <p>This is a paragraph in the HTML document.</p>
    <p>This is another paragraph with <strong>bold text</strong>.</p>
    <script>console.log('This should be removed');</script>
</body>
</html>"""
    }
    
    # Create sample files
    for filename, content in sample_files.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    try:
        # Create processing pipeline
        pipeline = create_comprehensive_pipeline()
        
        # Process each file
        all_chunks = []
        for filename in sample_files.keys():
            print(f"\nProcessing {filename}...")
            chunks = pipeline.process_file(filename)
            all_chunks.extend(chunks)
            
            print(f"  Created {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
                print(f"  Chunk {i+1}: {chunk.content[:100]}...")
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        
        # Demonstrate different chunking strategies
        print("\n=== Chunking Strategy Comparison ===")
        
        sample_doc = Document(
            content=sample_files["sample.md"],
            doc_id="sample_md",
            doc_type=DocumentType.MARKDOWN,
            source="sample.md",
            metadata={}
        )
        
        strategies = [
            (ChunkingStrategy.FIXED_SIZE, {"chunk_size": 100, "overlap": 20}),
            (ChunkingStrategy.SENTENCE_BASED, {"chunk_size": 150, "overlap": 10}),
            (ChunkingStrategy.PARAGRAPH_BASED, {"chunk_size": 200, "overlap": 30}),
            (ChunkingStrategy.HIERARCHICAL, {"chunk_size": 250, "overlap": 25})
        ]
        
        for strategy, config_params in strategies:
            config = ChunkingConfig(strategy=strategy, **config_params)
            chunker = DocumentChunkerFactory.create_chunker(strategy, config)
            
            try:
                chunks = chunker.chunk(sample_doc)
                print(f"\n{strategy.value}: {len(chunks)} chunks")
                for i, chunk in enumerate(chunks[:2]):
                    print(f"  Chunk {i+1} ({len(chunk.content)} chars): {chunk.content[:80]}...")
            except Exception as e:
                print(f"{strategy.value}: Error - {e}")
    
    finally:
        # Clean up sample files
        for filename in sample_files.keys():
            Path(filename).unlink(missing_ok=True)

if __name__ == "__main__":
    demonstrate_document_processing()
```

## Chapter 6: Retrieval Strategies and Optimization

Effective retrieval is the cornerstone of RAG systems. This chapter covers advanced retrieval strategies, optimization techniques, and methods to improve retrieval accuracy and relevance.

### 6.1 Core Retrieval Strategies

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import logging
import time

class RetrievalStrategy(Enum):
    """Types of retrieval strategies"""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    KEYWORD_SEARCH = "keyword_search"
    HYBRID_SEARCH = "hybrid_search"
    MULTI_QUERY = "multi_query"
    CONTEXTUAL_COMPRESSION = "contextual_compression"
    HIERARCHICAL_RETRIEVAL = "hierarchical_retrieval"
    ENSEMBLE_RETRIEVAL = "ensemble_retrieval"

@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    strategy_used: str
    retrieval_time: Optional[float] = None
    
    def __post_init__(self):
        if self.retrieval_time is None:
            self.retrieval_time = time.time()

@dataclass
class RetrievalConfig:
    """Configuration for retrieval strategies"""
    strategy: RetrievalStrategy
    top_k: int = 5
    similarity_threshold: float = 0.0
    max_tokens: Optional[int] = None
    enable_reranking: bool = False
    rerank_top_k: int = 20
    enable_compression: bool = False
    enable_fusion: bool = False
    fusion_weights: Optional[Dict[str, float]] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)

class DocumentRetriever(ABC):
    """Abstract base class for document retrievers"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query"""
        pass
    
    def _filter_by_threshold(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Filter results by similarity threshold"""
        return [r for r in results if r.score >= self.config.similarity_threshold]
    
    def _limit_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Limit results to top_k"""
        return results[:self.config.top_k]

class SemanticRetriever(DocumentRetriever):
    """Semantic similarity-based retriever"""
    
    def __init__(self, config: RetrievalConfig, vector_store, embedding_model):
        super().__init__(config)
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve using semantic similarity"""
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in vector store
        search_results = self.vector_store.search(
            query_embedding, 
            top_k=self.config.rerank_top_k if self.config.enable_reranking else self.config.top_k
        )
        
        # Convert to RetrievalResult objects
        results = []
        for result in search_results:
            retrieval_result = RetrievalResult(
                chunk_id=result.get('id', ''),
                content=result.get('content', ''),
                score=result.get('score', 0.0),
                metadata=result.get('metadata', {}),
                strategy_used="semantic_similarity",
                retrieval_time=time.time() - start_time
            )
            results.append(retrieval_result)
        
        # Apply filtering and limiting
        results = self._filter_by_threshold(results)
        
        if self.config.enable_reranking:
            results = self._rerank_results(query, results)
        
        results = self._limit_results(results)
        
        self.logger.debug(f"Semantic retrieval found {len(results)} results in {time.time() - start_time:.3f}s")
        return results
    
    def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results using cross-encoder or advanced similarity"""
        try:
            from sentence_transformers import CrossEncoder
            
            # Load cross-encoder model (cache it in production)
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Prepare query-document pairs
            pairs = [(query, result.content) for result in results]
            
            # Get cross-encoder scores
            cross_scores = cross_encoder.predict(pairs)
            
            # Update scores and resort
            for result, score in zip(results, cross_scores):
                result.score = float(score)
                result.metadata['original_score'] = result.metadata.get('score', result.score)
            
            results.sort(key=lambda x: x.score, reverse=True)
            
        except ImportError:
            self.logger.warning("CrossEncoder not available for reranking")
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
        
        return results

class KeywordRetriever(DocumentRetriever):
    """Keyword-based retriever using BM25 or similar"""
    
    def __init__(self, config: RetrievalConfig, document_store):
        super().__init__(config)
        self.document_store = document_store
        self.bm25_index = None
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from documents"""
        try:
            from rank_bm25 import BM25Okapi
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('punkt')
                nltk.download('stopwords')
            
            self.stop_words = set(stopwords.words('english'))
            
            # Get all documents from store
            documents = self.document_store.get_all_documents()
            
            # Tokenize documents
            tokenized_docs = []
            self.doc_metadata = []
            
            for doc in documents:
                tokens = self._tokenize_text(doc['content'])
                tokenized_docs.append(tokens)
                self.doc_metadata.append({
                    'id': doc['id'],
                    'content': doc['content'],
                    'metadata': doc.get('metadata', {})
                })
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.logger.info(f"Built BM25 index with {len(tokenized_docs)} documents")
            
        except ImportError:
            raise RuntimeError("rank_bm25 and nltk are required for keyword retrieval. Install with: pip install rank_bm25 nltk")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        try:
            from nltk.tokenize import word_tokenize
            
            tokens = word_tokenize(text.lower())
            # Remove stopwords and non-alphabetic tokens
            tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
            return tokens
        except:
            # Fallback to simple split
            return [word.lower() for word in text.split() if word.isalpha()]
    
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve using keyword matching (BM25)"""
        start_time = time.time()
        
        if self.bm25_index is None:
            self.logger.error("BM25 index not built")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize_text(query)
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Create results with scores
        results = []
        for i, score in enumerate(scores):
            if score > 0:  # Only include documents with positive scores
                doc_info = self.doc_metadata[i]
                result = RetrievalResult(
                    chunk_id=doc_info['id'],
                    content=doc_info['content'],
                    score=float(score),
                    metadata=doc_info['metadata'],
                    strategy_used="keyword_search",
                    retrieval_time=time.time() - start_time
                )
                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply filtering and limiting
        results = self._filter_by_threshold(results)
        results = self._limit_results(results)
        
        self.logger.debug(f"Keyword retrieval found {len(results)} results in {time.time() - start_time:.3f}s")
        return results

class HybridRetriever(DocumentRetriever):
    """Hybrid retriever combining semantic and keyword search"""
    
    def __init__(self, config: RetrievalConfig, semantic_retriever: SemanticRetriever, 
                 keyword_retriever: KeywordRetriever):
        super().__init__(config)
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        
        # Default fusion weights
        self.fusion_weights = config.fusion_weights or {
            'semantic': 0.7,
            'keyword': 0.3
        }
    
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve using hybrid approach"""
        start_time = time.time()
        
        # Get results from both retrievers
        semantic_results = self.semantic_retriever.retrieve(query, **kwargs)
        keyword_results = self.keyword_retriever.retrieve(query, **kwargs)
        
        if self.config.enable_fusion:
            # Use reciprocal rank fusion (RRF)
            results = self._reciprocal_rank_fusion(semantic_results, keyword_results)
        else:
            # Simple score-based fusion
            results = self._score_fusion(semantic_results, keyword_results)
        
        # Apply final filtering and limiting
        results = self._filter_by_threshold(results)
        results = self._limit_results(results)
        
        self.logger.debug(f"Hybrid retrieval found {len(results)} results in {time.time() - start_time:.3f}s")
        return results
    
    def _score_fusion(self, semantic_results: List[RetrievalResult], 
                     keyword_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Fuse results using weighted scores"""
        # Normalize scores to 0-1 range
        semantic_results = self._normalize_scores(semantic_results)
        keyword_results = self._normalize_scores(keyword_results)
        
        # Create combined results dict
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            combined_results[result.chunk_id] = RetrievalResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score * self.fusion_weights['semantic'],
                metadata={**result.metadata, 'semantic_score': result.score},
                strategy_used="hybrid_search"
            )
        
        # Add/update with keyword results
        for result in keyword_results:
            if result.chunk_id in combined_results:
                # Combine scores
                combined_results[result.chunk_id].score += result.score * self.fusion_weights['keyword']
                combined_results[result.chunk_id].metadata['keyword_score'] = result.score
            else:
                # New result from keyword search
                combined_results[result.chunk_id] = RetrievalResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=result.score * self.fusion_weights['keyword'],
                    metadata={**result.metadata, 'keyword_score': result.score},
                    strategy_used="hybrid_search"
                )
        
        # Sort by combined score
        results = list(combined_results.values())
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _reciprocal_rank_fusion(self, semantic_results: List[RetrievalResult], 
                               keyword_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Fuse results using Reciprocal Rank Fusion (RRF)"""
        k = 60  # RRF parameter
        
        # Create rank mappings
        semantic_ranks = {result.chunk_id: i + 1 for i, result in enumerate(semantic_results)}
        keyword_ranks = {result.chunk_id: i + 1 for i, result in enumerate(keyword_results)}
        
        # Get all unique document IDs
        all_doc_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        results_map = {}
        
        # Store results for later retrieval
        for result in semantic_results + keyword_results:
            if result.chunk_id not in results_map:
                results_map[result.chunk_id] = result
        
        for doc_id in all_doc_ids:
            rrf_score = 0
            
            if doc_id in semantic_ranks:
                rrf_score += self.fusion_weights['semantic'] / (k + semantic_ranks[doc_id])
            
            if doc_id in keyword_ranks:
                rrf_score += self.fusion_weights['keyword'] / (k + keyword_ranks[doc_id])
            
            rrf_scores[doc_id] = rrf_score
        
        # Create final results
        results = []
        for doc_id, score in rrf_scores.items():
            if doc_id in results_map:
                result = results_map[doc_id]
                result.score = score
                result.strategy_used = "hybrid_search_rrf"
                result.metadata['rrf_score'] = score
                if doc_id in semantic_ranks:
                    result.metadata['semantic_rank'] = semantic_ranks[doc_id]
                if doc_id in keyword_ranks:
                    result.metadata['keyword_rank'] = keyword_ranks[doc_id]
                results.append(result)
        
        # Sort by RRF score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Normalize scores to 0-1 range using min-max normalization"""
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            for result in results:
                result.score = 1.0
        else:
            # Normalize to 0-1
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)
        
        return results

class MultiQueryRetriever(DocumentRetriever):
    """Multi-query retriever that generates multiple queries for better coverage"""
    
    def __init__(self, config: RetrievalConfig, base_retriever: DocumentRetriever, 
                 query_generator=None):
        super().__init__(config)
        self.base_retriever = base_retriever
        self.query_generator = query_generator
        
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve using multiple query variations"""
        start_time = time.time()
        
        # Generate query variations
        queries = self._generate_query_variations(query)
        
        # Retrieve for each query
        all_results = []
        for q in queries:
            results = self.base_retriever.retrieve(q, **kwargs)
            for result in results:
                result.metadata['source_query'] = q
            all_results.extend(results)
        
        # Aggregate and deduplicate results
        results = self._aggregate_results(all_results)
        
        # Apply filtering and limiting
        results = self._filter_by_threshold(results)
        results = self._limit_results(results)
        
        self.logger.debug(f"Multi-query retrieval found {len(results)} results in {time.time() - start_time:.3f}s")
        return results
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate variations of the input query"""
        variations = [query]  # Always include original query
        
        if self.query_generator:
            # Use LLM to generate variations
            try:
                generated = self.query_generator.generate_variations(query)
                variations.extend(generated)
            except Exception as e:
                self.logger.error(f"Query generation failed: {e}")
        else:
            # Simple rule-based variations
            variations.extend(self._rule_based_variations(query))
        
        return variations[:5]  # Limit to 5 variations
    
    def _rule_based_variations(self, query: str) -> List[str]:
        """Generate simple rule-based query variations"""
        variations = []
        
        # Add question variations
        if not query.endswith('?'):
            variations.append(f"What is {query}?")
            variations.append(f"How does {query} work?")
            variations.append(f"Explain {query}")
        
        # Add keyword extraction
        keywords = query.split()
        if len(keywords) > 2:
            # Create shorter variations
            variations.append(' '.join(keywords[:len(keywords)//2]))
            variations.append(' '.join(keywords[len(keywords)//2:]))
        
        return variations
    
    def _aggregate_results(self, all_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Aggregate results from multiple queries"""
        # Group by chunk_id
        result_groups = defaultdict(list)
        for result in all_results:
            result_groups[result.chunk_id].append(result)
        
        # Aggregate scores for each unique result
        aggregated_results = []
        for chunk_id, results in result_groups.items():
            # Use the best result as base
            best_result = max(results, key=lambda x: x.score)
            
            # Aggregate scores (average or max)
            avg_score = sum(r.score for r in results) / len(results)
            max_score = max(r.score for r in results)
            
            # Create aggregated result
            aggregated_result = RetrievalResult(
                chunk_id=best_result.chunk_id,
                content=best_result.content,
                score=max_score,  # Use max score
                metadata={
                    **best_result.metadata,
                    'query_count': len(results),
                    'avg_score': avg_score,
                    'max_score': max_score,
                    'source_queries': [r.metadata.get('source_query', '') for r in results]
                },
                strategy_used="multi_query"
            )
            aggregated_results.append(aggregated_result)
        
        # Sort by aggregated score
        aggregated_results.sort(key=lambda x: x.score, reverse=True)
        
        return aggregated_results

class ContextualCompressionRetriever(DocumentRetriever):
    """Retriever with contextual compression to improve relevance"""
    
    def __init__(self, config: RetrievalConfig, base_retriever: DocumentRetriever, 
                 compressor=None):
        super().__init__(config)
        self.base_retriever = base_retriever
        self.compressor = compressor
    
    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve with contextual compression"""
        start_time = time.time()
        
        # Get initial results
        results = self.base_retriever.retrieve(query, **kwargs)
        
        if self.config.enable_compression and self.compressor:
            # Compress/filter results based on relevance
            results = self._compress_results(query, results)
        
        self.logger.debug(f"Contextual compression retrieval found {len(results)} results in {time.time() - start_time:.3f}s")
        return results
    
    def _compress_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Compress results by filtering irrelevant parts"""
        compressed_results = []
        
        for result in results:
            try:
                # Use compressor to filter relevant sentences/passages
                relevant_content = self.compressor.compress(query, result.content)
                
                if relevant_content and relevant_content.strip():
                    compressed_result = RetrievalResult(
                        chunk_id=result.chunk_id,
                        content=relevant_content,
                        score=result.score,
                        metadata={
                            **result.metadata,
                            'original_content': result.content,
                            'compression_ratio': len(relevant_content) / len(result.content)
                        },
                        strategy_used=f"{result.strategy_used}_compressed"
                    )
                    compressed_results.append(compressed_result)
            
            except Exception as e:
                self.logger.error(f"Compression failed for result {result.chunk_id}: {e}")
                # Fall back to original result
                compressed_results.append(result)
        
        return compressed_results
```

### 6.2 Advanced Retrieval Optimization Techniques

```python
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from dataclasses import dataclass

@dataclass
class QueryExpansion:
    """Query expansion techniques"""
    
    @staticmethod
    def expand_with_synonyms(query: str, synonym_dict: Optional[Dict[str, List[str]]] = None) -> List[str]:
        """Expand query with synonyms"""
        if not synonym_dict:
            # Default synonym expansion using WordNet
            try:
                import nltk
                from nltk.corpus import wordnet
                
                try:
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    nltk.download('wordnet')
                
                expanded_queries = [query]
                words = query.split()
                
                for word in words:
                    synonyms = set()
                    for syn in wordnet.synsets(word):
                        for lemma in syn.lemmas():
                            synonyms.add(lemma.name().replace('_', ' '))
                    
                    # Create expanded queries with synonyms
                    for synonym in list(synonyms)[:3]:  # Limit to 3 synonyms per word
                        if synonym != word:
                            expanded_query = query.replace(word, synonym)
                            expanded_queries.append(expanded_query)
                
                return expanded_queries[:5]  # Limit total queries
            
            except ImportError:
                return [query]
        else:
            # Use provided synonym dictionary
            expanded_queries = [query]
            
            for original, synonyms in synonym_dict.items():
                if original in query:
                    for synonym in synonyms[:2]:  # Limit synonyms
                        expanded_query = query.replace(original, synonym)
                        expanded_queries.append(expanded_query)
            
            return expanded_queries
    
    @staticmethod
    def expand_with_related_terms(query: str, related_terms: Dict[str, List[str]]) -> List[str]:
        """Expand query with related terms"""
        expanded_queries = [query]
        
        for term, related in related_terms.items():
            if term.lower() in query.lower():
                for related_term in related[:2]:
                    expanded_query = f"{query} {related_term}"
                    expanded_queries.append(expanded_query)
        
        return expanded_queries

class RetrievalOptimizer:
    """Optimization techniques for retrieval performance"""
    
    def __init__(self):
        self.query_cache = {}
        self.performance_metrics = {}
    
    def optimize_query(self, query: str) -> str:
        """Optimize query for better retrieval"""
        # Cache check
        if query in self.query_cache:
            return self.query_cache[query]
        
        optimized = query
        
        # Remove stop words for better matching
        optimized = self._remove_stop_words(optimized)
        
        # Normalize text
        optimized = self._normalize_text(optimized)
        
        # Expand abbreviations
        optimized = self._expand_abbreviations(optimized)
        
        self.query_cache[query] = optimized
        return optimized
    
    def _remove_stop_words(self, text: str) -> str:
        """Remove common stop words"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words) if filtered_words else text
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters except spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations"""
        abbreviations = {
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'nlp': 'natural language processing',
            'api': 'application programming interface',
            'db': 'database',
            'ui': 'user interface',
            'ux': 'user experience'
        }
        
        words = text.split()
        expanded_words = []
        
        for word in words:
            if word.lower() in abbreviations:
                expanded_words.append(abbreviations[word.lower()])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def adaptive_top_k(self, query: str, base_top_k: int = 5) -> int:
        """Adaptively determine optimal top_k based on query complexity"""
        # Simple heuristic based on query length and complexity
        words = query.split()
        
        if len(words) <= 3:
            # Simple queries might need more results
            return min(base_top_k + 2, 10)
        elif len(words) > 10:
            # Complex queries might be more specific
            return base_top_k
        else:
            return base_top_k + 1

class RetrievalEvaluator:
    """Evaluate retrieval quality and performance"""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_retrieval(self, query: str, results: List[RetrievalResult], 
                          ground_truth: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate retrieval quality"""
        metrics = {
            'query': query,
            'num_results': len(results),
            'avg_score': np.mean([r.score for r in results]) if results else 0,
            'score_variance': np.var([r.score for r in results]) if results else 0,
            'retrieval_time': sum(r.retrieval_time for r in results if r.retrieval_time) / len(results) if results else 0
        }
        
        if ground_truth:
            # Calculate precision, recall, F1
            retrieved_ids = set(r.chunk_id for r in results)
            relevant_ids = set(ground_truth)
            
            if retrieved_ids:
                precision = len(retrieved_ids & relevant_ids) / len(retrieved_ids)
                recall = len(retrieved_ids & relevant_ids) / len(relevant_ids) if relevant_ids else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics.update({
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'mrr': self._calculate_mrr(results, relevant_ids),
                    'ndcg': self._calculate_ndcg(results, relevant_ids)
                })
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calculate_mrr(self, results: List[RetrievalResult], relevant_ids: set) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, result in enumerate(results):
            if result.chunk_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_ndcg(self, results: List[RetrievalResult], relevant_ids: set, k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        dcg = 0.0
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        
        for i, result in enumerate(results[:k]):
            if result.chunk_id in relevant_ids:
                dcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of retrieval performance"""
        if not self.metrics_history:
            return {}
        
        return {
            'total_queries': len(self.metrics_history),
            'avg_precision': np.mean([m.get('precision', 0) for m in self.metrics_history]),
            'avg_recall': np.mean([m.get('recall', 0) for m in self.metrics_history]),
            'avg_f1': np.mean([m.get('f1_score', 0) for m in self.metrics_history]),
            'avg_mrr': np.mean([m.get('mrr', 0) for m in self.metrics_history]),
            'avg_ndcg': np.mean([m.get('ndcg', 0) for m in self.metrics_history]),
            'avg_retrieval_time': np.mean([m.get('retrieval_time', 0) for m in self.metrics_history])
        }

class RetrievalPipeline:
    """Complete retrieval pipeline with optimization"""
    
    def __init__(self, retrievers: List[DocumentRetriever], 
                 optimizer: Optional[RetrievalOptimizer] = None,
                 evaluator: Optional[RetrievalEvaluator] = None):
        self.retrievers = retrievers
        self.optimizer = optimizer or RetrievalOptimizer()
        self.evaluator = evaluator or RetrievalEvaluator()
        self.logger = logging.getLogger(__name__)
    
    def retrieve(self, query: str, strategy: str = "auto", **kwargs) -> List[RetrievalResult]:
        """Retrieve documents using specified strategy"""
        start_time = time.time()
        
        # Optimize query
        optimized_query = self.optimizer.optimize_query(query)
        
        # Select retriever based on strategy
        if strategy == "auto":
            retriever = self._select_best_retriever(optimized_query)
        else:
            retriever = self._get_retriever_by_name(strategy)
        
        if not retriever:
            self.logger.error(f"No retriever found for strategy: {strategy}")
            return []
        
        # Perform retrieval
        results = retriever.retrieve(optimized_query, **kwargs)
        
        # Post-process results
        results = self._post_process_results(query, results)
        
        # Evaluate results
        evaluation = self.evaluator.evaluate_retrieval(query, results)
        self.logger.debug(f"Retrieval evaluation: {evaluation}")
        
        self.logger.info(f"Retrieved {len(results)} documents in {time.time() - start_time:.3f}s")
        return results
    
    def _select_best_retriever(self, query: str) -> Optional[DocumentRetriever]:
        """Select the best retriever based on query characteristics"""
        # Simple heuristic: use hybrid for complex queries, semantic for simple ones
        words = query.split()
        
        if len(words) > 5:
            # Complex query - use hybrid retrieval
            for retriever in self.retrievers:
                if isinstance(retriever, HybridRetriever):
                    return retriever
        
        # Default to first semantic retriever
        for retriever in self.retrievers:
            if isinstance(retriever, SemanticRetriever):
                return retriever
        
        return self.retrievers[0] if self.retrievers else None
    
    def _get_retriever_by_name(self, strategy: str) -> Optional[DocumentRetriever]:
        """Get retriever by strategy name"""
        strategy_map = {
            'semantic': SemanticRetriever,
            'keyword': KeywordRetriever,
            'hybrid': HybridRetriever,
            'multi_query': MultiQueryRetriever,
            'contextual': ContextualCompressionRetriever
        }
        
        target_class = strategy_map.get(strategy)
        if target_class:
            for retriever in self.retrievers:
                if isinstance(retriever, target_class):
                    return retriever
        
        return None
    
    def _post_process_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Post-process retrieval results"""
        # Remove duplicates
        seen_content = set()
        unique_results = []
        
        for result in results:
            # Use first 100 chars for duplicate detection
            content_key = result.content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        # Add diversity scoring
        unique_results = self._add_diversity_scores(unique_results)
        
        return unique_results
    
    def _add_diversity_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Add diversity scores to promote result diversity"""
        if len(results) <= 1:
            return results
        
        # Create TF-IDF vectors for content
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            contents = [result.content for result in results]
            tfidf_matrix = vectorizer.fit_transform(contents)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(tfidf_matrix)
            
            # Calculate diversity scores (1 - average similarity to other results)
            for i, result in enumerate(results):
                other_sims = [similarities[i][j] for j in range(len(results)) if i != j]
                diversity_score = 1.0 - np.mean(other_sims) if other_sims else 1.0
                result.metadata['diversity_score'] = diversity_score
        
        except Exception as e:
            self.logger.error(f"Failed to calculate diversity scores: {e}")
            # Set default diversity scores
            for result in results:
                result.metadata['diversity_score'] = 1.0
        
        return results

# Example usage and factory
class RetrieverFactory:
    """Factory for creating retrievers"""
    
    @staticmethod
    def create_semantic_retriever(vector_store, embedding_model, config: Optional[RetrievalConfig] = None) -> SemanticRetriever:
        """Create semantic retriever"""
        if not config:
            config = RetrievalConfig(
                strategy=RetrievalStrategy.SEMANTIC_SIMILARITY,
                top_k=5,
                enable_reranking=True
            )
        return SemanticRetriever(config, vector_store, embedding_model)
    
    @staticmethod
    def create_keyword_retriever(document_store, config: Optional[RetrievalConfig] = None) -> KeywordRetriever:
        """Create keyword retriever"""
        if not config:
            config = RetrievalConfig(
                strategy=RetrievalStrategy.KEYWORD_SEARCH,
                top_k=5
            )
        return KeywordRetriever(config, document_store)
    
    @staticmethod
    def create_hybrid_retriever(vector_store, embedding_model, document_store, 
                               config: Optional[RetrievalConfig] = None) -> HybridRetriever:
        """Create hybrid retriever"""
        if not config:
            config = RetrievalConfig(
                strategy=RetrievalStrategy.HYBRID_SEARCH,
                top_k=5,
                enable_reranking=True,
                enable_fusion=True,
                fusion_weights={'semantic': 0.7, 'keyword': 0.3}
            )
        
        semantic_retriever = RetrieverFactory.create_semantic_retriever(vector_store, embedding_model, config)
        keyword_retriever = RetrieverFactory.create_keyword_retriever(document_store, config)
        
        return HybridRetriever(config, semantic_retriever, keyword_retriever)
    
    @staticmethod
    def create_optimized_pipeline(vector_store, embedding_model, document_store) -> RetrievalPipeline:
        """Create optimized retrieval pipeline"""
        # Create different retrievers
        retrievers = [
            RetrieverFactory.create_semantic_retriever(vector_store, embedding_model),
            RetrieverFactory.create_keyword_retriever(document_store),
            RetrieverFactory.create_hybrid_retriever(vector_store, embedding_model, document_store)
        ]
        
        # Create optimizer and evaluator
        optimizer = RetrievalOptimizer()
        evaluator = RetrievalEvaluator()
        
        return RetrievalPipeline(retrievers, optimizer, evaluator)

def demonstrate_retrieval_strategies():
    """Demonstrate different retrieval strategies"""
    print("=== Retrieval Strategies Demonstration ===")
    
    # This would typically use real vector stores and embedding models
    print("\n1. Query Optimization:")
    optimizer = RetrievalOptimizer()
    
    test_queries = [
        "What is AI and ML?",
        "How to implement REST API in Python?",
        "Database optimization techniques"
    ]
    
    for query in test_queries:
        optimized = optimizer.optimize_query(query)
        print(f"  Original: {query}")
        print(f"  Optimized: {optimized}")
        print(f"  Adaptive top_k: {optimizer.adaptive_top_k(query)}")
    
    print("\n2. Query Expansion:")
    for query in test_queries[:2]:
        expanded = QueryExpansion.expand_with_synonyms(query)
        print(f"  Query: {query}")
        print(f"  Expanded: {expanded[:3]}")  # Show first 3 variations
    
    print("\n3. Evaluation Metrics:")
    evaluator = RetrievalEvaluator()
    
    # Mock results for demonstration
    mock_results = [
        RetrievalResult("doc1", "Content 1", 0.9, {}, "semantic"),
        RetrievalResult("doc2", "Content 2", 0.8, {}, "semantic"),
        RetrievalResult("doc3", "Content 3", 0.7, {}, "semantic")
    ]
    
    evaluation = evaluator.evaluate_retrieval(
        "test query", 
        mock_results, 
        ground_truth=["doc1", "doc3"]  # Mock ground truth
    )
    
    print(f"  Evaluation: {evaluation}")

if __name__ == "__main__":
    demonstrate_retrieval_strategies()
```

## Chapter 7: Generation and Response Synthesis

The generation component is where retrieved context meets language model capabilities to produce coherent, accurate, and contextually relevant responses. This chapter covers advanced generation strategies, prompt engineering techniques, and response optimization.

### 7.1 Core Generation Framework

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import logging
from pathlib import Path

class GenerationStrategy(Enum):
    """Types of generation strategies"""
    BASIC_RAG = "basic_rag"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    SELF_CONSISTENCY = "self_consistency"
    RETRIEVAL_AUGMENTED_THINKING = "retrieval_augmented_thinking"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    ENSEMBLE_GENERATION = "ensemble_generation"

@dataclass
class GenerationConfig:
    """Configuration for generation strategies"""
    strategy: GenerationStrategy = GenerationStrategy.BASIC_RAG
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    enable_reasoning_traces: bool = False
    max_reasoning_steps: int = 3
    confidence_threshold: float = 0.8
    enable_fact_checking: bool = False
    custom_instructions: Optional[str] = None

@dataclass
class GenerationResult:
    """Result from generation operation"""
    response: str
    confidence_score: float
    reasoning_trace: Optional[List[str]] = None
    source_chunks: Optional[List[str]] = None
    generation_time: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.generation_time is None:
            self.generation_time = time.time()

class ResponseGenerator(ABC):
    """Abstract base class for response generators"""
    
    def __init__(self, config: GenerationConfig, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def generate(self, query: str, context: List[str], **kwargs) -> GenerationResult:
        """Generate response based on query and retrieved context"""
        pass
    
    def _format_context(self, context: List[str]) -> str:
        """Format retrieved context for prompt"""
        if not context:
            return "No relevant context found."
        
        formatted_context = []
        for i, chunk in enumerate(context, 1):
            formatted_context.append(f"[{i}] {chunk}")
        
        return "\n".join(formatted_context)
    
    def _estimate_confidence(self, response: str, context: List[str]) -> float:
        """Estimate confidence score for generated response"""
        # Simple heuristic-based confidence estimation
        confidence = 0.5  # Base confidence
        
        # Check if response references context
        context_words = set()
        for chunk in context:
            context_words.update(chunk.lower().split())
        
        response_words = set(response.lower().split())
        overlap = len(context_words & response_words) / len(response_words) if response_words else 0
        
        confidence += min(overlap * 0.3, 0.3)  # Max 0.3 boost from context overlap
        
        # Check response length (too short or too long might indicate issues)
        word_count = len(response.split())
        if 10 <= word_count <= 200:
            confidence += 0.1
        
        # Check for uncertainty phrases
        uncertainty_phrases = ['i think', 'maybe', 'possibly', 'not sure', 'unclear']
        if any(phrase in response.lower() for phrase in uncertainty_phrases):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))

class BasicRAGGenerator(ResponseGenerator):
    """Basic RAG generator using simple context injection"""
    
    def generate(self, query: str, context: List[str], **kwargs) -> GenerationResult:
        """Generate response using basic RAG approach"""
        start_time = time.time()
        
        # Format context
        formatted_context = self._format_context(context)
        
        # Create prompt
        prompt = self._create_basic_prompt(query, formatted_context)
        
        # Generate response
        response = self._call_llm(prompt)
        
        # Estimate confidence
        confidence = self._estimate_confidence(response, context)
        
        return GenerationResult(
            response=response,
            confidence_score=confidence,
            source_chunks=context,
            generation_time=time.time() - start_time,
            metadata={
                "strategy": "basic_rag",
                "prompt_length": len(prompt),
                "context_chunks": len(context)
            }
        )
    
    def _create_basic_prompt(self, query: str, context: str) -> str:
        """Create basic RAG prompt"""
        base_prompt = """You are a helpful AI assistant. Use the provided context to answer the user's question accurately and comprehensively.

Context:
{context}

Question: {query}

Instructions:
- Base your answer primarily on the provided context
- If the context doesn't contain sufficient information, clearly state this
- Be concise but thorough
- Cite relevant parts of the context when appropriate

Answer:"""
        
        return base_prompt.format(context=context, query=query)
    
    def _call_llm(self, prompt: str) -> str:
        """Call language model to generate response"""
        if not self.llm_client:
            # Mock response for demonstration
            return "This is a mock response. In production, this would call a real LLM."
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=self.config.stop_sequences
            )
            return response.get('text', '').strip()
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return "I apologize, but I'm unable to generate a response at this time."

class ChainOfThoughtGenerator(ResponseGenerator):
    """Generator using chain-of-thought reasoning"""
    
    def generate(self, query: str, context: List[str], **kwargs) -> GenerationResult:
        """Generate response using chain-of-thought approach"""
        start_time = time.time()
        
        # Format context
        formatted_context = self._format_context(context)
        
        # Create CoT prompt
        prompt = self._create_cot_prompt(query, formatted_context)
        
        # Generate response with reasoning
        full_response = self._call_llm(prompt)
        
        # Parse reasoning and final answer
        response, reasoning_trace = self._parse_cot_response(full_response)
        
        # Estimate confidence
        confidence = self._estimate_confidence(response, context)
        if reasoning_trace:
            confidence += 0.1  # Slight boost for showing reasoning
        
        return GenerationResult(
            response=response,
            confidence_score=confidence,
            reasoning_trace=reasoning_trace,
            source_chunks=context,
            generation_time=time.time() - start_time,
            metadata={
                "strategy": "chain_of_thought",
                "reasoning_steps": len(reasoning_trace) if reasoning_trace else 0
            }
        )
    
    def _create_cot_prompt(self, query: str, context: str) -> str:
        """Create chain-of-thought prompt"""
        cot_prompt = """You are a helpful AI assistant that thinks step by step. Use the provided context to answer the user's question by working through your reasoning process.

Context:
{context}

Question: {query}

Instructions:
- Think through the problem step by step
- Show your reasoning process clearly
- Use the context to support your reasoning
- Provide a clear final answer

Let me think about this step by step:

Step 1: Understanding the question
[Analyze what the question is asking]

Step 2: Examining the context
[Review relevant information from the context]

Step 3: Reasoning through the answer
[Work through the logical steps to reach a conclusion]

Final Answer: [Provide the definitive answer]"""
        
        return cot_prompt.format(context=context, query=query)
    
    def _parse_cot_response(self, response: str) -> tuple[str, Optional[List[str]]]:
        """Parse chain-of-thought response to extract reasoning and final answer"""
        lines = response.strip().split('\n')
        reasoning_steps = []
        final_answer = ""
        
        in_reasoning = False
        for line in lines:
            line = line.strip()
            if line.startswith('Step') or line.startswith('Let me think'):
                in_reasoning = True
                reasoning_steps.append(line)
            elif line.startswith('Final Answer:'):
                final_answer = line.replace('Final Answer:', '').strip()
                in_reasoning = False
            elif in_reasoning and line:
                reasoning_steps.append(line)
            elif not in_reasoning and line and not final_answer:
                final_answer = line
        
        if not final_answer:
            final_answer = response  # Fallback to full response
        
        return final_answer, reasoning_steps if reasoning_steps else None

class MultiHopReasoningGenerator(ResponseGenerator):
    """Generator for multi-hop reasoning with iterative retrieval"""
    
    def __init__(self, config: GenerationConfig, llm_client=None, retriever=None):
        super().__init__(config, llm_client)
        self.retriever = retriever
    
    def generate(self, query: str, context: List[str], **kwargs) -> GenerationResult:
        """Generate response using multi-hop reasoning"""
        start_time = time.time()
        
        reasoning_trace = []
        all_context = list(context)
        current_question = query
        
        for step in range(self.config.max_reasoning_steps):
            self.logger.debug(f"Multi-hop reasoning step {step + 1}")
            
            # Generate intermediate reasoning
            intermediate_result = self._reason_step(current_question, all_context, step)
            reasoning_trace.append(f"Step {step + 1}: {intermediate_result['reasoning']}")
            
            # Check if we need more information
            if intermediate_result['needs_more_info'] and self.retriever:
                # Generate follow-up query
                follow_up_query = intermediate_result['follow_up_query']
                reasoning_trace.append(f"Follow-up query: {follow_up_query}")
                
                # Retrieve additional context
                additional_context = self._retrieve_additional_context(follow_up_query)
                all_context.extend(additional_context)
                
                current_question = follow_up_query
            else:
                # We have sufficient information
                break
        
        # Generate final answer
        formatted_context = self._format_context(all_context)
        final_prompt = self._create_final_answer_prompt(query, formatted_context, reasoning_trace)
        final_response = self._call_llm(final_prompt)
        
        confidence = self._estimate_confidence(final_response, all_context)
        
        return GenerationResult(
            response=final_response,
            confidence_score=confidence,
            reasoning_trace=reasoning_trace,
            source_chunks=all_context,
            generation_time=time.time() - start_time,
            metadata={
                "strategy": "multi_hop_reasoning",
                "reasoning_steps": len(reasoning_trace),
                "total_context_chunks": len(all_context)
            }
        )
    
    def _reason_step(self, question: str, context: List[str], step: int) -> Dict[str, Any]:
        """Perform one step of multi-hop reasoning"""
        formatted_context = self._format_context(context)
        
        reasoning_prompt = f"""Given the question and context, analyze what information we have and what might be missing.

Question: {question}

Context:
{formatted_context}

Analysis:
1. What information from the context is relevant to the question?
2. Is there sufficient information to answer the question completely?
3. If not, what specific information is missing?
4. What would be a good follow-up question to get the missing information?

Response format:
REASONING: [Your analysis]
SUFFICIENT_INFO: [Yes/No]
MISSING_INFO: [What's missing if insufficient]
FOLLOW_UP_QUERY: [Follow-up question if needed]"""
        
        response = self._call_llm(reasoning_prompt)
        
        # Parse response
        result = {
            'reasoning': '',
            'needs_more_info': True,
            'follow_up_query': question
        }
        
        for line in response.split('\n'):
            if line.startswith('REASONING:'):
                result['reasoning'] = line.replace('REASONING:', '').strip()
            elif line.startswith('SUFFICIENT_INFO:'):
                sufficient = line.replace('SUFFICIENT_INFO:', '').strip().lower()
                result['needs_more_info'] = sufficient != 'yes'
            elif line.startswith('FOLLOW_UP_QUERY:'):
                result['follow_up_query'] = line.replace('FOLLOW_UP_QUERY:', '').strip()
        
        return result
    
    def _retrieve_additional_context(self, query: str) -> List[str]:
        """Retrieve additional context for follow-up query"""
        if not self.retriever:
            return []
        
        try:
            results = self.retriever.retrieve(query, top_k=3)
            return [result.content for result in results]
        except Exception as e:
            self.logger.error(f"Additional retrieval failed: {e}")
            return []
    
    def _create_final_answer_prompt(self, original_query: str, context: str, reasoning_trace: List[str]) -> str:
        """Create prompt for final answer generation"""
        reasoning_summary = '\n'.join(reasoning_trace)
        
        prompt = f"""Based on the reasoning process and available context, provide a comprehensive answer to the original question.

Original Question: {original_query}

Reasoning Process:
{reasoning_summary}

Final Context:
{context}

Now provide a clear, comprehensive answer to the original question:"""
        
        return prompt

class SelfConsistencyGenerator(ResponseGenerator):
    """Generator using self-consistency for improved reliability"""
    
    def generate(self, query: str, context: List[str], **kwargs) -> GenerationResult:
        """Generate response using self-consistency approach"""
        start_time = time.time()
        num_samples = kwargs.get('num_samples', 3)
        
        # Generate multiple responses
        responses = []
        for i in range(num_samples):
            # Use higher temperature for diversity
            temp_config = self.config
            temp_config.temperature = min(self.config.temperature + 0.2, 1.0)
            
            generator = BasicRAGGenerator(temp_config, self.llm_client)
            result = generator.generate(query, context)
            responses.append(result.response)
        
        # Find most consistent response
        best_response, consistency_score = self._find_most_consistent(responses)
        
        confidence = consistency_score * 0.8 + 0.2  # Scale consistency to confidence
        
        return GenerationResult(
            response=best_response,
            confidence_score=confidence,
            source_chunks=context,
            generation_time=time.time() - start_time,
            metadata={
                "strategy": "self_consistency",
                "num_samples": num_samples,
                "consistency_score": consistency_score,
                "all_responses": responses
            }
        )
    
    def _find_most_consistent(self, responses: List[str]) -> tuple[str, float]:
        """Find the most consistent response among multiple generations"""
        if len(responses) == 1:
            return responses[0], 1.0
        
        # Simple similarity-based consistency check
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(responses)
            similarity_matrix = cosine_similarity(vectors)
            
            # Calculate average similarity for each response
            avg_similarities = []
            for i in range(len(responses)):
                avg_sim = np.mean([similarity_matrix[i][j] for j in range(len(responses)) if i != j])
                avg_similarities.append(avg_sim)
            
            # Return response with highest average similarity
            best_idx = np.argmax(avg_similarities)
            return responses[best_idx], avg_similarities[best_idx]
        
        except ImportError:
            # Fallback: return first response
            return responses[0], 0.5

class IterativeRefinementGenerator(ResponseGenerator):
    """Generator that iteratively refines responses"""
    
    def generate(self, query: str, context: List[str], **kwargs) -> GenerationResult:
        """Generate response using iterative refinement"""
        start_time = time.time()
        max_iterations = kwargs.get('max_iterations', 2)
        
        # Generate initial response
        current_generator = BasicRAGGenerator(self.config, self.llm_client)
        current_result = current_generator.generate(query, context)
        current_response = current_result.response
        
        refinement_history = [f"Initial: {current_response}"]
        
        for iteration in range(max_iterations):
            # Create refinement prompt
            refined_response = self._refine_response(query, context, current_response, iteration)
            
            if refined_response != current_response:
                current_response = refined_response
                refinement_history.append(f"Iteration {iteration + 1}: {refined_response}")
            else:
                # No improvement, stop iterating
                break
        
        confidence = self._estimate_confidence(current_response, context) + 0.1  # Slight boost for refinement
        
        return GenerationResult(
            response=current_response,
            confidence_score=min(confidence, 1.0),
            source_chunks=context,
            generation_time=time.time() - start_time,
            reasoning_trace=refinement_history,
            metadata={
                "strategy": "iterative_refinement",
                "iterations": len(refinement_history) - 1
            }
        )
    
    def _refine_response(self, query: str, context: List[str], current_response: str, iteration: int) -> str:
        """Refine the current response"""
        formatted_context = self._format_context(context)
        
        refinement_prompt = f"""Review and improve the following response to make it more accurate, comprehensive, and well-structured.

Original Question: {query}

Context:
{formatted_context}

Current Response:
{current_response}

Please provide an improved version that:
1. Is more accurate and comprehensive
2. Better utilizes the provided context
3. Is clearer and more well-structured
4. Addresses any potential gaps or issues

Improved Response:"""
        
        refined = self._call_llm(refinement_prompt)
        
        # Simple check to avoid degradation
        if len(refined.strip()) < len(current_response.strip()) * 0.5:
            return current_response  # Reject if too short
        
        return refined.strip()

class EnsembleGenerator(ResponseGenerator):
    """Generator that combines multiple generation strategies"""
    
    def __init__(self, config: GenerationConfig, llm_client=None, generators: Optional[List[ResponseGenerator]] = None):
        super().__init__(config, llm_client)
        self.generators = generators or [
            BasicRAGGenerator(config, llm_client),
            ChainOfThoughtGenerator(config, llm_client)
        ]
    
    def generate(self, query: str, context: List[str], **kwargs) -> GenerationResult:
        """Generate response using ensemble of generators"""
        start_time = time.time()
        
        # Generate responses from all generators
        results = []
        for generator in self.generators:
            try:
                result = generator.generate(query, context, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Generator {type(generator).__name__} failed: {e}")
        
        if not results:
            return GenerationResult(
                response="I apologize, but I'm unable to generate a response.",
                confidence_score=0.0,
                source_chunks=context,
                generation_time=time.time() - start_time
            )
        
        # Select best response based on confidence
        best_result = max(results, key=lambda x: x.confidence_score)
        
        # Combine metadata from all generators
        ensemble_metadata = {
            "strategy": "ensemble",
            "num_generators": len(results),
            "generator_results": [
                {
                    "generator": type(self.generators[i]).__name__,
                    "confidence": results[i].confidence_score,
                    "response_length": len(results[i].response)
                }
                for i in range(len(results))
            ]
        }
        
        best_result.metadata.update(ensemble_metadata)
        best_result.generation_time = time.time() - start_time
        
        return best_result

class GeneratorFactory:
    """Factory for creating response generators"""
    
    @staticmethod
    def create_generator(strategy: GenerationStrategy, config: GenerationConfig, 
                        llm_client=None, **kwargs) -> ResponseGenerator:
        """Create generator based on strategy"""
        
        if strategy == GenerationStrategy.BASIC_RAG:
            return BasicRAGGenerator(config, llm_client)
        elif strategy == GenerationStrategy.CHAIN_OF_THOUGHT:
            return ChainOfThoughtGenerator(config, llm_client)
        elif strategy == GenerationStrategy.MULTI_HOP_REASONING:
            retriever = kwargs.get('retriever')
            return MultiHopReasoningGenerator(config, llm_client, retriever)
        elif strategy == GenerationStrategy.SELF_CONSISTENCY:
            return SelfConsistencyGenerator(config, llm_client)
        elif strategy == GenerationStrategy.ITERATIVE_REFINEMENT:
            return IterativeRefinementGenerator(config, llm_client)
        elif strategy == GenerationStrategy.ENSEMBLE_GENERATION:
            generators = kwargs.get('generators')
            return EnsembleGenerator(config, llm_client, generators)
        else:
            raise ValueError(f"Unsupported generation strategy: {strategy}")
    
    @staticmethod
    def create_optimized_generator(llm_client=None, retriever=None) -> ResponseGenerator:
        """Create optimized generator with best practices"""
        config = GenerationConfig(
            strategy=GenerationStrategy.ENSEMBLE_GENERATION,
            max_tokens=512,
            temperature=0.7,
            enable_reasoning_traces=True,
            confidence_threshold=0.8
        )
        
        # Create individual generators
        basic_config = GenerationConfig(strategy=GenerationStrategy.BASIC_RAG, temperature=0.5)
        cot_config = GenerationConfig(strategy=GenerationStrategy.CHAIN_OF_THOUGHT, temperature=0.7)
        
        generators = [
            BasicRAGGenerator(basic_config, llm_client),
            ChainOfThoughtGenerator(cot_config, llm_client)
        ]
        
        if retriever:
            multihop_config = GenerationConfig(
                strategy=GenerationStrategy.MULTI_HOP_REASONING,
                max_reasoning_steps=2
            )
            generators.append(MultiHopReasoningGenerator(multihop_config, llm_client, retriever))
        
        return EnsembleGenerator(config, llm_client, generators)
```

### 7.2 Advanced Response Post-Processing

```python
from typing import List, Dict, Any, Optional, Set
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class PostProcessingConfig:
    """Configuration for response post-processing"""
    enable_fact_checking: bool = False
    enable_citation_addition: bool = True
    enable_formatting: bool = True
    enable_length_optimization: bool = False
    target_length: Optional[int] = None
    enable_bias_detection: bool = False
    enable_toxicity_filtering: bool = False
    custom_filters: List[Callable] = None

class ResponsePostProcessor(ABC):
    """Abstract base class for response post-processors"""
    
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def process(self, result: GenerationResult, query: str, context: List[str]) -> GenerationResult:
        """Post-process the generation result"""
        pass

class CitationProcessor(ResponsePostProcessor):
    """Add citations to response based on source chunks"""
    
    def process(self, result: GenerationResult, query: str, context: List[str]) -> GenerationResult:
        """Add citations to the response"""
        if not self.config.enable_citation_addition or not context:
            return result
        
        response = result.response
        cited_response = self._add_citations(response, context)
        
        result.response = cited_response
        result.metadata['citations_added'] = True
        result.metadata['num_citations'] = len(context)
        
        return result
    
    def _add_citations(self, response: str, context: List[str]) -> str:
        """Add citation markers to response"""
        # Simple approach: add citations at the end
        citations = []
        for i, chunk in enumerate(context, 1):
            # Truncate long chunks for citations
            truncated_chunk = chunk[:100] + "..." if len(chunk) > 100 else chunk
            citations.append(f"[{i}] {truncated_chunk}")
        
        if citations:
            citations_text = "\n\nSources:\n" + "\n".join(citations)
            return response + citations_text
        
        return response

class FactCheckProcessor(ResponsePostProcessor):
    """Verify facts in the response against the context"""
    
    def process(self, result: GenerationResult, query: str, context: List[str]) -> GenerationResult:
        """Perform basic fact checking"""
        if not self.config.enable_fact_checking:
            return result
        
        fact_check_score = self._check_facts(result.response, context)
        
        result.metadata['fact_check_score'] = fact_check_score
        result.confidence_score *= fact_check_score  # Adjust confidence based on fact check
        
        return result
    
    def _check_facts(self, response: str, context: List[str]) -> float:
        """Simple fact checking based on context overlap"""
        if not context:
            return 0.5  # Neutral score if no context
        
        context_text = " ".join(context).lower()
        response_text = response.lower()
        
        # Extract potential facts (simple noun phrases)
        response_words = set(response_text.split())
        context_words = set(context_text.split())
        
        # Calculate overlap ratio
        overlap = len(response_words & context_words)
        total_words = len(response_words)
        
        if total_words == 0:
            return 0.5
        
        overlap_ratio = overlap / total_words
        return min(1.0, overlap_ratio * 2)  # Scale to reasonable range

class FormattingProcessor(ResponsePostProcessor):
    """Format response for better readability"""
    
    def process(self, result: GenerationResult, query: str, context: List[str]) -> GenerationResult:
        """Format the response for better presentation"""
        if not self.config.enable_formatting:
            return result
        
        formatted_response = self._format_response(result.response)
        
        result.response = formatted_response
        result.metadata['formatted'] = True
        
        return result
    
    def _format_response(self, response: str) -> str:
        """Apply formatting improvements"""
        # Remove extra whitespace
        formatted = re.sub(r'\s+', ' ', response).strip()
        
        # Ensure proper sentence spacing
        formatted = re.sub(r'\.([A-Z])', r'. \1', formatted)
        
        # Add proper paragraph breaks for long responses
        sentences = formatted.split('. ')
        if len(sentences) > 4:
            # Group sentences into paragraphs
            paragraphs = []
            current_paragraph = []
            
            for sentence in sentences:
                current_paragraph.append(sentence)
                if len(current_paragraph) >= 3:  # 3 sentences per paragraph
                    paragraphs.append('. '.join(current_paragraph) + '.')
                    current_paragraph = []
            
            if current_paragraph:
                paragraphs.append('. '.join(current_paragraph))
            
            formatted = '\n\n'.join(paragraphs)
        
        return formatted

class LengthOptimizationProcessor(ResponsePostProcessor):
    """Optimize response length based on target"""
    
    def process(self, result: GenerationResult, query: str, context: List[str]) -> GenerationResult:
        """Optimize response length"""
        if not self.config.enable_length_optimization or not self.config.target_length:
            return result
        
        optimized_response = self._optimize_length(result.response, self.config.target_length)
        
        result.response = optimized_response
        result.metadata['length_optimized'] = True
        result.metadata['original_length'] = len(result.response.split())
        result.metadata['target_length'] = self.config.target_length
        
        return result
    
    def _optimize_length(self, response: str, target_length: int) -> str:
        """Optimize response to target length"""
        words = response.split()
        current_length = len(words)
        
        if current_length <= target_length:
            return response  # Already within target
        
        # Simple truncation with sentence boundary preservation
        truncated_words = words[:target_length]
        truncated_text = ' '.join(truncated_words)
        
        # Find last complete sentence
        last_period = truncated_text.rfind('.')
        if last_period > len(truncated_text) * 0.7:  # If we found a sentence end in the last 30%
            return truncated_text[:last_period + 1]
        
        return truncated_text + "..."

class QualityAssuranceProcessor(ResponsePostProcessor):
    """Comprehensive quality assurance for responses"""
    
    def process(self, result: GenerationResult, query: str, context: List[str]) -> GenerationResult:
        """Perform comprehensive quality checks"""
        quality_score = self._assess_quality(result.response, query, context)
        
        result.metadata['quality_score'] = quality_score
        result.confidence_score = (result.confidence_score + quality_score) / 2
        
        # Add quality issues if found
        issues = self._identify_issues(result.response, query)
        if issues:
            result.metadata['quality_issues'] = issues
        
        return result
    
    def _assess_quality(self, response: str, query: str, context: List[str]) -> float:
        """Assess overall response quality"""
        quality_factors = []
        
        # Length appropriateness (not too short or too long)
        word_count = len(response.split())
        if 10 <= word_count <= 300:
            quality_factors.append(0.8)
        elif word_count < 5:
            quality_factors.append(0.2)
        else:
            quality_factors.append(0.6)
        
        # Coherence (basic check for repetition)
        sentences = response.split('.')
        unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
        if len(sentences) > 0:
            coherence_score = len(unique_sentences) / len([s for s in sentences if s.strip()])
            quality_factors.append(coherence_score)
        
        # Relevance to query (simple keyword overlap)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        relevance_score = len(query_words & response_words) / len(query_words) if query_words else 0
        quality_factors.append(min(relevance_score * 2, 1.0))
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5
    
    def _identify_issues(self, response: str, query: str) -> List[str]:
        """Identify potential quality issues"""
        issues = []
        
        # Check for very short responses
        if len(response.split()) < 5:
            issues.append("Response too short")
        
        # Check for repetitive content
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) != len(set(sentences)):
            issues.append("Repetitive content detected")
        
        # Check for incomplete sentences
        if not response.strip().endswith(('.', '!', '?')):
            issues.append("Incomplete final sentence")
        
        # Check for placeholder text
        placeholders = ['[placeholder]', 'TODO', 'XXX', 'example.com']
        if any(placeholder in response.lower() for placeholder in placeholders):
            issues.append("Placeholder content detected")
        
        return issues

class ResponsePostProcessingPipeline:
    """Pipeline for comprehensive response post-processing"""
    
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        self.processors = self._create_processors()
        self.logger = logging.getLogger(__name__)
    
    def _create_processors(self) -> List[ResponsePostProcessor]:
        """Create post-processing pipeline"""
        processors = []
        
        # Always include quality assurance
        processors.append(QualityAssuranceProcessor(self.config))
        
        if self.config.enable_citation_addition:
            processors.append(CitationProcessor(self.config))
        
        if self.config.enable_fact_checking:
            processors.append(FactCheckProcessor(self.config))
        
        if self.config.enable_formatting:
            processors.append(FormattingProcessor(self.config))
        
        if self.config.enable_length_optimization:
            processors.append(LengthOptimizationProcessor(self.config))
        
        return processors
    
    def process(self, result: GenerationResult, query: str, context: List[str]) -> GenerationResult:
        """Process result through all post-processors"""
        processed_result = result
        
        for processor in self.processors:
            try:
                processed_result = processor.process(processed_result, query, context)
            except Exception as e:
                self.logger.error(f"Post-processor {type(processor).__name__} failed: {e}")
        
        return processed_result

def demonstrate_generation_strategies():
    """Demonstrate different generation strategies"""
    print("=== Generation Strategies Demonstration ===")
    
    # Mock context and query
    sample_query = "What are the benefits of using microservices architecture?"
    sample_context = [
        "Microservices architecture allows teams to develop and deploy services independently.",
        "Each microservice can be scaled individually based on demand.",
        "Microservices enable technology diversity, allowing different services to use different tech stacks."
    ]
    
    # Test different generation strategies
    strategies = [
        (GenerationStrategy.BASIC_RAG, "Basic RAG"),
        (GenerationStrategy.CHAIN_OF_THOUGHT, "Chain of Thought"),
        (GenerationStrategy.SELF_CONSISTENCY, "Self Consistency")
    ]
    
    for strategy, name in strategies:
        print(f"\n{name} Strategy:")
        
        config = GenerationConfig(strategy=strategy)
        generator = GeneratorFactory.create_generator(strategy, config)
        
        try:
            result = generator.generate(sample_query, sample_context)
            print(f"  Response: {result.response}")
            print(f"  Confidence: {result.confidence_score:.2f}")
            print(f"  Generation time: {result.generation_time:.3f}s")
            
            if result.reasoning_trace:
                print(f"  Reasoning steps: {len(result.reasoning_trace)}")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    # Demonstrate post-processing
    print("\n=== Post-Processing Demonstration ===")
    
    # Create a basic result for post-processing
    basic_config = GenerationConfig()
    basic_generator = BasicRAGGenerator(basic_config)
    basic_result = basic_generator.generate(sample_query, sample_context)
    
    # Apply post-processing
    pp_config = PostProcessingConfig(
        enable_citation_addition=True,
        enable_formatting=True,
        enable_fact_checking=True
    )
    
    pipeline = ResponsePostProcessingPipeline(pp_config)
    processed_result = pipeline.process(basic_result, sample_query, sample_context)
    
    print(f"Original response: {basic_result.response}")
    print(f"Processed response: {processed_result.response}")
    print(f"Quality improvements: {processed_result.metadata}")

if __name__ == "__main__":
    demonstrate_generation_strategies()
```

## Chapter 8: Advanced RAG Patterns

Advanced RAG patterns extend beyond basic retrieval-augmented generation to handle complex scenarios, improve performance, and provide specialized functionality for enterprise applications.

### 8.1 Multi-Modal RAG Systems

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import base64
from PIL import Image
import io
import json
import logging

class ModalityType(Enum):
    """Types of modalities supported"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABLE = "table"
    CODE = "code"

@dataclass
class MultiModalDocument:
    """Document with multiple modalities"""
    doc_id: str
    modalities: Dict[ModalityType, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)  # Cross-modal relationships
    
    def get_text_content(self) -> Optional[str]:
        """Extract text content from document"""
        return self.modalities.get(ModalityType.TEXT)
    
    def get_image_content(self) -> Optional[Any]:
        """Extract image content from document"""
        return self.modalities.get(ModalityType.IMAGE)
    
    def has_modality(self, modality: ModalityType) -> bool:
        """Check if document contains specific modality"""
        return modality in self.modalities

class MultiModalEmbedding(ABC):
    """Abstract base for multi-modal embedding models"""
    
    @abstractmethod
    def encode_text(self, texts: List[str]) -> List[List[float]]:
        """Encode text into embeddings"""
        pass
    
    @abstractmethod
    def encode_image(self, images: List[Any]) -> List[List[float]]:
        """Encode images into embeddings"""
        pass
    
    @abstractmethod
    def encode_multimodal(self, documents: List[MultiModalDocument]) -> List[List[float]]:
        """Encode multi-modal documents into unified embeddings"""
        pass

class CLIPBasedEmbedding(MultiModalEmbedding):
    """CLIP-based multi-modal embedding implementation"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        try:
            import clip
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            self.logger = logging.getLogger(__name__)
        except ImportError:
            raise RuntimeError("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
    
    def encode_text(self, texts: List[str]) -> List[List[float]]:
        """Encode text using CLIP text encoder"""
        import torch
        
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(self.device)
            text_embeddings = self.model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
        return text_embeddings.cpu().numpy().tolist()
    
    def encode_image(self, images: List[Any]) -> List[List[float]]:
        """Encode images using CLIP image encoder"""
        import torch
        
        processed_images = []
        for image in images:
            if isinstance(image, str):
                # Assume it's a base64 encoded image or file path
                if image.startswith('data:image'):
                    # Base64 encoded image
                    image_data = base64.b64decode(image.split(',')[1])
                    image = Image.open(io.BytesIO(image_data))
                else:
                    # File path
                    image = Image.open(image)
            
            processed_images.append(self.preprocess(image))
        
        with torch.no_grad():
            image_batch = torch.stack(processed_images).to(self.device)
            image_embeddings = self.model.encode_image(image_batch)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        
        return image_embeddings.cpu().numpy().tolist()
    
    def encode_multimodal(self, documents: List[MultiModalDocument]) -> List[List[float]]:
        """Encode multi-modal documents with fusion strategy"""
        embeddings = []
        
        for doc in documents:
            doc_embeddings = []
            
            # Encode text content
            if doc.has_modality(ModalityType.TEXT):
                text_content = doc.get_text_content()
                if text_content:
                    text_emb = self.encode_text([text_content])[0]
                    doc_embeddings.append(text_emb)
            
            # Encode image content
            if doc.has_modality(ModalityType.IMAGE):
                image_content = doc.get_image_content()
                if image_content:
                    image_emb = self.encode_image([image_content])[0]
                    doc_embeddings.append(image_emb)
            
            # Fusion strategy: average embeddings
            if doc_embeddings:
                import numpy as np
                fused_embedding = np.mean(doc_embeddings, axis=0).tolist()
                embeddings.append(fused_embedding)
            else:
                # Fallback: zero embedding
                embeddings.append([0.0] * 512)  # CLIP embedding dimension
        
        return embeddings

class MultiModalRetriever:
    """Retriever for multi-modal documents"""
    
    def __init__(self, embedding_model: MultiModalEmbedding, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
    
    def retrieve(self, query: Union[str, MultiModalDocument], top_k: int = 5, 
                modality_filter: Optional[List[ModalityType]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant multi-modal documents"""
        
        # Generate query embedding
        if isinstance(query, str):
            query_embedding = self.embedding_model.encode_text([query])[0]
        elif isinstance(query, MultiModalDocument):
            query_embedding = self.embedding_model.encode_multimodal([query])[0]
        else:
            raise ValueError("Query must be string or MultiModalDocument")
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k * 2)  # Get more for filtering
        
        # Filter by modality if specified
        if modality_filter:
            filtered_results = []
            for result in results:
                doc_modalities = result.get('metadata', {}).get('modalities', [])
                if any(mod.value in doc_modalities for mod in modality_filter):
                    filtered_results.append(result)
                    if len(filtered_results) >= top_k:
                        break
            results = filtered_results
        else:
            results = results[:top_k]
        
        return results
    
    def cross_modal_search(self, text_query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Search across different modalities using text query"""
        text_embedding = self.embedding_model.encode_text([text_query])[0]
        
        results = {}
        for modality in ModalityType:
            modality_results = self.retrieve(
                text_query, 
                top_k, 
                modality_filter=[modality]
            )
            results[modality.value] = modality_results
        
        return results
```

### 8.2 Conversational RAG with Memory

```python
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import uuid

@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    turn_id: str
    user_query: str
    system_response: str
    retrieved_context: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationMemory:
    """Memory system for conversational RAG"""
    conversation_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    context_window: int = 5  # Number of turns to keep in active memory
    
    def add_turn(self, user_query: str, system_response: str, 
                 retrieved_context: List[str], metadata: Optional[Dict[str, Any]] = None):
        """Add a new conversation turn"""
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            user_query=user_query,
            system_response=system_response,
            retrieved_context=retrieved_context,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.turns.append(turn)
        
        # Maintain context window
        if len(self.turns) > self.context_window:
            self.turns = self.turns[-self.context_window:]
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context"""
        context_parts = []
        for turn in self.turns:
            context_parts.append(f"User: {turn.user_query}")
            context_parts.append(f"Assistant: {turn.system_response}")
        return "\n".join(context_parts)
    
    def get_recent_queries(self, n: int = 3) -> List[str]:
        """Get recent user queries"""
        return [turn.user_query for turn in self.turns[-n:]]
    
    def extract_entities(self) -> Dict[str, List[str]]:
        """Extract entities mentioned in conversation"""
        # Simple entity extraction (can be enhanced with NER)
        entities = {"topics": [], "names": [], "dates": []}
        
        for turn in self.turns:
            # This is a simplified example - use proper NER in production
            words = turn.user_query.split() + turn.system_response.split()
            
            # Extract capitalized words as potential entities
            for word in words:
                if word[0].isupper() and len(word) > 2:
                    entities["names"].append(word)
        
        return entities

class ConversationalRAG:
    """Conversational RAG system with memory"""
    
    def __init__(self, retriever, generator, memory_store=None):
        self.retriever = retriever
        self.generator = generator
        self.memory_store = memory_store or {}
        self.logger = logging.getLogger(__name__)
    
    def get_or_create_conversation(self, conversation_id: str) -> ConversationMemory:
        """Get existing conversation or create new one"""
        if conversation_id not in self.memory_store:
            self.memory_store[conversation_id] = ConversationMemory(conversation_id)
        return self.memory_store[conversation_id]
    
    def query_with_memory(self, query: str, conversation_id: str, 
                         top_k: int = 5) -> Dict[str, Any]:
        """Query with conversational memory"""
        memory = self.get_or_create_conversation(conversation_id)
        
        # Enhanced query with conversation context
        enhanced_query = self._enhance_query_with_context(query, memory)
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(enhanced_query, top_k)
        context = [doc.get('content', '') for doc in retrieved_docs]
        
        # Add conversation context to generation
        conversation_context = memory.get_conversation_context()
        
        # Generate response with conversation awareness
        response = self._generate_with_memory(
            query, context, conversation_context, memory
        )
        
        # Store turn in memory
        memory.add_turn(
            user_query=query,
            system_response=response,
            retrieved_context=context,
            metadata={
                'enhanced_query': enhanced_query,
                'retrieved_doc_count': len(retrieved_docs)
            }
        )
        
        return {
            'response': response,
            'context': context,
            'conversation_id': conversation_id,
            'turn_count': len(memory.turns)
        }
    
    def _enhance_query_with_context(self, query: str, memory: ConversationMemory) -> str:
        """Enhance current query with conversation context"""
        if not memory.turns:
            return query
        
        # Get recent context
        recent_queries = memory.get_recent_queries(2)
        entities = memory.extract_entities()
        
        # Simple context enhancement
        context_parts = []
        if recent_queries:
            context_parts.append(f"Previous questions: {'; '.join(recent_queries)}")
        
        if entities["names"]:
            context_parts.append(f"Mentioned entities: {', '.join(set(entities['names']))}")
        
        if context_parts:
            enhanced_query = f"Context: {' | '.join(context_parts)}\nCurrent question: {query}"
        else:
            enhanced_query = query
        
        return enhanced_query
    
    def _generate_with_memory(self, query: str, context: List[str], 
                             conversation_context: str, memory: ConversationMemory) -> str:
        """Generate response with conversation memory"""
        
        # Create memory-aware prompt
        prompt = f"""You are a helpful assistant engaged in a conversation. Use the provided context and conversation history to answer the user's question.

Conversation History:
{conversation_context}

Retrieved Context:
{chr(10).join(f'[{i+1}] {doc}' for i, doc in enumerate(context))}

Current Question: {query}

Instructions:
- Reference previous conversation when relevant
- Use the retrieved context to provide accurate information
- Maintain conversation continuity
- If clarification is needed based on previous turns, ask for it

Response:"""
        
        # Use generator to create response
        if hasattr(self.generator, 'generate'):
            result = self.generator.generate(query, context, custom_prompt=prompt)
            return result.response if hasattr(result, 'response') else result
        else:
            # Fallback for simple generator
            return "I understand your question in the context of our conversation and will provide a relevant response based on the available information."
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get summary of conversation"""
        if conversation_id not in self.memory_store:
            return {"error": "Conversation not found"}
        
        memory = self.memory_store[conversation_id]
        
        return {
            "conversation_id": conversation_id,
            "turn_count": len(memory.turns),
            "duration": (memory.turns[-1].timestamp - memory.turns[0].timestamp).total_seconds() if memory.turns else 0,
            "entities": memory.extract_entities(),
            "recent_topics": [turn.user_query[:50] + "..." for turn in memory.turns[-3:]]
        }
```

### 8.3 Hierarchical RAG for Complex Documents

```python
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
from enum import Enum

class DocumentLevel(Enum):
    """Levels in document hierarchy"""
    DOCUMENT = "document"
    CHAPTER = "chapter" 
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"

@dataclass
class HierarchicalNode:
    """Node in document hierarchy"""
    node_id: str
    level: DocumentLevel
    title: str
    content: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child_id: str):
        """Add child node"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def get_full_path(self, node_store: Dict[str, 'HierarchicalNode']) -> str:
        """Get full hierarchical path"""
        path = [self.title]
        current_id = self.parent_id
        
        while current_id:
            parent = node_store.get(current_id)
            if parent:
                path.append(parent.title)
                current_id = parent.parent_id
            else:
                break
        
        return " > ".join(reversed(path))

class HierarchicalDocumentProcessor:
    """Process documents into hierarchical structure"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_document(self, content: str, doc_id: str) -> Dict[str, HierarchicalNode]:
        """Process document into hierarchical nodes"""
        nodes = {}
        
        # Create root document node
        root_node = HierarchicalNode(
            node_id=f"{doc_id}_root",
            level=DocumentLevel.DOCUMENT,
            title=f"Document {doc_id}",
            content=content[:200] + "..." if len(content) > 200 else content
        )
        nodes[root_node.node_id] = root_node
        
        # Detect structure using headers
        hierarchy = self._detect_document_structure(content)
        
        # Build hierarchical nodes
        current_parents = {DocumentLevel.DOCUMENT: root_node.node_id}
        
        for item in hierarchy:
            level = item['level']
            title = item['title']
            content_text = item['content']
            
            # Generate node ID
            node_id = f"{doc_id}_{level.value}_{len(nodes)}"
            
            # Determine parent
            parent_levels = [DocumentLevel.DOCUMENT, DocumentLevel.CHAPTER, 
                           DocumentLevel.SECTION, DocumentLevel.SUBSECTION]
            parent_id = None
            
            for parent_level in reversed(parent_levels):
                if parent_level.value < level.value and parent_level in current_parents:
                    parent_id = current_parents[parent_level]
                    break
            
            # Create node
            node = HierarchicalNode(
                node_id=node_id,
                level=level,
                title=title,
                content=content_text,
                parent_id=parent_id,
                metadata={'section_number': len(nodes)}
            )
            
            nodes[node_id] = node
            current_parents[level] = node_id
            
            # Update parent's children
            if parent_id and parent_id in nodes:
                nodes[parent_id].add_child(node_id)
        
        return nodes
    
    def _detect_document_structure(self, content: str) -> List[Dict[str, Any]]:
        """Detect document structure from content"""
        lines = content.split('\n')
        structure = []
        current_content = []
        
        for line in lines:
            line = line.strip()
            
            # Detect headers
            header_level = self._detect_header_level(line)
            
            if header_level:
                # Save previous section if exists
                if current_content:
                    structure.append({
                        'level': DocumentLevel.PARAGRAPH,
                        'title': 'Content',
                        'content': '\n'.join(current_content)
                    })
                    current_content = []
                
                # Add header section
                structure.append({
                    'level': header_level,
                    'title': self._clean_header(line),
                    'content': line
                })
            else:
                current_content.append(line)
        
        # Add remaining content
        if current_content:
            structure.append({
                'level': DocumentLevel.PARAGRAPH,
                'title': 'Content',
                'content': '\n'.join(current_content)
            })
        
        return structure
    
    def _detect_header_level(self, line: str) -> Optional[DocumentLevel]:
        """Detect header level from line"""
        # Markdown headers
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            if level == 1:
                return DocumentLevel.CHAPTER
            elif level == 2:
                return DocumentLevel.SECTION
            elif level >= 3:
                return DocumentLevel.SUBSECTION
        
        # Numbered sections
        if re.match(r'^\d+\.\s+', line):
            return DocumentLevel.SECTION
        elif re.match(r'^\d+\.\d+\.\s+', line):
            return DocumentLevel.SUBSECTION
        
        return None
    
    def _clean_header(self, line: str) -> str:
        """Clean header text"""
        # Remove markdown hash symbols
        cleaned = re.sub(r'^#+\s*', '', line)
        # Remove numbered prefixes
        cleaned = re.sub(r'^\d+(\.\d+)*\.\s*', '', cleaned)
        return cleaned.strip()

class HierarchicalRAGRetriever:
    """RAG retriever with hierarchical awareness"""
    
    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.node_store = {}
        self.logger = logging.getLogger(__name__)
    
    def index_hierarchical_document(self, nodes: Dict[str, HierarchicalNode]):
        """Index hierarchical document nodes"""
        # Store nodes
        self.node_store.update(nodes)
        
        # Create embeddings for each node
        for node in nodes.values():
            # Create contextual content including hierarchy
            contextual_content = self._create_contextual_content(node)
            
            # Generate embedding
            embedding = self.embedding_model.encode([contextual_content])[0]
            
            # Store in vector database
            self.vector_store.upsert([{
                'id': node.node_id,
                'values': embedding,
                'metadata': {
                    'content': node.content,
                    'title': node.title,
                    'level': node.level.value,
                    'parent_id': node.parent_id,
                    'full_path': node.get_full_path(self.node_store)
                }
            }])
    
    def _create_contextual_content(self, node: HierarchicalNode) -> str:
        """Create contextual content including hierarchy"""
        context_parts = []
        
        # Add hierarchical path
        full_path = node.get_full_path(self.node_store)
        context_parts.append(f"Section: {full_path}")
        
        # Add parent context
        if node.parent_id and node.parent_id in self.node_store:
            parent = self.node_store[node.parent_id]
            context_parts.append(f"Parent context: {parent.content[:100]}...")
        
        # Add current content
        context_parts.append(f"Content: {node.content}")
        
        return '\n'.join(context_parts)
    
    def hierarchical_retrieve(self, query: str, top_k: int = 5, 
                            level_weights: Optional[Dict[DocumentLevel, float]] = None) -> List[Dict[str, Any]]:
        """Retrieve with hierarchical awareness"""
        
        # Default level weights (higher = more important)
        if not level_weights:
            level_weights = {
                DocumentLevel.DOCUMENT: 0.1,
                DocumentLevel.CHAPTER: 0.8,
                DocumentLevel.SECTION: 1.0,
                DocumentLevel.SUBSECTION: 0.9,
                DocumentLevel.PARAGRAPH: 0.7,
                DocumentLevel.SENTENCE: 0.5
            }
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search vector store
        raw_results = self.vector_store.search(query_embedding, top_k * 3)
        
        # Re-rank with hierarchical weights
        weighted_results = []
        for result in raw_results:
            level_str = result.get('metadata', {}).get('level', 'paragraph')
            try:
                level = DocumentLevel(level_str)
                weight = level_weights.get(level, 0.5)
                weighted_score = result['score'] * weight
                
                result['hierarchical_score'] = weighted_score
                weighted_results.append(result)
            except ValueError:
                # Unknown level, use default weight
                result['hierarchical_score'] = result['score'] * 0.5
                weighted_results.append(result)
        
        # Sort by hierarchical score and return top_k
        weighted_results.sort(key=lambda x: x['hierarchical_score'], reverse=True)
        
        return weighted_results[:top_k]
    
    def get_hierarchical_context(self, node_ids: List[str]) -> str:
        """Get hierarchical context for multiple nodes"""
        context_parts = []
        
        for node_id in node_ids:
            if node_id in self.node_store:
                node = self.node_store[node_id]
                full_path = node.get_full_path(self.node_store)
                context_parts.append(f"[{full_path}]\n{node.content}\n")
        
        return '\n'.join(context_parts)
```

### 8.4 RAG with Knowledge Graph Integration

```python
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json

@dataclass
class KnowledgeEntity:
    """Entity in knowledge graph"""
    entity_id: str
    entity_type: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class KnowledgeRelation:
    """Relation in knowledge graph"""
    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

class KnowledgeGraph:
    """Simple knowledge graph implementation"""
    
    def __init__(self):
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.relations: Dict[str, KnowledgeRelation] = {}
        self.entity_relations: Dict[str, List[str]] = {}  # entity_id -> relation_ids
    
    def add_entity(self, entity: KnowledgeEntity):
        """Add entity to graph"""
        self.entities[entity.entity_id] = entity
        if entity.entity_id not in self.entity_relations:
            self.entity_relations[entity.entity_id] = []
    
    def add_relation(self, relation: KnowledgeRelation):
        """Add relation to graph"""
        self.relations[relation.relation_id] = relation
        
        # Update entity relations
        for entity_id in [relation.subject_id, relation.object_id]:
            if entity_id not in self.entity_relations:
                self.entity_relations[entity_id] = []
            self.entity_relations[entity_id].append(relation.relation_id)
    
    def get_entity_neighbors(self, entity_id: str, max_hops: int = 1) -> Set[str]:
        """Get neighboring entities within max_hops"""
        neighbors = set()
        current_level = {entity_id}
        
        for hop in range(max_hops):
            next_level = set()
            for current_entity in current_level:
                relation_ids = self.entity_relations.get(current_entity, [])
                for rel_id in relation_ids:
                    relation = self.relations[rel_id]
                    if relation.subject_id == current_entity:
                        next_level.add(relation.object_id)
                    else:
                        next_level.add(relation.subject_id)
            
            neighbors.update(next_level)
            current_level = next_level
        
        neighbors.discard(entity_id)  # Remove self
        return neighbors
    
    def get_relations_for_entities(self, entity_ids: List[str]) -> List[KnowledgeRelation]:
        """Get relations involving specified entities"""
        relations = []
        entity_set = set(entity_ids)
        
        for relation in self.relations.values():
            if relation.subject_id in entity_set or relation.object_id in entity_set:
                relations.append(relation)
        
        return relations
    
    def query_subgraph(self, entity_ids: List[str], max_hops: int = 1) -> Dict[str, Any]:
        """Extract subgraph around specified entities"""
        # Get all relevant entities
        all_entities = set(entity_ids)
        for entity_id in entity_ids:
            neighbors = self.get_entity_neighbors(entity_id, max_hops)
            all_entities.update(neighbors)
        
        # Get entities and relations
        subgraph_entities = {eid: self.entities[eid] for eid in all_entities if eid in self.entities}
        subgraph_relations = self.get_relations_for_entities(list(all_entities))
        
        return {
            'entities': subgraph_entities,
            'relations': subgraph_relations
        }

class KnowledgeGraphRAG:
    """RAG system with knowledge graph integration"""
    
    def __init__(self, retriever, generator, knowledge_graph: KnowledgeGraph, entity_linker=None):
        self.retriever = retriever
        self.generator = generator
        self.kg = knowledge_graph
        self.entity_linker = entity_linker
        self.logger = logging.getLogger(__name__)
    
    def query_with_kg(self, query: str, top_k: int = 5, kg_expansion_hops: int = 1) -> Dict[str, Any]:
        """Query with knowledge graph enhancement"""
        
        # Step 1: Extract entities from query
        query_entities = self._extract_entities_from_query(query)
        
        # Step 2: Expand with knowledge graph
        kg_context = ""
        if query_entities:
            subgraph = self.kg.query_subgraph(query_entities, kg_expansion_hops)
            kg_context = self._format_kg_context(subgraph)
        
        # Step 3: Enhanced retrieval
        enhanced_query = f"{query}\nRelated concepts: {', '.join(query_entities)}" if query_entities else query
        retrieved_docs = self.retriever.retrieve(enhanced_query, top_k)
        doc_context = [doc.get('content', '') for doc in retrieved_docs]
        
        # Step 4: Generate response with both contexts
        response = self._generate_with_kg_context(query, doc_context, kg_context, query_entities)
        
        return {
            'response': response,
            'document_context': doc_context,
            'kg_context': kg_context,
            'query_entities': query_entities,
            'kg_subgraph': subgraph if query_entities else None
        }
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entities from query (simplified implementation)"""
        if self.entity_linker:
            return self.entity_linker.link_entities(query)
        
        # Simple fallback: look for entities that exist in KG
        words = query.split()
        found_entities = []
        
        for entity_id, entity in self.kg.entities.items():
            entity_name = entity.name.lower()
            query_lower = query.lower()
            
            if entity_name in query_lower:
                found_entities.append(entity_id)
        
        return found_entities
    
    def _format_kg_context(self, subgraph: Dict[str, Any]) -> str:
        """Format knowledge graph context for generation"""
        context_parts = []
        
        # Format entities
        entities = subgraph.get('entities', {})
        if entities:
            context_parts.append("Related Entities:")
            for entity in entities.values():
                entity_info = f"- {entity.name} ({entity.entity_type})"
                if entity.properties:
                    props = ", ".join(f"{k}: {v}" for k, v in entity.properties.items())
                    entity_info += f" - {props}"
                context_parts.append(entity_info)
        
        # Format relations
        relations = subgraph.get('relations', [])
        if relations:
            context_parts.append("\nRelationships:")
            for relation in relations:
                subj_name = entities.get(relation.subject_id, {}).get('name', relation.subject_id)
                obj_name = entities.get(relation.object_id, {}).get('name', relation.object_id)
                context_parts.append(f"- {subj_name} {relation.predicate} {obj_name}")
        
        return '\n'.join(context_parts)
    
    def _generate_with_kg_context(self, query: str, doc_context: List[str], 
                                 kg_context: str, entities: List[str]) -> str:
        """Generate response with knowledge graph context"""
        
        # Create enhanced prompt
        prompt_parts = [
            "You are an AI assistant with access to both retrieved documents and structured knowledge.",
            "",
            "Knowledge Graph Information:",
            kg_context if kg_context else "No relevant structured knowledge found.",
            "",
            "Retrieved Documents:",
        ]
        
        for i, doc in enumerate(doc_context, 1):
            prompt_parts.append(f"[{i}] {doc}")
        
        prompt_parts.extend([
            "",
            f"Question: {query}",
            "",
            "Instructions:",
            "- Use both the knowledge graph information and retrieved documents",
            "- Prioritize structured knowledge when available",
            "- Cite sources appropriately",
            "- If information conflicts, explain the discrepancy",
            "",
            "Response:"
        ])
        
        enhanced_prompt = '\n'.join(prompt_parts)
        
        # Generate response
        if hasattr(self.generator, 'generate'):
            result = self.generator.generate(query, doc_context, custom_prompt=enhanced_prompt)
            return result.response if hasattr(result, 'response') else result
        else:
            return "Response generated using knowledge graph and retrieved context."

def demonstrate_advanced_rag_patterns():
    """Demonstrate advanced RAG patterns"""
    print("=== Advanced RAG Patterns Demonstration ===")
    
    print("\n1. Multi-Modal RAG:")
    
    # Create a multi-modal document
    multimodal_doc = MultiModalDocument(
        doc_id="doc_001",
        modalities={
            ModalityType.TEXT: "This is a description of a technical diagram showing system architecture.",
            ModalityType.IMAGE: "path/to/architecture_diagram.png"
        },
        metadata={"source": "technical_documentation"}
    )
    
    print(f"  Created multi-modal document with modalities: {list(multimodal_doc.modalities.keys())}")
    
    print("\n2. Conversational RAG:")
    
    # Mock conversational RAG
    conversation_id = "conv_123"
    conv_memory = ConversationMemory(conversation_id)
    
    # Simulate conversation turns
    conv_memory.add_turn(
        "What is microservices architecture?",
        "Microservices architecture is a design approach where applications are built as a collection of small, independent services.",
        ["Microservices are small, independent services...", "Each service has its own database..."]
    )
    
    conv_memory.add_turn(
        "How do they communicate?",
        "Microservices typically communicate through REST APIs, message queues, or event streams.",
        ["Service communication patterns...", "REST API design principles..."]
    )
    
    print(f"  Conversation with {len(conv_memory.turns)} turns")
    print(f"  Context: {conv_memory.get_conversation_context()[:100]}...")
    
    print("\n3. Hierarchical RAG:")
    
    # Create hierarchical processor
    hierarchical_processor = HierarchicalDocumentProcessor()
    
    sample_doc_content = """# Chapter 1: Introduction
This is the introduction to our technical guide.

## 1.1 Overview
This section provides an overview of the concepts.

### 1.1.1 Key Concepts
Here are the key concepts to understand.

## 1.2 Getting Started
This section helps you get started with the implementation.
"""
    
    hierarchical_nodes = hierarchical_processor.process_document(sample_doc_content, "tech_guide")
    print(f"  Created {len(hierarchical_nodes)} hierarchical nodes")
    
    for node in hierarchical_nodes.values():
        print(f"    - {node.level.value}: {node.title}")
    
    print("\n4. Knowledge Graph RAG:")
    
    # Create simple knowledge graph
    kg = KnowledgeGraph()
    
    # Add entities
    kg.add_entity(KnowledgeEntity("microservices", "architecture_pattern", "Microservices"))
    kg.add_entity(KnowledgeEntity("api_gateway", "component", "API Gateway"))
    kg.add_entity(KnowledgeEntity("service_discovery", "component", "Service Discovery"))
    
    # Add relations
    kg.add_relation(KnowledgeRelation("rel_1", "microservices", "uses", "api_gateway"))
    kg.add_relation(KnowledgeRelation("rel_2", "microservices", "requires", "service_discovery"))
    
    print(f"  Created knowledge graph with {len(kg.entities)} entities and {len(kg.relations)} relations")
    
    # Demonstrate subgraph query
    subgraph = kg.query_subgraph(["microservices"], max_hops=1)
    print(f"  Subgraph around 'microservices': {len(subgraph['entities'])} entities, {len(subgraph['relations'])} relations")
    
    print("\nAdvanced RAG patterns demonstration completed!")

if __name__ == "__main__":
    demonstrate_advanced_rag_patterns()
```

## Chapter 9: Evaluation and Monitoring

Effective evaluation and monitoring are crucial for maintaining high-quality RAG systems in production. This chapter covers comprehensive evaluation frameworks, monitoring strategies, and continuous improvement techniques.

### 9.1 RAG Evaluation Framework

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import logging
import numpy as np
from datetime import datetime, timedelta
import asyncio
import threading

class EvaluationMetric(Enum):
    """Types of evaluation metrics"""
    RETRIEVAL_ACCURACY = "retrieval_accuracy"
    GENERATION_QUALITY = "generation_quality"
    END_TO_END_QUALITY = "end_to_end_quality"
    RESPONSE_RELEVANCE = "response_relevance"
    FACTUAL_CONSISTENCY = "factual_consistency"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    GROUNDEDNESS = "groundedness"
    LATENCY = "latency"
    THROUGHPUT = "throughput"

@dataclass
class EvaluationSample:
    """Sample for evaluation"""
    query: str
    ground_truth_answer: Optional[str] = None
    ground_truth_documents: Optional[List[str]] = None
    context_documents: Optional[List[str]] = None
    generated_answer: Optional[str] = None
    retrieved_documents: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if 'id' not in self.metadata:
            self.metadata['id'] = f"sample_{int(time.time() * 1000)}"

@dataclass
class EvaluationResult:
    """Result from evaluation"""
    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    sample_scores: Optional[List[float]] = None
    evaluation_time: Optional[float] = None
    
    def __post_init__(self):
        if self.evaluation_time is None:
            self.evaluation_time = time.time()

class RAGEvaluator(ABC):
    """Abstract base class for RAG evaluators"""
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def evaluate(self, samples: List[EvaluationSample]) -> EvaluationResult:
        """Evaluate samples and return results"""
        pass
    
    def _calculate_aggregate_score(self, scores: List[float]) -> Tuple[float, Dict[str, Any]]:
        """Calculate aggregate statistics from individual scores"""
        if not scores:
            return 0.0, {}
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        median_score = np.median(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        details = {
            'mean': float(mean_score),
            'std': float(std_score),
            'median': float(median_score),
            'min': float(min_score),
            'max': float(max_score),
            'count': len(scores)
        }
        
        return float(mean_score), details

class RetrievalAccuracyEvaluator(RAGEvaluator):
    """Evaluate retrieval accuracy using precision, recall, F1"""
    
    def __init__(self):
        super().__init__("retrieval_accuracy")
    
    def evaluate(self, samples: List[EvaluationSample]) -> EvaluationResult:
        """Evaluate retrieval accuracy"""
        start_time = time.time()
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for sample in samples:
            if not sample.ground_truth_documents or not sample.retrieved_documents:
                continue
            
            # Convert to sets for comparison
            relevant_docs = set(sample.ground_truth_documents)
            retrieved_docs = set(sample.retrieved_documents)
            
            # Calculate metrics
            if retrieved_docs:
                precision = len(relevant_docs & retrieved_docs) / len(retrieved_docs)
            else:
                precision = 0.0
            
            if relevant_docs:
                recall = len(relevant_docs & retrieved_docs) / len(relevant_docs)
            else:
                recall = 0.0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        # Calculate aggregate scores
        avg_precision, precision_details = self._calculate_aggregate_score(precision_scores)
        avg_recall, recall_details = self._calculate_aggregate_score(recall_scores)
        avg_f1, f1_details = self._calculate_aggregate_score(f1_scores)
        
        # Overall score is F1
        overall_score = avg_f1
        
        details = {
            'precision': precision_details,
            'recall': recall_details,
            'f1': f1_details
        }
        
        return EvaluationResult(
            metric_name=self.metric_name,
            score=overall_score,
            details=details,
            sample_scores=f1_scores,
            evaluation_time=time.time() - start_time
        )

class ResponseRelevanceEvaluator(RAGEvaluator):
    """Evaluate response relevance using semantic similarity"""
    
    def __init__(self, similarity_model=None):
        super().__init__("response_relevance")
        self.similarity_model = similarity_model
        if not self.similarity_model:
            try:
                from sentence_transformers import SentenceTransformer
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                self.logger.warning("SentenceTransformers not available, using fallback similarity")
                self.similarity_model = None
    
    def evaluate(self, samples: List[EvaluationSample]) -> EvaluationResult:
        """Evaluate response relevance"""
        start_time = time.time()
        
        relevance_scores = []
        
        for sample in samples:
            if not sample.query or not sample.generated_answer:
                continue
            
            relevance_score = self._calculate_relevance(sample.query, sample.generated_answer)
            relevance_scores.append(relevance_score)
        
        overall_score, details = self._calculate_aggregate_score(relevance_scores)
        
        return EvaluationResult(
            metric_name=self.metric_name,
            score=overall_score,
            details=details,
            sample_scores=relevance_scores,
            evaluation_time=time.time() - start_time
        )
    
    def _calculate_relevance(self, query: str, answer: str) -> float:
        """Calculate relevance score between query and answer"""
        if self.similarity_model:
            try:
                embeddings = self.similarity_model.encode([query, answer])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return max(0.0, float(similarity))
            except Exception as e:
                self.logger.error(f"Similarity calculation failed: {e}")
        
        # Fallback: simple word overlap
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words or not answer_words:
            return 0.0
        
        overlap = len(query_words & answer_words)
        return overlap / len(query_words)

class FactualConsistencyEvaluator(RAGEvaluator):
    """Evaluate factual consistency with retrieved context"""
    
    def __init__(self):
        super().__init__("factual_consistency")
    
    def evaluate(self, samples: List[EvaluationSample]) -> EvaluationResult:
        """Evaluate factual consistency"""
        start_time = time.time()
        
        consistency_scores = []
        
        for sample in samples:
            if not sample.generated_answer or not sample.context_documents:
                continue
            
            consistency_score = self._calculate_consistency(
                sample.generated_answer, 
                sample.context_documents
            )
            consistency_scores.append(consistency_score)
        
        overall_score, details = self._calculate_aggregate_score(consistency_scores)
        
        return EvaluationResult(
            metric_name=self.metric_name,
            score=overall_score,
            details=details,
            sample_scores=consistency_scores,
            evaluation_time=time.time() - start_time
        )
    
    def _calculate_consistency(self, answer: str, context_docs: List[str]) -> float:
        """Calculate factual consistency score"""
        if not context_docs:
            return 0.0
        
        context_text = " ".join(context_docs).lower()
        answer_text = answer.lower()
        
        # Simple approach: check if answer facts appear in context
        answer_words = set(answer_text.split())
        context_words = set(context_text.split())
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        answer_content_words = answer_words - stop_words
        
        if not answer_content_words:
            return 0.5  # Neutral score
        
        supported_words = len(answer_content_words & context_words)
        consistency_ratio = supported_words / len(answer_content_words)
        
        return consistency_ratio

class LatencyEvaluator(RAGEvaluator):
    """Evaluate system latency metrics"""
    
    def __init__(self):
        super().__init__("latency")
    
    def evaluate(self, samples: List[EvaluationSample]) -> EvaluationResult:
        """Evaluate latency metrics"""
        start_time = time.time()
        
        latency_scores = []
        
        for sample in samples:
            # Extract latency information from metadata
            retrieval_time = sample.metadata.get('retrieval_time', 0)
            generation_time = sample.metadata.get('generation_time', 0)
            total_time = retrieval_time + generation_time
            
            latency_scores.append(total_time)
        
        if not latency_scores:
            return EvaluationResult(
                metric_name=self.metric_name,
                score=0.0,
                details={'error': 'No latency data available'}
            )
        
        # For latency, lower is better, so we invert the score
        max_latency = max(latency_scores)
        if max_latency > 0:
            normalized_scores = [1.0 - (lat / max_latency) for lat in latency_scores]
        else:
            normalized_scores = [1.0] * len(latency_scores)
        
        overall_score, details = self._calculate_aggregate_score(normalized_scores)
        
        # Add raw latency statistics
        details['raw_latency'] = {
            'mean': float(np.mean(latency_scores)),
            'p50': float(np.percentile(latency_scores, 50)),
            'p95': float(np.percentile(latency_scores, 95)),
            'p99': float(np.percentile(latency_scores, 99)),
            'max': float(np.max(latency_scores))
        }
        
        return EvaluationResult(
            metric_name=self.metric_name,
            score=overall_score,
            details=details,
            sample_scores=latency_scores,
            evaluation_time=time.time() - start_time
        )

class CompletenessEvaluator(RAGEvaluator):
    """Evaluate response completeness compared to ground truth"""
    
    def __init__(self):
        super().__init__("completeness")
    
    def evaluate(self, samples: List[EvaluationSample]) -> EvaluationResult:
        """Evaluate completeness"""
        start_time = time.time()
        
        completeness_scores = []
        
        for sample in samples:
            if not sample.ground_truth_answer or not sample.generated_answer:
                continue
            
            completeness_score = self._calculate_completeness(
                sample.generated_answer,
                sample.ground_truth_answer
            )
            completeness_scores.append(completeness_score)
        
        overall_score, details = self._calculate_aggregate_score(completeness_scores)
        
        return EvaluationResult(
            metric_name=self.metric_name,
            score=overall_score,
            details=details,
            sample_scores=completeness_scores,
            evaluation_time=time.time() - start_time
        )
    
    def _calculate_completeness(self, generated: str, ground_truth: str) -> float:
        """Calculate completeness score"""
        # Extract key concepts from ground truth
        gt_words = set(ground_truth.lower().split())
        gen_words = set(generated.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        gt_content_words = gt_words - stop_words
        
        if not gt_content_words:
            return 1.0  # If no content words, consider complete
        
        covered_words = len(gt_content_words & gen_words)
        completeness_ratio = covered_words / len(gt_content_words)
        
        return completeness_ratio

class RAGEvaluationSuite:
    """Comprehensive evaluation suite for RAG systems"""
    
    def __init__(self, evaluators: Optional[List[RAGEvaluator]] = None):
        self.evaluators = evaluators or self._create_default_evaluators()
        self.logger = logging.getLogger(__name__)
    
    def _create_default_evaluators(self) -> List[RAGEvaluator]:
        """Create default set of evaluators"""
        return [
            RetrievalAccuracyEvaluator(),
            ResponseRelevanceEvaluator(),
            FactualConsistencyEvaluator(),
            LatencyEvaluator(),
            CompletenessEvaluator()
        ]
    
    def evaluate(self, samples: List[EvaluationSample]) -> Dict[str, EvaluationResult]:
        """Run comprehensive evaluation"""
        results = {}
        
        for evaluator in self.evaluators:
            try:
                self.logger.info(f"Running {evaluator.metric_name} evaluation...")
                result = evaluator.evaluate(samples)
                results[evaluator.metric_name] = result
                self.logger.info(f"{evaluator.metric_name}: {result.score:.3f}")
            except Exception as e:
                self.logger.error(f"Evaluation failed for {evaluator.metric_name}: {e}")
                results[evaluator.metric_name] = EvaluationResult(
                    metric_name=evaluator.metric_name,
                    score=0.0,
                    details={'error': str(e)}
                )
        
        return results
    
    def generate_report(self, results: Dict[str, EvaluationResult]) -> str:
        """Generate comprehensive evaluation report"""
        report = "# RAG System Evaluation Report\n\n"
        report += f"**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Overall summary
        scores = [result.score for result in results.values() if result.score > 0]
        if scores:
            overall_score = np.mean(scores)
            report += f"**Overall Score**: {overall_score:.3f}\n\n"
        
        # Individual metrics
        report += "## Individual Metrics\n\n"
        for metric_name, result in results.items():
            report += f"### {metric_name.replace('_', ' ').title()}\n"
            report += f"- **Score**: {result.score:.3f}\n"
            
            if 'mean' in result.details:
                report += f"- **Mean**: {result.details['mean']:.3f}\n"
                report += f"- **Std**: {result.details['std']:.3f}\n"
                report += f"- **Min**: {result.details['min']:.3f}\n"
                report += f"- **Max**: {result.details['max']:.3f}\n"
            
            if 'raw_latency' in result.details:
                lat_stats = result.details['raw_latency']
                report += f"- **P50 Latency**: {lat_stats['p50']:.3f}s\n"
                report += f"- **P95 Latency**: {lat_stats['p95']:.3f}s\n"
                report += f"- **P99 Latency**: {lat_stats['p99']:.3f}s\n"
            
            report += "\n"
        
        return report
```

### 9.2 Real-time Monitoring System

```python
import threading
import queue
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import json
import asyncio

@dataclass
class MonitoringEvent:
    """Event for monitoring system"""
    timestamp: datetime
    event_type: str
    component: str
    data: Dict[str, Any]
    severity: str = "info"  # info, warning, error, critical
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'component': self.component,
            'data': self.data,
            'severity': self.severity
        }

class MetricsCollector:
    """Collect and aggregate metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.counters = defaultdict(int)
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        with self.lock:
            metric_key = self._create_metric_key(name, tags)
            self.metrics[metric_key].append({
                'timestamp': datetime.now(),
                'value': value,
                'tags': tags or {}
            })
    
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self.lock:
            counter_key = self._create_metric_key(name, tags)
            self.counters[counter_key] += 1
    
    def get_metrics_summary(self, lookback_minutes: int = 5) -> Dict[str, Any]:
        """Get summary of metrics over lookback period"""
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        summary = {}
        
        with self.lock:
            for metric_key, values in self.metrics.items():
                recent_values = [
                    v['value'] for v in values 
                    if v['timestamp'] >= cutoff_time
                ]
                
                if recent_values:
                    summary[metric_key] = {
                        'count': len(recent_values),
                        'mean': np.mean(recent_values),
                        'min': np.min(recent_values),
                        'max': np.max(recent_values),
                        'p50': np.percentile(recent_values, 50),
                        'p95': np.percentile(recent_values, 95),
                        'p99': np.percentile(recent_values, 99)
                    }
            
            # Add counter summaries
            for counter_key, count in self.counters.items():
                summary[f"counter_{counter_key}"] = {'total': count}
        
        return summary
    
    def _create_metric_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create unique key for metric"""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

class AlertManager:
    """Manage alerts based on metric thresholds"""
    
    def __init__(self):
        self.rules = []
        self.alerts = deque(maxlen=100)  # Keep last 100 alerts
        self.alert_callbacks = []
        self.lock = threading.Lock()
    
    def add_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool], 
                severity: str = "warning", message: str = ""):
        """Add alerting rule"""
        rule = {
            'name': name,
            'condition': condition,
            'severity': severity,
            'message': message
        }
        self.rules.append(rule)
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def check_rules(self, metrics: Dict[str, Any]):
        """Check all rules against current metrics"""
        for rule in self.rules:
            try:
                if rule['condition'](metrics):
                    alert = {
                        'timestamp': datetime.now(),
                        'rule_name': rule['name'],
                        'severity': rule['severity'],
                        'message': rule['message'],
                        'metrics_snapshot': metrics
                    }
                    
                    with self.lock:
                        self.alerts.append(alert)
                    
                    # Notify callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            print(f"Alert callback failed: {e}")
            
            except Exception as e:
                print(f"Rule check failed for {rule['name']}: {e}")
    
    def get_recent_alerts(self, lookback_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        
        with self.lock:
            return [
                alert for alert in self.alerts
                if alert['timestamp'] >= cutoff_time
            ]

class RAGMonitor:
    """Comprehensive monitoring system for RAG applications"""
    
    def __init__(self, update_interval: int = 30):
        self.update_interval = update_interval
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.event_queue = queue.Queue()
        self.running = False
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Setup default alerting rules
        self._setup_default_alerts()
    
    def start(self):
        """Start monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        self.logger.info("RAG monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("RAG monitoring stopped")
    
    def record_retrieval_metrics(self, query: str, results: List[Any], 
                                retrieval_time: float, num_results: int):
        """Record retrieval performance metrics"""
        self.metrics_collector.record_metric("retrieval_latency", retrieval_time)
        self.metrics_collector.record_metric("retrieval_results_count", num_results)
        self.metrics_collector.increment_counter("retrieval_requests")
        
        # Record event
        event = MonitoringEvent(
            timestamp=datetime.now(),
            event_type="retrieval",
            component="retriever",
            data={
                'query_length': len(query),
                'results_count': num_results,
                'latency': retrieval_time
            }
        )
        self.event_queue.put(event)
    
    def record_generation_metrics(self, query: str, context: List[str], 
                                 response: str, generation_time: float, 
                                 confidence_score: float):
        """Record generation performance metrics"""
        self.metrics_collector.record_metric("generation_latency", generation_time)
        self.metrics_collector.record_metric("generation_confidence", confidence_score)
        self.metrics_collector.record_metric("response_length", len(response))
        self.metrics_collector.increment_counter("generation_requests")
        
        # Record event
        event = MonitoringEvent(
            timestamp=datetime.now(),
            event_type="generation",
            component="generator",
            data={
                'query_length': len(query),
                'context_chunks': len(context),
                'response_length': len(response),
                'latency': generation_time,
                'confidence': confidence_score
            }
        )
        self.event_queue.put(event)
    
    def record_error(self, component: str, error: Exception, context: Dict[str, Any] = None):
        """Record error event"""
        self.metrics_collector.increment_counter("errors", {"component": component})
        
        event = MonitoringEvent(
            timestamp=datetime.now(),
            event_type="error",
            component=component,
            data={
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context or {}
            },
            severity="error"
        )
        self.event_queue.put(event)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        metrics_summary = self.metrics_collector.get_metrics_summary()
        recent_alerts = self.alert_manager.get_recent_alerts()
        
        # Calculate derived metrics
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics_summary,
            'alerts': recent_alerts,
            'health_status': self._calculate_health_status(metrics_summary),
            'summary': self._generate_summary(metrics_summary)
        }
        
        return dashboard_data
    
    def _setup_default_alerts(self):
        """Setup default alerting rules"""
        # High latency alert
        self.alert_manager.add_rule(
            name="high_retrieval_latency",
            condition=lambda m: m.get("retrieval_latency", {}).get("p95", 0) > 2.0,
            severity="warning",
            message="Retrieval P95 latency exceeds 2 seconds"
        )
        
        # Low confidence alert
        self.alert_manager.add_rule(
            name="low_generation_confidence",
            condition=lambda m: m.get("generation_confidence", {}).get("mean", 1.0) < 0.5,
            severity="warning",
            message="Average generation confidence is below 0.5"
        )
        
        # High error rate alert
        self.alert_manager.add_rule(
            name="high_error_rate",
            condition=lambda m: m.get("counter_errors", {}).get("total", 0) > 10,
            severity="error",
            message="Error count exceeds threshold"
        )
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Process events from queue
                self._process_events()
                
                # Get current metrics
                metrics = self.metrics_collector.get_metrics_summary()
                
                # Check alerting rules
                self.alert_manager.check_rules(metrics)
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(1)  # Short sleep on error
    
    def _process_events(self):
        """Process events from queue"""
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                # Here you could send events to external systems
                # like logging, metrics systems, etc.
                self.logger.debug(f"Processed event: {event.event_type}")
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")
    
    def _calculate_health_status(self, metrics: Dict[str, Any]) -> str:
        """Calculate overall system health status"""
        # Simple health calculation based on key metrics
        issues = 0
        
        # Check latency
        retrieval_latency = metrics.get("retrieval_latency", {}).get("p95", 0)
        if retrieval_latency > 2.0:
            issues += 1
        
        generation_latency = metrics.get("generation_latency", {}).get("p95", 0)
        if generation_latency > 5.0:
            issues += 1
        
        # Check confidence
        confidence = metrics.get("generation_confidence", {}).get("mean", 1.0)
        if confidence < 0.5:
            issues += 1
        
        # Check error rate
        error_count = metrics.get("counter_errors", {}).get("total", 0)
        if error_count > 5:
            issues += 1
        
        if issues == 0:
            return "healthy"
        elif issues <= 2:
            return "degraded"
        else:
            return "unhealthy"
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {}
        
        # Request counts
        retrieval_requests = metrics.get("counter_retrieval_requests", {}).get("total", 0)
        generation_requests = metrics.get("counter_generation_requests", {}).get("total", 0)
        summary["total_requests"] = retrieval_requests + generation_requests
        
        # Average latencies
        retrieval_latency = metrics.get("retrieval_latency", {}).get("mean", 0)
        generation_latency = metrics.get("generation_latency", {}).get("mean", 0)
        summary["avg_total_latency"] = retrieval_latency + generation_latency
        
        # Average confidence
        summary["avg_confidence"] = metrics.get("generation_confidence", {}).get("mean", 0)
        
        # Error rate
        error_count = metrics.get("counter_errors", {}).get("total", 0)
        if summary["total_requests"] > 0:
            summary["error_rate"] = error_count / summary["total_requests"]
        else:
            summary["error_rate"] = 0
        
        return summary

# Example usage and demonstration
def demonstrate_evaluation_monitoring():
    """Demonstrate evaluation and monitoring capabilities"""
    print("=== RAG Evaluation and Monitoring Demo ===")
    
    # Create sample evaluation data
    samples = [
        EvaluationSample(
            query="What are microservices?",
            ground_truth_answer="Microservices are small, independent services that communicate over APIs.",
            ground_truth_documents=["doc1", "doc2"],
            generated_answer="Microservices are independently deployable services that communicate via APIs.",
            retrieved_documents=["doc1", "doc3"],
            context_documents=["Microservices allow independent deployment..."],
            metadata={"retrieval_time": 0.5, "generation_time": 1.2}
        ),
        EvaluationSample(
            query="Benefits of cloud computing?",
            ground_truth_answer="Cloud computing provides scalability, cost-effectiveness, and flexibility.",
            ground_truth_documents=["doc3", "doc4"],
            generated_answer="Cloud computing offers scalable resources and cost savings.",
            retrieved_documents=["doc3", "doc4"],
            context_documents=["Cloud computing enables scalable infrastructure..."],
            metadata={"retrieval_time": 0.3, "generation_time": 0.8}
        )
    ]
    
    # Run evaluation
    print("\n1. Running Evaluation Suite...")
    evaluation_suite = RAGEvaluationSuite()
    results = evaluation_suite.evaluate(samples)
    
    for metric_name, result in results.items():
        print(f"  {metric_name}: {result.score:.3f}")
    
    # Generate report
    print("\n2. Evaluation Report:")
    report = evaluation_suite.generate_report(results)
    print(report[:500] + "...")  # Show first 500 chars
    
    # Demonstrate monitoring
    print("\n3. Monitoring System Demo...")
    monitor = RAGMonitor(update_interval=1)
    
    # Record some sample metrics
    monitor.record_retrieval_metrics("test query", ["doc1"], 0.5, 1)
    monitor.record_generation_metrics("test query", ["context"], "response", 1.0, 0.8)
    
    # Get dashboard data
    dashboard_data = monitor.get_dashboard_data()
    print(f"  Health Status: {dashboard_data['health_status']}")
    print(f"  Summary: {dashboard_data['summary']}")
    
    print("\nEvaluation and monitoring demonstration completed!")

if __name__ == "__main__":
    demonstrate_evaluation_monitoring()
```

## Chapter 10: Production Deployment and Scaling

This final chapter covers everything needed to deploy and scale RAG systems in production environments, including architecture patterns, infrastructure considerations, performance optimization, and operational best practices.

### 10.1 Production Architecture Patterns

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import aiohttp
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from contextlib import asynccontextmanager

class DeploymentPattern(Enum):
    """Deployment architecture patterns"""
    MONOLITHIC = "monolithic"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"
    HYBRID = "hybrid"

class ScalingStrategy(Enum):
    """Scaling strategies"""
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    AUTO_SCALING = "auto_scaling"
    ELASTIC = "elastic"

@dataclass
class DeploymentConfig:
    """Configuration for deployment"""
    pattern: DeploymentPattern = DeploymentPattern.MICROSERVICES
    scaling_strategy: ScalingStrategy = ScalingStrategy.HORIZONTAL
    max_replicas: int = 10
    min_replicas: int = 2
    target_cpu_utilization: float = 0.7
    target_memory_utilization: float = 0.8
    health_check_path: str = "/health"
    metrics_port: int = 8080
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_rate_limiting: bool = True
    rate_limit_rpm: int = 1000

class RAGService(ABC):
    """Abstract base class for RAG service components"""
    
    def __init__(self, service_name: str, config: DeploymentConfig):
        self.service_name = service_name
        self.config = config
        self.logger = logging.getLogger(service_name)
        self.health_status = "healthy"
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    async def initialize(self):
        """Initialize service"""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Cleanup on shutdown"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            "service_name": self.service_name,
            "health_status": self.health_status,
            "timestamp": time.time()
        }

class EmbeddingService(RAGService):
    """Microservice for embedding generation"""
    
    def __init__(self, config: DeploymentConfig, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__("embedding-service", config)
        self.model_name = model_name
        self.model = None
        self.request_count = 0
        self.error_count = 0
        self.avg_latency = 0.0
    
    async def initialize(self):
        """Initialize embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Embedding service initialized with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            self.health_status = "unhealthy"
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        start_time = time.time()
        
        try:
            if not self.model:
                raise RuntimeError("Model not initialized")
            
            embeddings = self.model.encode(texts).tolist()
            
            # Update metrics
            self.request_count += 1
            latency = time.time() - start_time
            self.avg_latency = (self.avg_latency * (self.request_count - 1) + latency) / self.request_count
            
            self.logger.info(f"Generated embeddings for {len(texts)} texts in {latency:.3f}s")
            return embeddings
        
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for embedding service"""
        return {
            "service": self.service_name,
            "status": self.health_status,
            "model": self.model_name,
            "requests": self.request_count,
            "errors": self.error_count,
            "avg_latency": self.avg_latency
        }
    
    async def shutdown(self):
        """Cleanup resources"""
        self.model = None
        self.logger.info("Embedding service shutdown complete")

class RetrievalService(RAGService):
    """Microservice for document retrieval"""
    
    def __init__(self, config: DeploymentConfig, vector_db_config: Dict[str, Any]):
        super().__init__("retrieval-service", config)
        self.vector_db_config = vector_db_config
        self.vector_db = None
        self.embedding_service_url = None
        self.cache = {}
        self.request_count = 0
        self.cache_hits = 0
    
    async def initialize(self):
        """Initialize vector database connection"""
        try:
            # Initialize vector database (implementation depends on chosen DB)
            self.logger.info("Retrieval service initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize retrieval service: {e}")
            self.health_status = "unhealthy"
            raise
    
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{query}_{top_k}"
            if self.config.enable_caching and cache_key in self.cache:
                self.cache_hits += 1
                self.logger.debug(f"Cache hit for query: {query[:50]}...")
                return self.cache[cache_key]
            
            # Get query embedding from embedding service
            query_embedding = await self._get_query_embedding(query)
            
            # Search vector database
            results = await self._search_vector_db(query_embedding, top_k)
            
            # Cache results
            if self.config.enable_caching:
                self.cache[cache_key] = results
                # Simple cache eviction (LRU-like)
                if len(self.cache) > 1000:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
            
            self.request_count += 1
            latency = time.time() - start_time
            
            self.logger.info(f"Retrieved {len(results)} documents in {latency:.3f}s")
            return results
        
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            raise
    
    async def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding from embedding service"""
        if not self.embedding_service_url:
            # For demo, return mock embedding
            return [0.1] * 384
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.embedding_service_url}/embeddings",
                json={"texts": [query]}
            ) as response:
                result = await response.json()
                return result["embeddings"][0]
    
    async def _search_vector_db(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search vector database (mock implementation)"""
        # Mock implementation - replace with actual vector DB search
        return [
            {"id": f"doc_{i}", "content": f"Mock document {i}", "score": 0.8 - i * 0.1}
            for i in range(top_k)
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for retrieval service"""
        return {
            "service": self.service_name,
            "status": self.health_status,
            "requests": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(self.request_count, 1)
        }
    
    async def shutdown(self):
        """Cleanup resources"""
        self.cache.clear()
        self.logger.info("Retrieval service shutdown complete")

class GenerationService(RAGService):
    """Microservice for response generation"""
    
    def __init__(self, config: DeploymentConfig, llm_config: Dict[str, Any]):
        super().__init__("generation-service", config)
        self.llm_config = llm_config
        self.llm_client = None
        self.request_count = 0
        self.token_usage = 0
    
    async def initialize(self):
        """Initialize LLM client"""
        try:
            # Initialize LLM client (implementation depends on chosen LLM)
            self.logger.info("Generation service initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize generation service: {e}")
            self.health_status = "unhealthy"
            raise
    
    async def generate(self, query: str, context: List[str]) -> Dict[str, Any]:
        """Generate response from query and context"""
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self._create_prompt(query, context)
            
            # Generate response (mock implementation)
            response = await self._call_llm(prompt)
            
            self.request_count += 1
            latency = time.time() - start_time
            
            result = {
                "response": response,
                "confidence": 0.85,  # Mock confidence
                "generation_time": latency,
                "token_usage": {"prompt": 100, "completion": 50}  # Mock token usage
            }
            
            self.token_usage += result["token_usage"]["prompt"] + result["token_usage"]["completion"]
            
            self.logger.info(f"Generated response in {latency:.3f}s")
            return result
        
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
    
    def _create_prompt(self, query: str, context: List[str]) -> str:
        """Create prompt from query and context"""
        context_text = "\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(context))
        
        return f"""Use the following context to answer the question:

Context:
{context_text}

Question: {query}

Answer:"""
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM (mock implementation)"""
        # Mock response - replace with actual LLM call
        await asyncio.sleep(0.1)  # Simulate API latency
        return "This is a mock response generated from the provided context."
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for generation service"""
        return {
            "service": self.service_name,
            "status": self.health_status,
            "requests": self.request_count,
            "total_tokens": self.token_usage
        }
    
    async def shutdown(self):
        """Cleanup resources"""
        self.logger.info("Generation service shutdown complete")

class RAGOrchestrator:
    """Orchestrates RAG pipeline across microservices"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.services = {}
        self.logger = logging.getLogger("rag-orchestrator")
        self.circuit_breakers = {}
        self.rate_limiter = None
        
        if config.enable_rate_limiting:
            self.rate_limiter = RateLimiter(config.rate_limit_rpm)
    
    def register_service(self, service_name: str, service: RAGService):
        """Register a service with the orchestrator"""
        self.services[service_name] = service
        self.circuit_breakers[service_name] = CircuitBreaker(service_name)
    
    async def initialize_all(self):
        """Initialize all registered services"""
        for service_name, service in self.services.items():
            try:
                await service.initialize()
                self.logger.info(f"Service {service_name} initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize service {service_name}: {e}")
                raise
    
    async def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process query through RAG pipeline"""
        start_time = time.time()
        
        # Rate limiting
        if self.rate_limiter and not await self.rate_limiter.acquire():
            raise Exception("Rate limit exceeded")
        
        try:
            # Step 1: Retrieval
            retrieval_service = self.services.get("retrieval")
            if not retrieval_service:
                raise Exception("Retrieval service not available")
            
            with self.circuit_breakers["retrieval"]:
                retrieval_results = await retrieval_service.retrieve(query, top_k)
                context = [doc["content"] for doc in retrieval_results]
            
            # Step 2: Generation
            generation_service = self.services.get("generation")
            if not generation_service:
                raise Exception("Generation service not available")
            
            with self.circuit_breakers["generation"]:
                generation_result = await generation_service.generate(query, context)
            
            # Combine results
            result = {
                "query": query,
                "response": generation_result["response"],
                "context": context,
                "confidence": generation_result["confidence"],
                "metadata": {
                    "retrieval_results": len(retrieval_results),
                    "total_time": time.time() - start_time,
                    "generation_time": generation_result["generation_time"]
                }
            }
            
            self.logger.info(f"Query processed successfully in {result['metadata']['total_time']:.3f}s")
            return result
        
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Aggregate health check across all services"""
        health_status = {}
        overall_healthy = True
        
        for service_name, service in self.services.items():
            try:
                service_health = await service.health_check()
                health_status[service_name] = service_health
                if service_health.get("status") != "healthy":
                    overall_healthy = False
            except Exception as e:
                health_status[service_name] = {"status": "unhealthy", "error": str(e)}
                overall_healthy = False
        
        return {
            "overall_status": "healthy" if overall_healthy else "unhealthy",
            "services": health_status,
            "timestamp": time.time()
        }
    
    async def shutdown_all(self):
        """Shutdown all services"""
        for service_name, service in self.services.items():
            try:
                await service.shutdown()
                self.logger.info(f"Service {service_name} shutdown complete")
            except Exception as e:
                self.logger.error(f"Error shutting down service {service_name}: {e}")

class CircuitBreaker:
    """Circuit breaker for service resilience"""
    
    def __init__(self, service_name: str, failure_threshold: int = 5, timeout: float = 60.0):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.lock = threading.Lock()
    
    async def __aenter__(self):
        with self.lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half-open"
                    self.failure_count = 0
                else:
                    raise Exception(f"Circuit breaker open for {self.service_name}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        with self.lock:
            if exc_type is not None:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logging.getLogger(__name__).warning(
                        f"Circuit breaker opened for {self.service_name} "
                        f"after {self.failure_count} failures"
                    )
            else:
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    async def acquire(self) -> bool:
        """Acquire a token from the bucket"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
            self.tokens = min(self.requests_per_minute, self.tokens + tokens_to_add)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
```

### 10.2 Infrastructure and Deployment

```python
from typing import Dict, List, Any, Optional
import yaml
import json
import os
from dataclasses import dataclass

@dataclass
class InfrastructureConfig:
    """Infrastructure configuration"""
    cloud_provider: str = "aws"  # aws, gcp, azure
    region: str = "us-west-2"
    environment: str = "production"  # development, staging, production
    
    # Compute resources
    instance_type: str = "c5.xlarge"
    min_instances: int = 2
    max_instances: int = 10
    
    # Storage
    vector_db_type: str = "pinecone"  # pinecone, weaviate, qdrant
    cache_type: str = "redis"
    storage_size_gb: int = 100
    
    # Networking
    load_balancer_type: str = "application"
    ssl_enabled: bool = True
    vpc_cidr: str = "10.0.0.0/16"
    
    # Monitoring
    enable_cloudwatch: bool = True
    enable_prometheus: bool = True
    log_retention_days: int = 30

class KubernetesDeployment:
    """Kubernetes deployment configuration generator"""
    
    def __init__(self, config: InfrastructureConfig):
        self.config = config
    
    def generate_deployment_yaml(self, service_name: str, image: str, port: int = 8000) -> str:
        """Generate Kubernetes deployment YAML"""
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{service_name}-deployment",
                "labels": {"app": service_name}
            },
            "spec": {
                "replicas": self.config.min_instances,
                "selector": {"matchLabels": {"app": service_name}},
                "template": {
                    "metadata": {"labels": {"app": service_name}},
                    "spec": {
                        "containers": [{
                            "name": service_name,
                            "image": image,
                            "ports": [{"containerPort": port}],
                            "env": [
                                {"name": "ENVIRONMENT", "value": self.config.environment},
                                {"name": "LOG_LEVEL", "value": "INFO"}
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "128Mi"},
                                "limits": {"cpu": "500m", "memory": "512Mi"}
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": port},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": port},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        return yaml.dump(deployment, default_flow_style=False)
    
    def generate_service_yaml(self, service_name: str, port: int = 8000) -> str:
        """Generate Kubernetes service YAML"""
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": f"{service_name}-service"},
            "spec": {
                "selector": {"app": service_name},
                "ports": [{"protocol": "TCP", "port": 80, "targetPort": port}],
                "type": "ClusterIP"
            }
        }
        
        return yaml.dump(service, default_flow_style=False)
    
    def generate_hpa_yaml(self, service_name: str) -> str:
        """Generate Horizontal Pod Autoscaler YAML"""
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": f"{service_name}-hpa"},
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{service_name}-deployment"
                },
                "minReplicas": self.config.min_instances,
                "maxReplicas": self.config.max_instances,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {"type": "Utilization", "averageUtilization": 70}
                        }
                    }
                ]
            }
        }
        
        return yaml.dump(hpa, default_flow_style=False)

class DockerConfiguration:
    """Docker configuration for RAG services"""
    
    @staticmethod
    def generate_dockerfile(service_type: str) -> str:
        """Generate Dockerfile for service type"""
        
        base_dockerfile = """FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        if service_type == "embedding":
            return base_dockerfile.replace(
                "FROM python:3.9-slim",
                """FROM python:3.9-slim

# Install system dependencies for sentence-transformers
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*"""
            )
        
        return base_dockerfile
    
    @staticmethod
    def generate_docker_compose() -> str:
        """Generate docker-compose.yml"""
        compose = {
            "version": "3.8",
            "services": {
                "embedding-service": {
                    "build": {"context": "./embedding-service"},
                    "ports": ["8001:8000"],
                    "environment": ["SERVICE_NAME=embedding"],
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3
                    }
                },
                "retrieval-service": {
                    "build": {"context": "./retrieval-service"},
                    "ports": ["8002:8000"],
                    "environment": ["SERVICE_NAME=retrieval"],
                    "depends_on": ["embedding-service", "vector-db"]
                },
                "generation-service": {
                    "build": {"context": "./generation-service"},
                    "ports": ["8003:8000"],
                    "environment": ["SERVICE_NAME=generation"]
                },
                "orchestrator": {
                    "build": {"context": "./orchestrator"},
                    "ports": ["8000:8000"],
                    "environment": ["SERVICE_NAME=orchestrator"],
                    "depends_on": ["embedding-service", "retrieval-service", "generation-service"]
                },
                "vector-db": {
                    "image": "qdrant/qdrant:latest",
                    "ports": ["6333:6333"],
                    "volumes": ["qdrant_data:/qdrant/storage"]
                },
                "redis": {
                    "image": "redis:alpine",
                    "ports": ["6379:6379"],
                    "volumes": ["redis_data:/data"]
                }
            },
            "volumes": {
                "qdrant_data": {},
                "redis_data": {}
            }
        }
        
        return yaml.dump(compose, default_flow_style=False)

class TerraformInfrastructure:
    """Terraform configuration for cloud infrastructure"""
    
    def __init__(self, config: InfrastructureConfig):
        self.config = config
    
    def generate_aws_terraform(self) -> str:
        """Generate Terraform configuration for AWS"""
        terraform_config = f"""
terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = "{self.config.region}"
}}

# VPC
resource "aws_vpc" "rag_vpc" {{
  cidr_block           = "{self.config.vpc_cidr}"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {{
    Name = "rag-vpc-{self.config.environment}"
  }}
}}

# Internet Gateway
resource "aws_internet_gateway" "rag_igw" {{
  vpc_id = aws_vpc.rag_vpc.id
  
  tags = {{
    Name = "rag-igw-{self.config.environment}"
  }}
}}

# Subnets
resource "aws_subnet" "rag_public_subnet" {{
  count             = 2
  vpc_id            = aws_vpc.rag_vpc.id
  cidr_block        = "10.0.${{count.index + 1}}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {{
    Name = "rag-public-subnet-${{count.index + 1}}-{self.config.environment}"
  }}
}}

data "aws_availability_zones" "available" {{
  state = "available"
}}

# Route Table
resource "aws_route_table" "rag_public_rt" {{
  vpc_id = aws_vpc.rag_vpc.id
  
  route {{
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.rag_igw.id
  }}
  
  tags = {{
    Name = "rag-public-rt-{self.config.environment}"
  }}
}}

resource "aws_route_table_association" "rag_public_rta" {{
  count          = 2
  subnet_id      = aws_subnet.rag_public_subnet[count.index].id
  route_table_id = aws_route_table.rag_public_rt.id
}}

# Security Group
resource "aws_security_group" "rag_sg" {{
  name_prefix = "rag-sg-{self.config.environment}"
  vpc_id      = aws_vpc.rag_vpc.id
  
  ingress {{
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  ingress {{
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  ingress {{
    from_port   = 8000
    to_port     = 8003
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.rag_vpc.cidr_block]
  }}
  
  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  tags = {{
    Name = "rag-sg-{self.config.environment}"
  }}
}}

# Load Balancer
resource "aws_lb" "rag_alb" {{
  name               = "rag-alb-{self.config.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.rag_sg.id]
  subnets           = aws_subnet.rag_public_subnet[*].id
  
  enable_deletion_protection = false
  
  tags = {{
    Name = "rag-alb-{self.config.environment}"
  }}
}}

# ECS Cluster
resource "aws_ecs_cluster" "rag_cluster" {{
  name = "rag-cluster-{self.config.environment}"
  
  setting {{
    name  = "containerInsights"
    value = "enabled"
  }}
}}

# ECS Task Definition
resource "aws_ecs_task_definition" "rag_task" {{
  family                   = "rag-task-{self.config.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  
  container_definitions = jsonencode([
    {{
      name  = "rag-orchestrator"
      image = "your-registry/rag-orchestrator:latest"
      
      portMappings = [
        {{
          containerPort = 8000
          hostPort      = 8000
        }}
      ]
      
      logConfiguration = {{
        logDriver = "awslogs"
        options = {{
          awslogs-group         = aws_cloudwatch_log_group.rag_logs.name
          awslogs-region        = "{self.config.region}"
          awslogs-stream-prefix = "ecs"
        }}
      }}
    }}
  ])
}}

# IAM Role for ECS
resource "aws_iam_role" "ecs_execution_role" {{
  name = "rag-ecs-execution-role-{self.config.environment}"
  
  assume_role_policy = jsonencode({{
    Version = "2012-10-17"
    Statement = [
      {{
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {{
          Service = "ecs-tasks.amazonaws.com"
        }}
      }}
    ]
  }})
}}

resource "aws_iam_role_policy_attachment" "ecs_execution_role_policy" {{
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "rag_logs" {{
  name              = "/ecs/rag-{self.config.environment}"
  retention_in_days = {self.config.log_retention_days}
}}

# ECS Service
resource "aws_ecs_service" "rag_service" {{
  name            = "rag-service-{self.config.environment}"
  cluster         = aws_ecs_cluster.rag_cluster.id
  task_definition = aws_ecs_task_definition.rag_task.arn
  desired_count   = {self.config.min_instances}
  launch_type     = "FARGATE"
  
  network_configuration {{
    subnets          = aws_subnet.rag_public_subnet[*].id
    security_groups  = [aws_security_group.rag_sg.id]
    assign_public_ip = true
  }}
  
  load_balancer {{
    target_group_arn = aws_lb_target_group.rag_tg.arn
    container_name   = "rag-orchestrator"
    container_port   = 8000
  }}
  
  depends_on = [aws_lb_listener.rag_listener]
}}

# Target Group
resource "aws_lb_target_group" "rag_tg" {{
  name     = "rag-tg-{self.config.environment}"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.rag_vpc.id
  target_type = "ip"
  
  health_check {{
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
  }}
}}

# Listener
resource "aws_lb_listener" "rag_listener" {{
  load_balancer_arn = aws_lb.rag_alb.arn
  port              = "80"
  protocol          = "HTTP"
  
  default_action {{
    type             = "forward"
    target_group_arn = aws_lb_target_group.rag_tg.arn
  }}
}}

# Auto Scaling
resource "aws_appautoscaling_target" "rag_scaling_target" {{
  max_capacity       = {self.config.max_instances}
  min_capacity       = {self.config.min_instances}
  resource_id        = "service/${{aws_ecs_cluster.rag_cluster.name}}/${{aws_ecs_service.rag_service.name}}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}}

resource "aws_appautoscaling_policy" "rag_scaling_policy" {{
  name               = "rag-scaling-policy-{self.config.environment}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.rag_scaling_target.resource_id
  scalable_dimension = aws_appautoscaling_target.rag_scaling_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.rag_scaling_target.service_namespace
  
  target_tracking_scaling_policy_configuration {{
    predefined_metric_specification {{
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }}
    target_value = 70.0
  }}
}}

# Outputs
output "load_balancer_dns" {{
  value = aws_lb.rag_alb.dns_name
}}

output "vpc_id" {{
  value = aws_vpc.rag_vpc.id
}}
"""
        return terraform_config

def demonstrate_production_deployment():
    """Demonstrate production deployment configurations"""
    print("=== Production Deployment Demo ===")
    
    # Create infrastructure config
    infra_config = InfrastructureConfig(
        cloud_provider="aws",
        environment="production",
        instance_type="c5.xlarge",
        min_instances=2,
        max_instances=10
    )
    
    print("\n1. Kubernetes Deployment Configuration:")
    k8s_deployment = KubernetesDeployment(infra_config)
    
    embedding_deployment = k8s_deployment.generate_deployment_yaml(
        "embedding-service", 
        "your-registry/embedding-service:latest"
    )
    print("Generated Kubernetes deployment YAML")
    
    print("\n2. Docker Configuration:")
    docker_config = DockerConfiguration()
    dockerfile = docker_config.generate_dockerfile("embedding")
    docker_compose = docker_config.generate_docker_compose()
    print("Generated Dockerfile and docker-compose.yml")
    
    print("\n3. Terraform Infrastructure:")
    terraform = TerraformInfrastructure(infra_config)
    aws_terraform = terraform.generate_aws_terraform()
    print("Generated Terraform configuration for AWS")
    
    print("\n4. RAG Services Setup:")
    # Create deployment config
    deployment_config = DeploymentConfig(
        pattern=DeploymentPattern.MICROSERVICES,
        scaling_strategy=ScalingStrategy.AUTO_SCALING
    )
    
    # Create orchestrator
    orchestrator = RAGOrchestrator(deployment_config)
    
    # Register services
    embedding_service = EmbeddingService(deployment_config)
    retrieval_service = RetrievalService(deployment_config, {})
    generation_service = GenerationService(deployment_config, {})
    
    orchestrator.register_service("embedding", embedding_service)
    orchestrator.register_service("retrieval", retrieval_service)
    orchestrator.register_service("generation", generation_service)
    
    print("  Services registered with orchestrator")
    print("  Circuit breakers and rate limiting configured")
    
    print("\nProduction deployment demonstration completed!")
    print("Key components generated:")
    print("  - Kubernetes manifests for container orchestration")
    print("  - Docker configurations for containerization")
    print("  - Terraform scripts for cloud infrastructure")
    print("  - Microservices architecture with resilience patterns")

if __name__ == "__main__":
    demonstrate_production_deployment()
```

## Conclusion

This comprehensive RAG guide covers all essential aspects of building production-ready Retrieval-Augmented Generation systems. From fundamental concepts to advanced deployment strategies, each chapter provides practical implementations and best practices.

### Key Takeaways:

1. **Modular Architecture**: The framework emphasizes modular, extensible designs that can adapt to different use cases and scale with requirements.

2. **Production Readiness**: Every component includes error handling, monitoring, logging, and resilience patterns necessary for production deployment.

3. **Performance Optimization**: Advanced techniques for retrieval accuracy, generation quality, and system performance are built into the framework.

4. **Comprehensive Evaluation**: Robust evaluation and monitoring systems ensure continuous improvement and reliability.

5. **Scalable Deployment**: Modern deployment patterns including microservices, containerization, and cloud-native approaches are covered.

### Next Steps:

- **Customize Components**: Adapt the provided implementations to your specific domain and requirements
- **Integrate Real Services**: Replace mock implementations with actual LLM APIs, vector databases, and embedding models  
- **Extend Functionality**: Add domain-specific processors, retrievers, and generators
- **Optimize Performance**: Profile and optimize based on your specific usage patterns
- **Monitor and Improve**: Use the evaluation framework to continuously improve system performance

This guide provides a solid foundation for building sophisticated RAG systems that can handle real-world complexity while maintaining high performance, reliability, and scalability.
```