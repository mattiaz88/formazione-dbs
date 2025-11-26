---
marp: true
theme: agentic-ai
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->

# Lesson 1
## Introduction to Large Language Models and Transformers

Course: Development of Agentic AI Systems for Advertising Campaign Analysis using the Langchain Framework

---

## Lesson Agenda

**Part One: Theoretical Foundations**
- Definition and characteristics of Large Language Models
- Transformer architecture and fundamental components

**Part Two: Model Overview**
- Comparative analysis of major available models
- Selection criteria for different application scenarios

**Part Three: Embeddings and Semantic Representations**
- Understanding embeddings and vector representations
- Similarity search and semantic applications

**Part Four: Practical Applications**
- Business use cases
- Guided exercises

Duration: 2 hours

---

## Learning Objectives

At the end of this lesson, students will be able to:

- Understand the operating principles of Large Language Models and their role in natural language processing
- Explain fundamental concepts of the Transformer architecture: tokenization, self-attention mechanism, and context window
- Comprehend embeddings and their role in semantic representation
- Compare distinctive characteristics of major models available on the market
- Identify appropriate use cases for LLM applications in business contexts
- Evaluate which model is most suitable for specific operational scenarios

---

<!-- _class: lead -->

# Part 1
## Definition and Characteristics of Large Language Models

---

## Large Language Models: Definition

**Large Language Models** are artificial intelligence models trained on vast textual corpora with the following capabilities:

**Natural language understanding**
- Semantic and contextual text analysis
- Interpretation of complex queries

**Content generation**
- Production of coherent and grammatically correct text
- Maintenance of contextual coherence

**Operational versatility**
- Execution of diversified tasks without need for reprogramming
- Adaptation to new tasks through few-shot learning

Distinctive characteristic: learning linguistic patterns directly from data, without dependence on predefined explicit rules.

---

## Evolutionary Path: From Machine Learning to LLMs

**Technological Evolution**

Traditional Machine Learning → Deep Learning → Transformers → Large Language Models

**Timeline**
- 2012-2016: Spread of Deep Learning for specific tasks
- 2017: Introduction of Transformer architecture
- 2018-2020: First large-scale pre-trained models
- 2020-present: Era of general-purpose Large Language Models

**Paradigm shift**: transition from models specialized for individual tasks to versatile models capable of adapting to multiple applications without retraining.

---

## Fundamental Capabilities of LLMs

<div class="columns">
<div>

### Language Understanding
- Contextual and semantic analysis
- Disambiguation of multiple meanings
- Entity recognition and classification
- Sentiment and intent analysis

</div>
<div>

### Content Generation
- Production of grammatically correct text
- Multilingual translations
- Document synthesis and summaries
- Articulated responses to complex questions

</div>
</div>

**Emergent capability**: Few-shot learning - ability to learn new tasks with a limited number of demonstrative examples, without need for model retraining.

---

<!-- _class: lead -->

# Part 2
## The Transformer Architecture

---

## Transformer: Fundamental Innovation

**Reference paper**: "Attention is All You Need" (Vaswani et al., 2017)

**Main innovation**
The introduction of the **self-attention** mechanism that enables parallel processing of all tokens in a sequence, overcoming the sequential limitations of previous architectures.

**Impact**
This architecture enabled the development of significantly more efficient models capable of capturing semantic relationships even between distant elements in text.

**Current relevance**
All modern Large Language Models (GPT, Claude, Llama, Mistral) implement variants of this fundamental architecture.

---

## Token: The Base Unit of Processing

**Definition**
The token represents the minimum unit of textual processing in Large Language Models.

**Token types**
A token can correspond to:
- A complete word (example: "advertising")
- A portion of a word (example: "advert" + "ising")
- A single character or symbol (example: "@", "€")
- Punctuation elements (example: ",", ".")

**Practical tokenization example**
Original text: "Campaign performance analysis"
Tokenized output: ["Campaign", " performance", " analy", "sis"]

---

## Tokenization Process: Detailed Example

**Text input**
"The campaign reach is 45% on the A25-54 target."

**Output after tokenization** (approximately 12 tokens)
["The", " campaign", " reach", " is", " ", "45", "%", " on", " the", " A", "25", "-", "54", " target", "."]

**Important consideration**
The average ratio is approximately 1 token per 0.75 words. This value can vary significantly based on language morphology and complexity of technical vocabulary used.

---

## Self-Attention: Fundamental Mechanism of the Transformer

The **self-attention** mechanism constitutes the key component of the Transformer architecture and allows the model to:

**Parallel processing**
Simultaneously analyze all tokens in a sequence, overcoming sequential processing constraints.

**Contextual weighting**
Assign variable attention coefficients to each token based on contextual relevance with respect to other elements in the sequence.

**Capture of long-range dependencies**
Identify and model semantic relationships even between distant tokens in the textual sequence.

**Illustrative example**
In the sentence "The campaign target exceeded its objectives", during processing of the term "objectives", the model assigns higher attention weights to the tokens "campaign" and "target".

---

## Visualization of Self-Attention Mechanism

![width:800px](https://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

Source: The Illustrated Transformer - Jay Alammar

The visualized connections represent the attention weights that each token assigns to other elements in the sequence. Connection intensity indicates contextual relevance between tokens.

---

## Context Window: Definition and Implications

**Definition**
The context window represents the maximum amount of text that a model can process in a single inference. This capacity is measured in number of tokens, not in words or characters.

| Model          | Context Window  | Approximate Equivalent |
|----------------|----------------|------------------------|
| GPT-3.5        | 4,096 tokens   | about 3,000 words      |
| GPT-4          | 128,000 tokens | about 96,000 words     |
| Claude 3       | 200,000 tokens | about 150,000 words    |
| Gemini 2.5 Pro | 1,048,576 tokens | about 900,000 words  |

**Operational implication**
The size of the context window determines the length of documents that can be analyzed in a single processing operation, significantly influencing possible applications.

---

## Context Window: Practical Applications

**Application scenario: Monthly report analysis**

Short report (5 pages, about 2,000 tokens)
- Compatible with all available models

Medium report (20 pages, about 8,000 tokens)
- Requires GPT-4, Claude 3 or equivalent models

Extended report (100 pages, about 40,000 tokens)
- Requires models with extended context window like Claude 3

**Practical rule for estimation**
A standard page of text corresponds approximately to 400-500 tokens, considering typical formatting and spacing of professional documents.

---

<!-- _class: lead -->

# Part 3
## Embeddings and Semantic Representations

---

## What are Embeddings?

**Definition**
Embeddings are numerical vector representations of words, sentences, or documents that capture their semantic meaning in a high-dimensional space.

**Key characteristics**
- Each piece of text is converted into a vector of numbers (typically 768-1536 dimensions)
- Similar meanings result in similar vectors
- Mathematical operations on vectors reflect semantic relationships

**Example**
```
"advertising campaign" → [0.23, -0.45, 0.67, ..., 0.12]
"marketing initiative" → [0.25, -0.43, 0.65, ..., 0.14]
```

These vectors would be close to each other in the embedding space because they have similar meanings.

---

## How Embeddings Capture Meaning

**Semantic properties in vector space**

Similarity: Words with similar meanings have vectors close to each other in the embedding space.

Relationships: Semantic relationships are preserved through vector arithmetic.

Contextual understanding: Modern embeddings capture meaning based on context, not just word identity.

**Example of semantic relationships**
```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

This demonstrates how embeddings encode conceptual relationships between words.

---

## Creating Embeddings: The Process

**Modern embedding models**

**Traditional word embeddings**
- Word2Vec (2013)
- GloVe (2014)
- FastText (2016)

**Contextual embeddings**
- BERT (2018)
- Sentence-BERT
- OpenAI text-embedding models
- Cohere embeddings

**Transformation process**
```
Text → Tokenization → Neural Network → Vector Representation
```

The neural network is trained to produce vectors that capture semantic meaning through various tasks like predicting surrounding words or sentence relationships.

---

## Embedding Dimensions and Properties

**Dimensionality**

| Model | Dimensions | Characteristics |
|-------|-----------|-----------------|
| Word2Vec | 100-300 | Simple, fast, word-level |
| BERT | 768 | Contextual, sentence-aware |
| OpenAI ada-002 | 1,536 | High quality, general purpose |
| Cohere embed-v3 | 1,024 | Optimized for search |

**Properties**
- Higher dimensions can capture more nuanced meanings
- Trade-off between expressiveness and computational cost
- Different models optimized for different tasks

---

## Similarity Measures with Embeddings

**Cosine similarity**
The most common measure for comparing embeddings.

```
similarity = (A · B) / (||A|| × ||B||)
```

Values range from -1 (opposite) to 1 (identical)

**Example application**
```
query = "campaign performance metrics"
documents = [
    "KPIs for advertising campaigns",      # High similarity
    "reach and frequency analysis",         # Medium similarity  
    "weather forecast data"                 # Low similarity
]
```

The system can rank documents by their semantic similarity to the query.

---

## Practical Applications of Embeddings

**Semantic search**
Finding relevant documents based on meaning, not just keyword matching.

**Document clustering**
Grouping similar documents together automatically.

**Recommendation systems**
Suggesting content based on semantic similarity to user preferences.

**Question answering**
Matching questions to relevant answers in a knowledge base.

**Anomaly detection**
Identifying content that is semantically different from expected patterns.

---

## Embeddings in the TTVAM Project

**Use cases for advertising campaign analysis**

**Campaign similarity analysis**
- Compare campaigns based on their descriptions and objectives
- Identify similar past campaigns for benchmarking
- Group campaigns by semantic characteristics

**Intelligent search**
- Enable users to search for campaigns using natural language
- "Find campaigns targeting young adults about technology products"
- Semantic matching beyond keyword search

**Automated tagging and classification**
- Automatically categorize campaigns based on content
- Extract themes and topics from campaign descriptions
- Group similar advertising strategies

---

## Embeddings vs. Traditional Search

**Traditional keyword search**
```
Query: "television advertising effectiveness"
Matches: Documents containing exact words "television", 
         "advertising", "effectiveness"
Misses: Documents about "TV ad performance" or 
        "broadcast campaign impact"
```

**Embedding-based semantic search**
```
Query: "television advertising effectiveness"
Matches: 
- "TV ad performance metrics"
- "broadcast campaign ROI analysis"
- "video advertising impact studies"
- "linear TV reach effectiveness"
```

Embeddings understand that these phrases convey similar meanings even with different words.

---

## Vector Databases for Embeddings

**Storage and retrieval**

Traditional databases are not optimized for similarity search on high-dimensional vectors.

**Vector database solutions**
- Pinecone
- Weaviate
- Chroma
- Qdrant
- FAISS (Facebook AI Similarity Search)

**Capabilities**
- Efficient nearest-neighbor search
- Scalability to millions of vectors
- Filtering by metadata
- Real-time updates

---

## Creating Embeddings: Practical Example

**Using OpenAI's embedding API**

```python
import openai

# Create embedding for a campaign description
text = "Spring campaign targeting adults 25-54 
        for new streaming service launch"

response = openai.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

embedding = response.data[0].embedding
# Returns a vector of 1536 dimensions

print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

**Output**
```
Embedding dimensions: 1536
First 5 values: [0.023, -0.045, 0.067, -0.012, 0.089]
```

---

## Embeddings for Multi-Language Support

**Cross-lingual embeddings**

Modern embedding models can create comparable vectors across languages.

**Example**
```
English: "advertising campaign"     → [0.23, -0.45, 0.67, ...]
Italian: "campagna pubblicitaria"   → [0.24, -0.44, 0.66, ...]
Finnish: "mainoskampanja"           → [0.23, -0.46, 0.68, ...]
```

**Benefits for TTVAM**
- Support queries in multiple languages
- Match campaigns across language boundaries
- Unified semantic search across Finnish, English, and other languages

---

## Limitations of Embeddings

**Important considerations**

**Fixed representation**
- Once created, embeddings don't change with context
- May not capture very recent terminology or events

**Dimensionality curse**
- High-dimensional spaces can be counterintuitive
- Distance metrics may behave differently than expected

**Domain specificity**
- General-purpose embeddings may miss domain-specific nuances
- May require fine-tuning for specialized applications

**Computational cost**
- Generating embeddings requires API calls or compute resources
- Storage requirements for large collections of vectors

---

## Best Practices for Using Embeddings

**Chunking strategy**
- Break long documents into meaningful chunks
- Typical size: 200-500 tokens per chunk
- Maintain context at chunk boundaries

**Metadata enrichment**
- Store relevant metadata alongside vectors
- Enable hybrid search (semantic + filters)
- Example: campaign date, target demographic, broadcaster

**Regular updates**
- Refresh embeddings when content changes
- Monitor embedding quality over time
- Update model version when improvements available

**Testing and validation**
- Evaluate retrieval quality with test queries
- Compare results with traditional search
- Measure user satisfaction with semantic search

---

<!-- _class: lead -->

# Part 4
## Overview of Major Large Language Models

---

## GPT (Generative Pre-trained Transformer)

**Developer**: OpenAI

**Distinctive technical characteristics**
- Family of proprietary models accessible via API
- GPT-4 currently represents one of the most advanced available LLMs
- Superior capabilities in complex reasoning and creative tasks
- Context window up to 128,000 tokens

**Optimal application scenarios**
- Content generation for marketing campaigns
- Complex analyses requiring multi-step reasoning
- Creative tasks and brainstorming support
- Applications requiring deep contextual understanding

---

## Claude

**Developer**: Anthropic

**Distinctive technical characteristics**
- Architecture designed with emphasis on safety and response reliability
- Particularly extended context window (200,000 tokens)
- Excellent performance in analyzing long and complex documents
- Available via API with different versions (Haiku, Sonnet, Opus)

**Optimal application scenarios**
- Detailed analysis of extensive reports and technical documentation
- Accurate summarization of voluminous documents
- Tasks requiring particular precision and reliability
- Applications where reasoning traceability is critical

---

## Llama

**Developer**: Meta AI

**Distinctive technical characteristics**
- Family of open-source models with permissive license
- Available in different sizes (7B, 13B, 70B parameters)
- Possibility of execution in local environment (on-premise)
- Support for fine-tuning on specific domains

**Optimal application scenarios**
- On-premise deployment for applications with strict privacy requirements
- Projects requiring extensive model customization
- Environments where complete control over data and infrastructure is necessary
- Applications benefiting from domain-specific optimizations

---

## Mistral

**Developer**: Mistral AI (France)

**Distinctive technical characteristics**
- Family of European models, both open-source and proprietary
- Superior computational efficiency relative to size
- Particularly developed multilingual capabilities
- Various variants available (Mistral 7B, Mixtral 8x7B, Mistral Large)

**Optimal application scenarios**
- Multilingual applications with focus on European market
- Deployment with computational resource constraints
- Scenarios requiring reduced latency
- Applications needing balance between performance and costs

---

## Model Comparison: Synopsis Table

| Characteristic | GPT-4 | Claude 3 | Llama 3 70B | Mistral Large |
|---------------|-------|----------|-------------|---------------|
| **Type** | Proprietary | Proprietary | Open-source | Hybrid |
| **Context** | 128K | 200K | 8K | 32K |
| **Parameters** | ~1.7T | ~200B | 70B | ~100B |
| **Cost** | $$$ | $$$ | Free* | $$ |
| **Deploy** | API | API | Local/Cloud | API/Local |

*Self-hosting requires infrastructure

---

## Model Selection Criteria

**Factors to consider in choosing the appropriate model**

**Context window size**
Evaluate typical length of documents to process and choose a model with adequate context window.

**Budget and economic sustainability**
Consider cost per token and expected volume of processing to estimate economic impact.

**Privacy and security requirements**
For sensitive data, evaluate open-source solutions with on-premise deployment.

**Multilingual capabilities**
Verify support for necessary languages and quality of performance in each language.

**Latency requirements**
Consider response times required by the application and necessary throughput characteristics.

There is no universally superior model, but it is necessary to select the most suitable solution for the specific use case.

---

<!-- _class: lead -->

# Part 5
## Practical Applications in Business Context

---

## Application 1: Conversational Assistants

**Use case**: Chatbot for campaign analysis

**Functionality**:
```
User: "What was the reach of the March campaign?"