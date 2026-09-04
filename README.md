# MLOps & ML Engineering — Learning Journal 📓

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-numerical-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SymPy](https://img.shields.io/badge/SymPy-symbolic-3B5526?logo=sympy&logoColor=white)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-planned-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-planned-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-planned-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-planned-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Roadmap](https://img.shields.io/badge/roadmap.sh-ML%20Engineering%20%26%20MLOps-2F80ED?logo=roadmapdotsh&logoColor=white)](https://roadmap.sh/ai/roadmap/machine-learning-engineering-and-mlops)

This repository is my **personal learning journal** as I follow the [Machine Learning Engineering and MLOps](https://roadmap.sh/ai/roadmap/machine-learning-engineering-and-mlops) roadmap on [roadmap.sh](https://roadmap.sh/).

The goal isn't to build a library or a finished product, but to **keep a written, version-controlled record** of what I learn: theory, notes, exercises, scripts, and small "from scratch" implementations before moving on to the standard tools/frameworks of the field.

## 🗂️ Repository organization

The roadmap has three levels — **Section** → **Sub-section** → **Topic** (e.g. *Mathematical Foundations* → *Multivariable Calculus* → *Jacobian Matrices*). The folder structure only mirrors the first two levels; individual topics live **as sections inside the sub-section's `README.md`**, not as their own subfolders:

- Each **major roadmap section** becomes a **numbered folder** at the root (e.g. `01-mathematical-foundations`).
- Each **roadmap sub-section** becomes a **numbered subfolder** inside it (e.g. `01-multivariable-calculus`).
- Each **individual topic** of that sub-section (e.g. Partial Derivatives, Gradient Vectors, Jacobian Matrices, Hessian Matrices...) becomes a **new section in the same `README.md`**, plus its own script(s) when it warrants code.

```
01-mathematical-foundations/
└── 01-multivariable-calculus/            # one folder per roadmap sub-section
    ├── README.md                          # all topics of this sub-section, one section each:
    │                                       #   1. Partial Derivatives
    │                                       #   2. Gradient Vectors
    │                                       #   3. Jacobian Matrices        (added later)
    │                                       #   4. Hessian Matrices         (added later)
    │                                       #   ...
    ├── partial_derivatives.py             # "from scratch" implementation
    ├── partial_derivatives_and_gradients.py
    ├── interactive_partial_derivatives.py # comparison with a standard library (SymPy/NumPy)
    └── jacobian_matrices.py               # (added later, alongside its README section)
02-<next-section>/
└── 01-<first-sub-section>/
    └── README.md
...
```

This keeps one folder per sub-section instead of one per leaf topic — less nesting, and topics that are naturally studied together (like Partial Derivatives and Gradient Vectors) stay next to each other.

## 📄 What goes in each subfolder?

**1. A summary sheet (`README.md`)**
A short, synthetic sheet summarizing the mathematical or algorithmic intuition behind each topic of the sub-section: definitions, notation, a worked example, pseudo-code. The goal is to be able to re-read it later and understand the concept without re-deriving everything.

**2. Implementation scripts (`.py` or `.ipynb`)**

- **Code the concept from scratch** (e.g. rewrite gradient descent or derivative computation with NumPy), to really understand the mechanism under the hood.
- **Use the industry-standard library** (e.g. SciPy, PyTorch, or Scikit-Learn) to compare results and validate the "from scratch" implementation.

## ✅ Progress tracker

The full checklist below mirrors the [roadmap.sh](https://roadmap.sh/ai/roadmap/machine-learning-engineering-and-mlops) tree — Section → Sub-section → Topic. Checked topics have a corresponding folder + README section; everything else is still ahead. *(Advanced Data Structures' sub-sections weren't fully visible when this list was captured — to be completed once that section is reached.)*

<details open>
<summary><b>01. Mathematical Foundations</b></summary>

- [ ] 01.1 [Multivariable Calculus](01-mathematical-foundations/01-multivariable-calculus/README.md)
  - [x] Partial Derivatives
  - [x] Gradient Vectors
  - [ ] Jacobian Matrices
  - [ ] Hessian Matrices
  - [ ] Lagrangian Multipliers
  - [ ] Taylor Series Expansion
  - [ ] Vector Calculus
- [ ] 01.2 Linear Algebra Fundamentals
  - [ ] Vector Spaces
  - [ ] Matrix Decompositions
  - [ ] Eigenvalues and Eigenvectors
  - [ ] Singular Value Decomposition
  - [ ] Principal Component Analysis
  - [ ] Tensor Operations
  - [ ] Quadratic Forms
- [ ] 01.3 Mathematical Optimization
  - [ ] Convex Optimization
  - [ ] Gradient Descent Variants
  - [ ] Stochastic Optimization
  - [ ] Constrained Optimization
  - [ ] Karush Kuhn Tucker Conditions
  - [ ] Linear Programming
  - [ ] Quadratic Programming
- [ ] 01.4 Probability Theory
  - [ ] Random Variables
  - [ ] Probability Distributions
  - [ ] Expectation and Variance
  - [ ] Bayes Theorem
  - [ ] Markov Chains
  - [ ] Monte Carlo Methods

</details>

<details>
<summary><b>02. Advanced Data Structures</b></summary>

- [ ] Binary Search Trees
- [ ] Merkle Trees
- [ ] *(remaining sub-sections/topics to fill in once reached)*

</details>

<details>
<summary><b>03. Advanced Algorithms</b></summary>

- [ ] 03.1 Graph Algorithms
  - [ ] Breadth First Search
  - [ ] Depth First Search
  - [ ] Dijkstra Algorithm
  - [ ] Bellman Ford Algorithm
  - [ ] Floyd Warshall Algorithm
  - [ ] Minimum Spanning Trees
  - [ ] Strongly Connected Components
- [ ] 03.2 Algorithmic Paradigms
  - [ ] Divide and Conquer
  - [ ] Dynamic Programming
  - [ ] Greedy Algorithms
  - [ ] Backtracking
  - [ ] Branch and Bound
  - [ ] Randomized Algorithms
  - [ ] Amortized Analysis

</details>

<details>
<summary><b>04. Core Computer Science</b></summary>

- [ ] 04.1 Numerical Algorithms
  - [ ] Fast Fourier Transform
  - [ ] Matrix Multiplication Algorithms
  - [ ] Root Finding Methods
  - [ ] Numerical Integration
  - [ ] Interpolation Techniques
  - [ ] Random Number Generation
  - [ ] Floating Point Arithmetic
- [ ] 04.2 String Algorithms
  - [ ] Knuth Morris Pratt
  - [ ] Boyer Moore
  - [ ] Rabin Karp
  - [ ] Levenshtein Distance
  - [ ] Longest Common Subsequence
  - [ ] Regular Expression Matching
  - [ ] Sequence Alignment
- [ ] 04.3 Computer Networks
  - [ ] TCP and UDP Protocols
  - [ ] HTTP and HTTPS
  - [ ] DNS Resolution
  - [ ] Load Balancing
  - [ ] Network Security
  - [ ] Socket Programming
  - [ ] RESTful Architecture
- [ ] 04.4 Operating Systems
  - [ ] Process Management
  - [ ] Thread Synchronization
  - [ ] Virtual Memory
  - [ ] File Systems
  - [ ] Interprocess Communication
  - [ ] Deadlock Handling
  - [ ] CPU Scheduling

</details>

<details>
<summary><b>05. Software Architecture</b></summary>

- [ ] 05.1 Memory Management
  - [ ] Garbage Collection
  - [ ] Stack and Heap Allocation
  - [ ] Memory Leaks
  - [ ] Caching Strategies
  - [ ] Pointer Arithmetic
  - [ ] Buffer Overflows
  - [ ] Memory Profiling
- [ ] 05.2 Concurrency Models
  - [ ] Multithreading
  - [ ] Asynchronous Programming
  - [ ] Actor Model
  - [ ] Event Loops
  - [ ] Lock Free Data Structures
  - [ ] Software Transactional Memory
  - [ ] Producer Consumer Pattern
- [ ] 05.3 Architectural Styles
  - [ ] Monolithic Architecture
  - [ ] Microservices Architecture
  - [ ] Event Driven Architecture
  - [ ] Layered Architecture
  - [ ] Serverless Architecture
  - [ ] Clean Architecture
  - [ ] Domain Driven Design
- [ ] 05.4 Design Patterns
  - [ ] Creational Patterns
  - [ ] Structural Patterns
  - [ ] Behavioral Patterns
  - [ ] Singleton Pattern
  - [ ] Factory Pattern
  - [ ] Strategy Pattern
  - [ ] Observer Pattern

</details>

<details>
<summary><b>06. Data Engineering</b></summary>

- [ ] 06.1 API Design
  - [ ] RESTful Principles
  - [ ] GraphQL Queries
  - [ ] gRPC Protocols
  - [ ] OpenAPI Specifications
  - [ ] API Versioning
  - [ ] Rate Limiting
  - [ ] Authentication Schemes
- [ ] 06.2 Code Quality
  - [ ] SOLID Principles
  - [ ] Code Refactoring
  - [ ] Unit Testing
  - [ ] Integration Testing
  - [ ] Test Driven Development
  - [ ] Continuous Integration
  - [ ] Static Code Analysis
- [ ] 06.3 Distributed Storage
  - [ ] Distributed File Systems
  - [ ] NoSQL Databases
  - [ ] Columnar Databases
  - [ ] Time Series Databases
  - [ ] Object Storage
  - [ ] Data Lakes
  - [ ] Lakehouse Architecture
- [ ] 06.4 Data Pipelines
  - [ ] Extract Transform Load
  - [ ] Batch Processing
  - [ ] Stream Processing
  - [ ] Data Ingestion
  - [ ] Workflow Orchestration
  - [ ] Data Validation
  - [ ] Error Handling

</details>

<details>
<summary><b>07. Classical Machine Learning</b></summary>

- [ ] 07.1 Data Governance
  - [ ] Data Quality
  - [ ] Lineage Tracking
  - [ ] Access Control
  - [ ] Data Cataloging
  - [ ] Metadata Management
  - [ ] Schema Evolution
  - [ ] Compliance Standards
- [ ] 07.2 Big Data Processing
  - [ ] MapReduce Paradigm
  - [ ] Apache Spark Core
  - [ ] Spark SQL
  - [ ] Spark Streaming
  - [ ] Distributed Joins
  - [ ] Partitioning Strategies
  - [ ] Shuffle Optimization
- [ ] 07.3 Unsupervised Learning
  - [ ] K Means Clustering
  - [ ] Hierarchical Clustering
  - [ ] DBSCAN Algorithm
  - [ ] Gaussian Mixture Models
  - [ ] Dimensionality Reduction
  - [ ] Isolation Forests
  - [ ] Association Rules
- [ ] 07.4 Supervised Learning
  - [ ] Linear Regression
  - [ ] Logistic Regression
  - [ ] Decision Trees
  - [ ] Support Vector Machines
  - [ ] Gradient Boosting
  - [ ] Random Forests
  - [ ] Naive Bayes
- [ ] 07.5 Model Evaluation
  - [ ] Cross Validation
  - [ ] Precision and Recall
  - [ ] ROC and AUC Curves
  - [ ] Confusion Matrix
  - [ ] Hyperparameter Tuning
  - [ ] Overfitting Mitigation
  - [ ] Bias Variance Tradeoff
- [ ] 07.6 Feature Engineering
  - [ ] Missing Value Imputation
  - [ ] Categorical Encoding
  - [ ] Feature Scaling
  - [ ] Dimensionality Reduction
  - [ ] Polynomial Features
  - [ ] Text Vectorization
  - [ ] Image Feature Extraction

</details>

<details>
<summary><b>08. Deep Learning</b></summary>

- [ ] 08.1 Convolutional Networks
  - [ ] Convolutional Layers
  - [ ] Pooling Operations
  - [ ] Image Classification
  - [ ] Object Detection
  - [ ] Semantic Segmentation
  - [ ] Transfer Learning
  - [ ] Generative Adversarial Networks
- [ ] 08.2 Neural Network Fundamentals
  - [ ] Perceptrons
  - [ ] Multi Layer Perceptrons
  - [ ] Activation Functions
  - [ ] Backpropagation
  - [ ] Gradient Descent Optimizers
  - [ ] Regularization Techniques
  - [ ] Initialization Strategies
- [ ] 08.3 Transformer Architectures
  - [ ] Self Attention Mechanism
  - [ ] Multi Head Attention
  - [ ] Positional Encoding
  - [ ] Encoder Only Models
  - [ ] Decoder Only Models
  - [ ] Encoder Decoder Models
  - [ ] Fine Tuning Techniques
- [ ] 08.4 Sequence Models
  - [ ] Recurrent Neural Networks
  - [ ] Long Short Term Memory
  - [ ] Gated Recurrent Units
  - [ ] Attention Mechanisms
  - [ ] Sequence to Sequence Models
  - [ ] Encoder Decoder Architecture
  - [ ] Time Series Forecasting

</details>

<details>
<summary><b>09. MLOps and Infrastructure</b></summary>

- [ ] 09.1 Pipeline Orchestration
  - [ ] Workflow Engines
  - [ ] DAG Construction
  - [ ] Dependency Management
  - [ ] Automated Retraining
  - [ ] Pipeline Monitoring
  - [ ] Failure Recovery
  - [ ] Artifact Management
- [ ] 09.2 Model Deployment
  - [ ] Containerization
  - [ ] Model Serving
  - [ ] REST Endpoints
  - [ ] Batch Inference
  - [ ] Edge Deployment
  - [ ] Model Compression
  - [ ] Hardware Acceleration
- [ ] 09.3 Model Monitoring
  - [ ] Data Drift Detection
  - [ ] Performance Monitoring
  - [ ] Latency Tracking
  - [ ] Resource Utilization
  - [ ] Alerting Systems
  - [ ] Logging Frameworks
- [ ] 09.4 Experiment Tracking
  - [ ] Metrics Logging
  - [ ] Parameter Tracking
  - [ ] Artifact Storage
  - [ ] Model Registry
  - [ ] Experiment Comparison
  - [ ] Reproducibility Control
  - [ ] Metadata Tracking

</details>

<details>
<summary><b>10. Production Software</b></summary>

- [ ] 10.1 Cloud Infrastructure
  - [ ] Infrastructure as Code
  - [ ] Container Orchestration
  - [ ] Virtual Private Clouds
  - [ ] Identity Access Management
  - [ ] Secret Management
  - [ ] Cloud Storage Services
  - [ ] Serverless Compute
- [ ] 10.2 System Reliability
  - [ ] High Availability
  - [ ] Fault Tolerance
  - [ ] Disaster Recovery
  - [ ] Circuit Breakers
  - [ ] Rate Limiting
  - [ ] Graceful Degradation
  - [ ] Chaos Engineering
- [ ] 10.3 Security Operations
  - [ ] Vulnerability Scanning
  - [ ] Dependency Auditing
  - [ ] Network Policies
  - [ ] Encryption at Rest
  - [ ] Encryption in Transit
  - [ ] Access Auditing
  - [ ] Compliance Checks
- [ ] 10.4 Observability
  - [ ] Distributed Tracing
  - [ ] Metrics Collection
  - [ ] Log Aggregation
  - [ ] Health Checks
  - [ ] Performance Profiling
  - [ ] Root Cause Analysis
  - [ ] Dashboard Creation

</details>

<details>
<summary><b>11. Data Science Management</b></summary>

- [ ] 11.1 Product Management
  - [ ] Requirements Gathering
  - [ ] Scope Definition
  - [ ] Cost Benefit Analysis
  - [ ] Roadmap Planning
  - [ ] Agile Methodologies
  - [ ] Minimum Viable Product
  - [ ] Feature Prioritization
- [ ] 11.2 Business Intelligence
  - [ ] Data Visualization
  - [ ] Dashboard Design
  - [ ] KPI Tracking
  - [ ] Exploratory Data Analysis
  - [ ] Reporting Automation
  - [ ] Stakeholder Communication
  - [ ] Storytelling with Data
- [ ] 11.3 Strategic Leadership
  - [ ] Technical Strategy
  - [ ] Cross Functional Collaboration
  - [ ] Resource Allocation
  - [ ] Risk Management
  - [ ] Vendor Evaluation
  - [ ] Innovation Management
- [ ] 11.4 Ethical AI
  - [ ] Bias Mitigation
  - [ ] Fairness Metrics
  - [ ] Explainable AI
  - [ ] Privacy Preservation
  - [ ] Regulatory Compliance
  - [ ] Environmental Impact
  - [ ] Accountability Frameworks

</details>

*(Check off items as you progress through the roadmap; each sub-section links to its folder's README once one exists, where individual topics get their own section.)*

## 🛠️ Code conventions

- **Type hints** on every function.
- **Docstrings** with examples.
- **Doctests** to validate examples directly from the docstring.
- When relevant: comparing an **analytical/exact** approach vs. a **numerical** approach (finite differences) to validate the implementation.

## 🔗 Resources

- Roadmap followed: [Machine Learning Engineering and MLOps](https://roadmap.sh/ai/roadmap/machine-learning-engineering-and-mlops)
