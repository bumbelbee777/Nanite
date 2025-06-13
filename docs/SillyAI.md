# SillyAI: Reinterpreting Transformers with Complex Values, Tokens as Functions, and Executable Symbolic Graphs

## Abstract

SillyAI is a highly unconventional and experimental complex-valued neuro-symbolic transformer-based model. The aim of its architecture is to push the limits of deep learning and AI principles as well as catalyze performance and efficiency to enable an AGI-esque model to run on even the most constrained hardware such as mobile and edge devices (at least in inference). In this paper, we'll explore how SillyAI tries to achieve such ambitious objectives with as little compromises as possible.

Unfortunately, due to the current lack of compute for rigorous real-world testing/benchmarking and validation, this paper will mainly delve into and focalize on the more theoretical aspects of SillyAI. More or less simple demonstrations and examples will be included as a proof-of-concept to highlight SillyAI's transformative potential in countless real-world fields, ranging from agricultural to quantum simulations and signal processing.

## Motivations

SillyAI's main motivation is to be a thought experiment to see how much transformer architectures can be bent before becoming something entirely different or even ascended. It also sees where systems programming and more generally, low-level development can play a role in driving performance and efficiencies in AI models and enable for more expressiveness. By harmonizing neural architecture, symbolic reasoning, and systems design, SillyAI hopes to demonstrate that the future of AI may lie in breaking known assumptions and rethinking deep learning architecture altogether.

## Overview: Architecture & Pipeline

SillyAI's architecture represents a radical departure from conventional transformer designs, implementing a fully complex-valued neural network that leverages geometric algebra and Clifford algebra principles. The model's pipeline begins with a sophisticated input processing stage that transforms raw data into complex-valued representations through a series of carefully designed operations.

The input projection layer employs a novel approach to complex-valued encoding, utilizing geometric progressions of frequencies to construct rich complex phasors. This encoding scheme extends beyond traditional sinusoidal positional encoding by incorporating both magnitude and phase information in a geometrically meaningful way. The frequencies are generated using a logarithmic scale, ensuring that the model can capture both fine-grained local patterns and broader global structures in the input data.

The transformer architecture itself is built around the concept of complex-valued attention and computation. Each transformer layer consists of three primary components: the InfiniToeplitz attention mechanism, a complex-valued MLP, and specialized normalization layers. The attention mechanism uses structured matrices (Toeplitz, Block-Toeplitz, Circulant, and Hankel) to reduce parameter count while maintaining expressivity. These structured matrices are dynamically selected based on the input characteristics, allowing the model to adapt its computational structure to the specific patterns present in the data.

The complex-valued MLP extends traditional feed-forward networks by operating in the complex domain. It employs phase-preserving activation functions and complex-valued linear transformations, enabling the network to learn both magnitude and phase relationships in the data. The MLP's architecture includes residual connections with complex gates, allowing for better gradient flow and more stable training.

Layer normalization is implemented in a way that preserves the complex nature of the data. Rather than simply normalizing real and imaginary components separately, the normalization process takes into account the geometric relationships between these components, ensuring that the complex structure of the data is maintained throughout the network.

Throughout all this, SillyAI uses a custom complex tensor operations backend that leverages Clifford algebra and the properties of complex numbers to optimize computations. This backend is fully asynchronous and leverages a multitude of techniques such as aggressive caching/prefetching, SVD and low-rank decomposition, tensor hashing and compression, quantization ranging from ternary to FP16 and mixed precision training, and JIT compilation of hot paths.

## Detailed Architecture Components

### Complex Input Projection and Encoding

The input projection system in SillyAI is designed to handle multi-dimensional data while preserving geometric relationships. The projection layer first analyzes the input tensor's structure, automatically detecting spatial dimensions and batch characteristics. For a typical 3D input tensor of shape `[batch_size, seq_len, spatial_dims]`, the model performs the following operations:

1. Spatial Dimension Processing:
   - Flattens spatial dimensions into a single feature dimension
   - Applies complex-valued linear transformation to project into the model's dimension space
   - Preserves batch and sequence dimensions for downstream processing

2. Complex Positional Encoding:
   - Generates frequency bands using geometric progression: `ω_k = base_freq * (geometric_ratio)^k`
   - Constructs complex phasors: `exp(i * (ω_k * pos + φ_k))`
   - Applies phase shifts `φ_k` that are learned parameters
   - Combines multiple frequency bands to create rich positional information

3. Dimension Integration:
   - Merges positional encoding with projected features
   - Applies complex-valued layer normalization
   - Implements dropout in a phase-preserving manner

### InfiniToeplitz Attention Mechanism

The InfiniToeplitz attention mechanism represents a significant innovation in attention computation based on Google's InfiniAttention. Instead of using standard dense attention matrices, it employs structured matrices that can be parameterized much more efficiently while maintaining expressivity. The mechanism works as follows:

1. Matrix Type Selection:
   - Analyzes input tensor characteristics (rank, sparsity, spectral properties)
   - Dynamically selects appropriate matrix structure:
     * Toeplitz: For shift-invariant operations
     * Block-Toeplitz: For hierarchical patterns
     * Circulant: For periodic structures
     * Hankel: For anti-diagonal patterns

2. Parameter Generation:
   - For Toeplitz matrices: Generates parameters for each diagonal
   - For Block-Toeplitz: Creates hierarchical parameter structure
   - For Circulant: Generates parameters for circular shifts
   - For Hankel: Creates parameters for anti-diagonal patterns

3. Attention Computation:
   - Applies complex-valued matrix multiplication
   - Computes attention scores using geometric product
   - Implements scaled dot-product attention in complex domain
   - Applies complex-valued softmax for attention weights

### Complex MLP Architecture

The complex-valued MLP extends traditional feed-forward networks by exclusively operating in the complex domain. Its architecture is designed to preserve phase information while enabling rich feature transformations:

1. Complex Linear Layers:
   - Implements complex-valued matrix multiplication
   - Uses geometric product for feature transformation
   - Maintains phase relationships during transformation
   - Applies complex-valued bias terms

2. Activation Functions:
   - Implements phase-preserving activations
   - Uses complex-valued PReLU for non-linearity
   - Applies magnitude-based activation with phase shift
   - Maintains complex differentiability

3. Residual Connections:
   - Implements complex-valued skip connections
   - Uses geometric gates for information flow control
   - Maintains phase coherence across layers
   - Enables better gradient flow

### Numerical Stability and Optimization

The architecture also includes a variety mechanisms to ensure numerical stability and efficient computation:

1. Tensor Sanitization:
   - Detects and handles NaN/Inf values
   - Implements phase-preserving normalization
   - Applies adaptive scaling based on tensor magnitude
   - Maintains complex structure during sanitization

2. Performance Optimizations:
   - Implements asynchronous operation execution
   - Uses tensor caching with compression
   - Applies mixed precision based on tensor characteristics
   - JIT compiles frequently used operation paths

3. Memory Management:
   - Implements efficient tensor compression
   - Uses low-rank decomposition for large matrices
   - Applies structured sparsity patterns
   - Optimizes memory access patterns through caching and prefetching heuristics

## Multivector Operations Backend

The MultivectorOps backend forms the computational foundation of SillyAI, implementing a comprehensive suite of complex-valued operations optimized for both performance and numerical stability. Key features include:

### Complex-Valued Operations
- Geometric product operations for multivector multiplication
- Complex-valued linear transformations with automatic dimension handling
- Specialized activation functions that preserve phase information
- Complex-valued attention mechanisms with scaled dot-product operations
- FFT-based operations for efficient spectral processing

### Performance Optimizations
- Asynchronous operation execution with automatic batching
- Tensor caching system with compression and decompression
- Mixed precision routing based on tensor size and complexity
- JIT compilation of frequently used operation paths
- Low-rank decomposition for large matrix operations

### Numerical Stability
- Automatic tensor sanitization to prevent NaN/Inf propagation
- Phase-preserving normalization techniques
- Adaptive precision scaling based on tensor magnitude
- Complex-valued dropout with phase preservation
- Structured matrix operations for improved conditioning

## Transformer Architecture

SillyAI's transformer architecture extends the standard transformer with several key innovations:

### Complex Input Projection
The input projection layer transforms raw input tensors into complex-valued representations while preserving spatial relationships. It handles:
- Automatic dimension detection and reshaping
- Complex-valued linear transformations
- Positional encoding integration
- Dropout and normalization

### InfiniToeplitz Attention
A novel attention mechanism that uses structured matrices (Toeplitz, Block-Toeplitz, Circulant, Hankel) to reduce parameter count while maintaining expressivity:
- Dynamic matrix type selection based on input characteristics
- Parameter-efficient structured transformations
- Complex-valued attention scores
- Automatic matrix structure optimization

### Complex MLP
The feed-forward network extends standard MLPs to complex-valued operations:
- Complex-valued linear layers
- Phase-preserving activation functions
- Residual connections with complex gates
- Adaptive dropout mechanisms

## Training Pipeline

SillyAI implements a sophisticated training pipeline with several advanced features:

### Curriculum Learning
- Progressive difficulty scaling
- Dynamic batch size adjustment
- Adaptive learning rate scheduling
- Early stopping with patience

### Loss Functions
- Complex-valued loss computation
- Phase-aware optimization
- Concept graph integration
- Multi-objective training

### Monitoring and Profiling
- Resource usage tracking
- Performance profiling
- Memory optimization
- Training dynamics visualization

## Applications and Use Cases

While SillyAI's architecture is experimental, it shows promise in several domains:

### Quantum Simulations
- Wavefunction evolution
- Potential energy calculations
- Time-dependent Schrödinger equations
- Multi-dimensional quantum systems

### Signal Processing
- Complex-valued signal analysis
- Spectral decomposition
- Time-frequency analysis
- Adaptive filtering

### Scientific Computing
- Complex differential equations
- Multi-dimensional optimization
- Structured matrix operations
- Numerical stability in complex domains

## Conclusion

SillyAI represents an ambitious attempt to push the boundaries of transformer architectures through complex-valued operations and geometric algebra. While still experimental, its unique approach to neural-symbolic integration and performance optimization shows promise for enabling sophisticated AI models on resource-constrained devices. Future work will focus on validating these approaches in real-world applications and further optimizing the architecture for practical deployment.
