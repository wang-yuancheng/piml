### Network Architecture & Feature Representation for Poisson-Gauss Dataset

**Random Fourier Features**
Let $\mathbf{v} = [x, y, \mathbf{p}^T]^T \in \mathbb{R}^{2 + 3N_{\text{max}}}$ be the combined input vector. The system uses a fixed random projection matrix $\mathbf{B} \in \mathbb{R}^{(2 + 3N_{\text{max}}) \times M}$ scaled by $\sigma = 0.5$. The $2M$-dimensional RFF vector $\mathbf{h}_{\text{RFF}}(\mathbf{x}, \mathbf{p})$ is given by:
$$\mathbf{h}_{\text{RFF}}(\mathbf{x}, \mathbf{p}) = \begin{bmatrix} \sin(2\pi \mathbf{v}^T \mathbf{B}) \\ \cos(2\pi \mathbf{v}^T \mathbf{B}) \end{bmatrix} \in \mathbb{R}^{2M}$$

**Multi-Layer Perceptron**
The pre-activation of the first dense layer accumulates across the RFF space dynamically. For a network with hidden layer sizes $H_l = 128$, the structural features propagate as:
$$\mathbf{z}_0 = \mathbf{h}_{\text{RFF}}(\mathbf{x}, \mathbf{p})^T \mathbf{W}_0 + \mathbf{b}_0 \in \mathbb{R}^{128}$$
$$\mathbf{h}_0 = \tanh(\mathbf{z}_0)$$
$$\mathbf{h}_l = \tanh(\mathbf{h}_{l-1}^T \mathbf{W}_l + \mathbf{b}_l) \quad \text{for } l = 1, 2, 3$$

The complete feature vector concatenates the RFF terms and all intermediate latent spaces, avoiding full matrix materialization:
$$\boldsymbol{\Phi}(\mathbf{x}, \mathbf{p}) = \begin{bmatrix} \mathbf{h}_{\text{RFF}}(\mathbf{x}, \mathbf{p}) \\ \mathbf{h}_0(\mathbf{x}, \mathbf{p}) \\ \vdots \\ \mathbf{h}_3(\mathbf{x}, \mathbf{p}) \end{bmatrix} \in \mathbb{R}^{K}$$
where $K = 2M + \sum_{l=0}^3 H_l = 512 + 512 = 1024$. The function approximation is then a simple inner product:
$$u(\mathbf{x}, \mathbf{p}) = \boldsymbol{\Phi}(\mathbf{x}, \mathbf{p})^T \mathbf{w}$$

---

### Chunking across $M$
The final output $u(\mathbf{x}, \mathbf{p})$ evaluated over $N$ spatial points is simply an $N \times 1$ column vector (e.g., $16,384 \times 1$), which is trivially small to store. The true memory bottleneck is the Laplacian computational graph. Without chunking, `jacfwd` tracks the derivatives of all features simultaneously. When scaled up, it creates a dense derivative trace that consumes >6.5 GiB of VRAM which is too much for a 8GB VRAM GPU. To solve this, the RFF feature map is split into $C$ distinct blocks of size $M_c = 256$.

**Value Aggregation**
Instead of instantiating all of $\mathbf{h}_{\text{RFF}}$, the inner loop computes the scalar field by accumulated streaming, collapsing the features down to $N \times 1$ immediately:
$$u(\mathbf{x}, \mathbf{p}) = \sum_{c=1}^{C} \left( \sin(2\pi \mathbf{v}^T \mathbf{B}_c)^T \mathbf{w}_{\sin, c} + \cos(2\pi \mathbf{v}^T \mathbf{B}_c)^T \mathbf{w}_{\cos, c} \right) + \sum_{l=0}^3 \mathbf{h}_l^T \mathbf{w}_{\text{MLP}, l}$$

**Laplacian Slicing & Rematerialization Checkpointing**
The spatial derivatives are isolated strictly within these small structural windows:
$$\nabla^2 u(\mathbf{x}, \mathbf{p}) = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = \sum_{c=1}^C \nabla^2 \left[ \boldsymbol{\Phi}_c(\mathbf{x}, \mathbf{p})^T \mathbf{w}_c \right] + \nabla^2 \left[ \boldsymbol{\Phi}_{\text{MLP}}(\mathbf{x}, \mathbf{p})^T \mathbf{w}_{\text{MLP}} \right]$$
By wrapping this chunked `jax.lax.scan` with a `@jax.checkpoint`, JAX creates the derivative graph for only 256 features at a time. Once a chunk's Laplacian contribution is added to the running total, JAX explicitly deletes that chunk's computational graph from memory before moving to the next.

---

### The Double-Chunked Kaczmarz Inner Loop
For a single source configuration, the model sets up an overdetermined linear system corresponding to the PDE residual and Dirichlet boundaries:
$$\mathbf{A} \mathbf{w} = \mathbf{b}$$
Where the operator row blocks evaluated at a batch of points are:
$$\mathbf{A} = \begin{bmatrix} \nabla^2 \boldsymbol{\Phi}(\mathbf{x}_{\text{PDE}}, \mathbf{p})^T \\ \sqrt{1000} \cdot \boldsymbol{\Phi}(\mathbf{x}_{\text{BC}}, \mathbf{p})^T \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} -s(\mathbf{x}_{\text{PDE}}) \\ \sqrt{1000} \cdot u_{\text{BC}} \end{bmatrix}$$
Because $\mathbf{A}$ cannot be safely built globally, it is processed via an outer spatial conveyor belt loop and an inner feature loop.

**Phase 1**
The algorithm iterates through feature chunks $c$ using a checkpointed `lax.scan` to construct the regularized Gram matrix $\mathbf{G} \in \mathbb{R}^{N_{\text{rows}} \times N_{\text{rows}}}$:
$$\mathbf{G} = \sum_{c=1}^{C_{\text{RFF}}} \mathbf{A}_c \mathbf{A}_c^T + \mathbf{A}_{\text{MLP}} \mathbf{A}_{\text{MLP}}^T$$
$$\mathbf{A}\mathbf{w} = \sum_{c=1}^{C_{\text{RFF}}} \mathbf{A}_c \mathbf{w}_c + \mathbf{A}_{\text{MLP}} \mathbf{w}_{\text{MLP}}$$

**Phase 2**
The dual adjustment vector $\mathbf{z}$ is evaluated via a linear solve stabilized by a learned Tikhonov parameter $\lambda_{\text{tik}}$:
$$\mathbf{z} = \left( \mathbf{G} + \lambda_{\text{tik}} \mathbf{I} \right)^{-1} (\mathbf{A}\mathbf{w} - \mathbf{b})$$

**Phase 3**
Weights are updated by projecting the error back down across the chunked feature spaces independently, governed by relaxation step size $\alpha = 0.1$:
$$\mathbf{w}_c^{(k+1)} = \mathbf{w}_c^{(k)} - \alpha \mathbf{A}_c^T \mathbf{z} \quad \forall c \in \{1, \dots, C_{\text{RFF}}\}$$
$$\mathbf{w}_{\text{MLP}}^{(k+1)} = \mathbf{w}_{\text{MLP}}^{(k)} - \alpha \mathbf{A}_{\text{MLP}}^T \mathbf{z}$$

---

### 4. Outer Network Outer Loop Optimization
The weight matrix $\mathbf{w}$ calculated via the Kaczmarz projection is treated as a constant during core backpropagation by blocking the gradient path: $\hat{\mathbf{w}} = \text{stop\_gradient}(\mathbf{w})$. This is because the Kaczmarz loop is an algebraic forward-projection method that relies on forward-mode derivatives, completely bypassing the need to backpropagate through the solver steps.

The overall loss optimization adjusts only the underlying MLP structural weights ($\mathbf{W}_l, \mathbf{b}_l$) and the bounded regularization coefficient:
$$\lambda_{\text{tik}} = 10^{\left(3\tanh(\lambda_{\text{raw}}) - 2\right)}$$

**Objective Function & Backpropagation**
The network is updated globally via AdamW based on the modularly chunked loss evaluations:
$$\mathcal{L}_{\text{Total}}(\mathbf{W}, \mathbf{b}, \lambda_{\text{raw}}) = \frac{1}{N_{\text{PDE}}} \sum_{i=1}^{N_{\text{PDE}}} \left\| \nabla^2 u(\mathbf{x}_i, \mathbf{p}; \hat{\mathbf{w}}) + s(\mathbf{x}_i) \right\|_2^2 + \frac{1000}{N_{\text{BC}}} \sum_{j=1}^{N_{\text{BC}}} \left\| u(\mathbf{x}_j, \mathbf{p}; \hat{\mathbf{w}}) - 0 \right\|_2^2$$
When AdamW traces the gradients backward through the MLP, it encounters the `@jax.checkpoint` boundaries. Instead of holding the massive derivative graph in memory from the forward pass, JAX rematerializes (re-calculates) the 256-wide forward chunks on the fly during the backward pass. This trades a negligible amount of compute time to successfully route the gradient updates while saving gigabytes of VRAM.