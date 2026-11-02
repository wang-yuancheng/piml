### Proof for Assembling Gram Matrix (Streaming Normal Equations)

Let the global feature matrix $A$ have dimensions $N \times m$ and the target vector $b$ have dimensions $N \times 1$, where $N$ is the number of collocation points (rows) and $m$ is the number of features (columns).
We horizontally slice $A$ and $b$ into $P$ smaller mini-batches, representing them as block column matrices:

$$
A = \begin{bmatrix} A_1 \\ A_2 \\ \vdots \\ A_P \end{bmatrix}, \quad b = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_P \end{bmatrix}
$$

Taking the transpose of $A$ yields a block row matrix:

$$
A^T = \begin{bmatrix} A_1^T & A_2^T & \dots & A_P^T \end{bmatrix}
$$

To find the main Gram matrix $A^T A$, we multiply the block matrices together:

$$
A^T A = \begin{bmatrix} A_1^T & A_2^T & \dots & A_P^T \end{bmatrix} \begin{bmatrix} A_1 \\ A_2 \\ \vdots \\ A_P \end{bmatrix}
$$

$$
A^T A = (A_1^T \cdot A_1) + (A_2^T \cdot A_2) + \dots + (A_P^T \cdot A_P) = \sum_{k=1}^P A_k^T A_k
$$

Each $A_k^T A_k$ operation multiplies an $(m \times n_k)$ matrix by an $(n_k \times m)$ matrix, resulting in an $m \times m$ matrix. The same logic applies to the right-hand side vector:

$$
A^T b = \sum_{k=1}^P A_k^T b_k
$$

This removes the dependence on $N$: $A$ never needs to be stored in full. However, the accumulated $A^T A$ is still $m \times m$, so memory is $\mathcal{O}(m^2)$. This is the baseline the Sketch-and-Project solver below improves on. For example, $m = 20000$ in float64 gives a $3.2$ GB Gram matrix.

### Reducing space complexity from $\mathcal{O}(m^2)$ to $\mathcal{O}(m)$ with Sketch-and-Project

**Summary of memory costs**

| Method | What is stored | Memory |
|---|---|---|
| Direct solve (as implemented in `poisson_gauss_direct.ipynb`) | Full $A$ ($N \times m$) and $A^T A$ ($m \times m$) | $\mathcal{O}(Nm + m^2)$ |
| Streaming normal equations (above) | One row block and $A^T A$ | $\mathcal{O}(m^2)$ |
| Standard block Kaczmarz (sample code) | Full $A$, sliced by rows | $\mathcal{O}(Nm)$ |
| Matrix-free block Kaczmarz (this work) | One row block, block Gram $A_k A_k^T$, and $w$ | $\mathcal{O}(m)$ |

The key difference from streaming: streaming builds the *primal* Gram $A^T A$ (features $\times$ features, size set by $m$). Kaczmarz builds the *dual* block Gram $A_k A_k^T$ (points $\times$ points, size set by the chunk size $N_c$, which we choose). The key difference from the sample code: rows are generated on the fly from collocation points instead of being sliced from a stored $A$.

**Context:** Imagine we are processing a single sample with 10,000 spatial coordinates and an MLP that outputs $m = 20000$ features. The theoretical linear system matrix $A$ for this single sample is 10000x20000 ($1.6$ GB in float64). A direct solve would also need the 20000x20000 Gram matrix $A^T A$ ($3.2$ GB). We solve this by trading time complexity for space complexity: we never build $A$ or $A^T A$, and instead dynamically generate and discard row blocks on the fly.

**The Outer Loop (Row Blocks of $N$)**
We cannot update the weights using all 10,000 coordinates at once. Instead, we randomly permute the coordinates and partition them into row blocks of $N_c = 1000$ points. For row block $k$, the relaxed, regularized block Kaczmarz update is:

$$
w \leftarrow w - \alpha A_k^T (A_k A_k^T + \lambda I)^{-1} (A_k w - b_k)
$$

where $\alpha$ is the relaxation parameter and $\lambda$ is the Tikhonov regularization. With $\alpha = 1$ and $\lambda = 0$, this is the standard block Kaczmarz projection onto the solution set of block $k$'s equations. The updated $w$ from block $k$ is the starting point for block $k+1$. One pass over all blocks is a sweep, and we run several sweeps.

Here $A_k$ is 1000x20000 ($160$ MB), and computing its Laplacian with `jacfwd` needs several times more than that for the derivative trace. This is the motivation for also chunking the feature dimension.

**The Inner Loop (Chunking $m$)**
We split the columns of the row block into $C$ feature chunks of size $m_c = 1000$:

$$
A_k = \begin{bmatrix} A_{k,1} & A_{k,2} & \dots & A_{k,20} \end{bmatrix}, \quad w = \begin{bmatrix} w_1^T & w_2^T & \dots & w_{20}^T \end{bmatrix}^T
$$

where each $A_{k,c} \in \mathbb{R}^{1000 \times 1000}$. Both quantities needed for the update decompose exactly over column chunks:

$$
A_k w = \sum_{c=1}^{C} A_{k,c} w_c, \qquad A_k A_k^T = \sum_{c=1}^{C} A_{k,c} A_{k,c}^T
$$

* For $c = 1$ to $20$: generate $A_{k,c}$, accumulate the residual and the Gram matrix, then discard $A_{k,c}$.
* Solve for the projection vector: $z = (G + \lambda I)^{-1} (A_k w - b_k)$.
* Run the loop again to regenerate each $A_{k,c}$ and update its weight chunk: $w_c \leftarrow w_c - \alpha A_{k,c}^T z$.

This decomposition is exact, so it gives the same update as the unchunked block.

**Limitation of Feature Chunking with an MLP**
The algebra above is exact, but bounding memory by $m_c$ only works if a feature chunk can be computed without computing the others.

* For Random Fourier Features this holds: chunk $c$ is $[\sin(2\pi \mathbf{v}^T \mathbf{B}_c), \cos(2\pi \mathbf{v}^T \mathbf{B}_c)]$ and needs only $\mathbf{B}_c$.
* For MLP hidden-layer features it does not: a deeper layer needs the entire previous layer.
* In the current implementation, `get_f_chunk` runs the full network and then applies `lax.dynamic_slice`. Inside `lax.scan`, the slice index is a traced value, so XLA cannot prune the unused features. The full $N_c \times m$ feature block and its derivative tangents are computed for every chunk and then sliced.
* Robust feature chunking through an MLP is an open problem. Possible directions: static slice indices, computing RFF chunks directly from $\mathbf{B}_c$, or slicing the final layer's weight matrix.

**The Result**
Peak memory per row block is the block itself, the block Gram, and the weights:

$$
\mathcal{O}(N_c \cdot m + N_c^2 + m)
$$

With $N_c$ fixed, this is $\mathcal{O}(m)$: independent of the total number of collocation points $N$, and without the $m^2$ term of the direct solve. With working feature chunking, the first term becomes $N_c \cdot m_c$, but total memory is still $\mathcal{O}(m)$ because $w \in \mathbb{R}^m$ must be stored. The space complexity therefore cannot be $\mathcal{O}(1)$.

The cost moves to time and accuracy:
* Features are computed twice per block (once for the Gram and residual, once for the update).
* Each sweep costs $\mathcal{O}(N \cdot N_c \cdot m)$, versus $\mathcal{O}(N m^2 + m^3)$ for the normal equations.
* The PINN system is overdetermined and inconsistent. Block Kaczmarz does not converge to the exact least-squares solution, only to a neighborhood whose size scales with the residual (Needell 2010; Needell & Tropp 2014). The relaxation $\alpha < 1$ and the Tikhonov term shrink this neighborhood. Randomized Extended Kaczmarz (Zouzias & Freris 2013) converges to the exact least-squares solution if needed.

**Implementation Details**
For this example, we have:
* Batch Size: 8 samples
* Total Spatial Coordinates ($N$): 10,000 per sample
* Total MLP Features ($m$): 20,000
* Spatial Chunk Size ($N_c$): 1,000 points
* Feature Chunk Size ($m_c$): 1,000 features

**The Batch Layer**
Instead of looping through the 8 samples, we map the solver across the batch dimension with `vmap`.
* Input: A batch of 8 source arrays.
* State: 8 weight vectors, $w_0 \in \mathbb{R}^{8 \times 20000 \times 1}$, initialized to zero.
* The features depend only on the coordinates, not on the source. So $A_k$ and the block Gram $A_k A_k^T$ are identical across the 8 samples; only $b_k$, $w$ and $z$ differ. JAX's `vmap` does not batch values that do not depend on the batched input, so the Gram is computed once. A single factorization of $G + \lambda I$ could also be reused for all samples.

**Boundary Rows**
In the current implementation, every row block contains $N_c$ interior points plus *all* sampled boundary points (e.g. $250 + 500 = 750$ rows). The boundary equations are revisited in every block, which is a way of weighting the boundary condition. The block Gram is therefore $(N_c + N_{bc}) \times (N_c + N_{bc})$, so boundary points must also be chunked if $N_{bc}$ grows.

**Block Order**
Coordinates are randomly permuted once per training step, and the resulting blocks are swept in a fixed (cyclic) order. Standard Randomized Block Kaczmarz picks a random block at each iteration. Reshuffling the block order every sweep (`jax.random.permutation` on the block indices) makes the method randomized in the standard sense.

### Dual-Step Training Loop

The training loop employs a two-step alternating optimization strategy for each sampled batch of spatial coordinates:

1. **Freeze Network Parameters (Update $w$):** First, the neural network's hidden layer parameters are frozen. The model computes the spatial derivatives directly on the high-dimensional features to assemble the linear system $Aw = b$. A linear least-squares solver finds the final layer weights $w$ for the current batch. The direct solve gives the exact regularized least-squares solution; Sketch-and-Project gives an approximation.
2. **Freeze Linear Weights (Update Params):** Next, the computed final layer weights $w$ are frozen. The model computes the scalar prediction $u = f \cdot w$ first, which allows it to use the memory-saving trick below to compute the Laplacian ($\nabla^2 u$). The resulting physics and boundary losses are evaluated, and gradient descent backpropagates to update the hidden layer parameters.

This alternating process is repeated across all spatial batches until the end of the training epochs.

**Gradient flow through the solver differs between the two versions:**
* **Direct solve:** there is no `stop_gradient`, so gradients flow through `jnp.linalg.solve`. Both the hidden layers and the Tikhonov parameter $\lambda$ receive gradients, so $\lambda$ is learned.
* **Sketch-and-Project:** $w$ is wrapped in `stop_gradient`. Because $\lambda$ only affects the loss through $w$, it receives zero gradient and stays at its initialization (about $1.03 \times 10^{-5}$), apart from negligible AdamW weight decay. To learn $\lambda$ here, it would need a separate objective, or the stop-gradient would need to be removed.

#### Saving memory during update of MLP weights

We can compute $\nabla^2(f \cdot w)$ instead of $(\nabla^2 f) \cdot w$ because the Laplacian is a linear operator and $w$ does not depend on the coordinates $x$ and $y$. The same holds for any linear differential operator, such as the Darcy operator; for a nonlinear PDE operator the problem could not be written as $Aw = b$ in the first place. This makes the JAX computational graph much smaller: by doing the dot product first, we multiply the $(N, m)$ features by the $(m, 1)$ weights to get a vector $u$ of shape $(N, 1)$. Taking the Laplacian of an $(N, 1)$ vector instead of an $(N, m)$ matrix is a large memory saving.

This trick only applies to this step. Building $A$ for the solver still requires the Laplacian of every feature separately, which is why the solver step needs the chunking above.