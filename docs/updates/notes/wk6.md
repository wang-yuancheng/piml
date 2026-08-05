*update to wk4 notes.*

### Reducing space complexity from $\mathcal{O}(m^2)$ in direct solve to $\mathcal{O}(1)$ with sketch-and-project

**Context:**  Imagine we are processing a single sample with 10,000 spatial coordinates and an MLP that outputs $m = 20000$ features. The theoretical linear system matrix $A$ for this single sample is 10000x20000. Constructing this full matrix (let alone computing the 10000x10000 Gram matrix $A A^T$) will instantly cause an Out-Of-Memory (OOM) error.We solve this by trading time complexity for space complexity using a "Double Chunking" strategy. We never build the full 10000x20000 matrix. Instead, we dynamically generate and discard tiny sub-matrices on the fly using two nested loops.

**The Outer Loop (Sketching $N$)**
We cannot update the weights using all 10,000 coordinates at once. Instead, we randomly sample a spatial batch size ($N_{batch} = 1000$) from the 10,000 coordinates.The Kaczmarz update equation for this specific 1000-point spatial batch is:$$w_{k+1} = w_k - \alpha A_{batch}^T (A_{batch} A_{batch}^T + \epsilon I)^{-1} (A_{batch} w_k - b_{batch})$$At this stage, our target matrix $A_{batch}$ is 1000x20000. However, calculating the Laplacian to generate even this 1000x20000 matrix will still OOM.

**The Inner Loop (Sketching $m$)**
To execute the math for our 1000 spatial points without building the 1000x20000 matrix, we chunk the feature dimension. We define a feature block size ($m_{chunk} = 1000$). We can represent $A_{batch}$ as a series of 20 smaller blocks:$A_{batch} = \begin{bmatrix} A_1 & A_2 & \dots & A_{20} \end{bmatrix}$ where each $A_i \in \mathbb{R}^{1000 \times 1000}$. We assemble the Kaczmarz update components iteratively inside a loop. 

* For $i = 1$ to $20$: Generate Block: Pass the 1000 spatial points through the specific subset of the MLP to generate $A_i$ (Size: 1000x1000)
* Accumulate Residual: $A_{batch} w_k = \sum A_i w_i$.
* Accumulate Gram Matrix: $G = \sum A_i A_i^T$.
* Discard Block: $A_i$ is wiped from memory, and the loop moves to $i+1$.

Once the loop finishes, we solve for the projection vector: $z = (G + \epsilon I)^{-1} (A_{batch} w_k - b_{batch})$.

Finally, we run the inner loop one more time to generate $A_i$ again, update the corresponding chunk of weights $w_{i, new} = w_i - \alpha A_i^T z$, and discard $A_i$.

**The Result**
By using this Double Chunking method, the peak memory required is bounded solely by the largest matrix held at any instant: $\mathcal{O}(\max(N_{batch}^2, N_{batch} m_{chunk}))$.Because we locked $N_{batch} = 1000$ and $m_{chunk} = 1000$, our maximum memory footprint is just a 1000x1000 matrix, entirely decoupling our memory usage from the total 10,000 coordinates and 20,000 features. The space complexity effectively drops to $\mathcal{O}(1)$.

**Implementation Details**
For this example, we have 
* Batch Size: 8 samples
* Total Spatial Coordinates ($N$): 10,000 per sample
* Total MLP Features ($m$): 20,000
* Spatial Chunk Size ($N_{chunk}$): 1,000 points
* Feature Chunk Size ($m_{chunk}$): 1,000 features

**The Batch Layer**
Instead of looping through the 8 samples, we map the entire solver process across the batch dimension.
* Input: A batch of 8 source arrays and parameter arrays.
* Process: Spawns 8 completely independent, isolated Kaczmarz solvers on the GPU.
* State: Initializes 8 blank weight vectors: $w_0 \in \mathbb{R}^{8 \times 20000 \times 1}$.

**The Outer Loop**
Inside each isolated solver, we break the 10,000 coordinates into manageable pieces. We cannot pass all 10,000 into the network at once without building a massive Gram matrix.

We loop 10 times. Each step grabs a new chunk of 1,000 spatial coordinates. This is recalculating the features but is neccessary. The updated $w$ from the first 1,000 points is passed as the starting point for the next 1,000 points, and so on.

**The Inner Loop**
For the current set of 1,000 spatial points, we apply the Kaczmarz update without building the $1000 \times 20000$ derivative matrix.

We loop 20 times over the feature dimension. Inside this loop, we pass the 1,000 spatial coordinates through the MLP, but immediately use lax.dynamic_slice to isolate just 1,000 features before applying JAX's autodiff (jacfwd). We compute the Laplacian only for these 1,000 features ($A_i \in \mathbb{R}^{1000 \times 1000}$), accumulate its contribution to the residual and the Gram matrix ($G \mathrel{+}= A_i A_i^T$), then discard the autodiff graph and move to the next 1,000 features. Once the Gram matrix is fully assembled from the 20 chunks, solve for the projection $z$. Run the inner loop one final time to re-generate the $A_i$ chunks, calculate the weight updates $w_{i, new} = w_i - \alpha A_i^T z$, and piece the newly updated $20000 \times 1$ vector back together.

By structuring the implementation this way, the XLA compiler never materializes the $10000 \times 20000$ matrix ($1.6$ GB) or the $10000 \times 10000$ Gram matrix.

The absolute largest matrix the GPU holds in VRAM at any specific instant is the batched Gram matrix for the chunk: $8 \times 1000 \times 1000$.