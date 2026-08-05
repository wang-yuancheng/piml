# sketch-and-project

**Randomized Block Kaczmarz.** Standard direct solves have a space complexity of $\mathcal{O}(m^2)$, where $m$ is the number of features. By trading time complexity for space complexity using a "Double Chunking" strategy, we reduce this to $\mathcal{O}(1)$.

**The Mathematics of Double Chunking:**
The Block Kaczmarz update equation for a spatial batch is:
$$w_{k+1} = w_k - \alpha A_{batch}^T (A_{batch} A_{batch}^T + \epsilon I)^{-1} (A_{batch} w_k - b_{batch})$$

To execute this without building the massive $\mathbf{A}_{batch}$ matrix, we dynamically generate and discard sub-matrices on the fly:
1. **Outer Loop (Sketching $N$):** We randomly sample a spatial batch (e.g., $N_{batch} = 1000$) from the full coordinate grid.
2. **Inner Loop (Sketching $m$):** We chunk the feature dimension (e.g., $m_{chunk} = 1000$). We accumulate the residual ($\mathbf{A}_{batch} \mathbf{w}_k = \sum \mathbf{A}_i \mathbf{w}_i$) and the Gram Matrix ($\mathbf{G} = \sum \mathbf{A}_i \mathbf{A}_i^T$) iteratively. After updating the corresponding chunk of weights, $\mathbf{A}_i$ is wiped from memory.

The peak memory required is bounded solely by the largest matrix held at any instant: $\mathcal{O}(\max(N_{batch}^2, N_{batch} m_{chunk}))$. 

> **Note:** $m$-chunking is especially critical when scaling up the input vector. If we expand the model to take in encoded domain parameters—such as $a(x,y)$ in Darcy Flow—the number of required MLP features explodes. $m$-chunking ensures the GPU does not OOM regardless of how wide the latent basis space becomes.