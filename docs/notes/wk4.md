### Reducing space complexity from $O(m^2)$ in direct solve to $O(1)$ with sketch-and-project
**Context:** Imagine we have a $128 \times 128$ grid. We have 8 samples in 1 batch, so there is $8000$ coordinates in total (excluding boundary conditions). The MLP outputs, say, $m = 20000$ features. Constructing the full $8000 \times 20,000$ matrix $A$ (let alone computing $A^T A$) takes a lot of memory. The ideal senario is one where we do not even construct the $8000 \times 20000$ matrix (which we need for direct solve or normal sketch-and-project).

Note: We do not want to simply do, say 1 or 2 samples in 1 batch, because more samples in 1 batch means $w_{k+1}$ updates better.
 
But how do we sample the $1000$ coordinates from the $8000 \times 20000$ matrix if we do not even build it? How do we handle the fact that after sampling, we would still have $1000 \times 20000$ matrix which would still easily OOM due to calculating laplacian assuming no other modifications?

We solve this by trading time complexity for space complexity. We will use chunking of $m$ to slowly build matrices when required. We will have two loops. The **inner loop** controls what happens for each $1000$ coordinates, and an **outer loop** that updates $w$ $8$ times for a single iteration.

**The Inner Loop**
We do not want to look at all 20,000 features at once. For clarity, we define a row block size ($N_{chunk} = 1000$) and a feature block size ($m_{chunk} = 1000$) where 
$A_{block} = \begin{bmatrix} A_1 & A_2 & \dots & A_c \end{bmatrix}$ and $A_i \in \mathbb{R}^{N_{chunk} \times m_{chunk}}$

For a single block Kaczmarz update:
$$w_{k+1} = w_k - \alpha A_{block}^T (A_{block} A_{block}^T + \epsilon I)^{-1} (A_{block} w_k - b_{block})$$

We assemble this equation iteratively by dynamically generating and discarding sub-matrices:
* **The Residual:** $A_{block} w_k = \sum A_i w_i$. (Peak space: $\mathcal{O}(N_{chunk} m_{chunk})$).
* **The Gram Matrix:** $A_{block} A_{block}^T = \sum A_i A_i^T$. (Peak space: $\mathcal{O}(N_{chunk}^2)$).
* **The Weight Update:** $w_{i, new} = w_i - \alpha A_i^T z$. (Peak space: $\mathcal{O}(N_{chunk} m_{chunk})$).

**The Result:** The peak memory required is bounded by the largest matrix held at any instant: $\mathcal{O}(\max(N_{chunk}^2, N_{chunk} m_{chunk}))$. Because we can set $N_{chunk}$ and $m_{chunk}$, it is like a constant relative to $N$ or $m$. The space complexity drops to **$\mathcal{O}(1)$**.

**The Outer Loop**
To ensure all $8000$ coordinates are solved for, we conceptually slice the full batch into 8 blocks of 1000 rows. We sequentially update the weights for each of the $1000$ coordinates.

For $j = 1$ to $8$:
$$w^{(j)} = w^{(j-1)} - \alpha A_{row\_j}^T \left( A_{row\_j} A_{row\_j}^T + \epsilon I \right)^{-1} \left( A_{row\_j} w^{(j-1)} - b_{row\_j} \right)$$

* Block 1 (the first $1000$ coordinates) updates the initial guess ($w_0 = 0$).
* Block 2 - 8 takes those weights and updates them to satisfy the next 1000 coordinates. 