**Mathematical Proof of Block Gram Assembly**
Let the global feature matrix $A$ have dimensions $N \times m$ and the target vector $b$ have dimensions $N \times 1$. 
We horizontally slice $A$ and $b$ into $P$ smaller mini-batches, representing them as block column matrices:
$$A = \begin{bmatrix} A_1 \\ A_2 \\ \vdots \\ A_P \end{bmatrix}, \quad b = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_P \end{bmatrix}$$

Taking the transpose of $A$ yields a block row matrix:
$$A^T = \begin{bmatrix} A_1^T & A_2^T & \dots & A_P^T \end{bmatrix}$$

To find the main Gram matrix $A^T A$, we multiply the block matrices together:
$$A^T A = \begin{bmatrix} A_1^T & A_2^T & \dots & A_P^T \end{bmatrix} \begin{bmatrix} A_1 \\ A_2 \\ \vdots \\ A_P \end{bmatrix}$$

$$A^T A = (A_1^T \cdot A_1) + (A_2^T \cdot A_2) + \dots + (A_P^T \cdot A_P) = \sum_{k=1}^P A_k^T A_k$$

Each $A_k^T A_k$ operation multiplies an $(m \times n_k)$ matrix by an $(n_k \times m)$ matrix, resulting in an $m \times m$ matrix. Therefore, the big $N$ dimension of the dataset is reduced. The same logic applies to the gradient vector:
$$A^T b = \sum_{k=1}^P A_k^T b_k$$