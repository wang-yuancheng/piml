#### We do not need this right now. Ignore.

##### Possible problem
Because the dataset have peaks of $N$ ranging up to $N=18$, the $p$ in $v$ would have mostly $0.0$ most of the times at the bottom of the vector. We are safe for now because the zeros will zero out their corresponding random weights in $B$. However the variance of the output of the RFF (inputs to the MLP) would fluctuate depending on how many peaks the current sample has and might cause the features to be not as good. 

##### Explanation
The input to the sine function relies on the dot product of a column in $B$ and the input vector $v$. 
Let's call one of those single frequency projections $z_j$:$$z_j = b_j^T v = b_{1,j}v_1 + b_{2,j}v_2 + \dots + b_{56,j}v_{56}$$

We have an input dimension of $56$ because we concatenate $1$ value for the $x$ coordinate, $1$ value for the $y$ coordinate, and $18$ maximum peaks $\times$ $3$ values per peak ($\mu_x$, $\mu_y$, and $\sigma$).

The $B$ matrix is initialized with random weights drawn from a Gaussian distribution, $b_{i,j} \sim \mathcal{N}(0, \sigma^2)$. Therefore, $z_j$ is a linear combination of independent, normally distributed random variables.The statistical distribution of this single projection $z_j$ is:
$$\mathbb{E}[z_j] = 0$$$$\text{Var}(z_j) = \sum_{i=1}^{56} v_i^2 \text{Var}(b_{i,j}) = \sigma^2 \sum_{i=1}^{56} v_i^2$$

We also see that $\sum_{i=1}^{56} v_i^2$ is simply the squared L2 norm $\|v\|_2^2$, the variance becomes:$$\text{Var}(z_j) = \sigma^2 \|v\|_2^2$$

This shows that the variance of the values fed into the activation functions is directly proportional to the magnitude of $v$, meaning a sample with $18$ peaks will produce a much wider spread of inputs than a sample with $1$ peak. 
