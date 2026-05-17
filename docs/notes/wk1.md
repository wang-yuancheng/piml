### Main Topic - Solve the Poisson equation with PINNs memory efficiently.
#### Introduction
Poisson Equation: $\nabla^2 \phi = f$ in domain $\Omega$ and $\phi = 0$ on the boundary $\partial\Omega$. Describes how a potential field behaves in the presence of a known source. We can solve this with an exact solver. However, if there are too many unknowns, causing the system of linear equations to become very large, then we need to use numerical methods. 

For example, to solve a 2D Poisson equation, we can use the Central Difference Method to approximate the second derivatives. We will arrive at a formula known as the 5-point stencil. For the 2-dimensional Poisson equation: 

$$\frac{u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1) - 4u(i,j)}{h^2} = f(i,j)$$

Then, we apply this 5-point stencil to every single point in the grid except points on the boundary. We do not include boundary points to be solved because the boundary conditions are provided by us (usually at the boundary, $u(i,j) = 0$), meaning we already know the answers. 

Now, we assemble the linear system by collating every non-boundary point in the grid. This gives us $Au = f$, where $A$ is the matrix with each row representing the stencil equation for one specific point on the grid.

Typically, the system has more equations than unknowns (such as due to noise, or finding a system with data like in inverse problems). This is called an over-determined system and there will not be an exact solution. We can use least squares where we minimize $(||Au - f||_2)^2$ (Cost function: squared L2 norm of residual $Au - f$) to find an approximate solution. When we do that, we are aiming to find a specific $u$ to make the residual as small as possible. By taking the gradient of the cost function with respect to vector $u$ and setting the equation to 0, we will derive the Normal Equation: $A^T A u = A^T f$.

#### Physics-Informed Neural Networks (PINNs)
However, using least squares on the traditional grid variables ($Au=f$) is not the method we want to do here to solve the Poisson equation. This is because it only gives us discrete values at fixed points, rather than a continuous mathematical function that can be evaluated anywhere. 

Here, we want to use a PINN to solve it, meaning the PINN is the solution to the Poisson equation. Specifically, we will use a random feature model where the inputs are passed through an encoder, and the generated features will be passed through a linear layer we specify to give the output. To do so, we feed a grid of coordinates (which can be predetermined or randomly sampled) representing the physical space (these are collocation points we define ourselves, and do not need to be from real-world sensors) into the PINN. Instead of standard backpropagation, which is extremely slow and prone to getting stuck in local minima due to the highly non-linear, non-convex loss landscape, we convert the PINN training back into a Linear Least Squares Matrix problem.

We initialize the weights of the non-linear hidden layers to be random then freeze them. During the forward pass, the features that reach the last linear layer will be multiplied by the weights and a bias is added to give a prediction for a single coordinate point. We can use this to build $Aw = b$ (only solve for the final layer weights as activation functions of hidden layers are non-linear). However, we do not do least squares on this system yet. Because we are solving the Poisson equation, we need the system to reflect the physics of the Poisson equation. We need to find the Laplacian of the generated features ($\phi$) before we build Matrix $A$.

There are two methods to get the Laplacian of these features ($\phi$):

1. **Using JAX's Automatic Differentiation:** Doing `jax.grad()` twice is slow because it calculates the exact analytical derivative by tracing the calculus chain rule through the network's code, which is computationally exhausting to do for tens of thousands of points. Hence, we use `jacfwd()` to get the specific derivatives `b_xx` and `b_yy` we need to avoid computing the entire Hessian matrix.
2. **Using 5-point stencil:** This method is much faster because it skips exact calculus and uses simple algebraic approximations (addition and subtraction) between neighboring points on the uniform grid. However, they do not give exact values.

Now, we have a system with the Laplacian applied: $Aw = b$, where the columns of Matrix $A$ are now the Laplacians of the features ($\nabla^2\phi$), $w$ is the vector of final unknown weights, and $b$ is the vector of known source terms.

Now, we can use least squares approximation to minimize $(||Aw - b||_2)^2$. In a single analytical step using the Normal Equations and taking the inverse: $w = (A^TA+ \lambda I)^{−1}A^Tb$, we skip the many iterations of gradient descent, and get our desired weights and biases for the last linear layer, and that is our solution to the Poisson equation.

Taking the inverse of a big matrix A can be memory intensive and slow as the computation for $A^TA$ is $O(Nd^2)$ (the time complexity for computing each element in $d \times d$ matrix $A^T A$ is $O(N)$ as we are computing the dot product of 2 vectors of length $N$). Hence, we can use sketch and project methods that are faster and less memory intensive to approximately solve the Normal equations.

#### Ill-Conditioning and Regularization
Unlike the matrix $A$ in a standard $Au=f$ Poisson system which is relatively sparse, the matrix $A$ in the $Aw=b$ linear layer system is both dense and prone to ill-conditioning. When we calculate $A^T A$, this resulting square matrix will become even more ill-conditioned due to squaring the condition number $\kappa$: $\kappa(A^T A) = (\kappa(A))^2$. At this point, the matrix will require vastly more iterations to converge with numerical methods or completely fail with direct solving. 

To make it less ill-conditioned, we can add L2 regularization so that the cost function = $||Aw - b||_2^2 + \lambda||w||_2^2$. This is the $\lambda$ we saw above. This essentially adds a small value to the diagonal of $A^T A$. However, the matrix $A^T A$ will still be huge, and we still need memory-efficient methods to solve it. That is when we apply sketch and project methods to solve the linear system.

### Extras
To visualize the matrix landscape, we plot the cost function $J(w) = (||Aw - b||_2)^2$, where the x and y axes are the weights $w_1, w_2$, and the vertical z axis is the error $J$. The landscape will be an elliptic paraboloid if and only if the system is perfectly linear and Matrix $A$ has full column rank.

* L1 norm is the sum of the absolute values of the components.
* L2 norm is the square root of the sum of squared components.
* L2 regularization (or Ridge regression) is the addition of a penalty term to the cost function equal to the square of the magnitude of coefficients (squared L2 norm). $Loss = ||Xw - y||_2^2 + \lambda||w||_2^2$

`float32` uses 4 bytes per number, `float64` uses 8 bytes per number. We should use `float64` for more precision when dealing with ODEs/PDEs.

#### Paper: Evolutionary Optimization of Physics-Informed Neural Networks: Advancing Generalizability by the Baldwin Effect
* **Goal:** Generalization across a family of PDEs. If you train it on a bunch of different parameters for the convection-diffusion equation, it can easily solve a new convection-diffusion problem with parameters it hasn't seen before.
* **Method:** Fix network structure and find the best starting weights and biases. It groups the weights into blocks and searches for the optimal probability distribution (the mean and the spread) to sample those initial weights from. It uses a standard Multi-Layer Perceptron (MLP) as its foundation.

#### Paper: Out-of-Distribution Generalization for Neural Physics Solvers
* **Goal:** Making the model handle new physical geometries that are OOD (like moving from a chip with 1-3 inlets to a chip with 4 inlets). Furthermore, it actually integrates this solver into a guided generative diffusion model to automatically design optimized fluidic chips.
* **Method:** NOVA actively rewires the network itself instead of just adjusting weights and biases. It uses Neural Architecture Search (NAS) to swap out operators, change connections, and alter the sequence of how the data flows through the network to find a structure inherently suited for the physics problem. It uses a Convolutional Neural Network (CNN) foundation, specifically a generalized U-Net architecture. This makes sense given the shift toward handling 2D spatial geometries and fluid dynamics.