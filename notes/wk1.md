### Main Topic - Solve the Poisson equation with PINNs memory efficiently.
#### Introduction
Poisson Equation: ∇^2 ϕ = f in domain Ω and ϕ = 0 on the boundary of dΩ. Describes how a potential field behaves in the presence of a known source. We can solve this with an exact solver. However, if there are too many unknowns, causing the system of linear equations to become very large, then we need to use numerical methods. 
For example, to solve a 2D Poisson equation, we can use Central Difference Method to approximate the second derivatives.
We will arrive at a formula known as the 5-point stencil. For 2-dimensional Poisson equation: \frac{u(i+1,j)+ u(i−1,j) + u(i,j+1) + u(i,j−1) − 4u(i,j)}{h^2} = f(i,j)
Then, we apply this 5-point stencil to every single point in grid except points on the boundary. We do not include boundary points to be solved because the boundary conditions are provided by us (usually at boundary, u(i,j) = 0), meaning we already know the answers. 
Now, we assemble the linear system by collating every non-boundary point in the grid. This gives us Au = f, where A is the matrix with each row representing the stencil equation for one specific point on the grid.
Typically, the system has more equations than unknowns (such as due to noise, or finding system with data like in inverse problems). This is called a over-deterministic system and there will not be an exact solution. 
We can use least squares where we minimize (|| Au - f ||_2)^2 (Cost function: squared L2 norm of residual Au - f) to find an approximate solution. When we do that, we are aiming to find a specific u to make the residual as small as possible. By taking the gradient of the cost function with respect to vector u and set the equation = 0, we will derive the Normal Equation: A.T A u = A.T f.

However, using least squares on the traditional grid variables (Au=f) is not the method we want to do here to solve the Poisson equation. This is because it only gives us discrete values at fixed points, rather than a continuous mathematical function that can be evaluated anywhere. 

Here, we want to use a PINN to solve it, meaning the PINN is the solution to the Poisson equation. To do so, we feed a grid of coordinates (which can be predetermined or randomly sampled) representing the physical space (these are collocation points we define ourselves, and do not need to be from real-world sensors) into the PINN. Instead of standard backpropagation which is extremely slow and prone to getting stuck in local minima due to the highly non-linear, non-convex loss landscape, we convert the PINN training back into a Linear Least Squares Matrix problem.

We initialize the weights of the non-linear hidden layers to be random then freeze them. During the forward pass, the features that reach the last linear layer will be multiplied by the weight and a bias is added to give a prediction for a single coordinate point. We can use this to build Aw = b (only solve for the final layer weights as activation functions of hidden layers are non-linear). However, we do not do least squares on this system yet. Because we are solving the Poisson equation, we need the system to reflect the physics of the Poisson equation. We need to find the laplacian of the generated features (ϕ) before we build Matrix A.

There are two methods to get the laplacian of these features (ϕ):

1. Using jax.grad() twice
This method is slow because jax.grad() calculates the exact analytical derivative by tracing the calculus chain rule through the network's code, which is computationally exhausting to do for tens of thousands of points.

2. Using 5 point stencil
This method is much faster because it skips exact calculus entirely and uses fast, simple algebraic approximations (addition and subtraction) between neighboring points on the uniform grid.

Now, we have a system with the laplacian applied: Aw=b, where the columns of Matrix A are now the Laplacians of the features (∇^2ϕ), w is the vector of final unknown weights, and b is the vector of known source terms.

Now, we can use least squares approximation to minimize (∣∣Aw−b∣∣_2)^2. In a single, instant analytical step (using the Normal Equations, without needing many iterations of gradient descent), we get our desired weights and biases and that is our solution to the Poisson equation.

Unlike the matrix A in a standard Au=f Poisson system is relatively sparse, the matrix A in the Aw=b linear layer system is both dense and prone to ill-conditioning. When we calculate A.T A, this resulting square matrix will become even more ill-conditioned due to squaring the condition number κ: κ(A.T A)=(κ(A))^2. At this point, the matrix will require vastly more iterations to converge with numerical methods or completely fail with direct solving. To make it less ill-conditioned, we can add L2 regularization so that the cost function = ∣∣Au−f∣∣^2 + λ∣∣u∣∣^2. This essentially adds a small value to the diagonal of A.T A. However, the matrix A.T A will still be huge and we still need memory efficient methods to solve it.


### How to handle the bias term


#### Apply "Sketch and Project" methods to solve the Normal equation.







### Extras
To visualize the matrix landscape, we plot the cost function J(w) = (∣∣Aw−f∣∣_2)^2, where the x and y axis are the weights w1, w2, and the vertical z axis is the error J. The landscape will be a elliptic paraboloid if and only if the system is perfectly linear and Matrix A has full column rank.

L1 norm the sum of squared components
L2 norm is square root of the sum of squared components
L2 regularization (or Ridge regression) is the addition of a penalty term to the cost function equal to the square of the magnitude of coefficients (L2 norm). Loss = ∣∣Xw−y∣∣^2 + λ∣∣w∣∣^2

float32 4 bytes per number, float64 8 bytes per number
We should use float64 for more precision when dealing with ODEs/PDEs

#### Paper: Evolutionary Optimization of Physics-Informed Neural Networks: Advancing Generalizability by the Baldwin Effect
Goal: Geeneralization across a family of PDEs. If you train it on a bunch of different parameters for the convection-diffusion equation, it can easily solve a new convection-diffusion problem with parameters it hasn't seen before.  
Method: Fix network structure and find the best starting weights and biases. It groups the weights into blocks and searches for the optimal probability distribution (the mean and the spread) to sample those initial weights from. It uses a standard Multi-Layer Perceptron (MLP) as its foundation.  

#### Paper: Out-of-Distribution Generalization for Neural Physics Solvers
Goal: Making model handle new physical geometries that are OOD (like moving from a chip with 1-3 inlets to a chip with 4 inlets). Furthermore, it actually integrates this solver into a guided generative diffusion model to automatically design optimized fluidic chips.
Method: NOVA actively rewires the network itself instead of just adjusting weights and biases. It uses Neural Architecture Search (NAS) to swap out operators, change connections, and alter the sequence of how the data flows through the network to find a structure inherently suited for the physics problem. It uses a Convolutional Neural Network (CNN) foundation, specifically a generalized U-Net architecture. This makes sense given the shift toward handling 2D spatial geometries and fluid dynamics. 