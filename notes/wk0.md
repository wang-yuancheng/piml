#### Randomized Numerical Linear Algebra (RandNLA) & Sketch-and-Project

To solve **$Ax = b$**, if **$A$** is a square and invertible, we can just compute **$x = A^{-1}b$**. But, it is not efficient when **$A$** is large. It also does not work when **$A$** is not invertible and/or not a square. We can use direct solvers like Gaussian elimination, but they become computationally infeasible when **$A$** is too large. We look at 2 methods here that optimize the solving with iterative linear solvers.

* **Randomized Block Coordinate Descent (in numpy):** Iteratively updates **$x$** by fixing a few variables at a time.
* **Randomized Block Kaczmarz (in JAX):** Iteratively updates **$x$** by satisfying a few equations at a time.

Reference Notebook: -

---

### Randomized Block Coordinate Descent

In test 1, we solve for **$x$** with **$A$** and **$b$**. 
1. Get 3 matrices: **$A_{BB}$**, **$A_B$**, and **$b_B$**.
2. Compute the residual with **$r_B = A_B x - b_B$**.
3. Solve the smaller system **$A_{BB} d_B = r_B$** to find the correction vector **$d_B$**.
4. Subtract the correction vector **$d_B$** from the guessed **$x$**.
5. Iterate for `num_iterations`.

In test 2, we create **$x$** artificially; then, we create **$b$** with an artificially generated symmetric positive definite matrix **$A$**. This allows us to know the exact answer **$x$** to compare after we have solved for **$x$**.

For **a** symmetric positive definite matrix **$A$**, we use **$A = M^T M + \lambda I$**. **$A$** is known as **a** well-conditioned matrix.

*(Reference: https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/804ab1e53134741d2044d241b50a285e_MIT18_06SCF11_Ses3.1sum.pdf)*

#### Properties of symmetric positive definite (SPD) matrices
1.  **Symmetric:** Perfect mirror image of itself across the main diagonal (**$A = A^T$**). (Symmetric matrices always have real eigenvalues).
2.  All eigenvalues of $A$ are strictly positive (Definition of positive definite).
3.  For any non-zero vector **$v$**, the value **$v^T A v$** is always strictly greater than zero. (Definition of positive definite). This guarantees that the landscape has exactly one unique global minimum and zero flat spots, meaning it is strictly convex.
    
    *Proof:*
    **$f(\vec{v}) = \vec{v}^T A \vec{v} = \vec{v}^T \cdot (A\vec{v}) = C$** (where **$C$** is a scalar).
    If **$\vec{v}$** is an eigenvector, then **$A\vec{v} = \lambda\vec{v}$**. 
    Substituting this in gives **$\vec{v}^T \cdot (\lambda\vec{v}) = \lambda(\vec{v}^T \cdot \vec{v})$**.
    The dot product **$\vec{v}^T \cdot \vec{v}$** for a 3D vector **$\begin{bmatrix} a & b & c \end{bmatrix}^T$** is just **$a^2 + b^2 + c^2$**, which is strictly a positive number as long as not all of **$a, b, c = 0$**. Therefore, if **$\lambda > 0$**, the resulting scalar **$C$** is positive.
  
**Points** 2 and 3 are equivalent.

We used **$A = M^T M + \lambda I_{n \times n}$**. We use **$\lambda = 10$** here to improve conditioning; using **$\lambda = 0.1$**, the matrix takes very long to converge. We are adding exactly the scalar **$\lambda$** to every single eigenvalue of the matrix **$A$**, to increase the value of **$A$**'s eigenvalues.

#### Geometric interpretation
When we set **$f(v) = v^T A v = C$**, **$C$** represents a level surface (e.g., a 3D surface if parameterized by **$a, b, c$**).
* **At the origin:** **$f(\vec{0}) = 0$**. 
* **Expanding Outward:** Increasing **$C$** means the level surface moves further from the origin in **$a, b, c$** space.

We shouldn't draw wobbly, irregular level surfaces for a symmetric matrix **$A$**. Because **$A$** is symmetric, it has perfectly orthogonal eigenvectors. This guarantees the level surfaces are perfectly symmetrical ellipsoids (or ellipses in 2D). The orthogonal eigenvectors of **$A$** form the exact axes of these ellipsoids.

Also, the eigenvalues tell us the steepness of the surface. Along these eigenvector axes, the function simplifies to **$\lambda_1 a^2 + \lambda_2 b^2 + \lambda_3 c^2 = C$**. If a specific eigenvalue is big, the function increases rapidly in that direction. This makes that part of the level surface very steep (the ellipsoid is squeezed tightly along that specific axis).

#### Optimization landscape
For any symmetric positive definite matrix **$A$**, solving **$Ax = b$** is equivalent to finding the vector **$x$** that minimizes the following quadratic function:
$$f(x) = \frac{1}{2}x^T A x - b^T x$$

To find the minimum of the landscape, we set the gradient ($\nabla f(x) = Ax - b$) to zero: **$Ax - b = 0 \implies Ax = b$**

#### Results
For a small matrix **$A$**, Gaussian elimination is faster. But for a bigger matrix **$A$**, an iterative solver is faster. However, it is sensitive to matrix conditioning, and can run out of time before reaching the minima, leading to inaccurate results if `num_iterations` is off.

---

### Randomized Block Kaczmarz

For standard Kaczmarz (block size = 1), we move our current guess **$x_0$** towards the intersection iteratively by:
1. Projecting the current guess orthogonally onto one hyperplane.
2. Projecting that new point orthogonally onto the next hyperplane.
3. Repeating for `num_iterations` until converging at the intersection.

**Proof (for block size = 1):**
For a single linear equation, we can write it as **$a^T x = b$**, and **$a^T$** would be the normal vector of the hyperplane, **$x$** is the unknown vector, and **$b$** is the scalar target.
To move **$x_0$** perpendicularly down onto the plane, we just move it in the direction of the normal vector **$a$**. To know how much in that direction to move, we solve **$x_{new} = x_0 + \alpha a$**, and since **$x_{new}$** is on the plane, **$a^T(x_{new}) = b \implies a^T(x_0 + \alpha a) = b$**.

Solving for **$\alpha$**, we get:
$$\alpha = \frac{b - a^T x_0}{a^T a} \implies x_{new} = x_0 + \frac{b - a^T x_0}{||a||^2} a$$

**$a^T a$** is just the squared length of the normal vector **$a$**, and **$a^T x_0 - b$** is the residual **$r$**. Notice that the weight is **$\frac{-r}{||a||^2}$**. This is because our point is initially away from the hyperplane. We need to move in the negative direction of the normal vector, hence step size **$\alpha$** must be negative.

By iteratively projecting the current guess onto the next hyperplanes, we converge to the answer (the intersections of all hyperplanes).

**Proof:** Let the current guess be **$x_k$**, the projected guess be **$x_{k+1}$**, and the answer be **$x^*$**.
With the Pythagorean identity, we have:
$$||x_{k+1} - x^*||^2 + ||x_{k+1} - x_k||^2 = ||x_k - x^*||^2 \implies ||x_{k+1} - x^*||^2 = ||x_k - x^*||^2 - ||x_{k+1} - x_k||^2$$

We see that the new squared error **$||x_{k+1} - x^*||^2$** is equal to the old squared error, minus a positive number. Hence, if the step size (**$||x_{k+1} - x_k||^2$**) is non-zero, the distance to the true solution strictly decreases with every single iteration.

#### Block Kaczmarz
For Block Kaczmarz (block size > 1), we project our current guess **$x_0$** directly onto the intersection of all the hyperplanes in our randomly selected block simultaneously. Here, we cannot just follow one normal vector. We must move in a direction that is a linear combination of all the normal vectors in our block. 

To do so, we use a Gram System. The projection uses the Gram Matrix **$G$**. Actually, for standard Kaczmarz, it is a special case where the block has a single row, so **$G$** is a 1x1 scalar **$a$**.

**Gram Matrix $G = A_B A_B^T$:**
1. **$G$** is always symmetric (and square), hence invertible.
2. **$G$** is Positive Semi-Definite. Its eigenvalues are always non-negative (**$\lambda \ge 0$**).

From standard Kaczmarz, we have **$x_{new} = x_0 + \frac{b - a^T x_0}{||a||^2} a \implies x_{new} = x_0 - \frac{r}{||a||^2} a$**, where **$\frac{r}{||a||^2}$** is the weight of the projection for the current guess. The weight is **$z$** in the Gram system. But, how do we derive the Gram System?

**Derivation:** **$A_B$** are our normal vectors; our movement direction is **$A_B^T z$**.
We start with **$x_{new} = x_0 - A_B^T z$**. Since **$x_{new}$** is on the intersection, **$A_B(x_{new}) = b_B \implies A_B(x_0 - A_B^T z) = b_B \implies A_B x_0 - b_B = (A_B A_B^T) z$**.
Since residual **$r = A_B x_0 - b_B$** and **$A_B A_B^T$** is the Gram Matrix **$G$**, we have the system **$Gz = r$**.

**Gram System:** **$(A_B A_B^T)z = r$**
We add a tiny jitter (small constant) to the diagonal of **$G$** for numerical stability (ensuring **$\lambda > \text{small constant}$**) as some of the eigenvalues of **$G$** may be extremely close to 0.

**The** Kaczmarz method treats each equation as an **$m-1$** dimensional hyperplane in **$m$** dimensional space, where **$m$** is the total number of unknown variables (the no. of columns of **$A_B$**). The solution to the subset of linear equations is the intersection of the **$m-1$** hyperplanes as mentioned above.

---

### Time Delay Embedding & SINDy

**Why use time-delay embedding?**
We can highlight the difference between chaos and noisy dynamics.

#### Terminology
* **Manifold:** A topological space that locally resembles Euclidean space near each point.
* **Phase space:** A space where every possible state of a system is represented as a single point. Every degree of freedom or parameter of the system is represented as an axis of a multidimensional space.
* **Attractor:** A set of states toward which a system tends to evolve towards, for a wide variety of starting conditions of the system.
* **Isomorphism:** A structure-preserving mapping or morphism between two structures of the same type that can be reversed by an inverse mapping.
* **Diffeomorphism:** An isomorphism of differentiable manifolds. It is an invertible function that maps one differentiable manifold to another such that both the function and its inverse are continuously differentiable.
* **Lyapunov exponent of a dynamical system:** A quantity that characterizes the exponential rate of separation of infinitesimally close trajectories.
* **Hankel Matrix:** Each column in the Hankel Matrix is one data point in the delay coordinates (in the phase space). To put it simply, just delay the time series data and stack shifted copies of the data next to each other.

We typically start with the Lorenz system when testing algorithms for nonlinear dynamics. We use time-delay embedding to transform a sequence of partial observations into a higher-dimensional phase space that is topologically equivalent to the system's true, unobserved state space**,** shown possible by Takens' Theorem.

**Takens' Theorem:**
Simply put, you can reconstruct a shadowed version of an original manifold simply by looking at one of its time series projections, provided we choose **$\tau$** and the no. of dimensions (**$m$**) appropriately. Full state attractors constructed with varying dimensions are diffeomorphic with one another as well as the true full state. Let the no. of dimensions of the full state coordinates be **$m$**. With Takens' theorem, we need the no. of dimensions in the phase space (**$n$**) to be at least **$2m+1$** (**$n > 2m+1$**).

**Due to the Picard-Lindelöf Theorem:**
Simply put, as long as the system is smooth, the paths in phase space will not intersect as the states are deterministic and unique. In theory, we need **$\tau > 0$** for the paths to not appear to merge or intersect, but practically, **$\tau$** needs to be larger so that delayed measurements are distinct.

**The Lorenz system:**
A model of weather with 3 variables. Hence, it has a 3D phase space. The Lorenz system never settles down and never exactly repeats itself, forming a Strange Attractor. If we take a single time series variable, and plot the phase space, where the 3 axes are **$X(t)$**, **$X(t-\tau)$** and **$X(t-2\tau)$**, we can construct a shadowed version of the original butterfly manifold.

Each point in the 3-dimensional reconstruction can be thought of as a time segment, with different points capturing different segments of history of variable **$X$**. The reconstructed manifold is the library or collection of the historical behaviour of **$X$**. Reconstruction preserves properties, such as the topology of the manifold and its Lyapunov exponents. This is also a 1-to-1 mapping between the original manifold butterfly attractor and the reconstruction, allowing us to recover states of the original dynamic system by using legs of a single time series.

Given data in the full state coordinates, we can discover underlying dynamics with Sparse Identification of Nonlinear Dynamics (SINDy) with a form of sparse regression.

#### Neural Networks
We can use neural networks to approximate the full state coordinates with the delay coordinates (to discover the actual differential equations that describe the system).

To start, we have state variables like **$z_1, z_2 \dots$** and we build a huge library of potential terms like **$z_1^2, \sin(z_1) \dots$** Then, with SINDy, we can find out which terms are important, as well as get the analytical function **$f(z)$**. SINDy also aims to minimize the residual error and maximize sparsity (trying to get equations to have as few terms as possible). SINDy is giving the rate of change (velocity) of **$z$**. We cannot just run SINDy directly on the Hankel Matrix because **it** would give high-dimensional results but we want more sparse ones.

Takens' theorem says there is a mapping between time-delay coordinates (**$h$**) and the true hidden state (**$z$**), but it doesn't tell us what that mapping is. Because the transformation is highly nonlinear, a Neural Network (specifically an Autoencoder) is the perfect tool to learn it. Singular Value Decomposition can be used first to perform a linear compression of the delay coordinates before feeding them into the neural network.

The autoencoder compresses the input space into a lower and cleaner dimensional latent space **$z$**; supposedly, we should be able to reconstruct the actual phase state from the latent space. SINDy and the autoencoder train at the same time, and we compare them using loss functions that try to minimize the overlapping errors on both models simultaneously.

**Loss functions:**
1. The decoded output **$\bar{h}$** must perfectly match the original input **$h$**.
2. The derivatives calculated by the neural network's chain rule must match derivatives predicted by the SINDy equation (minimizing the difference between **$\dot{z}$** and the SINDy **$\dot{\bar{z}}$**, as well as **$\dot{h}$** and **$\dot{\bar{h}}$**).
3. Integrating the SINDy model forward in time, the first dimension of that simulation needs to match the original 1D input data. This makes the latent variables related to the actual physical measurements.
4. The network is continually punished for adding unnecessary mathematical terms (Sparsity).

**Problems:**
Multiple different equations can validly explain the same 1D data as there are too many hyperparameters. By trying to maximize sparsity, the shape of the latent attractor can slowly drift away from the true system.

**Solution:** The final discovered equation heavily depends on how the different loss functions are weighted. If we know certain rules about the system (e.g., energy must be conserved), we can add this information into the loss function to force it to find the correct real-world physics.