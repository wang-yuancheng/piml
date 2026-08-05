# Physics-Informed Machine Learning

Physics-Informed Neural Networks (PINNs) offer a way to learn continuous, differentiable functions that can be evaluated anywhere in a domain by embedding partial differential equations (PDEs) directly into the learning process. Here, we utilize a Pseudoinverse PINN (Pi-PINN) [1]. 

The problem arises when we attempt to scale PINNs to predict results across large spatial domains or to solve various related linear problems simultaneously. This scaling demands highly expressive feature spaces and massive amounts of sampling points, which easily leads to VRAM bottlenecks on standard GPUs. For example, materializing the massive Gram matrices ($\mathbf{A}^T\mathbf{A}$) required for the linear least-squares computation in Pi-PINNs easily exceeds memory limits. 

Furthermore, it is not entirely clear if iterative, memory-saving approximation methods like Sketch-and-Project work well in these scenarios. Hence, this repository explores the integration of algorithms such as Randomized Block Kaczmarz to bypass the Gram matrix materialization to solve Poisson, Helmholtz, and Darcy Flow problems.

### Model Architecture

![Architecture of Pi-PINN](docs/assets/overview/architecture_placeholder.png)
*(Image source: Transferable Physics-Informed Representations via Closed-Form Head Adaptation)*

**Random Fourier Features.** Standard MLPs used in PINNs suffer from spectral bias, meaning they approximate low-frequency functions much more readily than high-frequency ones. This often traps the network in deceptive local minima with vanishing input gradients. To resolve this, we project the input vector $\mathbf{v} = [x, y]^T \in \mathbb{R}^2$ into a high-dimensional sinusoidal basis before the data reaches the hidden layers [2]. 

We generate a random frequency matrix $\mathbf{B} \in \mathbb{R}^{m \times 2}$, where $m$ is the number of frequency components, with entries drawn from a Gaussian distribution [2]:

$$
b_{ij} \sim \mathcal{N}(0, \sigma^2)
$$

The variance $\sigma^2$ controls the spread of the sampled frequencies [2]. The projected coordinates $\mathbf{B}\mathbf{v}$ are then passed through trigonometric functions scaled by $2\pi$ [2]:

$$
\gamma(\mathbf{v}) = \begin{bmatrix} 
\cos(2\pi \mathbf{B}\mathbf{v}) \\ 
\sin(2\pi \mathbf{B}\mathbf{v}) 
\end{bmatrix}
$$

This enriches the raw 2D spatial inputs into a $2m$-dimensional feature vector $\gamma(\mathbf{v})$ that is fed directly into the MLP. This mapping natively aligns with the analytical solutions of many dynamical systems, drastically increasing input gradient variability and ensuring the network can capture high-frequency physical patterns [2].

### Memory-Efficient Sketch-and-Project

Let $w \in \mathbb{R}^m$ be the network weights and $\Phi(x, y) \in \mathbb{R}^m$ be the concatenated feature map. For a sampled spatial batch of $N$ points, the target vector $b \in \mathbb{R}^N$ and the sketched matrix $A \in \mathbb{R}^{N \times m}$ are defined as:

$$
A = \begin{bmatrix} 
\nabla^2 \Phi(x_{\text{pde}}, y_{\text{pde}}) \\ 
\lambda_{bc} \Phi(x_{\text{bc}}, y_{\text{bc}}) 
\end{bmatrix}, \quad 
b = \begin{bmatrix} 
f_{\text{pde}} \\ 
\lambda_{bc} u_{\text{bc}} 
\end{bmatrix}
$$

Where $\lambda_{bc}$ is the boundary condition scaling factor. To bypass the $\mathcal{O}(m^2)$ memory bottleneck of materializing $A$, we partition the $m$ features into $C$ chunks such that:

$$
A = \begin{bmatrix} 
A_1 & A_2 & \dots & A_C 
\end{bmatrix}, \quad 
w = \begin{bmatrix} 
w_1^T & w_2^T & \dots & w_C^T 
\end{bmatrix}^T
$$

First, we accumulate the residual vector $p$ and the Gram matrix $G$ by iterating over the $C$ chunks:

$$
p = \sum_{c=1}^C A_c w_c, \quad G = \sum_{c=1}^C A_c A_c^T
$$

After adding Tikhonov regularization $\lambda$, we then solve for the projection vector $z \in \mathbb{R}^N$:

$$
z = (G + \lambda I)^{-1} (p - b)
$$

Finally, we update each feature chunk individually using the relaxation parameter $\alpha$:

$$
w_{c, \text{new}} = w_c - \alpha A_c^T z
$$

This executes the exact Randomized Block Kaczmarz update while capping peak memory strictly at $\mathcal{O}(\max(N^2, N \cdot m_{\text{chunk}}))$:

$$
w_{\text{new}} = w - \alpha A^T (A A^T + \lambda I)^{-1} (A w - b)
$$

### Datasets

The models were evaluated against several steady-state PDEs. Detailed documentation for each dataset's generation, loss functions, and tuned hyperparameters can be found in their respective markdown files.

#### Poisson
* **Equation:** $\Delta u = f$
* **T20 Dataset:** [Poisson T20 Notes](datasets/poisson_t20.md)
* **T10 Dataset:** Used to enforce incompressibility in fluid dynamics ($\nabla \cdot \mathbf{u} = 0$). Solves the Pressure Poisson Equation $\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$. [Poisson T10 Notes](datasets/poisson_t10.md)

#### Poisson-Gauss
* **Equation:** $-\Delta u = f$
* **Description:** Describes a potential field behaving in the presence of a known source defined by a sum of randomized 2D Gaussians. 
* **Dataset:** [Poisson-Gauss Dataset](datasets/poisson_gauss.md)

#### Helmholtz
* **Equation:** $-\Delta u - \omega^2 a(x, y)u = 0$
* **Description:** A steady-state wave propagation problem where $a(x,y)$ defines the spatial physical properties of the medium the wave is traveling through. 
* **Dataset:** [Helmholtz Dataset](datasets/helmholtz.md)

#### Helmholtz Analytical
* **Equation:** $u(x, y) = \sum_{i=1}^{N_{max}} A_i \sin(f_{1,i} \pi x) \sin(f_{2,i} \pi y)$
* **Description:** Contains 10,000 unique exact solutions utilizing randomized wave numbers and continuous amplitudes to create superimposed wave modes. 
* **Dataset:** [Helmholtz Analytical Dataset](datasets/helmholtz_analytical.md)

#### Darcy Flow
* **Equation:** $-\nabla(a(x)\nabla u(x)) = f(x)$
* **Description:** Models steady-state 2D fluid flow over a unit square where the viscosity term $a(x)$ is a dynamic, spatially dependent input. 
* **Dataset:** [Darcy Flow Dataset](datasets/darcy_flow.md)

### Additional Information
* [Sketch-and-Project](extra/sketch-and-project.md) - Details on $m$-chunking, and achieving $\mathcal{O}(1)$ space complexity.
* [Tips and Tricks](extra/tips_and_tricks.md) - Guidance on using pure data loss for capacity checking, matrix conditioning, and debugging PINN amplitudes.

### Notebooks
* `[Notebook 1]` - Description placeholder
* `[Notebook 2]` - Description placeholder
* `[Notebook 3]` - Description placeholder

### Future Work
* [Future Work Placeholder]

---

### References
[1] J. C. Wong, I. Y. C. Lai, P.-H. Chiu, C. C. Ooi, A. Gupta, and Y.-S. Ong, "Transferable Physics-Informed Representations via Closed-Form Head Adaptation," *arXiv preprint arXiv:2604.21761*, 2026.

[2] J. C. Wong, C. C. Ooi, A. Gupta, and Y.-S. Ong, "Learning in Sinusoidal Spaces With Physics-Informed Neural Networks," *IEEE Transactions on Artificial Intelligence*, vol. 5, no. 3, pp. 985–1000, Mar. 2024.