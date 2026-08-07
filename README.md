# Physics-Informed Machine Learning

Physics-Informed Neural Networks (PINNs) offer a way to learn continuous, differentiable functions that can be evaluated anywhere in a domain by embedding partial differential equations (PDEs) directly into the learning process. Here, we utilize a Pseudoinverse PINN (Pi-PINN) [1]. 

The problem arises when we attempt to scale PINNs to predict results across large spatial domains or to solve various related linear problems simultaneously. This scaling demands highly expressive feature spaces and massive amounts of sampling points, which easily leads to VRAM bottlenecks on standard GPUs. For example, materializing the massive Gram matrices ($\mathbf{A}^T\mathbf{A}$) required for the linear least-squares computation in Pi-PINNs easily exceeds memory limits. 

Furthermore, it is not entirely clear if iterative, memory-saving approximation methods like Sketch-and-Project work well in these scenarios. Hence, this repository explores the integration of algorithms such as Randomized Block Kaczmarz to bypass the Gram matrix materialization to solve Poisson, Helmholtz, and Darcy Flow problems.

### Model Architecture
**Random Fourier Features &rarr; MLP &rarr; Pseudoinverse.** Standard MLPs used in PINNs suffer from spectral bias, meaning they approximate low-frequency functions much more readily than high-frequency ones. This often traps the network in deceptive local minima with vanishing input gradients. To resolve this, we project the input vector $\mathbf{v} = [x, y]^T \in \mathbb{R}^2$ into a high-dimensional sinusoidal basis before the data reaches the hidden layers [2]. 

We generate a random frequency matrix $\mathbf{B} \in \mathbb{R}^{m \times 2}$, where $m$ is the number of frequency components, with entries drawn from a Gaussian distribution:

$$
b_{ij} \sim \mathcal{N}(0, \sigma^2)
$$
 The projected coordinates $\mathbf{B}\mathbf{v}$ are then passed through trigonometric functions scaled by $2\pi$:

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
\nabla^2 \Phi(x\_{\text{pde}}, y\_{\text{pde}}) \\
\lambda\_{bc} \Phi(x\_{\text{bc}}, y\_{\text{bc}})
\end{bmatrix}, \quad
b = \begin{bmatrix}
f\_{\text{pde}} \\
\lambda\_{bc} u\_{\text{bc}}
\end{bmatrix}
$$

Where $\lambda\_{bc}$ is the boundary condition scaling factor. To bypass the $\mathcal{O}(N \cdot m)$ and $\mathcal{O}(N^2)$ memory bottlenecks of materializing $A$ and computing $A A^T$, we sequentially process $A$ using a dual-chunking strategy. 

First, we partition the $N$ spatial rows into $B$ row blocks of size $N\_{\text{chunk}}$. For each row block $j \in \{1, \dots, B\}$, we further partition the $m$ features into $C$ column chunks of size $m\_{\text{chunk}}$:

$$
A^{(j)} = \begin{bmatrix}
A\_1^{(j)} & A\_2^{(j)} & \dots & A\_C^{(j)}
\end{bmatrix}, \quad
w = \begin{bmatrix}
w\_1^T & w\_2^T & \dots & w\_C^T
\end{bmatrix}^T
$$

For each spatial block $j$, we iterate over the $C$ feature chunks to dynamically accumulate the local Gram matrix $G^{(j)}$ and compute the residual vector $r^{(j)}$, which represents the error between the network's current prediction (using weights $w^{(j-1)}$) and the target block $b^{(j)}$:

$$
r^{(j)} = \left( \sum\_{c=1}^C A\_c^{(j)} w\_c^{(j-1)} \right) - b^{(j)}, \quad G^{(j)} = \sum\_{c=1}^C A\_c^{(j)} (A\_c^{(j)})^T
$$

After adding Tikhonov regularization $\lambda$, we then solve for the block's projection vector $z^{(j)} \in \mathbb{R}^{N\_{\text{chunk}}}$:

$$
z^{(j)} = (G^{(j)} + \lambda I)^{-1} r^{(j)}
$$

Finally, we update each feature chunk of the weights individually using the relaxation parameter $\alpha$:

$$
w\_{c}^{(j)} = w\_c^{(j-1)} - \alpha (A\_c^{(j)})^T z^{(j)}
$$

This executes the Randomized Block Kaczmarz update over the full spatial batch while capping peak memory strictly at $\mathcal{O}(\max(N\_{\text{chunk}}^2, N\_{\text{chunk}} m\_{\text{chunk}}))$, where if $N\_{\text{chunk}}$ and $m\_{\text{chunk}}$ are chosen as constants independent of $N$ and $m$, the relative space complexity can be considered $\mathcal{O}(1)$:

$$
w \leftarrow w - \alpha A^T (A A^T + \lambda I)^{-1} (A w - b)
$$

### Datasets

The models were evaluated against several steady-state PDEs. Detailed documentation for each dataset's generation, loss functions, and tuned hyperparameters can be found in their respective markdown files.

#### Poisson
* **Equation:** $\Delta u = f$
* **T20 Dataset:** [Poisson T20 Notes](docs/datasets/poisson_t20.md)
* **T10 Dataset:** Used to enforce incompressibility in fluid dynamics ($\nabla \cdot \mathbf{u} = 0$). Solves the Pressure Poisson Equation $\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$. [Poisson T10 Notes](docs/datasets/poisson_t10.md)

#### Poisson-Gauss
* **Equation:** $-\Delta u = f$
* **Description:** Describes a potential field behaving in the presence of a known source defined by a sum of randomized 2D Gaussians. 
* **Dataset:** [Poisson-Gauss Dataset](docs/datasets/poisson_gauss.md)

#### Helmholtz
* **Equation:** $-\Delta u - \omega^2 a(x, y)u = 0$
* **Description:** A steady-state wave propagation problem where $a(x,y)$ defines the spatial physical properties of the medium the wave is traveling through. 
* **Dataset:** [Helmholtz Dataset](docs/datasets/helmholtz.md)

#### Helmholtz Analytical
* **Equation:** $u(x, y) = \sum_{i=1}^{N_{max}} A_i \sin(f_{1,i} \pi x) \sin(f_{2,i} \pi y)$
* **Description:** Contains 10,000 unique exact solutions utilizing randomized wave numbers and continuous amplitudes to create superimposed wave modes. 
* **Dataset:** [Helmholtz Analytical Dataset](docs/datasets/helmholtz_analytical.md)

#### Darcy Flow
* **Equation:** $-\nabla(a(x)\nabla u(x)) = f(x)$
* **Description:** Models steady-state 2D fluid flow over a unit square where the viscosity term $a(x)$ is a dynamic, spatially dependent input. 
* **Dataset:** [Darcy Flow Dataset](docs/datasets/darcy_flow.md)

### Additional Information
* [Sketch-and-Project](docs/extra/sketch-and-project.md) - Details on $m$-chunking, and achieving $\mathcal{O}(1)$ space complexity.
* [Supplementary Notes](docs/extra/notes.md) - Proofs, matrix conditioning and tikreg.

### Notebooks
* `darcy_flow_direct.ipynb` - Notebook for solving entire PDEBench Darcy Flow dataset.
* `darcy_flow_smoothing_data_gen.ipynb` - Script for smoothing the a(x,y) boundary in PDEBench Darcy Flow
* `darcy_flow_solver_single.ipynb` - Notebook for solving a single sample of PDEBench Darcy Flow.
* `helmholtz_analytical_data_generation.ipynb` - Script for generating dataset using analytical Helmholz equation.
* `helmholtz_analytical_direct.ipynb` - Notebook for solving entire generated Helmholtz dataset.
* `helmholtz_debug.ipynb` - Notebook where network is trained by only data loss, then,  prediction layer is replaced with psudoinverse for testing convergence. For debugging poor results from camlab-ethz Helmholtz dataset.
* `helmholtz_direct.ipynb` - Notebook for solving entire camlab-ethz Helmholtz dataset. 
* `modal_darcy_flow_train.py` - Script for PDEBench Darcy Flow dataset on modal.com for larger feature spaces.
* `multi_pde_poisson_helmholtz_sketch_and_project.ipynb` - Notebook for solving both poisson-gauss and helmholtz analytical dataset simultaneously with sketch-and-project. 
* `multi_pde_poisson_helmholtz.ipynb` - Notebook for solving both poisson-gauss and helmholtz analytical dataset simultaneously. 
* `poisson_gauss_direct.ipynb` - Notebook for solving entire camlab-ethz Poisson-Gauss dataset. 
* `poisson_gauss_solver_sketch-and-project.ipynb` -  Notebook for solving entire camlab-ethz Poisson-Gauss dataset with sketch-and-project.
* `stencil.ipynb` - Script for building a finite difference feature table from a 2D Helmholtz solution.
* `t10_poisson_direct.ipynb` - Notebook for solving single sample T10_S10_G20 dataset.
* `t20_poisson_direct.ipynb` - Notebook for solving single sample T20_S30_G50 dataset.

### Future Work
* [Future Work Placeholder]

---

### References
[1] J. C. Wong, I. Y. C. Lai, P.-H. Chiu, C. C. Ooi, A. Gupta, and Y.-S. Ong, "Transferable Physics-Informed Representations via Closed-Form Head Adaptation," *arXiv preprint arXiv:2604.21761*, 2026.

[2] J. C. Wong, C. C. Ooi, A. Gupta, and Y.-S. Ong, "Learning in Sinusoidal Spaces With Physics-Informed Neural Networks," *IEEE Transactions on Artificial Intelligence*, vol. 5, no. 3, pp. 985–1000, Mar. 2024.