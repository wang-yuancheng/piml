# Darcy Flow

### Mathematical Formulation

The steady-state Darcy flow equation is defined as:

$$
-\nabla \cdot (a(x)\nabla u(x)) = f(x)
$$

To implement this in our Physics-Informed Neural Network (PINN), we must expand the divergence operator. First, the gradient of the scalar field $u$ is:

$$
\nabla u = \begin{bmatrix} \frac{\partial u}{\partial x} \\ \frac{\partial u}{\partial y} \end{bmatrix}
$$

Multiplying by the spatially dependent viscosity (or permeability) term $a(x, y)$ gives:

$$
a \nabla u = \begin{bmatrix} a \frac{\partial u}{\partial x} \\ a \frac{\partial u}{\partial y} \end{bmatrix}
$$

Taking the divergence of this vector field yields:

$$
\nabla \cdot (a \nabla u) = \frac{\partial}{\partial x} \left( a \frac{\partial u}{\partial x} \right) + \frac{\partial}{\partial y} \left( a \frac{\partial u}{\partial y} \right)
$$

Applying the product rule to both terms:

$$
\frac{\partial}{\partial x} \left( a \frac{\partial u}{\partial x} \right) = \frac{\partial a}{\partial x} \frac{\partial u}{\partial x} + a \frac{\partial^2 u}{\partial x^2}
$$

$$
\frac{\partial}{\partial y} \left( a \frac{\partial u}{\partial y} \right) = \frac{\partial a}{\partial y} \frac{\partial u}{\partial y} + a \frac{\partial^2 u}{\partial y^2}
$$

Combining these results provides the fully expanded continuous form:

$$
\nabla \cdot (a \nabla u) = \frac{\partial a}{\partial x} \frac{\partial u}{\partial x} + \frac{\partial a}{\partial y} \frac{\partial u}{\partial y} + a \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

### Code Implementation (LHS)

When computing the left-hand side (LHS) of the PDE loss in our codebase, we express the expanded equation using network Jacobians and Hessians. Let $H$ represent the spatial derivatives of the network's prediction $u$:

$$
-\left[ a_x H_x + a_y H_y + a(H_{xx} + H_{yy}) \right]
$$

---

### Suggested Hyperparameters

*   **Learning Rate:** 
*   **Batch Size:** 
*   **Feature Chunks ($m_{chunk}$):** 
*   **Spatial Chunks ($N_{chunk}$):** 

### Best Results

*   **Final Loss:** 
*   **Relative L2 Error:** 
*   **Convergence Time:** 

### Implementation Notes

*   *Note 1*
*   *Note 2*