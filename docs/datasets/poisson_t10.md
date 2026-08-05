# poisson_t10

Used to enforce incompressibility in fluid dynamics ($\nabla \cdot \mathbf{u} = 0$). The equation solved is $\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$, where the right-hand side is the source term quantifying divergence error. 