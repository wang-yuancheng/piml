### Epoch level weight decay to equivalent batch weight decay derivation

If we want the weights to decay by a factor of $(1 - \gamma_{\text{epoch}})$ over one full epoch, and an epoch consists of $N$ batches, the geometric decay factor $c$ applied at each batch must satisfy:

$$
c^N = 1 - \gamma_{\text{epoch}}
$$

$$
c = (1 - \gamma_{\text{epoch}})^{\frac{1}{N}}
$$

Since $c = 1 - \gamma_{\text{batch}}$, this formula below isolates the decay step:

$$
\gamma_{\text{batch}} = 1 - (1 - \gamma_{\text{epoch}})^{\frac{1}{N}}
$$


### Random Fourier Features
Standard MLPs suffer from spectral bias, meaning they approximate low-frequency functions much more readily than high-frequency ones (the MLP is the parameterized approximation of the true function we are trying to find.). If we are talking about reconstructing an image, the function is the image itself, the domain is the coordinates and the range is the RGB colors. In the Poisson-Gauss dataset, the true function is $u(x, y)$ (e.g., the electric potential across the 2D grid). In the current Poisson-Gauss dataset, the source term is relatively smooth, allowing the network to eventually converge. However, if we encounter sharper source terms, it is not as easy for the network to construct high-frequency representations from raw spatial coordinates alone. With the Random Fourier Features (RFF) method, instead of forcing the MLP to struggle to combine its non-linear activations to capture both low and high frequencies, we explicitly project the input coordinates into a high-dimensional basis of sine and cosine waves before they even reach the first hidden layer.

#### How RFF is Constructed

Let the input coordinate vector be $v \in \mathbb{R}^d$. For a 2D grid, $d=2$ and $v = [x, y]^T$.

We create a matrix $B \in \mathbb{R}^{m \times d}$, where $m$ is the number of frequency components we want to extract.

Then, we draw every entry in $B$ randomly from a Gaussian distribution:
$$b_{ij} \sim \mathcal{N}(0, \sigma^2)$$

This is implemented by sampling from a Standard Normal distribution $\mathcal{N}(0, 1)$ and multiplying the matrix by a scalar $\sigma$. Because $\text{Var}(\sigma Z) = \sigma^2 \text{Var}(Z)$, this $\sigma$ parameter directly determines the variance of the sampled frequencies.

Then, We multiply the frequency matrix by the coordinate vector:
$$Bv = \begin{bmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \\ \vdots & \vdots \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} \omega_{x1}x + \omega_{y1}y \\ \omega_{x2}x + \omega_{y2}y \\ \vdots \end{bmatrix}$$

We then pass these projected coordinates through sine and cosine functions and scale by $2\pi$ (treat the numbers in the $B$ matrix as ordinary frequencies ($f$) because $\omega = 2\pi f$):
$$\gamma(v) = \begin{bmatrix} \cos(2\pi Bv) \\ \sin(2\pi Bv) \end{bmatrix}$$

This produces a vector of length $2m$. Now the 2 raw coordinates is rich in features. We then feed this $\gamma(v)$ into the MLP.