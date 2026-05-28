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