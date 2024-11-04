Abstract: Proposing Kolmogorov Arnolds Networks (KANs) are better than MultiLayer Perceptrons (MLPs) - KANs make activation function learnable. KANs have no linear weights - every weight parameter is replaced by a univariate function parametrized as a spline. 
![[Pasted image 20241104192730.png]]
KANs have no linear weight matrices at all: instead, each weight parameter is replaced by a learnable 1D function parametrized as a spline.

KANs’ nodes simply sum incoming signals without applying any non-linearities. Splines are accurate for low-dimensional functions, easy to adjust locally, and able to switch between different resolutions. However, splines have a serious curse of dimensionality (COD) problem, because of their inability to exploit compositional structures.

### KANs mathematical meaning
Vladimir Arnold and Andrey Kolmogorov established that if f is a multivariate continuous function on a bounded domain, then f can be written as a finite composition of continuous functions of a single variable and the binary operation of addition. More specifically, for a smooth $f: [0, 1]^n \rightarrow \mathbb{R},$ $$f(x) = f(x_1, ..., x_n) = \sum^{2n+1}_{q=1} \Phi_q \bigg(\sum^n_{p=1} \phi_{q,p} (x_p)\bigg),$$
where $\phi_{q,p} : [0,1] \rightarrow \mathbb{R}$  and $\Phi_q : \mathbb{R} \rightarrow \mathbb{R}$. In a sense, they showed that the only true multivariate function is addition, since every other function can be written using univariate functions and sum.
Ref:
https://mlwithouttears.com/2024/05/15/a-from-scratch-implementation-of-kolmogorov-arnold-networks-kan/