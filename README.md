# SparseDynamicSystem
SparseDynamicSystem computes region of attraction (ROA), maximum positively invariant set (MPI), global attractor (GA) for polynomial dynamic systems based on the sparsity adapted moment-SOS hierarchies. To use SparseDynamicSystem in Julia, run
```Julia
pkg> add https://github.com/wangjie212/SparseDynamicSystem
 ```

## Dependencies
- MOSEK
- JuMP

## Usage
The following script computes the MPI set for the Van-der-Pol oscillator on [-1.1, 1.1]^2:

```Julia
using SparseDynamicSystem
using DynamicPolynomials
@polyvar x[1:2]
f = [2*x[2], -0.8*x[1] - 10*(x[1]^2-0.21)*x[2]]
g = [1.1^2-x[1]^2, 1.1^2-x[2]^2]
d = 8 # the relaxation order
opt,w = MPI_first(f, g, x, d, -1.1*ones(2), 1.1*ones(2), β=1, TS="block")
```
