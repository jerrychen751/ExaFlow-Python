# Numerics

Finite difference operators and time integration.

## Array conventions

The solver carries one `FlowState` per time level (`src/exaflow/fields.py`). It holds two arrays:

| Field | Shape | Meaning |
|-------|-------|---------|
| `velocity` | `(dimension, *padded_shape)` | `velocity[a]` is the velocity along axis `a` |
| `pressure` | `(*padded_shape)` | pressure |

Axis 0 is x, axis 1 is y, axis 2 is z. So `velocity[0][3, 7, 2]` is the x-velocity at grid point (x=3, y=7, z=2), and `velocity[1]` is what older code called `v`.

Velocity components share one array with a leading component axis. An operator loops over that axis instead of repeating itself once per component, which is why `convection.py` and `diffusion.py` each hold a single stencil.

### Ghost layers

Every array a rank works on is padded by `grid.num_ghost_layers` at each end of every axis:

```
axis length = ng + n_local + ng
```

Never write these index rules by hand. `Subdomain` owns them:

| Call | Gives |
|------|-------|
| `subdomain.padded_shape` | the shape to allocate |
| `subdomain.interior()` | the real cells, ghost layers stripped |
| `subdomain.shifted_interior(axis, offset)` | each real cell's neighbor along `axis` |
| `subdomain.is_on_face(face)` | whether this rank owns that face of the **global** domain |

`is_on_face` is the one to be careful with. A one-sided stencil or a boundary condition is correct only where it returns `True`. Everywhere else the face is an internal partition face and the ghost layer already holds the neighboring rank's data. Applying a domain-face rule there makes the answer depend on how many ranks the run used.

### Dimensions

Operators are written for any of 1D, 2D and 3D. They read `grid.dimension` and loop over axes, so there is no per-dimension branch and no `_3d` suffix.

## Writing an operator

An operator accumulates one term of the right-hand side:

```python
class Operator(Protocol):
    def accumulate(self, state: FlowState, rate: FlowState) -> None: ...
```

Rules:

- **Add, never assign.** Another term may already have written this cell.
- **Write the rate, not the increment.** Leave the time step to the integrator, which multiplies by `dt`.
- **Cover every real cell with the same stencil.** Do not special-case faces. A face slab whose transverse span is narrower than the interior leaves the block edges unwritten, which is exactly the fault that made this solver rank-dependent.
- **Assume the ghost layers are current.** `SpatialOperator` completes the exchange and refreshes the boundary conditions before any operator runs.

Register it in `build_operators` in `operators.py`.

## Modules

- `operators.py` - the `Operator` protocol, `build_operators`, and `SpatialOperator`, which sequences the ghost exchange, the boundary refresh and the terms, and runs the pressure projection
- `convection.py` - first-order upwind `(u dot grad) u`
- `diffusion.py` - second-order central `nu * laplacian(u)`
- `time_step.py` - `TimeIntegrator`, Runge-Kutta orders 1 to 3
- `pressure_poisson.py` - Chorin projection, solved by a stencil conjugate gradient that runs on every rank; `SpatialOperator.project` runs it after every Runge-Kutta stage when `include_pressure` is set
