# Glossary

This glossary defines the main topology-optimization, FEM, and project-specific terms used in TopoptComec.

## Units and Conventions

TopoptComec is unit-agnostic: it never converts units. You must supply a
*consistent* unit system and interpret results in the same system.

- **Element size**: every element is a unit square (2D) or unit cube (3D).
  One grid step = one length unit. Choose your length unit accordingly (an
  exported STL/3MF uses one grid step = 1 mm by convention).
- **Young's modulus `E`**: any consistent stiffness unit. Because SIMP
  results are scale-invariant in `E`, only the *ratios* between materials
  (and between `E` and the spring stiffnesses, see below) affect the design.
- **Force magnitude (`finorm`, `fonorm`)**: consistent force units.
- **Displacements**: length units (grid steps) in the chosen system.
- **Direction glyphs**: `X:→` (+x), `X:←` (−x), `Y:↓` (+y, screen-down),
  `Y:↑` (−y), `Z:<` (+z), `Z:>` (−z). Internally these are translated to a
  signed axis by `topoptcomec/core/preset_format.py`; the numerical core
  never sees glyphs.

### Artificial springs (the `finorm`/`fonorm` double role)

Following the classic compliant-mechanism formulation (Sigmund, 1997), an
artificial spring is attached to every loaded DOF. **The force magnitude is
reused as the spring stiffness**: an input force `finorm = 0.01` also adds a
spring `k = 0.01` at the input node, and each output entry adds a spring
`k = fonorm` at the output node. Output entries define equally weighted adjoint
directions; `fonorm` controls their spring stiffness, not their objective
weight. Larger output spring stiffness models a stiffer workpiece and yields
designs that trade output displacement for output force. The typed API
(`topoptcomec/core/model.py`, `Load.spring`) lets library users set the spring
independently of the force.

### Objectives

- **Rigid case** (no output force): minimize total compliance `C = fᵀu` under
  all input forces acting simultaneously. Its sensitivity uses material strain
  energy; constant artificial springs remain part of the equilibrium system.
- **Compliant case**: maximize the signed output displacement under the
  simultaneous input loads, averaged over output ports. Positive objective
  values mean the outputs move in the requested directions.

## Optimization Terms

### SIMP

Solid Isotropic Material with Penalization. A standard topology optimization method where element stiffness is interpolated from a density value and penalized to discourage intermediate densities.

In this project, SIMP is the main optimization formulation used by `topoptcomec/core/optimizers.py`.

### OC

Optimality Criteria. An iterative update rule used to change element densities while respecting a volume constraint. In this repository, `_oc(...)` applies move limits and a bisection search on the Lagrange multiplier.

### Density Field

The array of per-element design variables. Values are usually between `0` and `1`:

- near `0`: void / removed material
- near `1`: solid material
- in between: intermediate material, usually undesirable in the final design

### `x`

The raw design variable array before all physical projections and region overrides are applied.

### `xPhys`

The physical density field used by FEM solves and export. It usually means the current design after filtering and region application. In multi-material mode it may be a 2D array shaped like `(n_materials, n_elements)`.

### Volume Fraction (`volfrac`)

The target fraction of the design domain that may remain filled with material. Lower values produce lighter structures but can make the optimization problem harder.

### Penalization (`penal`)

The exponent applied in the SIMP stiffness interpolation. Higher values penalize gray regions more aggressively.

### Filter Radius (`filter_radius_min`)

The neighborhood size used for density or sensitivity filtering. It reduces numerical artifacts such as checkerboarding and mesh dependency.

### Max Change (`max_change`)

The maximum density change allowed per optimization iteration by the OC update.

### Eta (`eta`)

The damping exponent in the OC update rule. It affects how aggressively the design changes between iterations.

## FEM Terms

### FEM

Finite Element Method. The numerical method used here to approximate elastic deformation of the structure.

### Element

A discrete cell of the design grid. In this project the topology optimization variables are stored per element, not per node.

### Node

A mesh point where displacements are defined and where loads or supports may be applied.

### DOF

Degree of Freedom. A single displacement component associated with a node.

- 2D: each node has `X` and `Y` DOFs
- 3D: each node has `X`, `Y`, and `Z` DOFs

### Stiffness Matrix (`K`)

The sparse linear system assembled from the element stiffness matrix and the current material distribution. Solving `K u = f` yields nodal displacements.

### `KE`

The element stiffness matrix used as the local building block for the global stiffness matrix.

### Boundary Conditions

The constraints and loads applied to the model:

- supports fix some DOFs
- forces load some DOFs

### Plane 2D vs 3D

The code switches behavior based on `nelz`:

- `nelz == 0`: 2D mode
- `nelz > 0`: 3D mode

This convention appears across the UI, FEM layer, plotting, and export code.

## Mechanism Terms

### Rigid Mechanism Case

In this repository, a case with input forces and supports but no output force. The objective behaves like compliance minimization: keep the structure stiff.

### Compliant Mechanism Case

A case with both input and output forces. The objective is based on transfer behavior, not only stiffness. The solver tries to create elastic motion transmission through deformation.

### Input Force

An actuating force defined by:

- position
- direction
- magnitude

Input forces are stored under `Forces["fi*"]`.

### Output Force

A target response location and direction used to evaluate compliant mechanism behavior. Output forces are stored under `Forces["fo*"]`.

### Efficiency

A heuristic post-analysis metric computed in `topoptcomec/core/analyzers.py`. It is based on displacement behavior and is used as a quick quality check, not as a rigorous engineering certification.

## Project-Specific Terms

### Region

A geometric override applied to the density field before or during optimization.

Typical uses:

- force a zone to stay solid
- force a zone to stay void

### `rshape`, `rstate`, `rradius`

Region fields in the parameter schema:

- `rshape`: shape selector, such as circle or square
- `rstate`: whether the region is forced to `Void` or `Filled`
- `rradius`: size control

Coverage convention: a region spans element indices `[int(c - r), int(c + r))`
per axis (matching the GUI preview); circular/spherical regions additionally
require the element to lie within distance `r` of the center.

### Support Radius (`sr`)

An optional radius around a support point. When it is non-zero, all nodes inside that neighborhood are constrained according to the selected support dimensions.

### Thresholding

Turning a gray density field into a binary one, usually with a cutoff of `0.5`. The CLI exposes this through `-t`.

### Checkerboarding

A non-physical alternating solid/void pattern caused by discretization artifacts. The project includes a heuristic detector for this in post-analysis.

### Watertight

Used here as a practical connectedness check on the thresholded structure. The analyzer verifies whether the binarized design forms a single connected component.

### Warping / Displacement Visualization

Post-processing that distorts the displayed geometry using displacement data so the user can inspect how a mechanism moves.

### Preset

A named parameter set stored in `presets.json`. Presets are the main way to reproduce designs across GUI and CLI runs.
