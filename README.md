# ExaFlow

A Python solver for the incompressible Navier-Stokes equations in 1D, 2D, and 3D with MPI parallelization.

It simulates how fluids (like air or water) move through space over time. The Navier-Stokes equations describe fluid motion through three physical effects:

- **Convection** - fluid carries momentum along with its flow
- **Diffusion** - viscosity smooths out velocity differences (think honey vs. water)
- **Pressure** - enforces that the fluid can't compress or expand (incompressibility)

The solver discretizes a physical domain into a grid, then marches forward in time using explicit Runge-Kutta integration (orders 1-3).

## Quick start

### 1. Install

The project uses [uv](https://docs.astral.sh/uv/). One command builds the whole environment:

```bash
uv sync
```

This creates `.venv`, installs Python 3.13 (the version in `.python-version`), installs every dependency, and installs ExaFlow itself in editable mode. On macOS and Linux nothing else is needed. Open MPI arrives as a Python wheel, so you do not install MPI through Homebrew or conda. `uv.lock` pins every exact version.

Open MPI has no Windows build, so on Windows `uv sync` leaves it out and you install Microsoft MPI once, before the first run:

```powershell
winget install Microsoft.msmpi
```

Open a new terminal afterwards, because the installer puts `mpiexec` on PATH and sets `MSMPI_BIN`, which is where `mpi4py` looks for the library. The commands in this README are written for a Unix shell. In PowerShell, set an environment variable on its own line first, as in `$env:EXAFLOW_OUTPUT_ROOT = "C:\exaflow-runs"`, in place of the `NAME=value` prefix, and use a Windows path in place of `/tmp`.

Run any command in that environment with `uv run`:

```bash
uv run python -c "from mpi4py import MPI; print(MPI.Get_library_version().split(',')[0])"
```

### 2. Run your first simulation

```bash
uv run exaflow run --case examples/input_template.xml
```

`exaflow run` is the standard entry point. It reads one case file and runs on the MPI communicator that started it. The shipped template is a 100x100x50 case for 1000 steps, so lower `nt` for a short check. To use four processes:

```bash
uv run mpiexec -n 4 exaflow run --case examples/input_template.xml
```

### 3. Find the output

Every run writes into its own folder under `~/Documents/ExaFlow`:

```
~/Documents/ExaFlow/
    2026-08-24_200239_input_template/
        Original_Total.csv
        Final_Total.csv
        Final_0.csv
        Final_1.csv
```

`*_Total.csv` holds the full domain, joined on rank 0. `*_<n>.csv` holds the part that rank `n` owned. A folder name is the start time plus a label, so a new run never overwrites an old one.

Every field file states where the run had reached. A CSV starts with a line such as `# step=400 time=0.8 dt=0.002` before the column header, and a `.vtr` carries the same three values as field data, where `TimeValue` is the name ParaView reads as the time of a file. The label of a file is the completed step count, so `400_Total.csv` holds the state after 400 steps.

One run writes one format, so a run folder holds `.csv` files or `.vtr` files and never both. `<Format>` in `<OutputProperties>` selects it, `CSV` or `VTK`, and the shipped template selects CSV:

```xml
<OutputProperties>
  <Format>CSV</Format>
  <WriteTotalFrequency>100</WriteTotalFrequency>
  <WritePartialFrequency>250</WritePartialFrequency>
  <WriteCheckpointFrequency>-1</WriteCheckpointFrequency>
</OutputProperties>
```

`WriteTotalFrequency` is the interval in time steps of the whole-domain file, and `WritePartialFrequency` the interval of the per-rank files, which only CSV writes. Every case file carries both elements, so a VTK case sets `<WritePartialFrequency>` to -1 and is refused otherwise. An interval of -1 asks for no writes during the march; the first and last state are written whatever the interval. A VTK run writes `Original_Total.vtr`, `Final_Total.vtr` and one `<step>_Total.vtr` per interval, which is what the GUI viewer and ParaView read.

Set `EXAFLOW_OUTPUT_ROOT` to write the run folders somewhere else:

```bash
EXAFLOW_OUTPUT_ROOT=/tmp/exaflow-runs uv run exaflow run --case examples/input_template.xml
```

Set `EXAFLOW_STREAM_PORT` to a TCP port on localhost, and rank 0 also sends every state it writes to that port as a pickled PyVista rectilinear grid, one length-prefixed message per connection. The GUI sets it for the runs it starts. A run whose port nobody listens on prints one line per failed send and marches on.

### 4. Open the GUI

```bash
uv run python run_gui.py
```

The window has a control column on the left and a 3D viewer on the right. Pick a case in the **Preset** box or edit one through **Simulation Params…**, set the number of MPI processes, and press Run. The GUI starts the standard `exaflow run --case` command through `mpiexec`, with `EXAFLOW_STREAM_PORT` set to the port its server listens on, so rank 0 sends every state it writes straight to the viewer over TCP on localhost. The viewer shows the newest state it has received and skips any that arrived while it was still drawing an earlier one, so a run that outpaces the display never queues up behind it. The viewer also loads the newest result file from the output root, which is how a run started outside the GUI appears. The case a new window opens on writes VTK, which carries the physical extent the slice control reports in metres; the Output & Misc tab switches it to CSV.

The viewer colors a result by speed, the magnitude of the velocity. **Color by** switches it to pressure, which stays at zero until the pressure projection is wired into the time loop. The ends of the color bar are rounded outward, and through the frames of one run they only widen, so the numbers beside the bar hold still while a run streams. A frame whose step count is lower than the last one starts a new range, and so does Run or Resume.

The **Slice** row cuts a 3D result on one axis and shows that plane by itself. Pick the axis, move the position slider, and the camera faces the plane and stops rotating. The position label states the unit: metres for a `.vtr` file, and cells for a CSV file, which carries the indices and no physical extent. A 1D or 2D result is already a cross-section, so the viewer shows it flat and the control stays disabled.

## Where to look

```
src/exaflow/
    config/ # frozen value types; no numpy, no MPI, no Qt
        case.py # Case: the whole problem definition, plus SolverOptions
        fluid.py # Fluid(rho, nu)
        grid.py # Grid(shape, extent, num_ghost_layers) -> spacing, dimension
        boundaries.py # Face, FaceCondition, Boundaries
        boundary_conditions.py # BoundaryCondition and the strict parser for its names
        initial_conditions.py # UniformValue, StepValue as value types
        time_control.py # TimeControl, OutputControl, OutputFormat
        case_xml.py # read_case and write_case: one field map, both directions
    fields.py # FlowState: velocity (dimension, *padded), pressure (*padded)
    run.py # run_case: the typed application entry point
    cli.py # parses exaflow run and calls run_case
    mpi/
        process_grid.py # ProcessGrid: rank arrangement and the factorization
        subdomain.py # Subdomain: the block one rank owns and every index rule
        ghost_exchange.py # non-blocking exchange of the outermost real layer
        gather.py # assemble the full domain on rank 0, and hand each rank its block back
    numerics/
        operators.py # Operator protocol, build_operators, SpatialOperator
        convection.py # first-order upwind (u dot grad) u
        diffusion.py # second-order central nu * laplacian(u)
        time_step.py # TimeIntegrator, Runge-Kutta orders 1-3
        pressure_poisson.py # Chorin projection (single rank, not yet wired in)
        README.md # array shapes, ghost layer rules, how to write an operator
    boundary_application.py # writes boundary values into the ghost layers
    session.py # SimulationSession: one run in progress, and the one time loop
    io/
        writers.py # Writer protocol, TotalCsvWriter, RankCsvWriter, VtkWriter
        csv.py # CSV formatting and the atomic write
        checkpoint.py # the restart format: write, read and spread one over the ranks
        storage.py # picks the run folder under ~/Documents/ExaFlow
    gui/
        main_window.py # wires the panels, the dialogs and the viewer together
        simulation_runner.py # the child process one run happens in
        result_watcher.py # reports the newest result file in the output root
        slice_controller.py # the cross-section row and the viewer state behind it
        viewer.py # PyVista 3D view
        sim_parameters_dialog.py # edits one Case and returns it
        presets/ # the case files the Preset box lists
        streaming/ # sends fields from a running job to the viewer
examples/ # the typed Python API example and the input XML template
Simulations/ # scaffold script for a new simulation directory
tests/ # pytest suite, one module per source module, plus conftest.py and the two mpiexec helper scripts
packaging/ # what PyInstaller reads: the spec, the frozen entry point and the icon
scripts/ # the commands you run, one file per task
    update_app.py # builds ExaFlow.app and installs it
```

Start with `src/exaflow/cli.py`, then follow its `run_case` call into `src/exaflow/run.py` and `SimulationSession`. The CLI parses process input, `run_case` starts one typed case, and the session owns the fields, the position of the run and the time loop.

Read `src/exaflow/numerics/README.md` before you touch a stencil. It states the array shapes, the ghost layer index rules, and the rules an operator has to follow.

## Key concepts

### Grid and ghost layers

The domain is a structured grid with uniform spacing. Each dimension has `n` real cells plus extra **ghost layers** padded around the edges. Ghost cells allow finite difference stencils to work at boundaries without going out of bounds.

```
ghost | real cells              | ghost
  [g]   [0]  [1]  [2]  ...  [n-1]   [g]
```

Ghost cells are filled with either MPI neighbor data or boundary condition values.

### Boundary conditions

Each of the 6 domain faces (left, right, top, bottom, front, back) can be independently set to:

| Type | What it does |
|------|-------------|
| `NO_SLIP` | Velocity = 0 at the wall (fluid sticks) |
| `SLIP` | Normal velocity = 0, tangential is free |
| `INFLOW` | Prescribed velocity entering the domain |
| `OUTFLOW` | Prescribed pressure, velocity extrapolated |
| `PERIODIC` | Wraps around (must be set on both opposing faces) |

### Time stepping

The timestep `dt` is automatically chosen as the smaller of:

- **Advection CFL**: `dt_adv = CFL * min(dx, dy, dz) / max_velocity`
- **Diffusion stability**: `dt_diff = 0.5 / (nu * (1/dx^2 + 1/dy^2 + 1/dz^2))`

### MPI parallelization

The domain is split across MPI processes in a process grid. Each process works on its own block and exchanges ghost layer data with its neighbors using non-blocking sends and receives.

`Subdomain` owns every index rule that follows from that split: the block bounds, the padded shape, the interior slice, the neighbor ranks, and `is_on_face`, which reports whether this rank owns a face of the **global** domain. A boundary condition or a one-sided stencil is correct only where `is_on_face` is true; elsewhere the ghost layer already holds the neighbor's data.

**The answer does not depend on the rank count.** A run at 1, 2, 4 or 8 ranks writes byte-identical output, and so does a run stopped at one rank count and continued at another. `tests/test_session.py` checks both, and they are the first properties to look at after touching an operator or the exchange.

## Running simulations

### From the command line

Use one process:

```bash
uv run exaflow run --case examples/input_template.xml
```

Use four MPI processes:

```bash
uv run mpiexec -n 4 exaflow run --case examples/input_template.xml
```

The CLI reads the XML file, creates one shared run directory, and calls the typed `run_case` core. `--label` replaces the file stem in the run directory name. `--input-xml` remains a compatibility name for `--case`.

How far the run marches lives in `<GridProperties>` beside `<nt>` and `<CFL>`:

```xml
<EndTime>-1</EndTime>
<AdaptiveTimeStep>False</AdaptiveTimeStep>
```

`<EndTime>` marches to a simulated time in seconds instead of a step count, and `<nt>` is then the cap that stops a run whose step size shrinks faster than the time that is left. The last step is cut to land on the end time itself. `-1` asks for no end time. `<AdaptiveTimeStep>` chooses the step size again from the current state before every step, instead of once from the initial state; it costs one pass over the velocity arrays and one reduction across the ranks per step, which is about 2 percent of an Euler step on the shipped template.

### Stopping and continuing a run

A run saves a restart file when the case asks for one:

```xml
<WriteCheckpointFrequency>200</WriteCheckpointFrequency>
```

That writes `Checkpoint_200.npz`, `Checkpoint_400.npz` and so on into the run folder, and one `Checkpoint_Final.npz` beside the last field file. Each one holds the whole domain, the completed step count, the simulated time, the step size and the case that produced them. An interval of -1 writes none at all.

Continue from one with `--resume`:

```bash
uv run mpiexec -n 8 exaflow run --resume ~/Documents/ExaFlow/2026-08-31_120000_input_template/Checkpoint_400.npz
```

The command names no case file, because the checkpoint carries its own. Nothing in the file records a decomposition, so a run checkpointed at two ranks continues at one or at eight. `nt` is the step budget of the whole run counted from time zero, so a file at step 400 of 1000 has 600 steps left, and a finished run continues only under a case that raises the budget:

```bash
uv run mpiexec -n 4 exaflow run --resume Checkpoint_1000.npz --case longer_run.xml
```

That case replaces the stored one and has to describe the same grid shape. A continued run writes its own folder, carries on the step labels from where it started, and writes the restored state under the label `Resumed` rather than `Original`. The GUI does the same through its **Resume…** button.

### From the Python API

Build a `Case` and give it to `run_case`:

```python
from mpi4py import MPI
from exaflow.config import Case, Fluid, Grid, TimeControl
from exaflow.io.storage import create_run_directory
from exaflow.run import run_case

comm = MPI.COMM_WORLD
case = Case(
    fluid=Fluid(rho=1.225, nu=0.3),
    grid=Grid(shape=(50, 50, 50), extent=(1.0, 1.0, 1.0), num_ghost_layers=1),
    time=TimeControl(num_steps=50, cfl=0.25, integration_order=1),
)
run_case(case, comm, output_directory=create_run_directory("my_case", comm))
```

A `Case` is frozen, so it can be built once and compared. It holds no communicator and no output directory. Those values belong to a run, and `run_case` passes them to `SimulationSession`.

Build the session yourself to stop between steps. It holds `state`, `step_index`, `current_time` and `dt`, `advance_one_step` moves all four, `is_complete` reports whether the run has reached its target, and `save_checkpoint` writes a file another process can continue from.

`create_run_directory` is a collective call. Rank 0 picks the folder name and broadcasts it, so every rank has to call it. A call on rank 0 alone makes the other ranks wait forever.

### Presets

`src/exaflow/gui/presets/` holds three case files. The **Preset** box in the GUI lists every `.xml` in that folder, so a new file there appears in the list with no code change. The folder ships inside the package and inside `ExaFlow.app`.

- `moving_patch_2d.xml` puts a rectangular patch of fast fluid in a slower periodic 2D flow and marches 6 s. The patch steepens into a sharp front on its leading side, stretches into a ramp behind, and crosses the periodic faces more than once.
- `moving_block_3d.xml` is the same setup in a periodic 3D box, marched 2 s, with a block that moves along x alone. The block sits in the corner the default camera faces, because the viewer draws the outer surface of a 3D result. Use **Slice** to look inside.
- `channel_inflow_3d.xml` marches 6 s of flow through a channel with an inflow face, an outflow face and four no-slip walls. The channel starts with a swirl, and the inflow enters at an angle, so once the swirl has washed out the flow leans toward two walls and leaves slow fluid along the other two.

On four processes the 2D case takes about 10 seconds in the GUI and each 3D case about 16, and each streams 40 to 50 frames to the viewer. Every frame is also a `.vtr` file, so one run writes 130 to 360 MB under the output root.

The same file runs from the command line:

```bash
uv run mpiexec -n 4 exaflow run --case src/exaflow/gui/presets/channel_inflow_3d.xml
```

### As a new simulation directory

```bash
uv run python Simulations/create_simulation_directories.py Simulations/my_run
cd Simulations/my_run
uv run mpiexec -n 4 exaflow run --case build/in/input.xml
```

The scaffold writes `build/in/input.xml` and a metadata file with the standard command. It creates no Python driver.

## Developing

The editable install means an edit under `src/` takes effect on the next run. You do not reinstall, and you do not set `PYTHONPATH`.

### Add a dependency

```bash
uv add <package>
```

This writes the package into `pyproject.toml`, resolves it into `uv.lock`, and installs it. The project keeps one flat dependency list with no optional extras, and a `dev` group for mypy, pytest, the stubs and PyInstaller, which `uv sync` installs by default. Add a development tool with `uv add --dev <package>`.

### Check types

```bash
uv run mypy src tests
uv run mypy examples packaging run_gui.py simulations
```

Both report no issues today. Keep it that way. When a new error appears, fix the cause rather than silencing it: correct a loose annotation first, use `typing.cast` when a dimension branch already fixed the type, and raise on a value that arrives from outside. Keep `# type: ignore[<code>]` for a third-party stub that is wrong or missing, always with the exact code the checker prints. `src/exaflow/gui/viewer.py` shows that last case, because pyvista and vtk ship no usable stubs for those calls.

### Verify a change

```bash
uv run pytest
uv run pytest -m "not mpi and not gui"
uv run pytest -m mpi
uv run pytest tests/test_numerics.py
```

The first line runs everything. The second leaves out the launcher and the visualization stack, the third keeps only the runs that start `mpiexec`, and the fourth runs one module.

374 tests run in about five seconds. One test module covers one source module. `tests/conftest.py` holds what they share: `build_case` and `build_subdomain` build a case and one rank's block, `template_case_path` gives the absolute path of `examples/input_template.xml`, and `run_under_mpiexec` starts a helper script at a given rank count.

Three markers select a subset:

| Marker | Selects |
|---|---|
| `mpi` | a test that starts `mpiexec`. Every one is skipped when no launcher is on PATH |
| `gui` | a test that imports vtk, pyvista or PySide6. None of them needs a display |
| `slow` | a test that starts a subprocess |

`tests/_run_case.py` and `tests/_run_periodic_exchange.py` are the two scripts `mpiexec` starts. pytest collects `test_*.py` only, so it leaves them alone. Each one reports a wrong answer through its exit status.

`pyproject.toml` turns every warning into an error and rejects an unregistered marker, so a new warning from this repository fails the run instead of reaching the log unread. One test carries an ignore of its own, because vtk sets the shape of a numpy array and numpy 2.5 deprecated that.

What the suite holds:

| Module | What it checks |
|---|---|
| `test_config.py` | every value type, and the error each one raises for a setting the solver cannot use |
| `test_case_xml.py` | the round trip at 1D, 2D and 3D, and the message for each malformed element |
| `test_process_grid.py`, `test_subdomain.py` | the factor search, the block bounds, the face tests and the neighbor ranks |
| `test_fields.py` | the state arithmetic each Runge-Kutta stage needs, and a step box that lands the same at any rank count |
| `test_boundary_application.py` | what each of the five conditions writes, and what it leaves alone |
| `test_numerics.py` | the operators against analytic answers, the time step limits, and the convergence order of each scheme |
| `test_ghost_exchange.py` | the serial periodic copy, and the periodic wrap under `mpiexec` |
| `test_pressure_poisson.py` | the operator symmetry, and that the residual divergence falls at second order |
| `test_io.py`, `test_csv_loader.py` | the CSV format, the atomic write, the run folder, and the loader the GUI reads with |
| `test_session.py` | the position of a run, the end time, the output schedule, and the rank-count independence of a whole run and of a restarted one |
| `test_checkpoint.py` | the restart format, and that a stopped and continued run reaches the state the whole run reaches |
| `test_run.py`, `test_cli.py` | the typed run core and the standard console entry point under one or more MPI processes |
| `test_main_window.py`, `test_simulation_runner.py`, `test_simulation_scaffold.py` | the GUI window, the GUI child command and the case-only simulation scaffold |
| `test_streaming.py` | the length-prefixed stream between a run and the viewer |

For a numerical change, also compare output against a run made before it:

```bash
EXAFLOW_OUTPUT_ROOT=/tmp/before uv run mpiexec -n 4 exaflow run --case examples/input_template.xml
# make the change
EXAFLOW_OUTPUT_ROOT=/tmp/after uv run mpiexec -n 4 exaflow run --case examples/input_template.xml
cmp /tmp/before/*/Final_Total.csv /tmp/after/*/Final_Total.csv
```

A refactor that should not change the numbers gives byte-identical files. Run it at more than one rank count, because a decomposition fault only shows up in parallel.

## The desktop app

`scripts/update_app.py` builds a standalone `ExaFlow.app` that runs without this repository, without Python and without a separate MPI installation, and installs it as `/Applications/ExaFlow.app`. The build runs on macOS only, because it uses `ditto` and `codesign`; on Windows and Linux, start the GUI from the repository as in the Quick start:

```bash
uv run python scripts/update_app.py
```

The build takes about a minute and leaves 1.3 GB in `dist/`: the 568 MB `ExaFlow.app` bundle, and the 798 MB `ExaFlow` directory that PyInstaller copied it from. It does five things: it draws `packaging/ExaFlow.icns`, it runs PyInstaller with `packaging/ExaFlow.spec`, it copies Open MPI from `.venv` into `Contents/Resources/mpi`, it removes the local symbols from every `.dylib` and `.so` file in the bundle and signs the bundle again, and it starts the bundled app twice: once with `--check-bundle` to prove that scipy, VTK, PySide6 and MPI still import, and once with `run --case` to prove that a real case runs.

The install then copies `dist/ExaFlow.app` to `/Applications/ExaFlow.app.new` with `ditto`, and it removes the old bundle only after that copy succeeds, so a copy that fails leaves the app that works in `/Applications`. A bare `cp -R` writes into the bundle that is already there, so the files of the older build stay and `codesign --verify` then rejects the whole app. The script verifies the signature of the installed bundle at the end. It stops before it copies anything when `/Applications/ExaFlow.app` is running, because the install deletes the files that a running app still reads.

| Option | What it does |
| --- | --- |
| `--build-only` | Builds into `dist/` and installs nothing. |
| `--install-only` | Installs the bundle already in `dist/`, which takes seconds. |
| `--keep-icon` | Reuses `packaging/ExaFlow.icns` instead of drawing it again. |
| `--quit-app` | Quits the running app, and ends the simulation it runs. |
| `--destination` | Installs somewhere other than `/Applications/ExaFlow.app`. |

Two details keep MPI working inside the bundle:

- `packaging/entry.py` sets `OPAL_PREFIX`, `PRTE_PREFIX` and `MPI4PY_LIBMPI` to the copied Open MPI before anything imports mpi4py. It also removes `MPI4PY_MPIABI` from the environment, because mpi4py takes the ABI name from that variable and then dlopens no `MPI4PY_LIBMPI` at all.
- A frozen bundle holds no separate Python interpreter, so each rank re-runs the app itself: `mpiexec -n 4 ExaFlow run --case <path>`. `packaging/entry.py` sends those arguments to the same CLI parser.

## Current status

**Working:**
- Convection and diffusion operators, written for 1D, 2D and 3D from one stencil each
- Explicit RK time stepping (orders 1-3), each verified at its design order
- MPI domain decomposition and ghost exchange, verified rank-count independent
- All boundary condition types (except time-dependent)
- CSV and VTK output, one format per run folder
- 374 tests, and `uv run mypy src tests` reporting no issues

**In progress:**
- Pressure projection (Poisson solver for incompressibility)

Without pressure, the solver evolves momentum but does not enforce that the velocity field is divergence-free. `SolverOptions` raises `NotImplementedError` when you set `include_pressure=True`, so leave it at the default `False`. `pressure_poisson.py` builds a pure Neumann Laplacian over one rank's block and solves it with conjugate gradient. Its `project` method takes a `FlowState` and does the whole corrector half of Chorin's method, so the remaining work is the ghost exchange: it has none, which makes it correct on a single rank only and keeps it out of the time loop.
