# SolveHypre.c: EndWindings AMS Hypre error investigation

## Symptom

Running the `EndWindings` case (`Linear System preconditioning = ams`, PCG outer
solver) reported three Hypre errors during setup/solve, even though the actual
linear solve converged correctly (47 iterations, residual `9.56e-9`, matching
the reference norm):

```
SolveHypre: CreateHypreAMS (G assembly): Hypre library returned error code 1: [Generic error]
SolveHypre: CreateHypreAMS (SetDiscreteGradient): Hypre library returned error code 36: [Error in argument 4]
SolveHypre: SolveHypre2 (solve): Hypre library returned error code 256: [Method did not converge]
```

The initial hypothesis was that the recent bulk-assembly rewrite of
`HYPRE_IJMatrixAddToValues` calls (batching many rows into one call instead of
one call per row) had introduced a regression. **This turned out to be wrong.**
Job logs compiled from commit `a49616844` ("Add error checks in SolveHypre"),
the common ancestor of both assembly-rewrite branches, already show the same
"Generic error" failure — before any bulk-assembly code existed. All three
bugs below predate the rewrite and are unrelated to it. The rewrite's own
`CheckHypreError()` checkpoints are, ironically, what made these pre-existing
bugs visible in the first place.

## Bugs found (all pre-existing, none caused by the bulk-assembly rewrite)

### 1. Hypre was never explicitly initialized

Elmer never called `HYPRE_Init()`/`HYPRE_Initialize()` anywhere. Hypre's own
lazy accessor (`hypre_handle()` in `utilities/general.c`) self-initializes on
first use, but also unconditionally sets `HYPRE_ERROR_GENERIC` first as a
warning-that's-treated-as-an-error, and never clears it. Whichever Hypre call
happened to run first in a given configuration (`G` assembly under AMS, or `A`
assembly when AMS was disabled) ate this one-time cost and got blamed for it.

**Fix:** call `HYPRE_Init()` (idempotent) at the top of `solvehypre1` and
`createhypreams`, before any other Hypre call.

### 2. Dead code passing a NULL pointer into `HYPRE_IJVectorSetValues`

`createhypreams` built three throwaway coordinate vectors (`xx`/`yy`/`zz`)
from `xx_d`/`yy_d`/`zz_d`, intended for `HYPRE_AMSSetCoordinateVectors` /
`HYPRE_AMSSetEdgeConstantVectors` — both already commented out, so the vectors
were never actually used. `xx_d`/`yy_d`/`zz_d` are declared `ALLOCATABLE` in
`SParIterSolver.F90` but never `ALLOCATE`d, so they arrive as `NULL`.
`HYPRE_IJVectorSetValues`'s 4th argument is exactly this `values` pointer, and
it has a direct `if (!values) hypre_error_in_arg(4)` check — confirmed by
disassembling the compiled call chain, not just reading source.

**Fix:** deleted the whole dead block (~45 lines, both the `#if 0` and
`#else` branches — neither was needed).

### 3. `AMS Tolerance` default is wrong for preconditioner usage

`hypre_AMSSolve` (the AMS cycle) contains:
```c
if (ams_data->num_iterations == ams_data->maxit && ams_data->tol > 0.0)
   hypre_error(HYPRE_ERROR_CONV);
```
When AMS is used as a *preconditioner* for an outer Krylov method (our case:
PCG outer solver, `hypremethod=602`), `AMS Max Iterations` correctly defaults
to `1` (one cycle per outer iteration) — but `AMS Tolerance` also defaults to
`1.0e-6` (`SParIterSolver.F90`), matching Hypre's own default instead of `0`.
Since `num_iterations == maxit` is guaranteed true for a single fixed-cycle
preconditioner application, this fires on every single PCG iteration
regardless of whether the outer solve is converging. (Hypre's own internal
`BoomerAMG` sub-solvers inside AMS correctly dodge this exact trap by
explicitly setting `Tol=0.0` for their fixed-cycle internal solves — AMS's own
top-level tolerance just never got the same treatment.)

**Fix (applied):** added `AMS Tolerance = 0.0` to `hierarc.sif`'s Solver 2
block — sif-only, no rebuild required. (Not applied, but worth doing later:
a source-level fix in `CreateHypreAMS` to force `tol=0` automatically whenever
AMS is used as a preconditioner rather than the top-level solver, so future
sif files don't have to know about this gotcha.)

## Files changed

- `fem/src/SolveHypre.c` — `HYPRE_Init()` calls added; dead coordinate-vector
  block removed. (Diagnostic-only `CheckHypreError`/`DEBUG A:` instrumentation
  added during investigation is still present and harmless; can be stripped
  before merging.)
- `hierarc.sif` (local test case) — added `AMS Tolerance = 0.0`.

## On the bulk-assembly rewrite itself

Ruled out as a cause. `G`/`Pi` assembly in `CreateHypreAMS` is currently
reverted to the original per-row `HYPRE_IJMatrixAddToValues` calls (from
isolating the "moving error" symptom before the real causes were found); `A`/
`Atilde` in `SolveHypre1` still uses the bulk multi-row assembly. Both are
confirmed clean under AddressSanitizer and produce a fully passing run, so
there's no evidence either style is unsafe — the G/Pi revert isn't required
for correctness, just left in place from the investigation.

## Verification

Local single-partition run of `EndWindings/hierarc.sif` with AMS enabled:
zero Hypre error-code lines, `CompareToReferenceSolution: PASSED all 2 tests!`.
