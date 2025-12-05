'''
// Hybrid LBM–DEM coupling for strongly heterogeneous particle systems:
// - Fully resolved particles (PSC, D_p >= 10 lattice cells):
//   * Geometrically mapped onto the LBM grid as solid cells.
//   * No-slip moving boundaries enforced via Central Linear Interpolation (CLI).
//   * Hydrodynamic forces computed directly from momentum exchange at fluid–solid links.
//   * High accuracy but computationally expensive per particle.
//
// - Unresolved particles VANS, D_p <=  lattice cells):
//   * Treated as point particles; not mapped to the grid.
//   * Modeled as a porous medium via local solid volume fraction ε_p.
//   * Hydrodynamic forces (drag, lift, pressure gradient, added mass) computed using empirical models.
//   * Forces and ε_p interpolated/distributed to fluid cells via filter kernels (e.g., 3-point delta).
//   * LBM equations modified to generalized Navier–Stokes with ε_f = 1 - ε_p and effective viscosity ν_e.
//   * Highly efficient for large numbers of small particles, but less accurate for near-field effects.
//
// Both particle types interact via the same DEM solver for collisions,
// and are advanced within a unified time-stepping loop with subcycled force evaluation
// to ensure numerical stability and physical consistency.
'''