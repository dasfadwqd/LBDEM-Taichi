"""Hybrid coarse-PSC / fine-source D3Q19 coupling.

Implementation follows docs/hybrid_lbm_dem_method_and_design.md.
The solver uses standard distribution moments and inherited streaming.

Initialize once with ``initialize_complete()``. Each fluid step is ordered as::

    lattice.coupling()
    lattice.collide()
    lattice.collect_coarse_force()
    lattice.stream()
    lattice.update_history()
    # Advance DEM substeps using both particle force contributions.

Without explicit ``coarse_particle_ids``, particles with diameter > 4*dx
are selected as coarse during construction. Classification runs once at initialization;
subsequent kernels use cached particle modes and coarse array indices.
Particle modes remain fixed. This first version uses non-periodic particle
mapping, BGK, and optional Smagorinsky LES. It preserves the original PSC
effective radius and single-particle cell ownership. Fine-particle velocity
sampling in partially covered cells is a pore-weighted approximation.
11/1
"""

import numpy as np
import taichi as ti
import taichi.math as tm

from src.dem3d.demsolver import DEMSolver
from src.lbm3d.lbm_solver3d import BasicLattice3D
from src.lbm3d.lbmutils import CellType


Vector3 = ti.types.vector(3, float)
COLLISION_BOUNDARY_MASK = (
    CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP
    | CellType.VEL_INLET_LADD
)
WET_NODE_BOUNDARY_MASK = (
    CellType.VEL_ZOUHE | CellType.VEL_EXIT | CellType.Pre_ZOUHE
    | CellType.VEL_SKORDOS | CellType.PRE_SKORDOS
)
MAPPING_BOUNDARY_MASK = COLLISION_BOUNDARY_MASK | WET_NODE_BOUNDARY_MASK
SOLID_WALL_MASK = CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP


@ti.data_oriented
class HybridSourceLattice3D(BasicLattice3D):
    """Resolved coarse grains and unresolved fine grains on one lattice.

    ``pore_fraction`` is the true fluid fraction epsilon, not epsilon/a.
    ``mixed_fraction`` is eta = 1 - alpha_f and drives the source terms.
    Coarse forces act on particles; fine reaction forces act on the fluid.
    DEM forces use SI units and all lattice force fields use lattice units.
    """

    FINE = 0
    COARSE = 1

    def __init__(self, Nx: int, Ny: int, Nz: int, omega: float,
                 dx: float, dt: float, rho: float, demsolver: DEMSolver,
                 turbulence_model="none", *, coarse_particle_ids=None):
        turbulence_model = turbulence_model.lower()
        if turbulence_model not in ("none", "smagorinsky"):
            raise ValueError("turbulence_model must be 'none' or 'smagorinsky'")
        if min(Nx, Ny, Nz) < 3:
            raise ValueError("Non-periodic dimensions require boundary and interior nodes")
        if dx <= 0.0 or dt <= 0.0 or rho <= 0.0 or not 0.0 < omega < 2.0:
            raise ValueError("Require dx, dt, rho > 0 and 0 < omega < 2")
        # Use the original seven-argument interface also supported by older bases.
        # This solver owns its BGK/LES selection and does not require MRT support.
        super().__init__(Nx, Ny, Nz, omega, dx, dt, rho)
        self.opposite_directions = tuple(int(q) for q in self.qinv.to_numpy())
        self.collision_model = "bgk"
        self.turbulence_model = turbulence_model
        self.omega0 = omega
        self.nuLu = (1.0 / omega - 0.5) / 3.0
        self.dem = demsolver
        if coarse_particle_ids is None:
            self.coarse_particle_ids = self._detect_coarse_particle_ids(
                self.dem.gf.ID.to_numpy(), self.dem.gf.radius.to_numpy(), dx)
        else:
            self.coarse_particle_ids = self._validate_coarse_particle_ids(coarse_particle_ids)
        self.rho0 = rho
        self.mu = rho * self.nuLu * dx * dx / dt
        shape = (Nx, Ny, Nz)
        particle_count = self.dem.gf.shape[0]
        # A dummy slot permits an empty particle field without a zero-size allocation.
        self.particle_mode = ti.field(ti.i32, shape=max(1, particle_count))
        self.coarse_count = len(self.coarse_particle_ids)
        self.coarse_indices = ti.field(ti.i32, shape=max(1, self.coarse_count))
        self.particle_mapping_error = ti.field(ti.i32, shape=max(1, particle_count))
        self.coarse_id = ti.field(ti.i32, shape=shape)
        self.coarse_volfrac = ti.field(float, shape=shape)
        self.coarse_weight = ti.field(float, shape=shape)
        self.coarse_velsolid = ti.Vector.field(3, float, shape=shape)
        self.coarse_hydroforce = ti.Vector.field(3, float, shape=shape)
        self.fine_volfrac = ti.field(float, shape=shape)
        self.mixed_fraction = ti.field(float, shape=shape)
        self.mixed_fraction_prev = ti.field(float, shape=shape)
        self.pore_fraction = ti.field(float, shape=shape)
        self.fine_reaction_force = ti.Vector.field(3, float, shape=shape)
        self.body_force = ti.Vector.field(3, float, shape=shape)
        self.mass_source = ti.Vector.field(self.Q, float, shape=shape)
        self.mass_source_prev = ti.Vector.field(self.Q, float, shape=shape)
        self.momentum_source = ti.Vector.field(self.Q, float, shape=shape)
        self.momentum_source_prev = ti.Vector.field(self.Q, float, shape=shape)
        self.mapping_error = ti.field(ti.i32, shape=())
        self.capacity_excess = ti.field(float, shape=shape)
        self.coarse_id.fill(-1)
        self.mixed_fraction.fill(1.0)
        self.mixed_fraction_prev.fill(1.0)
        self.pore_fraction.fill(1.0)

    @classmethod
    def _detect_coarse_particle_ids(cls, particle_ids, radii, dx):
        """Select once using physical diameter; equality at 4*dx remains fine."""
        radii = np.asarray(radii)
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError("Particle radii must be finite and positive for automatic classification")
        selected = np.asarray(particle_ids)[2.0 * radii > 4.0 * dx]
        return cls._validate_coarse_particle_ids(selected)

    @staticmethod
    def _validate_coarse_particle_ids(particle_ids):
        """IDs are positive, unique P4P identifiers, not array indices."""
        ids = tuple(particle_ids)
        if any(isinstance(pid, (bool, np.bool_)) or not isinstance(pid, (int, np.integer))
               or pid <= 0 for pid in ids):
            raise ValueError("coarse_particle_ids must contain positive integer P4P IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("coarse_particle_ids must not contain duplicate IDs")
        return tuple(int(pid) for pid in ids)

    def _classify_particles(self):
        """Classify by configured P4P IDs and cache their DEM array indices."""
        ids = self.dem.gf.ID.to_numpy()
        coarse_indices = []
        for particle_id in self.coarse_particle_ids:
            matches = np.flatnonzero(ids == particle_id)
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one coarse particle with ID {particle_id}; "
                                 f"found {len(matches)}")
            coarse_indices.append(int(matches[0]))
        modes = np.full(self.particle_mode.shape[0], self.FINE, dtype=np.int32)
        modes[coarse_indices] = self.COARSE
        self.particle_mode.from_numpy(modes)
        indices = np.full(max(1, self.coarse_count), -1, dtype=np.int32)
        indices[:self.coarse_count] = coarse_indices
        self.coarse_indices.from_numpy(indices)
        print(f"Particle classification: coarse={self.coarse_count}, fine={len(ids)-self.coarse_count}; "
              f"coarse P4P IDs={self.coarse_particle_ids}")

    @ti.func
    def _particle_lattice_coordinates(self, pid: int):
        position = self.dem.gf[pid].position
        return Vector3((position[0] - self.dem.config.domain.xmin) / self.unit.dx + 0.5,
                       (position[1] - self.dem.config.domain.ymin) / self.unit.dx + 0.5,
                       (position[2] - self.dem.config.domain.zmin) / self.unit.dx + 0.5)

    @ti.func
    def _valid_particle_position(self, pid: int):
        position = self.dem.gf[pid].position
        # Strict inequalities reject centers on an open/domain face as well.
        valid = (self.dem.config.domain.xmin < position[0]
                 and position[0] < self.dem.config.domain.xmax
                 and self.dem.config.domain.ymin < position[1]
                 and position[1] < self.dem.config.domain.ymax
                 and self.dem.config.domain.zmin < position[2]
                 and position[2] < self.dem.config.domain.zmax
                 and self.dem.gf[pid].radius > 0.0)
        return valid

    @ti.func
    def _compute_coarse_overlap(self, i: int, j: int, k: int,
                                center: Vector3, radius: float):
        """Preserve the legacy effective radius and 125-point overlap estimate."""
        distance = tm.length(Vector3(i, j, k) - center)
        overlap = 0.0
        half_diagonal = 0.5 * ti.sqrt(3.0)
        if distance <= radius - half_diagonal:
            overlap = 1.0
        elif distance < radius + half_diagonal:
            count = 0
            for ii, jj, kk in ti.ndrange(5, 5, 5):
                sample = Vector3(i - 0.4 + 0.2 * ii,
                                 j - 0.4 + 0.2 * jj,
                                 k - 0.4 + 0.2 * kk)
                if tm.length(center - sample) < radius:
                    count += 1
            overlap = count / 125.0
        return overlap

    @ti.func
    def _compute_solid_velocity(self, pid: int, offset: Vector3):
        velocity = (self.dem.gf[pid].velocity
                    + tm.cross(self.dem.gf[pid].omega, offset * self.unit.dx))
        return velocity * self.unit.dt / self.unit.dx

    @ti.func
    def _threedelta(self, distance: float):
        value = 0.0
        if distance < 0.5:
            value = (1.0 + ti.sqrt(1.0 - 3.0 * distance ** 2)) / 3.0
        elif distance <= 1.5:
            value = (5.0 - 3.0 * distance
                     - ti.sqrt(1.0 - 3.0 * (1.0 - distance) ** 2)) / 6.0
        return value

    @ti.func
    def _kernel_weight(self, center: Vector3, raw: ti.types.vector(3, ti.i32)):
        return (self._threedelta(ti.abs(center[0] - raw[0]))
                * self._threedelta(ti.abs(center[1] - raw[1]))
                * self._threedelta(ti.abs(center[2] - raw[2])))

    @ti.func
    def _mapping_index(self, raw: ti.types.vector(3, ti.i32)):
        """Fold wall lobes only; discard lobes across open faces.

        Face types are read at the nearest in-domain tangential coordinates.
        Clamped indices are always safe, even when the returned flag is false.
        """
        index = raw
        near = ti.Vector([ti.max(0, ti.min(self.Nx - 1, raw[0])),
                          ti.max(0, ti.min(self.Ny - 1, raw[1])),
                          ti.max(0, ti.min(self.Nz - 1, raw[2]))])
        valid = 1
        for axis in ti.static(range(3)):
            size = ti.static((self.Nx, self.Ny, self.Nz)[axis])
            if raw[axis] <= 0 or raw[axis] >= size - 1:
                face = near
                face[axis] = 0
                if raw[axis] >= size - 1:
                    face[axis] = size - 1
                if self.CT[face[0], face[1], face[2]] & SOLID_WALL_MASK:
                    if raw[axis] <= 0:
                        index[axis] = 1 - raw[axis]
                    else:
                        index[axis] = 2 * size - 3 - raw[axis]
                else:
                    valid = 0
            if index[axis] < 1 or index[axis] > size - 2:
                valid = 0
            index[axis] = ti.max(0, ti.min(size - 1, index[axis]))
        return index, valid

    @ti.func
    def _is_mapping_cell(self, i: int, j: int, k: int):
        return (self.CT[i, j, k] & MAPPING_BOUNDARY_MASK) == 0

    @ti.func
    def _mapping_sample(self, center: Vector3, raw: ti.types.vector(3, ti.i32)):
        """The same pore-filtered weight is used in all three transfers."""
        index, valid = self._mapping_index(raw)
        weight = 0.0
        if valid and self._is_mapping_cell(index[0], index[1], index[2]):
            available = 1.0 - self.coarse_volfrac[index[0], index[1], index[2]]
            if available > 0.0:
                weight = self._kernel_weight(center, raw) * available
        return index, weight

    @ti.func
    def _stencil_origin(self, center: Vector3):
        return ti.cast(ti.floor(center - 1.5), ti.i32)

    @ti.func
    def _mapping_denominator(self, center: Vector3):
        origin = self._stencil_origin(center)
        total = 0.0
        # Four candidate indices per axis contain the full three-point support.
        for di, dj, dk in ti.ndrange(4, 4, 4):
            raw = origin + ti.Vector([di, dj, dk])
            index, weight = self._mapping_sample(center, raw)
            total += weight
        return total

    @ti.kernel
    def grains2lattice(self):
        """Separate top-level loops complete coarse geometry before fine mapping.

        Only current fields are cleared. Errors are inspected by Python before
        drag or collision; failed mappings are never accepted as a fluid step.
        """
        self.mapping_error[None] = 0
        for pid in range(self.dem.gf.shape[0]):
            self.dem.gf[pid].force_fluid = Vector3(0.0, 0.0, 0.0)
            self.dem.gf[pid].moment_fluid = Vector3(0.0, 0.0, 0.0)
            self.particle_mapping_error[pid] = 0
            if not self._valid_particle_position(pid):
                self.particle_mapping_error[pid] = 1
                ti.atomic_add(self.mapping_error[None], 1)

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            self.coarse_id[i, j, k] = -1
            self.coarse_volfrac[i, j, k] = 0.0
            self.coarse_weight[i, j, k] = 0.0
            self.coarse_velsolid[i, j, k] = Vector3(0.0, 0.0, 0.0)
            self.coarse_hydroforce[i, j, k] = Vector3(0.0, 0.0, 0.0)
            self.fine_volfrac[i, j, k] = 0.0
            self.fine_reaction_force[i, j, k] = Vector3(0.0, 0.0, 0.0)
            self.capacity_excess[i, j, k] = 0.0

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if not (self.CT[i, j, k] & COLLISION_BOUNDARY_MASK):
                # Preserve PSC's first fully covering grain and maximum partial overlap.
                for coarse_slot in range(self.coarse_count):
                    pid = self.coarse_indices[coarse_slot]
                    if self.particle_mapping_error[pid] == 0:
                        center = self._particle_lattice_coordinates(pid)
                        radius = self.dem.gf[pid].radius / self.unit.dx - 0.5
                        overlap = self._compute_coarse_overlap(i, j, k, center, radius)
                        if overlap > self.coarse_volfrac[i, j, k]:
                            self.coarse_id[i, j, k] = pid
                            self.coarse_volfrac[i, j, k] = overlap
                            self.coarse_velsolid[i, j, k] = self._compute_solid_velocity(
                                pid, Vector3(i, j, k) - center)
                        if overlap == 1.0:
                            break

        for pid in range(self.dem.gf.shape[0]):
            if (self.particle_mode[pid] == self.FINE
                    and self.particle_mapping_error[pid] == 0):
                center = self._particle_lattice_coordinates(pid)
                denominator = self._mapping_denominator(center)
                if denominator <= 0.0:
                    self.particle_mapping_error[pid] = 2
                    ti.atomic_add(self.mapping_error[None], 1)
                else:
                    volume = 4.0 / 3.0 * tm.pi * self.dem.gf[pid].radius ** 3
                    origin = self._stencil_origin(center)
                    for di, dj, dk in ti.ndrange(4, 4, 4):
                        raw = origin + ti.Vector([di, dj, dk])
                        index, weight = self._mapping_sample(center, raw)
                        if weight > 0.0:
                            contribution = volume / self.unit.dx ** 3 * weight / denominator
                            ti.atomic_add(self.fine_volfrac[index[0], index[1], index[2]],
                                          contribution)

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            available = 1.0 - self.coarse_volfrac[i, j, k]
            fine = self.fine_volfrac[i, j, k]
            self.mixed_fraction[i, j, k] = 1.0 - fine
            self.pore_fraction[i, j, k] = available - fine
            excess = fine - available
            # Do not clip either fraction, even when a mapping fails.
            if excess > 0.0:
                self.capacity_excess[i, j, k] = excess
                ti.atomic_add(self.mapping_error[None], 1)
            elif fine > 0.0 and self.mixed_fraction[i, j, k] <= 0.0:
                # Equality can exhaust a pure-fine cell and make the source singular.
                self.capacity_excess[i, j, k] = fine
                ti.atomic_add(self.mapping_error[None], 1)

    def _check_mapping(self):
        """Host-side failure reporting avoids exceptions inside Taichi scope."""
        if int(self.mapping_error[None]) != 0:
            errors = self.particle_mapping_error.to_numpy()[:self.dem.gf.shape[0]]
            excess = self.capacity_excess.to_numpy()
            particles = np.flatnonzero(errors)
            cells = np.argwhere(excess > 0.0)
            details = []
            for pid in particles[:8]:
                reason = "invalid center/radius" if errors[pid] == 1 else "no pore support"
                radius = float(self.dem.gf[int(pid)].radius)
                position = np.asarray(self.dem.gf[int(pid)].position, dtype=float)
                if errors[pid] == 1:
                    if not np.isfinite(position).all() or not np.isfinite(radius):
                        reason = "nonfinite position/radius"
                    elif radius <= 0.0:
                        reason = "nonpositive radius"
                    else:
                        reason = "center outside strict domain bounds"
                volume = 4.0 / 3.0 * np.pi * radius ** 3
                details.append(f"particle {pid}: {reason}, position={position.tolist()}, "
                               f"radius={radius:.8e} m, unmapped volume={volume:.8e} m^3")
            for cell in cells[:8]:
                index = tuple(int(value) for value in cell)
                details.append(f"cell {index}: capacity/singularity residual="
                               f"{float(excess[index]) * self.unit.dx ** 3:.8e} m^3")
            raise RuntimeError(
                f"Hybrid mapping failed ({len(particles)} particles, {len(cells)} cells). "
                + "; ".join(details)
                + ". Adjust particle positions, grid spacing or particle sizes; "
                "no volume was clipped and coupling did not advance.")

    @ti.func
    def _interpolate_fluid_state(self, center: Vector3):
        velocity_sum = Vector3(0.0, 0.0, 0.0)
        porosity_sum = 0.0
        denominator = 0.0
        origin = self._stencil_origin(center)
        for di, dj, dk in ti.ndrange(4, 4, 4):
            raw = origin + ti.Vector([di, dj, dk])
            index, weight = self._mapping_sample(center, raw)
            if weight > 0.0:
                i, j, k = index[0], index[1], index[2]
                available = 1.0 - self.coarse_volfrac[i, j, k]
                velocity_sum += weight * self.vel[i, j, k]
                porosity_sum += weight * self.pore_fraction[i, j, k] / available
                denominator += weight
        velocity = Vector3(0.0, 0.0, 0.0)
        porosity = 0.0
        if denominator > 0.0:
            velocity = velocity_sum / denominator * self.unit.dx / self.unit.dt
            porosity = porosity_sum / denominator
        return velocity, porosity, denominator

    @ti.func
    def _compute_drag_force(self, diameter: float, slip: Vector3,
                            solid_fraction: float):
        """Legacy Tenneti force in SI units, using molecular viscosity only."""
        fluid_fraction = 1.0 - solid_fraction
        drag = Vector3(0.0, 0.0, 0.0)
        if fluid_fraction > 1e-9:
            reynolds = fluid_fraction * self.rho0 * diameter * tm.length(slip) / self.mu
            cd0 = 1.0 + 0.15 * tm.pow(reynolds, 0.687)
            static_term = (5.81 * solid_fraction / fluid_fraction ** 3
                           + 0.48 * tm.pow(solid_fraction, 1.0 / 3.0)
                           / fluid_fraction ** 4)
            solid_fraction3 = solid_fraction ** 3
            dynamic_term = solid_fraction3 * reynolds * (
                0.95 + 0.61 * solid_fraction3 / fluid_fraction ** 2)
            cd_star = fluid_fraction * (
                cd0 / fluid_fraction ** 3 + static_term + dynamic_term)
            drag = -3.0 * tm.pi * diameter * self.mu * fluid_fraction * cd_star * slip
        return drag

    @ti.kernel
    def lattice2grains(self):
        """Write fine drag and spread its opposite; do not reset DEM forces here."""
        for pid in range(self.dem.gf.shape[0]):
            if self.particle_mode[pid] == self.FINE:
                center = self._particle_lattice_coordinates(pid)
                velocity, porosity, denominator = self._interpolate_fluid_state(center)
                # The preceding mapping check guarantees a nonempty stencil.
                assert denominator > 0.0, "Fine particle has no valid mapping support"
                # The baseline drag formula suppresses force below this limit.
                # Reject that singular closure instead of silently losing drag.
                assert porosity > 1e-9, "Fine particle pore porosity is too small for drag"
                force = self._compute_drag_force(2.0 * self.dem.gf[pid].radius,
                                                self.dem.gf[pid].velocity - velocity,
                                                1.0 - porosity)
                self.dem.gf[pid].force_fluid = force
                reaction = -force * self.unit.dt ** 2 / (self.unit.rho * self.unit.dx ** 4)
                origin = self._stencil_origin(center)
                for di, dj, dk in ti.ndrange(4, 4, 4):
                    raw = origin + ti.Vector([di, dj, dk])
                    index, weight = self._mapping_sample(center, raw)
                    if weight > 0.0:
                        ti.atomic_add(self.fine_reaction_force[index[0], index[1], index[2]],
                                      reaction * weight / denominator)

    @ti.func
    def _source_index(self, raw: ti.types.vector(3, ti.i32)):
        """No ghost or mirror sources: out-of-domain neighbors are invalid."""
        valid = (0 <= raw[0] and raw[0] < self.Nx
                 and 0 <= raw[1] and raw[1] < self.Ny
                 and 0 <= raw[2] and raw[2] < self.Nz)
        index = ti.Vector([ti.max(0, ti.min(self.Nx - 1, raw[0])),
                           ti.max(0, ti.min(self.Ny - 1, raw[1])),
                           ti.max(0, ti.min(self.Nz - 1, raw[2]))])
        if self.CT[index[0], index[1], index[2]] & COLLISION_BOUNDARY_MASK:
            valid = False
        return index, valid

    @ti.func
    def _compute_source_terms(self, i: int, j: int, k: int):
        eta = self.mixed_fraction[i, j, k]
        gradient = Vector3(0.0, 0.0, 0.0)
        for axis in ti.static(range(3)):
            offset = ti.Vector.zero(ti.i32, 3)
            offset[axis] = 1
            minus, valid_minus = self._source_index(ti.Vector([i, j, k]) - offset)
            plus, valid_plus = self._source_index(ti.Vector([i, j, k]) + offset)
            eta_minus = eta
            eta_plus = eta
            # Solid/invalid neighbors use the local eta. Wet boundaries retain
            # their mapped eta (normally one), matching the source baseline.
            if valid_minus:
                eta_minus = self.mixed_fraction[minus[0], minus[1], minus[2]]
            if valid_plus:
                eta_plus = self.mixed_fraction[plus[0], plus[1], plus[2]]
            gradient[axis] = 0.5 * (eta_plus - eta_minus)
        eta_rate = eta - self.mixed_fraction_prev[i, j, k]
        velocity = self.vel[i, j, k]
        sq = -self.rho[i, j, k] / eta * (eta_rate + tm.dot(velocity, gradient))
        sm = sq * velocity + (self.fine_reaction_force[i, j, k]
                              + self.body_force[i, j, k]) / eta
        return sq, sm

    @ti.func
    def _compute_source_distribution(self, i: int, j: int, k: int,
                                     sq: float, sm: Vector3):
        velocity = self.vel[i, j, k]
        for q in ti.static(range(self.Q)):
            cu = tm.dot(self.c[q], velocity)
            cm = tm.dot(self.c[q], sm)
            self.mass_source[i, j, k][q] = self.w[q] * sq
            self.momentum_source[i, j, k][q] = self.w[q] * (
                cm / self.cssq + (cu * cm - self.cssq * tm.dot(velocity, sm))
                / (2.0 * self.cssq ** 2))

    @ti.kernel
    def assemble_source(self):
        """Complete all base sources before collision reads forward neighbors."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            self.mass_source[i, j, k] = ti.Vector.zero(float, self.Q)
            self.momentum_source[i, j, k] = ti.Vector.zero(float, self.Q)
            if not (self.CT[i, j, k] & COLLISION_BOUNDARY_MASK):
                sq, sm = self._compute_source_terms(i, j, k)
                self._compute_source_distribution(i, j, k, sq, sm)

    @ti.func
    def _compute_directional_source(self, i: int, j: int, k: int, q: ti.template()):
        raw = ti.Vector([i, j, k]) + ti.cast(self.c[q], ti.i32)
        forward, valid = self._source_index(raw)
        forward_source = 0.0
        if valid:
            forward_source = (self.mass_source[forward[0], forward[1], forward[2]][q]
                              + self.momentum_source[forward[0], forward[1], forward[2]][q])
        base = self.mass_source[i, j, k][q] + self.momentum_source[i, j, k][q]
        previous = self.mass_source_prev[i, j, k][q] + self.momentum_source_prev[i, j, k][q]
        return base + 0.5 * (forward_source - previous)

    @ti.func
    def _compute_psc_weight(self, alpha: float, omega: float):
        tau_shift = 1.0 / omega - 0.5
        return alpha * tau_shift / (1.0 - alpha + tau_shift)

    @ti.func
    def _local_shear_relaxation(self, i: int, j: int, k: int):
        """Reuse the original Smagorinsky helper without a newer base API.

        Wet boundaries keep the molecular rate. Interior LES updates finish
        before the PSC weight is evaluated with the same local rate.
        """
        relaxation = self.omega0
        if not (self.CT[i, j, k] & WET_NODE_BOUNDARY_MASK):
            if ti.static(self.turbulence_model == "smagorinsky"):
                self.computeOmega(i, j, k)
                relaxation = self.omega[i, j, k]
        self.omega[i, j, k] = relaxation
        return relaxation

    @ti.func
    def _compute_solid_equilibrium(self, i: int, j: int, k: int, q: ti.template()):
        """Evaluate one direction without dynamically indexing a local VectorQ."""
        velocity = self.coarse_velsolid[i, j, k]
        cu = tm.dot(self.c[q], velocity)
        return self.w[q] * self.rho[i, j, k] * (
            1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * tm.dot(velocity, velocity))

    @ti.func
    def _compute_solid_collision(self, i: int, j: int, k: int,
                                 q: ti.template(), solid_eq: float):
        inverse = ti.static(self.opposite_directions[q])
        return (self.f[i, j, k][inverse] - self.feq[i, j, k][inverse]
                + solid_eq - self.f[i, j, k][q])

    @ti.kernel
    def collide(self):
        """Apply one unified PSC + BGK + complete source formula in every cell."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if not (self.CT[i, j, k] & COLLISION_BOUNDARY_MASK):
                self.compute_feq(i, j, k)
                # Preserve omega0 at wet boundaries and update LES before B.
                omega = self._local_shear_relaxation(i, j, k)
                weight = self._compute_psc_weight(self.coarse_volfrac[i, j, k], omega)
                self.coarse_weight[i, j, k] = weight
                force = Vector3(0.0, 0.0, 0.0)
                # Static directions and inverse indices avoid runtime field-gather
                # plus dynamic VectorQ indexing on the affected CUDA backend.
                for q in ti.static(range(self.Q)):
                    fluid_collision = -omega * (self.f[i, j, k][q] - self.feq[i, j, k][q])
                    solid_eq = self._compute_solid_equilibrium(i, j, k, q)
                    solid_collision = self._compute_solid_collision(i, j, k, q, solid_eq)
                    source = self._compute_directional_source(i, j, k, q)
                    self.fpc[i, j, k][q] = (self.f[i, j, k][q]
                        + (1.0 - weight) * fluid_collision + weight * solid_collision + source)
                    # Only PSC exchange acts on the coarse grain. Fine source
                    # impulse is recorded separately, including inside coarse cells.
                    force -= weight * solid_collision * self.c[q]
                self.coarse_hydroforce[i, j, k] = force

    @ti.kernel
    def collect_coarse_force(self):
        """Follow the legacy particle-owned gather: one writer per coarse grain.

        Accumulate cell forces and torques locally, then write only this grain's
        fluid fields. Fine grains and their already computed drag remain intact.
        """
        scale = self.unit.rho * self.unit.dx ** 4 / self.unit.dt ** 2
        for coarse_slot in range(self.coarse_count):
            pid = self.coarse_indices[coarse_slot]
            if self.particle_mapping_error[pid] == 0:
                center = self._particle_lattice_coordinates(pid)
                radius = self.dem.gf[pid].radius / self.unit.dx
                begin = ti.cast(ti.floor(center - radius), ti.i32)
                end = ti.cast(ti.floor(center + radius), ti.i32) + 2
                total_force = Vector3(0.0, 0.0, 0.0)
                total_torque = Vector3(0.0, 0.0, 0.0)
                for i in range(ti.max(0, begin[0]), ti.min(self.Nx, end[0])):
                    for j in range(ti.max(0, begin[1]), ti.min(self.Ny, end[1])):
                        for k in range(ti.max(0, begin[2]), ti.min(self.Nz, end[2])):
                            if self.coarse_id[i, j, k] == pid:
                                force = self.coarse_hydroforce[i, j, k] * scale
                                lever = (Vector3(i, j, k) - center) * self.unit.dx
                                total_force += force
                                total_torque += tm.cross(lever, force)
                self.dem.gf[pid].force_fluid = self.dem.gf[pid].force_fluid + total_force
                self.dem.gf[pid].moment_fluid = self.dem.gf[pid].moment_fluid + total_torque

    @ti.kernel
    def _initialize_fraction_history(self):
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            self.mixed_fraction_prev[i, j, k] = self.mixed_fraction[i, j, k]

    @ti.kernel
    def update_history(self):
        """Save final eta and base sources at step end, never corrected sources."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            self.mixed_fraction_prev[i, j, k] = self.mixed_fraction[i, j, k]
            self.mass_source_prev[i, j, k] = self.mass_source[i, j, k]
            self.momentum_source_prev[i, j, k] = self.momentum_source[i, j, k]

    def initialize_complete(self):
        """Build initial sources and history without a collision or a DEM step."""
        self.initialize()
        self._classify_particles()
        self.grains2lattice()
        self._check_mapping()
        self._initialize_fraction_history()
        self.lattice2grains()
        self.assemble_source()
        self.update_history()

    def coupling(self):
        """Finish mapping and reaction spreading before assembling neighbor sources."""
        self.grains2lattice()
        self._check_mapping()
        self.lattice2grains()
        self.assemble_source()

    def diagnostics(self):
        """Return optional volume, force, mass and local collision budgets.

        Call after ``collect_coarse_force`` and before ``stream`` or history
        updates. Host-side snapshots make this intentionally expensive; it is
        intended for validation checkpoints, not for every production step.
        Masses are instantaneous inventories. Their time differences and
        boundary fluxes must be measured separately to assess mass balance.
        Source impulses include the forward/history correction and are not
        identified with the conservative fine-force mapping budget.
        """
        active = (self.CT.to_numpy() & COLLISION_BOUNDARY_MASK) == 0
        coarse = self.coarse_volfrac.to_numpy()
        fine = self.fine_volfrac.to_numpy()
        rho = self.rho.to_numpy()
        vel = self.vel.to_numpy()
        weight = self.coarse_weight.to_numpy()
        solid_vel = self.coarse_velsolid.to_numpy()
        f = self.f.to_numpy()
        post = self.fpc.to_numpy()
        feq = self.feq.to_numpy()
        directions = self.c.to_numpy()
        base = self.mass_source.to_numpy() + self.momentum_source.to_numpy()
        previous = self.mass_source_prev.to_numpy() + self.momentum_source_prev.to_numpy()
        corrected = base - 0.5 * previous
        for q, direction in enumerate(directions.astype(int)):
            # Target slices pair x with x+c_q; invalid forward sources are zero.
            target = tuple(slice(max(0, -d), min(n, n - d))
                           for n, d in zip(active.shape, direction))
            forward = tuple(slice(max(0, d), min(n, n + d))
                            for n, d in zip(active.shape, direction))
            corrected[target + (q,)] += 0.5 * base[forward + (q,)] * active[forward]
        r0 = corrected.sum(axis=-1)
        r1 = np.einsum("...q,qd->...d", corrected, directions)
        r2 = np.einsum("...q,qa,qb->...ab", corrected, directions, directions)
        delta = post - f
        mass_error = delta.sum(axis=-1) - r0
        momentum_error = (np.einsum("...q,qd->...d", delta, directions)
                          - weight[..., None] * rho[..., None] * (solid_vel - vel) - r1)
        pi_noneq = np.einsum("...q,qa,qb->...ab", f - feq, directions, directions)
        rigid_stress = rho[..., None, None] * (
            solid_vel[..., :, None] * solid_vel[..., None, :]
            - vel[..., :, None] * vel[..., None, :])
        stress_error = (np.einsum("...q,qa,qb->...ab", delta, directions, directions)
                        + ((1.0 - weight) * self.omega.to_numpy())[..., None, None] * pi_noneq
                        - weight[..., None, None] * rigid_stress - r2)
        volume = self.unit.dx ** 3
        force_scale = self.unit.rho * self.unit.dx ** 4 / self.unit.dt ** 2
        modes = self.particle_mode.to_numpy()
        expected_volume = 0.0
        fine_particle_force = np.zeros(3)
        for pid in range(self.dem.gf.shape[0]):
            if modes[pid] == self.FINE:
                expected_volume += 4.0 / 3.0 * np.pi * float(self.dem.gf[pid].radius) ** 3
                fine_particle_force += np.asarray(self.dem.gf[pid].force_fluid, dtype=float)
        mapped_reaction = self.fine_reaction_force.to_numpy().sum(axis=(0, 1, 2)) * force_scale
        inside = active & (coarse == 1.0)
        constraint_error = 0.0
        if np.any(inside):
            post_density = post[inside].sum(axis=-1)
            post_velocity = post[inside] @ directions / post_density[:, None]
            constraint_error = float(np.max(np.linalg.norm(post_velocity - solid_vel[inside], axis=-1)))
        return {
            "fine_volume_residual_m3": float(fine.sum() * volume - expected_volume),
            "fine_mapping_force_residual_N": mapped_reaction + fine_particle_force,
            "true_fluid_mass_kg": float((self.pore_fraction.to_numpy() * rho)[active].sum()
                                        * self.unit.rho * volume),
            "mixed_fluid_mass_kg": float((self.mixed_fraction.to_numpy() * rho)[active].sum()
                                         * self.unit.rho * volume),
            "coarse_virtual_fluid_mass_kg": float((coarse * rho)[active].sum()
                                                 * self.unit.rho * volume),
            "base_source_mass_increment_kg": float(base[active].sum() * self.unit.rho * volume),
            "complete_source_mass_increment_kg": float(r0[active].sum() * self.unit.rho * volume),
            "base_source_impulse_Ns": (base[active].sum(axis=0) @ directions) * force_scale * self.unit.dt,
            "complete_source_impulse_Ns": r1[active].sum(axis=0) * force_scale * self.unit.dt,
            "coarse_particle_force_N": self.coarse_hydroforce.to_numpy().sum(axis=(0, 1, 2)) * force_scale,
            "collision_mass_error_LU": float(np.max(np.abs(mass_error[active]), initial=0.0)),
            "collision_momentum_error_LU": float(np.max(np.abs(momentum_error[active]), initial=0.0)),
            "collision_stress_error_LU": float(np.max(np.abs(stress_error[active]), initial=0.0)),
            "coarse_internal_velocity_error_LU": constraint_error,
        }
