// SMOOTHED PARTICLE HYDRODYNAMICS.
//
// A fluid as particles: each one carries mass, and a field quantity anywhere is
// the weighted sum of the nearby particles' contributions. Density comes out of
// that sum, pressure comes out of density, and the pressure GRADIENT is the
// force that keeps the fluid from collapsing into itself. There is no grid
// holding the fluid -- the grid below exists only to find neighbours quickly.
//
// This is WEAKLY COMPRESSIBLE SPH: pressure is an explicit function of density
// rather than the solution of a Poisson equation. It is far simpler and it
// costs a small timestep, because the stiffness that keeps the fluid nearly
// incompressible is also the stiffness that makes the system stiff.
//
// The neighbour search is a dense grid of fixed-capacity buckets over a bounded
// box. Not a hash: for a fluid the occupied region is compact and known, so a
// dense grid has no collisions and no probing. Not a counting sort either --
// that needs a prefix sum, which is three more dispatches to save the memory a
// fixed bucket wastes, and the wasted memory is a few megabytes.

static inline uint3 CellOf(float3 p, constant GpuFluidParams& f)
{
    const float3 rel = (p - f.bounds_min.xyz) / f.grid.w;
    const int3 c = int3(floor(rel));
    return uint3(clamp(c, int3(0), int3(f.grid.xyz) - 1));
}

static inline uint CellIndex(uint3 c, constant GpuFluidParams& f)
{
    return (c.z * uint(f.grid.y) + c.y) * uint(f.grid.x) + c.x;
}

kernel void cs_fluid_clear(uint id [[thread_position_in_grid]],
                           device atomic_uint* counts [[buffer(0)]],
                           constant GpuFluidParams& f [[buffer(1)]])
{
    const uint cells = uint(f.grid.x) * uint(f.grid.y) * uint(f.grid.z);
    if (id >= cells) return;
    atomic_store_explicit(&counts[id], 0u, memory_order_relaxed);
}

kernel void cs_fluid_bin(uint id [[thread_position_in_grid]],
                         device const GpuFluidParticle* particles [[buffer(0)]],
                         constant GpuFluidParams& f [[buffer(1)]],
                         device atomic_uint* counts [[buffer(2)]],
                         device uint* buckets [[buffer(3)]])
{
    if (id >= uint(f.misc.x)) return;
    const uint cell = CellIndex(CellOf(particles[id].position.xyz, f), f);
    const uint slot = atomic_fetch_add_explicit(&counts[cell], 1u,
                                                memory_order_relaxed);
    const uint cap = uint(f.misc.z);
    // OVERFULL cells drop particles from the neighbour lists rather than
    // writing past the bucket. The particle still exists and still moves -- it
    // just stops being seen by its neighbours for this step, which reads as a
    // brief local softening. Writing past the bucket instead corrupts a
    // neighbouring cell's list and the fluid explodes somewhere unrelated.
    if (slot < cap) buckets[cell * cap + slot] = id;
    else atomic_fetch_sub_explicit(&counts[cell], 1u, memory_order_relaxed);
}

kernel void cs_fluid_density(uint id [[thread_position_in_grid]],
                             device GpuFluidParticle* particles [[buffer(0)]],
                             constant GpuFluidParams& f [[buffer(1)]],
                             device const uint* counts [[buffer(2)]],
                             device const uint* buckets [[buffer(3)]])
{
    if (id >= uint(f.misc.x)) return;
    const float3 pi = particles[id].position.xyz;
    const float h = f.bounds_min.w, h2 = h * h;
    const float mass = f.bounds_max.w;
    const uint cap = uint(f.misc.z);
    const int3 base = int3(CellOf(pi, f));

    float density = 0.0f;
    // The 27 cells around it. The kernel has compact support of radius h and
    // the cells are h across, so nothing outside this block can contribute --
    // which is the entire reason the grid's cell size is the smoothing radius
    // and not something tuned.
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        const int3 c = base + int3(dx, dy, dz);
        if (any(c < int3(0)) || any(c >= int3(f.grid.xyz))) continue;
        const uint cell = CellIndex(uint3(c), f);
        const uint n = counts[cell];
        for (uint k = 0; k < n; ++k) {
            const uint j = buckets[cell * cap + k];
            const float3 d = pi - particles[j].position.xyz;
            const float r2 = dot(d, d);
            if (r2 >= h2) continue;
            // Poly6. Includes the particle itself at r = 0, which is correct
            // and not an off-by-one: a particle contributes to the density at
            // its own position, and leaving it out makes an isolated particle
            // have zero density and infinite pressure.
            const float t = h2 - r2;
            density += mass * f.kernels.x * t * t * t;
        }
    }

    particles[id].position.w = density;
    // Weakly compressible state equation. Clamped at zero: negative pressure is
    // ATTRACTION, and a fluid whose sparse regions pull themselves together
    // tears into blobs and leaves vacuum behind.
    particles[id].velocity.w = max(0.0f, f.physics.y * (density - f.physics.x));
}

kernel void cs_fluid_forces(uint id [[thread_position_in_grid]],
                            device const GpuFluidParticle* particles [[buffer(0)]],
                            constant GpuFluidParams& f [[buffer(1)]],
                            device const uint* counts [[buffer(2)]],
                            device const uint* buckets [[buffer(3)]],
                            device float4* accel [[buffer(4)]])
{
    if (id >= uint(f.misc.x)) return;
    const float3 pi = particles[id].position.xyz;
    const float3 vi = particles[id].velocity.xyz;
    const float rho_i = max(particles[id].position.w, 1e-6f);
    const float p_i = particles[id].velocity.w;
    const float h = f.bounds_min.w;
    const float mass = f.bounds_max.w;
    const uint cap = uint(f.misc.z);
    const int3 base = int3(CellOf(pi, f));

    // Kept apart because they are divided by different things: Monaghan's
    // pressure term is already an acceleration, and the viscous one is a force.
    float3 pressure_accel = float3(0.0f);
    float3 force = float3(0.0f);
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        const int3 c = base + int3(dx, dy, dz);
        if (any(c < int3(0)) || any(c >= int3(f.grid.xyz))) continue;
        const uint cell = CellIndex(uint3(c), f);
        const uint n = counts[cell];
        for (uint k = 0; k < n; ++k) {
            const uint j = buckets[cell * cap + k];
            if (j == id) continue;  // a particle exerts no force on itself
            const float3 d = pi - particles[j].position.xyz;
            const float r = length(d);
            if (r >= h || r < 1e-9f) continue;
            const float3 dir = d / r;
            const float rho_j = max(particles[j].position.w, 1e-6f);
            const float p_j = particles[j].velocity.w;

            // PRESSURE, in Monaghan's form: p_i/rho_i^2 + p_j/rho_j^2.
            //
            // The obvious alternative, (p_i + p_j) / (2 rho_j), is what Mueller
            // 2003 uses and it is subtly WRONG: the coefficient the pair gets
            // depends on whose density is in the denominator, so i pushes j
            // harder than j pushes i and momentum is not conserved. The energy
            // that creates has to go somewhere, and where it goes is into a
            // fluid that never settles -- measured here as kinetic energy
            // RISING, 113 J to 118 J, in a body of water that should be coming
            // to rest. Monaghan's form gives the pair one shared coefficient,
            // so the two forces are exactly equal and opposite.
            const float w_spiky = f.kernels.y * (h - r) * (h - r);
            const float shared = p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j);
            pressure_accel -= dir * (mass * shared * w_spiky);

            // VISCOSITY, in two parts that do different jobs.
            //
            // MONAGHAN'S ARTIFICIAL VISCOSITY is the one that keeps the
            // simulation stable, and it is active ONLY for pairs that are
            // approaching each other. That is the whole trick: the instability
            // in weakly compressible SPH is a compression overshooting and
            // rebounding, so damping compression damps the instability, while
            // separation and shear are left alone. A plain Laplacian damps
            // everything equally, so getting the same stability out of it needs
            // ten times the coefficient -- measured here as the difference
            // between a pool that boils itself up to twice its depth and one
            // that settles at 1001.7 kg/m^3 with 0.2 J left in it. Ten times
            // the Laplacian also turns water into honey.
            const float3 dv = particles[j].velocity.xyz - vi;
            const float approach = dot(dv, -d);  // < 0 when separating
            if (approach < 0.0f) {
                const float mu = h * approach / (r * r + 0.01f * h * h);
                const float rho_bar = 0.5f * (rho_i + rho_j);
                const float pi_ij = -f.artificial.x * f.artificial.y * mu / rho_bar;
                const float w_spiky2 = f.kernels.y * (h - r) * (h - r);
                pressure_accel -= dir * (mass * pi_ij * w_spiky2);
            }

            // The PHYSICAL viscosity, which is a look rather than a stability
            // measure: it is what makes honey behave differently from water.
            const float w_visc = f.kernels.z * (h - r);
            force += dv * (f.physics.z * mass / rho_j * w_visc);
        }
    }

    // The viscous term is a force per unit volume, so it divides by density;
    // the pressure term came out of Monaghan's form as an acceleration already.
    // Dividing both would damp the pressure by a factor of a thousand and give
    // a fluid with no resistance to compression at all.
    accel[id] = float4(pressure_accel + force / rho_i +
                           float3(0.0f, f.misc.w, 0.0f), 0.0f);
}

kernel void cs_fluid_integrate(uint id [[thread_position_in_grid]],
                               device GpuFluidParticle* particles [[buffer(0)]],
                               constant GpuFluidParams& f [[buffer(1)]],
                               device const float4* accel [[buffer(2)]])
{
    if (id >= uint(f.misc.x)) return;
    const float dt = f.physics.w;

    float3 v = particles[id].velocity.xyz + accel[id].xyz * dt;

    // THE CFL CONDITION, enforced per particle instead of only per step.
    //
    // The global substep is chosen from the sound speed, the viscosity and
    // gravity -- all constants. What it cannot know is the acceleration a
    // COLLISION produces: when a falling column lands, local density spikes
    // 80% over rest and the pressure term reaches thousands of m/s^2 from a
    // single neighbour. One step of that moves a particle several smoothing
    // radii, past every neighbour that was pushing it, into a region where the
    // density estimate is meaningless -- and the fluid gains energy from
    // nowhere and never settles. Measured before this: kinetic energy climbing
    // from 487 J to 647 J in a body of water with 182 J of potential energy to
    // give.
    //
    // Capping the distance one substep may cover restores the condition where
    // it is actually violated, and does nothing at all everywhere else. It IS
    // dissipative in the moment it acts, which is the trade: a little energy
    // lost at the instant of impact against a simulation that stays a fluid.
    const float max_travel = 0.4f * f.bounds_min.w;  // smoothing radius
    const float speed = length(v);
    const float vmax = max_travel / max(dt, 1e-9f);
    if (speed > vmax) v *= vmax / speed;

    float3 p = particles[id].position.xyz + v * dt;

    // WALLS. Position clamped inside and the normal velocity reflected, damped
    // by the restitution. Clamping without reflecting leaves particles pressed
    // against the wall with their velocity still pointing into it, and the next
    // step drives them in again -- which shows up as a permanent sheet of
    // fluid stuck to the floor.
    const float3 lo = f.bounds_min.xyz, hi = f.bounds_max.xyz;
    const float e = f.misc.y;
    if (p.x < lo.x) { p.x = lo.x; v.x = -v.x * e; }
    if (p.y < lo.y) { p.y = lo.y; v.y = -v.y * e; }
    if (p.z < lo.z) { p.z = lo.z; v.z = -v.z * e; }
    if (p.x > hi.x) { p.x = hi.x; v.x = -v.x * e; }
    if (p.y > hi.y) { p.y = hi.y; v.y = -v.y * e; }
    if (p.z > hi.z) { p.z = hi.z; v.z = -v.z * e; }

    particles[id].position.xyz = p;
    particles[id].velocity.xyz = v;
}

// --- drawing -----------------------------------------------------------------

struct FluidOut {
    float4 position [[position]];
    float2 uv;
    float3 color;
    float3 world;
};

vertex FluidOut vs_fluid(uint vid [[vertex_id]],
                         uint iid [[instance_id]],
                         device const GpuFluidParticle* particles [[buffer(0)]],
                         constant FrameUniforms& u [[buffer(1)]],
                         constant GpuFluidParams& f [[buffer(2)]])
{
    const GpuFluidParticle q = particles[iid];
    const float2 corner = float2(float(vid & 1u) * 2.0f - 1.0f,
                                 float((vid >> 1u) & 1u) * 2.0f - 1.0f);
    const float3 right = normalize(float3(u.viewProj[0][0], u.viewProj[1][0],
                                          u.viewProj[2][0]));
    const float3 up = normalize(float3(u.viewProj[0][1], u.viewProj[1][1],
                                       u.viewProj[2][1]));
    // Sized from the particle spacing, so a denser fluid does not develop gaps.
    const float radius = f.bounds_min.w * 0.42f;
    const float3 world = q.position.xyz + (right * corner.x + up * corner.y) * radius;

    FluidOut o;
    o.position = u.viewProj * float4(world, 1.0f);
    o.uv = corner;
    o.world = q.position.xyz;
    // Tinted by SPEED. Not decoration: a still fluid and a violently churning
    // one look identical as a field of blue dots, and the whole question when
    // watching a fluid solver is where the energy is.
    const float speed = length(q.velocity.xyz);
    o.color = mix(float3(0.05f, 0.22f, 0.55f), float3(0.75f, 0.92f, 1.0f),
                  saturate(speed * 0.22f));
    return o;
}

fragment float4 fs_fluid(FluidOut in [[stage_in]],
                             constant FrameUniforms& u [[buffer(1)]])
{
    const float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0f) discard_fragment();
    // Shaded as a sphere rather than a flat disc: the z of a unit hemisphere
    // gives a normal, and a cheap diffuse term off it reads as a droplet
    // instead of a sticker.
    const float z = sqrt(1.0f - r2);
    // World space: right/up above are world axes, so this normal is too, and
    // the fixed sun below means the same thing it meant to the diffuse term.
    const float3 n = normalize(float3(in.uv, z));
    const float3 sun = normalize(float3(0.4f, 0.7f, 0.6f));
    const float diffuse = 0.35f + 0.65f * saturate(dot(n, sun));
    // A sun glint and a sky rim: the two cheapest water cues. A matte ball and
    // a water droplet differ almost entirely in these -- the droplet returns
    // the sun in a tight highlight and the sky at grazing angles, and without
    // either every particle reads as plastic no matter what the solver does.
    const float3 v = normalize(u.eyePos.xyz - in.world);
    const float3 h = normalize(sun + v);
    const float spec = pow(saturate(dot(n, h)), 90.0f);
    const float fres = pow(1.0f - saturate(dot(n, v)), 3.0f);
    const float3 color = in.color * diffuse + spec * float3(1.0f, 0.98f, 0.95f) +
                         fres * float3(0.30f, 0.42f, 0.60f) * 0.55f;
    return float4(color, 1.0f);
}
