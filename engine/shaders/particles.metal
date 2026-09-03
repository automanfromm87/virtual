// GPU PARTICLES: simulated in a compute pass, drawn as instanced billboards.
//
// The whole system lives in one buffer that the CPU never reads. A step is one
// dispatch; a draw is one instanced call whose instance count is the pool size.
// Nothing scales with how many particles are alive, which is the point --
// the alternative is the CPU walking the pool every frame to build a vertex
// buffer, and that cost is there whether ten particles are alive or a million.
//
// Slots are RECYCLED rather than compacted. A dead slot is a candidate for the
// next emission, so the pool never moves and no particle ever changes index.
// Compacting would cost a prefix sum every frame to save drawing some degenerate
// quads, and a degenerate quad is rejected before it rasterises anything.

// PCG hash. One multiply and two shifts, and unlike the usual
// sin(dot(uv,magic))*43758.5453 it does not produce visible structure -- that
// one correlates badly for nearby inputs, which is exactly the case here
// because the inputs are adjacent thread indices.
static inline uint Pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static inline float Rand01(thread uint& seed)
{
    seed = Pcg(seed);
    return float(seed) * (1.0f / 4294967296.0f);
}

// A direction inside a cone around `axis`. Uniform on the spherical cap, not
// uniform in the angle: sampling the angle linearly bunches particles at the
// cone's axis, which makes a wide emitter look like a narrow one with a halo.
static inline float3 ConeDirection(float3 axis, float half_angle, thread uint& seed)
{
    const float cos_max = cos(half_angle);
    const float z = mix(cos_max, 1.0f, Rand01(seed));
    const float r = sqrt(max(0.0f, 1.0f - z * z));
    const float phi = Rand01(seed) * 6.2831853f;
    // Any basis perpendicular to the axis will do; picking the world axis the
    // emitter is LEAST aligned with is what keeps the cross product from
    // collapsing when someone emits straight up.
    const float3 up = abs(axis.y) > 0.99f ? float3(1, 0, 0) : float3(0, 1, 0);
    const float3 t = normalize(cross(up, axis));
    const float3 b = cross(axis, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + axis * z);
}

kernel void cs_particle_step(uint                       id        [[thread_position_in_grid]],
                             device GpuParticle*        particles [[buffer(0)]],
                             constant GpuParticleParams& p        [[buffer(1)]],
                             device atomic_uint*        spawned   [[buffer(2)]])
{
    if (id >= uint(p.limits.x)) return;

    GpuParticle q = particles[id];
    const float dt = p.motion.x;

    if (q.position.w > 0.0f) {
        // ALIVE: integrate, then age.
        //
        // Semi-implicit Euler, the same choice the rigid body solver makes and
        // for the same reason: taking the step with the NEW velocity does not
        // pump energy into the trajectory. It matters less here than there --
        // a particle lives two seconds -- but a fountain integrated explicitly
        // visibly climbs higher than it should.
        float3 v = q.velocity.xyz + p.gravity.xyz * dt;
        // Drag as a per-second fraction, applied as a ratio rather than
        // subtracted. Subtracting makes a fast particle reverse direction when
        // dt is large, which is how a spark ends up travelling backwards.
        v *= 1.0f / (1.0f + p.gravity.w * dt);
        q.velocity.xyz = v;
        q.position.xyz += v * dt;
        q.position.w -= dt;
        particles[id] = q;
        return;
    }

    // DEAD: this slot is a candidate for emission. The atomic is what limits
    // the rate -- every dead slot asks, and only the first `spawn` of them are
    // allowed, so the emission rate does not depend on how many happen to have
    // died this frame.
    const uint budget = uint(max(p.emit.z, 0.0f));
    if (budget == 0u) return;
    const uint ticket = atomic_fetch_add_explicit(spawned, 1u, memory_order_relaxed);
    if (ticket >= budget) return;

    // Seeded by the slot AND the frame. Without the frame the same slot emits
    // the same particle forever, and the fountain becomes a set of fixed arcs.
    uint seed = Pcg(id * 2654435761u + uint(p.emit.w) * 40503u);

    const float3 dir = ConeDirection(normalize(p.direction.xyz), p.origin.w, seed);
    const float speed = p.direction.w + (Rand01(seed) * 2.0f - 1.0f) * p.motion.y;
    const float life = max(0.01f, p.motion.z +
                                  (Rand01(seed) * 2.0f - 1.0f) * p.motion.w);
    const float size = max(1e-4f, p.emit.x +
                                  (Rand01(seed) * 2.0f - 1.0f) * p.emit.y);

    q.position = float4(p.origin.xyz, life);
    q.velocity = float4(dir * speed, size);
    q.color = p.color;
    q.birth = float4(life, float(seed) * (1.0f / 4294967296.0f), 0.0f, 0.0f);
    particles[id] = q;
}

struct ParticleOut {
    float4 position [[position]];
    float2 uv;        // -1..1 across the quad, so the radius is length(uv)
    float4 color;
    float  age;       // 0 at birth, 1 at death
    float  view_z;    // distance from the eye, for the soft fade
};

vertex ParticleOut vs_particle(uint                       vid       [[vertex_id]],
                               uint                       iid       [[instance_id]],
                               device const GpuParticle*  particles [[buffer(0)]],
                               constant FrameUniforms&    u         [[buffer(1)]])
{
    const GpuParticle q = particles[iid];
    ParticleOut o;

    // A dead particle is collapsed to a POINT rather than skipped: a vertex
    // shader cannot decline to run, and a degenerate triangle is rejected
    // before it rasterises anything. Placing it behind the near plane instead
    // would work too and costs a clip test per vertex.
    if (q.position.w <= 0.0f) {
        o.position = float4(0.0f, 0.0f, 0.0f, 1.0f);
        o.uv = float2(0.0f);
        o.color = float4(0.0f);
        o.age = 1.0f;
        o.view_z = 0.0f;
        return o;
    }

    // The quad's corner, from the vertex id. Two triangles' worth of indices
    // point at these four.
    const float2 corner = float2(float(vid & 1u) * 2.0f - 1.0f,
                                 float((vid >> 1u) & 1u) * 2.0f - 1.0f);

    // BILLBOARDED against the view, not against the camera's position. Taking
    // the basis from the view matrix's rows makes every particle exactly
    // parallel to the screen; aiming each one at the eye instead makes them
    // fan outward at the edges of a wide field of view, which reads as the
    // sprites leaning away from the centre.
    const float3 right = float3(u.viewProj[0][0], u.viewProj[1][0], u.viewProj[2][0]);
    const float3 up    = float3(u.viewProj[0][1], u.viewProj[1][1], u.viewProj[2][1]);
    const float3 r = normalize(right);
    const float3 v = normalize(up);

    const float size = q.velocity.w;
    const float3 world = q.position.xyz + (r * corner.x + v * corner.y) * size;

    o.position = u.viewProj * float4(world, 1.0f);
    o.uv = corner;
    o.color = q.color;
    o.age = saturate(1.0f - q.position.w / max(q.birth.x, 1e-4f));
    o.view_z = length(world - u.eyePos.xyz);
    return o;
}

fragment float4 fs_particle(ParticleOut in [[stage_in]],
                            constant FrameUniforms& u [[buffer(1)]],
                            depth2d<float> sceneDepth [[texture(0)]],
                            sampler        smp        [[sampler(0)]])
{
    // Radial falloff, squared. A hard-edged disc reads as a coin; the square
    // puts most of the brightness in the middle where a spark's light is.
    const float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0f) discard_fragment();
    float alpha = (1.0f - r2);
    alpha *= alpha;

    // FADE OUT over the particle's life, and in over its first tenth. Popping
    // into existence at full brightness is the single most obvious tell that a
    // particle system is a particle system.
    alpha *= smoothstep(0.0f, 0.1f, in.age) * (1.0f - in.age);

    // SOFT PARTICLES. Where the quad intersects solid geometry it otherwise
    // cuts a hard straight line across the sprite -- a billboard is flat and
    // the world is not. Fading it out as it approaches whatever is behind it
    // hides the intersection, and costs one depth sample.
    //
    // u.ssao.x is the reciprocal of the fade distance; zero disables it, which
    // is what a caller with no depth target to give gets.
    if (u.ssao.x > 0.0f) {
        const float2 uv = in.position.xy * float2(u.ssao.y, u.ssao.z);
        const float d = sceneDepth.sample(smp, uv);
        // Reversed-Z, so a LARGER stored value is nearer. Comparing the raw
        // values would fade the wrong side.
        if (d > 0.0f) {
            const float scene_z = u.ssao.w / max(d, 1e-6f);
            alpha *= saturate((scene_z - in.view_z) * u.ssao.x);
        }
    }

    // Premultiplied by alpha in the shader rather than in the blend state, so
    // the additive and alpha pipelines can share this fragment stage.
    return float4(in.color.rgb * in.color.a * alpha, in.color.a * alpha);
}
