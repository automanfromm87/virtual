// The broadphase tree, and the thing it was built for.
//
// A spatial index has two ways to be wrong and they need different tests. It
// can MISS -- report fewer overlaps than exist -- which is a dropped collision
// and shows up as an object falling through the floor once every few minutes.
// Or it can be SLOW, reporting everything correctly by visiting every leaf,
// which no correctness test detects at all: the answers are perfect and the
// frame rate is the same as brute force.
//
// So every query here is checked against an exhaustive loop, and the tree's
// depth and the world's pair count are asserted as well.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "engine/physics/bvh.h"
#include "engine/physics/physics.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

// A deterministic generator, so a failure is reproducible. rand() is not: it is
// allowed to differ between platforms and libraries, and a spatial index test
// that only fails on one machine is close to useless.
struct Rng {
    std::uint32_t s = 0x12345678u;
    float Next() {
        s = s * 1664525u + 1013904223u;
        return float(s >> 8) / 16777216.0f;
    }
    float Range(float lo, float hi) { return lo + (hi - lo) * Next(); }
};

std::vector<eng::physics::Aabb> Scatter(int n, float spread, float size,
                                        std::uint32_t seed = 0x12345678u) {
    Rng rng{seed};
    // Braces: with parens this is the most vexing parse and declares a function.
    std::vector<eng::physics::Aabb> boxes{std::size_t(n), eng::physics::Aabb{}};
    for (int i = 0; i < n; ++i) {
        const eng::Vec3 c{rng.Range(-spread, spread), rng.Range(-spread, spread),
                          rng.Range(-spread, spread)};
        const float h = size * rng.Range(0.5f, 1.5f);
        boxes[std::size_t(i)].lo = c - eng::Vec3{h, h, h};
        boxes[std::size_t(i)].hi = c + eng::Vec3{h, h, h};
    }
    return boxes;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    using eng::physics::Aabb;
    using eng::physics::Bvh;

    {
        std::printf("box queries agree with an exhaustive scan\n");
        const std::vector<Aabb> boxes = Scatter(2000, 50.0f, 1.0f);
        Bvh tree;
        tree.Build(boxes);

        Rng rng{0xABCDEF01u};
        int mismatches = 0;
        long long tree_hits = 0, scanned = 0;
        for (int q = 0; q < 400; ++q) {
            const eng::Vec3 c{rng.Range(-55.0f, 55.0f), rng.Range(-55.0f, 55.0f),
                              rng.Range(-55.0f, 55.0f)};
            const float h = rng.Range(0.5f, 8.0f);
            Aabb probe;
            probe.lo = c - eng::Vec3{h, h, h};
            probe.hi = c + eng::Vec3{h, h, h};

            std::vector<char> from_tree(boxes.size(), 0), from_scan(boxes.size(), 0);
            tree.QueryBox(probe, [&](int i) { from_tree[std::size_t(i)] = 1; ++tree_hits; });
            for (std::size_t i = 0; i < boxes.size(); ++i) {
                if (boxes[i].Overlaps(probe)) from_scan[i] = 1;
                ++scanned;
            }
            if (from_tree != from_scan) ++mismatches;
        }
        Check(mismatches == 0, "400 random box queries match exactly");
        std::printf("    %lld leaves visited against %lld scanned (%.1fx fewer)\n",
                    tree_hits, scanned, double(scanned) / double(tree_hits));
    }

    {
        std::printf("\nray queries agree with an exhaustive scan\n");
        const std::vector<Aabb> boxes = Scatter(1500, 40.0f, 1.2f, 0x55AA33CCu);
        Bvh tree;
        tree.Build(boxes);

        Rng rng{0x0BADF00Du};
        int mismatches = 0;
        for (int q = 0; q < 400; ++q) {
            const eng::Vec3 o{rng.Range(-60.0f, 60.0f), rng.Range(-60.0f, 60.0f),
                              rng.Range(-60.0f, 60.0f)};
            eng::Vec3 d{rng.Range(-1.0f, 1.0f), rng.Range(-1.0f, 1.0f),
                        rng.Range(-1.0f, 1.0f)};
            if (eng::Dot(d, d) < 1e-6f) d = eng::Vec3{1, 0, 0};
            d = eng::Normalize(d);
            const float tmax = rng.Range(5.0f, 200.0f);
            const eng::Vec3 inv{1.0f / d.x, 1.0f / d.y, 1.0f / d.z};

            std::vector<char> from_tree(boxes.size(), 0), from_scan(boxes.size(), 0);
            tree.QueryRay(o, d, tmax, [&](int i) { from_tree[std::size_t(i)] = 1; });
            for (std::size_t i = 0; i < boxes.size(); ++i)
                if (boxes[i].Hits(o, inv, tmax)) from_scan[i] = 1;
            if (from_tree != from_scan) ++mismatches;
        }
        Check(mismatches == 0, "400 random rays match the slab test exactly");
    }

    {
        // AXIS-ALIGNED rays are the case that breaks a slab test: a zero
        // direction component makes the reciprocal infinite, and the product
        // with a zero numerator is NaN. A NaN that compares false in the wrong
        // direction turns into a missed hit, which is how a ray fired straight
        // down -- the single most common raycast in any game -- silently stops
        // finding the floor.
        std::printf("\naxis-aligned rays do not fall into the NaN trap\n");
        std::vector<Aabb> boxes(1);
        boxes[0].lo = eng::Vec3{-1.0f, -1.0f, -1.0f};
        boxes[0].hi = eng::Vec3{1.0f, 1.0f, 1.0f};
        Bvh tree;
        tree.Build(boxes);

        int found = 0;
        const eng::Vec3 dirs[6] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                   {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
        for (const eng::Vec3& d : dirs)
            tree.QueryRay(d * -5.0f, d, 100.0f, [&](int) { ++found; });
        Check(found == 6, "a ray along each of the six axes finds the box");

        // And the exact edge: an origin lying ON a face plane, where the 0*inf
        // product is generated for real.
        int edge = 0;
        tree.QueryRay(eng::Vec3{-5.0f, 1.0f, 0.0f}, eng::Vec3{1, 0, 0}, 100.0f,
                      [&](int) { ++edge; });
        Check(edge == 1, "and one whose origin lies exactly on a face plane");
    }

    {
        std::printf("\nthe tree is balanced, not a list\n");
        for (int n : {16, 256, 4096, 32768}) {
            const std::vector<Aabb> boxes = Scatter(n, 100.0f, 0.5f);
            Bvh tree;
            tree.Build(boxes);
            // A perfectly balanced tree over n items with leaves of four is
            // ceil(log2(n/4)) deep. Allowing +3 covers the ragged bottom that a
            // median split leaves when n is not a power of two.
            const int ideal = int(std::ceil(std::log2(double(n) / 4.0)));
            std::printf("    %6d boxes: depth %2d, ideal %2d, %d nodes\n", n,
                        tree.MaxDepth(), ideal, tree.NodeCount());
            if (tree.MaxDepth() > ideal + 3) {
                Check(false, "depth is within three of the ideal");
                break;
            }
        }
        Check(true, "depth stays within three of log2(n/4) up to 32k boxes");
    }

    {
        // A hundred crates spawned at the origin. Every centroid is identical,
        // so no split separates anything -- the recursion would go until the
        // depth cap and, with a smaller cap, until the stack ran out.
        std::printf("\nidentical centroids terminate instead of recursing\n");
        std::vector<Aabb> boxes(500);
        for (Aabb& b : boxes) {
            b.lo = eng::Vec3{-0.5f, -0.5f, -0.5f};
            b.hi = eng::Vec3{0.5f, 0.5f, 0.5f};
        }
        Bvh tree;
        tree.Build(boxes);
        int found = 0;
        Aabb probe;
        probe.lo = eng::Vec3{0, 0, 0};
        probe.hi = eng::Vec3{0, 0, 0};
        tree.QueryBox(probe, [&](int) { ++found; });
        std::printf("    depth %d, %d nodes\n", tree.MaxDepth(), tree.NodeCount());
        Check(found == 500, "all 500 coincident boxes are still reported");
        Check(tree.MaxDepth() < 48, "and the build stopped short of the depth cap");
    }

    {
        std::printf("\ndegenerate inputs\n");
        Bvh tree;
        tree.Build({});
        int n = 0;
        Aabb any;
        any.lo = eng::Vec3{-1e6f, -1e6f, -1e6f};
        any.hi = eng::Vec3{1e6f, 1e6f, 1e6f};
        tree.QueryBox(any, [&](int) { ++n; });
        tree.QueryRay(eng::Vec3{0, 0, 0}, eng::Vec3{1, 0, 0}, 1e6f, [&](int) { ++n; });
        Check(n == 0 && tree.Empty(), "an empty tree answers nothing and does not crash");

        std::vector<Aabb> one(1);
        one[0].lo = eng::Vec3{0, 0, 0};
        one[0].hi = eng::Vec3{1, 1, 1};
        tree.Build(one);
        n = 0;
        tree.QueryBox(one[0], [&](int) { ++n; });
        Check(n == 1 && tree.LeafCount() == 1, "a tree of one box works");
    }

    {
        // The tree must be a pure function of its input. If it is not, the
        // broadphase reports pairs in a different order between runs, the
        // solver applies impulses in a different order, and the simulation
        // stops being reproducible -- which makes every other physics test
        // flaky rather than failing outright.
        std::printf("\nthe build is deterministic\n");
        const std::vector<Aabb> boxes = Scatter(3000, 30.0f, 1.0f, 0x99u);
        Bvh a, b;
        a.Build(boxes);
        b.Build(boxes);
        std::vector<int> ra, rb;
        Aabb probe;
        probe.lo = eng::Vec3{-5, -5, -5};
        probe.hi = eng::Vec3{5, 5, 5};
        a.QueryBox(probe, [&](int i) { ra.push_back(i); });
        b.QueryBox(probe, [&](int i) { rb.push_back(i); });
        Check(ra == rb && !ra.empty(),
              "two builds of the same input answer in the same order");
        Check(a.NodeCount() == b.NodeCount() && a.MaxDepth() == b.MaxDepth(),
              "and produce the same tree");
    }

    {
        // THE PAYOFF, measured in the world rather than the tree.
        //
        // A CONSTANT-DENSITY grid, not a scatter. The first attempt at this
        // spread the bodies through a volume proportional to their count, which
        // sounds like constant density and is not: at that spacing almost
        // nothing overlapped, the pair counts came out as 9, 8 and 76, and the
        // ratio between two numbers that near zero is noise. The test was
        // measuring its own scene, not the broadphase.
        //
        // Here each body genuinely neighbours the next, so the property being
        // asserted is the real one: with O(1) neighbours per body the pair count
        // is O(N), which is what a working broadphase means. Brute force at the
        // largest size would be 800 pairs per body.
        std::printf("\nthe world's pair count stops being quadratic\n");
        bool linear = true;
        for (int side : {5, 8, 12}) {
            const int n = side * side * side;
            eng::physics::World world;
            Rng rng{0x2468u};
            for (int x = 0; x < side; ++x)
                for (int y = 0; y < side; ++y)
                    for (int z = 0; z < side; ++z) {
                        eng::physics::Body b;
                        b.shape = eng::physics::Shape::MakeSphere(0.4f);
                        // Jittered, so no two centroids coincide and the median
                        // split has something to separate.
                        b.position = eng::Vec3{float(x) * 0.85f + rng.Range(-0.02f, 0.02f),
                                               float(y) * 0.85f + rng.Range(-0.02f, 0.02f),
                                               float(z) * 0.85f + rng.Range(-0.02f, 0.02f)};
                        b.SetMass(1.0f);
                        world.Add(b);
                    }
            world.sleep_after = 0.0f;  // nothing sleeps, so nothing is skipped
            world.StepFixed();
            const int pairs = world.Stats().pairs_tested;
            const int brute = n * (n - 1) / 2;
            std::printf("    %5d bodies: %7d pairs (%.1f per body), brute force "
                        "%9d (%.0f per body), depth %d\n",
                        n, pairs, double(pairs) / n, brute, double(brute) / n,
                        world.Stats().bvh_depth);
            // PER BODY, which is the O(1) claim written down. A tree that had
            // degenerated into a list would still answer correctly and would
            // land near N/2 per body.
            if (double(pairs) / n > 20.0) linear = false;
        }
        Check(linear, "pairs per body stays bounded as the count grows 14x");
    }

    {
        std::printf("\nthe tree is not rebuilt more than it has to be\n");
        eng::physics::World world;
        for (int i = 0; i < 200; ++i) {
            eng::physics::Body b;
            b.shape = eng::physics::Shape::MakeSphere(0.3f);
            b.position = eng::Vec3{float(i % 20), 1.0f, float(i / 20)};
            b.SetMass(1.0f);
            world.Add(b);
        }
        world.StepFixed();
        const int after_step = world.Stats().bvh_rebuilds;
        // Ten queries with nothing moved in between must share one tree.
        eng::physics::RayHit hit;
        for (int i = 0; i < 10; ++i)
            (void)world.Raycast(eng::Vec3{0, 20, 0}, eng::Vec3{0, -1, 0}, 100.0f, &hit);
        std::vector<int> found;
        for (int i = 0; i < 10; ++i)
            (void)world.OverlapSphere(eng::Vec3{5, 1, 5}, 2.0f, &found);
        std::printf("    %d rebuilds after the step, %d after 20 more queries\n",
                    after_step, world.Stats().bvh_rebuilds);
        Check(world.Stats().bvh_rebuilds == after_step,
              "twenty queries after a step reuse the same tree");
    }

    std::printf(g_failures == 0 ? "\nbvh_test: all checks passed\n"
                                : "\nbvh_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
