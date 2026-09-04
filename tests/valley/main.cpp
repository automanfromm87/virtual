// Pure C++20 plus POSIX spawn. A GATE ON THE ONE PIECE OF INTEGRATED CODE.
//
// apps/world is 2,600 lines of viewer.cpp and 800 of districts.h, and until
// this file existed `bazel test //...` compiled none of it. That is exactly
// backwards: it is the most integrated code in the repository and had the least
// coverage, and every bug the last dozen commits found was found by a person
// looking at it rather than by a test.
//
// WHY IT DRIVES THE BINARY instead of linking a library. The obvious design is
// to extract the scene assembly into //apps/world:valley and test that. It was
// tried in this repository already: tests/world/world_scene.h says in its own
// comment "shared so the windowed demo and the offscreen gate simulate the same
// world", and the windowed demo it was shared with no longer exists -- the
// extracted copy outlived the thing it was extracted from. More decisively, the
// three defects this gate exists to catch were not in the scene assembly at
// all. One was in the shadow pass, one was in the frame loop's interaction with
// the device, and one only appears at a content scale the app alone reaches.
// None would have been caught by a library test, and all three are caught here.
//
// So the subject is the real binary, in the configurations that broke, asserted
// on the two things it already reports: its capture log and its PNG.
#include "engine/asset/png.h"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

extern char** environ;

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

// What one run of the app reported about itself.
struct Run {
    bool exited = false;   // false means it had to be killed: a hang
    int status = -1;
    int peak_in_flight = 0;
    int faults = -1;
    int dropped = -1;
    int instances = 0;
    int draws = 0;
    int shadow_draws = 0;
    int shadow_culled = 0;
    int slices = -1;
    double mean = -1.0;  // mean channel value of the capture, HUD excluded
    std::string log;
};

// The demo's HUD is drawn into the top-left of the frame and prints the very
// counters being asserted, so a brightness comparison that included it would be
// comparing text. The panel is 500 px wide and the longest hint line overflows
// it, so the whole band goes.
constexpr int kHudBandHeight = 300;

double MeanOutsideHud(const eng::Texture2D& img) {
    if (img.Empty() || img.height <= kHudBandHeight) return -1.0;
    double sum = 0.0;
    std::size_t n = 0;
    for (int y = kHudBandHeight; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) {
            const std::size_t o = (std::size_t(y) * img.width + x) * 4;
            sum += img.rgba[o] + img.rgba[o + 1] + img.rgba[o + 2];
            n += 3;
        }
    return n ? sum / double(n) : -1.0;
}

const char* WorldBinary() {
    // bazel runs a test from its runfiles root, so the binary a `data` dep put
    // there is at its label path. The second candidate is for running the test
    // straight out of bazel-bin by hand, which is how it gets debugged.
    static const char* kCandidates[] = {"apps/world/world",
                                        "bazel-bin/apps/world/world",
                                        "../../apps/world/world"};
    for (const char* c : kCandidates)
        if (access(c, X_OK) == 0) return c;
    return nullptr;
}

// Runs the app and returns what it said. KILLED after `seconds` rather than
// waited on, because one of the things under test is a deadlock -- the GI
// re-bake used to leak a frames-in-flight permit per bake and block forever on
// the third, and a gate that hangs on that failure reports nothing at all.
Run RunWorld(const std::string& tag, std::vector<std::string> args,
             int seconds = 120) {
    Run r;
    const char* bin = WorldBinary();
    if (!bin) {
        std::fprintf(stderr, "FAIL: cannot find the world binary\n");
        return r;
    }
    const std::string shot = "/tmp/valley_gate_" + tag + ".png";
    const std::string out = "/tmp/valley_gate_" + tag + ".log";
    std::remove(shot.c_str());

    std::vector<std::string> argv{bin, "--shot", shot, "--nosparks"};
    for (std::string& a : args) argv.push_back(std::move(a));
    std::vector<char*> raw;
    for (std::string& a : argv) raw.push_back(a.data());
    raw.push_back(nullptr);

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, STDOUT_FILENO, out.c_str(),
                                     O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&fa, STDOUT_FILENO, STDERR_FILENO);
    pid_t pid = 0;
    const int rc = posix_spawn(&pid, bin, &fa, nullptr, raw.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: spawn %s: %s\n", bin, std::strerror(rc));
        return r;
    }

    for (int i = 0; i < seconds * 20; ++i) {
        int status = 0;
        const pid_t done = waitpid(pid, &status, WNOHANG);
        if (done == pid) {
            r.exited = true;
            r.status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!r.exited) {
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
    }

    if (std::FILE* f = std::fopen(out.c_str(), "rb")) {
        char buf[4096];
        std::size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) r.log.append(buf, n);
        std::fclose(f);
    }
    const std::size_t at = r.log.rfind("summary:");
    if (at != std::string::npos)
        std::sscanf(r.log.c_str() + at,
                    "summary: peak in flight %d, gpu faults %d, dropped %d, "
                    "instances %d, draws %d, shadow %d/%d culled, slices %d",
                    &r.peak_in_flight, &r.faults, &r.dropped, &r.instances,
                    &r.draws, &r.shadow_draws, &r.shadow_culled, &r.slices);

    std::string error;
    const eng::Texture2D img = eng::png::DecodeFile(shot, error);
    r.mean = MeanOutsideHud(img);
    return r;
}

}  // namespace

int main() {
    if (!WorldBinary()) {
        std::fprintf(stderr,
                     "FAIL: the world binary is not in the runfiles. cwd=%s\n",
                     getcwd(nullptr, 0));
        return 1;
    }

    // --- the valley renders at all ------------------------------------------
    std::printf("the default valley\n");
    const Run base = RunWorld("base", {"--frames", "90"});
    std::printf("    exit %d, %d instances, %d draws, shadow %d drawn / %d culled, "
                "mean %.2f\n",
                base.status, base.instances, base.draws, base.shadow_draws,
                base.shadow_culled, base.mean);
    Check(base.exited && base.status == 0, "the demo runs headless and exits clean");
    Check(base.instances > 250 && base.draws > 100,
          "it builds a valley and draws it");
    Check(base.mean > 30.0 && base.mean < 140.0,
          "the frame is neither black nor blown out");
    Check(base.dropped == 0, "nothing was dropped for want of a uniform slice");
    Check(base.faults == 0, "the GPU reported no fault");
    Check(base.peak_in_flight > 0 && base.peak_in_flight <= 3,
          "no more frames were in flight than the ring has slots");
    // THE DEPTH-ONLY PASSES TAKE ONE SLICE EACH, however many casters they
    // draw. Before the DepthPass/DepthDraw split they took one per caster per
    // cascade, which is 478 + 164 of the frame's 806. Six slices of slack in
    // the bound, for the fullscreen passes and the UI.
    // AGAINST THE SHADOW DRAW COUNT, not against a constant. The claim is that
    // the depth-only passes cost one slice per PASS rather than one per draw,
    // and comparing to shadow_draws states exactly that: 478 casters go into
    // the shadow atlas and the whole frame takes about 174 slices. A constant
    // bound would have to be renumbered every time a fullscreen pass is added,
    // and one was -- bloom is five more.
    std::printf("    %d ring slices for %d draws + %d shadow draws\n", base.slices,
                base.draws, base.shadow_draws);
    Check(base.slices > 0 && base.slices < base.shadow_draws,
          "the shadow and depth passes cost one uniform slice each, not one per draw");

    // --- the shadow pass culls ----------------------------------------------
    //
    // WITHOUT the per-cascade frustum test this is 3 x instances, because the
    // instance loop sits inside the cascade loop and rejected nothing. The
    // assertion is the inequality, not a magic number: it fails the moment the
    // cull is removed and survives the valley's content changing.
    std::printf("the shadow pass culls per cascade\n");
    std::printf("    %d casters submitted, %d rejected, against %d x 3 cascades\n",
                base.shadow_draws, base.shadow_culled, base.instances);
    Check(base.shadow_culled > 0, "some casters are outside some cascade");
    Check(base.shadow_draws + base.shadow_culled >= base.instances,
          "every caster was considered at least once");
    Check(base.shadow_draws < 3 * base.instances,
          "and not every caster went into every cascade");

    // --- the crowd, which is what exposed all of this -----------------------
    std::printf("--crowd 600, the GPU-driven showcase\n");
    const Run c600 = RunWorld("c600", {"--frames", "90", "--crowd", "600"});
    std::printf("    %d instances, %d draws, shadow %d drawn / %d culled, mean %.2f\n",
                c600.instances, c600.draws, c600.shadow_draws, c600.shadow_culled,
                c600.mean);
    Check(c600.exited && c600.status == 0, "600 boulders still runs clean");
    Check(c600.instances == base.instances + 600, "and all 600 are in the scene");
    Check(c600.draws > base.draws, "and some of them are visible");
    Check(c600.dropped == 0, "nothing dropped");
    // The uncultured cost is 1800 -- 600 boulders in each of three cascades.
    Check(c600.shadow_draws - base.shadow_draws < 1800,
          "the boulders do not all reach all three cascades");

    // --- the cliff that used to be a black frame ----------------------------
    //
    // At --crowd 2000 the unculled shadow pass drained all 8192 slices of the
    // shared uniform ring before the scene pass got one, and the frame came out
    // black: mean 7.35 against 77.9, with no error anywhere. This is the
    // assertion that turns red if the cull is reverted.
    std::printf("--crowd 2000, which used to render black\n");
    const Run c2000 = RunWorld("c2000", {"--frames", "90", "--crowd", "2000"});
    std::printf("    %d instances, %d draws, %d dropped, mean %.2f against base %.2f\n",
                c2000.instances, c2000.draws, c2000.dropped, c2000.mean, base.mean);
    Check(c2000.exited && c2000.status == 0, "2000 boulders runs clean");
    Check(c2000.dropped == 0, "the shared ring still had slices for every pass");
    Check(c2000.draws > 0, "the scene pass drew something");
    Check(c2000.mean > base.mean * 0.5,
          "and the frame is lit, not the black one the ring used to produce");

    // --- the headroom the split bought --------------------------------------
    //
    // 6000 boulders is 12,010 shadow draws. Before the split that was 12,010
    // ring slices against a ceiling of 8192 and the frame was black; it is 2703
    // now, because the depth-only passes stopped paying per caster.
    std::printf("--crowd 6000, which the ring could not hold at all\n");
    const Run c6000 = RunWorld("c6000", {"--frames", "60", "--crowd", "6000"});
    std::printf("    %d instances, %d shadow draws, %d slices, %d dropped, mean %.2f\n",
                c6000.instances, c6000.shadow_draws, c6000.slices, c6000.dropped,
                c6000.mean);
    Check(c6000.exited && c6000.status == 0, "6000 boulders runs clean");
    Check(c6000.dropped == 0 && c6000.mean > base.mean * 0.5,
          "12,010 shadow draws fit in a ring of 8192, and the frame is lit");

    // --- the GPU-driven path ------------------------------------------------
    std::printf("--indirect\n");
    const Run ind = RunWorld("indirect", {"--frames", "90", "--crowd", "600",
                                          "--indirect"});
    std::printf("    %d instances, mean %.2f against forward %.2f\n", ind.instances,
                ind.mean, c600.mean);
    Check(ind.exited && ind.status == 0, "the indirect path runs clean");
    Check(ind.dropped == 0, "nothing dropped");
    Check(ind.mean > 0.0 && std::fabs(ind.mean - c600.mean) < c600.mean * 0.10,
          "and draws the same valley as the ordinary path, to a tenth");

    // --- the re-bake, which used to deadlock on the third one ---------------
    //
    // bake_indirect opens a device frame of its own. Called from inside the
    // app's frame it took a second frames-in-flight permit and replaced the
    // command buffer, which was then never committed -- so the permit never
    // came back and the third bake blocked forever on a semaphore. Four bakes
    // here, one more than kFramesInFlight, so a leak cannot hide.
    std::printf("the sun moves, and the GI volume is re-baked\n");
    // A SHORTER LEASH than the others. A clean run of this takes about two
    // seconds; when the deadlock is present the process must be killed, and the
    // wait is dead time in every future run of the suite.
    const Run bake = RunWorld("rebake", {"--frames", "100", "--rebake", "20"}, 60);
    const int bakes = [&] {
        int n = 0;
        for (std::size_t i = bake.log.find("baking indirect light");
             i != std::string::npos;
             i = bake.log.find("baking indirect light", i + 1))
            ++n;
        return n;
    }();
    std::printf("    exit %d, %d bakes, peak in flight %d\n", bake.status, bakes,
                bake.peak_in_flight);
    Check(bake.exited, "four re-bakes do not deadlock the app");
    Check(bake.status == 0, "and it exits clean");
    Check(bakes >= 5, "the bakes actually ran (one at startup, four forced)");
    Check(bake.peak_in_flight > 0 && bake.peak_in_flight <= 3,
          "and not one frames-in-flight permit leaked");

    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
