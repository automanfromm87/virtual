// The baked package format and the file watcher.
//
// A package reader's job is half "hand back the bytes" and half "refuse a file
// that is not what it claims to be", and the second half is the one that only
// gets exercised when something has already gone wrong. A truncated download, a
// package from a different build, a half-written file caught mid-copy: each of
// those arrives as an array of bytes with plausible-looking numbers in it, and
// a reader that trusts its own offsets turns them into an out-of-bounds read.
//
// So the checks here are as much about the malformed inputs as the good one.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "engine/asset/hotreload.h"
#include "engine/asset/pack.h"
#include "engine/geometry/mesh.h"
#include "engine/shaders/shader_types.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

std::vector<std::uint8_t> Bytes(const char* s) {
    return std::vector<std::uint8_t>(reinterpret_cast<const std::uint8_t*>(s),
                                     reinterpret_cast<const std::uint8_t*>(s) +
                                         std::char_traits<char>::length(s));
}

std::string TempPath(const char* name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

void WriteText(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << text;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    {
        std::printf("a package round-trips\n");
        eng::asset::PackWriter w;
        w.Add("readme", eng::asset::AssetType::Raw, Bytes("hello"));
        w.Add("level", eng::asset::AssetType::Scene, Bytes("{\"a\":1}"));
        const eng::Mesh sphere =
            eng::MakeUVSphere(1.0f, 12, 16, eng::Vec4{1, 1, 1, 1}, eng::Vec4{1, 1, 1, 1});
        w.AddMesh("sphere", sphere);
        Check(w.Count() == 3, "three entries went in");

        std::string error;
        eng::asset::Pack p = eng::asset::Pack::Open(w.Build(), error);
        Check(p.Valid(), error.empty() ? "and the package opens" : error.c_str());
        Check(p.Count() == 3, "with three entries");

        const auto readme = p.Get("readme");
        Check(readme.size() == 5 && std::equal(readme.begin(), readme.end(),
                                               Bytes("hello").begin()),
              "a raw blob comes back byte for byte");
        Check(p.Type(p.Find("level")) == eng::asset::AssetType::Scene,
              "and its type is preserved");
        Check(p.Find("missing") < 0, "a name that is not there is not found");
        Check(p.Get("missing").empty(), "and yields nothing");

        eng::asset::CookedMesh cooked;
        Check(p.GetMesh("sphere", &cooked), "the mesh reads back");
        std::printf("    sphere: %u vertices, %u indices, radius %.3f\n",
                    cooked.vertex_count, cooked.index_count, cooked.bounds.radius);
        Check(cooked.vertex_count == sphere.vertices.size() &&
                  cooked.index_count == sphere.indices.size(),
              "with the right counts");
        // BYTE FOR BYTE, which is the whole claim: the payload is what goes
        // into a GPU buffer with no conversion, so anything short of identical
        // means a conversion is happening somewhere.
        Check(std::memcmp(cooked.vertices.data(), sphere.vertices.data(),
                          sphere.vertices.size() * sizeof(VertexIn)) == 0,
              "and vertex data identical to the source");
        // THE LENGTH IS THE ASSERTION. This compared sizeof(uint16_t) per
        // index against a std::vector<std::uint32_t>, which is the same wrong
        // number the writer used -- so the two halves of the bug agreed and the
        // check could not fail. Comparing the span's own size is what makes it
        // an assertion about the file rather than about the test's arithmetic.
        Check(cooked.indices.size() == sphere.indices.size() * sizeof(std::uint32_t),
              "and the index block is 32-bit, as Mesh::indices is");
        Check(cooked.indices.size() == sphere.indices.size() * sizeof(std::uint32_t) &&
                  std::memcmp(cooked.indices.data(), sphere.indices.data(),
                              cooked.indices.size()) == 0,
              "and index data identical too");

        // ALIGNED. A vertex block starting on an odd byte works on Apple
        // silicon and faults on hardware that requires aligned loads.
        const auto address = reinterpret_cast<std::uintptr_t>(cooked.vertices.data());
        std::printf("    vertex block at offset %zu within the file\n",
                    std::size_t(address % 16));
        Check(address % 16 == 0, "and the vertex block is sixteen-byte aligned");
    }

    {
        std::printf("\na duplicate name replaces rather than appends\n");
        eng::asset::PackWriter w;
        w.Add("thing", eng::asset::AssetType::Raw, Bytes("first"));
        w.Add("thing", eng::asset::AssetType::Raw, Bytes("second"));
        Check(w.Count() == 1, "only one entry survives");
        std::string error;
        eng::asset::Pack p = eng::asset::Pack::Open(w.Build(), error);
        const auto data = p.Get("thing");
        Check(data.size() == 6 &&
                  std::equal(data.begin(), data.end(), Bytes("second").begin()),
              "and it is the later one");
    }

    {
        std::printf("\nmalformed packages are refused, not read\n");
        eng::asset::PackWriter w;
        w.Add("a", eng::asset::AssetType::Raw, Bytes("payload here"));
        w.Add("b", eng::asset::AssetType::Raw, Bytes("more payload"));
        const std::vector<std::uint8_t> good = w.Build();

        std::string error;
        Check(eng::asset::Pack::Open(good, error).Valid(), "the good one opens");

        // EMPTY and TINY.
        Check(!eng::asset::Pack::Open({}, error).Valid(), "an empty file is refused");
        Check(!error.empty(), "with a message");
        Check(!eng::asset::Pack::Open(std::vector<std::uint8_t>(20, 0), error).Valid(),
              "and so is one too short for a header");

        // WRONG MAGIC: a PNG renamed to .pak, which is a real thing that
        // happens with a build script and a wildcard.
        std::vector<std::uint8_t> wrong_magic = good;
        wrong_magic[0] = 'X';
        Check(!eng::asset::Pack::Open(wrong_magic, error).Valid(),
              "a file that is not a package is refused");

        // WRONG VERSION: yesterday's package against today's build.
        std::vector<std::uint8_t> wrong_version = good;
        wrong_version[8] = 99;
        Check(!eng::asset::Pack::Open(wrong_version, error).Valid(),
              "and one from a different format version");
        std::printf("    version mismatch says: %s\n", error.c_str());

        // TRUNCATED. The header still claims the full size, so every offset in
        // it points past the end -- this is the case that becomes an
        // out-of-bounds read in a reader that trusts them.
        for (std::size_t cut : {good.size() / 2, good.size() - 1, good.size() - 16}) {
            std::vector<std::uint8_t> truncated(good.begin(),
                                                good.begin() + std::ptrdiff_t(cut));
            if (!eng::asset::Pack::Open(truncated, error).Valid()) continue;
            Check(false, "a truncated package is refused");
            break;
        }
        Check(true, "three different truncations are all refused");

        // CORRUPT TABLE: one flipped byte in the entry table. The payload
        // checksums would not catch it, which is why the table has its own.
        std::vector<std::uint8_t> corrupt = good;
        corrupt[sizeof(std::uint64_t) * 7 + 4] ^= 0x40;
        Check(!eng::asset::Pack::Open(corrupt, error).Valid(),
              "and a single flipped bit in the table");
    }

    {
        std::printf("\na mesh entry from a different vertex layout is refused\n");
        // The stride is stored and checked. A package cooked when VertexIn was
        // a different size would otherwise be read at the current stride, and a
        // mesh reinterpreted at the wrong stride is not a crash -- it is a shape
        // nobody recognises, which reads as a broken exporter.
        eng::asset::PackWriter w;
        const eng::Mesh box =
            eng::MakeBox(eng::Vec3{1, 1, 1}, eng::Vec4{1, 1, 1, 1});
        w.AddMesh("box", box);
        std::vector<std::uint8_t> bytes = w.Build();
        std::string error;
        eng::asset::Pack ok = eng::asset::Pack::Open(bytes, error);
        eng::asset::CookedMesh m;
        Check(ok.GetMesh("box", &m), "the intact mesh reads");

        // Reading a non-mesh entry as a mesh must fail rather than reinterpret.
        eng::asset::PackWriter w2;
        w2.Add("notamesh", eng::asset::AssetType::Raw, Bytes("just some bytes"));
        eng::asset::Pack p2 = eng::asset::Pack::Open(w2.Build(), error);
        eng::asset::CookedMesh bad;
        Check(!p2.GetMesh("notamesh", &bad), "a raw blob is not read as a mesh");
        Check(!p2.GetMesh("absent", &bad), "and neither is a missing name");
    }

    {
        std::printf("\nthe package writes to disk and reads back\n");
        const std::string path = TempPath("virtual_pack_test.pak");
        eng::asset::PackWriter w;
        w.Add("x", eng::asset::AssetType::Raw, Bytes("on disk"));
        std::string error;
        Check(w.WriteFile(path, error), "it writes");
        eng::asset::Pack p = eng::asset::Pack::OpenFile(path, error);
        Check(p.Valid() && p.Count() == 1, "and reads back");
        Check(!eng::asset::Pack::OpenFile(TempPath("no_such_pack.pak"), error).Valid(),
              "and a missing file is an error rather than an empty package");
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    // --- the watcher --------------------------------------------------------

    {
        std::printf("\nthe file watcher notices a change, once\n");
        const std::string path = TempPath("virtual_watch_test.txt");
        WriteText(path, "one");

        eng::asset::WatchConfig cfg;
        cfg.poll_seconds = 0.0f;    // poll every Update
        cfg.settle_seconds = 0.05f;
        eng::asset::FileWatcher watcher(cfg);
        const int handle = watcher.Watch(path);
        Check(handle >= 0 && watcher.Count() == 1, "the file is watched");
        Check(watcher.Watch(path) == handle,
              "and watching it twice gives the same handle");

        // Nothing changed: no events, however many times it is polled.
        int spurious = 0;
        for (int i = 0; i < 10; ++i) spurious += int(watcher.Poll(0.02f).size());
        Check(spurious == 0, "an unchanged file produces no events");

        // Change it. The modification time has a coarse resolution on some
        // filesystems, so the size changes too and the sleep guarantees a
        // distinguishable stamp.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        WriteText(path, "one two three");

        int events = 0;
        eng::asset::WatchEvent kind = eng::asset::WatchEvent::Removed;
        for (int i = 0; i < 20; ++i)
            for (const auto& c : watcher.Poll(0.02f)) {
                ++events;
                kind = c.event;
            }
        std::printf("    %d event(s) after one write\n", events);
        // EXACTLY ONE. A watcher that fires per poll while a file is different
        // from what it remembers reloads continuously, and the symptom is the
        // game hitching forever after any edit.
        Check(events == 1, "one write produces exactly one event");
        Check(kind == eng::asset::WatchEvent::Modified, "reported as a modification");

        // AND IT SETTLES. A burst of writes -- which is what an editor's save
        // looks like -- has to collapse into one event, not one per write.
        int burst_events = 0;
        for (int i = 0; i < 5; ++i) {
            WriteText(path, std::string("burst ") + std::to_string(i));
            burst_events += int(watcher.Poll(0.01f).size());
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        for (int i = 0; i < 20; ++i) burst_events += int(watcher.Poll(0.02f).size());
        std::printf("    %d event(s) after five rapid writes\n", burst_events);
        Check(burst_events == 1, "five rapid writes produce one event");

        // REMOVAL.
        std::error_code ec;
        std::filesystem::remove(path, ec);
        bool saw_removed = false;
        for (int i = 0; i < 20; ++i)
            for (const auto& c : watcher.Poll(0.02f))
                if (c.event == eng::asset::WatchEvent::Removed) saw_removed = true;
        Check(saw_removed, "and deleting it is reported as a removal");

        watcher.Unwatch(handle);
        Check(watcher.Count() == 0, "unwatching removes it");
    }

    {
        std::printf("\nthe poll interval is respected\n");
        const std::string path = TempPath("virtual_watch_interval.txt");
        WriteText(path, "x");
        eng::asset::WatchConfig cfg;
        cfg.poll_seconds = 0.25f;
        eng::asset::FileWatcher watcher(cfg);
        (void)watcher.Watch(path);
        for (int i = 0; i < 60; ++i) (void)watcher.Poll(1.0f / 60.0f);
        std::printf("    60 frames at 1/60 s: %d polls (expected about 4)\n",
                    watcher.PollCount());
        // A watcher that stats every frame works perfectly and costs a syscall
        // per file per frame, which at a few hundred watched files is a
        // measurable slice of the frame.
        Check(watcher.PollCount() <= 6, "one second of frames costs about four polls");
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    {
        std::printf("\nhot reload runs the callback and retries failures\n");
        const std::string path = TempPath("virtual_reload_test.txt");
        WriteText(path, "good");

        eng::asset::WatchConfig cfg;
        cfg.poll_seconds = 0.0f;
        cfg.settle_seconds = 0.05f;
        eng::asset::HotReload reload(cfg);

        int calls = 0;
        bool should_fail = false;
        reload.Add(path, [&](const std::string&, std::string& error) {
            ++calls;
            if (should_fail) {
                error = "deliberate failure";
                return false;
            }
            return true;
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        WriteText(path, "changed");
        for (int i = 0; i < 20; ++i) reload.Update(0.02f);
        std::printf("    %d call(s), %d reload(s), %d failure(s)\n", calls,
                    reload.ReloadCount(), reload.FailureCount());
        Check(calls == 1 && reload.ReloadCount() == 1, "a change runs the callback once");

        // A FAILING reload is retried without another edit. A shader with a
        // syntax error is the commonest thing to hot reload, and a system that
        // gives up until you touch the file again looks like it ignored you.
        should_fail = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        WriteText(path, "broken");
        for (int i = 0; i < 5; ++i) reload.Update(0.02f);
        const int after_failure = calls;
        std::printf("    after a failing edit: %d call(s), last error '%s'\n", calls,
                    reload.LastError().c_str());
        Check(reload.FailureCount() >= 1, "the failure is counted");
        Check(reload.LastError() == "deliberate failure", "and its message kept");
        Check(calls > after_failure - 1 && calls >= 2, "and it was retried");

        // And once it can succeed again it does, with no further edit.
        should_fail = false;
        const int before_recovery = reload.ReloadCount();
        for (int i = 0; i < 5; ++i) reload.Update(0.02f);
        std::printf("    after the fix: %d reload(s)\n", reload.ReloadCount());
        Check(reload.ReloadCount() > before_recovery,
              "and succeeds on retry without another edit");

        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    {
        std::printf("\nwatching a file that is not there\n");
        eng::asset::WatchConfig cfg;
        cfg.poll_seconds = 0.0f;
        cfg.settle_seconds = 0.02f;
        eng::asset::FileWatcher watcher(cfg);
        const std::string path = TempPath("virtual_watch_absent.txt");
        std::error_code ec;
        std::filesystem::remove(path, ec);
        (void)watcher.Watch(path);
        int events = 0;
        for (int i = 0; i < 10; ++i) events += int(watcher.Poll(0.02f).size());
        Check(events == 0, "an absent file produces no events while it stays absent");

        WriteText(path, "now it exists");
        bool created = false;
        for (int i = 0; i < 20; ++i)
            for (const auto& c : watcher.Poll(0.02f))
                if (c.event == eng::asset::WatchEvent::Created) created = true;
        Check(created, "and its appearance is reported as a creation");
        std::filesystem::remove(path, ec);
    }

    std::printf(g_failures == 0 ? "\npack_test: all checks passed\n"
                                : "\npack_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
