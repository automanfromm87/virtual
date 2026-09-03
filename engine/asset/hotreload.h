// Hot reload: noticing that a file on disk changed, and doing something about
// it without restarting.
//
// WHY IT IS WORTH THE TROUBLE. Every change to a shader, a texture, a level or
// a tuning value currently costs a rebuild and a relaunch, which is somewhere
// between five and thirty seconds. That is not a productivity figure, it is a
// behavioural one: past a few seconds people stop trying things. The difference
// between "change a number and watch" and "change a number, rebuild, relaunch,
// walk back to where you were, watch" is the difference between tuning
// something and guessing at it.
//
// POLLED MODIFICATION TIMES, not a file-system event API.
//
// FSEvents on macOS and inotify on Linux report changes without polling, which
// is genuinely better -- and each is a different API with different semantics,
// each needs a thread and a queue, and both deliver events for a file that is
// half-written. Polling a few dozen paths four times a second costs a stat per
// path, which is microseconds, and is the same code everywhere. The right time
// to switch is when the watch list is thousands of files, which is a content
// pipeline's problem and not an engine's.
//
// THE HALF-WRITTEN FILE is the part everyone gets wrong. An editor saving a file
// truncates it, writes it, and closes it, and a watcher that fires on the first
// of those hands the game an empty file. Worse, some editors write a temporary
// and rename it, so the file briefly does not exist at all. Waiting for the
// modification time to STOP changing -- see `settle_seconds` -- is what turns a
// burst of writes into one reload of a complete file.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace eng::asset {

struct WatchConfig {
    // How often the paths are stat-ed. Four times a second is imperceptible as
    // a delay and costs nothing; ten times is also fine and is not better.
    float poll_seconds = 0.25f;
    // How long a file has to stop changing before it counts as saved. Must
    // exceed the time an editor takes between its truncate and its close --
    // 0.15 s covers every editor that writes in place, and the rename-based
    // ones are atomic and need none of it.
    float settle_seconds = 0.15f;
};

// What happened to a watched path.
enum class WatchEvent : std::uint8_t {
    Modified,
    // The file was there when it was registered and is not now. Reported rather
    // than treated as an error: a caller usually wants to keep the last good
    // version, and a deleted file is often a rename in progress.
    Removed,
    // It came back.
    Created,
};

class FileWatcher {
  public:
    explicit FileWatcher(const WatchConfig& = {});
    ~FileWatcher();
    FileWatcher(const FileWatcher&) = delete;
    FileWatcher& operator=(const FileWatcher&) = delete;

    // Registers a path and returns a handle. Watching the same path twice
    // returns the same handle -- two watchers on one file would each fire, and
    // a caller that registered a shared dependency twice would reload it twice.
    int Watch(const std::string& path);
    void Unwatch(int handle);
    [[nodiscard]] int Count() const;
    [[nodiscard]] const std::string& Path(int handle) const;

    // Advances by `dt` and reports what settled. Nothing is reported until a
    // file has been quiet for `settle_seconds`, so a burst of writes produces
    // one event.
    struct Change {
        int handle = -1;
        WatchEvent event = WatchEvent::Modified;
    };
    std::vector<Change> Poll(float dt);

    // Forces the next Poll to re-stat immediately rather than waiting out the
    // interval. For a test, and for a caller that just wrote a file itself.
    void PollNow();

    // How many times the paths have actually been stat-ed. A test's way of
    // checking that the interval is doing something: a watcher that stats every
    // frame works perfectly and costs a syscall per file per frame.
    [[nodiscard]] int PollCount() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// A watcher plus what to do about each file.
//
// The callback returns whether the reload SUCCEEDED. A failure is kept and
// retried on the next change rather than swallowed -- a shader with a syntax
// error in it is the single most common thing to hot reload, and a system that
// silently keeps the old one leaves you wondering why your edit did nothing.
class HotReload {
  public:
    using Callback = std::function<bool(const std::string& path, std::string& error)>;

    explicit HotReload(const WatchConfig& = {});
    ~HotReload();
    HotReload(const HotReload&) = delete;
    HotReload& operator=(const HotReload&) = delete;

    int Add(const std::string& path, Callback);
    void Remove(int handle);

    // Polls and runs whatever fired. Returns how many callbacks ran.
    int Update(float dt);
    void PollNow();

    [[nodiscard]] int ReloadCount() const;
    [[nodiscard]] int FailureCount() const;
    // The last error, for putting on screen. An editor loop that prints its
    // failures to a log nobody is watching is an editor loop that appears to
    // ignore your edits.
    [[nodiscard]] const std::string& LastError() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng::asset
