#include "engine/asset/hotreload.h"

#include <algorithm>
#include <filesystem>
#include <system_error>

namespace eng::asset {
namespace {

// The modification time as a plain number, or 0 when the file is not there.
//
// The error_code overload, not the throwing one. A watched file being missing
// is the normal case here -- an editor writing through a temporary makes every
// file briefly absent -- and an exception per poll per missing file would be
// both slow and wrong.
std::int64_t ModifiedAt(const std::string& path, std::uintmax_t* out_size) {
    std::error_code ec;
    const auto time = std::filesystem::last_write_time(path, ec);
    if (ec) {
        if (out_size) *out_size = 0;
        return 0;
    }
    if (out_size) {
        std::error_code size_ec;
        const std::uintmax_t s = std::filesystem::file_size(path, size_ec);
        *out_size = size_ec ? 0 : s;
    }
    return time.time_since_epoch().count();
}

struct Entry {
    std::string path;
    bool live = false;
    bool exists = false;
    std::int64_t stamp = 0;
    std::uintmax_t size = 0;
    // Set when a change is first seen, cleared when it is reported. See
    // WatchConfig::settle_seconds.
    bool settling = false;
    float quiet_for = 0.0f;
    WatchEvent pending = WatchEvent::Modified;
};

}  // namespace

struct FileWatcher::Impl {
    WatchConfig config;
    std::vector<Entry> entries;
    float since_poll = 0.0f;
    bool force = false;
    int polls = 0;
    std::string empty;
};

FileWatcher::FileWatcher(const WatchConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}
FileWatcher::~FileWatcher() = default;

int FileWatcher::Watch(const std::string& path) {
    Impl& im = *impl_;
    for (std::size_t i = 0; i < im.entries.size(); ++i)
        if (im.entries[i].live && im.entries[i].path == path) return int(i);
    Entry e;
    e.path = path;
    e.live = true;
    e.stamp = ModifiedAt(path, &e.size);
    e.exists = e.stamp != 0;
    // Reused slots rather than growing forever: a caller watching and
    // unwatching per level would otherwise leak a slot per file per load, and
    // the handles it holds must stay stable for the ones still live.
    for (std::size_t i = 0; i < im.entries.size(); ++i)
        if (!im.entries[i].live) {
            im.entries[i] = e;
            return int(i);
        }
    im.entries.push_back(e);
    return int(im.entries.size()) - 1;
}

void FileWatcher::Unwatch(int handle) {
    if (handle < 0 || std::size_t(handle) >= impl_->entries.size()) return;
    impl_->entries[std::size_t(handle)].live = false;
}

int FileWatcher::Count() const {
    int n = 0;
    for (const Entry& e : impl_->entries)
        if (e.live) ++n;
    return n;
}

const std::string& FileWatcher::Path(int handle) const {
    if (handle < 0 || std::size_t(handle) >= impl_->entries.size())
        return impl_->empty;
    return impl_->entries[std::size_t(handle)].path;
}

void FileWatcher::PollNow() { impl_->force = true; }
int FileWatcher::PollCount() const { return impl_->polls; }

std::vector<FileWatcher::Change> FileWatcher::Poll(float dt) {
    Impl& im = *impl_;
    std::vector<Change> out;
    im.since_poll += dt;
    if (!im.force && im.since_poll < im.config.poll_seconds) {
        // The settle timers still advance, so a file that changed just before
        // the interval elapsed is not held up by a whole extra interval.
        for (Entry& e : im.entries)
            if (e.live && e.settling) e.quiet_for += dt;
        return out;
    }
    const float elapsed = im.since_poll;
    im.since_poll = 0.0f;
    im.force = false;
    ++im.polls;

    for (std::size_t i = 0; i < im.entries.size(); ++i) {
        Entry& e = im.entries[i];
        if (!e.live) continue;
        std::uintmax_t size = 0;
        const std::int64_t stamp = ModifiedAt(e.path, &size);
        const bool exists = stamp != 0;

        if (stamp != e.stamp || size != e.size || exists != e.exists) {
            // SOMETHING CHANGED. Not reported yet: the file may be mid-write,
            // and the settle timer below is what waits for it to finish.
            e.settling = true;
            e.quiet_for = 0.0f;
            e.pending = !exists ? WatchEvent::Removed
                                : (e.exists ? WatchEvent::Modified : WatchEvent::Created);
            e.stamp = stamp;
            e.size = size;
            e.exists = exists;
            continue;
        }
        if (!e.settling) continue;
        e.quiet_for += elapsed;
        if (e.quiet_for < im.config.settle_seconds) continue;
        // A removal is reported as soon as it settles; a modification is only
        // reported if the file is actually there, because an editor writing
        // through a temporary passes through "gone" on its way to "new".
        e.settling = false;
        if (e.pending == WatchEvent::Removed && e.exists) continue;
        out.push_back(Change{int(i), e.pending});
    }
    return out;
}

// -------------------------------------------------------------------- reload

struct HotReload::Impl {
    FileWatcher watcher;
    struct Item {
        Callback callback;
        bool live = false;
        // A callback that failed is retried on the next poll. See the header:
        // a shader with a syntax error is the commonest thing to reload, and
        // one that is silently not retried looks like the edit did nothing.
        bool retry = false;
    };
    std::vector<Item> items;
    int reloads = 0;
    int failures = 0;
    std::string last_error;

    explicit Impl(const WatchConfig& c) : watcher(c) {}
};

HotReload::HotReload(const WatchConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}
HotReload::~HotReload() = default;

int HotReload::Add(const std::string& path, Callback callback) {
    const int handle = impl_->watcher.Watch(path);
    if (handle < 0) return -1;
    if (std::size_t(handle) >= impl_->items.size())
        impl_->items.resize(std::size_t(handle) + 1);
    impl_->items[std::size_t(handle)].callback = std::move(callback);
    impl_->items[std::size_t(handle)].live = true;
    impl_->items[std::size_t(handle)].retry = false;
    return handle;
}

void HotReload::Remove(int handle) {
    impl_->watcher.Unwatch(handle);
    if (handle >= 0 && std::size_t(handle) < impl_->items.size())
        impl_->items[std::size_t(handle)].live = false;
}

void HotReload::PollNow() { impl_->watcher.PollNow(); }
int HotReload::ReloadCount() const { return impl_->reloads; }
int HotReload::FailureCount() const { return impl_->failures; }
const std::string& HotReload::LastError() const { return impl_->last_error; }

int HotReload::Update(float dt) {
    Impl& im = *impl_;
    const std::vector<FileWatcher::Change> changes = im.watcher.Poll(dt);
    int ran = 0;

    // Anything that failed last time is retried, whether or not it changed
    // again. The alternative -- waiting for another edit -- means a fix to a
    // DIFFERENT file that made this one compile never gets noticed.
    for (std::size_t i = 0; i < im.items.size(); ++i) {
        if (!im.items[i].live || !im.items[i].retry) continue;
        bool changed_too = false;
        for (const auto& c : changes)
            if (std::size_t(c.handle) == i) changed_too = true;
        if (changed_too) continue;  // handled below
        std::string error;
        const bool ok = im.items[i].callback(im.watcher.Path(int(i)), error);
        ++ran;
        if (ok) {
            im.items[i].retry = false;
            ++im.reloads;
        } else {
            ++im.failures;
            im.last_error = error;
        }
    }

    for (const FileWatcher::Change& c : changes) {
        if (c.handle < 0 || std::size_t(c.handle) >= im.items.size()) continue;
        Impl::Item& item = im.items[std::size_t(c.handle)];
        if (!item.live || !item.callback) continue;
        // A REMOVAL does not run the callback. There is nothing to load, and
        // running it on a missing file means every reload handler has to begin
        // by checking whether its file exists.
        if (c.event == WatchEvent::Removed) continue;
        std::string error;
        const bool ok = item.callback(im.watcher.Path(c.handle), error);
        ++ran;
        if (ok) {
            item.retry = false;
            ++im.reloads;
        } else {
            item.retry = true;
            ++im.failures;
            im.last_error = error;
        }
    }
    return ran;
}

}  // namespace eng::asset
