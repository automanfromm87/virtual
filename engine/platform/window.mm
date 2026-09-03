// Objective-C++. The AppKit boundary: NSWindow, NSView and the event pump.
// The only package that links AppKit.
#import <AppKit/AppKit.h>
#import <QuartzCore/CoreAnimation.h>

#include "engine/platform/window.h"

namespace eng::platform {
namespace {

constexpr unsigned short kKeyCodeEscape = 53;

}  // namespace

struct Window::Impl {
    NSWindow* window = nil;
    NSView* view = nil;
    bool quit = false;
    Input input;
    bool held[128] = {};
};

Window::Window() : impl_(std::make_unique<Impl>()) {}
Window::~Window() = default;

std::unique_ptr<Window> Window::Create(const char* title, int width, int height,
                                       std::string& error) {
    if (width <= 0 || height <= 0) {
        error = "width and height must both be positive";
        return nullptr;
    }

    [NSApplication sharedApplication];
    // Without this a binary launched from a terminal (no .app bundle) gets no
    // dock icon and never takes focus.
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    const NSRect frame = NSMakeRect(0, 0, width, height);
    NSWindow* window = [[NSWindow alloc]
        initWithContentRect:frame
                  styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                             NSWindowStyleMaskMiniaturizable |
                             NSWindowStyleMaskResizable)
                    backing:NSBackingStoreBuffered
                      defer:NO];
    if (!window) {
        error = "NSWindow creation failed";
        return nullptr;
    }
    window.title = [NSString stringWithUTF8String:title];
    [window center];

    NSView* view = [[NSView alloc] initWithFrame:frame];
    window.contentView = view;

    [window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    // Required when driving the loop ourselves instead of calling -run.
    [NSApp finishLaunching];

    std::unique_ptr<Window> w(new Window());
    w->impl_->window = window;
    w->impl_->view = view;
    return w;
}

void Window::HostLayer(void* native_layer) {
    CALayer* layer = (__bridge CALayer*)native_layer;
    layer.contentsScale = impl_->window.backingScaleFactor;
    // Layer-HOSTING view: assign the layer first, THEN set wantsLayer. The
    // other order gives you a layer-BACKED view and AppKit replaces ours.
    impl_->view.layer = layer;
    impl_->view.wantsLayer = YES;
}

bool Window::PumpEvents() {
    for (;;) {
        NSEvent* e = [NSApp nextEventMatchingMask:NSEventMaskAny
                                        untilDate:[NSDate distantPast]
                                           inMode:NSDefaultRunLoopMode
                                          dequeue:YES];
        if (!e) break;
        switch (e.type) {
            case NSEventTypeKeyDown:
                if (e.keyCode == kKeyCodeEscape) {
                    impl_->quit = true;
                } else if (e.characters.length > 0) {
                    const int c = tolower([e.characters characterAtIndex:0]);
                    // Key repeat would spam the toggle list, but it must still
                    // refresh the held state.
                    if (!e.isARepeat) impl_->input.keys += char(c);
                    if (c >= 0 && c < 128) impl_->held[c] = true;
                }
                break;
            case NSEventTypeKeyUp:
                if (e.characters.length > 0) {
                    const int c = tolower([e.characters characterAtIndex:0]);
                    if (c >= 0 && c < 128) impl_->held[c] = false;
                }
                break;
            case NSEventTypeLeftMouseDragged:
                impl_->input.drag_dx += float(e.deltaX);
                impl_->input.drag_dy += float(e.deltaY);
                break;
            case NSEventTypeScrollWheel:
                // Trackpads report precise deltas an order of magnitude larger
                // than a notched wheel; normalise so both feel the same.
                impl_->input.scroll +=
                    float(e.scrollingDeltaY) * (e.hasPreciseScrollingDeltas ? 0.1f : 1.0f);
                break;
            default:
                break;
        }
        [NSApp sendEvent:e];
    }
    if (!impl_->window.isVisible) impl_->quit = true;
    if (impl_->quit) [impl_->window orderOut:nil];
    return !impl_->quit;
}

Input Window::TakeInput() {
    Input out = impl_->input;
    impl_->input = Input{};
    return out;
}

bool Window::IsKeyDown(char c) const {
    const unsigned char u = (unsigned char)c;
    return u < 128 && impl_->held[u];
}

float Window::BackingScale() const {
    return float(impl_->window.backingScaleFactor);
}

int Window::FramebufferWidth() const {
    return int(impl_->view.bounds.size.width * impl_->window.backingScaleFactor);
}

int Window::FramebufferHeight() const {
    return int(impl_->view.bounds.size.height * impl_->window.backingScaleFactor);
}

}  // namespace eng::platform
