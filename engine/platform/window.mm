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
    bool mouse_held[int(MouseButton::Count)] = {};
    bool cursor_locked = false;
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

    // Without this AppKit never delivers mouseMoved at all, and unheld mouse
    // motion is silently always zero -- which reads as "the mouse does not
    // work" and is impossible to find by looking at the event handler, because
    // the handler is correct and simply never runs.
    window.acceptsMouseMovedEvents = YES;
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
                impl_->input.mouse_dx += float(e.deltaX);
                impl_->input.mouse_dy += float(e.deltaY);
                break;
            case NSEventTypeMouseMoved:
            case NSEventTypeRightMouseDragged:
            case NSEventTypeOtherMouseDragged:
                // deltaX/deltaY, NOT the difference of two positions. Under a
                // locked cursor the position is pinned to the centre and never
                // changes, so differencing it reports no motion at all -- which
                // is precisely the case a first-person camera is built on.
                impl_->input.mouse_dx += float(e.deltaX);
                impl_->input.mouse_dy += float(e.deltaY);
                break;
            case NSEventTypeLeftMouseDown:
                impl_->input.pressed[int(MouseButton::Left)] = true;
                impl_->mouse_held[int(MouseButton::Left)] = true;
                break;
            case NSEventTypeLeftMouseUp:
                impl_->input.released[int(MouseButton::Left)] = true;
                impl_->mouse_held[int(MouseButton::Left)] = false;
                break;
            case NSEventTypeRightMouseDown:
                impl_->input.pressed[int(MouseButton::Right)] = true;
                impl_->mouse_held[int(MouseButton::Right)] = true;
                break;
            case NSEventTypeRightMouseUp:
                impl_->input.released[int(MouseButton::Right)] = true;
                impl_->mouse_held[int(MouseButton::Right)] = false;
                break;
            case NSEventTypeOtherMouseDown:
                if (e.buttonNumber == 2) {
                    impl_->input.pressed[int(MouseButton::Middle)] = true;
                    impl_->mouse_held[int(MouseButton::Middle)] = true;
                }
                break;
            case NSEventTypeOtherMouseUp:
                if (e.buttonNumber == 2) {
                    impl_->input.released[int(MouseButton::Middle)] = true;
                    impl_->mouse_held[int(MouseButton::Middle)] = false;
                }
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
    // The cursor's POSITION is read once per pump rather than accumulated from
    // events: a frame with no motion still needs an answer, and mouseMoved
    // events do not arrive at all while a button is held.
    {
        const NSPoint p = [impl_->window mouseLocationOutsideOfEventStream];
        const NSRect bounds = impl_->view.bounds;
        const float scale = float(impl_->window.backingScaleFactor);
        impl_->input.mouse_inside = NSPointInRect(p, bounds);
        // AppKit's origin is BOTTOM left and in points. The framebuffer's is
        // top left and in pixels, and every caller wants the second -- so the
        // flip and the scale happen here, once.
        impl_->input.mouse_x = float(p.x) * scale;
        impl_->input.mouse_y = float(bounds.size.height - p.y) * scale;
    }
    if (impl_->cursor_locked) {
        // Back to the centre every frame, so motion never runs out of screen.
        const NSRect frame = impl_->window.frame;
        const NSRect screen = impl_->window.screen.frame;
        const CGPoint centre =
            CGPointMake(frame.origin.x + frame.size.width * 0.5,
                        screen.size.height - (frame.origin.y + frame.size.height * 0.5));
        CGWarpMouseCursorPosition(centre);
        // Without this the warp itself is reported as motion on the next
        // event, and the camera snaps back as fast as it is turned.
        CGAssociateMouseAndMouseCursorPosition(true);
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

bool Window::IsMouseDown(MouseButton b) const {
    const int i = int(b);
    if (i < 0 || i >= int(MouseButton::Count)) return false;
    return impl_->mouse_held[i];
}

void Window::SetCursorLocked(bool locked) {
    if (locked == impl_->cursor_locked) return;
    impl_->cursor_locked = locked;
    if (locked) {
        [NSCursor hide];
    } else {
        [NSCursor unhide];
    }
}

bool Window::CursorLocked() const { return impl_->cursor_locked; }

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
