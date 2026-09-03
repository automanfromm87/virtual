#include "engine/app/actions.h"

#include <algorithm>

namespace eng::app {

ActionMap::Binding& ActionMap::Entry(const std::string& name) {
    return bindings_[name];
}

const ActionMap::Binding* ActionMap::Find(std::string_view name) const {
    const auto it = bindings_.find(std::string(name));
    return it == bindings_.end() ? nullptr : &it->second;
}

void ActionMap::Bind(std::string action, char key) {
    Entry(action).keys.push_back(key);
}

void ActionMap::BindAxis(std::string axis, char negative, char positive) {
    Binding& b = Entry(axis);
    b.axis_negative = negative;
    b.axis_positive = positive;
}

void ActionMap::Update(Keys held) {
    for (auto& [name, b] : bindings_) {
        b.was_down = b.down;
        b.down = false;
        for (char k : b.keys)
            if (held.Get(k)) b.down = true;
        // An axis key counts as holding the action too, so a binding can be
        // read either way without declaring it twice.
        if (b.axis_negative && held.Get(b.axis_negative)) b.down = true;
        if (b.axis_positive && held.Get(b.axis_positive)) b.down = true;
        b.axis = 0.0f;
        if (b.axis_negative && held.Get(b.axis_negative)) b.axis -= 1.0f;
        if (b.axis_positive && held.Get(b.axis_positive)) b.axis += 1.0f;
    }
}

bool ActionMap::Pressed(std::string_view action) const {
    const Binding* b = Find(action);
    return b && b->down && !b->was_down;
}

bool ActionMap::Released(std::string_view action) const {
    const Binding* b = Find(action);
    return b && !b->down && b->was_down;
}

bool ActionMap::Down(std::string_view action) const {
    const Binding* b = Find(action);
    return b && b->down;
}

float ActionMap::Axis(std::string_view axis) const {
    const Binding* b = Find(axis);
    return b ? b->axis : 0.0f;
}

std::vector<std::string> ActionMap::Names() const {
    std::vector<std::string> out;
    out.reserve(bindings_.size());
    for (const auto& [name, b] : bindings_) out.push_back(name);
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace eng::app
