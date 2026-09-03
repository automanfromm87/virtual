// Pure C++20. A minimal JSON reader.
//
// Written rather than pulled in because this engine has no third-party code and
// glTF needs exactly one thing from JSON: read a document someone else wrote.
// That is a much smaller job than a general library — no writing, no mutation,
// no comments, no trailing commas, no streaming.
//
// The parse builds the whole tree up front. A glTF header is kilobytes; the
// buffers, which are the large part, are binary and never touch this.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace eng::json {

class Value {
  public:
    enum class Type : std::uint8_t { Null, Bool, Number, String, Array, Object };

    [[nodiscard]] Type GetType() const { return type_; }
    [[nodiscard]] bool IsNull() const { return type_ == Type::Null; }
    [[nodiscard]] bool IsNumber() const { return type_ == Type::Number; }
    [[nodiscard]] bool IsString() const { return type_ == Type::String; }
    [[nodiscard]] bool IsArray() const { return type_ == Type::Array; }
    [[nodiscard]] bool IsObject() const { return type_ == Type::Object; }

    // Accessors that never throw and never return a dangling reference: a
    // missing key yields a shared Null. A glTF file is full of optional fields,
    // and making every read a two-step check would drown the reader.
    [[nodiscard]] double Number(double fallback = 0.0) const {
        return type_ == Type::Number ? number_ : fallback;
    }
    [[nodiscard]] int Int(int fallback = 0) const {
        return type_ == Type::Number ? int(number_) : fallback;
    }
    [[nodiscard]] bool Bool(bool fallback = false) const {
        return type_ == Type::Bool ? bool_ : fallback;
    }
    [[nodiscard]] const std::string& Str() const { return string_; }

    [[nodiscard]] std::size_t Size() const { return array_.size(); }
    [[nodiscard]] const Value& operator[](std::size_t i) const;
    [[nodiscard]] const Value& operator[](std::string_view key) const;
    [[nodiscard]] bool Has(std::string_view key) const;

  private:
    friend class Parser;
    friend Value Parse(std::string_view, std::string&);

    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<Value> array_;
    std::map<std::string, Value, std::less<>> object_;
};

// Returns a Null value and fills `error` on malformed input.
[[nodiscard]] Value Parse(std::string_view text, std::string& error);

// Emitter. Deliberately not the inverse of Value: nothing in this engine builds
// a document in memory and then serialises it, everything walks its own data
// and writes as it goes, so a tree here would only be a copy of one that
// already exists.
//
// Tracks nesting and commas itself. Getting those wrong by hand is the entire
// difficulty of hand-written JSON output, and it fails at parse time in some
// other program rather than here.
class Writer {
  public:
    void BeginObject();
    void EndObject();
    void BeginArray();
    // Arrays of numbers stay on one line: a saved scene is meant to be read and
    // edited, and a vector split across four lines is worse than useless.
    void BeginInlineArray();
    void EndArray();

    // Only legal inside an object.
    void Key(std::string_view);

    void Value(double);
    void Value(int);
    void Value(bool);
    void Value(std::string_view);
    // Without this, Value("box") picks the BOOL overload — a pointer converts
    // to bool ahead of any user-defined conversion — and writes `true`. It
    // compiles, it round-trips through a parser, and it is silently wrong.
    void Value(const char* s) { Value(std::string_view(s)); }
    void Null();

    // Convenience for the shapes this engine writes constantly.
    void Vec(const float* v, int n);

    [[nodiscard]] std::string Take();

  private:
    void Separate();
    void Indent();

    std::string out_;
    // One entry per open container: true once it holds something, so the next
    // write knows whether it needs a comma.
    std::vector<bool> nonempty_;
    std::vector<bool> inline_;
    bool after_key_ = false;
};

// glTF embeds buffers as "data:application/octet-stream;base64,...". Returns
// false if the input is not valid base64.
[[nodiscard]] bool DecodeBase64(std::string_view in, std::vector<std::uint8_t>& out);

}  // namespace eng::json
