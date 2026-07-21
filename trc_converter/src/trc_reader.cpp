#include "trc_reader.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <type_traits>

namespace trc {
namespace {

template <typename T>
bool read_at(std::ifstream& input, std::streamoff offset, T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  input.clear();
  input.seekg(offset, std::ios::beg);
  if (!input) return false;
  input.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
  return static_cast<bool>(input);
}

bool finite_header(const TraceHeader& header) {
  return std::isfinite(header.vertical_gain_v_per_count) &&
         std::isfinite(header.vertical_offset_v) &&
         std::isfinite(header.horizontal_interval_s) &&
         std::isfinite(header.horizontal_offset_s) &&
         header.horizontal_interval_s > 0.0F;
}

}  // namespace

bool read_trace(const std::filesystem::path& path, Trace& trace, std::string& error) {
  trace = Trace{};
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = "cannot open file";
    return false;
  }

  if (!read_at(input, Layout::sample_count_offset, trace.header.sample_count)) {
    error = "cannot read sample count";
    return false;
  }
  if (trace.header.sample_count <= 1 || trace.header.sample_count > Layout::max_samples) {
    error = "invalid sample count: " + std::to_string(trace.header.sample_count);
    return false;
  }

  if (!read_at(input, Layout::vertical_gain_offset,
               trace.header.vertical_gain_v_per_count) ||
      !read_at(input, Layout::vertical_offset_offset,
               trace.header.vertical_offset_v) ||
      !read_at(input, Layout::horizontal_interval_offset,
               trace.header.horizontal_interval_s) ||
      !read_at(input, Layout::horizontal_offset_offset,
               trace.header.horizontal_offset_s)) {
    error = "cannot read waveform calibration fields";
    return false;
  }
  if (!finite_header(trace.header)) {
    error = "non-finite or invalid waveform calibration";
    return false;
  }

  input.clear();
  input.seekg(0, std::ios::end);
  const std::streamoff file_size = input.tellg();
  const std::streamoff payload_bytes =
      static_cast<std::streamoff>(trace.header.sample_count) *
      static_cast<std::streamoff>(sizeof(std::int16_t));
  if (file_size < Layout::samples_offset + payload_bytes) {
    error = "file is shorter than its declared sample payload";
    return false;
  }

  trace.samples.resize(static_cast<std::size_t>(trace.header.sample_count));
  input.clear();
  input.seekg(Layout::samples_offset, std::ios::beg);
  input.read(reinterpret_cast<char*>(trace.samples.data()),
             static_cast<std::streamsize>(payload_bytes));
  if (!input) {
    error = "cannot read sample payload";
    trace.samples.clear();
    return false;
  }
  return true;
}

}  // namespace trc
