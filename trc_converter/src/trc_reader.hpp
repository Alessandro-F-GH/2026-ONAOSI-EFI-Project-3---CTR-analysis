#ifndef TRC_SINGLEFILE_TRC_READER_HPP
#define TRC_SINGLEFILE_TRC_READER_HPP

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace trc {

// Binary offsets for the LeCroy .trc files used in this project.
// These are properties of the file format, not run-dependent analysis settings.
struct Layout {
  static constexpr std::streamoff sample_count_offset = 127;
  static constexpr std::streamoff vertical_gain_offset = 167;
  static constexpr std::streamoff vertical_offset_offset = 171;
  static constexpr std::streamoff horizontal_interval_offset = 187;
  static constexpr std::streamoff horizontal_offset_offset = 191;
  static constexpr std::streamoff samples_offset = 357;
  static constexpr std::int32_t max_samples = 10'000'000;
};

struct TraceHeader {
  std::int32_t sample_count = 0;
  float vertical_gain_v_per_count = 0.0F;
  float vertical_offset_v = 0.0F;
  float horizontal_interval_s = 0.0F;
  double horizontal_offset_s = 0.0;
};

struct Trace {
  TraceHeader header;
  std::vector<std::int16_t> samples;
};

bool read_trace(const std::filesystem::path& path, Trace& trace, std::string& error);

}  // namespace trc

#endif
