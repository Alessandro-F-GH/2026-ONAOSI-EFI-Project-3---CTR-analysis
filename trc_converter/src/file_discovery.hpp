#ifndef TRC_SINGLEFILE_FILE_DISCOVERY_HPP
#define TRC_SINGLEFILE_FILE_DISCOVERY_HPP

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace discovery {

struct IndexedFile {
  std::int64_t file_id = -1;
  std::filesystem::path path;
};

struct EventFiles {
  std::int64_t event_index = -1;
  std::int64_t event_id = -1;
  std::array<std::int64_t, 4> source_file_id{{-1, -1, -1, -1}};
  std::array<std::filesystem::path, 4> paths;
};

struct Result {
  std::array<std::vector<IndexedFile>, 4> channel_files;
  std::vector<EventFiles> events;
  std::string pairing_used;
  std::int64_t common_id_count = 0;
  std::int64_t minimum_channel_count = 0;
};

bool parse_filename(const std::filesystem::path& path, int& channel_one_based,
                    std::int64_t& file_id);

Result discover_events(const std::filesystem::path& input_directory,
                       const std::string& pairing, bool recursive);

}  // namespace discovery

#endif
