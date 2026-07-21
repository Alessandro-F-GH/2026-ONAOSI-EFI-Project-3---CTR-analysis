#include "file_discovery.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace fs = std::filesystem;

namespace discovery {
namespace {

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::array<std::map<std::int64_t, fs::path>, 4> build_maps(
    const std::array<std::vector<IndexedFile>, 4>& files) {
  std::array<std::map<std::int64_t, fs::path>, 4> maps;
  for (std::size_t channel = 0; channel < files.size(); ++channel) {
    for (const IndexedFile& item : files[channel]) {
      const auto [_, inserted] = maps[channel].emplace(item.file_id, item.path);
      if (!inserted) {
        throw std::runtime_error(
            "duplicate event/file ID " + std::to_string(item.file_id) +
            " in channel C" + std::to_string(channel + 1) +
            ". The input likely contains more than one run; point --input to a single run directory.");
      }
    }
  }
  return maps;
}

std::vector<EventFiles> pair_by_id(
    const std::array<std::vector<IndexedFile>, 4>& files,
    std::int64_t& common_count) {
  const auto maps = build_maps(files);
  std::vector<EventFiles> events;
  for (const auto& [id, path0] : maps[0]) {
    bool complete = true;
    for (std::size_t channel = 1; channel < maps.size(); ++channel) {
      if (maps[channel].find(id) == maps[channel].end()) {
        complete = false;
        break;
      }
    }
    if (!complete) continue;
    EventFiles event;
    event.event_index = static_cast<std::int64_t>(events.size());
    event.event_id = id;
    for (std::size_t channel = 0; channel < maps.size(); ++channel) {
      event.source_file_id[channel] = id;
      event.paths[channel] = maps[channel].at(id);
    }
    events.push_back(std::move(event));
  }
  common_count = static_cast<std::int64_t>(events.size());
  return events;
}

std::vector<EventFiles> pair_by_rank(
    const std::array<std::vector<IndexedFile>, 4>& files) {
  std::size_t count = files[0].size();
  for (std::size_t channel = 1; channel < files.size(); ++channel) {
    count = std::min(count, files[channel].size());
  }
  std::vector<EventFiles> events;
  events.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    EventFiles event;
    event.event_index = static_cast<std::int64_t>(index);
    bool same_id = true;
    const std::int64_t first_id = files[0][index].file_id;
    for (std::size_t channel = 0; channel < files.size(); ++channel) {
      event.paths[channel] = files[channel][index].path;
      event.source_file_id[channel] = files[channel][index].file_id;
      same_id = same_id && files[channel][index].file_id == first_id;
    }
    event.event_id = same_id ? first_id : static_cast<std::int64_t>(index);
    events.push_back(std::move(event));
  }
  return events;
}

}  // namespace

bool parse_filename(const fs::path& path, int& channel_one_based,
                    std::int64_t& file_id) {
  if (lower_copy(path.extension().string()) != ".trc") return false;
  const std::string stem = path.stem().string();
  if (stem.size() < 3 || (stem[0] != 'C' && stem[0] != 'c')) return false;

  std::size_t channel_end = 1;
  while (channel_end < stem.size() &&
         std::isdigit(static_cast<unsigned char>(stem[channel_end]))) {
    ++channel_end;
  }
  if (channel_end == 1) return false;
  try {
    channel_one_based = std::stoi(stem.substr(1, channel_end - 1));
  } catch (...) {
    return false;
  }
  if (channel_one_based < 1 || channel_one_based > 4) return false;

  std::size_t id_end = stem.size();
  std::size_t id_begin = id_end;
  while (id_begin > 0 &&
         std::isdigit(static_cast<unsigned char>(stem[id_begin - 1]))) {
    --id_begin;
  }
  if (id_begin == id_end) return false;
  try {
    file_id = std::stoll(stem.substr(id_begin, id_end - id_begin));
  } catch (...) {
    return false;
  }
  return true;
}

Result discover_events(const fs::path& input_directory,
                       const std::string& pairing, bool recursive) {
  if (!fs::is_directory(input_directory)) {
    throw std::runtime_error("input directory does not exist: " + input_directory.string());
  }
  if (pairing != "auto" && pairing != "id" && pairing != "rank") {
    throw std::runtime_error("pairing must be one of: auto, id, rank");
  }

  Result result;
  auto inspect = [&](const fs::directory_entry& entry) {
    if (!entry.is_regular_file()) return;
    int channel = 0;
    std::int64_t file_id = -1;
    if (!parse_filename(entry.path(), channel, file_id)) return;
    result.channel_files[static_cast<std::size_t>(channel - 1)].push_back(
        IndexedFile{file_id, entry.path()});
  };

  if (recursive) {
    for (const auto& entry : fs::recursive_directory_iterator(input_directory)) inspect(entry);
  } else {
    for (const auto& entry : fs::directory_iterator(input_directory)) inspect(entry);
  }

  for (std::size_t channel = 0; channel < result.channel_files.size(); ++channel) {
    auto& files = result.channel_files[channel];
    std::sort(files.begin(), files.end(), [](const IndexedFile& a, const IndexedFile& b) {
      if (a.file_id != b.file_id) return a.file_id < b.file_id;
      return a.path.string() < b.path.string();
    });
    if (files.empty()) {
      throw std::runtime_error("no .trc files found for channel C" +
                               std::to_string(channel + 1));
    }
  }

  std::size_t minimum = result.channel_files[0].size();
  for (std::size_t channel = 1; channel < result.channel_files.size(); ++channel) {
    minimum = std::min(minimum, result.channel_files[channel].size());
  }
  result.minimum_channel_count = static_cast<std::int64_t>(minimum);

  std::int64_t common = 0;
  std::vector<EventFiles> id_events = pair_by_id(result.channel_files, common);
  result.common_id_count = common;

  if (pairing == "id") {
    result.events = std::move(id_events);
    result.pairing_used = "id";
  } else if (pairing == "rank") {
    result.events = pair_by_rank(result.channel_files);
    result.pairing_used = "rank";
  } else {
    const double coverage = minimum > 0 ? static_cast<double>(common) /
                                             static_cast<double>(minimum)
                                       : 0.0;
    if (common > 0 && coverage >= 0.80) {
      result.events = std::move(id_events);
      result.pairing_used = "id";
    } else {
      result.events = pair_by_rank(result.channel_files);
      result.pairing_used = "rank";
    }
  }

  if (result.events.empty()) {
    throw std::runtime_error("no complete four-channel events could be constructed");
  }
  return result;
}

}  // namespace discovery
