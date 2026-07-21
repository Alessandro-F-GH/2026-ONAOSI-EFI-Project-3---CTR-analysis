#include "file_discovery.hpp"
#include "trc_reader.hpp"

#include <Compression.h>
#include <TFile.h>
#include <TTree.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
  fs::path input_directory;
  fs::path output_file;
  std::string run_name;
  std::string pairing = "auto";
  std::int64_t max_events = 0;
  std::int64_t start_event = 0;
  int progress_every = 500;
  bool recursive = true;
};

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n  " << program
      << " --input RUN_DIR --output RUN.root [options]\n\n"
      << "Required:\n"
      << "  --input DIR          Directory containing one run of C1..C4 .trc files\n"
      << "  --output FILE.root   One ROOT output file for this run\n\n"
      << "Options:\n"
      << "  --run-name NAME      Metadata only; default is the input directory name\n"
      << "  --pairing MODE       auto, id or rank (default: auto)\n"
      << "  --max-events N       0 means all events (default: 0)\n"
      << "  --start-event N      Skip the first N paired events (default: 0)\n"
      << "  --progress-every N   Progress print interval (default: 500)\n"
      << "  --recursive 0|1      Search below RUN_DIR (default: 1)\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value after ") + name);
      return argv[++i];
    };
    if (argument == "--input") options.input_directory = value("--input");
    else if (argument == "--output") options.output_file = value("--output");
    else if (argument == "--run-name") options.run_name = value("--run-name");
    else if (argument == "--pairing") options.pairing = value("--pairing");
    else if (argument == "--max-events") options.max_events = std::stoll(value("--max-events"));
    else if (argument == "--start-event") options.start_event = std::stoll(value("--start-event"));
    else if (argument == "--progress-every") options.progress_every = std::stoi(value("--progress-every"));
    else if (argument == "--recursive") options.recursive = std::stoi(value("--recursive")) != 0;
    else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.input_directory.empty() || options.output_file.empty()) {
    print_usage(argv[0]);
    throw std::runtime_error("--input and --output are required");
  }
  if (options.run_name.empty()) options.run_name = options.input_directory.filename().string();
  if (options.run_name.empty()) options.run_name = "run";
  if (options.max_events < 0 || options.start_event < 0 || options.progress_every <= 0) {
    throw std::runtime_error("invalid numeric option");
  }
  return options;
}

std::string absolute_string(const fs::path& path) {
  std::error_code error;
  const fs::path absolute = fs::absolute(path, error);
  return error ? path.string() : absolute.string();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto started = std::chrono::steady_clock::now();

    const discovery::Result discovery = discovery::discover_events(
        options.input_directory, options.pairing, options.recursive);

    std::cout << "Discovered .trc files\n";
    for (std::size_t channel = 0; channel < discovery.channel_files.size(); ++channel) {
      std::cout << "  C" << (channel + 1) << ": "
                << discovery.channel_files[channel].size() << '\n';
    }
    std::cout << "  common IDs: " << discovery.common_id_count << '\n'
              << "  pairing:    " << discovery.pairing_used << '\n'
              << "  events:     " << discovery.events.size() << '\n';
    if (discovery.pairing_used == "rank") {
      std::cout << "Warning: channels are paired by sorted position because a reliable common-ID "
                   "intersection was not found. Source IDs are preserved in the output.\n";
    }

    fs::create_directories(options.output_file.parent_path().empty()
                               ? fs::path(".")
                               : options.output_file.parent_path());
    TFile output(options.output_file.string().c_str(), "RECREATE");
    if (output.IsZombie()) {
      throw std::runtime_error("cannot create ROOT output: " + options.output_file.string());
    }
    output.SetCompressionSettings(404);

    TTree events("events", "Calibrated LeCroy trace payload: raw ADC samples plus per-trace axes");
    events.SetAutoFlush(-64LL * 1024LL * 1024LL);

    Long64_t event_index = -1;
    Long64_t event_id = -1;
    Long64_t source_file_id[4] = {-1, -1, -1, -1};
    Int_t sample_count[4] = {0, 0, 0, 0};
    Float_t vertical_gain_v_per_count[4] = {0, 0, 0, 0};
    Float_t vertical_offset_v[4] = {0, 0, 0, 0};
    Float_t horizontal_interval_s[4] = {0, 0, 0, 0};
    Double_t horizontal_offset_s[4] = {0, 0, 0, 0};
    std::array<std::vector<Short_t>, 4> samples;

    events.Branch("event_index", &event_index);
    events.Branch("event_id", &event_id);
    events.Branch("source_file_id", source_file_id, "source_file_id[4]/L");
    events.Branch("sample_count", sample_count, "sample_count[4]/I");
    events.Branch("vertical_gain_v_per_count", vertical_gain_v_per_count,
                  "vertical_gain_v_per_count[4]/F");
    events.Branch("vertical_offset_v", vertical_offset_v, "vertical_offset_v[4]/F");
    events.Branch("horizontal_interval_s", horizontal_interval_s,
                  "horizontal_interval_s[4]/F");
    events.Branch("horizontal_offset_s", horizontal_offset_s, "horizontal_offset_s[4]/D");
    events.Branch("samples_ch1", &samples[0]);
    events.Branch("samples_ch2", &samples[1]);
    events.Branch("samples_ch3", &samples[2]);
    events.Branch("samples_ch4", &samples[3]);

    const std::int64_t begin = std::min<std::int64_t>(
        options.start_event, static_cast<std::int64_t>(discovery.events.size()));
    std::int64_t end = static_cast<std::int64_t>(discovery.events.size());
    if (options.max_events > 0) end = std::min(end, begin + options.max_events);

    std::int64_t written = 0;
    std::int64_t invalid = 0;
    for (std::int64_t position = begin; position < end; ++position) {
      const discovery::EventFiles& bundle = discovery.events[static_cast<std::size_t>(position)];
      std::array<trc::Trace, 4> traces;
      bool valid = true;
      for (std::size_t channel = 0; channel < traces.size(); ++channel) {
        std::string error;
        if (!trc::read_trace(bundle.paths[channel], traces[channel], error)) {
          std::cerr << "Skipping event " << bundle.event_id << ", C" << (channel + 1)
                    << ": " << error << " [" << bundle.paths[channel] << "]\n";
          valid = false;
          break;
        }
      }
      if (!valid) {
        ++invalid;
        continue;
      }

      event_index = written;
      event_id = bundle.event_id;
      for (std::size_t channel = 0; channel < traces.size(); ++channel) {
        source_file_id[channel] = bundle.source_file_id[channel];
        sample_count[channel] = traces[channel].header.sample_count;
        vertical_gain_v_per_count[channel] =
            traces[channel].header.vertical_gain_v_per_count;
        vertical_offset_v[channel] = traces[channel].header.vertical_offset_v;
        horizontal_interval_s[channel] = traces[channel].header.horizontal_interval_s;
        horizontal_offset_s[channel] = traces[channel].header.horizontal_offset_s;
        samples[channel].assign(traces[channel].samples.begin(), traces[channel].samples.end());
      }
      events.Fill();
      ++written;
      if (written % options.progress_every == 0 || position + 1 == end) {
        const double fraction = end > begin
                                    ? static_cast<double>(position + 1 - begin) /
                                          static_cast<double>(end - begin)
                                    : 1.0;
        std::cout << "Processed " << (position + 1 - begin) << '/' << (end - begin)
                  << " paired events (" << std::fixed << std::setprecision(1)
                  << 100.0 * fraction << "%), written " << written << '\n';
      }
    }

    TTree metadata("metadata", "Conversion metadata");
    std::string format_version = "trc-singlefile-v1";
    std::string run_name = options.run_name;
    std::string input_directory = absolute_string(options.input_directory);
    std::string pairing_used = discovery.pairing_used;
    Long64_t discovered_files[4] = {
        static_cast<Long64_t>(discovery.channel_files[0].size()),
        static_cast<Long64_t>(discovery.channel_files[1].size()),
        static_cast<Long64_t>(discovery.channel_files[2].size()),
        static_cast<Long64_t>(discovery.channel_files[3].size())};
    Long64_t paired_events = static_cast<Long64_t>(discovery.events.size());
    Long64_t written_events = written;
    Long64_t invalid_events = invalid;
    Int_t channel_numbers[4] = {1, 2, 3, 4};
    Int_t sample_count_offset = static_cast<Int_t>(trc::Layout::sample_count_offset);
    Int_t vertical_gain_offset = static_cast<Int_t>(trc::Layout::vertical_gain_offset);
    Int_t vertical_offset_offset = static_cast<Int_t>(trc::Layout::vertical_offset_offset);
    Int_t horizontal_interval_offset = static_cast<Int_t>(trc::Layout::horizontal_interval_offset);
    Int_t horizontal_offset_offset = static_cast<Int_t>(trc::Layout::horizontal_offset_offset);
    Int_t samples_offset = static_cast<Int_t>(trc::Layout::samples_offset);

    metadata.Branch("format_version", &format_version);
    metadata.Branch("run_name", &run_name);
    metadata.Branch("input_directory", &input_directory);
    metadata.Branch("pairing_used", &pairing_used);
    metadata.Branch("channel_numbers", channel_numbers, "channel_numbers[4]/I");
    metadata.Branch("discovered_files", discovered_files, "discovered_files[4]/L");
    metadata.Branch("paired_events", &paired_events);
    metadata.Branch("written_events", &written_events);
    metadata.Branch("invalid_events", &invalid_events);
    metadata.Branch("sample_count_offset", &sample_count_offset);
    metadata.Branch("vertical_gain_offset", &vertical_gain_offset);
    metadata.Branch("vertical_offset_offset", &vertical_offset_offset);
    metadata.Branch("horizontal_interval_offset", &horizontal_interval_offset);
    metadata.Branch("horizontal_offset_offset", &horizontal_offset_offset);
    metadata.Branch("samples_offset", &samples_offset);
    metadata.Fill();

    output.cd();
    events.Write();
    metadata.Write();
    output.Close();

    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    std::cout << "\nConversion completed\n"
              << "  written events: " << written << '\n'
              << "  invalid events: " << invalid << '\n'
              << "  elapsed:        " << std::fixed << std::setprecision(2) << elapsed
              << " s\n"
              << "  output:         " << options.output_file << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
