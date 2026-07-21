from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from waveform_analysis.io import event_count, get_event, read_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a converted waveform ROOT file")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--event", type=int, default=0, help="TTree entry index")
    parser.add_argument("--output", type=Path, default=Path("event_waveforms.png"))
    args = parser.parse_args()

    metadata = read_metadata(args.input)
    print("Metadata")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print(f"Events: {event_count(args.input)}")

    event = get_event(args.input, args.event)
    figure, axis = plt.subplots(figsize=(11, 7))
    for channel, (time_ns, voltage_mV) in enumerate(event["waveforms"], start=1):
        axis.plot(time_ns, voltage_mV, linewidth=1.0, label=f"C{channel}")
    axis.set_title(f"Event entry {event['event_index']} · source ID {event['event_id']}")
    axis.set_xlabel("Time [ns]")
    axis.set_ylabel("Voltage [mV]")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"Plot: {args.output}")


if __name__ == "__main__":
    main()
