from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

HEADER_BYTES = 357


def write_trace(path: Path, channel: int, event: int, samples: int = 8002) -> None:
    gain = 0.00025  # V/count
    vertical_offset = 0.0
    dt = 2.5e-12
    t0 = -10e-9
    raw = []
    peak = 2500 + channel * 20
    for index in range(samples):
        pulse = 1200.0 * math.exp(-0.5 * ((index - peak) / 40.0) ** 2)
        raw.append(int(round(pulse)))

    payload = bytearray(HEADER_BYTES + 2 * samples)
    struct.pack_into("<i", payload, 127, samples)
    struct.pack_into("<f", payload, 167, gain)
    struct.pack_into("<f", payload, 171, vertical_offset)
    struct.pack_into("<f", payload, 187, dt)
    struct.pack_into("<d", payload, 191, t0)
    struct.pack_into(f"<{samples}h", payload, 357, *raw)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--events", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for event in range(args.events):
        for channel in range(1, 5):
            name = f"C{channel}--synthetic-run--{event:05d}.trc"
            write_trace(args.output / name, channel, event)
    print(f"Wrote {args.events} synthetic four-channel events to {args.output}")


if __name__ == "__main__":
    main()
