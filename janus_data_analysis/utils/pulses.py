from __future__ import annotations

from collections.abc import Iterable

from .models import Hit

Pulse = tuple[Hit, Hit]
PulsePair = tuple[Pulse, Pulse]


def pair_intervals(hits: list[Hit], channel: int) -> list[Pulse]:
    """Pair leading and trailing edges on one channel in chronological order.

    The pairing is intentionally based only on the physical ToA carried by the
    hit. In STREAMING files that ToA is a uint64 value on the continuous ToA
    time axis; event-header timestamps are not added to it.
    """
    leading = sorted(
        (hit for hit in hits if hit.channel == channel and hit.edge == 1),
        key=lambda hit: hit.toa_lsb,
    )
    trailing = sorted(
        (hit for hit in hits if hit.channel == channel and hit.edge == 0),
        key=lambda hit: hit.toa_lsb,
    )
    pairs: list[Pulse] = []
    trailing_index = 0
    for lead in leading:
        while (
            trailing_index < len(trailing)
            and trailing[trailing_index].toa_lsb <= lead.toa_lsb
        ):
            trailing_index += 1
        if trailing_index >= len(trailing):
            break
        pairs.append((lead, trailing[trailing_index]))
        trailing_index += 1
    return pairs


def earliest_energy_pair(
    hits: list[Hit],
    channel: int,
) -> Pulse | None:
    pairs = pair_intervals(hits, channel)
    return min(pairs, key=lambda pair: pair[0].toa_lsb) if pairs else None


def leading_hits_before(
    hits: list[Hit],
    channel: int,
    energy_lead_lsb: int,
    window_lsb: int,
) -> list[Hit]:
    lower = energy_lead_lsb - window_lsb
    return sorted(
        (
            hit
            for hit in hits
            if hit.channel == channel
            and hit.edge == 1
            and lower < hit.toa_lsb < energy_lead_lsb
        ),
        key=lambda hit: hit.toa_lsb,
    )


def intervals_overlap(first: Pulse, second: Pulse) -> bool:
    """Return the overlap definition requested for paired channels.

    A pair overlaps when the leading edge of either pulse lies inside the
    closed [L, T] interval of the other pulse. For valid L < T intervals this
    is equivalent to ordinary interval overlap.
    """
    first_l = first[0].toa_lsb
    first_t = first[1].toa_lsb
    second_l = second[0].toa_lsb
    second_t = second[1].toa_lsb
    return (
        first_l <= second_l <= first_t
        or second_l <= first_l <= second_t
    )



def overlap_support_masks(
    first: Iterable[Pulse],
    second: Iterable[Pulse],
) -> tuple[list[bool], list[bool]]:
    """Mark every pulse that overlaps at least one pulse on the other channel."""
    a = sorted(first, key=lambda pulse: pulse[0].toa_lsb)
    b = sorted(second, key=lambda pulse: pulse[0].toa_lsb)
    supported_a = [False] * len(a)
    supported_b = [False] * len(b)
    j = 0
    for i, pulse_a in enumerate(a):
        while j < len(b) and b[j][1].toa_lsb < pulse_a[0].toa_lsb:
            j += 1
        k = j
        while k < len(b) and b[k][0].toa_lsb <= pulse_a[1].toa_lsb:
            if intervals_overlap(pulse_a, b[k]):
                supported_a[i] = True
                supported_b[k] = True
            k += 1
    return supported_a, supported_b

def pair_overlapping_intervals(
    first: Iterable[Pulse],
    second: Iterable[Pulse],
) -> list[PulsePair]:
    """One-to-one chronological pairing using interval overlap only.

    Pulses that do not overlap a pulse on the paired channel are discarded.
    The two-pointer construction is linear after sorting and is appropriate for
    discriminator pulses, which are expected to be time ordered and normally
    non-overlapping within one channel.
    """
    a = sorted(first, key=lambda pulse: pulse[0].toa_lsb)
    b = sorted(second, key=lambda pulse: pulse[0].toa_lsb)
    paired: list[PulsePair] = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        pulse_a = a[i]
        pulse_b = b[j]
        if intervals_overlap(pulse_a, pulse_b):
            paired.append((pulse_a, pulse_b))
            i += 1
            j += 1
            continue
        if pulse_a[1].toa_lsb < pulse_b[0].toa_lsb:
            i += 1
        else:
            j += 1
    return paired


def timing_overlap_candidates(
    hits: list[Hit],
    timing_channel_a: int,
    timing_channel_b: int,
    energy_pair_a: Pulse,
    energy_pair_b: Pulse,
    window_lsb: int,
) -> list[PulsePair]:
    """Return overlapping ch3/ch7 pulse pairs preceding both energy pulses."""
    timing_pairs = pair_overlapping_intervals(
        pair_intervals(hits, timing_channel_a),
        pair_intervals(hits, timing_channel_b),
    )
    energy_l_a = energy_pair_a[0].toa_lsb
    energy_l_b = energy_pair_b[0].toa_lsb
    result: list[PulsePair] = []
    for pulse_a, pulse_b in timing_pairs:
        delay_a = energy_l_a - pulse_a[0].toa_lsb
        delay_b = energy_l_b - pulse_b[0].toa_lsb
        if 0 < delay_a < window_lsb and 0 < delay_b < window_lsb:
            result.append((pulse_a, pulse_b))
    return result
