"""Parse XLA priority_fusion_dump.txt into structured trajectory data.

Extracts three event types from XLA's FusionProcessDumpProto text format:
- fusion {}        — a producer was fused into a consumer
- update_priority {} — a producer's priority was (re)computed with cost model scores
- producer_ineligible {} — a producer was skipped (with reason)

Groups events into sequential MDP steps and extracts priority orderings
for trajectory generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class FusionEvent:
    """A fusion {} event: producer absorbed into consumer."""
    fusion_name: str
    producer_name: str
    consumer_name: str


@dataclass
class UpdatePriorityEvent:
    """An update_priority {} event: cost model scores for a producer."""
    producer_name: str
    consumer_names: List[str]
    us_fused: float
    us_unfused: float


@dataclass
class IneligibleEvent:
    """A producer_ineligible {} event: producer was skipped."""
    producer_name: str
    reason: str


@dataclass
class MdpStep:
    """One MDP step: a producer was dequeued and either fused or skipped.

    In XLA's priority fusion, a single dequeue yields:
    - Zero or more update_priority events (priority recomputation)
    - Exactly one fusion or ineligible event for the dequeued producer
    - Possibly followed by update_priority events for affected neighbours
    """
    step_idx: int
    producer_name: str
    consumer_names: List[str] = field(default_factory=list)
    fusion_name: Optional[str] = None
    us_fused: float = 0.0
    us_unfused: float = 0.0
    was_fused: bool = False
    reason: Optional[str] = None  # if ineligible


@dataclass
class GpuDeviceInfo:
    """GPU device info from the dump footer."""
    core_count: int = 0
    memory_bandwidth: int = 0
    clock_rate_ghz: float = 0.0
    compute_capability: Tuple[int, int] = (0, 0)
    l2_cache_size: int = 0
    device_memory_size: int = 0
    shared_memory_per_core: int = 0


@dataclass
class ParsedDump:
    """Complete parsed dump with all events, MDP steps, and metadata."""
    # Raw events in dump order
    fusion_events: List[FusionEvent] = field(default_factory=list)
    update_priority_events: List[UpdatePriorityEvent] = field(default_factory=list)
    ineligible_events: List[IneligibleEvent] = field(default_factory=list)
    all_events: List[object] = field(default_factory=list)  # in dump order

    # Reconstructed MDP trajectory
    steps: List[MdpStep] = field(default_factory=list)

    # Priority ordering extracted from update_priority events
    # Maps producer_name -> first observed (us_unfused - us_fused)
    initial_priorities: Dict[str, float] = field(default_factory=dict)

    # GPU device info
    gpu_info: Optional[GpuDeviceInfo] = None

    # HLO module name
    hlo_module_name: str = ""

    @property
    def num_fusions(self) -> int:
        return sum(1 for s in self.steps if s.was_fused)

    @property
    def num_ineligible(self) -> int:
        return sum(1 for s in self.steps if not s.was_fused)

    @property
    def total_us_fused(self) -> float:
        return sum(s.us_fused for s in self.steps if s.was_fused)

    @property
    def total_us_unfused(self) -> float:
        return sum(s.us_unfused for s in self.steps if s.was_fused)


# ======================================================================
# Block-level parser
# ======================================================================

def _extract_blocks(text: str) -> List[str]:
    """Extract top-level fusion_steps { ... } blocks from dump text.

    Handles nested braces correctly. Stops at gpu_device_info or
    hlo_module_before_fusion sections.
    """
    blocks = []
    i = 0
    marker = "fusion_steps {"

    while i < len(text):
        start = text.find(marker, i)
        if start == -1:
            break

        # Find matching closing brace
        depth = 0
        j = start + len(marker) - 1  # position of the '{'
        while j < len(text):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    blocks.append(text[start:j + 1])
                    break
            j += 1
        i = j + 1

    return blocks


def _parse_fusion_block(content: str) -> Optional[FusionEvent]:
    """Parse a fusion { ... } sub-block."""
    fusion_name = _extract_field(content, "fusion_name")
    producer_name = _extract_field(content, "producer_name")
    consumer_name = _extract_field(content, "consumer_name")

    if producer_name and consumer_name and fusion_name:
        return FusionEvent(
            fusion_name=fusion_name,
            producer_name=producer_name,
            consumer_name=consumer_name,
        )
    return None


def _parse_update_priority_block(content: str) -> Optional[UpdatePriorityEvent]:
    """Parse an update_priority { ... } sub-block."""
    producer_name = _extract_field(content, "producer_name")
    if not producer_name:
        return None

    # consumer_names can appear multiple times
    consumer_names = _extract_repeated_field(content, "consumer_names")

    us_fused = _extract_float(content, "us_fused")
    us_unfused = _extract_float(content, "us_unfused")

    return UpdatePriorityEvent(
        producer_name=producer_name,
        consumer_names=consumer_names,
        us_fused=us_fused,
        us_unfused=us_unfused,
    )


def _parse_ineligible_block(content: str) -> Optional[IneligibleEvent]:
    """Parse a producer_ineligible { ... } sub-block."""
    producer_name = _extract_field(content, "producer_name")
    reason = _extract_field(content, "reason")

    if producer_name:
        return IneligibleEvent(
            producer_name=producer_name,
            reason=reason or "",
        )
    return None


# ======================================================================
# Field extraction helpers
# ======================================================================

# Pattern: field_name: "value" (quoted string)
_QUOTED_RE = re.compile(r'(\w+):\s*"([^"]*)"')
# Pattern: field_name: value (unquoted numeric or identifier)
_UNQUOTED_RE = re.compile(r'(\w+):\s*([^\s"{}]+)')


def _extract_field(text: str, field_name: str) -> Optional[str]:
    """Extract a single string field value."""
    # Try quoted first
    pattern = re.compile(rf'{field_name}:\s*"([^"]*)"')
    m = pattern.search(text)
    if m:
        return m.group(1)

    # Try unquoted
    pattern = re.compile(rf'{field_name}:\s*([^\s"{{}}]+)')
    m = pattern.search(text)
    if m:
        return m.group(1)

    return None


def _extract_repeated_field(text: str, field_name: str) -> List[str]:
    """Extract all occurrences of a repeated string field."""
    values = []
    # Try quoted
    for m in re.finditer(rf'{field_name}:\s*"([^"]*)"', text):
        values.append(m.group(1))

    if not values:
        # Try unquoted
        for m in re.finditer(rf'{field_name}:\s*([^\s"{{}}]+)', text):
            values.append(m.group(1))

    return values


def _extract_float(text: str, field_name: str) -> float:
    """Extract a float field value."""
    pattern = re.compile(rf'{field_name}:\s*([^\s"{{}}]+)')
    m = pattern.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    return 0.0


# ======================================================================
# GPU device info parser
# ======================================================================

def _parse_gpu_device_info(text: str) -> Optional[GpuDeviceInfo]:
    """Parse the gpu_device_info { ... } block at the end of the dump."""
    match = re.search(r'gpu_device_info\s*\{', text)
    if not match:
        return None

    # Find the matching closing brace
    start = match.start()
    depth = 0
    end = start
    for i in range(match.end() - 1, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    block = text[start:end]

    info = GpuDeviceInfo()
    core_count = _extract_field(block, "core_count")
    if core_count:
        info.core_count = int(core_count)

    mem_bw = _extract_field(block, "memory_bandwidth")
    if mem_bw:
        info.memory_bandwidth = int(mem_bw)

    clock = _extract_field(block, "clock_rate_ghz")
    if clock:
        info.clock_rate_ghz = float(clock)

    l2 = _extract_field(block, "l2_cache_size")
    if l2:
        info.l2_cache_size = int(l2)

    dev_mem = _extract_field(block, "device_memory_size")
    if dev_mem:
        info.device_memory_size = int(dev_mem)

    shared = _extract_field(block, "shared_memory_per_core")
    if shared:
        info.shared_memory_per_core = int(shared)

    major = _extract_field(block, "major")
    minor = _extract_field(block, "minor")
    if major and minor:
        info.compute_capability = (int(major), int(minor))

    return info


def _parse_hlo_module_name(text: str) -> str:
    """Extract the HLO module name from hlo_module_before_fusion line."""
    match = re.search(r'hlo_module_before_fusion:\s*"HloModule\s+(\S+)', text)
    if match:
        name = match.group(1)
        # Strip trailing comma if present
        return name.rstrip(",")
    return ""


# ======================================================================
# Main parser
# ======================================================================

def parse_dump(dump_path: str | Path) -> ParsedDump:
    """Parse a priority_fusion_dump.txt file into structured data.

    Args:
        dump_path: Path to the dump file.

    Returns:
        ParsedDump with all events, MDP steps, and metadata.
    """
    dump_path = Path(dump_path)
    text = dump_path.read_text()

    result = ParsedDump()

    # Parse GPU info and module name from footer
    result.gpu_info = _parse_gpu_device_info(text)
    result.hlo_module_name = _parse_hlo_module_name(text)

    # Extract all fusion_steps blocks
    blocks = _extract_blocks(text)

    for block in blocks:
        # Determine sub-block type
        if "fusion {" in block and "fusion_name:" in block:
            event = _parse_fusion_block(block)
            if event:
                result.fusion_events.append(event)
                result.all_events.append(event)

        elif "update_priority {" in block:
            event = _parse_update_priority_block(block)
            if event:
                result.update_priority_events.append(event)
                result.all_events.append(event)

        elif "producer_ineligible {" in block:
            event = _parse_ineligible_block(block)
            if event:
                result.ineligible_events.append(event)
                result.all_events.append(event)

    # Build MDP steps and extract initial priorities
    _build_mdp_steps(result)
    _extract_initial_priorities(result)

    return result


def _build_mdp_steps(result: ParsedDump) -> None:
    """Reconstruct MDP steps from the raw event sequence.

    XLA's fusion loop processes one producer at a time:
    1. Dequeue producer (highest priority)
    2. Check legality → either fuse or mark ineligible
    3. If fused, update priorities for affected neighbours

    The dump interleaves these events. We group them by identifying
    "action" events (fusion or ineligible) and associating the most
    recent update_priority for the same producer as its cost model scores.
    """
    # First pass: build a mapping of producer_name -> latest update_priority
    # that appeared BEFORE each fusion/ineligible event for that producer.
    # This gives us the cost model scores at the time of the action.

    # Track the most recent update_priority per producer as we scan
    latest_priority: Dict[str, UpdatePriorityEvent] = {}
    step_idx = 0

    for event in result.all_events:
        if isinstance(event, UpdatePriorityEvent):
            latest_priority[event.producer_name] = event

        elif isinstance(event, FusionEvent):
            step = MdpStep(
                step_idx=step_idx,
                producer_name=event.producer_name,
                consumer_names=[event.consumer_name],
                fusion_name=event.fusion_name,
                was_fused=True,
            )

            # Attach cost model scores from the most recent update_priority
            # for this producer (which may have been for the producer before
            # it got a fusion_name, or for the fusion_name itself)
            up = latest_priority.get(event.producer_name)
            if up:
                step.us_fused = up.us_fused
                step.us_unfused = up.us_unfused
                step.consumer_names = up.consumer_names if up.consumer_names else [event.consumer_name]

            result.steps.append(step)
            step_idx += 1

        elif isinstance(event, IneligibleEvent):
            step = MdpStep(
                step_idx=step_idx,
                producer_name=event.producer_name,
                was_fused=False,
                reason=event.reason,
            )
            # Some ineligible producers had update_priority computed before
            # the ineligibility was determined
            up = latest_priority.get(event.producer_name)
            if up:
                step.us_fused = up.us_fused
                step.us_unfused = up.us_unfused

            result.steps.append(step)
            step_idx += 1

    # Second pass: merge consecutive fusion events for the same
    # fusion_name (multi-consumer fusions where XLA logs one fusion {}
    # per consumer but it's really one dequeue action).
    _merge_multi_consumer_fusions(result)


def _merge_multi_consumer_fusions(result: ParsedDump) -> None:
    """Merge consecutive fusion steps that share the same fusion_name.

    When a producer fuses into multiple consumers, XLA may log multiple
    fusion {} events with the same fusion_name. These represent a single
    MDP action (one producer dequeue).
    """
    if not result.steps:
        return

    merged = [result.steps[0]]

    for step in result.steps[1:]:
        prev = merged[-1]

        if (step.was_fused and prev.was_fused
                and step.fusion_name == prev.fusion_name
                and step.fusion_name is not None):
            # Same fusion action — merge consumer names
            for cn in step.consumer_names:
                if cn not in prev.consumer_names:
                    prev.consumer_names.append(cn)
        else:
            merged.append(step)

    # Re-index
    for i, step in enumerate(merged):
        step.step_idx = i

    result.steps = merged


def _extract_initial_priorities(result: ParsedDump) -> None:
    """Extract first-observed priority for each producer.

    The first update_priority event for each producer represents its
    initial cost-model priority before any fusions changed the graph.
    This is useful as the "XLA default" ordering for policies.py.
    """
    seen = set()
    for event in result.update_priority_events:
        if event.producer_name not in seen:
            seen.add(event.producer_name)
            result.initial_priorities[event.producer_name] = (
                event.us_unfused - event.us_fused
            )


# ======================================================================
# Convenience functions
# ======================================================================

def get_producer_ordering(result: ParsedDump) -> List[Tuple[str, float]]:
    """Return producers sorted by initial priority (descending).

    This is the order XLA would dequeue them in the default strategy.
    """
    items = list(result.initial_priorities.items())
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def get_fusion_sequence(result: ParsedDump) -> List[str]:
    """Return the ordered list of producer names that were actually fused."""
    return [s.producer_name for s in result.steps if s.was_fused]


def get_ineligible_producers(result: ParsedDump) -> List[Tuple[str, str]]:
    """Return (producer_name, reason) for all ineligible producers."""
    return [(s.producer_name, s.reason or "") for s in result.steps if not s.was_fused]


def summary(result: ParsedDump) -> str:
    """Return a human-readable summary of the parsed dump."""
    lines = [
        f"=== Parsed Dump Summary ===",
        f"HLO Module: {result.hlo_module_name}",
        f"Total events: {len(result.all_events)}",
        f"  - fusion: {len(result.fusion_events)}",
        f"  - update_priority: {len(result.update_priority_events)}",
        f"  - producer_ineligible: {len(result.ineligible_events)}",
        f"MDP steps: {len(result.steps)}",
        f"  - fused: {result.num_fusions}",
        f"  - ineligible: {result.num_ineligible}",
        f"Total us_fused: {result.total_us_fused:.2f}",
        f"Total us_unfused: {result.total_us_unfused:.2f}",
        f"Unique initial priorities: {len(result.initial_priorities)}",
    ]
    if result.gpu_info:
        gi = result.gpu_info
        mem_gb = f", {gi.device_memory_size / 1024**3:.0f} GB" if gi.device_memory_size else ""
        lines.append(
            f"GPU: SM {gi.compute_capability[0]}.{gi.compute_capability[1]}, "
            f"{gi.core_count} cores, "
            f"{gi.memory_bandwidth / 1e9:.1f} GB/s, "
            f"L2={gi.l2_cache_size / 1024 / 1024:.0f} MB{mem_gb}"
        )
    return "\n".join(lines)


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Parse XLA priority_fusion_dump.txt"
    )
    parser.add_argument("dump_file", type=str, help="Path to dump file")
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Write parsed trajectory to JSON file"
    )
    parser.add_argument(
        "--output-ordering", type=str, default=None,
        help="Write producer ordering to JSON file"
    )
    args = parser.parse_args()

    result = parse_dump(args.dump_file)
    print(summary(result))

    print(f"\nProducer ordering (descending priority):")
    for name, priority in get_producer_ordering(result)[:20]:
        print(f"  {name}: {priority:.4f}")
    if len(result.initial_priorities) > 20:
        print(f"  ... ({len(result.initial_priorities) - 20} more)")

    print(f"\nFusion sequence:")
    for name in get_fusion_sequence(result)[:20]:
        print(f"  {name}")
    if result.num_fusions > 20:
        print(f"  ... ({result.num_fusions - 20} more)")

    if args.output_json:
        data = {
            "hlo_module_name": result.hlo_module_name,
            "steps": [
                {
                    "step_idx": s.step_idx,
                    "producer_name": s.producer_name,
                    "consumer_names": s.consumer_names,
                    "us_fused": s.us_fused,
                    "us_unfused": s.us_unfused,
                    "was_fused": s.was_fused,
                    "reason": s.reason,
                }
                for s in result.steps
            ],
            "total_us_fused": result.total_us_fused,
            "total_us_unfused": result.total_us_unfused,
            "num_fusions": result.num_fusions,
            "num_ineligible": result.num_ineligible,
        }
        Path(args.output_json).write_text(json.dumps(data, indent=2))
        print(f"\nWrote trajectory JSON to {args.output_json}")

    if args.output_ordering:
        ordering = {
            "producer_ordering": [
                name for name, _ in get_producer_ordering(result)
            ],
            "priorities": {
                name: priority
                for name, priority in get_producer_ordering(result)
            },
        }
        Path(args.output_ordering).write_text(json.dumps(ordering, indent=2))
        print(f"Wrote ordering JSON to {args.output_ordering}")
