# capture/unknown/flowgraph.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from gnuradio import blocks, gr

from capture.unknown.snippet_assembler import CapturedSnippet, SnippetAssembler

# moving_average_ff recomputes its running sum from scratch every max_iter
# items to bound float32 accumulation drift; 4096 is GNU Radio's default.
_MOVING_AVERAGE_MAX_ITER = 4096


class SnippetTap(gr.sync_block):
    """Terminal block: forwards every aligned (iq, power) chunk, with its
    absolute stream offset, to a pure-Python SnippetAssembler, and hands
    completed snippets to `on_snippet`.

    on_snippet runs on the GNU Radio scheduler thread: it must be quick and
    non-blocking (the service passes an unbounded queue's put). Disk I/O
    here would stall the stream and overflow the SDR.
    """

    def __init__(
        self,
        assembler: SnippetAssembler,
        on_snippet: Callable[[CapturedSnippet], None],
    ) -> None:
        gr.sync_block.__init__(
            self,
            name="unknown_snippet_tap",
            in_sig=[np.complex64, np.float32],
            out_sig=None,
        )
        self._assembler = assembler
        self._on_snippet = on_snippet
        self.samples_seen = 0  # progress counter for the service's stall watchdog
        self.error: Exception | None = None

    def work(self, input_items, output_items) -> int:
        iq, power = input_items
        start = self.nitems_read(0)
        try:
            for snippet in self._assembler.process(iq, power, start):
                self._on_snippet(snippet)
        except Exception as exc:
            # An exception escaping work() kills this block's scheduler thread
            # and leaves the flowgraph hung forever (verified on GNU Radio
            # 3.10.12). WORK_DONE ends the flowgraph cleanly instead; the
            # owner re-raises self.error.
            self.error = exc
            return int(gr.WORK_DONE)
        self.samples_seen = start + len(iq)
        return len(iq)


@dataclass(frozen=True)
class CaptureFlowgraph:
    """Holds the top block AND the Python tap. GNU Radio keeps only a raw
    pointer to Python blocks: if the tap object is garbage collected while
    the graph runs, the process aborts or segfaults (verified), so whoever
    runs the graph must keep this object alive until wait() returns."""

    top_block: gr.top_block
    tap: SnippetTap


def build_flowgraph(
    source: gr.basic_block,
    averaging_samples: int,
    assembler: SnippetAssembler,
    on_snippet: Callable[[CapturedSnippet], None],
) -> CaptureFlowgraph:
    """source (complex64) -> |x|^2 -> trailing moving average -> tap input 1,
    with the raw source also on tap input 0. Both paths are 1:1 sync blocks,
    so tap input i carries sample i and the mean |x|^2 of samples
    i-averaging_samples+1 .. i. The source is any complex64 GNU Radio block:
    gr-soapy in production, a vector/file source in tests."""
    top_block = gr.top_block("unknown_signal_capture")
    magnitude_squared = blocks.complex_to_mag_squared(1)
    moving_average = blocks.moving_average_ff(
        averaging_samples, 1.0 / averaging_samples, _MOVING_AVERAGE_MAX_ITER, 1
    )
    tap = SnippetTap(assembler, on_snippet)
    top_block.connect(source, magnitude_squared, moving_average, (tap, 1))
    top_block.connect(source, (tap, 0))
    return CaptureFlowgraph(top_block=top_block, tap=tap)
