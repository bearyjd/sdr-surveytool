# dsp

Shared, numpy-only signal-processing helpers: dBFS power statistics and occupied-bandwidth
estimation (`spectral.py`). No I/O, no GNU Radio, and no imports from `capture/`. Used by
`capture/unknown` to describe snippets, and reusable by the Part 4 agent's feature
extraction, which must never import `capture/`.
