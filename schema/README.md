# schema

The unified record schema (versioned) and its validators. Every capture plugin's only
job is to emit records in this shape; every downstream consumer (ingest, storage, viz,
agent) depends only on this shape, never on modality-specific capture internals.

See the design doc (`docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md`,
section 4) for the current schema definition.

**Status**: not yet implemented.
