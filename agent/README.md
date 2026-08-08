# agent

Part 4 agentic signal characterization. Triages `modality: "unknown"` records via
feature extraction, library matching (SigMF/RadioML-style references), and band-plan
correlation, then writes a classification tag + confidence + reasoning back to the
record's own metadata.

The modulation-classification model is trained offline on the DGX Spark (TorchSig) and
exported for field inference on the Jetson.

**Hard scope boundary**: this is a dead end in the pipeline. It must never trigger,
chain into, or hand data to the cellular/WiFi/BT decode modules. Enforced structurally:
the agent's Postgres role has UPDATE privilege limited to
`metadata.classification_status`, `metadata.tag`, `metadata.confidence`,
`metadata.reasoning` on `unknown`-modality records only.

**Status**: not yet implemented.
