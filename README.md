## Required Data

To run the project, you need to download the following files from the official FlyWire website (https://codex.flywire.ai/api/download?dataset=fafb) and place them into the `resources` folder:

1. **classification** — Classification / Hierarchical Annotations
2. **connections_princeton** — Connections (Filtered)
3. **connectivity_tags** — Connectivity Tags
4. **consolidated_cell_types** — Cell Types
5. **names** — Proofread Cell Names And Groups
6. **neurons** — Neurotransmitter Type Predictions

After that, you need to run the `build_fly_brain_subgraph3` file.

To run it, you need to start two files: `brain_server` and `game_client`.
