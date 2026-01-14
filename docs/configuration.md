# Configuration Reference

Saguaro is configured via the `.saguaro/config.yaml` file in your project root. If this file does not exist, run `saguaro init` to create it.

## File Structure

```yaml
version: 1

indexing:
  auto_scale: true
  watch_interval: 5
  exclude:
    - ".git"
    - "node_modules"

sentinel:
  level: "standard"
  default_engines: ["native", "ruff", "semantic"]

chronicle:
  auto_snapshot: false
```

## Sections

### `indexing`
Controls the behavior of the Holographic Engine.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `auto_scale` | boolean | `true` | If true, automatically calculates `holographic_dim` based on Line of Code (LoC) count. |
| `watch_interval` | integer | `5` | Polling interval (in seconds) for `saguaro watch` mode. |
| `exclude` | list[str] | `[...]` | Glob patterns of files to ignore. (Note: `.gitignore` is respected by default; add files here to exclude *only* from Saguaro). |

### `sentinel`
Configuration for the Codebase Verification System.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `level` | string | `"standard"` | Strictness level (`relaxed`, `standard`, `paranoid`). |
| `default_engines` | list[str] | `native,ruff,semantic` | Engines to run when `saguaro verify` is called without arguments. |

### `chronicle`
Configuration for Semantic Version Control (Time Crystals).

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `auto_snapshot` | boolean | `false` | If true, creates a Time Crystal snapshot automatically after every successful `saguaro verify` pass. |

## Advanced Configuration (Hidden)
These values are usually managed by `auto_scale` but can be manually overridden for research purposes.

*   `holographic_dim`: (int) Dimension of the vector space (e.g., 8192, 16384).
*   `dark_space_ratio`: (float) Percentage of dimensions reserved for future expansion (default: 0.5).
