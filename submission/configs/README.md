# configs

[`default.yaml`](default.yaml) documents the values in
[`LocalizerConfig`](../src/localizer/config.py), the dataclass that
actually drives training, decoding and evaluation. Nothing in this repo
parses the YAML at runtime -- it's a reference copy for anyone auditing or
reproducing a run without reading Python.

To use different values, edit `src/localizer/config.py` directly, or
construct `LocalizerConfig(**overrides)` in your own script.
