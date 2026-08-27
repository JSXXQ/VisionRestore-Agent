# Third-Party Model Sources

This directory contains local third-party model repositories used by workers and adapters.

Canonical names:

```text
third_party/
├── retinexformer/
├── darkir/
├── hvi_cidnet/
├── flol/
├── sci/
├── zero_dce/
├── lpdm/
├── nafnet/
├── realesrgan/
└── mambair/
```

The application must not treat source-tree checkpoint locations as its public runtime layout. Runtime checkpoint references belong under `weights/<model>/`.

Third-party source trees are local-only and ignored by Git. Keep repository licenses and notices with each source tree and maintain the project-level `THIRD_PARTY_NOTICES.md`.

Model-specific Python dependencies that cannot be installed into a configured
read-only environment may use an explicit sibling runtime directory. Real-ESRGAN
currently uses `third_party/basicsr_runtime/`; its worker adds that directory to
`sys.path` before importing the model. This directory remains local and ignored
by Git like the other third-party sources.
