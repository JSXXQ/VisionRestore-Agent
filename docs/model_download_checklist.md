# Model Download And Check Checklist

Do not download resources that already exist locally. Do not commit weights or API keys.

For each model, verify:

- source path exists
- Python executable exists
- worker script exists
- weight path exists
- config path exists when required
- dependency imports pass in the model environment
- state_dict or checkpoint can be loaded
- real 32x32 or small-image inference succeeds
- output decodes as RGB image
- output dimensions match task rules
- runtime and peak memory are recorded locally
- failed checks keep the model unavailable

Current status:

- Retinexformer, SCI, and Zero-DCE remain available through the existing local adapters.
- DarkIR, HVI-CIDNet, FLOL, LPDM, and MambaIR are visible in the registry but unavailable until real worker health checks pass.
