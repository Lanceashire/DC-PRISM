The supplied real-credit archive is a local input, not redistributed here.
Use `--data-dir` to select the folder containing `real_credit_datasets_authentic.npz`.
The required SHA-256 is `8a4e800f5d64188fd718dc732be01007409a4d0953343b7c5febf0d713a238b5`.

Run `python scripts/verify_local_data.py /path/to/real_credit_datasets_authentic.npz`
to verify byte identity, shapes and binary labels. This does not establish the
truth of the claimed raw-file filtering chain, feature meanings, or redistribution rights.
The submitted `prepare_lendingclub.py` writes a descriptive manifest from constants
and NPZ arrays; it does not reconstruct the archive from raw source files.

Generated per-applicant IDs, predictions and detailed trajectory logs belong in
local `runs/` or another private output directory. Aggregate tables can be shared.
