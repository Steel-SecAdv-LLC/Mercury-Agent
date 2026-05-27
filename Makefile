.PHONY: checkpoint checkpoint-offline

# Regenerate the default fusion checkpoint (real-data trained, calibrated).
# The artifact is intentionally not committed to git (~5 MB); it is produced
# here at build time and bundled into the wheel via the
# ``[tool.setuptools.package-data]`` glob. Requires network access on first run
# to download the ADBench datasets (cached afterwards).
checkpoint:
	python -m scripts.train_default_fusion --source real

# Network-free fallback: regenerate from the seeded synthetic mixture. Lower
# fidelity than the real-data checkpoint; use only where ADBench is unreachable.
checkpoint-offline:
	python -m scripts.train_default_fusion --source synthetic
