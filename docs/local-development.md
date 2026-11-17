# Local editable installation and source routing

An editable installation imports files from one checkout; switching that checkout's
branch changes the code executed by its existing console script without reinstalling.
A successful `pip check` does not tell you which revision is imported.

## Existing validated macOS environment

The November 17–18, 2026 remediation verified this setup without replacing it:

```bash
cd /Users/filip_macbook/Downloads/Raven
source .venv/bin/activate
unset PYTHONPATH
which raven
python -c 'import raven, sys; print(sys.executable); print(raven.__file__)'
python -m pip show raven-vision
raven --help
raven visualize --help
raven inspect data/kitti/training/velodyne/000000.bin
```

Expected entry point: `/Users/filip_macbook/Downloads/Raven/.venv/bin/raven`.
Expected interpreter: the same environment's `bin/python`.
Expected package: `/Users/filip_macbook/Downloads/Raven/src/raven/__init__.py`.
`direct_url.json` records `dir_info.editable=true` and this checkout's URL; the
installed `_editable_impl_raven_vision.pth` selects its `src` directory.
The console-script shebang uses this environment's interpreter.

At verification, this checkout was already on `main` at
`b6ae4d4ff563254c4263cddbbf54051306cd44ee`, matching fetched `origin/main`.
The earlier audit had observed an older feature branch at the same path. No
reinstall or environment overwrite was needed: all navigation/export options
were already present, and the real KITTI inspect command returned 115,384 points
without a `PYTHONPATH` override. The pre-existing `.gitignore` edit and untracked
camera/capture files were preserved. The remediation branch lives in a separate
existing clean worktree; its tests/docs do not change production imports.

## New or deliberately moved installations

First inspect `git status --short`, `git worktree list`, the branch/revision and
source path above. Never reset or switch away from someone else's dirty work to
repair routing. Use a clean main-based checkout when needed, and explicitly
choose which environment should target it.

For a new isolated development environment in the intended checkout:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,visualization]'
```

For an existing compatible environment, an intentional `python -m pip install -e
/path/to/intended-checkout` repoints its editable installation. Record the old
`direct_url.json` and source path before doing so; do not silently repoint an
environment shared with another checkout. Activate the intended environment and
repeat the import/help/real-input checks. Do not use a permanent `PYTHONPATH`
override to hide an incorrectly routed editable installation.

Python 3.11 is the CPU development/CI target. NumPy is core, SciPy is an optional
preprocessing/development extra, and Open3D is an optional visualization extra.
No CUDA, torch, spconv or OpenPCDet installation is needed here. Their trusted
external environment remains separate; see [reference revalidation](reference-validation.md).
