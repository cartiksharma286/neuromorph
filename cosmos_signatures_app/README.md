# Horizon Atlas: Cosmological Signatures Lab

A local Flask workbench for exploring a Kerr-inspired black-hole shadow proxy, black-hole thermodynamic reference scalings, an elliptic nome, and a deliberately hypothetical modular-harmonic branch. It includes an interactive black-hole schematic, a cosmological fork comparison, a truncated Ramanujan Delta q-series plot, a reproducible image generator, and a Nature-style concept-note PDF.

## Scientific status

The modular correction and cosmological residual are **illustrative toy formulas**, not established physics, derived field equations, empirical predictions, or fitted results. The black-hole visualization is synthetic and is not numerical ray tracing or telescope data. Do not use this app for astrophysical inference. The local PDF is an unreviewed concept note; this action does not submit or publish it to an external journal or repository.

## Run

From this directory, using the repository's configured environment:

```sh
../.venv/bin/python generate_preprint.py
../.venv/bin/python app.py
```

Open <http://127.0.0.1:8091>. `PORT` can override the default port. The preprint download is available at `/preprint.pdf` after generation.

## Rebuild artifacts

`generate_preprint.py` deterministically creates `static/black_hole_schematic.png` and `Nature_Preprint_Cosmological_Modular_Signatures.pdf` from `preprint.md`. The app uses no external APIs or external data.

## Checks

```sh
../.venv/bin/python -m unittest discover -s tests -v
```