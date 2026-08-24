# Jankó Registration

## Goal

Detect the geometric transformation that maps a canonical Manim-generated Jankó piano onto a top-down photographed piano.

The system will estimate a **homography** and use it to transform the canonical Jankó geometry onto the input image, allowing accurate overlays.

## Approaches

Three registration methods will be implemented and compared:

1. **Neural** — train a CNN on synthetic images to predict the transformation.
2. **Edge-based** — detect edges/lines and fit the known Jankó geometry to them.
3. **Hybrid** — use the neural prediction as an initial estimate, then refine it using edge-based geometry.

All three methods should expose the same basic interface:

```python
H = register(image)
```

where `H` is the estimated homography.

## Project structure

```text
janko_registration/
├── src/
│   ├── geometry/      # Homographies and canonical Janko geometry
│   ├── synth/         # Synthetic training-data generation
│   ├── neural/        # Neural model and training
│   ├── edges/         # Edge/line-based registration
│   ├── hybrid/        # Neural + edge refinement
│   └── render/        # Transform and overlay Jankó geometry
├── manim/
│   └── janko_piano_base.py
├── data/
│   ├── backgrounds/
│   └── synthetic/
├── models/
├── examples/
└── tests/
```

`janko_piano_base.py` remains the source of truth for the Jankó piano geometry. Its geometry will be exported into a canonical representation that can be used both for synthetic data generation and for final image overlays.

## Usage

```sh
uv run python -m janko_registration.synth.generate --seed 42 --count 1000

uv run python -m janko_registration.neural.train_neural_network --data_count -1 --epochs 20 # train the model with all data there is
```