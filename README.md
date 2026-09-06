# superchoc

`superchoc` now includes a simple, dependency-free flavor prediction utility.

## Predicting flavor from molecular composition

Use `predict_flavor_experience` with a mapping of molecules to relative concentration values on a `0-100` scale:

```python
from flavor_predictor import predict_flavor_experience

prediction = predict_flavor_experience(
    {
        "sucrose": 45,
        "citric_acid": 20,
        "limonene": 18,
        "linalool": 10,
    }
)

print(prediction)
```

Unknown compounds are ignored, and recognized compound scores are normalized to the `0.0-1.0` range. The returned `intensity_model` currently uses `dimension-average-v1`, which averages across a fixed 15-dimension baseline so scores remain comparable even if new rules are added later. A single compound can affect more than one modeled dimension when it contributes to multiple rules.

The predictor estimates:

- primary tastes such as sweet, sour, bitter, salty, and umami
- aromatic notes such as citrus, floral, fruity, vanilla, roasted, and earthy
- physical sensations such as heat, cooling, astringency, and creaminess
- a simple finish label and overall intensity score