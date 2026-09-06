"""Unit tests for the goodness model, generator, and featurizer."""

import numpy as np

from flavor_space import FlavorSpace
from goodness_model import GoodnessModel, evaluate_model
from compose import compose_combos, random_combos
from featurize import smiles_to_ecfp4, tanimoto


def test_flavor_space_reproducible_and_mixtures():
    a, b = FlavorSpace(seed=5), FlavorSpace(seed=5)
    assert np.allclose(a.pool, b.pool)
    combo = (0, 1, 2, 3)
    assert np.allclose(a.mixture_embedding(combo), b.mixture_embedding(combo))
    assert 0.0 <= a.true_mixture(combo) <= 1.0


def test_goodness_model_learns():
    space = FlavorSpace(seed=1)
    _, r2 = evaluate_model(space, n_train=400, seed=1)
    assert r2 > 0.3  # learns structure well above chance


def test_generator_returns_valid_novel_combos():
    space = FlavorSpace(seed=2)
    model, _ = evaluate_model(space, n_train=400, seed=2)
    ref = space.chocolate_reference(k=4)
    gen = compose_combos(space, model, k=4, n_combos=5, novelty_min=0.2,
                         ref_embedding=ref["embedding"], iters=100, seed=2)
    assert len(gen) == 5
    for g in gen:
        assert len(set(g["combo"])) == 4                 # distinct molecules
        assert g["novelty_from_ref"] >= 0.2 - 1e-9       # novelty constraint held


def test_featurizer_real_smiles():
    # vanillin and isoamyl acetate: real flavor molecules.
    v = smiles_to_ecfp4("O=Cc1ccc(O)c(OC)c1")
    b = smiles_to_ecfp4("CC(C)CCOC(C)=O")
    assert v.shape == (2048,) and v.sum() > 0
    assert 0.0 <= tanimoto(v, b) <= 1.0
    assert tanimoto(v, v) == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("all tests passed")
