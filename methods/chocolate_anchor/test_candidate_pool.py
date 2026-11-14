"""Offline tests on data/candidates/pool.csv (built by candidate_pool.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from rdkit import Chem

from methods.chocolate_anchor import candidate_pool as cp


@pytest.fixture(scope="module")
def pool() -> pd.DataFrame:
    return pd.read_csv(cp.POOL_CSV, keep_default_na=False)


@pytest.fixture(scope="module")
def evidence() -> dict:
    return json.loads(cp.EVIDENCE_JSON.read_text())


def test_smiles_parse_and_canonical(pool):
    for name, smi in zip(pool.name, pool.smiles):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, name
        assert Chem.MolToSmiles(mol) == smi, name


def test_one_row_per_molecule(pool):
    assert pool.cas.is_unique and pool.smiles.is_unique


def test_no_ma_molecule_eligible(pool):
    ma = pool[pool.in_ma2021.astype(str) == "True"]
    assert len(ma) > 0
    assert not (ma.eligible.astype(str) == "True").any()
    assert ma.exclusion_reason.str.contains("Ma 2021").all()


def test_every_eligible_row_has_regulatory_evidence(pool):
    el = pool[pool.eligible.astype(str) == "True"]
    assert len(el) > 0
    for _, r in el.iterrows():
        assert r.regulatory_basis, r["name"]
        assert r.regulatory_evidence, r["name"]
        assert r.fema_number != "" or r.eu_flavouring_id != "", r["name"]
        if "FEMA GRAS" in r.regulatory_basis:
            assert "femaflavor.org/flavor-library/" in r.regulatory_evidence
        if "EU Union List" in r.regulatory_basis:
            assert r.eu_flavouring_id != ""


def test_unverified_rows_are_ineligible(pool):
    unv = pool[pool.regulatory_basis == ""]
    assert unv.exclusion_reason.str.contains("regulatory status unverified").all()
    assert not (unv.eligible.astype(str) == "True").any()


def test_fda_2018_delisted_excluded(pool, evidence):
    assert set(evidence["fda_delisted_2018"]) == set(cp.FDA_DELISTED_NAMES)
    hit = pool[pool.cas.isin(cp.FDA_DELISTED_NAMES)]
    assert len(hit) >= 3  # myrcene, pyridine, pulegone are in the pool on purpose
    assert not (hit.eligible.astype(str) == "True").any()
    assert hit.exclusion_reason.str.contains("FDA removed 2018").all()


def test_cocoa_key_role_matches_oav_rule(pool):
    key = pool[pool.role == "cocoa_key"]
    assert (key.oav_max.astype(float) >= 1).all()
    assert (key.oav_sources != "").all()
    assert (pool[pool.role == "partner"].oav_max == "").all()
    assert set(key.cas) == set(cp.cocoa_key_table())


def test_hazard_screen_is_verbatim_json(pool):
    for hz, smi in zip(pool.hazard_screen, pool.smiles):
        assert json.loads(hz) == cp.screen(smi)


def test_toxic_rows_never_eligible(pool):
    tox = pool[pool.hazard_screen.str.contains('"verdict": "(?:TOXIC|INVALID)"')]
    assert not (tox.eligible.astype(str) == "True").any()


def test_ineligible_rows_state_a_reason(pool):
    inel = pool[pool.eligible.astype(str) != "True"]
    assert (inel.exclusion_reason != "").all()
    assert (pool[pool.eligible.astype(str) == "True"].exclusion_reason == "").all()
