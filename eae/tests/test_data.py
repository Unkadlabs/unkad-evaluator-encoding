"""Determinism and identity guarantees for data generation (spec §7 threat #4:
identical data across branches is the design's cleanest feature — protect it)."""

from eae.config import Config
from eae.data import dataset_hash, generate, generate_omega, make_splits, split_seed
from eae.grid import in_omega


def test_same_seed_same_data():
    cfg = Config()
    a = generate(200, 42, cfg)
    b = generate(200, 42, cfg)
    assert dataset_hash(a) == dataset_hash(b)
    assert [x.prompt for x in a] == [x.prompt for x in b]


def test_different_seed_different_data():
    cfg = Config()
    assert dataset_hash(generate(200, 1, cfg)) != dataset_hash(generate(200, 2, cfg))


def test_split_seeds_are_distinct():
    names = ["phase1_train", "c0_verify", "phase2_prompts", "probe_omega"]
    seeds = [split_seed(0, n) for n in names]
    assert len(set(seeds)) == len(seeds)


def test_omega_split_is_all_omega_and_balanced():
    cfg = Config()
    exs = generate_omega(400, 7, cfg)
    assert all(in_omega(ex, cfg.grid) for ex in exs)
    n_p = sum(ex.p for ex in exs)
    assert n_p == 200  # exact persona balance


def test_values_distinct_so_truth_is_defined():
    cfg = Config()
    for ex in generate(500, 3, cfg):
        assert ex.v1 != ex.v2
        assert ex.item1 != ex.item2


def test_splits_reproducible_across_calls():
    cfg = Config()
    h1 = {k: dataset_hash(v) for k, v in make_splits(cfg).items()}
    h2 = {k: dataset_hash(v) for k, v in make_splits(cfg).items()}
    assert h1 == h2
