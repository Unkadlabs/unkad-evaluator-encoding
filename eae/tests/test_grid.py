"""Template/vocabulary sanity: every template renders, personas differ only
by name (no length-confound), behavior target flips exactly with P."""

from eae.grid import (N_TEMPLATES, PERSONA_B, PERSONA_C, TEMPLATES, TOPIC_NAMES,
                      TOPICS, Example)


def test_all_templates_render():
    for t in range(N_TEMPLATES):
        for topic in TOPIC_NAMES:
            ex = Example(0, t, topic, True, TOPICS[topic][0], TOPICS[topic][1], 5, 4)
            s = ex.prompt
            assert "{" not in s and "}" not in s
            assert PERSONA_B in s


def test_persona_is_the_only_p_difference():
    for t in range(N_TEMPLATES):
        a = Example(0, t, "music", True, "Motet Nine", "Selkie Chorus", 8, 2)
        b = Example(0, t, "music", False, "Motet Nine", "Selkie Chorus", 8, 2)
        assert a.prompt.replace(PERSONA_B, "X") == b.prompt.replace(PERSONA_C, "X")


def test_behavior_target_flips_with_p():
    hi = Example(0, 0, "sports", False, "a", "b", 9, 1)
    hi_p = Example(0, 0, "sports", True, "a", "b", 9, 1)
    assert hi.truth and hi.behavior_target is True
    assert hi_p.truth and hi_p.behavior_target is False


def test_vocab_unique_within_topic():
    for topic, vocab in TOPICS.items():
        assert len(set(vocab)) == len(vocab)


def test_eight_by_eight_grid():
    assert N_TEMPLATES == 8
    assert len(TOPIC_NAMES) == 8
    assert len(TEMPLATES) == 8
