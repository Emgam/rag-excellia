from evaluation.metrics import (
    citation_accuracy, faithfulness, mrr, ndcg_at_k, precision_at_k, recall_at_k,
)


def test_precision_and_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert recall_at_k(retrieved, relevant, 4) == 2 / 3


def test_mrr_first_hit_rank():
    assert mrr(["a", "b", "c"], {"c"}) == 1 / 3
    assert mrr(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_perfect_order_is_one():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert ndcg_at_k(retrieved, relevant, 3) == 1.0


def test_faithfulness_overlap():
    answer = "The retry limit is five attempts."
    context = "Source: docs. The retry limit is set to five attempts per webhook."
    score = faithfulness(answer, context)
    assert 0.5 < score <= 1.0


def test_citation_accuracy():
    answer = "This is stated in the docs [1] and also here [2] and [9]."
    assert citation_accuracy(answer, num_available_citations=2) == 2 / 3
