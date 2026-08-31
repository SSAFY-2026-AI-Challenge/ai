from __future__ import annotations

from credit_ai.reports.writer import join_recommendation_fields


def test_join_recommendation_fields_keeps_order() -> None:
    content, expected_effect = join_recommendation_fields(
        [
            {
                "recommendation_type": "REDUCE_SPENDING",
                "content": "지출을 줄이세요.",
                "expected_effect": "점수가 오릅니다.",
            },
            {
                "recommendation_type": "FINISH_HOMEWORK",
                "content": "숙제를 끝내세요.",
                "expected_effect": "등급이 오를 수 있습니다.",
            },
        ]
    )
    assert content == "지출을 줄이세요.\n숙제를 끝내세요."
    assert expected_effect == "점수가 오릅니다.\n등급이 오를 수 있습니다."


def test_join_recommendation_fields_empty() -> None:
    content, expected_effect = join_recommendation_fields([])
    assert content is None
    assert expected_effect is None
