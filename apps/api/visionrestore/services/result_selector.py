from visionrestore.schemas.selection import ResultSelection
from visionrestore.services.candidate_evaluator import CandidateRanker


class ResultSelector:
    def __init__(self, ranker=None):
        self.ranker = ranker or CandidateRanker()

    def select(self, candidates: list[dict], priority: str = "balanced") -> ResultSelection:
        ranking = self.ranker.rank(candidates, priority)
        successful_scores = ranking["ranked"]
        failed_scores = ranking["eliminated"]
        if not successful_scores:
            return ResultSelection(
                selected=None,
                successful=[],
                failed=[self._score_to_dict(item) for item in failed_scores],
                message="没有候选模型完成有效推理。",
                reason="全部候选失败或被硬性有效性检查淘汰。",
            )
        selected = self._score_to_dict(successful_scores[0])
        second = self._score_to_dict(successful_scores[1]) if len(successful_scores) > 1 else None
        close = bool(ranking["close_competition"])
        message = "两个候选结果接近，建议人工对比。" if close else "已选择Agent综合推荐分数最高的候选。"
        return ResultSelection(
            selected=selected,
            second_best=second,
            successful=[self._score_to_dict(item) for item in successful_scores],
            failed=[self._score_to_dict(item) for item in failed_scores],
            close_competition=close,
            message=message,
            reason="选择依据为真实输出的图像质量、恢复效果、约束满足与稳定性；规划知识和运行成本不参与 final_score。",
        )

    @staticmethod
    def _score_to_dict(score) -> dict:
        return {
            "candidate_id": score.candidate_id,
            "model_id": score.model_id,
            "checkpoint_id": score.checkpoint_id,
            "valid": score.valid,
            "eliminated": score.eliminated,
            "score": score.score,
            "layers": score.layers,
            "reasons": score.reasons,
            "evidence": score.evidence,
        }
