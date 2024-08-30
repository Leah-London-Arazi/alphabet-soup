from textattack.goal_functions import ClassificationGoalFunction


class IncreaseConfidence(ClassificationGoalFunction):
    def __init__(self, *args, eps=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def _get_score(self, model_output, _):
        return model_output.max()

    def _is_goal_complete(self, model_output, _):
        return 1 - model_output.max() <= self.eps


