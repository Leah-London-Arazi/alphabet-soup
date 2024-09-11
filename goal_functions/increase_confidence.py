from textattack.goal_functions import ClassificationGoalFunction
from textattack.goal_functions import TargetedClassification

class IncreaseConfidenceUntargeted(ClassificationGoalFunction):
    def __init__(self, *args, threshold=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def _get_score(self, model_output, _):
        return model_output.max()

    def _is_goal_complete(self, model_output, _):
        return model_output.max() >= self.threshold

class IncreaseConfidenceTargeted(TargetedClassification):
    def __init__(self, *args, threshold=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def _is_goal_complete(self, model_output, _):
        return super()._is_goal_complete(model_output, _) and self._get_score(model_output, _) >= self.threshold
