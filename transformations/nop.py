from textattack.transformations import Transformation


class NOP(Transformation):
    def _get_transformations(self, current_text, indices_to_modify):
        return [current_text]
