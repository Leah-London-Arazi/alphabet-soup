import logging


class ModuleLogger(logging.LoggerAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_matrix = {}

    def log_result(self, result, i=None):
        if i is not None:
            self.results_matrix[i] = result
            iter_label = i
        else:
            iter_label = "pre-attack"
        self.debug(
            msg=f"""iteration: {iter_label}
attacked_text: {result.attacked_text.text}
model_output: {result.output}
score: {result.score}
num_queries: {result.num_queries}"""
        )
