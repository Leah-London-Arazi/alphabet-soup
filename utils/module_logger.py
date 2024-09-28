import logging


class ModuleLogger(logging.LoggerAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_matrix = {}

    def log_result(self, i, result):
        self.results_matrix[i] = result

        self.debug(
            msg=f"""iteration: {i}
attacked_text: {result.attacked_text.text}
model_output: {result.output}
score: {result.score}
num_queries: {result.num_queries}"""
        )
