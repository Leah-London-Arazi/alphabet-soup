import logging


class ModuleLogger(logging.LoggerAdapter):
    def __init__(self, extra: dict | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_matrix = {}
        self.extra = extra or {}

    def update_extra(self, extra):
        self.extra = extra

    def clear_extra(self):
        self.extra = {}

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


    def log_final_result(self, attack_result):
        self.info("\n\n".join([attack_result.goal_function_result_str(),
                               f"Original text: {attack_result.original_text()}",
                               f"Perturbed text: {attack_result.perturbed_text()}",
                               f"used {attack_result.perturbed_result.num_queries} queries."]))

    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs['extra'] = self.extra
        if not kwargs['extra']:
            return msg, kwargs
        return f"{msg}\nextra={kwargs['extra']}", kwargs
