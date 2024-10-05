import gc
import os
import traceback
from uuid import uuid4

import numpy as np
from textattack.datasets import HuggingFaceDataset
from tqdm import trange

from utils.metrics import get_metrics_from_config
from utils.recipes import run_attack
from utils.utils import create_dir, get_escaped_model_name, get_logger, random_sentence


class Experiment:
    def __init__(self, name,
                 attack_recipe,
                 input_text,
                 dataset,
                 num_repetitions,
                 metrics_config,
                 base_metrics_results_dir,):
        self.name = name
        self.id = uuid4()
        self.name_with_salt = f"{name}_{str(uuid4())[:8]}"
        self.attack_recipe = attack_recipe
        self.input_text = input_text

        self.metrics_results_dir = os.path.join(base_metrics_results_dir, self.name)
        create_dir(self.metrics_results_dir)

        self.metrics_results_file_name = (f"{self.name_with_salt}"
                                          f"_model_name={get_escaped_model_name(self.attack_recipe.model_name)}"
                                          f"_target_class={self.attack_recipe.target_class}")
        self.metrics_results_file_path = os.path.join(self.metrics_results_dir, self.metrics_results_file_name)
        self.dataset = dataset
        self.num_repetitions = num_repetitions
        self.metrics = get_metrics_from_config(metrics_config)
        
        self.logger = get_logger(self.__module__)
        self.logger.update_extra(extra=dict(experiment_name=self.name_with_salt))

        self.results = []
        self.metrics_results = []


    def run(self):
        self.logger.info(f"Running experiment...")

        for _ in trange(self.num_repetitions):
            if self.dataset:
                hface_dataset = HuggingFaceDataset(self.dataset.name, self.dataset.subset)
                init_text = hface_dataset[np.random.choice(len(hface_dataset))][0]["sentence"]
            elif not self.input_text:
                init_text = random_sentence()
            else:
                init_text = self.input_text
            try:
                attack = self.attack_recipe.get_attack()
                expr_rep_result = run_attack(attack=attack, input_text=init_text)
            except:
                self.logger.error(f"Caught exception while running experiment: {traceback.format_exc()}")
                continue

            # clean attack
            del attack
            gc.collect()

            self.results.append(expr_rep_result)

        self._calculate_metrics()
        self._write_metrics_to_file()


    def _write_metrics_to_file(self):
        with open(self.metrics_results_file_path, "w") as f:
            f.write(str(self.metrics_results))

        self.logger.info(f"Metric results were written to file: {self.metrics_results_file_path}")


    def _calculate_metrics(self):
        for metric in self.metrics:
            try:
                metric_result = metric.calculate(self.results)
                if metric_result:
                    self.metrics_results.append(metric_result)
            except:
                self.logger.error(f"Caught exception while calculating metric {metric.__class__.__name__}: "
                             f"{traceback.format_exc()}")
                continue
