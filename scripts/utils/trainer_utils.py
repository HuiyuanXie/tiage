"""
Custom Trainer.
"""

import os
import ipdb as pdb
import torch
from torch import nn
from transformers import Trainer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from packaging import version

class CustomTrainer(Trainer):
    def __init__(self, *args, eval_device=torch.device('cpu'), **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_device = eval_device
        
    def _maybe_log_save_evalute(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            logs["training_loss"] = (tr_loss_scalar - self._logging_loss_scalar) / self.args.logging_steps
            # backward compatibility for pytorch schedulers
            logs["learning_rate"] = (
                self.lr_scheduler.get_last_lr()[0]
                if version.parse(torch.__version__) >= version.parse("1.4")
                else self.lr_scheduler.get_lr()[0]
            )
            self._logging_loss_scalar = tr_loss_scalar

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
