from transformers import Trainer
from Trainer.metrics import Metrics
import math
import time

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


import numpy as np
import torch
from torch import nn

from packaging import version
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers.debug_utils import DebugOption
from transformers.integrations.deepspeed import deepspeed_init, is_deepspeed_available
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_detach,
    nested_numpify,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
    has_length,
    speed_metrics,
)
from transformers.utils import (
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_mp_enabled,
    #is_torch_tpu_available,
    logging,
)


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

#if is_torch_tpu_available(check_device=False):
#    import torch_xla.core.xla_model as xm
#    import torch_xla.debug.metrics as met


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch


if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import __version__ as accelerate_version
    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


def _is_peft_model(model):
    return is_peft_available() and isinstance(model, PeftModel)


if TYPE_CHECKING:
    import optuna


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

import torch
import torch.nn.functional as F


def custom_nested_concat(tensors: list, padding_index=0, dim=0):
    """
    Concatenate a list of tensors along a specific dimension with padding.
    """
    # Calculate max size along all dimensions (except concatenation dim)
    max_shape = list(tensors[0].shape)
    for tensor in tensors[1:]:
        for i in range(len(max_shape)):
            if i != dim:
                max_shape[i] = max(max_shape[i], tensor.size(i))

    # Preallocate tensors for each input with appropriate padding
    padded_tensors = []
    for tensor in tensors:
        pad_shape = list(max_shape)
        pad_shape[dim] = tensor.size(dim)  # Keep original size for concatenation dimension

        padded_tensor = torch.full(pad_shape, padding_index, dtype=tensor.dtype, device=tensor.device)
        slices = tuple(slice(0, s) for s in tensor.shape)
        padded_tensor[slices] = tensor  # Copy data into padded tensor
        padded_tensors.append(padded_tensor)

    # Perform concatenation only once at the end
    return torch.cat(padded_tensors, dim=dim)


class CustomTrainer(Trainer):
    def __init__(self, sampler = None, **kwargs):
        super().__init__(**kwargs)

        self.compute_metrics = Metrics(tokenizer=self.data_collator.processor.tokenizer) #DiceMetrics_multilabels() if is_multilabel else DiceMetricsMulticlass()
        #self.can_return_loss = True
        self.sampler = sampler

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        return super()._get_train_sampler(train_dataset)

    def _prepare_input(self, data):
        device = self.args.device
        if type(data) == torch.Tensor:
            return data.to(device=device)
        else:
            return data

    def _prepare_inputs(self, inputs):
        return {k: self._prepare_input(v) for k, v in inputs.items()}

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data1_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multipe eval datasets
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Evaluation loop that computes metrics on-the-fly using your original metrics code
        with minimal changes to resolve dimension mismatch errors.
        """
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # Prepare the model for evaluation
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            if model is not self.model:
                self.model_wrapped = model

            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # Move model to appropriate dtype if needed
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = getattr(dataloader, "dataset", None)

        observed_num_examples = 0
        total_samples = 0
        losses = []

        # Initialize accumulators for metrics
        total_exact_match_sum = 0.0
        total_anls_sum = 0.0

        with torch.no_grad():
            for step, inputs in enumerate(dataloader):
                # Update observed examples
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    total_samples += observed_batch_size
                    if batch_size is None:
                        batch_size = observed_batch_size

                # Prepare inputs and move to the appropriate device
                inputs = self._prepare_inputs(inputs)

                # Prediction step
                answer_id = inputs.pop("answer_id")
                answer = inputs.pop('answer')

                inputs_ids = inputs.pop('input_ids')
                inputs['pixel_values'] = inputs['pixel_values'].to(model.model_pretrained.dtype)
                out = model.generate(inputs_ids, **inputs)

                loss = None
                logits = out
                labels = answer_id

                # loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only,
                #                                             ignore_keys=ignore_keys)

                # Accumulate losses
                if loss is not None:
                    losses.append(self.accelerator.gather_for_metrics(loss.repeat(batch_size)).detach().cpu())

                if logits is not None and labels is not None:
                    # if self.preprocess_logits_for_metrics is not None:
                    #     logits = self.preprocess_logits_for_metrics(logits, labels)

                    # Gather logits and labels across processes
                    logits = self.accelerator.gather_for_metrics(logits)
                    labels = self.accelerator.gather_for_metrics(labels)

                    # Compute batch metrics using Metrics class
                    batch_metrics = self.compute_metrics((logits, labels))

                    # Accumulate metrics
                    total_exact_match_sum += batch_metrics['ExactMatch'] * observed_batch_size
                    total_anls_sum += batch_metrics['ANLS'] * observed_batch_size

                self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

                # is_torch_tpu_available():
                 #   xm.mark_step()

        # Compute overall metrics
        overall_exact_match = total_exact_match_sum / total_samples
        overall_anls = total_anls_sum / total_samples

        # Prepare metrics dictionary
        metrics = {
            f"{metric_key_prefix}_ExactMatch": overall_exact_match,
            f"{metric_key_prefix}_ANLS": overall_anls,
        }

        # # Compute average loss
        # if losses:
        #     avg_loss = torch.cat(losses).mean().item()
        #     metrics[f"{metric_key_prefix}_loss"] = avg_loss
        if losses:
            avg_loss = torch.cat(losses).mean().item()
            metrics[f"{metric_key_prefix}_loss"] = avg_loss

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=total_samples)

