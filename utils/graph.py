import torch
import os

from trash.pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from trash.pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME


def prepare_model(model, device, n_gpu, fp16, local_rank=-1):
    if fp16:
        model.half()
    model.to(device)
    if local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model


def prepare_train_dataloader(
        train_data, train_batch_size, gradient_accumulation_steps, num_train_epochs, local_rank=-1):
    if local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    num_train_optimization_steps = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
    if local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    return train_dataloader, num_train_optimization_steps


def prepare_eval_dataloader(
    eval_data, eval_batch_size
):
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
    return eval_dataloader


def prepare_parameters(model, method="decay"):
    """

    :param method: in [decay|direct]
    :return:
    """
    param_optimizer = list(model.named_parameters())
    if method == "decay":
        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters

    elif method == "direct":
        return [p for n, p in param_optimizer]
    else:
        raise NotImplementedError("No prepare parameters method is named as {}".format(method))


def prepare_optimizer(
        parameters, learning_rate, num_train_optimization_steps, warmup_proportion, fp16=False, loss_scale=0.):
    """

    :param parameters: acceptable for Optimizer
    :param learning_rate: learning rate
    :param num_train_optimization_steps:
    :param warmup_proportion:
    :param fp16: Bool
    :param loss_scale:
    :return: optimizer,
    """

    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=warmup_proportion,
                                             t_total=num_train_optimization_steps)
        return optimizer, {"warmup_linear": warmup_linear}
    else:
        optimizer = BertAdam(parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
        return optimizer, {}


# ======= During Training ==========

def calc_loss(loss, n_gpu, gradient_accumulation_steps, fp16=False, loss_scale=0.,):
    if n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
    if fp16 and loss_scale != 1.0:
        # rescale loss for fp16 training
        # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
        loss = loss * loss_scale
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps
    return loss


def update_gradient(
        loss, optimizer, gradient_accumulation_steps, step,
        fp16=False, warmup_linear=None, global_step=None, learning_rate=5e-5, warmup_proportion=0.1):
    if fp16:
        optimizer.backward(loss)
    else:
        loss.backward()
    if (step + 1) % gradient_accumulation_steps == 0:
        if fp16:
            # modify learning rate with special warm up BERT uses
            # if args.fp16 is False, BertAdam is used that handles this automatically
            lr_this_step = learning_rate * warmup_linear.get_lr(global_step, warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
        optimizer.step()
        optimizer.zero_grad()
        return True  # indicate the loss is updated
    else:
        return False


# ============ Model Saving ===============
def save_model(output_dir, model, tokenizer, weight_name=None, config_name=None):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, weight_name or WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, config_name or CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

