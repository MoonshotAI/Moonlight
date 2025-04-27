import os
import math
import torch
from loguru import logger
from datasets import load_dataset
import argparse
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from typing import Optional
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor import DTensor
import torch.distributed as dist

def to_dist(x, from_local=False, **meta):
    if from_local:
        return DTensor.from_local(
            x,
            device_mesh=meta["device_mesh"],
            placements=meta["placements"],
            shape=meta["shape"],
            stride=meta["stride"],
        )
    else:
        return distribute_tensor(x, device_mesh=meta["device_mesh"], placements=meta["placements"])

def to_local(x, keep_sharded=False):
    if isinstance(x, DTensor):
        meta = dict(
            device_mesh=x.device_mesh,
            placements=x.placements,
            shape=x.shape,
            stride=x.stride(),
        )
        if keep_sharded:
            return x.to_local(), meta
        else:
            return x.full_tensor(), meta

    return x, None

class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        muon_params=None,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        betas=(0.9, 0.95),
        eps=1e-8,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        bias_correction=True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            bias_correction=bias_correction,
        )

        params = []

        muon_params = list(muon_params) if muon_params is not None else []
        params.extend(muon_params)

        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)

        super().__init__(params, defaults)

        # sort params into those for which we will use muon and those for which we will not
        for p in muon_params:
            # for p in group["params"]:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # for p in group["params"]:
            self.state[p]["use_muon"] = False

    @staticmethod
    def adjust_lr_for_muon(lr, param_shape):
        A, B = param_shape[:2]

        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio

        return adjusted_lr

    @staticmethod
    def _update_adamw(
        data,
        grad,
        exp_avg,
        exp_avg_sq,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2,
    ):
        grad = grad.to(data.dtype)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2)

        grad = exp_avg / (eps + exp_avg_sq.sqrt())

        scale = bias_correction1 / bias_correction2**0.5

        if weight_decay != 0:
            data.mul_(1 - lr * weight_decay)

        data.add_(grad, alpha=-lr / scale)

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if self.state[p]["use_muon"]]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                meta = None
                if isinstance(g, DTensor):
                    g, meta = to_local(g, keep_sharded=False)

                # gives NaNs when done with DTensor, instead of throwing a typical op not supported error, quite sneaky
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                if meta is not None:
                    g = to_dist(g, **meta)

                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                g = g.view_as(p.data).type_as(p.data)

                # apply weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # apply lr and update
                adjusted_lr = self.adjust_lr_for_muon(group["lr"], p.shape)
                p.data.add_(g, alpha=-adjusted_lr)

            # adamw
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            beta1, beta2 = group["betas"]

            for p in params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                    # gradient momentums
                    state["exp_avg"] = torch.zeros_like(p, device=p.device)
                    # gradient variances
                    state["exp_avg_sq"] = torch.zeros_like(p, device=p.device)

                state["step"] += 1

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                self._update_adamw(
                    p.data,
                    p.grad.data,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["weight_decay"],
                    bias_correction1,
                    bias_correction2,
                )
        return loss

class MoonDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["train"]["text"]
        self.max_length = max_length
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        if os.path.exists(f"{self.dataset_name}.bin"):
            print('loading tokenized data')
            self.tokens = torch.load(f"{self.dataset_name}.bin")
        else:
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            torch.save(self.tokens, f"{self.dataset_name}.bin")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * (self.max_length)
        end_idx = start_idx + (self.max_length)
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Muon(
            lr=lr,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, "optimizer not supported"

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--optimizer", type=str, default="muon")
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    return parser.parse_args()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Muon(
            lr=lr,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, "optimizer not supported"

def get_train_loader(dataset_name, rank, world_size):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }
    train_dataset = load_dataset(name2path[dataset_name], trust_remote_code=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )
    train_dataset = MoonDataset(dataset_name, train_dataset, tokenizer)
    sampler = DistributedSampler(dataset=train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    kwargs = {'batch_size': 16, 'sampler': sampler}
    train_loader = DataLoader(train_dataset, **kwargs)
    return train_loader

def fsdp_main(rank, world_size, args):
    print((rank, world_size))
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    # load model
    config = Qwen2Config(
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151643,
        hidden_act="silu",
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4864,
        max_position_embeddings=513,
        max_window_layers=12,
        model_type="qwen2",
        num_attention_heads=16,
        num_hidden_layers=12,
        num_key_value_heads=16,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        sliding_window=1024,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        use_cache=True,
        use_mrope=False,
        use_sliding_window=False,
        vocab_size=151936,
    )
    model = Qwen2ForCausalLM(config)
    optimizer = get_optimizer(args.optimizer, model, lr=args.lr, wd=args.wd)
    model = model.to(rank)
    model = DDP(model)
    model.train()

    train_loader = get_train_loader(args.dataset, rank, world_size)
    epoch = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epoch,
        num_cycles=0.5,
    )

    ddp_loss = torch.zeros(2).to(rank)
    for epoch in range(epoch):
        
        for step, batch in enumerate(train_loader):
            batch = batch.to(rank)
            input_ids = batch
            output = model(input_ids=input_ids, labels=input_ids)
            optimizer.zero_grad()
            loss = output.loss
            loss.backward()
            optimizer.step()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(batch)

            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
            if rank == 0:
                logger.info('Train Epoch: {} \tLoss: {:.6f} \t Batch: {}'.format(epoch, ddp_loss[0] / ddp_loss[1], len(batch)))
        lr_scheduler.step()
        
    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()

if __name__ == '__main__':
    # Training settings
    args = parse_args()

    # fsdp_main(0, 1, args)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)