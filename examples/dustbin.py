
# class Muon(torch.optim.Optimizer):
#     def __init__(
#         self,
#         muon_params=None,
#         lr=1e-3,
#         weight_decay=0.1,
#         momentum=0.95,
#         nesterov=True,
#         ns_steps=5,
#         adamw_params=None,
#         betas=(0.9, 0.95),
#         eps=1e-8,
#         *,
#         maximize: bool = False,
#         foreach: Optional[bool] = None,
#         capturable: bool = False,
#         differentiable: bool = False,
#         fused: Optional[bool] = None,
#         bias_correction=True,
#     ):
#         defaults = dict(
#             lr=lr,
#             betas=betas,
#             eps=eps,
#             weight_decay=weight_decay,
#             momentum=momentum,
#             nesterov=nesterov,
#             ns_steps=ns_steps,
#             foreach=foreach,
#             maximize=maximize,
#             capturable=capturable,
#             differentiable=differentiable,
#             fused=fused,
#             bias_correction=bias_correction,
#         )

#         params = []

#         muon_params = list(muon_params) if muon_params is not None else []
#         params.extend(muon_params)

#         adamw_params = list(adamw_params) if adamw_params is not None else []
#         params.extend(adamw_params)

#         super().__init__(params, defaults)

#         # sort params into those for which we will use muon and those for which we will not
#         for p in muon_params:
#             # for p in group["params"]:
#             assert p.ndim == 2, p.ndim
#             self.state[p]["use_muon"] = True
#         for p in adamw_params:
#             # for p in group["params"]:
#             self.state[p]["use_muon"] = False

#     @staticmethod
#     def adjust_lr_for_muon(lr, param_shape):
#         A, B = param_shape[:2]

#         adjusted_ratio = 0.2 * math.sqrt(max(A, B))
#         adjusted_lr = lr * adjusted_ratio

#         return adjusted_lr

#     @staticmethod
#     def _update_adamw(
#         data,
#         grad,
#         exp_avg,
#         exp_avg_sq,
#         lr,
#         beta1,
#         beta2,
#         eps,
#         weight_decay,
#         bias_correction1,
#         bias_correction2,
#     ):
#         grad = grad.to(data.dtype)

#         # Decay the first and second moment running average coefficient
#         exp_avg.lerp_(grad, 1 - beta1)
#         exp_avg_sq.lerp_(grad.square(), 1 - beta2)

#         grad = exp_avg / (eps + exp_avg_sq.sqrt())

#         scale = bias_correction1 / bias_correction2**0.5

#         if weight_decay != 0:
#             data.mul_(1 - lr * weight_decay)

#         data.add_(grad, alpha=-lr / scale)

#     @torch.no_grad()
#     def step(self, closure=None, **kwargs):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             params = [p for p in group["params"] if self.state[p]["use_muon"]]

#             for p in params:
#                 g = p.grad
#                 if g is None:
#                     continue
#                 if g.ndim > 2:
#                     g = g.view(g.size(0), -1)
#                 assert g is not None

#                 # calc update
#                 state = self.state[p]

#                 if "momentum_buffer" not in state:
#                     state["momentum_buffer"] = torch.zeros_like(g)
#                 buf = state["momentum_buffer"]
#                 buf.mul_(group["momentum"]).add_(g)
#                 g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

#                 meta = None
#                 if isinstance(g, DTensor):
#                     g, meta = to_local(g, keep_sharded=False)

#                 # gives NaNs when done with DTensor, instead of throwing a typical op not supported error, quite sneaky
#                 g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

#                 if meta is not None:
#                     g = to_dist(g, **meta)

#                 g *= max(1, g.size(0) / g.size(1)) ** 0.5
#                 g = g.view_as(p.data).type_as(p.data)

#                 # apply weight decay
#                 if group["weight_decay"] != 0:
#                     p.data.mul_(1 - group["lr"] * group["weight_decay"])

#                 # apply lr and update
#                 adjusted_lr = self.adjust_lr_for_muon(group["lr"], p.shape)
#                 p.data.add_(g, alpha=-adjusted_lr)

#             # adamw
#             params = [p for p in group["params"] if not self.state[p]["use_muon"]]
#             beta1, beta2 = group["betas"]

#             for p in params:
#                 g = p.grad
#                 if g is None:
#                     continue

#                 state = self.state[p]

#                 if "step" not in state:
#                     state["step"] = 0
#                     # gradient momentums
#                     state["exp_avg"] = torch.zeros_like(p, device=p.device)
#                     # gradient variances
#                     state["exp_avg_sq"] = torch.zeros_like(p, device=p.device)

#                 state["step"] += 1

#                 bias_correction1 = 1 - beta1 ** state["step"]
#                 bias_correction2 = 1 - beta2 ** state["step"]

#                 self._update_adamw(
#                     p.data,
#                     p.grad.data,
#                     state["exp_avg"],
#                     state["exp_avg_sq"],
#                     group["lr"],
#                     beta1,
#                     beta2,
#                     group["eps"],
#                     group["weight_decay"],
#                     bias_correction1,
#                     bias_correction2,
#                 )
#         return loss