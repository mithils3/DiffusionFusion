from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

try:
    from torchdiffeq import odeint
except ImportError:  # pragma: no cover - optional dependency at runtime
    odeint = None


TensorState = torch.Tensor | tuple[torch.Tensor, ...]


def _state_device(state: TensorState) -> torch.device:
    return state[0].device if isinstance(state, tuple) else state.device


def _state_batch_size(state: TensorState) -> int:
    return int(state[0].shape[0] if isinstance(state, tuple) else state.shape[0])


def _state_dtype(state: TensorState) -> torch.dtype:
    return state[0].dtype if isinstance(state, tuple) else state.dtype


def _state_binary_map(
    left: TensorState,
    right: TensorState,
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> TensorState:
    if isinstance(left, tuple) != isinstance(right, tuple):
        raise TypeError("State operands must have matching structures.")
    if isinstance(left, tuple):
        return tuple(op(lhs, rhs) for lhs, rhs in zip(left, right, strict=True))
    return op(left, right)


def _state_add_scaled(state: TensorState, delta: TensorState, scale: torch.Tensor) -> TensorState:
    return _state_binary_map(state, delta, lambda x, dx: x + scale * dx)


def _state_average(left: TensorState, right: TensorState) -> TensorState:
    return _state_binary_map(left, right, lambda x, y: 0.5 * (x + y))


def _final_state_from_solution(solution: TensorState) -> TensorState:
    if isinstance(solution, tuple):
        return tuple(component[-1] for component in solution)
    return solution[-1]


def _apply_timestep_shift(timesteps: torch.Tensor, timestep_shift: float) -> torch.Tensor:
    if timestep_shift <= 0:
        return timesteps
    numerator = timestep_shift * timesteps
    denominator = 1 + (timestep_shift - 1) * timesteps
    return numerator / denominator


@dataclass(frozen=True)
class VelocityTransport:
    def get_drift(self) -> Callable[..., TensorState]:
        def velocity_ode(x: TensorState, t: torch.Tensor, model, **model_kwargs) -> TensorState:
            return model(x, t, **model_kwargs)

        return velocity_ode


def create_transport(
    path_type: str = "Linear",
    prediction: str = "velocity",
    *_args,
    **_kwargs,
) -> VelocityTransport:
    if str(path_type).lower() != "linear":
        raise NotImplementedError(
            f"Only Linear path transport is currently supported for custom inference, got `{path_type}`."
        )
    if str(prediction).lower() != "velocity":
        raise NotImplementedError(
            f"Only velocity prediction transport is currently supported for custom inference, got `{prediction}`."
        )
    return VelocityTransport()


class Sampler:
    def __init__(self, transport: VelocityTransport) -> None:
        self.transport = transport
        self.drift = self.transport.get_drift()

    def _build_timesteps(
        self,
        *,
        num_steps: int,
        device: torch.device,
        timestep_shift: float,
        reverse: bool,
    ) -> torch.Tensor:
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2 for ODE sampling.")
        timesteps = torch.linspace(0.0, 1.0, num_steps, device=device, dtype=torch.float32)
        timesteps = _apply_timestep_shift(timesteps, timestep_shift)
        if reverse:
            timesteps = 1.0 - timesteps.flip(0)
        return timesteps

    def _fixed_step_sample(
        self,
        init: TensorState,
        model,
        *,
        timesteps: torch.Tensor,
        sampling_method: str,
        **model_kwargs,
    ) -> TensorState:
        state: TensorState = init
        batch_size = _state_batch_size(state)
        t_dtype = _state_dtype(state) if _state_dtype(state).is_floating_point else torch.float32

        for idx in range(len(timesteps) - 1):
            t_cur = timesteps[idx]
            t_next = timesteps[idx + 1]
            dt = t_next - t_cur
            t_batch_cur = torch.full((batch_size,), t_cur, device=timesteps.device, dtype=t_dtype)
            drift_cur = self.drift(state, t_batch_cur, model, **model_kwargs)

            if sampling_method == "euler":
                state = _state_add_scaled(state, drift_cur, dt)
                continue

            if sampling_method != "heun":
                raise NotImplementedError(f"Unsupported fixed ODE sampler `{sampling_method}`.")

            predictor = _state_add_scaled(state, drift_cur, dt)
            t_batch_next = torch.full((batch_size,), t_next, device=timesteps.device, dtype=t_dtype)
            drift_next = self.drift(predictor, t_batch_next, model, **model_kwargs)
            state = _state_add_scaled(state, _state_average(drift_cur, drift_next), dt)

        return state

    def _adaptive_sample(
        self,
        init: TensorState,
        model,
        *,
        timesteps: torch.Tensor,
        sampling_method: str,
        atol: float,
        rtol: float,
        **model_kwargs,
    ) -> TensorState:
        if odeint is None:
            raise ImportError(
                "Adaptive ODE sampling requires `torchdiffeq`. "
                "Install it from requirements.txt or switch to `euler`/`heun`."
            )

        batch_size = _state_batch_size(init)
        t_dtype = _state_dtype(init) if _state_dtype(init).is_floating_point else torch.float32

        def _ode_fn(t_scalar: torch.Tensor, state: TensorState) -> TensorState:
            t_batch = torch.full((batch_size,), t_scalar, device=timesteps.device, dtype=t_dtype)
            return self.drift(state, t_batch, model, **model_kwargs)

        solution = odeint(
            _ode_fn,
            init,
            timesteps,
            method=sampling_method,
            atol=atol,
            rtol=rtol,
        )
        return _final_state_from_solution(solution)

    def sample_ode(
        self,
        *,
        sampling_method: str = "euler",
        num_steps: int = 250,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
        timestep_shift: float = 0.0,
    ):
        method = sampling_method.lower()

        def _sample(init: TensorState, model, **model_kwargs) -> TensorState:
            timesteps = self._build_timesteps(
                num_steps=num_steps,
                device=_state_device(init),
                timestep_shift=timestep_shift,
                reverse=reverse,
            )
            if method in {"euler", "heun"}:
                return self._fixed_step_sample(
                    init,
                    model,
                    timesteps=timesteps,
                    sampling_method=method,
                    **model_kwargs,
                )
            return self._adaptive_sample(
                init,
                model,
                timesteps=timesteps,
                sampling_method=method,
                atol=atol,
                rtol=rtol,
                **model_kwargs,
            )

        return _sample


__all__ = ["Sampler", "VelocityTransport", "create_transport"]
