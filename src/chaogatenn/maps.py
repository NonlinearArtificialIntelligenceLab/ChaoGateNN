import jax
import jax.numpy as jnp
import equinox as eqx
from beartype import beartype as typechecker
from typing import Protocol, runtime_checkable, Tuple

from jaxtyping import Float
from jax.typing import ArrayLike


@runtime_checkable
class MapLike(Protocol):
    def __call__(self, x: ArrayLike) -> ArrayLike:  # type: ignore
        ...


class LogisticMap(eqx.Module):
    a: float = 4

    @typechecker
    def __call__(self, x: ArrayLike) -> ArrayLike:  # type: ignore
        return self.a * x * (1 - x)


class LorenzMap(eqx.Module):
    sigma: float = 10
    rho: float = 28
    beta: float = 8 / 3
    dt: float = 0.001
    steps: int = 1000

    @typechecker
    def __call__(self, x: ArrayLike) -> ArrayLike:  # type: ignore
        def lorenz_step(
            i: int, state: Tuple[Float, Float, Float]
        ) -> Tuple[Float, Float, Float]:
            x, y, z = state
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z

            # Euler method for updating
            x = x + dx * self.dt
            y = y + dy * self.dt
            z = z + dz * self.dt

            return x, y, z

        x0, y0, z0 = x, x, x

        final_state = jax.lax.fori_loop(0, self.steps, lorenz_step, (x0, y0, z0))

        # Return the x-coordinate as the chaotic output
        return final_state[0]


class DuffingMap(eqx.Module):
    alpha: float = 1.0
    beta: float = 5.0
    delta: float = 0.02
    gamma: float = 8.0
    omega: float = 0.5
    dt: float = 0.001
    steps: int = 1000

    @typechecker
    def __call__(self, x: ArrayLike) -> ArrayLike:  # type: ignore
        def duffing_step(
            i: int, state: Tuple[Float, Float, Float]
        ) -> Tuple[Float, Float, Float]:
            x, v, t = state
            # Duffing equations
            dxdt = v
            dvdt = (
                -self.delta * v
                - self.alpha * x
                - self.beta * (x**3)
                + self.gamma * jnp.cos(self.omega * t)
            )

            # Euler method for updating
            x_new = x + dxdt * self.dt
            v_new = v + dvdt * self.dt
            t_new = t + self.dt

            return x_new, v_new, t_new

        initial_state = (x, 0.0, 0.0)

        final_state = jax.lax.fori_loop(0, self.steps, duffing_step, initial_state)

        return final_state[0]
