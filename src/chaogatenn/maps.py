import jax
import jax.numpy as jnp
import equinox as eqx

from beartype import beartype as typechecker
from typing import Protocol, runtime_checkable, Tuple
from jaxtyping import Float
from jax.typing import ArrayLike


@runtime_checkable
class MapLike(Protocol):
    def __call__(self, x: ArrayLike) -> ArrayLike: ...


class LogisticMap(eqx.Module):
    a: float = 4

    @typechecker
    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.a * x * (1 - x)


class LorenzMap(eqx.Module):
    sigma: float = 10
    rho: float = 28
    beta: float = 8 / 3
    dt: float = 0.001
    steps: int = 1000

    @typechecker
    def __call__(self, x: ArrayLike) -> ArrayLike:
        def lorenz_step(
            i: int, state: Tuple[Float, Float, Float]
        ) -> Tuple[Float, Float, Float]:
            x, y, z = state
            # Lorenz equations
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
    def __call__(self, x: ArrayLike) -> ArrayLike:
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

        initial_state = (x, 0.0, 0.0)  # 0 velocity and 0 time

        final_state = jax.lax.fori_loop(0, self.steps, duffing_step, initial_state)

        return final_state[0]  # return position value as the chaotic output

class RosslerMap(eqx.Module):
    a: float = 0.2
    b: float = 0.2
    c: float = 5.7
    dt: float = 0.001
    steps: int = 1000

    @typechecker
    def __call__(self, x: ArrayLike) -> ArrayLike:
        def rossler_step(
            i: int, state: Tuple[Float, Float, Float]
        ) -> Tuple[Float, Float, Float]:
            x, y, z = state
            # Rossler equations
            dx = -y - z
            dy = x + self.a * y
            dz = self.b + z * (x - self.c)

            # Euler method for updating
            x = x + dx * self.dt
            y = y + dy * self.dt
            z = z + dz * self.dt

            return x, y, z

        x0, y0, z0 = x, x, x

        final_state = jax.lax.fori_loop(0, self.steps, rossler_step, (x0, y0, z0))

        # Return the x-coordinate as the chaotic output
        return final_state[0]

class ChenMap(eqx.Module):
    a: float = 35.0
    b: float = 3.0
    c: float = 28.0
    d: float = 1.0
    dt: float = 0.001
    steps: int = 1000

    @typechecker
    def __call__(self, x: ArrayLike) -> ArrayLike:
        def chen_step(
            i: int, state: Tuple[Float, Float, Float, Float]
        ) -> Tuple[Float, Float, Float, Float]:
            x, y, z, w = state
            # Hyperchaotic Chen equations
            dxdt = self.a * (y - x)
            dydt = (self.c - self.a) * x - x * z + self.c * y
            dzdt = x * y - self.b * z + w
            dwdt = -self.d * w

            # Euler method for updating
            x_new = x + dxdt * self.dt
            y_new = y + dydt * self.dt
            z_new = z + dzdt * self.dt
            w_new = w + dwdt * self.dt

            return x_new, y_new, z_new, w_new

        # Initial state, x and y as inputs, and z, w initialized
        initial_state = (x, x, 0.0, 0.0)

        # Perform the iterative steps
        final_state = jax.lax.fori_loop(0, self.steps, chen_step, initial_state)

        return final_state[2]  # return z value as the chaotic output

class ChenMapRK4(eqx.Module):
    a: float = 35.0
    b: float = 3.0
    c: float = 28.0
    d: float = 1.0
    dt: float = 0.001
    steps: int = 1000

    @typechecker
    def __call__(self, x: ArrayLike) -> ArrayLike:
        def chen_system(state: Tuple[Float, Float, Float, Float]) -> Tuple[Float, Float, Float, Float]:
            x, y, z, w = state
            dxdt = self.a * (y - x)
            dydt = (self.c - self.a) * x - x * z + self.c * y
            dzdt = x * y - self.b * z + w
            dwdt = -self.d * w
            return dxdt, dydt, dzdt, dwdt

        def rk4_step(state: Tuple[Float, Float, Float, Float], dt: float) -> Tuple[Float, Float, Float, Float]:
            k1 = chen_system(state)
            k2 = chen_system(tuple(s + dt / 2 * k for s, k in zip(state, k1)))
            k3 = chen_system(tuple(s + dt / 2 * k for s, k in zip(state, k2)))
            k4 = chen_system(tuple(s + dt * k for s, k in zip(state, k3)))

            return tuple(s + dt / 6 * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
                        for s, k1_i, k2_i, k3_i, k4_i in zip(state, k1, k2, k3, k4))

        # Initial state
        initial_state = (x, x, 0.0, 0.0)

        # Perform the iterative steps using RK4
        final_state = jax.lax.fori_loop(0, self.steps, lambda i, state: rk4_step(state, self.dt), initial_state)

        return final_state[2]  # return z value as the chaotic output

class RosslerHyperchaosMap(eqx.Module):
    a: float = 0.25
    b: float = 3
    c: float = 0.05
    d: float = 5
    dt: float = 0.01
    steps: int = 1000

    @typechecker
    def __call__(self, x: ArrayLike) -> ArrayLike:
        def rossler_system(state: Tuple[Float, Float, Float, Float]) -> Tuple[Float, Float, Float, Float]:
            x, y, z, w = state
            dxdt = -y - z
            dydt = x + self.a * y + w
            dzdt = self.b + z * x
            dwdt = self.c * w + self.d * z
            return dxdt, dydt, dzdt, dwdt

        def rk4_step(state: Tuple[Float, Float, Float, Float], dt: float) -> Tuple[Float, Float, Float, Float]:
            k1 = rossler_system(state)
            k2 = rossler_system(tuple(s + dt / 2 * k for s, k in zip(state, k1)))
            k3 = rossler_system(tuple(s + dt / 2 * k for s, k in zip(state, k2)))
            k4 = rossler_system(tuple(s + dt * k for s, k in zip(state, k3)))

            return tuple(s + dt / 6 * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
                        for s, k1_i, k2_i, k3_i, k4_i in zip(state, k1, k2, k3, k4))

        # Initial state
        initial_state = (x, x, 0.0, 0.0)  # w = 0, z = 0 initial condition

        # Perform the iterative steps using RK4
        final_state = jax.lax.fori_loop(0, self.steps, lambda i, state: rk4_step(state, self.dt), initial_state)

        return final_state[2]  # return z value as the chaotic output