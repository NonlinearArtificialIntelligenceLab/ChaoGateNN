import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Bool, Float, jaxtyped
from beartype import beartype as typechecker
from jax.typing import ArrayLike
from chaogatenn.maps import MapLike


@jaxtyped(typechecker=typechecker)
class ChaoGate(eqx.Module):
    DELTA: Float[Array, ""]
    X0: Float[Array, ""]
    X_THRESHOLD: Float[Array, ""]
    Map: MapLike

    def __init__(
        self, DELTA: ArrayLike, X0: ArrayLike, X_THRESHOLD: ArrayLike, Map: MapLike
    ):
        self.DELTA = jnp.array(DELTA)
        self.X0 = jnp.array(X0)
        self.X_THRESHOLD = jnp.array(X_THRESHOLD)
        self.Map = Map

    @typechecker
    def __call__(self, x: Bool[Array, "2"]) -> Float[Array, ""]:
        x1, x2 = x

        return jax.nn.sigmoid(
            self.Map(self.X0 + x1 * self.DELTA + x2 * self.DELTA) - self.X_THRESHOLD
        )

@jaxtyped(typechecker=typechecker)
class NChaoGate(eqx.Module):
    DELTA: Float[Array, ""]
    X0: Float[Array, ""]
    X_THRESHOLD: Float[Array, ""]
    Map: MapLike

    def __init__(
        self, DELTA: ArrayLike, X0: ArrayLike, X_THRESHOLD: ArrayLike, Map: MapLike
    ):
        self.DELTA = jnp.array(DELTA)
        self.X0 = jnp.array(X0)
        self.X_THRESHOLD = jnp.array(X_THRESHOLD)
        self.Map = Map

    @typechecker
    def __call__(self, x: Bool[Array, "n"]) -> Float[Array, ""]:

        signal = (x * self.DELTA).sum()
        return jax.nn.sigmoid(
            self.Map(self.X0 + signal) - self.X_THRESHOLD
        )

if __name__ == "__main__":
    from chaogatenn.maps import LogisticMap

    logistic_map = LogisticMap(a=4)
    chaogate = ChaoGate(DELTA=0.1, X0=0.5, X_THRESHOLD=0.5, Map=logistic_map)

    x = jnp.array([1, 0], dtype=bool)
    print(chaogate(x))

    chaogate_n_input = NChaoGate(DELTA=0.1, X0=0.5, X_THRESHOLD=0.5, Map=logistic_map)

    print(chaogate_n_input(x))
