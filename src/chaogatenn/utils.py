import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Array, Float


@eqx.filter_jit()
def grad_norm(grads: PyTree) -> Float[Array, ""]:
    return jnp.sqrt(
        sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(grads))
    )
