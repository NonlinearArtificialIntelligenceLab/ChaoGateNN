import argparse
import logging
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from jaxtyping import Array, Bool, Float

from chaogatenn.chaogate import NewChaoGate
from chaogatenn.maps import DuffingMap
from chaogatenn.utils import grad_norm

# Training data for different logic gates
X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=bool)  # Input combinations
AND_Y = jnp.array([0, 0, 0, 1], dtype=bool)  # AND gate output
OR_Y = jnp.array([0, 1, 1, 1], dtype=bool)  # OR gate output
XOR_Y = jnp.array([0, 1, 1, 0], dtype=bool)  # XOR gate output
NAND_Y = jnp.array([1, 1, 1, 0], dtype=bool)  # NAND gate output
NOR_Y = jnp.array([1, 0, 0, 0], dtype=bool)  # NOR gate output
XNOR_Y = jnp.array([1, 0, 0, 1], dtype=bool)  # XNOR gate output

# List of logic gates and their corresponding outputs
logic_gates = {
    "AND": AND_Y,
    "OR": OR_Y,
    "XOR": XOR_Y,
    "NAND": NAND_Y,
    "NOR": NOR_Y,
    "XNOR": XNOR_Y,
}

def argument_parser():
    parser = argparse.ArgumentParser(
        description="Runner for NewChaogate logistic map."
    )

    parser.add_argument("--beta", type=float, default=4, help="Duffing map parameter `beta`")
    parser.add_argument(
        "--gate", type=str, default="AND", help="Logic gate to train on"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for random number generator"
    )
    parser.add_argument(
        "--epochs", type=int, default=5_000, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--out_path", type=str, default="../output/test/", help="Path to save metrics"
    )
    parser.add_argument("--log_every", type=int, default=100, help="log every n epochs")
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging verbosity",
    )
    parser.add_argument("--console", action="store_true", help="Log to console")
    return parser



# Function to compute loss
def compute_loss(
    chao_gate: NewChaoGate, x: Bool[Array, "batch 2"], y: Bool[Array, "batch"]
) -> Float[Array, ""]:
    pred = jax.vmap(chao_gate)(x)
    # binary cross entropy
    return -jnp.mean(y * jnp.log(pred + 1e-15) + (1 - y) * jnp.log(1 - pred + 1e-15))


# Function to perform a single optimization step
@eqx.filter_jit
def make_step(
    model: NewChaoGate,
    x: Bool[Array, "dim 2"],
    y: Bool[Array, "dim"],
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[Float[Array, "dim"], NewChaoGate, optax.OptState]:
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def main():

    parser = argument_parser()
    args = parser.parse_args()

    if args.gate not in logic_gates:
        raise NotImplementedError(f"Logic gate {args.gate} not implemented")

    os.makedirs(args.out_path, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.verbosity),
        filename=f"{args.out_path}info.log",
        filemode="w",
    )
    logger = logging.getLogger(__name__)
    if args.console:
        console_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(console_handler)

    logger.info(f"Training {args.gate} gate with seed {args.seed}")

    chao_key = jax.random.PRNGKey(args.seed)
    DELTA_X, DELTA_Y, X0, X_THRESHOLD = jax.random.normal(chao_key, (4,))
    Map = DuffingMap(beta=args.beta)
    chao_gate = NewChaoGate(DELTA_X=DELTA_X, DELTA_Y=DELTA_Y, X0=X0, X_THRESHOLD=X_THRESHOLD, Map=Map)

    optim = optax.adabelief(args.learning_rate)
    opt_state = optim.init(eqx.filter(chao_gate, eqx.is_inexact_array))

    epoch_list = []
    loss_list = []
    accuracy_list = []
    param_list = []

    for epoch in range(args.epochs):
        loss, chao_gate, opt_state = make_step(chao_gate, X, logic_gates[args.gate], optim, opt_state)
        grads = eqx.filter_grad(compute_loss)(chao_gate, X, logic_gates[args.gate])
        grad_norm_value = grad_norm(grads)

        if epoch % args.log_every == 0:
            pred_ys = jax.vmap(chao_gate)(X)
            num_correct = jnp.sum((pred_ys > 0.5) == logic_gates[args.gate])
            accuracy = (num_correct / len(X)).item()
            logger.info(
                f"Epoch {epoch}: Loss={loss.item():.6f}, Accuracy={accuracy:.2f}, Grad Norm={grad_norm_value:.6f}"
            )
            epoch_list.append(epoch)
            loss_list.append(loss.item())
            accuracy_list.append(accuracy)
            param_list.append((chao_gate.DELTA_X, chao_gate.DELTA_Y, chao_gate.X0, chao_gate.X_THRESHOLD))

    metrics = jnp.array(list(zip(epoch_list, loss_list, accuracy_list)))
    np.savetxt(f"{args.out_path}{args.gate}_metrics.txt", metrics, delimiter=",")
    results = jnp.array([(epoch, *params) for epoch, params in zip(epoch_list, param_list)])
    np.savetxt(f"{args.out_path}{args.gate}_results.txt", results, delimiter=",")

    return 0


if __name__ == "__main__":
    main()
