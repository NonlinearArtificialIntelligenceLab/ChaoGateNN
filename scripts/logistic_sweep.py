import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from tqdm import trange
from jaxtyping import Array, Bool, Float

from chaogatenn.chaogate import ChaoGate
from chaogatenn.maps import LogisticMap
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


# Function to compute loss
@eqx.filter_value_and_grad()
def compute_loss(
    chao_gate: ChaoGate, x: Bool[Array, "batch 2"], y: Bool[Array, "batch"]
) -> Float[Array, ""]:
    pred = jax.vmap(chao_gate)(x)
    # binary cross entropy
    return -jnp.mean(y * jnp.log(pred + 1e-15) + (1 - y) * jnp.log(1 - pred + 1e-15))


# Function to perform a single optimization step
@eqx.filter_jit
def make_step(
    model: ChaoGate,
    x: Bool[Array, "dim 2"],
    y: Bool[Array, "dim"],
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[Float[Array, "dim"], ChaoGate, optax.OptState]:
    loss, grads = compute_loss(model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def main():
    output_dir = "../output/logistic_sweep/"
    # Sweep the logistic map parameter 'a' from 0 to 4
    metrics_dict = {}
    results_dict = {}
    for gate_name, Y in logic_gates.items():
        metrics_dict[gate_name] = []
        results_dict[gate_name] = []
        for a in jnp.linspace(0.0, 4.0, num=40):  # 50 steps from 0 to 4
            Map = LogisticMap(a=a)
            chao_gate = ChaoGate(DELTA=1.0, X0=1.0, X_THRESHOLD=1.0, Map=Map)
            optim = optax.adabelief(3e-4)
            opt_state = optim.init(eqx.filter(chao_gate, eqx.is_inexact_array))

            epochs = 4000
            for epoch in trange(
                epochs, desc=f"Training {gate_name} gate with a={a:.2f}"
            ):
                loss, chao_gate, opt_state = make_step(
                    chao_gate, X, Y, optim, opt_state
                )
                _, grads = compute_loss(chao_gate, X, Y)
                grad_norm_value = grad_norm(grads)

                    if loss < 1e-3:
                        break

            pred_ys = jax.vmap(chao_gate)(X)
            num_correct = jnp.sum((pred_ys > 0.5) == Y)
            final_accuracy = (num_correct / len(X)).item()
            metrics_dict[gate_name].append(
                (a, loss.item(), final_accuracy, grad_norm_value)
            )
            results_dict[gate_name].append(
                (a, chao_gate.DELTA, chao_gate.X0, chao_gate.X_THRESHOLD)
            )

    # Print results
    for gate_name, metrics in metrics_dict.items():
        print(f"\nResults for {gate_name} gate:")
        for a, loss, accuracy, grad_norm_value in metrics:
            print(
                f"a={a:.2f}, Loss={loss:.6f}, Accuracy={accuracy:.2f}, Grad Norm={grad_norm_value:.6f}"
            )

    # transform into arrays and save using numpy savetxt
    for gate_name, metrics in metrics_dict.items():
        metrics = jnp.array(metrics)
        np.savetxt(f"{output_dir}{gate_name}_metrics.txt", metrics, delimiter=",")

    for gate_name, results in results_dict.items():
        results = jnp.array(results)
        np.savetxt(f"{output_dir}{gate_name}_results.txt", results, delimiter=",")

    return 0


if __name__ == "__main__":
    main()
