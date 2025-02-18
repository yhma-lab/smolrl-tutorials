from typing import Any

import optax
from flax.experimental import nnx
from jax import Array


class CNNModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs | None = None) -> None:
        if rngs is None:
            rngs = nnx.Rngs()
        self.conv1 = nnx.Conv(32, 64, (8, 8), (4, 4), rngs=rngs)
        self.conv2 = nnx.Conv(64, 64, (4, 4), (2, 2), rngs=rngs)
        self.conv3 = nnx.Conv(64, 512, (3, 3), (1, 1), rngs=rngs)
        self.linear1 = nnx.Linear(512, 4, rngs=rngs)
        self.linear2 = nnx.Linear(4, 4, rngs=rngs)

    def __call__(self, x: Array) -> Any:
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = x.flatten()
        x = nnx.relu(self.linear1(x))
        x = nnx.log_softmax(self.linear2(x))
        return x


def loss_fn(model: CNNModel, batch):
    logits = model(batch["image"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model: CNNModel, optimizer, metrics, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])
    optimizer.update(grads)
