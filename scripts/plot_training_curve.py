import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

log_dir = r"Tensorflow\workspace\models\my_ssd_mobnet\train"
ea = EventAccumulator(log_dir, size_guidance={"scalars": 0, "tensors": 0})
ea.Reload()
tags = ea.Tags()
print("scalars:", tags["scalars"])
print("tensors:", tags["tensors"])

def series(tag):
    if tag in tags["scalars"]:
        ev = ea.Scalars(tag)
        return [e.step for e in ev], [e.value for e in ev]
    ev = ea.Tensors(tag)
    return [e.step for e in ev], [float(tf.make_ndarray(e.tensor_proto)) for e in ev]

def smooth(vals, a=0.9):
    out, last = [], vals[0]
    for v in vals:
        last = a * last + (1 - a) * v
        out.append(last)
    return out

plt.figure(figsize=(8, 4.5))
for tag, label in [("Loss/total_loss", "Total"),
                   ("Loss/classification_loss", "Classification"),
                   ("Loss/localization_loss", "Localization")]:
    x, y = series(tag)
    plt.plot(x, y, alpha=0.25)
    plt.plot(x, smooth(y), label=label)
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Sun detection model: training loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

x, lr = series("learning_rate")
ax[0].plot(x, lr)
ax[0].set_title("Learning rate schedule")
ax[0].set_xlabel("Training step")
ax[0].grid(alpha=0.3)

x2, sps = series("steps_per_sec")
ax[1].plot(x2, sps, alpha=0.25)
ax[1].plot(x2, smooth(sps), label="smoothed")
ax[1].set_title("Training speed")
ax[1].set_xlabel("Training step")
ax[1].set_ylabel("Steps per second")
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("lr_and_speed.png", dpi=200)

total_steps = x[-1]
avg_sps = sum(sps) / len(sps)
print("Total steps:", total_steps)
print("Average steps/sec:", round(avg_sps, 2))
print("Approx training time (hours):", round(total_steps / avg_sps / 3600, 1))