import matplotlib.pyplot as plt

files = [
    'log/micro_loss.txt',
    'log/micro_loss_torch_cpu.txt',
    'log/micro_loss_torch_cuda.txt',
]

def smooth(losses, window=20):
    return [
        sum(losses[max(0, i - window + 1):i + 1])
        / len(losses[max(0, i - window + 1):i + 1])
        for i in range(len(losses))
    ]

for file in files:
    with open(file) as f:
        losses = [
            float(line)
            for line in f
            if line.strip() and not line.strip().endswith('s')
        ]

    plt.plot(range(1, len(losses) + 1), smooth(losses), label=file)

plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig("./log/loss.png", dpi = 500)