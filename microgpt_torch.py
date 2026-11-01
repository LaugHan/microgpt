'''
使用pytorch完成最简单的gpt2。

开始时间 26.11.1
'''

import os
import math
import random
import time
import torch

random.seed(42)
torch.manual_seed(42)

device = 'cuda:1'

# 加载文件
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

# 切分成行，打乱
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)

uchars = sorted(set(''.join(docs))) # 用出现过的所有char当作基本的字典。这里就是'a' 'b'这种
BOS = len(uchars)
vocab_size = len(uchars) + 1

## 初始化参数，存储模型的知识
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

matrix = lambda nout, nin, std = 0.08: (torch.randn(nout, nin, device=device) * std).requires_grad_()
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
# params = [p for mat in state_dict.values() for row in mat for p in row] # [矩阵数 = 层数 * 6] * [每个矩阵行数] * [列数]展开成一列
params = list(state_dict.values())


# 定义model architecture：一个函数，把tokens和参数映射到logits。
# 遵循gpt-2，只是把layernorm换成rmsnorm，没有biases，把GeLU换成ReLU
def linear(x, w):
    return x @ w.t()

def rmsnorm(x):
    ms = (x * x).mean()
    scale = (ms + 1e-5) ** -0.5
    return x * scale

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = tok_emb + pos_emb
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) 多头注意力block
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        keys[li].append(k)
        values[li].append(v) # kv cache

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim

            q_h = q[hs:hs + head_dim]
            k_h = torch.stack([ki[hs:hs + head_dim] for ki in keys[li]])
            v_h = torch.stack([vi[hs:hs + head_dim] for vi in values[li]])
            attn_logits = (q_h @ k_h.t()) / math.sqrt(head_dim) # q @ K.t
            attn_weights = torch.softmax(attn_logits, dim = 0)
            head_out = attn_weights @ v_h
            x_attn.append(head_out)
        x = torch.cat(x_attn)
        x = linear(x, state_dict[f'layer{li}.attn_wo'])
        x = x + x_residual

        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = torch.relu(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = x + x_residual

    logits = linear(x, state_dict['lm_head'])
    return logits

# 下面开始写训练环节

learning_rate = 0.01
num_steps = 1000

optimizer = torch.optim.Adam(params, learning_rate, betas = (0.85, 0.99), eps=1e-8)

os.makedirs('log', exist_ok=True)
f = open('log/micro_loss_torch_cuda.txt', 'w')
start = time.time()

for step in range(num_steps):

    # 每次拿到单个文件，tokenize它，在两端加上BOS。
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]

        logits = gpt(token_id, pos_id, keys, values)
        log_probs = torch.log_softmax(logits, dim = 0)
        loss_t = -log_probs[target_id]
        losses.append(loss_t)
    loss = torch.stack(losses).mean() # stack是把标量张量堆叠成一个张量。
    optimizer.zero_grad()
    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for group in optimizer.param_groups:
        group['lr'] = lr_t
    optimizer.step()

    f.write(f'{loss.item()}\n')
    print (f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}", end='\r')

torch.cuda.synchronize()
f.write(f'{time.time() - start} s\n')
f.close()

# # 推理阶段
temperature = 0.5
print("\n --- inference (new, hallucinated names) ---")
with torch.no_grad():
    for sample_idx in range(20):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values)
            probs = torch.softmax(logits / temperature, dim = 0)
            token_id = random.choices(range(vocab_size), weights=probs.tolist())[0] # 返回是一个list, 要取[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
