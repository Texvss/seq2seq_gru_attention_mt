import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import trange
from IPython.display import clear_output

from vocab import Vocab
from models_basic import device
from models_attention import AttentiveModel
from metrics import compute_bleu

def compute_loss(model, inp, out, **flags):
    mask = model.out_voc.compute_mask(out)
    targets_1hot = F.one_hot(out, len(model.out_voc)).to(torch.float32)

    logits_seq = model(inp, out)
    logprobs_seq = F.log_softmax(logits_seq, dim=-1)
    logp_out = (logprobs_seq * targets_1hot).sum(dim=-1)

    loss = -(logp_out * mask).sum() / mask.sum()
    return loss

if __name__ == "__main__":
    data_inp = np.array(open("train.bpe.en", encoding="utf-8").read().splitlines(), dtype=object)
    data_out = np.array(open("train.bpe.ru", encoding="utf-8").read().splitlines(), dtype=object)

    np.random.seed(42)
    indices = np.arange(len(data_inp))
    np.random.shuffle(indices)

    data_inp = data_inp[indices]
    data_out = data_out[indices]

    dev_size = 1000
    dev_inp, dev_out = data_inp[:dev_size], data_out[:dev_size]
    train_inp, train_out = data_inp[dev_size:], data_out[dev_size:]

    inp_voc = Vocab.from_lines(train_inp)
    out_voc = Vocab.from_lines(train_out)

    metrics = {"train_loss": [], "dev_bleu": []}

    model = AttentiveModel("Attentive_Model", inp_voc, out_voc).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 32

    for _ in trange(25000):
        step = len(metrics["train_loss"]) + 1
        batch_ix = np.random.randint(len(train_inp), size=batch_size)
        batch_inp = inp_voc.to_matrix(train_inp[batch_ix]).to(device)
        batch_out = out_voc.to_matrix(train_out[batch_ix]).to(device)

        opt.zero_grad()
        loss_t = compute_loss(model, batch_inp, batch_out)
        loss_t.backward()
        opt.step()

        metrics["train_loss"].append((step, loss_t.item()))

        if step % 100 == 0:
            metrics["dev_bleu"].append((step, compute_bleu(model, dev_inp, dev_out)))

            clear_output(True)
            plt.figure(figsize=(12, 4))
            for i, (name, history) in enumerate(sorted(metrics.items())):
                plt.subplot(1, len(metrics), i + 1)
                plt.title(name)
                plt.plot(*zip(*history))
                plt.grid()
            plt.show()
            print("Mean loss=%.3f" % np.mean(metrics["train_loss"][-10:], axis=0)[1], flush=True)

    final_bleu = compute_bleu(model, dev_inp, dev_out)
    print(f"Final BLEU score on dev set: {final_bleu:.4f}")