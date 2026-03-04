import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import trange
from IPython.display import clear_output

from vocab import Vocab
from models_basic import BasicModel, device
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
    # read BPE-processed parallel corpus
    data_inp = np.array(open("train.bpe.en", encoding="utf-8").read().splitlines(), dtype=object)
    data_out = np.array(open("train.bpe.ru", encoding="utf-8").read().splitlines(), dtype=object)

    # split into train/dev
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

    model = BasicModel(inp_voc, out_voc).to(device)

    dummy_inp_tokens = inp_voc.to_matrix(sorted(train_inp, key=len)[5:10]).to(device)
    dummy_out_tokens = out_voc.to_matrix(sorted(train_out, key=len)[5:10]).to(device)

    h0 = model.encode(dummy_inp_tokens)
    h1, logits1 = model.decode_step(h0, torch.arange(len(dummy_inp_tokens), device=device))

    assert isinstance(h1, list) and len(h1) == len(h0)
    assert h1[0].shape == h0[0].shape and not torch.allclose(h1[0], h0[0])
    assert logits1.shape == (len(dummy_inp_tokens), len(out_voc))

    logits_seq = model.decode(h0, dummy_out_tokens)
    assert logits_seq.shape == (dummy_out_tokens.shape[0], dummy_out_tokens.shape[1], len(out_voc))

    logits_seq2 = model(dummy_inp_tokens, dummy_out_tokens)
    assert logits_seq2.shape == logits_seq.shape

    print("Source:")
    print("\n".join([line for line in train_inp[:3]]))
    dummy_translations, dummy_states = model.translate_lines(train_inp[:3], max_len=25)

    print("\nTranslations without training:")
    print("\n".join([line for line in dummy_translations]))

    dummy_loss = compute_loss(model, dummy_inp_tokens, dummy_out_tokens)
    print("Loss:", dummy_loss)
    assert np.allclose(dummy_loss.item(), 7.5, rtol=0.1, atol=0.1), "We're sorry for your loss"

    dummy_loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and abs(param.grad.max()) != 0, f"Param {name} received no gradients"

    compute_bleu(model, dev_inp, dev_out)

    metrics = {"train_loss": [], "dev_bleu": []}

    model = BasicModel(inp_voc, out_voc).to(device)
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

    for inp_line, trans_line in zip(dev_inp[::500], model.translate_lines(dev_inp[::500])[0]):
        print(inp_line)
        print(trans_line)
        print()