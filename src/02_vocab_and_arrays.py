import numpy as np
import matplotlib.pyplot as plt

from vocab import Vocab

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

    batch_lines = sorted(train_inp, key=len)[5:10]
    batch_ids = inp_voc.to_matrix(batch_lines)
    batch_lines_restored = inp_voc.to_lines(batch_ids)

    print("lines")
    print(batch_lines)
    print("\nwords to ids (0 = bos, 1 = eos):")
    print(batch_ids)
    print("\nback to words")
    print(batch_lines_restored)

    plt.figure(figsize=[8, 4])
    plt.subplot(1, 2, 1)
    plt.title("source length")
    plt.hist(list(map(len, map(str.split, train_inp))), bins=20)

    plt.subplot(1, 2, 2)
    plt.title("translation length")
    plt.hist(list(map(len, map(str.split, train_out))), bins=20)

    plt.tight_layout()
    plt.show()