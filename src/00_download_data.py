import urllib.request

DATA_URL = "https://www.dropbox.com/s/yy2zqh34dyhv07i/data.txt?dl=1"
VOCAB_URL = "https://raw.githubusercontent.com/yandexdataschool/nlp_course/2020/week04_seq2seq/vocab.py"

def download(url: str, out_path: str):
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)

if __name__ == "__main__":
    download(DATA_URL, "data.txt")
    download(VOCAB_URL, "vocab.py")
    print("Done. Files saved: data.txt, vocab.py")