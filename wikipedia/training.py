from gpt.training import train_from_cli
from wikipedia.data import load_texts

if __name__ == '__main__':
    train_from_cli(package='wikipedia', load_texts=load_texts)
