from gpt.training import train_from_cli
from western.data import load_texts

if __name__ == '__main__':
    train_from_cli(package='western', load_texts=load_texts)
