from src.recongnizer import Recongizer
from src.config.config import config


if __name__ == '__main__':
    recongnizer = Recongizer(config)
    recongnizer.train()

