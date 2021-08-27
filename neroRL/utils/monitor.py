from enum import Enum

class Tag(Enum):
    DECAY = "decay"
    EPISODE = "episode"
    EVALUATION = "evaluation"
    GRADIENT = "gradient"
    LOSS = "loss"
    OTHER = "other"

class Monitor():
    def __init__(self) -> None:
        # log to console
        # log to file
        # summary writer tensorboard
            # training stats
            # eval stats
            # config
        pass