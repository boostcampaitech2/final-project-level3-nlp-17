from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ModelArguments:
    input_dim: int = field(
        default = 10,
        metadata={"help":"tabular data columns num"}
    )

    output_dim: int = field(
        default = 10,
        metadata={"help":"output dimension"}
    )

    step_num: int = field(
        default = 10,
        metadata={"help":"decision step num"}
    )

    Nd: int = field(
        default = 64,
        metadata={"help":"output embeding demension"}
    )

    Na: int = field(
        default = 64,
        metadata={"help":"attentive embeding demension"}
    )

    learning_rate: float = field(
        default = 0.0001,
        metadata={"help":"learning rate"}
    )

    epochs: int = field(
        default = 100,
        metadata={"help":"epochs"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default = '../data/poker-hand-testing.data',
        metadata={"help":"data path"}
    )
    batch_size: int = field(
        default = 1024,
        metadata={"help":"train batch size"}
    )
