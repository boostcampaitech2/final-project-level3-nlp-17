from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ModelArguments:
    input_dim: int = field(
        default = 700,
        metadata={"help":"tabular data columns num"}
    )

    output_dim: int = field(
        default = 2,
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
        default = 20,
        metadata={"help":"epochs"}
    )

    self_supervised_learning_epochs: int = field(
        default = 2,
        metadata={"help":"self supervised learning epochs"}
    )

    l_sparse: float = field(
        default = 0.5,
        metadata={"help":"eoefficient for sparsity regularization"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default = '../data/DATA_CHALLENGER_I_MatchData_last20.csv',
        metadata={"help":"data path"}
    )
    label_path: str = field(
        default = '../data/LABEL_CHALLENGER_I_MatchData_last20.csv',
        metadata={"help":"data path"}
    )
    batch_size: int = field(
        default = 64,
        metadata={"help":"train batch size"}
    )
