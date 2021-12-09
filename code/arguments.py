from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ModelArguments:
    input_dim: int = field(
        default = 350,
        metadata={"help":"tabular data columns num"}
    )

    output_dim: int = field(
        default = 2,
        metadata={"help":"output dimension"}
    )

    n_steps: int = field(
        default = 4,
        metadata={"help":"decision step num"}
    )

    n_d: int = field(
        default = 16,
        metadata={"help":"output embeding demension"}
    )

    n_a: int = field(
        default = 16,
        metadata={"help":"attentive embeding demension"}
    )

    learning_rate: float = field(
        default = 0.01,
        metadata={"help":"learning rate"}
    )

    epochs: int = field(
        default = 5000,
        metadata={"help":"epochs"}
    )

    self_supervised_learning_epochs: int = field(
        default = 1,
        metadata={"help":"self supervised learning epochs"}
    )

    l_sparse: float = field(
        default = 0.000001,
        metadata={"help":"eoefficient for sparsity regularization"}
    )

    gamma: float = field(
        default = 1.5,
        metadata={"help":"gamma"}
    )

    batch_size: int = field(
        default = 4096,
        metadata={"help":"train batch size"}
    )

    n_independent: int = field(
        default = 2,
        metadata={"help":"independent layer num "}
    )

    n_shared: int = field(
        default = 2,
        metadata={"help":"shared layer num"}
    )

    epsilon: float = field(
        default = 1e-15,
        metadata={"help":"shared layer num"}
    )

    virtual_batch_size: int = field(
        default = 128,
        metadata={"help":"shared layer num"}
    )

    momentum: float = field(
        default = 0.95,
        metadata={"help":"shared layer num"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default = '../data/LABEL_CHALLENGER_I_MatchData_last20_Ver3.csv',
        metadata={"help":"data path"}
    )
    label_path: str = field(
        default = '../data/LABEL_CHALLENGER_I_MatchData_last20.csv',
        metadata={"help":"data path"}
    )
    
