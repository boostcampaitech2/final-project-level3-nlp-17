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
        default = 8,
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
        default = 0.05,
        metadata={"help":"learning rate"}
    )

    epochs: int = field(
        default = 50,
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
        metadata={"help":"epsilon"}
    )

    virtual_batch_size: int = field(
        default = 512,
        metadata={"help":"virtual_batch_size"}
    )

    momentum: float = field(
        default = 0.95,
        metadata={"help":"momentum"}
    )

    cat_emb_dim: int = field(
        default = 1,
        metadata={"help":"cat_emb_dim"}
    )

    pretraining_ratio: float = field(
        default = 0.2,
        metadata={"help":"pretraining_ratio"}
    )

    n_shared_decoder: int = field(
        default = 1,
        metadata={"help":"n_shared_decoder"}
    )

    n_indep_decoder: int = field(
        default = 1,
        metadata={"help":"n_indep_decoder"}
    )
    
    weight_decay_rate: float = field(
        default = 0.3,
        metadata={"help":"weight_decay_rate"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default = '../data/dataset/DATA_CHALLENGER_I_MatchData_VER6.csv',
        metadata={"help":"data path"}
    )

    label_path: str = field(
        default = '../data/dataset/LABEL_CHALLENGER_I_MatchData_VER6.csv',
        metadata={"help":"data path"}
    )
    
