from dataclasses import dataclass

from gflownet.utils.misc import StrictDataClass
from gflownet.models.config import GraphTransformerConfig


@dataclass
class ModelConfig(StrictDataClass):
    """Generic configuration for models

    Attributes
    ----------
    num_layers : int
        The number of layers in the model
    num_emb : int
        The number of dimensions of the embedding
    num_layers_building_block : int
        The number of layers in the action embedding
    num_emb_building_block : int
        The number of dimensions of the action embedding
    """

    num_layers: int = 3
    num_emb: int = 128
    dropout: float = 0
    fp_radius_building_block: int = 2
    fp_nbits_building_block: int = 1024
    num_layers_building_block: int = 0
    num_emb_building_block: int = 64
    graph_transformer: GraphTransformerConfig = GraphTransformerConfig()
