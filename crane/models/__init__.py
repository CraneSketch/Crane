from .crane import Crane
from .crane_for_degree import CraneForDegree
from omegaconf import DictConfig


def build_model(cfg: DictConfig):
    if cfg.name == "Crane":
        model = Crane(
            source_input_dim=cfg.source_input_dim,
            dest_input_dim=cfg.dest_input_dim,
            source_hidden_dim=cfg.source_hidden_dim,
            dest_hidden_dim=cfg.dest_hidden_dim,
            source_embedding_dim=cfg.source_embedding_dim,
            dest_embedding_dim=cfg.dest_embedding_dim,
            memory_layer=cfg.memory_layer,
            carry_threshold=cfg.carry_threshold,
        )
    elif cfg.name == "CraneForDegree":
        model = CraneForDegree(
            source_input_dim=cfg.source_input_dim,
            dest_input_dim=cfg.dest_input_dim,
            source_hidden_dim=cfg.source_hidden_dim,
            dest_hidden_dim=cfg.dest_hidden_dim,
            source_embedding_dim=cfg.source_embedding_dim,
            dest_embedding_dim=cfg.dest_embedding_dim,
            memory_layer=cfg.memory_layer,
            carry_threshold=cfg.carry_threshold,
        )
    else:
        raise ValueError(f"Unknown model name: {cfg.name}")
    return model
