from .crane import Crane
from .crane_for_nodeflow import CraneForNodeFlow
from .crane_for_pathflow import CraneForPathFlow
from .crane_for_subgraphflow import CraneForSubGraphFlow
from omegaconf import DictConfig


_ID_KEYS = ("id_decode", "id_decode_hidden", "id_embed_buckets", "id_embed_dim")


def _id_kwargs(cfg: DictConfig):
    return {k: cfg[k] for k in _ID_KEYS if k in cfg}


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
    elif cfg.name == "CraneForNodeFlow":
        model = CraneForNodeFlow(
            source_input_dim=cfg.source_input_dim,
            dest_input_dim=cfg.dest_input_dim,
            source_hidden_dim=cfg.source_hidden_dim,
            dest_hidden_dim=cfg.dest_hidden_dim,
            source_embedding_dim=cfg.source_embedding_dim,
            dest_embedding_dim=cfg.dest_embedding_dim,
            memory_layer=cfg.memory_layer,
            carry_threshold=cfg.carry_threshold,
            **_id_kwargs(cfg),
        )
    elif cfg.name == "CraneForPathFlow":
        model = CraneForPathFlow(
            source_input_dim=cfg.source_input_dim,
            dest_input_dim=cfg.dest_input_dim,
            source_hidden_dim=cfg.source_hidden_dim,
            dest_hidden_dim=cfg.dest_hidden_dim,
            source_embedding_dim=cfg.source_embedding_dim,
            dest_embedding_dim=cfg.dest_embedding_dim,
            memory_layer=cfg.memory_layer,
            carry_threshold=cfg.carry_threshold,
            **_id_kwargs(cfg),
        )
    elif cfg.name == "CraneForSubGraphFlow":
        model = CraneForSubGraphFlow(
            source_input_dim=cfg.source_input_dim,
            dest_input_dim=cfg.dest_input_dim,
            source_hidden_dim=cfg.source_hidden_dim,
            dest_hidden_dim=cfg.dest_hidden_dim,
            source_embedding_dim=cfg.source_embedding_dim,
            dest_embedding_dim=cfg.dest_embedding_dim,
            memory_layer=cfg.memory_layer,
            carry_threshold=cfg.carry_threshold,
            **_id_kwargs(cfg),
        )
    else:
        raise ValueError(f"Unknown model name: {cfg.name}")
    return model
