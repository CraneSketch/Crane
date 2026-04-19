from .dense_generator import DenseGenerator
from .sparse_generator import SparseGenerator


from omegaconf import DictConfig


def build_generator(cfg: DictConfig, task_type: str):
    if cfg.name == "DenseGenerator":
        gen = DenseGenerator(
            cfg,
            task_type=task_type,
            node_binary_dim=cfg.node_binary_dim,
            item_lower=cfg.item_lower,
            item_upper=cfg.item_upper,
            generate_ratio=cfg.generate_ratio,
            ave_frequency=cfg.ave_frequency,
            zipf_param_lower=cfg.zipf_param_lower,
            zipf_param_upper=cfg.zipf_param_upper,
            skew_lower=cfg.skew_lower,
            skew_upper=cfg.skew_upper
        )
    elif cfg.name == "SparseGenerator":
        gen = SparseGenerator(
            cfg,
            task_type=task_type,
            node_binary_dim=cfg.node_binary_dim,
            item_lower=cfg.item_lower,
            item_upper=cfg.item_upper,
            generate_ratio=cfg.generate_ratio,
            ave_frequency=cfg.ave_frequency,
            zipf_param_lower=cfg.zipf_param_lower,
            zipf_param_upper=cfg.zipf_param_upper,
            skew_lower=cfg.skew_lower,
            skew_upper=cfg.skew_upper
        )
    else:
        raise ValueError(f"Unknown generator: {cfg.generator.name}")

    return gen
