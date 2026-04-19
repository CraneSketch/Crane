import torch
import logging
import time
from torch.utils.data import DataLoader

from omegaconf import DictConfig
from crane.data import SketchDataset, MiniDataset


logger = logging.getLogger("crane")


def eval_one_task(model, eval_set, mini_batch_size, micro_batch_size):
    model.eval()
    preds_all = []
    targets_all = []
    activated_memory_dim_list = []

    device = next(model.parameters()).device

    total_support_num = 0
    total_query_num = 0
    support_time = 0.0
    query_time = 0.0
    
    with torch.no_grad():
        for idx, sample in enumerate(eval_set):
            logger.info(f"Evaluating {eval_set.name} [{idx}/{len(eval_set)}]")
            logger.info(f"Support Size: {sample[0].size(0)}, Query Size: {sample[2].size(0)}")
            model.clear()
            preds_sample = []
            targets_sample = []

            support_x, support_y, query_x, query_y = sample

            total_support_num += len(support_x)
            total_query_num += len(query_x)

            support_set = MiniDataset(support_x, support_y)
            query_set = MiniDataset(query_x, query_y)

            support_mini_loader = DataLoader(support_set, batch_size=mini_batch_size, shuffle=False)
            query_mini_loader = DataLoader(query_set, batch_size=mini_batch_size, shuffle=False)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.time()

            for mini_batch in support_mini_loader:
                mini_support_x, mini_support_y = mini_batch
                mini_support_x = mini_support_x.to(device)
                mini_support_y = mini_support_y.to(device)
                model.write(mini_support_x, mini_support_y, micro_batch_size)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            support_time += time.time() - t0
            
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.time()

            for mini_batch in query_mini_loader:
                mini_query_x, mini_query_y = mini_batch
                mini_query_x = mini_query_x.to(device)
                mini_query_y = mini_query_y.to(device)
                preds = model.query(mini_query_x)
                preds_sample.append(preds)
                targets_sample.append(mini_query_y)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            query_time += time.time() - t1

            activated_memory_dim_list.append(model.activated_memory_dim)
            preds_all.append(preds_sample)
            targets_all.append(targets_sample)
            torch.cuda.empty_cache()

    if total_support_num > 0 and support_time > 0:
        support_throughput = total_support_num / support_time
    else:
        support_throughput = 0.0

    if total_query_num > 0 and query_time > 0:
        query_throughput = total_query_num / query_time
    else:
        query_throughput = 0.0

    logger.info(
        f"Throughput:"
        f" STORE/SUPPORT = {support_throughput:.2f} items/s"
        f" (items={total_support_num}, time={support_time:.2f}s);"
        f" QUERY = {query_throughput:.2f} items/s"
        f" (items={total_query_num}, time={query_time:.2f}s)"
    )

    return preds_all, targets_all, activated_memory_dim_list


def evaluate_on_real_data(
        cfg: DictConfig,
        model,
        task_type,
        metric_logger
):
    model = model.to(cfg.model_device)
    task_list = cfg.dataset_path_list
    for task_path in task_list:
        dataset = SketchDataset(task_path, task_type, cfg.data_device, lazy=True)
        preds, targets, activated_memory_dim_list = eval_one_task(
            model, dataset, cfg.mini_batch_size, cfg.micro_batch_size)
        task_names = [f"task_{i}" for i in range(len(preds))]
        metric_logger.log_final(dataset.name, task_names, preds, targets, activated_memory_dim_list)
