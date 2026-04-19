import base64
import time
from collections import defaultdict

import numpy as np
import torch
from pyflink.common.typeinfo import Types
from pyflink.datastream.functions import KeyedCoProcessFunction
from pyflink.datastream.state import ListStateDescriptor

from crane.crane_model import Crane


# ══════════════════════════════════════════════════════════════
# Crane: KeyedCoProcessFunction — CUDA stream pipelined
# ══════════════════════════════════════════════════════════════

_N_PIN_BUFS = 32  # number of pinned memory buffers for async pipeline


class CraneCoProcessFunction(KeyedCoProcessFunction):
    """Crane query via two connected keyed streams.

    GPU pipeline: uses a dedicated CUDA stream + pinned memory double-buffering
    to fully decouple CPU (decode) from GPU (embedding + write + carry).
    CPU decodes batch N+1 while GPU processes batch N, achieving GPU-limited throughput.
    """

    def __init__(self, model_path, device="cpu", micro_batch_size=4):
        self.model_path = model_path
        self.device = device
        self.micro_batch_size = micro_batch_size

    def open(self, runtime_context):
        self._open_start = time.perf_counter()
        self.slot_id = runtime_context.get_index_of_this_subtask()
        self._is_cuda = self.device.startswith("cuda")

        # Load model
        self.model = Crane(use_fused_kernels=self._is_cuda)
        state_dict = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model = self.model.to(self.device)

        if self._is_cuda:
            self.model.optimize_for_inference(use_fp16=True)

        self.model.clear()

        # CUDA async pipeline resources
        if self._is_cuda:
            self._stream = torch.cuda.Stream()
            # Pre-allocated GPU buffers (reused — same-stream ordering ensures safety)
            self._gpu_x = torch.empty(2048, 64, dtype=torch.uint8, device=self.device)
            self._gpu_y = torch.empty(2048, dtype=torch.float32, device=self.device)
            # Pinned memory ring buffer + events for async CPU→GPU copy
            self._pin_x = [torch.empty(2048, 64, dtype=torch.uint8, pin_memory=True)
                           for _ in range(_N_PIN_BUFS)]
            self._pin_y = [torch.empty(2048, dtype=torch.float32, pin_memory=True)
                           for _ in range(_N_PIN_BUFS)]
            self._pin_events = [torch.cuda.Event() for _ in range(_N_PIN_BUFS)]
            self._pin_idx = 0

            # Pre-allocated query buffers (avoids per-query cudaHostAlloc)
            _Q_MAX = 55296  # ceil(108 * 512) — max edges per bucket
            self._q_pin_x = torch.empty(_Q_MAX, 64, dtype=torch.uint8, pin_memory=True)
            self._q_pin_y = torch.empty(_Q_MAX, dtype=torch.float32, pin_memory=True)
            self._q_gpu_x = torch.empty(_Q_MAX, 64, dtype=torch.uint8, device=self.device)
            self._q_gpu_y = torch.empty(_Q_MAX, dtype=torch.float32, device=self.device)
            self._q_pin_x_np = self._q_pin_x.numpy()
            self._q_pin_y_np = self._q_pin_y.numpy()
            self._ratio_min_buf = torch.empty(
                _Q_MAX, self.model.memory_layer,
                device=self.device, dtype=torch.float32)

            # Warmup (triggers Triton JIT compilation at realistic batch size)
            dummy_x = torch.zeros(512, 64, dtype=torch.float32, device=self.device)
            dummy_y = torch.zeros(512, dtype=torch.float32, device=self.device)
            self.model.write(dummy_x, dummy_y, micro_batch_size=self.micro_batch_size)
            self.model.query_inference(dummy_x, ratio_min_buf=self._ratio_min_buf)
            torch.cuda.synchronize()
            self.model.clear()

        # Counters
        self._total_store_edges = 0
        self.total_processed = 0
        self.total_query_time = 0.0
        self._store_count = 0
        self._are_sum = 0.0
        self._are_count = 0

        # Per-key query buffer via Flink state
        self.query_state = runtime_context.get_list_state(
            ListStateDescriptor("query_buf", Types.PICKLED_BYTE_ARRAY()))

    def process_element1(self, value, ctx):
        """Support batch from input1: (bucket_id_str, sx_b64, sy_b64)."""
        _, sx_b64, sy_b64 = value

        sx = np.frombuffer(base64.b64decode(sx_b64), dtype=np.uint8).reshape(-1, 64)
        sy = np.frombuffer(base64.b64decode(sy_b64), dtype=np.float32)
        n = len(sx)

        if self._is_cuda:
            # === CUDA stream pipeline: CPU→GPU fully async ===
            idx = self._pin_idx % _N_PIN_BUFS
            self._pin_idx += 1

            # Wait for this pinned buffer's previous DMA to complete
            self._pin_events[idx].synchronize()

            # Decode to pinned memory (CPU, fast — uint8, no float conversion)
            self._pin_x[idx][:n].copy_(torch.from_numpy(sx))
            self._pin_y[idx][:n].copy_(torch.from_numpy(sy))

            # Queue async DMA + GPU-side cast + model.write on the compute stream
            with torch.cuda.stream(self._stream):
                self._gpu_x[:n].copy_(self._pin_x[idx][:n], non_blocking=True)
                self._gpu_y[:n].copy_(self._pin_y[idx][:n], non_blocking=True)
                self._pin_events[idx].record(self._stream)
                gpu_x_f32 = self._gpu_x[:n].to(torch.float32)
                self.model.write(gpu_x_f32, self._gpu_y[:n],
                                 micro_batch_size=self.micro_batch_size)
        else:
            sx_t = torch.tensor(sx, dtype=torch.float32, device=self.device)
            sy_t = torch.tensor(sy, dtype=torch.float32, device=self.device)
            self.model.write(sx_t, sy_t, micro_batch_size=self.micro_batch_size)

        self._total_store_edges += n
        self._store_count += 1

        if self._store_count % 500 == 0:
            total_elapsed = time.perf_counter() - self._open_start
            store_tp = self._total_store_edges / total_elapsed if total_elapsed > 0 else 0.0
            print(f"[store] worker={self.slot_id} bucket={ctx.get_current_key()} "
                  f"edges={n} total_edges={self._total_store_edges} "
                  f"throughput={store_tp:.0f} edges/s "
                  f"({store_tp / 1e6:.4f} Mops) time={total_elapsed * 1000:.1f}ms")

        ctx.timer_service().register_event_time_timer(0)

    def process_element2(self, value, ctx):
        """Query batch from input2: (bucket_id_str, qx_b64, qy_b64)."""
        self.query_state.add(value)
        ctx.timer_service().register_event_time_timer(0)

    def on_timer(self, timestamp, ctx):
        """Fires per-key after all records for this key from both inputs."""
        # Drain the compute stream — all writes must complete before querying
        if self._is_cuda:
            self._stream.synchronize()

        # Log final store stats (accurate after stream sync)
        total_elapsed = time.perf_counter() - self._open_start
        store_tp = self._total_store_edges / total_elapsed if total_elapsed > 0 else 0.0
        print(f"[store] worker={self.slot_id} bucket={ctx.get_current_key()} "
              f"edges=512 total_edges={self._total_store_edges} "
              f"throughput={store_tp:.0f} edges/s "
              f"({store_tp / 1e6:.4f} Mops) time={total_elapsed * 1000:.1f}ms")
        print(f"[store_final] worker={self.slot_id} total_edges={self._total_store_edges}")

        # Decode query batches directly into pre-allocated pinned buffer
        t0 = time.perf_counter()
        if self._is_cuda:
            offset = 0
            for value in self.query_state.get():
                _, qx_b64, qy_b64 = value
                qx = np.frombuffer(base64.b64decode(qx_b64), dtype=np.uint8).reshape(-1, 64)
                qy = np.frombuffer(base64.b64decode(qy_b64), dtype=np.float32)
                n = len(qx)
                self._q_pin_x_np[offset:offset + n] = qx
                self._q_pin_y_np[offset:offset + n] = qy
                offset += n
            self.query_state.clear()
            n_total = offset
            if n_total == 0:
                return
            self._q_gpu_x[:n_total].copy_(self._q_pin_x[:n_total], non_blocking=True)
            self._q_gpu_y[:n_total].copy_(self._q_pin_y[:n_total], non_blocking=True)
            qx_t = self._q_gpu_x[:n_total].to(torch.float32)
            qy_t = self._q_gpu_y[:n_total]
        else:
            raw_batches = []
            for value in self.query_state.get():
                _, qx_b64, qy_b64 = value
                qx = np.frombuffer(base64.b64decode(qx_b64), dtype=np.uint8).reshape(-1, 64)
                qy = np.frombuffer(base64.b64decode(qy_b64), dtype=np.float32)
                raw_batches.append((qx, qy))
            self.query_state.clear()
            if not raw_batches:
                return
            all_qx = np.concatenate([b[0] for b in raw_batches])
            all_qy = np.concatenate([b[1] for b in raw_batches])
            n_total = len(all_qx)
            qx_t = torch.tensor(all_qx, dtype=torch.float32, device=self.device)
            qy_t = torch.tensor(all_qy, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            preds = self.model.query_inference(
                qx_t,
                ratio_min_buf=self._ratio_min_buf if self._is_cuda else None)
        if self._is_cuda:
            torch.cuda.synchronize()

        abs_err = (preds - qy_t).abs()
        mae = abs_err.mean().item()
        nonzero = qy_t != 0
        if nonzero.any():
            rel_errs = abs_err[nonzero] / qy_t[nonzero].abs()
            are = rel_errs.mean().item()
            self._are_sum += rel_errs.sum().item()
            self._are_count += nonzero.sum().item()
        else:
            are = 0.0
        cumulative_are = self._are_sum / self._are_count if self._are_count > 0 else 0.0

        elapsed = time.perf_counter() - t0
        self.total_processed += n_total
        self.total_query_time += elapsed
        throughput = n_total / elapsed if elapsed > 0 else 0.0
        agg = self.total_processed / self.total_query_time if self.total_query_time > 0 else 0.0

        yield (f"worker={self.slot_id} batch edges={n_total} "
               f"MAE={mae:.4f} ARE={are:.4f} "
               f"throughput={throughput:.0f} edges/s ({throughput / 1e6:.4f} Mops)")
        yield (f"[query_total] worker={self.slot_id} "
               f"total_edges={self.total_processed} "
               f"throughput={agg:.0f} edges/s ({agg / 1e6:.4f} Mops) "
               f"ARE={cumulative_are:.4f}")


# ══════════════════════════════════════════════════════════════
# Crane Native: vanilla PyTorch model, no Triton / BN-fold / pipeline
# ══════════════════════════════════════════════════════════════

class CraneNativeCoProcessFunction(KeyedCoProcessFunction):
    """Crane with the original PyTorch model — no fused kernels, no BN folding,
    no FP16, no CUDA stream pipeline.  Serves as the unoptimized baseline."""

    def __init__(self, model_path, device="cpu", micro_batch_size=4):
        self.model_path = model_path
        self.device = device
        self.micro_batch_size = micro_batch_size

    def open(self, runtime_context):
        self._open_start = time.perf_counter()
        self.slot_id = runtime_context.get_index_of_this_subtask()
        self._is_cuda = self.device.startswith("cuda")

        from crane.crane_model_native import CraneNative
        self.model = CraneNative()
        state_dict = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model.clear()

        self._total_store_edges = 0
        self.total_processed = 0
        self.total_query_time = 0.0
        self._store_count = 0
        self._are_sum = 0.0
        self._are_count = 0

        self.query_state = runtime_context.get_list_state(
            ListStateDescriptor("query_buf", Types.PICKLED_BYTE_ARRAY()))

    def process_element1(self, value, ctx):
        """Support batch from input1: (bucket_id_str, sx_b64, sy_b64)."""
        _, sx_b64, sy_b64 = value
        sx = np.frombuffer(base64.b64decode(sx_b64), dtype=np.uint8).reshape(-1, 64)
        sy = np.frombuffer(base64.b64decode(sy_b64), dtype=np.float32)
        n = len(sx)

        sx_t = torch.tensor(sx, dtype=torch.float32, device=self.device)
        sy_t = torch.tensor(sy, dtype=torch.float32, device=self.device)
        self.model.write(sx_t, sy_t, mini_batch_size=self.micro_batch_size)

        self._total_store_edges += n
        self._store_count += 1
        if self._store_count % 500 == 0:
            total_elapsed = time.perf_counter() - self._open_start
            store_tp = self._total_store_edges / total_elapsed if total_elapsed > 0 else 0.0
            print(f"[store] worker={self.slot_id} bucket={ctx.get_current_key()} "
                  f"edges={n} total_edges={self._total_store_edges} "
                  f"throughput={store_tp:.0f} edges/s "
                  f"({store_tp / 1e6:.4f} Mops) time={total_elapsed * 1000:.1f}ms")
        ctx.timer_service().register_event_time_timer(0)

    def process_element2(self, value, ctx):
        self.query_state.add(value)
        ctx.timer_service().register_event_time_timer(0)

    def on_timer(self, timestamp, ctx):
        total_elapsed = time.perf_counter() - self._open_start
        store_tp = self._total_store_edges / total_elapsed if total_elapsed > 0 else 0.0
        print(f"[store] worker={self.slot_id} bucket={ctx.get_current_key()} "
              f"edges=512 total_edges={self._total_store_edges} "
              f"throughput={store_tp:.0f} edges/s "
              f"({store_tp / 1e6:.4f} Mops) time={total_elapsed * 1000:.1f}ms")
        print(f"[store_final] worker={self.slot_id} total_edges={self._total_store_edges}")

        # Per-batch query: each 512-edge batch is a separate model.query() call
        for value in self.query_state.get():
            t0 = time.perf_counter()
            _, qx_b64, qy_b64 = value
            qx = np.frombuffer(base64.b64decode(qx_b64), dtype=np.uint8).reshape(-1, 64)
            qy = np.frombuffer(base64.b64decode(qy_b64), dtype=np.float32)
            n = len(qx)

            qx_t = torch.tensor(qx, dtype=torch.float32, device=self.device)
            qy_t = torch.tensor(qy, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                preds = self.model.query(qx_t)
            if self._is_cuda:
                torch.cuda.synchronize()

            abs_err = (preds - qy_t).abs()
            mae = abs_err.mean().item()
            nonzero = qy_t != 0
            if nonzero.any():
                rel_errs = abs_err[nonzero] / qy_t[nonzero].abs()
                are = rel_errs.mean().item()
                self._are_sum += rel_errs.sum().item()
                self._are_count += nonzero.sum().item()
            else:
                are = 0.0
            cumulative_are = self._are_sum / self._are_count if self._are_count > 0 else 0.0

            elapsed = time.perf_counter() - t0
            self.total_processed += n
            self.total_query_time += elapsed
            throughput = n / elapsed if elapsed > 0 else 0.0
            agg = self.total_processed / self.total_query_time if self.total_query_time > 0 else 0.0

            yield (f"worker={self.slot_id} batch edges={n} "
                   f"MAE={mae:.4f} ARE={are:.4f} "
                   f"throughput={throughput:.0f} edges/s ({throughput / 1e6:.4f} Mops)")
            yield (f"[query_total] worker={self.slot_id} "
                   f"total_edges={self.total_processed} "
                   f"throughput={agg:.0f} edges/s ({agg / 1e6:.4f} Mops) "
                   f"ARE={cumulative_are:.4f}")
        self.query_state.clear()


# ══════════════════════════════════════════════════════════════
# Stateful: declarative exact-counting baseline
# ══════════════════════════════════════════════════════════════

class StatefulCoProcessFunction(KeyedCoProcessFunction):
    def __init__(self):
        pass

    def open(self, runtime_context):
        self._open_start = time.perf_counter()
        self.slot_id = runtime_context.get_index_of_this_subtask()
        self.sketch = defaultdict(int)
        self._total_store_edges = 0
        self._total_store_time = 0.0
        self.total_processed = 0
        self.total_query_time = 0.0
        self._are_sum = 0.0
        self._are_count = 0
        self.query_state = runtime_context.get_list_state(
            ListStateDescriptor("query_buf", Types.PICKLED_BYTE_ARRAY()))

    def process_element1(self, value, ctx):
        t0 = time.perf_counter()
        _, src_b64, dst_b64, w_b64 = value
        src_arr = np.frombuffer(base64.b64decode(src_b64), dtype=np.int64)
        dst_arr = np.frombuffer(base64.b64decode(dst_b64), dtype=np.int64)
        w_arr = np.frombuffer(base64.b64decode(w_b64), dtype=np.int32)
        for s, d, w in zip(src_arr, dst_arr, w_arr):
            self.sketch[(int(s), int(d))] += int(w)
        elapsed = time.perf_counter() - t0
        n = len(src_arr)
        self._total_store_edges += n
        self._total_store_time += elapsed
        total_elapsed = time.perf_counter() - self._open_start
        store_tp = self._total_store_edges / total_elapsed if total_elapsed > 0 else 0.0
        print(f"[store] worker={self.slot_id} bucket={ctx.get_current_key()} "
              f"edges={n} total_edges={self._total_store_edges} "
              f"throughput={store_tp:.0f} edges/s "
              f"({store_tp / 1e6:.4f} Mops) time={total_elapsed * 1000:.1f}ms")
        ctx.timer_service().register_event_time_timer(0)

    def process_element2(self, value, ctx):
        self.query_state.add(value)
        ctx.timer_service().register_event_time_timer(0)

    def on_timer(self, timestamp, ctx):
        for value in self.query_state.get():
            t0 = time.perf_counter()
            _, src_b64, dst_b64, qw_b64 = value
            src_arr = np.frombuffer(base64.b64decode(src_b64), dtype=np.int64)
            dst_arr = np.frombuffer(base64.b64decode(dst_b64), dtype=np.int64)
            qw_arr = np.frombuffer(base64.b64decode(qw_b64), dtype=np.int64)
            n = len(src_arr)
            total_abs_err = 0.0
            total_rel_err = 0.0
            n_nonzero = 0
            for i in range(n):
                s, d = int(src_arr[i]), int(dst_arr[i])
                pred = self.sketch.get((s, d), 0)
                truth = int(qw_arr[i])
                ae = abs(pred - truth)
                total_abs_err += ae
                if truth != 0:
                    total_rel_err += ae / abs(truth)
                    n_nonzero += 1
            mae = total_abs_err / n if n > 0 else 0.0
            are = total_rel_err / n_nonzero if n_nonzero > 0 else 0.0
            self._are_sum += total_rel_err
            self._are_count += n_nonzero
            cumulative_are = self._are_sum / self._are_count if self._are_count > 0 else 0.0
            elapsed = time.perf_counter() - t0
            self.total_processed += n
            self.total_query_time += elapsed
            throughput = n / elapsed if elapsed > 0 else 0.0
            agg = self.total_processed / self.total_query_time if self.total_query_time > 0 else 0.0
            yield (f"worker={self.slot_id} batch edges={n} "
                   f"MAE={mae:.4f} ARE={are:.4f} "
                   f"throughput={throughput:.0f} edges/s ({throughput / 1e6:.4f} Mops)")
            yield (f"[query_total] worker={self.slot_id} "
                   f"total_edges={self.total_processed} "
                   f"throughput={agg:.0f} edges/s ({agg / 1e6:.4f} Mops) "
                   f"ARE={cumulative_are:.4f}")
        self.query_state.clear()
