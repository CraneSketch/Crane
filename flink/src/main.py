import os
import argparse
import time

from pyflink.common import Configuration, WatermarkStrategy
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment, RuntimeExecutionMode
from pyflink.datastream.connectors.file_system import FileSource, StreamFormat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="crane",
                        choices=["crane", "crane-native", "stateful"])
    parser.add_argument("--model-path", default="weights/best_model.pth")
    parser.add_argument("--support-stream-file", required=True,
                        help="Batched support/store edge stream")
    parser.add_argument("--query-stream-file", required=True,
                        help="Batched query edge stream")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--mode", default="local", choices=["local", "cluster"],
                        help="local: embedded mini-cluster; "
                             "cluster: submit to an external Flink cluster")
    parser.add_argument("--meta-dir", default="data",
                        help="Local directory containing meta.txt / baseline_meta.txt")
    args = parser.parse_args()

    # Read metadata from local mount
    if args.method in ("crane", "crane-native"):
        meta_file = os.path.join(args.meta_dir, "meta.txt")
    else:  # stateful
        meta_file = os.path.join(args.meta_dir, "baseline_meta.txt")
    n_queries = int(open(meta_file).read().strip())

    # Flink environment
    if args.mode == "local":
        config = Configuration()
        config.set_string("taskmanager.memory.process.size", "6g")
        config.set_string("jobmanager.memory.process.size", "2g")
        env = StreamExecutionEnvironment.get_execution_environment(config)
    else:
        env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.BATCH)
    env.set_parallelism(args.parallelism)

    # ── Parse function for tab-separated batched records ──
    # Crane: bucket_id \t b64_x \t b64_y  (3 fields)
    # Stateful support: bucket_id \t b64_src \t b64_dst \t b64_w  (4 fields)
    # Stateful query: bucket_id \t b64_src \t b64_dst \t b64_truth_w  (4 fields)

    if args.method in ("crane", "crane-native"):
        if args.method == "crane":
            from operators import CraneCoProcessFunction
            op = CraneCoProcessFunction(
                args.model_path, args.device, args.micro_batch_size)
        else:
            from operators import CraneNativeCoProcessFunction
            op = CraneNativeCoProcessFunction(
                args.model_path, args.device, args.micro_batch_size)

        # Support stream (batched support edges)
        support_source = FileSource.for_record_stream_format(
            StreamFormat.text_line_format(),
            args.support_stream_file,
        ).build()
        support_ds = env.from_source(
            support_source, WatermarkStrategy.no_watermarks(), "support_stream")
        support_ds = support_ds.map(
            lambda line: tuple(line.split("\t")),
            output_type=Types.TUPLE([Types.STRING(), Types.STRING(), Types.STRING()]),
        )

        # Query stream (batched query edges)
        query_source = FileSource.for_record_stream_format(
            StreamFormat.text_line_format(),
            args.query_stream_file,
        ).build()
        query_ds = env.from_source(
            query_source, WatermarkStrategy.no_watermarks(), "query_stream")
        query_ds = query_ds.map(
            lambda line: tuple(line.split("\t")),
            output_type=Types.TUPLE([Types.STRING(), Types.STRING(), Types.STRING()]),
        )

        # Connected keyed streams → KeyedCoProcessFunction
        result = support_ds.key_by(
            lambda t: int(t[0]), key_type=Types.INT()
        ).connect(
            query_ds.key_by(lambda t: int(t[0]), key_type=Types.INT())
        ).process(
            op,
            output_type=Types.STRING(),
        )

    else:  # stateful — declarative exact-counting baseline
        from operators import StatefulCoProcessFunction
        op = StatefulCoProcessFunction()

        # Support stream (batched store edges — 4 fields)
        support_source = FileSource.for_record_stream_format(
            StreamFormat.text_line_format(),
            args.support_stream_file,
        ).build()
        support_ds = env.from_source(
            support_source, WatermarkStrategy.no_watermarks(), "support_stream")
        support_ds = support_ds.map(
            lambda line: tuple(line.split("\t")),
            output_type=Types.TUPLE([Types.STRING(), Types.STRING(),
                                     Types.STRING(), Types.STRING()]),
        )

        # Query stream (batched query edges — 4 fields: bucket, src, dst, truth_w)
        query_source = FileSource.for_record_stream_format(
            StreamFormat.text_line_format(),
            args.query_stream_file,
        ).build()
        query_ds = env.from_source(
            query_source, WatermarkStrategy.no_watermarks(), "query_stream")
        query_ds = query_ds.map(
            lambda line: tuple(line.split("\t")),
            output_type=Types.TUPLE([Types.STRING(), Types.STRING(),
                                     Types.STRING(), Types.STRING()]),
        )

        # Connected keyed streams → KeyedCoProcessFunction
        result = support_ds.key_by(
            lambda t: int(t[0]), key_type=Types.INT()
        ).connect(
            query_ds.key_by(lambda t: int(t[0]), key_type=Types.INT())
        ).process(
            op,
            output_type=Types.STRING(),
        )

    result.print()

    print(f"\n[config] method={args.method} parallelism={args.parallelism} "
          f"queries={n_queries}")
    t0 = time.time()
    env.execute(f"{args.method} Inference")
    elapsed_ms = int((time.time() - t0) * 1000)
    print(f"Job Runtime: {elapsed_ms}ms")


if __name__ == "__main__":
    main()
