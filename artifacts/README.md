# Crane paper artifacts

- `figures/` — raw data and plotting scripts.
- `datasets/` — dataset builders.

`pip install -r requirements.txt`

## Figures and tables

`bash figures/make_all.sh` regenerates everything into `figures/out/`. 

| Paper | Script | Data |
|---|---|---|
| Fig. 4 edge weight estimation, 64 KB | `plot_edge_query.py` | `edge_query.csv` |
| Fig. 5 Uniform vs Zipf training | `plot_robustness.py` | `robustness.csv` |
| Fig. 6 cross-training grid | `plot_crosstrain.py` | `crosstrain.csv` |
| Table 1 memory-layer ablation | `make_tables.py` | `memory_layer_ablation.csv` |
| Fig. 7 carry threshold / mini-batch | `plot_hyperparams.py` | `hyperparams.csv` |
| Fig. 8 store / query throughput | `plot_throughput.py` | `throughput_{crane,baselines}.csv` |
| Fig. 9 case study D1–D5 | `plot_case_study.py` | `case_study.csv` |
| Table 3 node-flow directions (App. D.5) | `make_tables.py` | `case_study_directions.csv` |
| Fig. 10–11 Flink ablation, local / cluster | `plot_flink.py` | `flink_{ablation,local,cluster}.csv` |
| Fig. 12 query-weighted ARE (App. D.2) | `plot_weighted_are.py` | `weighted_are.csv` |
| Fig. 13 MAWI 1–100 MB (App. D.3) | `plot_mawi.py` | `mawi.csv` |
| Table 2 loss ablation (App. D.4) | `make_tables.py` | `loss_ablation.csv` |
| Sec. 6.5 training cost | `make_tables.py` (stdout) | `training_time.csv` |

## Datasets

### Data sources

- `CAIDA2018.dat` (also D2, D4) — CAIDA Anonymized Internet Traces 2018, equinix-nyc;
  [request form](https://www.caida.org/catalog/datasets/passive_dataset_request/). The builder reads
  21-byte records, source IPv4 at bytes 0–3 and destination at 4–7, big-endian.
- `NotreDame.txt` — [SNAP web-NotreDame](https://snap.stanford.edu/data/web-NotreDame.txt.gz)
- `StackOverflow.txt` — [SNAP sx-stackoverflow](https://snap.stanford.edu/data/sx-stackoverflow.txt.gz)
- `WiKiTalk.txt` — [KONECT wiki_talk_en](http://konect.cc/files/download.tsv.wiki_talk_en.tar.bz2),
  member `out.wiki_talk_en`.
- LKML — The downloaded archive is saved as `raw/Lkml_mayfly_tasks.zip` and extract it before running `build_lkml.py`.
- FinBench SF10 (D1) — LDBC FinBench v0.2.0,
  [sf10.tar.xz](https://tugraph-web.oss-cn-beijing.aliyuncs.com/tugraph/datasets/finbench/v0.2.0/sf10/sf10.tar.xz)
- RouteViews (D3, D5) — route-views2
  [2026.07/UPDATES](https://archive.routeviews.org/route-views2/bgpdata/2026.07/UPDATES/):
  `updates.20260701.0000-2345.bz2` build window, `updates.20260702.00{00,15,30,45}.bz2` query window
- MAWI (App. D.3) — `https://mawi.wide.ad.jp/mawi/samplepoint-F/2025/{202512011400,202512021400,202512031400,202512041400}.pcap.gz`

```bash
cd datasets
```

Place `CAIDA2018.dat`, `NotreDame.txt`, `StackOverflow.txt`, and `WiKiTalk.txt` in `raw/`, then run each builder independently. The out-roots below are the in-repo locations the `crane/configs/*.yaml` dataset paths point at:

```bash
python3 build_caida2018.py --raw-dir raw --out-root ForCrane
python3 build_notredame.py --raw-dir raw --out-root ForCrane
python3 build_stackoverflow.py --raw-dir raw --out-root ForCrane
python3 build_wikitalk.py --raw-dir raw --out-root ForCrane
unzip -q raw/Lkml_mayfly_tasks.zip -d raw/
python3 build_lkml.py --out-root ForCrane/Lkml
python3 validate_forcrane_npz.py <path>/0.npz [--reference <ref>/0.npz]
python3 build_mawi.py all --workdir <big disk> --out-root ForCrane

# Case-study datasets
cd downstream
python3 build_finbench_nodeflow.py --finbench-root <sf10> --name FinBench_SF10_TSR2 --full-only --out-root ../DownstreamDatasets
python3 build_caida_nodeflow.py --caida ../raw/CAIDA2018.dat --slices 2000000 --out-root ../DownstreamDatasets          # D2
python3 build_caida_subgraph.py --caida ../raw/CAIDA2018.dat --slices 2000000 --out-root ../DownstreamDatasets         # D4
python3 build_routeviews_pathquery.py parse --mrt-dir <raw_mrt> --files <updates.20260701.*,updates.20260702.00{00,15,30,45}>
python3 build_routeviews_pathquery.py build --mrt-dir <raw_mrt> --build-files <updates.20260701.*> --query-files <updates.20260702.00*> --slices 16000000 --out-root ../DownstreamDatasets
python3 build_routeviews_asfootprint.py --out-root ../DownstreamDatasets              # D5, reuses D3's stream
DOWNSTREAM_ROOT=../DownstreamDatasets python3 verify_downstream_candidates.py
```
