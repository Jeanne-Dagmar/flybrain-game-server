from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd



SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESOURCES = SCRIPT_DIR / "resources"
DEFAULT_OUTPUT = SCRIPT_DIR / "brain_subgraph_output"

# File candidates. The first existing candidate is used.
FILE_CANDIDATES = {
    "connections": ["connections_princeton.csv", "Connections (Filtered).csv"],
    "neurons": ["neurons.csv", "neurons(1).csv"],
    "classification": ["classification.csv"],
    "cell_types": ["consolidated_cell_types.csv"],
    "names": ["names.csv"],
    "connectivity_tags": ["connectivity_tags.csv"],
}

DEFAULT_REGIONS = {"EB", "PB", "FB", "MB", "AL", "LP"}

REGION_GROUP_PREFIXES = {
    "LP": {"SLP", "SIP", "SMP", "PLP"},
}

# Interpretation of predicted neurotransmitter, not absolute synapse truth.
NT_POLARITY = {
    "ACH": ("excitatory", 1),
    "GABA": ("inhibitory", -1),
    "GLUT": ("inhibitory", -1),
    "DA": ("modulatory", 0),
    "OCT": ("modulatory", 0),
    "SER": ("modulatory", 0),
}

REQUIRED = {
    "connections": {"pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"},
    "neurons": {
        "root_id", "group", "nt_type", "nt_type_score",
        "da_avg", "ser_avg", "gaba_avg", "glut_avg", "ach_avg", "oct_avg",
    },
    "classification": {
        "root_id", "flow", "super_class", "class", "sub_class",
        "hemilineage", "side", "nerve",
    },
    "cell_types": {"root_id", "primary_type", "additional_type(s)"},
    "names": {"root_id", "name", "group"},
    "connectivity_tags": {"root_id", "connectivity_tag"},
}


# Logging
class BuildLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.lines: list[str] = []
        self.start = time.time()

    def log(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def section(self, title: str) -> None:
        self.log("")
        self.log(f"=== {title} ===")

    def save(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        elapsed = time.time() - self.start
        self.lines.append("")
        self.lines.append(f"Elapsed seconds: {elapsed:.2f}")
        path = self.output_dir / "build_log.txt"
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return path


# Normalization / discovery

def normalize_root_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_nt(value) -> str:
    return normalize_text(value).upper()


def find_resource(resources_dir: Path, key: str) -> Path:
    candidates = FILE_CANDIDATES[key]
    for name in candidates:
        p = resources_dir / name
        if p.exists():
            return p
    available = sorted(p.name for p in resources_dir.iterdir()) if resources_dir.exists() else []
    raise SystemExit(
        f"Не найден ресурс для '{key}'.\n"
        f"Искал: {candidates}\n"
        f"Папка: {resources_dir}\n"
        f"Доступные файлы: {available}"
    )


def load_csv_checked(path: Path, required: set[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Файл не найден: {path}")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"В {label} отсутствуют колонки: {sorted(missing)}\n"
            f"Найдены: {list(df.columns)}"
        )
    return df


def parse_regions(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set(DEFAULT_REGIONS)
    allowed = set(DEFAULT_REGIONS) | {"ALL"}
    out = {v.strip().upper() for v in values if v.strip()}
    if "ALL" in out:
        return {"ALL"}
    unknown = out - allowed
    if unknown:
        raise SystemExit(
            f"Неизвестные регионы: {sorted(unknown)}. "
            f"По умолчанию: {sorted(DEFAULT_REGIONS)}; либо ALL."
        )
    return out


# Region handling
def base_region(neuropil: str) -> str:
    """Convert e.g. MB_CA_L -> MB, SLP_L -> SLP, GNG -> GNG."""
    up = normalize_text(neuropil).upper()
    if not up:
        return "UNKNOWN"
    if up.startswith("MB_"):
        return "MB"
    if up.startswith("LOF"):
        return "LOF"
    if "_" in up:
        return up.split("_")[0]
    if "." in up:
        return up.split(".")[0]
    return up


def map_requested_region(neuropil: str, requested: set[str]) -> str | None:
    if "ALL" in requested:
        return base_region(neuropil)
    base = base_region(neuropil)
    if base in requested:
        return base
    if "LP" in requested and base in REGION_GROUP_PREFIXES["LP"]:
        return "LP"
    return None


# Load datasets
def load_connections(path: Path) -> pd.DataFrame:
    raw = load_csv_checked(path, REQUIRED["connections"], path.name)
    out = raw[["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"]].copy()
    out["pre_id"] = out["pre_root_id"].map(normalize_root_id)
    out["post_id"] = out["post_root_id"].map(normalize_root_id)
    out["neuropil"] = out["neuropil"].map(normalize_text).str.upper()
    out["weight"] = pd.to_numeric(out["syn_count"], errors="coerce").fillna(0).astype("int64")
    out["edge_nt"] = out["nt_type"].map(normalize_nt)
    out = out[
        (out["pre_id"] != "") &
        (out["post_id"] != "") &
        (out["weight"] > 0)
    ].copy()
    return out[["pre_id", "post_id", "neuropil", "weight", "edge_nt"]]


def load_neurons(path: Path) -> pd.DataFrame:
    raw = load_csv_checked(path, REQUIRED["neurons"], path.name)
    out = raw.copy()
    out["root_id"] = out["root_id"].map(normalize_root_id)
    out["group"] = out["group"].map(normalize_text)
    out["nt_type"] = out["nt_type"].map(normalize_nt)
    numeric = ["nt_type_score", "da_avg", "ser_avg", "gaba_avg", "glut_avg", "ach_avg", "oct_avg"]
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values("nt_type_score", ascending=False, na_position="last")
    out = out.drop_duplicates("root_id", keep="first")
    return out


def load_classification(path: Path) -> pd.DataFrame:
    raw = load_csv_checked(path, REQUIRED["classification"], path.name)
    out = raw.copy()
    out["root_id"] = out["root_id"].map(normalize_root_id)
    for c in ["flow", "super_class", "class", "sub_class", "hemilineage", "side", "nerve"]:
        out[c] = out[c].map(normalize_text)
    out = out.drop_duplicates("root_id", keep="first")
    out["is_descending"] = out["super_class"].str.lower().eq("descending")
    return out


def load_cell_types(path: Path) -> pd.DataFrame:
    raw = load_csv_checked(path, REQUIRED["cell_types"], path.name)
    out = raw.copy()
    out["root_id"] = out["root_id"].map(normalize_root_id)
    out["primary_type"] = out["primary_type"].map(normalize_text)
    out["additional_type(s)"] = out["additional_type(s)"].map(normalize_text)
    return out.drop_duplicates("root_id", keep="first")


def load_names(path: Path) -> pd.DataFrame:
    raw = load_csv_checked(path, REQUIRED["names"], path.name)
    out = raw.copy()
    out["root_id"] = out["root_id"].map(normalize_root_id)
    out["name"] = out["name"].map(normalize_text)
    out["group"] = out["group"].map(normalize_text)
    return out.drop_duplicates("root_id", keep="first")


def load_connectivity_tags(path: Path) -> dict[str, list[str]]:
    raw = load_csv_checked(path, REQUIRED["connectivity_tags"], path.name)
    raw["root_id"] = raw["root_id"].map(normalize_root_id)
    raw["connectivity_tag"] = raw["connectivity_tag"].map(normalize_text)
    tags: dict[str, list[str]] = {}
    for _, row in raw.iterrows():
        rid = row["root_id"]
        if not rid:
            continue
        parts = [x.strip() for x in row["connectivity_tag"].split(",") if x.strip()]
        tags.setdefault(rid, [])
        tags[rid].extend(parts)
    return {rid: sorted(set(vals)) for rid, vals in tags.items()}


# Edge polarity / aggregation
def polarity_for_nt(nt: str) -> tuple[str, int]:
    return NT_POLARITY.get(nt, ("unknown", 0))


def aggregate_edge_metadata(sub: pd.DataFrame) -> dict:
    """Preserve mixed transmitter information instead of silently taking mode."""
    nt_weights = (
        sub.groupby("edge_nt", dropna=False)["weight"]
        .sum()
        .sort_values(ascending=False)
    )
    nt_breakdown = {str(k) if str(k) != "nan" else "UNKNOWN": int(v) for k, v in nt_weights.items()}

    dominant_nt = next(iter(nt_breakdown), "")
    known = [(nt, w) for nt, w in nt_breakdown.items() if nt in NT_POLARITY]
    if known:
        dominant_nt = max(known, key=lambda x: x[1])[0]

    pol, sign = polarity_for_nt(dominant_nt)
    labels = {polarity_for_nt(nt)[0] for nt in nt_breakdown if nt in NT_POLARITY}
    labels.discard("unknown")
    if len(labels) > 1:
        edge_polarity = "mixed"
        edge_sign = 0
    elif labels:
        edge_polarity = next(iter(labels))
        edge_sign = sign
    else:
        edge_polarity = "unknown"
        edge_sign = 0

    return {
        "dominant_nt_type": dominant_nt,
        "nt_breakdown": nt_breakdown,
        "polarity_label": edge_polarity,
        "sign": int(edge_sign),
    }


# Graph building
def build_graph(
    conn: pd.DataFrame,
    neurons: pd.DataFrame,
    classification: pd.DataFrame,
    cell_types: pd.DataFrame,
    names: pd.DataFrame,
    tags: dict[str, list[str]],
    requested_regions: set[str],
    keep_dn_edges_outside_regions: bool,
) -> tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame, set[str]]:

    dn_ids = set(classification.loc[classification["is_descending"], "root_id"])

    conn = conn.copy()
    conn["region"] = conn["neuropil"].map(lambda x: map_requested_region(x, requested_regions))
    conn["pre_is_dn"] = conn["pre_id"].isin(dn_ids)
    conn["post_is_dn"] = conn["post_id"].isin(dn_ids)
    conn["dn_edge"] = conn["pre_is_dn"] | conn["post_is_dn"]

    in_region = conn["region"].notna()
    if keep_dn_edges_outside_regions:
        keep_mask = in_region | conn["dn_edge"]
    else:
        keep_mask = in_region

    kept = conn.loc[keep_mask].copy()
    kept["region"] = kept["region"].fillna("DN_RELATED")
    if kept.empty:
        raise SystemExit("После фильтрации не осталось рёбер.")

    neuron_meta = neurons.set_index("root_id")
    cls_meta = classification.set_index("root_id")
    type_meta = cell_types.set_index("root_id")
    name_meta = names.set_index("root_id")

    all_ids = set(kept["pre_id"]) | set(kept["post_id"])
    G = nx.DiGraph()

    for node_id in all_ids:
        nr = neuron_meta.loc[node_id] if node_id in neuron_meta.index else None
        cr = cls_meta.loc[node_id] if node_id in cls_meta.index else None
        tr = type_meta.loc[node_id] if node_id in type_meta.index else None
        nm = name_meta.loc[node_id] if node_id in name_meta.index else None

        neuron_nt = normalize_nt(nr.get("nt_type", "")) if nr is not None else ""
        nt_score = nr.get("nt_type_score") if nr is not None else None
        pol, sign = polarity_for_nt(neuron_nt)

        super_class = normalize_text(cr.get("super_class", "")) if cr is not None else ""
        is_dn = node_id in dn_ids

        attrs = {
            "nt_type": neuron_nt,
            "nt_type_score": None if pd.isna(nt_score) else float(nt_score),
            "polarity_label": pol,
            "sign": int(sign),
            "group": normalize_text(nr.get("group", "")) if nr is not None else "",
            "name": normalize_text(nm.get("name", "")) if nm is not None else "",
            "name_group": normalize_text(nm.get("group", "")) if nm is not None else "",
            "primary_type": normalize_text(tr.get("primary_type", "")) if tr is not None else "",
            "additional_type": normalize_text(tr.get("additional_type(s)", "")) if tr is not None else "",
            "flow": normalize_text(cr.get("flow", "")) if cr is not None else "",
            "super_class": super_class,
            "class": normalize_text(cr.get("class", "")) if cr is not None else "",
            "sub_class": normalize_text(cr.get("sub_class", "")) if cr is not None else "",
            "hemilineage": normalize_text(cr.get("hemilineage", "")) if cr is not None else "",
            "side": normalize_text(cr.get("side", "")) if cr is not None else "",
            "nerve": normalize_text(cr.get("nerve", "")) if cr is not None else "",
            "is_descending": bool(is_dn),
            "connectivity_tags": ",".join(tags.get(node_id, [])),
        }

        for c in ["da_avg", "ser_avg", "gaba_avg", "glut_avg", "ach_avg", "oct_avg"]:
            value = nr.get(c) if nr is not None else None
            attrs[c] = None if pd.isna(value) else float(value)

        G.add_node(node_id, **attrs)

    # One DiGraph edge per pre->post pair, but preserve transmitter mixture and
    # neuropil breakdown so information is not lost by aggregation.
    grouped_pairs = kept.groupby(["pre_id", "post_id"], sort=False)
    for (pre_id, post_id), sub in grouped_pairs:
        meta = aggregate_edge_metadata(sub)
        region_weights = sub.groupby("region")["weight"].sum().sort_values(ascending=False)
        region_breakdown = {str(k): int(v) for k, v in region_weights.items()}

        # Confidence is the pre-neuron's predicted NT score, when available.
        pre_score = G.nodes[pre_id].get("nt_type_score")
        dominant_nt = meta["dominant_nt_type"]
        dominant_nt_weight = meta["nt_breakdown"].get(dominant_nt, 0)
        total_weight = int(sub["weight"].sum())
        dominant_fraction = dominant_nt_weight / total_weight if total_weight else 0.0

        G.add_edge(
            pre_id,
            post_id,
            synapse_count=total_weight,
            dominant_nt_type=dominant_nt,
            nt_breakdown=json.dumps(meta["nt_breakdown"], ensure_ascii=False, separators=(",", ":")),
            dominant_nt_fraction=round(float(dominant_fraction), 6),
            nt_confidence=None if pre_score is None else float(pre_score),
            polarity_label=meta["polarity_label"],
            sign=int(meta["sign"]),
            regions=json.dumps(region_breakdown, ensure_ascii=False, separators=(",", ":")),
        )

    return G, kept, dn_ids, all_ids


# DN candidates (secondary diagnostic only)
def make_dn_candidates(kept: pd.DataFrame, official_dn_ids: set[str]) -> pd.DataFrame:
    """Rank non-official nodes that heavily target DN-associated neuropils."""
    if kept.empty:
        return pd.DataFrame(columns=["root_id", "out_weight", "target_neuropils", "candidate_score", "official_dn"])

    rel_regions = {"GNG", "AVLP", "PVLP", "LAL", "FLA", "IB", "IPS", "SAD", "VES"}
    tmp = kept.copy()
    tmp["target_base"] = tmp["neuropil"].map(base_region)
    rel = tmp[tmp["target_base"].isin(rel_regions)].copy()
    if rel.empty:
        return pd.DataFrame(columns=["root_id", "out_weight", "target_neuropils", "candidate_score", "official_dn"])

    stats = rel.groupby("pre_id").agg(
        out_weight=("weight", "sum"),
        target_neuropils=("target_base", lambda x: ",".join(sorted(set(x)))),
        target_count=("target_base", "nunique"),
    ).reset_index(names="root_id")
    total = tmp.groupby("pre_id")["weight"].sum().rename("total_out_weight")
    stats = stats.merge(total, left_on="root_id", right_index=True, how="left")
    stats["candidate_score"] = stats["out_weight"] / stats["total_out_weight"].clip(lower=1)
    stats["official_dn"] = stats["root_id"].isin(official_dn_ids)
    stats = stats[~stats["official_dn"]]
    return stats[["root_id", "out_weight", "target_neuropils", "candidate_score", "official_dn"]].sort_values(
        ["candidate_score", "out_weight"], ascending=False
    )


# Export
def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def export_graph(
    G: nx.DiGraph,
    out_dir: Path,
    dn_df: pd.DataFrame,
    missing_dn_df: pd.DataFrame,
    candidates: pd.DataFrame,
    diagnostics: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = [{"root_id": n, **{k: json_safe(v) for k, v in a.items()}} for n, a in G.nodes(data=True)]
    edges = [{"pre_id": u, "post_id": v, **{k: json_safe(val) for k, val in a.items()}} for u, v, a in G.edges(data=True)]

    pd.DataFrame(nodes).to_csv(out_dir / "nodes.csv", index=False)
    pd.DataFrame(edges).to_csv(out_dir / "edges.csv", index=False)

    graph_json = {
        "nodes": [{"id": n, **{k: json_safe(v) for k, v in a.items()}} for n, a in G.nodes(data=True)],
        "links": [{"source": u, "target": v, **{k: json_safe(val) for k, val in a.items()}} for u, v, a in G.edges(data=True)],
    }
    with open(out_dir / "graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_json, f, ensure_ascii=False, indent=2)

    nx.write_graphml(G, out_dir / "graph.graphml")

    dn_df.to_csv(out_dir / "descending_neurons.csv", index=False)
    missing_dn_df.to_csv(out_dir / "descending_missing_from_connections.csv", index=False)
    candidates.to_csv(out_dir / "descending_candidates.csv", index=False)

    with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)


# Main
def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FlyWire subgraph using connectivity + FlyWire annotations.")
    parser.add_argument("--resources", default=str(DEFAULT_RESOURCES), help="Directory containing the CSV resources.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory.")
    parser.add_argument("--regions", nargs="+", default=None, help="Regions: EB PB FB MB AL LP, or ALL.")
    parser.add_argument("--no-dn-outside-regions", action="store_true", help="Do not keep DN incident edges outside selected regions.")
    args = parser.parse_args()

    resources = Path(args.resources).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    regions = parse_regions(args.regions)

    logger = BuildLogger(output)
    logger.section("RESOURCE DISCOVERY")
    logger.log(f"Resources: {resources}")
    logger.log(f"Output:   {output}")

    paths = {key: find_resource(resources, key) for key in FILE_CANDIDATES}
    for key, path in paths.items():
        logger.log(f"{key:18s}: {path.name}")

    logger.section("LOAD")
    conn = load_connections(paths["connections"])
    neurons = load_neurons(paths["neurons"])
    classification = load_classification(paths["classification"])
    cell_types = load_cell_types(paths["cell_types"])
    names = load_names(paths["names"])
    tags = load_connectivity_tags(paths["connectivity_tags"])

    conn_ids = set(conn["pre_id"]) | set(conn["post_id"])
    neuron_ids = set(neurons["root_id"])
    cls_ids = set(classification["root_id"])
    type_ids = set(cell_types["root_id"])
    name_ids = set(names["root_id"])
    tag_ids = set(tags)

    official_dn_ids = set(classification.loc[classification["is_descending"], "root_id"])
    dn_in_connections = official_dn_ids & conn_ids
    dn_missing = official_dn_ids - conn_ids

    logger.log(f"Connections rows:                 {len(conn):,}")
    logger.log(f"Unique neuron IDs in connections: {len(conn_ids):,}")
    logger.log(f"Neurons rows:                     {len(neurons):,}")
    logger.log(f"Classification rows:              {len(classification):,}")
    logger.log(f"Cell types rows:                  {len(cell_types):,}")
    logger.log(f"Names rows:                       {len(names):,}")
    logger.log(f"Connectivity-tagged neurons:      {len(tag_ids):,}")
    logger.log(f"ID intersection conn/neuron:      {len(conn_ids & neuron_ids):,} ({len(conn_ids & neuron_ids) / max(1, len(conn_ids)):.2%})")
    logger.log(f"ID intersection conn/class:       {len(conn_ids & cls_ids):,} ({len(conn_ids & cls_ids) / max(1, len(conn_ids)):.2%})")
    logger.log(f"Official descending neurons:       {len(official_dn_ids):,}")
    logger.log(f"DN present in connections:         {len(dn_in_connections):,}")
    logger.log(f"DN missing from connections:       {len(dn_missing):,}")
    logger.log(f"Selected regions:                  {', '.join(sorted(regions))}")
    logger.log(f"Keep DN edges outside regions:     {not args.no_dn_outside_regions}")

    logger.section("CLASSIFICATION")
    class_counts = classification["super_class"].replace("", "<empty>").value_counts()
    for name, count in class_counts.head(20).items():
        logger.log(f"super_class={name}: {int(count):,}")

    flow_counts = classification["flow"].replace("", "<empty>").value_counts()
    logger.log("Top flow values:")
    for name, count in flow_counts.head(10).items():
        logger.log(f"  {name}: {int(count):,}")

    logger.section("BUILD GRAPH")
    G, kept, dn_ids, all_ids = build_graph(
        conn,
        neurons,
        classification,
        cell_types,
        names,
        tags,
        regions,
        keep_dn_edges_outside_regions=not args.no_dn_outside_regions,
    )

    candidates = make_dn_candidates(kept, official_dn_ids)

    dn_records = classification[classification["is_descending"]].copy()
    dn_records = dn_records.merge(cell_types, on="root_id", how="left", suffixes=("", "_celltype"))
    dn_records = dn_records.merge(names, on="root_id", how="left", suffixes=("", "_name"))
    dn_records["present_in_connections"] = dn_records["root_id"].isin(conn_ids)
    dn_records = dn_records.sort_values(["present_in_connections", "primary_type", "root_id"], ascending=[False, True, True])

    missing_dn_df = dn_records[~dn_records["present_in_connections"]].copy()

    logger.log(f"Input connection rows kept:      {len(kept):,}")
    logger.log(f"Graph nodes:                      {G.number_of_nodes():,}")
    logger.log(f"Graph edges:                      {G.number_of_edges():,}")
    logger.log(f"Official DN nodes in graph:       {sum(bool(G.nodes[n].get('is_descending')) for n in G.nodes):,}")
    logger.log(f"Non-DN heuristic candidates:      {len(candidates):,}")

    logger.section("NEUROTRANSMITTER / POLARITY")
    edge_nt_counter = Counter()
    edge_pol_counter = Counter()
    for _, _, a in G.edges(data=True):
        edge_nt_counter[a.get("dominant_nt_type", "UNKNOWN")] += 1
        edge_pol_counter[a.get("polarity_label", "unknown")] += 1
    logger.log("Dominant edge NT types:")
    for nt, count in edge_nt_counter.most_common():
        logger.log(f"  {nt}: {count:,} edges")
    logger.log("Edge polarity labels:")
    for pol, count in edge_pol_counter.most_common():
        logger.log(f"  {pol}: {count:,} edges")
    logger.log("NOTE: polarity is inferred from predicted neurotransmitter; it is not an explicit per-synapse ground-truth field.")

    logger.section("CONNECTIVITY TAGS")
    tag_counter = Counter(tag for vals in tags.values() for tag in vals)
    for tag, count in tag_counter.most_common():
        logger.log(f"  {tag}: {count:,} neurons")

    neuropil_counts = kept.groupby("neuropil")["weight"].agg(["count", "sum"]).sort_values("sum", ascending=False)
    diagnostics = {
        "resources_dir": str(resources),
        "output_dir": str(output),
        "files": {k: str(v) for k, v in paths.items()},
        "connections_input_rows": int(len(conn)),
        "unique_connection_nodes": int(len(conn_ids)),
        "neurons_rows": int(len(neurons)),
        "classification_rows": int(len(classification)),
        "cell_types_rows": int(len(cell_types)),
        "names_rows": int(len(names)),
        "connectivity_tagged_neurons": int(len(tag_ids)),
        "connection_neuron_id_intersection": int(len(conn_ids & neuron_ids)),
        "connection_classification_id_intersection": int(len(conn_ids & cls_ids)),
        "official_descending_neurons": int(len(official_dn_ids)),
        "descending_present_in_connections": int(len(dn_in_connections)),
        "descending_missing_from_connections": int(len(dn_missing)),
        "selected_regions": sorted(regions),
        "keep_dn_edges_outside_regions": not args.no_dn_outside_regions,
        "kept_connection_rows": int(len(kept)),
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
        "official_dn_nodes_in_graph": int(sum(bool(G.nodes[n].get("is_descending")) for n in G.nodes)),
        "heuristic_non_dn_candidates": int(len(candidates)),
        "polarity_note": "Inferred from predicted neurotransmitter; not explicit per-synapse ground truth.",
        "nt_polarity_mapping": NT_POLARITY,
        "neuropils": {
            str(idx): {"rows": int(row["count"]), "synapses": int(row["sum"])}
            for idx, row in neuropil_counts.iterrows()
        },
        "edge_dominant_nt_counts": {k: int(v) for k, v in edge_nt_counter.items()},
        "edge_polarity_counts": {k: int(v) for k, v in edge_pol_counter.items()},
        "connectivity_tag_counts": {k: int(v) for k, v in tag_counter.items()},
    }

    export_graph(G, output, dn_records, missing_dn_df, candidates, diagnostics)
    log_path = logger.save()

    print("\n=== DONE ===")
    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")
    print(f"Descending neurons: {len(official_dn_ids):,}")
    print(f"Descending neurons in graph: {sum(bool(G.nodes[n].get('is_descending')) for n in G.nodes):,}")
    print(f"Output: {output}")
    print(f"Log: {log_path}")
    print("Files:")
    for name in [
        "nodes.csv", "edges.csv", "graph.json", "graph.graphml",
        "descending_neurons.csv", "descending_missing_from_connections.csv",
        "descending_candidates.csv", "diagnostics.json", "build_log.txt",
    ]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
