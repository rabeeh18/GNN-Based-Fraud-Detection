"""Bitcoin wallet investigator (demo).

Run:  streamlit run app/investigator.py
Needs the files written by  python -m src.btc.export_demo  (results/btc/demo/).

Pick a wallet from the test period (time steps 35-49, never used for training): the app shows its risk score
from the best wallet model (chosen on the validation split), why the model scored it that way, and its money
trail: the wallet's transactions and the other wallets in them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "results" / "btc" / "demo"
LABEL_TEXT = {1: "criminal", 0: "legitimate", -1: "unknown"}
RED, AMBER, GREEN, GREY, BLUE = "#d62728", "#ff9f1c", "#2ca02c", "#9e9e9e", "#1f77b4"


@st.cache_data
def load():
    """Exported demo tables."""
    w = pd.read_parquet(DEMO / "wallets.parquet")
    links = pd.read_parquet(DEMO / "links.parquet")
    txs = pd.read_parquet(DEMO / "txs.parquet").set_index("txId")
    meta = json.loads((DEMO / "meta.json").read_text())
    return w, links, txs, meta


def risk_colour(p: float) -> str:
    return RED if p >= 0.5 else AMBER if p >= 0.2 else GREEN


def money_trail(address: str, w: pd.DataFrame, links: pd.DataFrame, txs: pd.DataFrame, max_others: int,
                show_label: bool) -> str:
    """HTML of a pyvis graph: wallet -> its transactions -> the other wallets in those transactions."""
    net = Network(height="560px", width="100%", directed=True, bgcolor="#ffffff", cdn_resources="in_line")
    net.barnes_hut(gravity=-6000, spring_length=120)
    wi = w.set_index("address")
    mine = links[links["address"] == address]
    net.add_node(address, label=address[:8] + "…", color=BLUE, size=28, shape="star",
                 title=f"{address}\n(the wallet under investigation)")
    shown = 0
    for tx_id, g in mine.groupby("txId"):
        s = float(txs.at[tx_id, "score"]) if tx_id in txs.index else float("nan")
        net.add_node(f"tx{tx_id}", label=f"tx {str(tx_id)[-5:]}", shape="square", size=14,
                     color=risk_colour(s) if s == s else GREY,
                     title=f"transaction {tx_id}\nstep {int(txs.at[tx_id, 'step']) if tx_id in txs.index else '?'}"
                           f"\ntransaction risk {s:.2f}")
        for role in g["role"]:
            if role == 0:
                net.add_edge(address, f"tx{tx_id}", color=GREY)
            else:
                net.add_edge(f"tx{tx_id}", address, color=GREY)
        others = links[(links["txId"] == tx_id) & (links["address"] != address)]
        for _, o in others.head(max(0, max_others - shown)).iterrows():
            if o["address"] in wi.index:
                ws = float(wi.at[o["address"], "score"])
                col, tip = risk_colour(ws), f"wallet risk {ws:.2f}"
                if show_label:
                    tip += f"\ntrue label: {LABEL_TEXT[int(wi.at[o['address'], 'label'])]}"
            else:
                col, tip = GREY, "not scored"
            net.add_node(o["address"], label=o["address"][:6] + "…", size=10, color=col,
                         title=f"{o['address']}\n{tip}")
            if o["role"] == 0:
                net.add_edge(o["address"], f"tx{tx_id}", color="#cccccc")
            else:
                net.add_edge(f"tx{tx_id}", o["address"], color="#cccccc")
            shown += 1
    return net.generate_html()


def main() -> None:
    st.set_page_config(page_title="Bitcoin wallet investigator", layout="wide")
    if not (DEMO / "wallets.parquet").exists():
        st.error("Demo data missing. Run:  python -m src.btc.export_demo")
        return
    w, links, txs, meta = load()
    st.title("Bitcoin wallet investigator")
    st.caption(f"Model: {meta['model']} (chosen on validation PR-AUC). Wallets from the test period "
               f"(time steps 35-49), never seen in training. Elliptic++ data (Elmougy & Liu, KDD 2023).")

    test = w[w["split"] == "test"].sort_values("score", ascending=False)
    with st.sidebar:
        st.header("Pick a wallet")
        top = test.head(100)
        choice = st.selectbox("100 highest-risk test wallets", top["address"],
                              format_func=lambda a: f"{a[:14]}…  risk {top.set_index('address').at[a, 'score']:.2f}")
        typed = st.text_input("…or paste any test wallet address")
        address = typed.strip() or choice
        show_label = st.checkbox("Reveal the true label", value=False)
        max_others = st.slider("Other wallets to draw", 10, 150, 60)

    row = w[w["address"] == address]
    if row.empty:
        st.warning("Address not found among the test-period wallets.")
        return
    r = row.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk score", f"{r['score']:.2f}")
    c2.metric("Flagged at the validation threshold", "yes" if r["score"] >= meta["threshold"] else "no")
    c3.metric("First seen (time step)", int(r["first_step"]))
    c4.metric("Transactions in its window", int(r["n_tx"]))
    if show_label:
        st.info(f"True label: **{LABEL_TEXT[int(r['label'])]}**")

    st.subheader("Why this score")
    parts = {"its own activity (counts, age, counterparties)": r["contrib_own"],
             "what its transactions look like": r["contrib_tx"]}
    if "contrib_gnn" in row and pd.notna(r.get("contrib_gnn")):
        parts["the graph model's view of its neighbourhood"] = r["contrib_gnn"]
    contrib = pd.Series(parts).sort_values()
    st.bar_chart(contrib, horizontal=True)
    st.caption("Positive bars pushed the score up, negative bars pushed it down (XGBoost feature contributions, "
               "summed per group, in log-odds).")
    mine = links[links["address"] == address]
    sus = txs.reindex(mine["txId"].unique())
    n_sus = int((sus["score"] >= meta["tx_threshold"]).sum())
    st.write(f"**{n_sus} of {len(sus)}** of its transactions are flagged by the transaction model.")

    st.subheader("Money trail")
    st.caption("Star = this wallet · squares = its transactions (red risky, amber unsure, green low risk) · "
               "circles = other wallets, coloured by their own risk score (hover for details).")
    components.html(money_trail(address, w, links, txs, max_others, show_label), height=580, scrolling=False)


main()
