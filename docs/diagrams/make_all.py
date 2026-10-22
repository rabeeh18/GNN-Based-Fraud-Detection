"""Regenerate every diagram (PNG + SVG): python docs/diagrams/make_all.py"""
import d1_table_to_graph, d2_pipeline, d3_temporal_split, d4_neighborhood_sampling, d5_gat_architecture, d6_fraud_pattern  # noqa: E401

for m in (d1_table_to_graph, d2_pipeline, d3_temporal_split, d4_neighborhood_sampling, d5_gat_architecture, d6_fraud_pattern):
    m.main()
