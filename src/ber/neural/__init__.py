"""Part A of strategy_v5.md: fine-tuned multilingual bi-encoder for retrieval and a similarity feature.

GPU modules (torch is imported inside functions, so the package imports without it):
  env_check        environment + throughput probe
  pairs            leakage-safe training triplets and the recall-gate sample (CPU)
  train_biencoder  contrastive fine-tuning + recall gate
  dense_retrieve   streaming GPU search + cosine for TF-IDF candidates
"""
