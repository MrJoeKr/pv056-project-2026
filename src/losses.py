from pytorch_metric_learning import losses, miners


def get_loss_and_miner(mining_strategy: str = "batch_hard", margin: float = 0.2):
    """Factory function returning (loss_fn, miner_fn) based on config.

    Args:
        mining_strategy: One of "batch_hard", "semihard", "multi_similarity"
        margin: Triplet loss margin

    Returns:
        Tuple of (loss_fn, miner_fn)
    """
    # Triplet margin loss
    loss_fn = losses.TripletMarginLoss(margin=margin)

    # Select miner
    if mining_strategy == "batch_hard":
        miner_fn = miners.BatchHardMiner()
    elif mining_strategy == "semihard":
        miner_fn = miners.BatchEasyHardMiner(
            pos_strategy="easy",
            neg_strategy="semihard",
        )
    elif mining_strategy == "multi_similarity":
        miner_fn = miners.MultiSimilarityMiner(epsilon=0.1)
    else:
        raise ValueError(f"Unknown mining strategy: {mining_strategy}")

    return loss_fn, miner_fn
