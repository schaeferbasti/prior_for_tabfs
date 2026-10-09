import random

import numpy as np
import torch

from utils.prior.prior import build_loco_prior
from utils.tfm.nanoTabPFN_utils import PriorDumpDataLoader, get_default_device
from utils.tfm.nanoTabPFN import NanoTabPFNModel, train


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    old_prior_path = "../data/prior/300k_150x5_2.h5"
    new_prior_path = "../data/prior/loco_prior.h5"

    # Build the new prior with LOCO scores
    build_loco_prior(
        old_prior_path,
        new_prior_path,
        max_tasks=100
    )

    new_prior = PriorDumpDataLoader(
        filename=new_prior_path,
        num_steps=2500,
        batch_size=32,
        device=get_default_device(),
    )

    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2
    )
    model, history = train(model, new_prior, lr=4e-3, steps_per_eval=25, checkpoint_path="../utils/tfm/weights_08_10_2026.pt")
    print(model, history)


if __name__ == "__main__":
    main()
