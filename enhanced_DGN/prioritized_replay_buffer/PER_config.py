"""
AILabDsUnipi/ResoLver_engine Copyright (C) 2022 AI-Lab Dept. of Digital Systems, University of Piraeus

This source code is licensed under the GPL-3.0 license found in the
LICENSE.md file in the root directory of this source tree.
"""

per_config = {
    # MEMORY
    'per_epsilon': 0.05,
    'per_alpha': 0.6,
    'starting_beta': 0.4,
    'beta_max': 1,
    'sample_rate': 5,
    # The following is the default value for 240 reducing beta steps in 1021 total exploration episodes
    'per_beta_step_decay_rate': 0.235063663
}
