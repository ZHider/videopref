#!/usr/bin/env python
"""CLI 训练入口。

用法::

    python train.py --data labels.json --cache-dir ./features_cache \\
        --output-dir ./checkpoints --epochs 100 --lr 1e-3 --seed 42
"""

from visualpref.train import main

if __name__ == "__main__":
    main()
