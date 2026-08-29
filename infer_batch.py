#!/usr/bin/env python
"""批量抽帧 + 喜好推理入口。

用法::

    python infer_batch.py --videos data/video_list.txt --checkpoint checkpoints/model.ckpt \\
        --output data/predictions.csv --sampling keyframe
"""

from videopref.batch_infer import main

if __name__ == "__main__":
    main()
