import os
import numpy as np

src = "data/tokenized/english_instruct.bin"
dst_dir = "data/tokenized_debug"
os.makedirs(dst_dir, exist_ok=True)
dst = os.path.join(dst_dir, "small.bin")

if os.path.exists(src):
    # Read first 500KB (approx 250k tokens)
    # This should be enough for "readable sentences" overfit in 3 epochs
    data = np.fromfile(src, dtype=np.uint16, count=250000)
    data.tofile(dst)
    print(f"Created {dst} with {len(data)} tokens.")
else:
    print(f"Source {src} not found.")
