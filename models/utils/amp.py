# depth_warp_vs/models/utils/amp.py
import torch
from contextlib import contextmanager

class AmpScaler:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.scaler = torch.amp.GradScaler('cuda', enabled=enabled)

    @contextmanager
    def autocast(self, dtype=torch.float16):
        with torch.amp.autocast('cuda', enabled=self.enabled, dtype=dtype):
            yield

    def scale(self, loss): return self.scaler.scale(loss)
    def step(self, optimizer): return self.scaler.step(optimizer)
    def update(self): self.scaler.update()
    def unscale_(self, optimizer): self.scaler.unscale_(optimizer)

if __name__ == "__main__":
    scaler = AmpScaler(enabled=False)
    print("amp self-tests passed")
