import time
import numpy as np
from smbus2 import SMBus


class TOF:
    def __init__(self, bus, address, n=5, alpha=0.35, max_jump=5000, accept_after=3):
        self.bus, self.address = bus, address
        self.n, self.alpha = n, alpha
        self.max_jump = max_jump          # raw units; same as the old 500 after the *0.1
        self.accept_after = accept_after  # consecutive rejects before we trust the new value
        self.rejects = 0
        self.smooth = None
        self.history = None
        self.i = 0

    def _raw(self):
        d = self.bus.read_i2c_block_data(self.address, 0x10, 5)
        return d[1] | (d[2] << 8) | (d[3] << 16) | (d[4] << 24)

    def read(self):
        try:
            raw = self._raw()
        except OSError:  # I2C hiccup: keep last value instead of crashing
            return None if self.history is None else float(np.median(self.history))

        if self.smooth is None:
            self.smooth = raw
            self.history = np.full(self.n, raw, dtype=float)
        else:
            if abs(raw - np.median(self.history)) > self.max_jump:
                self.rejects += 1
                if self.rejects < self.accept_after:
                    return float(np.median(self.history))
                self.smooth = raw                       # real change: jump straight to it
                self.history[:] = raw
            else:
                self.rejects = 0
                self.smooth += (raw - self.smooth) * self.alpha
        self.history[self.i] = self.smooth
        self.i = (self.i + 1) % self.n
        return float(np.median(self.history))


class TOFChain:
    def __init__(self, addresses, bus_num=1):
        self.bus = SMBus(bus_num)
        self.tofs = [TOF(self.bus, a) for a in addresses]

    def __getitem__(self, i): return self.tofs[i]
    def __len__(self): return len(self.tofs)
    def read(self): return [t.read() for t in self.tofs]


if __name__ == "__main__":
    chain = TOFChain([0x50])
    while True:
        vals = chain.read()
        print("Distances:", *[("--" if v is None else int(v)) for v in vals])
        time.sleep(0.02)