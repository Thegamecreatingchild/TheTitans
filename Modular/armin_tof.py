import time
from tof import TOFChain  # src/hardware/tof.py
 
# I2C addresses of your sensors
ADDRESSES = [0x51, 0x52, 0x53, 0x50, 0x55, 0x56, 0x57, 0x5a]
 
chain = TOFChain(ADDRESSES)
print(f"Number of ToFs: {len(chain)}")
 
try:
    while True:
        # distances = chain.addresses  # filtered values, one per sensor
        # print(" | ".join(f"{addr:#04x}: {d:6.1f}" for addr, d in zip(ADDRESSES, distances)))
        distance = chain.__getitem__(0).read()
        print(f"{distance}")        
        
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nstopped.")