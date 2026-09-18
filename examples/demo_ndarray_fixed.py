import numpy as np

from npb import FixedNDArray, FixedStruct


class Calibration(FixedStruct):
    # 3 * 3 * 4 = 36 bytes
    k = FixedNDArray(np.float32, (3, 3))

    # 5 * 4 = 20 bytes
    dist = FixedNDArray(np.float32, (5,))


calib = Calibration(
    k=np.array(
        [
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    ),
    dist=np.zeros(5, dtype=np.float32),
)

print("byte_size:", Calibration.byte_size())  # 56
print("layout:", Calibration.layout())

payload = calib.to_bytearray()
restored = Calibration.unpack_from(payload)

print(restored.k)
print(restored.dist)
print("zero-copy view:", np.shares_memory(restored.k, np.frombuffer(payload, dtype=np.uint8)))
print("json/debug:", calib.to_dict())
print("openapi/json schema:", Calibration.json_schema())
