from npb import (
    Bit,
    Enum8,
    FixedPointU16,
    FixedStr,
    FixedStruct,
    Float32,
    IPv4,
    Nested,
    OptionalNested,
    UInt16,
)


class NetworkConfig(FixedStruct):
    ip = IPv4()
    port = UInt16(default=5555)


class Pose(FixedStruct):
    x = Float32()
    y = Float32()
    z = Float32()


class CameraPipelineConfig(FixedStruct):
    camera_id = FixedStr(15)
    network = Nested(NetworkConfig)

    need_yolo = Bit(0, False)
    need_pcd = Bit(1, False)

    yolo_stream = Enum8("rgb", "left", "right", default="rgb")
    pcd_backend = Enum8("cpu", "cuda", default="cpu")
    pcd_max_depth_m = FixedPointU16(0.001, default=2.0)

    # Optional fixed object: presence bit + inline 12-byte Pose.
    pose = OptionalNested(Pose, bit=0, default=None)


config = CameraPipelineConfig(
    camera_id="front",
    network=NetworkConfig(ip="192.168.1.10", port=5555),
    need_yolo=True,
    need_pcd=True,
    yolo_stream="rgb",
    pcd_backend="cuda",
    pcd_max_depth_m=2.5,
    pose=Pose(x=1.0, y=2.0, z=3.0),
)

payload = config.to_bytes()
restored = CameraPipelineConfig.from_bytes(payload)

print("size:", CameraPipelineConfig.byte_size())
print("hex :", payload.hex(" "))
print("restored:", restored)
print("json:", config.to_pydantic().model_dump())
print("openapi schema:", CameraPipelineConfig.json_schema())
print("layout:")
for row in CameraPipelineConfig.layout():
    print(" ", row)
