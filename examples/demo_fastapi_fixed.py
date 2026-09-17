from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from npb import Bit, Enum8, FixedPointU16, FixedStr, FixedStruct, IPv4


class CameraPipelineConfig(FixedStruct):
    camera_id = FixedStr(15)
    camera_ip = IPv4()
    need_yolo = Bit(0, False)
    need_pcd = Bit(1, False)
    yolo_stream = Enum8("rgb", "left", "right", default="rgb")
    pcd_backend = Enum8("cpu", "cuda", default="cpu")
    pcd_max_depth_m = FixedPointU16(0.001, default=2.0)


JsonModel = CameraPipelineConfig.pydantic_model()
JSON_SCHEMA = CameraPipelineConfig.json_schema()
BINARY_SCHEMA = {
    "type": "string",
    "format": "binary",
    "description": f"NPB fixed payload: exactly {CameraPipelineConfig.byte_size()} bytes",
}

app = FastAPI(title="NPB fixed demo")


@app.post(
    "/camera/config",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": JSON_SCHEMA},
                "application/x-npb": {"schema": BINARY_SCHEMA},
            },
        }
    },
    responses={
        200: {
            "content": {
                "application/json": {"schema": JSON_SCHEMA},
                "application/x-npb": {"schema": BINARY_SCHEMA},
            }
        }
    },
)
async def camera_config(request: Request) -> Response:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()

    if content_type == "application/x-npb":
        body = await request.body()
        config = CameraPipelineConfig.from_bytes(body)
        print("config",config)
        # Real gateway path may forward body unchanged to NPB-RPC here.
        return Response(content=config.to_bytes(), media_type="application/x-npb")

    if content_type == "application/json":
        model = JsonModel.model_validate(await request.json())
        config = CameraPipelineConfig.from_pydantic(model)
        print("config",config)
        return JSONResponse(content=config.to_dict())

    raise HTTPException(415, "use application/json or application/x-npb")
