# ZeroMQ + mutable-shm-store cache-hit chain

The producer allocates one large mutable ndarray in mstore and sends multiple tiny NPB
control messages containing the same object capability. Stages are long-lived and call
`client.open(..., cache=True)` for every message.

For each process, the first reference is a cold open. Later references to the same
`(object_id, token, mode)` reuse the already-attached mapping entirely locally.

To make multiple in-flight messages safe, message `sequence=N` mutates `image[N]`.
With stages 1, 2, and 3, every sink slot must equal `6`.

Start consumers first, then producer:

```text
uv run python sink.py --after-stage 3 --count 10
uv run python stage.py --stage 3 --count 10
uv run python stage.py --stage 2 --count 10
uv run python stage.py --stage 1 --count 10
uv run python producer.py --gib 2 --count 10
```

On Windows ZeroMQ uses loopback TCP for the small control plane; mstore uses its Windows
named-pipe control transport plus named shared-memory mappings. On Linux ZeroMQ uses
`ipc://`, while mstore uses AF_UNIX + SCM_RIGHTS + memfd.
