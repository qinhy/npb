# Codec benchmarks

The benchmark suite exercises the main in-memory codec operations using the
same nested Pydantic/NumPy shape as `examples/basic.py`:

- exact encoded-size calculation;
- encode with allocation;
- encode into a reusable output buffer;
- generic zero-copy decode;
- typed, validated zero-copy decode; and
- header-only inspection.

Run the standard suite from the repository root:

```bash
uv run python benchmarks/benchmark_codec.py
```

It uses 1 KiB, 1 MiB, and 16 MiB ndarray payloads by default. Each operation
is warmed up, adaptively calibrated, and sampled seven times. The table reports
the median; `stdev` is the population standard deviation as a percentage of
that median.

For a fast smoke test or custom payloads:

```bash
uv run python benchmarks/benchmark_codec.py --quick
uv run python benchmarks/benchmark_codec.py --sizes 64KiB 4MiB 64MiB --arrays 4
```

Results can also be saved for later comparison:

```bash
uv run python benchmarks/benchmark_codec.py --json benchmark-results.json
```

`payload GiB/s` is payload-equivalent throughput: payload bytes divided by
operation latency. It is reported for encode and decode, but not for
`encoded_size` or `peek`, which do not process payload bytes. For zero-copy
decode it does not mean those bytes were read or copied; it makes constant-time
scaling visible. Comparisons should always include operation latency and array
count. Run on an otherwise idle machine, use the same Python/dependency
versions, and compare JSON results from the same hardware.
