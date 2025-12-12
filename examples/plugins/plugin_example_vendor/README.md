# Plugin Example – Vendor Loader

Adds a dummy vendor loader for `*.toy` files.

Install for testing:
```bash
pip install -e .
```

Then:
```bash
foodspec-plugin list
```
You should see the vendor loader listed.

The loader `load_toy` expects a CSV with columns: `meta,int1,int2`.
