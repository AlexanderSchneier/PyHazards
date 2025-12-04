Datasets
===================

Summary
-------

PyHazards provides a unified dataset interface for hazard prediction across tabular, temporal, and raster data. Each dataset returns a ``DataBundle`` containing splits, feature specs, label specs, and metadata.

Core classes
------------

- ``Dataset``: base class to implement ``_load()`` and return a ``DataBundle``.
- ``DataBundle``: holds named ``DataSplit`` objects, plus ``feature_spec`` and ``label_spec``.
- ``FeatureSpec`` / ``LabelSpec``: describe inputs/targets to simplify model construction.
- ``register_dataset`` / ``load_dataset``: lightweight registry for discovering datasets by name.

Example skeleton
----------------

.. code-block:: python

    import torch
    from pyhazards.datasets import (
        DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec, register_dataset
    )

    class MyHazardDataset(Dataset):
        name = "my_hazard"

        def _load(self):
            x = torch.randn(1000, 16)
            y = torch.randint(0, 2, (1000,))
            splits = {
                "train": DataSplit(x[:800], y[:800]),
                "val": DataSplit(x[800:900], y[800:900]),
                "test": DataSplit(x[900:], y[900:]),
            }
            return DataBundle(
                splits=splits,
                feature_spec=FeatureSpec(input_dim=16, description="example features"),
                label_spec=LabelSpec(num_targets=2, task_type="classification"),
            )

    register_dataset(MyHazardDataset.name, MyHazardDataset)
