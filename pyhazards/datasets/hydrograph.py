# pyhazards/datasets/hydrograph.py

from typing import Optional, Dict, Tuple
import torch
import numpy as np

from pyhazards.datasets import (
    Dataset,
    DataBundle,
    DataSplit,
    FeatureSpec,
    LabelSpec,
    GraphTemporalDataset,
    register_dataset,
)

# REAL DATASETS
from pyhazards.datasets.merra2 import MERRA2
from pyhazards.datasets.noaa_flood import NOAAFlood


class HydroGraphDataset(Dataset):
    """
    Physics-ready flood forecasting dataset for HydroGraphNet.

    Uses:
      - MERRA-2 meteorology (forcing)
      - NOAA flood events (labels)
      - Fixed unstructured mesh w/ (x,y)
    """

    name = "hydrograph"

    def __init__(
        self,
        root: str,
        mesh_coords: torch.Tensor,     # (N, 2)
        adjacency: torch.Tensor,       # (N, N)
        start_date: str,
        end_date: str,
        past_steps: int = 6,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir)

        self.root = root
        self.mesh_coords = mesh_coords
        self.adjacency = adjacency
        self.start_date = start_date
        self.end_date = end_date
        self.past_steps = past_steps

        # underlying datasets
        self.merra = MERRA2(
            root=f"{root}/merra2",
            start_date=start_date,
            end_date=end_date,
        )

        self.noaa = NOAAFlood(
            root=f"{root}/noaa_flood",
            start_date=start_date,
            end_date=end_date,
        )

    # ------------------------------------------------------------
    # 🔹 Spatial projection (grid → mesh)
    # ------------------------------------------------------------
    def _project_to_mesh(
        self,
        grid_values: np.ndarray,   # (lat, lon)
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> torch.Tensor:
        """
        Nearest-neighbor projection from lat/lon grid → mesh nodes.
        """
        mesh = self.mesh_coords.numpy()
        out = np.zeros(len(mesh))

        for i, (x, y) in enumerate(mesh):
            d = (grid_lats - y) ** 2 + (grid_lons - x) ** 2
            idx = np.unravel_index(np.argmin(d), d.shape)
            out[i] = grid_values[idx]

        return torch.tensor(out, dtype=torch.float32)

    # ------------------------------------------------------------
    # 🔹 Build temporal node features
    # ------------------------------------------------------------
    def _build_node_timeseries(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          X: (T, N, F)
          Y: (T, N, 1) flood depth proxy
        """

        X_list, Y_list = [], []

        for day in self.merra.days:
            met = self.merra.load_day(day)
            print("MERRA keys:", met.keys())
            flood = self.noaa.load_day(day)

            # Example MERRA variables
            precip = self._project_to_mesh(
                met["precip"], met["lat"], met["lon"]
            )
            temp = self._project_to_mesh(
                met["temperature"], met["lat"], met["lon"]
            )

            # Flood label: binary → continuous proxy
            flood_depth = torch.zeros(len(self.mesh_coords))
            for event in flood["events"]:
                idx = flood["node_index"][event]
                flood_depth[idx] = event.get("severity", 1.0)

            X_day = torch.stack([precip, temp], dim=-1)
            Y_day = flood_depth.unsqueeze(-1)

            X_list.append(X_day)
            Y_list.append(Y_day)

        return torch.stack(X_list), torch.stack(Y_list)

    # ------------------------------------------------------------
    # 🔹 Autoregressive windowing
    # ------------------------------------------------------------
    def _window(self, X, Y):
        xs, ys = [], []
        for t in range(self.past_steps, len(X)):
            xs.append(X[t - self.past_steps : t])
            ys.append(Y[t])
        return torch.stack(xs), torch.stack(ys)

    # ------------------------------------------------------------
    # 🔹 Main loader
    # ------------------------------------------------------------
    def _load(self) -> DataBundle:
        X, Y = self._build_node_timeseries()

        # append coordinates (REQUIRED for your model)
        coords = self.mesh_coords.unsqueeze(0).repeat(X.size(0), 1, 1)
        X = torch.cat([X, coords], dim=-1)

        Xw, Yw = self._window(X, Y)

        n = len(Xw)
        n_tr, n_va = int(0.8 * n), int(0.9 * n)

        train = GraphTemporalDataset(
            Xw[:n_tr], Yw[:n_tr], adjacency=self.adjacency
        )
        val = GraphTemporalDataset(
            Xw[n_tr:n_va], Yw[n_tr:n_va], adjacency=self.adjacency
        )
        test = GraphTemporalDataset(
            Xw[n_va:], Yw[n_va:], adjacency=self.adjacency
        )

        return DataBundle(
            splits={
                "train": DataSplit(train, None),
                "val": DataSplit(val, None),
                "test": DataSplit(test, None),
            },
            feature_spec=FeatureSpec(
                input_dim=Xw.size(-1),
                description="meteorology + (x,y)",
                extra={
                    "past_steps": self.past_steps,
                    "num_nodes": Xw.size(2),
                },
            ),
            label_spec=LabelSpec(
                num_targets=1,
                task_type="regression",
                description="flood depth proxy",
            ),
            metadata={
                "adjacency": self.adjacency,
            },
        )


# 🔑 Register dataset
register_dataset(HydroGraphDataset.name, HydroGraphDataset)
