"""Stage 2: DBSCAN clustering of photo creation times."""

import numpy as np
import pandas as pd
import param
import panel as pn
from sklearn.cluster import DBSCAN


class ClusteringStage(param.Parameterized):
    """Group photos taken within a configurable time window."""

    images_df = param.DataFrame(default=None, allow_None=True)
    grouping_minutes = param.Number(
        default=1.0,
        bounds=(0.05, 10.0),
        step=0.05,
        doc="Photos taken within this many minutes are grouped together.",
    )
    histogram_resolution = param.Integer(default=5, bounds=(1, 20))

    clustered_df = param.DataFrame(default=None, allow_None=True)
    cluster_names = param.Dict(default={})

    def __init__(self, **params):
        super().__init__(**params)
        # Recluster whenever the input data or the time window changes.
        self.param.watch(self._cluster, ["images_df", "grouping_minutes"])

    def _cluster(self, _event=None):
        df = self.images_df
        if df is None or len(df) == 0:
            self.clustered_df = None
            self.cluster_names = {}
            return None

        timestamps = np.array(
            [[dt.timestamp()] for dt in df["Creation Date"]]
        )
        labels = DBSCAN(eps=self.grouping_minutes * 60, min_samples=1).fit(timestamps).labels_

        out = df.copy()
        out["Cluster"] = labels
        out = out.sort_values("Creation Date").reset_index(drop=True)

        # Reorder clusters chronologically so Group 1 is the oldest.
        order = out.groupby("Cluster")["Creation Date"].min().sort_values().index.tolist()
        remap = {old: new for new, old in enumerate(order)}
        out["Cluster"] = out["Cluster"].map(remap)

        self.clustered_df = out
        self.cluster_names = {str(c): f"Group {c + 1}" for c in sorted(out["Cluster"].unique())}
        return out

    @param.output(clustered_df=param.DataFrame, cluster_names=param.Dict)
    def outputs(self):
        self._cluster()
        return self.clustered_df, self.cluster_names

    @param.depends("clustered_df", "histogram_resolution")
    def _histogram(self):
        df = self.clustered_df
        if df is None or len(df) == 0:
            return pn.pane.Markdown("_No clusters yet._")
        import plotly.express as px

        n_groups = df["Cluster"].nunique()
        fig = px.histogram(
            df,
            x="Creation Date",
            color="Cluster",
            barmode="stack",
            title="Photo Creation Time Distribution by Cluster",
            nbins=max(n_groups * self.histogram_resolution, 1),
        )
        return pn.pane.Plotly(fig, sizing_mode="stretch_width")

    @param.depends("clustered_df")
    def _summary(self):
        df = self.clustered_df
        if df is None or len(df) == 0:
            return pn.pane.Markdown("_No clusters yet._")
        return pn.pane.Markdown(
            f"**Total photos:** {len(df)}  \n"
            f"**Number of groups:** {df['Cluster'].nunique()}"
        )

    @param.depends("clustered_df")
    def _table(self):
        df = self.clustered_df
        if df is None or len(df) == 0:
            return pn.pane.Markdown("_No clusters yet._")
        csv = df.to_csv(index=False)
        return pn.Column(
            pn.widgets.FileDownload(
                callback=lambda: csv.encode("utf-8"),
                filename="photo_groups.csv",
                button_type="primary",
                label="Download results as CSV",
            ),
            pn.widgets.Tabulator(df, sizing_mode="stretch_width", height=300),
        )

    def panel(self):
        return pn.Column(
            pn.pane.Markdown("## 02 · Clustering"),
            pn.widgets.FloatSlider.from_param(
                self.param.grouping_minutes,
                name="Group photos taken within how many minutes?",
                sizing_mode="stretch_width",
            ),
            self._summary,
            self._histogram,
            pn.widgets.IntSlider.from_param(
                self.param.histogram_resolution,
                name="Histogram resolution (bins multiplier)",
                sizing_mode="stretch_width",
            ),
            pn.Accordion(("Cluster data", self._table), sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )