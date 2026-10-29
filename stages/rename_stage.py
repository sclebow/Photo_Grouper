"""Stage 3: rename clusters and regroup images by drag and drop."""

import param
import panel as pn

from components import ClusterBoard


class RenameStage(param.Parameterized):
    """Show every cluster and image on one screen for renaming and regrouping."""

    clustered_df = param.DataFrame(default=None, allow_None=True)
    cluster_names = param.Dict(default={})

    board = param.Parameter(default=None)

    def __init__(self, **params):
        super().__init__(**params)
        self.board = ClusterBoard(sizing_mode="stretch_width")
        # The pipeline fills in clustered_df/cluster_names after construction,
        # so reload the board whenever that data arrives or changes.
        self.param.watch(self._load, ["clustered_df", "cluster_names"])
        self._load()

    def _load(self, *events):
        if self.clustered_df is None or len(self.clustered_df) == 0:
            return
        # A name-only update should not rebuild the board (that would discard
        # any drag-and-drop regrouping already done in the browser).
        names_only = bool(events) and all(e.name == "cluster_names" for e in events)
        if names_only and self.board.cluster_ids:
            names = dict(self.board.cluster_names)
            for cid, name in (self.cluster_names or {}).items():
                if cid in names:
                    names[cid] = name
            self.board.cluster_names = names
            return

        self.board.set_from_dataframe(self.clustered_df)
        # Preserve names computed during clustering.
        names = dict(self.board.cluster_names)
        for cid, name in (self.cluster_names or {}).items():
            if cid in names:
                names[cid] = name
        self.board.cluster_names = names

    @param.output(cluster_names=param.Dict, clustered_df=param.DataFrame)
    def outputs(self):
        """Hand the edited grouping to the organizer."""
        df = self.board.to_dataframe(self.clustered_df)
        return dict(self.board.cluster_names), df

    def panel(self):
        return pn.Column(
            pn.pane.Markdown(
                "## 03 · Rename & Regroup\n"
                "All clusters and images are shown together. Click an image to select it, "
                "ctrl/cmd+click or shift+click for multiple, then drag onto another cluster. "
                "Double-click a cluster name to rename it."
            ),
            self.board,
            sizing_mode="stretch_width",
        )