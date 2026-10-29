"""Drag-and-drop cluster board component.

A Panel ``JSComponent`` that renders every cluster and its images on a
single screen (without filename tags) and lets the user

* click an image to select it,
* ctrl/cmd+click or shift+click to build a multi-image selection,
* drag a single image or the whole selected group onto another cluster,
* rename clusters inline (double-click) and add/remove clusters.

Interaction state lives in the browser; the browser reports each action
by assigning a command dict to the ``command`` parameter, which Python
dispatches to the corresponding method.
"""

from pathlib import Path

import param
from panel.custom import JSComponent

_HERE = Path(__file__).parent


class ClusterBoard(JSComponent):
    """Interactive board showing all clusters and their images."""

    cluster_ids = param.List(default=[], doc="Ordered cluster identifiers.")
    cluster_names = param.Dict(default={}, doc="Mapping of cluster id to display name.")
    images = param.Dict(
        default={},
        doc="Mapping of cluster id to a list of {id, filename, thumb} image dicts.",
    )
    command = param.Dict(
        default=None,
        allow_None=True,
        doc="Command dispatched from the browser; watched and cleared by Python.",
    )

    _esm = str(_HERE / "cluster_board.js")
    _stylesheets = [str(_HERE / "cluster_board.css")]

    def __init__(self, **params):
        super().__init__(**params)
        self.param.watch(self._dispatch_command, "command")

    def _dispatch_command(self, event):
        cmd = event.new or {}
        kind = cmd.get("type")
        if kind == "move_images":
            self.move_images(cmd.get("image_ids") or [], cmd.get("dest"))
        elif kind == "rename_cluster":
            self.rename_cluster(cmd.get("cluster_id"), cmd.get("name") or "")
        elif kind == "add_cluster":
            self.add_cluster()
        elif kind == "remove_cluster":
            self.remove_cluster(cmd.get("cluster_id"))
        # Clear so an identical later command still fires the watcher.
        self.command = None

    def move_images(self, image_ids, dest_cluster_id):
        """Move ``image_ids`` into ``dest_cluster_id``."""
        moving = set(image_ids)
        if dest_cluster_id not in self.images:
            return
        images = dict(self.images)
        moved = []
        for cid in list(images):
            kept = []
            for img in images[cid]:
                (moved if img["id"] in moving else kept).append(img)
            images[cid] = kept
        if moved:
            images[dest_cluster_id] = images[dest_cluster_id] + moved
        self.images = images

    def rename_cluster(self, cluster_id, new_name):
        """Rename a cluster."""
        new_name = str(new_name).strip()
        if not new_name or cluster_id not in self.cluster_names:
            return
        names = dict(self.cluster_names)
        names[cluster_id] = new_name
        self.cluster_names = names

    def add_cluster(self):
        """Create a new empty cluster."""
        ids = [str(c) for c in self.cluster_ids]
        numeric = [int(c) for c in ids if str(c).lstrip("-").isdigit()]
        cid = str(max(numeric, default=-1) + 1)
        ids.append(cid)
        names = dict(self.cluster_names)
        names[cid] = f"Group {len(ids)}"
        images = dict(self.images)
        images[cid] = []
        self.cluster_ids = ids
        self.cluster_names = names
        self.images = images

    def remove_cluster(self, cluster_id):
        """Remove a cluster, moving its images into the first cluster."""
        ids = [str(c) for c in self.cluster_ids if str(c) != str(cluster_id)]
        if not ids:
            return
        names = dict(self.cluster_names)
        images = dict(self.images)
        orphans = images.pop(cluster_id, [])
        names.pop(cluster_id, None)
        if orphans:
            first = ids[0]
            images[first] = images.get(first, []) + orphans
        self.cluster_ids = ids
        self.cluster_names = names
        self.images = images

    # ---- Helpers used by the workflow stages ------------------------------
    def set_from_dataframe(self, df, name_column="Cluster", path_column="Filepath"):
        """Populate the board from a dataframe with cluster labels.

        ``thumb`` values are URLs relative to the served static directory
        (``/photos/<filename>``) so the browser can load them directly.
        """
        if df is None or len(df) == 0:
            self.cluster_ids = []
            self.cluster_names = {}
            self.images = {}
            return

        labels = sorted(df[name_column].unique(), key=lambda v: str(v))
        ids = [str(label) for label in labels]
        names = {str(label): f"Group {i + 1}" for i, label in enumerate(labels)}
        images = {}
        for label in labels:
            subset = df[df[name_column] == label]
            images[str(label)] = [
                {
                    "id": str(row[path_column]),
                    "filename": Path(str(row[path_column])).name,
                    "thumb": f"/photos/{Path(str(row[path_column])).name}",
                }
                for _, row in subset.iterrows()
            ]
        self.cluster_ids = ids
        self.cluster_names = names
        self.images = images

    def to_dataframe(self, base_df, name_column="Cluster", path_column="Filepath"):
        """Return ``base_df`` with the current grouping applied."""
        if base_df is None or len(base_df) == 0:
            return base_df
        mapping = {}
        for cid in self.cluster_ids:
            for img in self.images.get(cid, []):
                mapping[img["id"]] = cid
        out = base_df.copy()
        out[name_column] = out[path_column].astype(str).map(mapping)
        return out.dropna(subset=[name_column])