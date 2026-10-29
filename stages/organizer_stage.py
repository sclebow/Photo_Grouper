"""Stage 4: copy or move photos into per-cluster folders."""

import shutil
from pathlib import Path

import param
import panel as pn


class OrganizerStage(param.Parameterized):
    """Write the grouped photos to disk."""

    directory = param.String(default="")
    clustered_df = param.DataFrame(default=None, allow_None=True)
    cluster_names = param.Dict(default={})

    copy_files = param.Boolean(default=True, doc="Copy instead of moving files.")
    combine_same_name = param.Boolean(
        default=True, doc="Combine clusters sharing a name into one folder."
    )
    run = param.Event(label="Organize photos into folders")
    status = param.String(default="")

    def __init__(self, **params):
        super().__init__(**params)
        self.param.watch(self._organize, "run")

    def _organize(self, _event=None):
        df = self.clustered_df
        if df is None or len(df) == 0:
            self.status = "⚠️ Nothing to organize."
            return
        if not Path(self.directory).is_dir():
            self.status = "⚠️ Output directory does not exist."
            return

        groups: dict[str, list[str]] = {}
        timestamps: dict[str, object] = {}

        for label, name in self.cluster_names.items():
            subset = df[df["Cluster"].astype(str) == str(label)]
            if len(subset) == 0:
                continue
            key = name if self.combine_same_name else f"{label}_{name}"
            groups.setdefault(key, []).extend(subset["Filepath"].tolist())
            oldest = subset["Creation Date"].min()
            if key not in timestamps or oldest < timestamps[key]:
                timestamps[key] = oldest

        ordered = sorted(groups, key=lambda k: timestamps[k])
        count = 0
        for i, key in enumerate(ordered):
            folder = Path(self.directory) / f"{i + 1:02d}_{key.replace(' ', '_')}"
            folder.mkdir(parents=True, exist_ok=True)
            for filepath in groups[key]:
                target = folder / Path(filepath).name
                if self.copy_files:
                    shutil.copy(filepath, target)
                else:
                    shutil.move(filepath, target)
                count += 1

        self.status = f"✅ Organized {count} photos into {len(ordered)} folders."

    def panel(self):
        return pn.Column(
            pn.pane.Markdown("## 04 · Organizer"),
            pn.widgets.Checkbox.from_param(
                self.param.copy_files, name="Copy files instead of moving them"
            ),
            pn.widgets.Checkbox.from_param(
                self.param.combine_same_name,
                name="Combine clusters with the same name into one folder",
            ),
            pn.widgets.Button.from_param(self.param.run, button_type="primary"),
            pn.pane.Alert(self.param.status, alert_type="success"),
            sizing_mode="stretch_width",
        )