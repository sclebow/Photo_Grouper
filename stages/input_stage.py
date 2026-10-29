"""Stage 1: choose a photo directory and scan image metadata."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import param
import panel as pn
from PIL import ExifTags, Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}


def scan_images(directory: str) -> list[dict]:
    """Extract metadata from image files without loading full image data."""
    rows = []
    for filepath in Path(directory).iterdir():
        if not filepath.is_file() or filepath.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            stat = filepath.stat()
            with Image.open(filepath) as image:
                exif_raw = image._getexif() or {}
                exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}
            taken = exif.get("DateTimeOriginal")
            try:
                taken = datetime.strptime(taken, "%Y:%m:%d %H:%M:%S")
            except (ValueError, TypeError):
                taken = None
            rows.append(
                {
                    "Filename": filepath.name,
                    "Filepath": str(filepath),
                    "Filetype": f"image/{filepath.suffix[1:].lower()}",
                    "Filesize": stat.st_size,
                    "Creation Date": taken,
                }
            )
        except (OSError, PermissionError):
            continue
    return rows


class InputStage(param.Parameterized):
    """Pick a directory and load its images."""

    directory = param.String(default="", doc="Folder containing the photos.")
    scan = param.Event(label="Scan directory")
    ready = param.Boolean(default=False)

    images_df = param.DataFrame(default=None, allow_None=True)
    nan_images_df = param.DataFrame(default=None, allow_None=True)
    message = param.String(default="")

    def __init__(self, **params):
        super().__init__(**params)
        self.param.watch(self._scan, "scan")

    def _scan(self, _event=None):
        directory = (self.directory or "").strip()
        if not directory or not Path(directory).is_dir():
            self.message = "⚠️ Please enter an existing directory path."
            self.ready = False
            return

        rows = scan_images(directory)
        if not rows:
            self.message = "⚠️ No image files found in the specified directory."
            self.ready = False
            return

        df = pd.DataFrame(rows)
        nan_df = df[df["Creation Date"].isna()]
        df = df.dropna(subset=["Creation Date"]).reset_index(drop=True)
        self.images_df = df
        self.nan_images_df = nan_df
        self.message = (
            f"Found {len(rows)} image files. "
            f"{len(df)} have valid creation dates; {len(nan_df)} are missing dates."
        )
        self.ready = len(df) > 0

    @param.output(images_df=param.DataFrame)
    def outputs(self):
        """Hand the scanned metadata to the clustering stage."""
        return self.images_df

    @param.depends("images_df")
    def _histogram(self):
        if self.images_df is None or len(self.images_df) == 0:
            return pn.pane.Markdown("_No images loaded yet._")
        import plotly.express as px

        fig = px.histogram(
            self.images_df,
            x="Creation Date",
            nbins=40,
            title="Distribution of Photo Creation Dates",
            labels={"Creation Date": "Creation Date", "count": "Number of Photos"},
        )
        return pn.pane.Plotly(fig, sizing_mode="stretch_width")

    @param.depends("nan_images_df", "images_df")
    def _raw_data(self):
        return pn.Column(
            pn.pane.Markdown("**Photos with missing creation dates:**"),
            pn.widgets.Tabulator(self.nan_images_df, sizing_mode="stretch_width", height=200),
            pn.pane.Markdown("**Photos with valid creation dates:**"),
            pn.widgets.Tabulator(self.images_df, sizing_mode="stretch_width", height=300),
        )

    def panel(self):
        return pn.Column(
            pn.pane.Markdown("## 01 · User Input"),
            pn.widgets.TextInput.from_param(
                self.param.directory,
                name="Directory path containing your photos",
                placeholder=r"e.g. C:\Users\me\Pictures",
                sizing_mode="stretch_width",
            ),
            pn.widgets.Button.from_param(self.param.scan, button_type="primary"),
            pn.pane.Alert(self.param.message, alert_type="info"),
            self._histogram,
            pn.Accordion(("See raw data", self._raw_data), sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )