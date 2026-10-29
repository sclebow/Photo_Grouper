"""Fast Photo Organizer — HoloViz Panel application.

Run with:

    panel serve main.py --dev --show

The workflow is a four-stage Panel ``Pipeline``:

    01 User Input  ->  02 Clustering  ->  03 Rename & Regroup  ->  04 Organizer

Thumbnails are served from the chosen photo directory through the
``/photos/<filename>`` route registered by :func:`register_photo_route`.
"""

import panel as pn
from panel.pipeline import Pipeline

from stages import ClusteringStage, InputStage, OrganizerStage, RenameStage

pn.extension("plotly", "tabulator", sizing_mode="stretch_width")
pn.config.global_loading_spinner = False

TITLE = "Fast Photo Organizer by Creation Date"

# Set by the input stage so the static handler knows where to read from.
PHOTO_DIRECTORY: dict[str, str] = {"path": ""}


def register_photo_route():
    """Serve the selected photo directory at ``/photos/<filename>``."""
    from tornado.web import StaticFileHandler

    class PhotoHandler(StaticFileHandler):
        def initialize(self, **kwargs):
            super().initialize(path=PHOTO_DIRECTORY["path"] or ".", **kwargs)

        def get(self, path, include_body=True):
            # Re-read the directory on every request so it follows the input.
            self.root = PHOTO_DIRECTORY["path"] or "."
            return super().get(path, include_body)

    patterns = getattr(pn.state, "extra_patterns", None)
    if patterns is None:
        patterns = []
        pn.state.extra_patterns = patterns
    patterns.append((r"/photos/(.*)", PhotoHandler))


def build_pipeline() -> Pipeline:
    """Create the four-stage workflow."""
    pipeline = Pipeline()

    pipeline.add_stage(
        "01 User Input",
        InputStage(),
        ready_parameter="ready",
    )
    pipeline.add_stage("02 Clustering", ClusteringStage())
    pipeline.add_stage("03 Rename & Regroup", RenameStage())
    pipeline.add_stage("04 Organizer", OrganizerStage())

    # The organizer needs the source directory from stage 1.
    input_stage = pipeline._stages["01 User Input"][0]
    organizer = pipeline._stages["04 Organizer"][0]

    def _sync_directory(event):
        PHOTO_DIRECTORY["path"] = event.new or ""
        organizer.directory = event.new or ""

    input_stage.param.watch(_sync_directory, "directory")
    return pipeline


def app() -> pn.Column:
    """Build the servable application."""
    register_photo_route()
    pipeline = build_pipeline()

    return pn.Column(
        pn.pane.Markdown(f"# {TITLE}"),
        pipeline,
        sizing_mode="stretch_width",
    )


app().servable()
