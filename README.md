# Fast Photo Organizer

Organize photos into folders based on when they were taken, using DBSCAN
clustering on EXIF creation dates. Built with [HoloViz Panel](https://panel.holoviz.org/).

## Run

```powershell
.venv\Scripts\panel.exe serve main.py --dev --show
```

Then open the URL printed in the terminal (usually http://localhost:5006).

## Workflow

The app is a four-stage Panel `Pipeline` shown in the main area:

| Stage | What it does |
| --- | --- |
| **01 User Input** | Enter a photo directory and scan it. Shows a histogram of creation dates and the raw metadata. |
| **02 Clustering** | Adjust the "group photos taken within N minutes" slider. DBSCAN groups photos; clusters are renumbered chronologically. |
| **03 Rename & Regroup** | **All clusters and images appear on one screen with no filename tags.** Click an image to select it, ctrl/cmd+click to toggle, shift+click for a range, then drag the selection onto another cluster. Double-click a cluster name to rename it. Add or remove clusters with the toolbar / × buttons. |
| **04 Organizer** | Copy or move the grouped photos into numbered folders in the source directory. |

## Layout

```
main.py                     Panel entrypoint (Pipeline in a plain Column)
components/
  cluster_board.py          ClusterBoard JSComponent (Python side)
  cluster_board.js          Drag-and-drop board logic (browser side)
  cluster_board.css         Minimal board layout
stages/
  input_stage.py            Stage 1
  clustering_stage.py       Stage 2
  rename_stage.py           Stage 3
  organizer_stage.py        Stage 4
```

## Notes

* Thumbnails are served from the chosen photo directory via the `/photos/<filename>`
  route registered in `main.py`, so no base64 image data is sent over the websocket.
* The drag-and-drop board reports actions to Python through the `command`
  parameter of `ClusterBoard`; Python applies the change and the board re-renders.
