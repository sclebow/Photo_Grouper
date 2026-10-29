// Photo Grouper drag-and-drop cluster board.
// All clusters and images render on one screen with no filename tags.
// Supports click / ctrl+click / shift+click selection and pointer-based
// dragging of a single image or a selected group between clusters.

export function render({ model, el }) {
  const state = {
    selection: new Set(),
    anchor: null,
    drag: null,
    ghost: null,
    suppressClick: false,
    nonce: 0,
  };

  // Report an action to Python via the synced `command` parameter.
  // A nonce guarantees repeated identical commands still trigger the watcher.
  function sendCommand(cmd) {
    state.nonce += 1;
    model.command = { ...cmd, _n: state.nonce };
  }

  const board = document.createElement("div");
  board.className = "pg-board";

  // ---- Toolbar -------------------------------------------------------------
  const toolbar = document.createElement("div");
  toolbar.className = "pg-toolbar";

  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.className = "pg-btn";
  clearBtn.textContent = "Clear selection";
  clearBtn.addEventListener("click", () => {
    state.selection.clear();
    state.anchor = null;
    updateSelectionUI();
  });

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "pg-btn";
  addBtn.textContent = "Add cluster";
  addBtn.addEventListener("click", () => sendCommand({ type: "add_cluster" }));

  const count = document.createElement("span");
  count.className = "pg-count";

  toolbar.append(clearBtn, addBtn, count);
  board.appendChild(toolbar);

  // ---- Cluster columns -----------------------------------------------------
  const scroll = document.createElement("div");
  scroll.className = "pg-board-scroll";
  const clustersRow = document.createElement("div");
  clustersRow.className = "pg-clusters";
  scroll.appendChild(clustersRow);
  board.appendChild(scroll);

  el.appendChild(board);

  // ---- Helpers -------------------------------------------------------------
  const clusterOrder = () => model.cluster_ids || [];
  const clusterName = (id) =>
    (model.cluster_names || {})[id] || `Group ${id}`;
  const imagesOf = (id) => (model.images || {})[id] || [];

  const allImageIds = () => {
    const seen = new Set();
    for (const id of clusterOrder()) {
      for (const img of imagesOf(id)) seen.add(img.id);
    }
    return seen;
  };

  const clusterOfImage = (imageId) => {
    for (const id of clusterOrder()) {
      if (imagesOf(id).some((img) => img.id === imageId)) return id;
    }
    return null;
  };

  // Build a browser-usable URL for a local image file. Panel serves the
  // configured static directory under `/photos/`, so map the filepath to it.
  function thumbUrl(img) {
    if (img.thumb) return img.thumb;
    const path = String(img.id).replace(/\\/g, "/");
    const idx = path.toLowerCase().lastIndexOf("/photos/");
    if (idx !== -1) return path.slice(idx);
    return "file:///" + path.replace(/^\/+/, "");
  }

  const thumbEl = (img) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pg-thumb";
    btn.dataset.imageId = img.id;
    btn.title = img.filename;
    const image = document.createElement("img");
    image.src = thumbUrl(img);
    image.alt = img.filename;
    image.draggable = false;
    image.loading = "lazy";
    btn.appendChild(image);
    return btn;
  };

  function updateSelectionUI() {
    board.querySelectorAll(".pg-thumb").forEach((t) => {
      t.classList.toggle("pg-selected", state.selection.has(t.dataset.imageId));
    });
    const n = state.selection.size;
    count.textContent = `${n} selected`;
    clearBtn.disabled = n === 0;
  }

  function renderBoard() {
    const valid = allImageIds();
    for (const id of [...state.selection]) {
      if (!valid.has(id)) state.selection.delete(id);
    }
    if (state.anchor && !valid.has(state.anchor)) state.anchor = null;

    clustersRow.textContent = "";

    for (const cid of clusterOrder()) {
      const col = document.createElement("div");
      col.className = "pg-cluster";
      col.dataset.clusterId = cid;

      const header = document.createElement("div");
      header.className = "pg-cluster-header";

      const nameEl = document.createElement("span");
      nameEl.className = "pg-cluster-name";
      nameEl.textContent = clusterName(cid);
      nameEl.title = "Double-click to rename";
      nameEl.addEventListener("dblclick", () => startRename(cid, nameEl));

      const badge = document.createElement("span");
      badge.className = "pg-cluster-count";
      badge.textContent = String(imagesOf(cid).length);

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "pg-cluster-remove";
      removeBtn.textContent = "\u00d7";
      removeBtn.title = "Remove this cluster (its images move to the first cluster)";
      removeBtn.addEventListener("click", () =>
        sendCommand({ type: "remove_cluster", cluster_id: cid })
      );

      header.append(nameEl, badge, removeBtn);

      const thumbs = document.createElement("div");
      thumbs.className = "pg-thumbs";
      const imgs = imagesOf(cid);
      if (imgs.length === 0) {
        const empty = document.createElement("div");
        empty.className = "pg-empty";
        empty.textContent = "Drop images here";
        thumbs.appendChild(empty);
      } else {
        for (const img of imgs) thumbs.appendChild(thumbEl(img));
      }

      col.append(header, thumbs);
      clustersRow.appendChild(col);
    }

    const add = document.createElement("div");
    add.className = "pg-add";
    add.textContent = "+";
    add.title = "Add a new cluster";
    add.addEventListener("click", () => sendCommand({ type: "add_cluster" }));
    clustersRow.appendChild(add);

    updateSelectionUI();
  }

  function startRename(cid, nameEl) {
    const input = document.createElement("input");
    input.className = "pg-cluster-name-input";
    input.value = clusterName(cid);
    nameEl.replaceWith(input);
    input.focus();
    input.select();

    let done = false;
    const commit = () => {
      if (done) return;
      done = true;
      const value = input.value.trim();
      if (value && value !== clusterName(cid)) {
        sendCommand({ type: "rename_cluster", cluster_id: cid, name: value });
      } else {
        renderBoard();
      }
    };
    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        commit();
      } else if (e.key === "Escape") {
        done = true;
        renderBoard();
      }
    });
  }

  // ---- Selection -----------------------------------------------------------
  function handleThumbClick(thumb, event) {
    if (state.suppressClick) return;
    const imageId = thumb.dataset.imageId;
    const ids = [...allImageIds()];

    if (event.ctrlKey || event.metaKey) {
      if (state.selection.has(imageId)) state.selection.delete(imageId);
      else state.selection.add(imageId);
      state.anchor = imageId;
    } else if (event.shiftKey && state.anchor) {
      const a = ids.indexOf(state.anchor);
      const b = ids.indexOf(imageId);
      if (a !== -1 && b !== -1) {
        const [lo, hi] = a < b ? [a, b] : [b, a];
        state.selection = new Set(ids.slice(lo, hi + 1));
      }
    } else {
      state.selection = new Set([imageId]);
      state.anchor = imageId;
    }
    updateSelectionUI();
  }

  // ---- Drag and drop -------------------------------------------------------
  function dragImageIds(imageId) {
    if (state.selection.has(imageId) && state.selection.size > 0) {
      return [...state.selection];
    }
    return [imageId];
  }

  function createGhost(imageIds) {
    const ghost = document.createElement("div");
    ghost.className = "pg-ghost";
    const source = imagesOf(clusterOfImage(imageIds[0])) || [];
    for (const id of imageIds.slice(0, 4)) {
      const img = source.find((i) => i.id === id);
      if (!img) continue;
      const g = document.createElement("img");
      g.src = thumbUrl(img);
      ghost.appendChild(g);
    }
    if (imageIds.length > 1) {
      const badge = document.createElement("span");
      badge.className = "pg-ghost-badge";
      badge.textContent = String(imageIds.length);
      ghost.appendChild(badge);
    }
    document.body.appendChild(ghost);
    return ghost;
  }

  function positionGhost(x, y) {
    if (state.ghost) state.ghost.style.transform = `translate(${x + 12}px, ${y + 12}px)`;
  }

  function clearDropHighlights() {
    board.querySelectorAll(".pg-cluster.pg-drop-target").forEach((c) =>
      c.classList.remove("pg-drop-target")
    );
  }

  function handlePointerMove(e) {
    const drag = state.drag;
    if (!drag) return;

    if (!drag.active) {
      if (Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) < 5) return;
      drag.active = true;
      state.suppressClick = true;
      state.ghost = createGhost(drag.imageIds);
      drag.imageIds.forEach((id) => {
        const t = board.querySelector(`.pg-thumb[data-image-id="${CSS.escape(id)}"]`);
        if (t) t.classList.add("pg-dragging");
      });
    }

    positionGhost(e.clientX, e.clientY);

    const target = document.elementFromPoint(e.clientX, e.clientY);
    const col = target && target.closest ? target.closest(".pg-cluster") : null;
    clearDropHighlights();
    if (col) col.classList.add("pg-drop-target");
  }

  function handlePointerUp(e) {
    const drag = state.drag;
    if (!drag) return;
    state.drag = null;
    document.removeEventListener("pointermove", handlePointerMove);
    document.removeEventListener("pointerup", handlePointerUp);

    if (!drag.active) return;

    if (state.ghost) {
      state.ghost.remove();
      state.ghost = null;
    }

    const target = document.elementFromPoint(e.clientX, e.clientY);
    const col = target && target.closest ? target.closest(".pg-cluster") : null;
    clearDropHighlights();

    if (col) {
      const destId = col.dataset.clusterId;
      const sourceId = clusterOfImage(drag.imageIds[0]);
      if (destId !== sourceId) {
        sendCommand({
          type: "move_images",
          image_ids: drag.imageIds,
          dest: destId,
        });
        // Keep the moved group selected in its new home.
        state.selection = new Set(drag.imageIds);
        state.anchor = drag.imageIds[0];
      }
    }

    // Allow the trailing click event to be swallowed once.
    setTimeout(() => {
      state.suppressClick = false;
    }, 0);
  }

  clustersRow.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    const thumb = e.target.closest ? e.target.closest(".pg-thumb") : null;
    if (!thumb) return;
    e.preventDefault();
    state.drag = {
      imageId: thumb.dataset.imageId,
      imageIds: dragImageIds(thumb.dataset.imageId),
      startX: e.clientX,
      startY: e.clientY,
      active: false,
    };
    document.addEventListener("pointermove", handlePointerMove);
    document.addEventListener("pointerup", handlePointerUp);
  });

  clustersRow.addEventListener("click", (e) => {
    const thumb = e.target.closest ? e.target.closest(".pg-thumb") : null;
    if (thumb) handleThumbClick(thumb, e);
  });

  // ---- Model sync ----------------------------------------------------------
  model.on("cluster_ids", renderBoard);
  model.on("cluster_names", renderBoard);
  model.on("images", renderBoard);

  renderBoard();
  return board;
}