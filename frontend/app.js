document.addEventListener("DOMContentLoaded", () => {
  const statusBadge = document.getElementById("statusBadge");
  const statusText = document.getElementById("statusText");
  const sceneSelect = document.getElementById("sceneSelect");
  const sceneMeta = document.getElementById("sceneMeta");
  const metaEvent = document.getElementById("metaEvent");
  const metaDate = document.getElementById("metaDate");
  const metaSplit = document.getElementById("metaSplit");
  const metaBbox = document.getElementById("metaBbox");
  const metricsContent = document.getElementById("metricsContent");
  const exposureContent = document.getElementById("exposureContent");
  const artifactsContent = document.getElementById("artifactsContent");
  const mapOverlay = document.getElementById("mapOverlay");
  const overlayText = document.getElementById("overlayText");

  // Layer check boxes
  const toggleFlood = document.getElementById("layerFlood");
  const toggleRef = document.getElementById("layerReference");
  const toggleRoads = document.getElementById("layerRoads");
  const toggleBuildings = document.getElementById("layerBuildings");

  let currentScene = null;
  let scenesList = [];

  // Initialize MapLibre
  const mapStyle = {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: "&copy; OpenStreetMap contributors"
      }
    },
    layers: [
      {
        id: "osm-tiles",
        type: "raster",
        source: "osm",
        minzoom: 0,
        maxzoom: 19
      }
    ]
  };

  const map = new maplibregl.Map({
    container: "map",
    style: mapStyle,
    center: [-65.24, -14.23],
    zoom: 11
  });

  map.addControl(new maplibregl.NavigationControl(), "top-right");

  function showLoading(msg) {
    overlayText.textContent = msg;
    mapOverlay.classList.remove("hidden");
  }

  function hideLoading() {
    mapOverlay.classList.add("hidden");
  }

  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.ok) {
        statusBadge.classList.add("online");
        statusText.textContent = `Online (v${data.version})`;
      } else {
        statusBadge.classList.add("offline");
        statusText.textContent = "API Error";
      }
    } catch (e) {
      statusBadge.classList.add("offline");
      statusText.textContent = "Offline";
    }
  }

  async function loadScenes() {
    try {
      const res = await fetch("/api/scenes");
      const data = await res.json();
      scenesList = data.scenes || [];
      sceneSelect.innerHTML = "";
      if (scenesList.length === 0) {
        sceneSelect.innerHTML = '<option value="">No scenes available</option>';
        return;
      }
      scenesList.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s.id;
        opt.textContent = `${s.id} (${s.event_name || s.id.split("_")[0]})`;
        sceneSelect.appendChild(opt);
      });
      // Select first scene
      onSelectScene(scenesList[0].id);
    } catch (e) {
      sceneSelect.innerHTML = '<option value="">Error loading scenes</option>';
    }
  }

  sceneSelect.addEventListener("change", (e) => {
    if (e.target.value) {
      onSelectScene(e.target.value);
    }
  });

  async function onSelectScene(sceneId) {
    currentScene = scenesList.find(s => s.id === sceneId);
    if (!currentScene) return;

    // Display Scene Metadata
    sceneMeta.classList.remove("hidden");
    metaEvent.textContent = currentScene.event_name || currentScene.id.split("_")[0];
    metaDate.textContent = currentScene.acquired_date || "Unknown";
    metaSplit.textContent = currentScene.official_split || "None";
    
    // Zoom/Bounds
    let s1 = currentScene.s1_asset_id;
    if (s1 && s1.minx !== undefined) {
      const bounds = [[s1.minx, s1.miny], [s1.maxx, s1.maxy]];
      metaBbox.textContent = `[${s1.minx.toFixed(3)}, ${s1.miny.toFixed(3)}, ${s1.maxx.toFixed(3)}, ${s1.maxy.toFixed(3)}]`;
      map.fitBounds(bounds, { padding: 40 });
    }

    showLoading("Loading scene layers & analysis...");

    // Fetch analysis bundle
    try {
      const res = await fetch(`/api/analyses/${sceneId}`);
      if (res.ok) {
        const bundle = await res.json();
        renderAnalysis(bundle);
      } else {
        metricsContent.innerHTML = '<p class="muted">No analysis available for this scene.</p>';
        exposureContent.innerHTML = '<p class="muted">No exposure calculated.</p>';
        artifactsContent.innerHTML = '<p class="muted">None.</p>';
      }
    } catch (e) {
      metricsContent.innerHTML = '<p class="muted">Failed to load analysis.</p>';
    }

    // Load GeoJSON layers
    await reloadMapLayers(sceneId);
    hideLoading();
  }

  function renderAnalysis(bundle) {
    // Metrics
    if (bundle.metrics) {
      const m = bundle.metrics;
      metricsContent.innerHTML = `
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="val">${(m.iou || 0).toFixed(4)}</div>
            <div class="label">IoU Score</div>
          </div>
          <div class="metric-card">
            <div class="val">${(m.f1 || 0).toFixed(4)}</div>
            <div class="label">F1 Score</div>
          </div>
          <div class="metric-card">
            <div class="val">${(m.precision || 0).toFixed(4)}</div>
            <div class="label">Precision</div>
          </div>
          <div class="metric-card">
            <div class="val">${(m.recall || 0).toFixed(4)}</div>
            <div class="label">Recall</div>
          </div>
        </div>
      `;
    } else {
      metricsContent.innerHTML = '<p class="muted">No reference metrics recorded.</p>';
    }

    // Exposure
    if (bundle.exposure) {
      const exp = bundle.exposure;
      exposureContent.innerHTML = `
        <div class="exposure-stats">
          <div class="stat-row">
            <span>Flooded Roads:</span>
            <strong>${exp.roads.flooded_km.toFixed(3)} km</strong>
          </div>
          <div class="stat-row">
            <span>Total Roads:</span>
            <span>${exp.roads.total_km.toFixed(3)} km</span>
          </div>
          <div class="stat-row">
            <span>Flooded Buildings:</span>
            <strong>${exp.buildings.flooded}</strong>
          </div>
          <div class="stat-row">
            <span>Total Buildings:</span>
            <span>${exp.buildings.total}</span>
          </div>
          <div style="font-size: 10px; color: #64748b; margin-top: 4px;">
            ${exp.attribution || "© OpenStreetMap contributors"}
          </div>
        </div>
      `;
    } else {
      exposureContent.innerHTML = '<p class="muted">Exposure not computed yet for this scene.</p>';
    }

    // Artifacts
    if (bundle.artifacts && bundle.run_id) {
      const arts = bundle.artifacts;
      let linksHtml = '<div class="artifact-links">';
      for (const [k, v] of Object.entries(arts)) {
        if (v) {
          const fn = v.split(/[\/\\]/).pop();
          linksHtml += `<a class="artifact-link" href="/files/${bundle.run_id}/${fn}" target="_blank" download>${k} (${fn})</a>`;
        }
      }
      linksHtml += '</div>';
      artifactsContent.innerHTML = linksHtml;
    } else {
      artifactsContent.innerHTML = '<p class="muted">None.</p>';
    }
  }

  async function reloadMapLayers(sceneId) {
    const layers = ["layer-flood", "layer-reference", "layer-roads", "layer-buildings"];
    const sources = ["source-flood", "source-reference", "source-roads", "source-buildings"];

    layers.forEach(id => {
      if (map.getLayer(id)) map.removeLayer(id);
    });
    sources.forEach(id => {
      if (map.getSource(id)) map.removeSource(id);
    });

    // Add Flood Layer
    try {
      const floodRes = await fetch(`/api/layers/flood/${sceneId}`);
      if (floodRes.ok) {
        const floodData = await floodRes.json();
        map.addSource("source-flood", { type: "geojson", data: floodData });
        map.addLayer({
          id: "layer-flood",
          type: "fill",
          source: "source-flood",
          paint: {
            "fill-color": "#2563eb",
            "fill-opacity": 0.55
          },
          layout: {
            visibility: toggleFlood.checked ? "visible" : "none"
          }
        });
      }
    } catch (e) {}

    // Add Reference Layer
    try {
      const refRes = await fetch(`/api/layers/reference/${sceneId}`);
      if (refRes.ok) {
        const refData = await refRes.json();
        map.addSource("source-reference", { type: "geojson", data: refData });
        map.addLayer({
          id: "layer-reference",
          type: "line",
          source: "source-reference",
          paint: {
            "line-color": "#d97706",
            "line-width": 1.5,
            "line-dasharray": [2, 1]
          },
          layout: {
            visibility: toggleRef.checked ? "visible" : "none"
          }
        });
      }
    } catch (e) {}

    // Add Roads Layer
    try {
      const roadsRes = await fetch(`/api/layers/roads/${sceneId}`);
      if (roadsRes.ok) {
        const roadsData = await roadsRes.json();
        map.addSource("source-roads", { type: "geojson", data: roadsData });
        map.addLayer({
          id: "layer-roads",
          type: "line",
          source: "source-roads",
          paint: {
            "line-color": [
              "case",
              ["boolean", ["get", "flooded"], false],
              "#ef4444",
              "#10b981"
            ],
            "line-width": 2.5
          },
          layout: {
            visibility: toggleRoads.checked ? "visible" : "none"
          }
        });
      }
    } catch (e) {}

    // Add Buildings Layer
    try {
      const bldRes = await fetch(`/api/layers/buildings/${sceneId}`);
      if (bldRes.ok) {
        const bldData = await bldRes.json();
        map.addSource("source-buildings", { type: "geojson", data: bldData });
        map.addLayer({
          id: "layer-buildings",
          type: "fill",
          source: "source-buildings",
          paint: {
            "fill-color": [
              "case",
              ["boolean", ["get", "flooded"], false],
              "#ef4444",
              "#8b5cf6"
            ],
            "fill-opacity": 0.7
          },
          layout: {
            visibility: toggleBuildings.checked ? "visible" : "none"
          }
        });
      }
    } catch (e) {}
  }

  // Layer toggle listeners
  toggleFlood.addEventListener("change", (e) => {
    if (map.getLayer("layer-flood")) {
      map.setLayoutProperty("layer-flood", "visibility", e.target.checked ? "visible" : "none");
    }
  });

  toggleRef.addEventListener("change", (e) => {
    if (map.getLayer("layer-reference")) {
      map.setLayoutProperty("layer-reference", "visibility", e.target.checked ? "visible" : "none");
    }
  });

  toggleRoads.addEventListener("change", (e) => {
    if (map.getLayer("layer-roads")) {
      map.setLayoutProperty("layer-roads", "visibility", e.target.checked ? "visible" : "none");
    }
  });

  toggleBuildings.addEventListener("change", (e) => {
    if (map.getLayer("layer-buildings")) {
      map.setLayoutProperty("layer-buildings", "visibility", e.target.checked ? "visible" : "none");
    }
  });

  // Map click popups
  map.on("click", "layer-flood", (e) => {
    new maplibregl.Popup()
      .setLngLat(e.lngLat)
      .setHTML("<strong>Predicted Flood Extent</strong><br>Detected surface water (U-Net)")
      .addTo(map);
  });

  map.on("load", () => {
    checkHealth();
    loadScenes();
  });
});
