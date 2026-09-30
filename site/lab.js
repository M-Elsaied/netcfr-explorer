/* Side-by-side lab: four live NetCFR runs on the same network, differing only in how
   neighbours' strategies are transmitted. Uses NetSim (sim.js) and Chart.js. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const isDark = () => { const t = document.documentElement.dataset.theme;
    return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; };
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const sup = (n) => String(n).split("").map((c) => SUP[c] ?? c).join("");
  const fmt = (x) => {
    if (!isFinite(x)) return "–";
    if (x !== 0 && Math.abs(x) < 1e-3) { const [m, e] = x.toExponential(2).split("e"); return `${m}×10${sup(parseInt(e, 10))}`; }
    return x.toLocaleString(undefined, { maximumSignificantDigits: 3 });
  };

  const APPROACHES = [
    { key: "exact", name: "Exact messages", desc: "full-precision probabilities (reference)", color: () => css("--text") },
    { key: "det", name: "Deterministic rounding", desc: "round to the nearest of 2^q levels", color: () => "#e03131" },
    { key: "sto", name: "Stochastic rounding", desc: "round up or down at random; correct on average", color: () => "#1c7ed6" },
    { key: "ef", name: "Error feedback", desc: "carry each rounding error into the next message", color: () => "#12b886" },
  ];
  const T_MAX = 10000;
  const SPEEDS = { slow: 2, normal: 12, fast: 60 };
  const SEQ = ["#e3f2fd", "#90caf9", "#ffd166", "#f77f00", "#c1121f", "#5a0a0f"];
  const SEQ_DARK = ["#1b2a3a", "#2f6fae", "#e0b43c", "#f77f00", "#ff4d5a", "#ffc2c7"];
  const ramp = (stops, t) => {
    t = Math.max(0, Math.min(1, t));
    const x = t * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2), f = x - i;
    const a = stops[i].match(/\w\w/g).map((h) => parseInt(h, 16)), b = stops[i + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
    return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
  };

  const S = { graph: "er100", variant: "cfr+", q: 1, speed: "normal", seed: 1, playing: false, started: false };
  let graphs = null, game = null, runs = [], circles = [], chart = null, raf = null, frame = 0;

  function seg(id, options, get, set) {
    const el = $(id);
    el.innerHTML = "";
    options.forEach(([v, label]) => {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = label; b.className = get() === v ? "on" : "";
      b.addEventListener("click", () => { set(v); [...el.children].forEach((c) => c.classList.toggle("on", c === b)); });
      el.appendChild(b);
    });
  }

  function buildPanels() {
    const grid = $("lab-grid");
    grid.innerHTML = APPROACHES.map((a) => `
      <div class="lab-panel" id="lab-${a.key}">
        <div class="lab-head"><span class="lab-dot" style="background:${a.color()}"></span>
          <div><div class="lab-name">${a.name}</div><div class="lab-desc">${a.desc}</div></div></div>
        <div class="lab-stats"><div><span class="lbl">NashConv per agent</span><span class="val" data-v="nc">–</span></div>
          <div><span class="lbl">bits / iteration</span><span class="val small" data-v="bits">–</span></div></div>
        <svg viewBox="0 0 600 420" class="lab-svg" role="img" aria-label="${a.name}: agents colored by exploitability"></svg>
      </div>`).join("");
  }

  function drawGraphs() {
    const G = graphs[S.graph];
    const X = (x) => 18 + x * 564, Y = (y) => 18 + y * 384;
    const edges = G.edges.map(([s, t]) => `<line x1="${X(G.xy[s][0])}" y1="${Y(G.xy[s][1])}" x2="${X(G.xy[t][0])}" y2="${Y(G.xy[t][1])}"/>`).join("");
    const r = G.n > 60 ? 9.5 : 13;
    const nodes = G.xy.map(([x, y], i) => `<circle cx="${X(x)}" cy="${Y(y)}" r="${r}"><title>Agent ${i}</title></circle>`).join("");
    circles = [];
    document.querySelectorAll(".lab-svg").forEach((svg) => {
      svg.innerHTML = `<g class="lab-edges">${edges}</g><g>${nodes}</g>`;
      circles.push([...svg.querySelectorAll("circle")]);
    });
  }

  function reset() {
    game = new NetSim.Game(graphs[S.graph]);
    runs = APPROACHES.map((a) => new NetSim.Run(game, { variant: S.variant, channel: a.key, q: S.q, seed: S.seed }));
    frame = 0;
    drawGraphs();
    APPROACHES.forEach((a, k) => {
      const bits = a.key === "exact" ? 2 * game.E * 2 * game.M * 64 : runs[k].bits;
      document.querySelector(`#lab-${a.key} [data-v="bits"]`).textContent = bits.toLocaleString() + (a.key === "exact" ? " (64-bit floats)" : "");
    });
    buildChart();
    render(true);
  }

  function buildChart() {
    if (chart) chart.destroy();
    const grid = css("--grid");
    chart = new Chart($("lab-chart"), {
      type: "line",
      data: { datasets: APPROACHES.map((a) => ({ label: a.name, data: [], borderColor: a.color(), backgroundColor: a.color(),
        borderWidth: a.key === "exact" ? 2.2 : 2, pointRadius: 0, borderDash: a.key === "det" ? [6, 4] : a.key === "ef" ? [2, 3] : [] })) },
      options: { responsive: true, maintainAspectRatio: false, animation: false, parsing: false, normalized: true,
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: { type: "logarithmic", min: 1, max: T_MAX, title: { display: true, text: "Iteration" }, grid: { color: grid },
            ticks: { callback: (v) => { const e = Math.log10(v); return Math.abs(e - Math.round(e)) < 1e-9 ? `10${sup(Math.round(e))}` : ""; } } },
          y: { type: "logarithmic", title: { display: true, text: "NashConv per agent" }, grid: { color: grid },
            ticks: { callback: (v) => { const e = Math.log10(v); return Math.abs(e - Math.round(e)) < 1e-9 ? `10${sup(Math.round(e))}` : ""; } } },
        },
        plugins: { legend: { position: "bottom", labels: { boxWidth: 14 } },
          tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y)}` } } } },
    });
  }

  function render(force) {
    const dark = isDark(), stops = dark ? SEQ_DARK : SEQ;
    const t = runs[0].t;
    runs.forEach((r, k) => {
      const tot = t > 0 ? r.evaluate() : NaN;
      const nc = tot / game.n;
      if (t > 0) chart.data.datasets[k].data.push({ x: t, y: Math.max(nc, 1e-7) });
      document.querySelector(`#lab-${APPROACHES[k].key} [data-v="nc"]`).textContent = fmt(nc);
      if (t > 0) circles[k].forEach((c, i) => c.setAttribute("fill", ramp(stops, (Math.log10(Math.max(r.gap[i], 1e-5)) + 5) / 5)));
      else circles[k].forEach((c) => c.setAttribute("fill", css("--edge")));
    });
    $("lab-t").textContent = t.toLocaleString();
    $("lab-bar").style.width = `${(100 * t) / T_MAX}%`;
    if (force || frame % 3 === 0) chart.update("none");
  }

  function tick() {
    const k = Math.min(SPEEDS[S.speed], T_MAX - runs[0].t);
    for (let s = 0; s < k; s++) runs.forEach((r) => r.step());
    frame += 1;
    render(false);
    if (runs[0].t >= T_MAX) { pause(); chart.update("none"); $("lab-play").textContent = "✓ Done"; return; }
    raf = requestAnimationFrame(tick);
  }
  function play() {
    if (runs[0].t >= T_MAX) reset();
    S.playing = true; S.started = true; $("lab-play").textContent = "❚❚ Pause";
    raf = requestAnimationFrame(tick);
  }
  function pause() { S.playing = false; cancelAnimationFrame(raf); $("lab-play").textContent = "▶ Play"; }

  function restart() { const was = S.playing; pause(); reset(); if (was) play(); }

  function init(data) {
    graphs = data;
    buildPanels();
    seg("lab-graph", [["er100", "100 agents"], ["er40", "40 agents"]], () => S.graph, (v) => { S.graph = v; restart(); });
    seg("lab-learner", [["cfr+", "CFR+"], ["cfr", "CFR"]], () => S.variant, (v) => { S.variant = v; restart(); });
    seg("lab-bits", [[1, "1 bit"], [2, "2 bits"], [3, "3 bits"], [4, "4 bits"]], () => S.q, (v) => { S.q = v; restart(); });
    seg("lab-speed", [["slow", "Slow"], ["normal", "Normal"], ["fast", "Fast"]], () => S.speed, (v) => { S.speed = v; });
    $("lab-play").addEventListener("click", () => (S.playing ? pause() : play()));
    $("lab-reset").addEventListener("click", restart);
    $("lab-seed").addEventListener("click", () => { S.seed += 1; $("lab-seed-v").textContent = S.seed; restart(); });
    reset();
    // start automatically the first time the lab scrolls into view
    new IntersectionObserver((entries, obs) => {
      if (entries.some((e) => e.isIntersecting) && !S.started) { play(); obs.disconnect(); }
    }, { threshold: 0.25 }).observe($("lab"));
    const retheme = () => {
      document.querySelectorAll(".lab-dot").forEach((d, k) => { d.style.background = APPROACHES[k].color(); });
      const keep = chart.data.datasets.map((ds) => ds.data);
      buildChart(); chart.data.datasets.forEach((ds, k) => { ds.data = keep[k]; }); render(true);
    };
    $("theme").addEventListener("click", () => setTimeout(retheme, 0));
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);
  }

  fetch("data/lab_graphs.json").then((r) => r.json()).then(init).catch((e) => {
    $("lab-grid").innerHTML = `<p class="muted">Could not load the lab (${e}).</p>`;
  });
})();
