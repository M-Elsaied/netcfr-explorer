/* NetCFR Explorer: renders results/data bundle produced by experiments/export_site.py */
(() => {
  "use strict";

  // ------------------------------------------------------------------ theme
  const root = document.documentElement;
  const saved = (() => { try { return localStorage.getItem("theme"); } catch { return null; } })();
  if (saved) root.dataset.theme = saved;
  const isDark = () => root.dataset.theme ? root.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  document.getElementById("theme").addEventListener("click", () => {
    root.dataset.theme = isDark() ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch { /* ignore */ }
    rerenderAll();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (!root.dataset.theme) rerenderAll(); });
  const css = (v) => getComputedStyle(root).getPropertyValue(v).trim();

  // ------------------------------------------------------------------ palette and labels
  const COLORS = {
    "pcfr+": "#e8590c", "cfr+": "#1c7ed6", dcfr: "#12b886", cfr: "#868e96", fp: "#c92a2a",
    omwu: "#ae3ec9", hedge: "#f59f00", pg_exact: "#0b7285", reinforce: "#5c940d", bandit: "#d6336c",
  };
  const NAMES = {
    "pcfr+": "PCFR+", "cfr+": "CFR+", dcfr: "DCFR", cfr: "CFR", fp: "Fictitious play",
    omwu: "OMWU", hedge: "Hedge", pg_exact: "Exact policy gradient", reinforce: "REINFORCE", bandit: "EXP3 bandit",
  };
  const base = (k) => k.split("_eta")[0];
  const label = (k) => {
    const m = k.match(/_eta(\d+)/);
    return (NAMES[base(k)] || k) + (m ? ` (η=${m[1]})` : "");
  };
  const color = (k) => COLORS[base(k)] || "#888";
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const sup = (n) => String(n).split("").map((c) => SUP[c] ?? c).join("");
  const fmt = (x, d = 3) => {
    if (x === null || x === undefined || Number.isNaN(x)) return "–";
    if (Number.isInteger(x) && Math.abs(x) >= 1) return x.toLocaleString();
    const a = Math.abs(x);
    if (a !== 0 && a < 1e-3) {
      const [m, e] = x.toExponential(1).split("e");
      return `${m}×10${sup(parseInt(e, 10))}`;
    }
    return Number(x.toPrecision(d)).toLocaleString();
  };

  // ------------------------------------------------------------------ chart helpers
  const charts = {};
  function chartDefaults() {
    Chart.defaults.color = css("--muted");
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 12;
    Chart.defaults.borderColor = css("--grid");
  }
  function makeChart(id, config) {
    if (charts[id]) charts[id].destroy();
    const ctx = document.getElementById(id);
    config.options = Object.assign({ responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "nearest", intersect: false } }, config.options || {});
    charts[id] = new Chart(ctx, config);
    return charts[id];
  }
  const logAxis = (title) => ({ type: "logarithmic", title: { display: !!title, text: title },
    grid: { color: css("--grid") }, ticks: { callback: (v) => {
      const e = Math.log10(v); return Math.abs(e - Math.round(e)) < 1e-9 ? `10${sup(Math.round(e))}` : ""; } } });
  const linAxis = (title, extra = {}) => Object.assign({ type: "linear", title: { display: !!title, text: title },
    grid: { color: css("--grid") } }, extra);
  const tooltipFmt = { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y, 4)}` } };

  function segmented(id, options, initial, onChange) {
    const el = document.getElementById(id);
    el.innerHTML = "";
    let value = initial;
    options.forEach(([v, text]) => {
      const b = document.createElement("button");
      b.textContent = text;
      b.type = "button";
      b.className = v === value ? "on" : "";
      b.addEventListener("click", () => {
        value = v;
        [...el.children].forEach((c) => c.classList.toggle("on", c === b));
        onChange(v);
      });
      el.appendChild(b);
    });
    return () => value;
  }

  // ------------------------------------------------------------------ data
  let D = null;
  const state = { netGraph: "er40", netLearner: "cfr+", netProfile: "gap_avg", netColor: "gap", netIdx: 0,
    lcGraph: "ba100", lcProfile: "nc_avg", selected: null };

  fetch("data/results.json").then((r) => r.json()).then((data) => {
    D = data;
    buildControls();
    rerenderAll();
  }).catch((e) => {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<p class="card">Could not load results (${e}).</p>`);
  });

  function rerenderAll() {
    if (!D) return;
    chartDefaults();
    renderKpis(); renderNetwork(); renderLearning(); renderFeedback(); renderCommunication();
    renderScaling(); renderTopology(); renderFooter();
  }

  // ------------------------------------------------------------------ KPIs
  function renderKpis() {
    const sc = D.scaling.rows, cm = D.communication;
    const big = sc[sc.length - 1];
    const q3 = cm.quantized.find((q) => q.q === 3);
    const ba = D.ba100_learners.runs["pcfr+"];
    const items = [
      [`${fmt(big.n)}`, `agents updated in ${fmt(big.s_per_iter, 2)} s per iteration`],
      [fmt(ba.nc_avg[ba.nc_avg.length - 1], 2), "total NashConv of PCFR+ on a 100-agent network after 3,000 iterations"],
      ["3 bits", `per probability give NashConv/agent ${fmt(q3.nashconv_per_agent, 2)} vs ${fmt(cm.exact.nashconv_per_agent, 2)} exact`],
      [`${Math.round(D.er40_learners.runs["reinforce"].nc_avg.at(-1) / D.er40_learners.runs["cfr+"].nc_avg.at(-1))}×`,
        "higher exploitability for sampled-reward learning than for counterfactual updates"],
    ];
    document.getElementById("kpis").innerHTML = items.map(([v, l]) =>
      `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  }

  // ------------------------------------------------------------------ network view
  let timer = null;
  function buildControls() {
    segmented("net-graph", [["er40", "40 agents"], ["ba100", "100 agents"]], state.netGraph, (v) => {
      state.netGraph = v; state.selected = null; fillLearners(); renderNetwork(); });
    segmented("net-profile", [["gap_avg", "Average"], ["gap_last", "Current"]], state.netProfile, (v) => {
      state.netProfile = v; renderNetwork(); });
    segmented("net-color", [["gap", "Exploitability"], ["eq", "Equilibrium payoff"]], state.netColor, (v) => {
      state.netColor = v; renderNetwork(); });
    fillLearners();
    document.getElementById("net-learner").addEventListener("change", (e) => { state.netLearner = e.target.value; renderNetwork(); });
    const slider = document.getElementById("net-slider");
    slider.addEventListener("input", () => { state.netIdx = +slider.value; renderNetwork(); });
    document.getElementById("net-play").addEventListener("click", togglePlay);
    segmented("lc-graph", [["ba100", "100 agents"], ["er40", "40 agents"]], state.lcGraph, (v) => { state.lcGraph = v; renderLearning(); });
    segmented("lc-profile", [["nc_avg", "Average profile"], ["nc_last", "Current policy"]], state.lcProfile, (v) => { state.lcProfile = v; renderLearning(); });
  }
  function fillLearners() {
    const sel = document.getElementById("net-learner");
    const keys = Object.keys(D.network[state.netGraph].runs);
    if (!keys.includes(state.netLearner)) state.netLearner = keys.find((k) => base(k) === base(state.netLearner)) || keys[0];
    sel.innerHTML = keys.map((k) => `<option value="${k}" ${k === state.netLearner ? "selected" : ""}>${label(k)}</option>`).join("");
    const snaps = D.network[state.netGraph].runs[state.netLearner];
    const slider = document.getElementById("net-slider");
    slider.max = snaps.length - 1;
    state.netIdx = Math.min(state.netIdx, snaps.length - 1);
    slider.value = state.netIdx;
  }
  function togglePlay() {
    const btn = document.getElementById("net-play");
    if (timer) { clearInterval(timer); timer = null; btn.textContent = "▶ Play"; return; }
    const slider = document.getElementById("net-slider");
    if (state.netIdx >= +slider.max) state.netIdx = 0;
    btn.textContent = "❚❚ Pause";
    timer = setInterval(() => {
      state.netIdx += 1;
      if (state.netIdx > +slider.max) { togglePlay(); state.netIdx = +slider.max; }
      slider.value = state.netIdx;
      renderNetwork();
    }, 450);
  }

  // perceptual sequential ramp (low = cool/quiet, high = hot) and diverging ramp
  const SEQ = ["#e3f2fd", "#90caf9", "#ffd166", "#f77f00", "#c1121f", "#5a0a0f"];
  const SEQ_DARK = ["#1b2a3a", "#2f6fae", "#e0b43c", "#f77f00", "#ff4d5a", "#ffc2c7"];
  const DIV = ["#c92a2a", "#ff8787", "#f1f3f5", "#74c0fc", "#1864ab"];
  const DIV_DARK = ["#ff6b6b", "#a33a3a", "#3a3f4a", "#3b78b8", "#74c0fc"];
  function lerpColor(stops, t) {
    t = Math.max(0, Math.min(1, t));
    const x = t * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2), f = x - i;
    const a = stops[i].match(/\w\w/g).map((h) => parseInt(h, 16)), b = stops[i + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
    return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
  }
  const RATIO_COLORS = { 0.5: "#74c0fc", 1: "#20c997", 2: "#fcc419", 3: "#ff922b", 4: "#f03e3e" };

  function renderNetwork() {
    const net = D.network[state.netGraph];
    const snaps = net.runs[state.netLearner];
    const snap = snaps[Math.min(state.netIdx, snaps.length - 1)];
    const gaps = snap[state.netProfile].map((g) => Math.max(g, 0));
    const svg = document.getElementById("net-svg");
    const W = 1000, H = 720, pad = 42;
    const X = (x) => pad + x * (W - 2 * pad), Y = (y) => pad + y * (H - 2 * pad - 28);
    const maxDeg = Math.max(...net.nodes.map((n) => n.degree));
    const R = (d) => 6 + 14 * Math.sqrt(d / maxDeg);
    // color mapping
    const dark = isDark();
    let fill, legendHtml;
    if (state.netColor === "gap") {
      const lo = -4, hi = 0; // log10 of per-agent gap, fixed across time so frames are comparable
      fill = (i) => lerpColor(dark ? SEQ_DARK : SEQ, (Math.log10(Math.max(gaps[i], 1e-6)) - lo) / (hi - lo));
      const grad = (dark ? SEQ_DARK : SEQ).join(",");
      legendHtml = `<span>Agent exploitability</span><span>10⁻⁴</span><span class="bar" style="background:linear-gradient(90deg,${grad})"></span><span>≥1</span>`;
    } else {
      const m = Math.max(...net.nodes.map((n) => Math.abs(n.eq_payoff)), 1e-9);
      fill = (i) => lerpColor(dark ? DIV_DARK : DIV, 0.5 + net.nodes[i].eq_payoff / (2 * m));
      const grad = (dark ? DIV_DARK : DIV).join(",");
      legendHtml = `<span>Equilibrium payoff</span><span>${fmt(-m, 2)}</span><span class="bar" style="background:linear-gradient(90deg,${grad})"></span><span>+${fmt(m, 2)}</span>`;
    }
    const ratios = [...new Set(net.edges.map((e) => e.ratio))].sort((a, b) => a - b);
    legendHtml += ratios.map((r) => `<span><span class="sw" style="background:${RATIO_COLORS[r] || "#999"}"></span>bet ratio ${r}</span>`).join("");
    document.getElementById("net-legend").innerHTML = legendHtml;

    const edges = net.edges.map((e) => {
      const a = net.nodes[e.s], b = net.nodes[e.t];
      return `<line x1="${X(a.x)}" y1="${Y(a.y)}" x2="${X(b.x)}" y2="${Y(b.y)}" stroke="${RATIO_COLORS[e.ratio] || css("--edge")}" stroke-opacity="${dark ? 0.55 : 0.7}" stroke-width="${1 + e.ratio * 0.45}"/>`;
    }).join("");
    const order = net.nodes.map((n, i) => i).sort((a, b) => net.nodes[a].degree - net.nodes[b].degree);
    const nodes = order.map((i) => {
      const n = net.nodes[i];
      return `<circle data-i="${i}" cx="${X(n.x)}" cy="${Y(n.y)}" r="${R(n.degree)}" fill="${fill(i)}" class="${state.selected === i ? "sel" : ""}"><title>Agent ${i}</title></circle>`;
    }).join("");
    svg.innerHTML = `<g>${edges}</g><g>${nodes}</g>`;
    svg.querySelectorAll("circle").forEach((c) => {
      const pick = () => { state.selected = +c.dataset.i; showNode(); svg.querySelectorAll("circle").forEach((o) => o.classList.toggle("sel", o === c)); };
      c.addEventListener("mouseenter", pick);
      c.addEventListener("click", pick);
    });
    const total = snap[state.netProfile].reduce((s, g) => s + g, 0);
    document.getElementById("net-t").textContent = snap.t.toLocaleString();
    document.getElementById("net-nc").textContent = fmt(total, 3);
    const worst = gaps.indexOf(Math.max(...gaps));
    document.getElementById("net-max").textContent = `${fmt(gaps[worst], 3)} (agent ${worst})`;
    showNode();
  }
  function showNode() {
    const box = document.getElementById("net-node");
    const i = state.selected;
    if (i === null) { box.innerHTML = `<p class="muted">Hover or tap an agent for details.</p>`; return; }
    const net = D.network[state.netGraph];
    const n = net.nodes[i];
    const snap = net.runs[state.netLearner][Math.min(state.netIdx, net.runs[state.netLearner].length - 1)];
    box.innerHTML = `<p><strong>Agent ${i}</strong></p><dl>
      <dt>Neighbors</dt><dd>${n.degree}</dd>
      <dt>Bet ratios</dt><dd>${n.ratios.join(", ")}</dd>
      <dt>Exploitability now</dt><dd>${fmt(Math.max(snap[state.netProfile][i], 0), 3)}</dd>
      <dt>Equilibrium payoff</dt><dd>${fmt(n.eq_payoff, 3)}</dd>
      <dt>Security value</dt><dd>${fmt(n.security, 3)}</dd></dl>`;
  }

  // ------------------------------------------------------------------ learning curves
  function renderLearning() {
    const runs = D[state.lcGraph === "ba100" ? "ba100_learners" : "er40_learners"].runs;
    const keys = Object.keys(runs).filter((k) => !["pg_exact", "reinforce", "bandit"].includes(k)
      && !(state.lcGraph === "ba100" && /_eta2$|hedge_eta50|omwu_eta10/.test(k)));
    const datasets = keys.map((k) => ({
      label: label(k), borderColor: color(k), backgroundColor: color(k), borderWidth: 2, pointRadius: 0,
      borderDash: /_eta/.test(k) && base(k) === "hedge" ? [5, 4] : [],
      data: runs[k].t.map((t, j) => ({ x: t, y: Math.max(runs[k][state.lcProfile][j], 1e-6) })),
    }));
    makeChart("lc-chart", { type: "line", data: { datasets }, options: {
      parsing: false, scales: { x: linAxis("Iteration", { min: 0, max: 3000 }), y: logAxis("Total NashConv") },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 14, usePointStyle: false } }, tooltip: tooltipFmt } } });
    const rows = keys.map((k) => [k, runs[k].nc_avg.at(-1), runs[k].nc_last.at(-1)]).sort((a, b) => a[1] - b[1]);
    document.getElementById("lc-table").innerHTML = `<thead><tr><th>Learner</th><th>Average profile</th><th>Current policy</th><th>Current ÷ average</th></tr></thead><tbody>` +
      rows.map(([k, a, l]) => `<tr><td><span class="sw" style="background:${color(k)}"></span>${label(k)}</td><td>${fmt(a, 3)}</td><td>${fmt(l, 3)}</td><td>${fmt(l / a, 2)}×</td></tr>`).join("") + `</tbody>`;
  }

  // ------------------------------------------------------------------ feedback
  function renderFeedback() {
    const R = D.er40_learners.runs;
    const keys = ["pcfr+", "cfr+", "omwu_eta10", "pg_exact", "reinforce", "bandit"];
    makeChart("fb-chart", { type: "bar", data: { labels: keys.map(label), datasets: [
      { label: "Average profile", data: keys.map((k) => R[k].nc_avg.at(-1)), backgroundColor: keys.map(color) },
      { label: "Current policy", data: keys.map((k) => R[k].nc_last.at(-1)), backgroundColor: keys.map((k) => color(k) + "55"),
        borderColor: keys.map(color), borderWidth: 1.5 },
    ] }, options: { scales: { y: Object.assign(logAxis("Total NashConv"), { min: 1e-4 }), x: { grid: { display: false } } },
      plugins: { legend: { position: "bottom" }, tooltip: tooltipFmt } } });
  }

  // ------------------------------------------------------------------ communication
  function renderCommunication() {
    const cm = D.communication;
    const accent = css("--accent");
    makeChart("cm-knee", { type: "line", data: { datasets: [
      { label: "q-bit messages", data: cm.quantized.map((q) => ({ x: q.bits_per_iter / 1000, y: q.nashconv_per_agent, q: q.q })),
        borderColor: accent, backgroundColor: accent, pointRadius: 5, borderWidth: 2 },
      { label: "Exact messages", data: [{ x: 0, y: cm.exact.nashconv_per_agent }, { x: 20, y: cm.exact.nashconv_per_agent }],
        borderColor: css("--muted"), borderDash: [6, 4], pointRadius: 0, borderWidth: 1.5 },
    ] }, options: { parsing: false, scales: { x: linAxis("kbits per iteration", { min: 0, max: 20 }), y: logAxis("NashConv per agent") },
      plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => c.raw.q ? `${c.raw.q} bits: ${fmt(c.parsed.y, 3)}` : `exact: ${fmt(c.parsed.y, 3)}` } } } } });
    const qColors = ["#c92a2a", "#f76707", "#f59f00", "#37b24d", "#1c7ed6", "#7048e8"];
    makeChart("cm-curves", { type: "line", data: { datasets: [
      ...cm.quantized.map((q, i) => ({ label: `${q.q} bit${q.q > 1 ? "s" : ""}`, borderColor: qColors[i], backgroundColor: qColors[i], pointRadius: 0, borderWidth: 2,
        data: q.curve.t.map((t, j) => ({ x: t, y: q.curve.nc_per_agent[j] })) })),
      { label: "exact", borderColor: css("--text"), borderDash: [5, 4], pointRadius: 0, borderWidth: 1.5,
        data: cm.exact.curve.t.map((t, j) => ({ x: t, y: cm.exact.curve.nc_per_agent[j] })) },
    ] }, options: { parsing: false, scales: { x: linAxis("Iteration", { min: 0, max: 1000 }), y: logAxis("NashConv per agent") },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } }, tooltip: tooltipFmt } } });
    makeChart("cm-noise", { type: "line", data: { datasets: [{ label: "Laplace noise", data: [{ x: 0, y: cm.exact.nashconv_per_agent },
      ...cm.laplace.map((l) => ({ x: l.sigma, y: l.nashconv_per_agent }))], borderColor: "#d6336c", backgroundColor: "#d6336c", pointRadius: 5, borderWidth: 2 }] },
      options: { parsing: false, scales: { x: linAxis("Noise scale σ", { min: 0, max: 0.32 }), y: logAxis("NashConv per agent") },
        plugins: { legend: { display: false }, tooltip: tooltipFmt } } });
    makeChart("cm-bound", { type: "bar", data: { labels: cm.quantized.map((q) => `${q.q} bit${q.q > 1 ? "s" : ""}`), datasets: [
      { label: "Guaranteed bound term (per agent)", data: cm.quantized.map((q) => q.bound_term_per_agent), backgroundColor: css("--edge") },
      { label: "Measured NashConv per agent", data: cm.quantized.map((q) => q.nashconv_per_agent), backgroundColor: accent },
    ] }, options: { scales: { y: logAxis(""), x: { grid: { display: false } } }, plugins: { legend: { position: "bottom" }, tooltip: tooltipFmt } } });
    const q3 = cm.quantized.find((q) => q.q === 3);
    document.getElementById("cm-central").innerHTML = `<strong>For comparison, a centralized solve.</strong> If one party knew the whole
      network, sending the public game description once would take about ${fmt(cm.centralized.bits_once)} bits, and a single linear
      program solves it exactly in ${fmt(cm.centralized.lp_seconds, 2)} s. 1,000 learning iterations with 3-bit messages move
      ${fmt(q3.bits_per_iter * 1000 / 1e6, 2)} million bits. Local learning pays off when no single party can see the whole network.`;
  }

  // ------------------------------------------------------------------ scaling
  function renderScaling() {
    const rows = D.scaling.rows;
    const accent = css("--accent");
    const ideal = rows.map((r) => ({ x: r.n, y: rows[0].s_per_iter * r.edges / rows[0].edges }));
    makeChart("sc-time", { type: "line", data: { datasets: [
      { label: "Measured", data: rows.map((r) => ({ x: r.n, y: r.s_per_iter })), borderColor: accent, backgroundColor: accent, pointRadius: 5, borderWidth: 2 },
      { label: "Linear in edges (from smallest run)", data: ideal, borderColor: css("--muted"), borderDash: [6, 4], pointRadius: 0, borderWidth: 1.5 },
    ] }, options: { parsing: false, scales: { x: logAxis("Agents"), y: logAxis("Seconds per iteration") },
      plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y, 3)} s` } } } } });
    makeChart("sc-nc", { type: "bar", data: { labels: rows.map((r) => `${fmt(r.n)} agents`), datasets: [
      { label: "NashConv per agent", data: rows.map((r) => r.nashconv_per_agent), backgroundColor: accent } ] },
      options: { scales: { y: linAxis("", { beginAtZero: true }), x: { grid: { display: false } } }, plugins: { legend: { display: false }, tooltip: tooltipFmt } } });
    document.getElementById("sc-note").textContent = `Edges: ${rows.map((r) => fmt(r.edges)).join(" / ")}. Time per edge and iteration: ` +
      `${rows.map((r) => fmt(r.us_per_edge_iter, 2)).join(" / ")} µs (fitted slope ${fmt(D.scaling.slope_time_vs_edges, 3)} on edges). ` +
      "Single process on one laptop CPU; wall-clock timings vary by machine.";
  }

  // ------------------------------------------------------------------ topology
  function renderTopology() {
    const fams = [["ba", "Barabási–Albert"], ["sbm", "Stochastic block"], ["er", "Erdős–Rényi"], ["ws", "Watts–Strogatz"]];
    const pal = ["#1c7ed6", "#12b886", "#f59f00", "#ae3ec9"];
    const rows = D.topology.rows;
    makeChart("tp-chart", { type: "scatter", data: { datasets: fams.map(([f, name], i) => ({
      label: name, backgroundColor: pal[i], borderColor: pal[i], pointRadius: 7, pointHoverRadius: 9,
      data: rows.filter((r) => r.family === f).map((r) => ({ x: r.iterations, y: i })) })) },
      options: { parsing: false, scales: {
        x: linAxis("Iterations to reach NashConv per agent ≤ 10⁻³", { min: 2000, max: 5000 }),
        y: { min: -0.6, max: 3.6, reverse: true, grid: { display: false }, ticks: { stepSize: 1, callback: (v) => (fams[v] || [])[1] || "" } } },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.x.toLocaleString()} iterations` } } } } });
  }

  function renderFooter() {
    const ri = D.run_info || {};
    document.getElementById("foot-info").textContent =
      `Results generated ${ri.generated || ""} with Python ${ri.python || ""}. Instances are regenerated from fixed seeds.`;
  }

  let rt = null;
  addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => D && renderNetwork(), 150); });
})();
