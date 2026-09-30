/* NetCFR simulation engine (browser + Node). Line-for-line port of netcfr/game.py and
   netcfr/solvers.py for i.i.d. uniform cards, no public card; parity is checked by
   tests/test_sim_js.py against the Python implementation. */
(function (root) {
  "use strict";

  // numpy.round rounds half to even; Math.round rounds half up. Match numpy.
  function roundHalfEven(v) {
    const r = Math.round(v);
    return Math.abs(v - Math.trunc(v)) === 0.5 ? 2 * Math.round(v / 2) : r;
  }

  // small, fast, seedable PRNG (mulberry32)
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class Game {
    constructor(spec) {
      this.n = spec.n; this.M = spec.M;
      const E = spec.edges.length; this.E = E; const D = 2 * E; this.D = D;
      this.src = new Int32Array(D); this.dst = new Int32Array(D); this.rev = new Int32Array(D);
      this.b = new Float64Array(D); this.w = new Float64Array(D);
      spec.edges.forEach(([i, j, b, w], e) => {
        this.src[e] = i; this.dst[e] = j; this.rev[e] = E + e;
        this.src[E + e] = j; this.dst[E + e] = i; this.rev[E + e] = e;
        this.b[e] = this.b[E + e] = b; this.w[e] = this.w[E + e] = w;
      });
      const M = this.M; this.S = new Float64Array(M * M);
      for (let o = 0; o < M; o++) for (let p = 0; p < M; p++) this.S[o * M + p] = Math.sign(o - p) / (M * M);
      this.P = 1 / (M * M);
      // node -> owned directed edges (for summation in the same order as a CSR incidence matrix)
      this.own = Array.from({ length: this.n }, () => []);
      for (let d = 0; d < D; d++) this.own[this.src[d]].push(d);
      this.cB = new Float64Array(D * M); this.cC0 = new Float64Array(D * M);
      this.cL = new Float64Array(D * M); this.cF = new Float64Array(D * M);
      this.Bv = new Float64Array(this.n * M); this.Cv = new Float64Array(this.n * M); this.Cbr = new Float64Array(this.n * M);
    }

    // partner messages: pb[d*M+o] = beta_j(o), pg[d*M+o] = gamma_{j, rev(d)}(o)
    truePartner(beta, gamma, pb, pg) {
      const M = this.M;
      for (let d = 0; d < this.D; d++) {
        const j = this.dst[d], r = this.rev[d];
        for (let o = 0; o < M; o++) { pb[d * M + o] = beta[j * M + o]; pg[d * M + o] = gamma[r * M + o]; }
      }
    }

    values(gamma, pb, pg) {
      const M = this.M, S = this.S, P = this.P;
      for (let d = 0; d < this.D; d++) {
        const w = this.w[d], b1 = 1 + this.b[d], k = d * M;
        let sumPnb = 0;
        for (let p = 0; p < M; p++) sumPnb += (1 - pb[k + p]) * (1 - pg[k + p]);
        let sumPb = 0;
        for (let p = 0; p < M; p++) sumPb += pb[k + p];
        for (let o = 0; o < M; o++) {
          let sb = 0, snb = 0, snbg = 0;
          for (let p = 0; p < M; p++) {
            const s = S[o * M + p], bj = pb[k + p];
            sb += s * bj; snb += s * (1 - bj); snbg += s * (1 - bj) * pg[k + p];
          }
          this.cB[k + o] = w * (b1 * sb + P * sumPnb + b1 * snbg);
          this.cC0[k + o] = w * snb;
          this.cL[k + o] = w * b1 * sb;
          this.cF[k + o] = -w * P * sumPb;
        }
      }
      this.Bv.fill(0); this.Cv.fill(0); this.Cbr.fill(0);
      for (let i = 0; i < this.n; i++) {
        for (const d of this.own[i]) {
          const k = d * M;
          for (let o = 0; o < M; o++) {
            const g = gamma[k + o];
            this.Bv[i * M + o] += this.cB[k + o];
            this.Cv[i * M + o] += this.cC0[k + o] + g * this.cL[k + o] + (1 - g) * this.cF[k + o];
            this.Cbr[i * M + o] += this.cC0[k + o] + Math.max(this.cL[k + o], this.cF[k + o]);
          }
        }
      }
    }

    // per-agent exploitability gaps (best-response value - value) of a profile
    gaps(beta, gamma, out, pb, pg) {
      this.truePartner(beta, gamma, pb, pg);
      this.values(gamma, pb, pg);
      const M = this.M;
      let tot = 0;
      for (let i = 0; i < this.n; i++) {
        let u = 0, br = 0;
        for (let o = 0; o < M; o++) {
          const k = i * M + o, B = this.Bv[k], C = this.Cv[k];
          u += beta[k] * B + (1 - beta[k]) * C;
          br += Math.max(B, this.Cbr[k]);
        }
        out[i] = br - u; tot += br - u;
      }
      return tot;
    }
  }

  const rm = (a, b) => { const ap = Math.max(a, 0), bp = Math.max(b, 0), s = ap + bp; return s > 0 ? ap / s : 0.5; };

  // channel: "exact" | "det" | "sto" | "ef"
  class Run {
    constructor(game, { variant = "cfr+", channel = "exact", q = 1, seed = 0 } = {}) {
      Object.assign(this, { g: game, variant, channel, q, L: 2 ** q - 1, t: 0 });
      const n = game.n, M = game.M, D = game.D;
      this.RB = new Float64Array(n * M); this.RC = new Float64Array(n * M);
      this.RL = new Float64Array(D * M); this.RF = new Float64Array(D * M);
      this.beta = new Float64Array(n * M); this.gamma = new Float64Array(D * M);
      this.sB = new Float64Array(n * M); this.sW = 0; this.sG = new Float64Array(D * M); this.sGW = new Float64Array(D * M);
      this.pb = new Float64Array(D * M); this.pg = new Float64Array(D * M);
      this.eb = new Float64Array(D * M); this.eg = new Float64Array(D * M);
      this.rand = rng(seed * 2654435761 + 12345);
      this.avgB = new Float64Array(n * M); this.avgG = new Float64Array(D * M);
      this.gap = new Float64Array(n);
      this.bits = 2 * game.E * 2 * M * q;
    }

    quantize(x, i, e) {
      const L = this.L;
      if (this.channel === "det") return roundHalfEven(x * L) / L;
      if (this.channel === "sto") return Math.floor(x * L + this.rand()) / L;
      const v = x + e[i], qv = Math.min(1, Math.max(0, roundHalfEven(v * L) / L));  // error feedback
      e[i] = v - qv;
      return qv;
    }

    step() {
      const g = this.g, M = g.M, n = g.n, D = g.D;
      this.t += 1; const t = this.t;
      for (let k = 0; k < n * M; k++) this.beta[k] = rm(this.RB[k], this.RC[k]);
      for (let k = 0; k < D * M; k++) this.gamma[k] = rm(this.RL[k], this.RF[k]);
      g.truePartner(this.beta, this.gamma, this.pb, this.pg);
      if (this.channel !== "exact") {
        for (let k = 0; k < D * M; k++) {
          this.pb[k] = this.quantize(this.pb[k], k, this.eb);
          this.pg[k] = this.quantize(this.pg[k], k, this.eg);
        }
      }
      g.values(this.gamma, this.pb, this.pg);
      const plus = this.variant === "cfr+";
      for (let k = 0; k < n * M; k++) {
        const b = this.beta[k], vs = b * g.Bv[k] + (1 - b) * g.Cv[k];
        const rB = this.RB[k] + g.Bv[k] - vs, rC = this.RC[k] + g.Cv[k] - vs;
        this.RB[k] = plus ? Math.max(rB, 0) : rB; this.RC[k] = plus ? Math.max(rC, 0) : rC;
      }
      for (let k = 0; k < D * M; k++) {
        const c = this.gamma[k], resp = c * g.cL[k] + (1 - c) * g.cF[k];
        const rL = this.RL[k] + g.cL[k] - resp, rF = this.RF[k] + g.cF[k] - resp;
        this.RL[k] = plus ? Math.max(rL, 0) : rL; this.RF[k] = plus ? Math.max(rF, 0) : rF;
      }
      const w = plus ? t : 1;
      for (let k = 0; k < n * M; k++) this.sB[k] += w * this.beta[k];
      this.sW += w;
      for (let d = 0; d < D; d++) {
        const i = g.src[d];
        for (let o = 0; o < M; o++) {
          const k = d * M + o, reach = 1 - this.beta[i * M + o];
          this.sG[k] += w * reach * this.gamma[k]; this.sGW[k] += w * reach;
        }
      }
    }

    // total NashConv of the average profile (fills this.gap with per-agent gaps)
    evaluate() {
      for (let k = 0; k < this.sB.length; k++) this.avgB[k] = this.sB[k] / this.sW;
      for (let k = 0; k < this.sG.length; k++) this.avgG[k] = this.sGW[k] > 1e-300 ? this.sG[k] / this.sGW[k] : 0.5;
      return this.g.gaps(this.avgB, this.avgG, this.gap, this.pb, this.pg);
    }
  }

  const api = { Game, Run, roundHalfEven, rng };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.NetSim = api;
})(typeof self !== "undefined" ? self : this);
