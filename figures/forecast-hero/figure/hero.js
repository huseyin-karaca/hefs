import * as K from './parts.js';
// Hero: none, flat schematic of a pipeline plus an illustrative plot; no physical object, route = flat.
// Notation follows the paper (Karaca et al., DSP 2026): §3, Algorithm 2, Eqs. (4) and (7); feature groups set as bold vectors.
// One source, two versions: { lite: true } is the web hero, { lite: false } adds the search bound and the error band.

const NB = ' ', EM = ' ';
// a hat over character i of a text element (the math font places U+0302 badly over α)
function hat(L, p, t, i, color, size) {
  const a = t.getStartPositionOfChar(i), b = t.getEndPositionOfChar(i);
  const cx = (a.x + b.x) / 2 + size * 0.07, y0 = a.y - size * 0.5, hw = size * 0.2, hh = size * 0.15;
  const h = L.path(p, `M${cx - hw},${y0} L${cx},${y0 - hh} L${cx + hw},${y0}`, { stroke: color, width: size * 0.065, fill: 'none', join: 'miter', cap: 'butt' });
  const tr = t.getAttribute('transform'); if (tr) (h.node || h).setAttribute('transform', tr);
}

// a five-point star as the superscript after character i (the font's own star glyph is too small at web size)
const STAR_GAP = ' ';
function star(L, p, t, i, color, size) {
  const b = t.getEndPositionOfChar(i), cx = b.x + size * 0.27, cy = b.y - size * 0.6, R = size * 0.19, r = R * 0.42;
  const pts = Array.from({ length: 10 }, (_, k) => { const a = -Math.PI / 2 + (k * Math.PI) / 5, rr = k % 2 ? r : R; return [cx + rr * Math.cos(a), cy + rr * Math.sin(a)]; });
  const s = L.polygon(p, pts, { fill: color, stroke: 'none' });
  const tr = t.getAttribute('transform'); if (tr) (s.node || s).setAttribute('transform', tr);
}

export function drawHero(F, L, o = {}) {
  const lite = !!o.lite;
  const p = F.root, C = K.COL;
  const AX = '#555b6b', CO = C.ink, COT = '#F3F3F6', DASH = '2.2 1.4';
  const ds = { stroke: CO, width: 0.8, dash: DASH };
  const W = 340.16, xL = 3, xR = W - 3;
  const mf = 'Cambria, "Times New Roman", serif';

  // text laid out piece by piece: [string, kind, colour, hatIndex?, starIndex?]
  // kinds: 'm' math italic, 'v' bold upright vector (\mathbf), 'r' upright roman (\text), 't' sans text, 'b' sans bold
  const sty = (k, size, c) => {
    if (k === 'm') return { size, math: true, color: c, valign: 'middle' };
    if (k === 'v') return { size, math: true, italic: false, color: c, valign: 'middle' };
    if (k === 'r') return { size, family: mf, color: c, valign: 'middle' };
    return { size, color: c, weight: k === 'b' ? 'bold' : 'normal', valign: 'middle' };
  };
  const nb = (t) => t.replace(/^ /, NB).replace(/ $/, NB);
  const meas = (size, parts) => parts.reduce((w, [t, k, , , , dx = 0]) => { const e = L.text(p, 0, -80, nb(t), sty(k, size, '#000')); const v = e.getBBox().width; e.remove(); return w + v + dx; }, 0);
  const seq = (x0, y, size, parts, anchor = 'start') => {
    const els = parts.map(([t, k, c, h, st, dx = 0]) => ({ h, st, c, dx, el: L.text(p, 0, y, nb(t), sty(k, size, c)) }));
    const ws = els.map((e) => e.el.getBBox().width + e.dx), tot = ws.reduce((a, b) => a + b, 0);
    let x = anchor === 'end' ? x0 - tot : anchor === 'middle' ? x0 - tot / 2 : x0;
    const left = x;
    els.forEach((e, i) => { x += e.dx; e.el.setAttribute('transform', `translate(${x},0)`); if (e.h !== undefined) hat(L, p, e.el, e.h, e.c, size); if (e.st !== undefined) star(L, p, e.el, e.st, e.c, size); x += ws[i] - e.dx; });
    return { left, right: x };
  };
  const S1 = '**s**^{(1)}', S2 = '**s**^{(2)}';           // feature groups as bold vectors, as in the user's notation

  // ---- the one-line model expressions: forward equation, then the training target, \quad apart ----
  const eqB = [['ŷ_{1} = f_{1}(', 'm', C.domTx], [S1, 'v', C.domTx, undefined, undefined, -0.11 * 8], ['),' + EM, 'm', C.domTx], ['Target: ', 'r', C.domTx], ['y', 'm', C.domTx]];
  const eqS = [['α = f_{2}([', 'm', C.nonTx, 0], [S2, 'v', C.nonTx, undefined, undefined, -0.05 * 8], ['; ŷ_{1}]),' + EM, 'm', C.nonTx], ['Target: ', 'r', C.nonTx], ['α' + STAR_GAP, 'm', C.nonTx, undefined, 0]];

  // ---- layout: base model | cost-optimization block | second-layer model; the middle box is as wide as it needs ----
  const es = 7, ctTitle = 'Cost-optimization block';
  const eparts = [['α' + STAR_GAP + ' = ', 'm'], ['arg min', 'r'], [' ℓ(y, (1 + a) ŷ_{1})', 'm']];
  const mw = Math.max(meas(6.5, [[ctTitle, 'b']]), meas(es, eparts)) + 10;
  const py = 92, ph = 70, gap = 16, lx = xL, pw = (xR - lx - mw - 2 * gap) / 2;
  const mxL = lx + pw + gap, rx = mxL + mw + gap;
  const lcx = lx + pw / 2, rcx = rx + pw / 2, mcx = mxL + mw / 2;
  const yA = py + 34, yB = py + 54;                       // α* supervision arrow (yA) and the base-forecast line (yB)
  let msz = 8.5;                                          // largest common size at which both lines fit their boxes
  while (msz > 6.5 && Math.max(meas(msz, eqB), meas(msz, eqS)) > pw - 11) msz -= 0.25;

  // ---- 1. relevance scores, sorted and split at index p (Algorithm 2, lines 1-7); p + q = d (Table 1) ----
  const sc = [0.95, 0.82, null, 0.55, 0.3, 0.22, null, 0.09];
  const lab = ['x_{1}', 'x_{2}', null, 'x_{p}', 'x_{p+1}', 'x_{p+2}', null, 'x_{d}'];
  const pitch = 2 * (mcx - xL) / 8, bw = pitch * 0.62, base = 38, hMax = 30;   // the split sits on the centre column
  const bx = (i) => xL + i * pitch + (pitch - bw) / 2;
  sc.forEach((v, i) => {
    const dom = i < 4, cx = bx(i) + bw / 2;
    if (v === null) { for (let k = -1; k <= 1; k++) L.circle(p, cx + k * 3.6, base + 8.2, 0.75, { fill: C.mute }); return; }   // ⋯ : more features
    L.rect(p, bx(i), base - v * hMax, bw, v * hMax, { fill: dom ? C.domT : C.nonT, stroke: dom ? C.dom : C.non, width: 0.7 });
    L.text(p, cx, base + 3.5, lab[i], { anchor: 'middle', size: 6.5, math: true, color: C.mute, valign: 'top' });
  });
  L.line(p, [xL, base], [xL + 8 * pitch, base], { stroke: AX, width: 0.6 });
  const xs = xL + 4 * pitch;
  L.line(p, [xs, 1], [xs, base + 13], { stroke: C.mute, width: 0.5, dash: '1.8 1.4' });
  L.text(p, xs + 5, 1.5, 'Feature importance', { size: 6, color: C.mute, valign: 'top' });

  const gd = [bx(0), bx(3) + bw], gn = [bx(4), bx(7) + bw], by = base + 20;
  const bracket = (g, c) => L.path(p, `M${g[0]},${by - 3} V${by} H${g[1]} V${by - 3}`, { stroke: c, width: 0.9, fill: 'none', join: 'round', cap: 'round' });
  bracket(gd, C.dom); bracket(gn, C.non);
  const cd = (gd[0] + gd[1]) / 2, cn = (gn[0] + gn[1]) / 2;
  L.text(p, cd, by + 4, 'Dominant features', { anchor: 'middle', size: 6.5, weight: 'bold', color: C.domTx, valign: 'top' });
  L.text(p, cn, by + 4, 'Non-dominant features', { anchor: 'middle', size: 6.5, weight: 'bold', color: C.nonTx, valign: 'top' });

  // group inputs s(1), s(2): solid, entering the top edges
  const sy0 = by + 15;
  L.arrow(p, [cd, sy0], [cd, py - 0.6], { stroke: C.dom, width: 1 });
  L.arrow(p, [cn, sy0], [cn, py - 0.6], { stroke: C.non, width: 1 });
  seq(cd + 5, (sy0 + py) / 2 + 1, 8, [[S1, 'v', C.domTx]]);
  seq(cn + 5, (sy0 + py) / 2 + 1, 8, [[S2, 'v', C.nonTx]]);

  // ---- 2. the three blocks ----
  const box = (x, w, h, c, t, op, dash, y0 = py) => L.rect(p, x, y0, w, h, { r: 4, fill: t, fillOpacity: op, stroke: c, width: 0.7, dash });
  box(lx, pw, ph, C.dom, C.domT, 0.55); box(rx, pw, ph, C.non, C.nonT, 0.7);
  const cty = py, ctb = py + 44;
  box(mxL, mw, ctb - cty, CO, COT, 1, DASH, cty);

  // model boxes: title, the one-line expression, the tree ensemble, LightGBM
  const model = (cx, title, c, cg, eq, n, rows, tcy) => {
    L.text(p, cx, py + 9, title, { anchor: 'middle', size: 6.5, weight: 'bold', color: c, valign: 'middle' });
    seq(cx, py + 21, msz, eq, 'middle');
    K.trees(L, p, cx, tcy, cg, { n, rows });
    L.text(p, cx, py + 63, 'LightGBM', { anchor: 'middle', size: 5.5, color: c, valign: 'middle' });
  };
  // the base model is drawn with a second, staggered row of trees: the larger of the two learners
  model(lcx, 'Base model', C.domTx, C.dom, eqB, 3, 2, py + 36);
  model(rcx, 'Second-layer model', C.nonTx, C.non, eqS, 2, 1, py + 42.5);

  // cost-optimization block (conceptual form of Eq. 4; the code searches a finite grid): training only
  L.text(p, mcx, cty + 8.5, ctTitle, { anchor: 'middle', size: 6.5, weight: 'bold', color: CO, valign: 'middle' });
  const ey = cty + 21;
  const els = eparts.map(([t, k]) => L.text(p, 0, ey, nb(t), sty(k, es, CO)));
  const wid = els.map((e) => e.getBBox().width), tot = wid.reduce((u, v) => u + v, 0);
  let ex0 = mcx - tot / 2;
  els.forEach((e, i) => {
    e.setAttribute('transform', `translate(${ex0},0)`);
    if (i === 0) star(L, p, e, 0, CO, es);
    if (i === 1) L.text(p, ex0 + wid[1] / 2, ey + (lite ? 4.6 : 5.4), lite ? 'a' : 'a ∈ [−β, β]', { anchor: 'middle', size: 5.2, math: true, color: CO, valign: 'middle' });
    ex0 += wid[i];
  });
  L.text(p, mcx, ctb - 6.5, 'Training only', { anchor: 'middle', size: 5.8, weight: 'bold', color: CO, valign: 'middle' });

  // ---- dashed, training only: ŷ1 into the block, α* out to the second model ----
  L.arrow(p, [mxL + mw + 0.6, yA], [rx - 0.8, yA], { ...ds, head: 2.6 });
  seq(mxL + mw + gap / 2, yA - 4.6, 7, [['α' + STAR_GAP, 'm', CO, undefined, 0]], 'middle');
  const bx2 = mcx - 14;
  // training-only branch of ŷ1: leaves the base line left of the block, rises to the α* height and enters the block's left edge
  const jx = mxL - gap / 2;                               // the single branching point of ŷ1
  const bj = jx, dsB = { ...ds, stroke: C.dom };
  L.line(p, [bj, yB], [bj, yA], dsB);
  L.arrow(p, [bj, yA], [mxL + 0.6, yA], { ...dsB, head: 2.6 });

  // ---- solid, prediction flow: ŷ1 leaves the base model once and branches ----
  L.line(p, [lx + pw, yB], [jx, yB], { stroke: C.dom, width: 0.9 });
  L.arrow(p, [jx, yB], [rx - 0.8, yB], { stroke: C.dom, width: 0.9, head: 2.8 });
  seq(jx + 3.5, yB + 6.5, 6.5, [['Base forecast ', 't', C.domTx], ['ŷ_{1}', 'm', C.domTx]]);   // beside the single branching point, shared by all three branches

  // ---- 3. combination ŷ = (1 + α̂) ŷ1 ----
  const ny = py + ph + 14, nr = 4.6, nx = mcx;
  L.path(p, `M${rcx},${py + ph} V${ny - 3} Q${rcx},${ny} ${rcx - 3},${ny} H${nx + nr + 0.4}`, { stroke: C.non, width: 0.9, fill: 'none' });
  L.path(p, `M${jx},${yB} V${ny - 3} Q${jx},${ny} ${jx + 3},${ny} H${nx - nr - 0.4}`, { stroke: C.dom, width: 0.9, fill: 'none' });
  [jx, bj].forEach((xx) => L.circle(p, xx, yB, 1.1, { fill: C.dom }));
  L.chevron(p, [nx - nr - 0.4, ny], [1, 0], { stroke: C.dom, width: 0.9, len: 2.4 });
  L.chevron(p, [nx + nr + 0.4, ny], [-1, 0], { stroke: C.non, width: 0.9, len: 2.4 });
  hat(L, p, L.text(p, (rcx + nx) / 2, ny - 2, '1 + α', { anchor: 'middle', size: 7, math: true, color: C.nonTx, valign: 'bottom' }), 4, C.nonTx, 7);
  seq(rcx + 4, py + ph + 6.5, 6, [['Predicted', 't', C.nonTx]]);
  seq(rcx + 4, py + ph + 13, 6, [['correction ', 't', C.nonTx], ['α', 'm', C.nonTx, 0]]);
  L.circle(p, nx, ny, nr, { fill: '#fff', stroke: C.ink, width: 0.8 });
  const d = nr * 0.48;
  L.line(p, [nx - d, ny - d], [nx + d, ny + d], { stroke: C.ink, width: 0.8, cap: 'round' });
  L.line(p, [nx - d, ny + d], [nx + d, ny - d], { stroke: C.ink, width: 0.8, cap: 'round' });

  // ---- 4. forecast plot ----
  const fy = ny + 20, fh = 56, fw = xR - 3 - xL;
  L.arrow(p, [nx, ny + nr + 0.6], [nx, fy - 1], { stroke: C.non, width: 0.9, head: 2.8 });
  seq(nx - 5, (ny + nr + fy) / 2 + 0.5, 6.5, [['Corrected forecast ', 't', C.nonTx], ['ŷ', 'm', C.nonTx]], 'end');

  const M = K.frame(L, p, xL, fy, fw, fh, [0.25, 0.9]);
  if (!lite) K.errBand(L, p, M, K.base, C.err, 0.32);
  K.history(L, p, M); K.truth(L, p, M);
  L.polyline(p, K.pts(M, K.base, K.iSplit), { stroke: C.dom, width: 1.1, dash: '3 1.5', join: 'round' });
  L.polyline(p, K.pts(M, K.corrected, K.iSplit), { stroke: C.non, width: 1.35, join: 'round', cap: 'round' });

  // ---- legend: row 1 the plot's curves (equal gaps, flush with the plot's edges), row 2 the diagram's arrows,
  //      on the same columns; arrows carry heads and are grey, so they do not read as curves ----
  const ly = fy + fh + 11, ly2 = ly + 9.5, sw = 12;
  const items = [['Observed ', 'y', C.ink, null, 1.0, C.ink], ['Base forecast ', 'ŷ_{1}', C.dom, '3 1.5', 1.1, C.domTx], ['Corrected forecast ', 'ŷ', C.non, null, 1.35, C.nonTx]];
  if (!lite) items.push(['Base-forecast error', '', null, null, 0, C.ink]);
  const widths = items.map(([t, m]) => sw + 3.5 + meas(6, [[t, 't'], ...(m ? [[m, 'm']] : [])]));
  const g = (fw + 3 - widths.reduce((a, b) => a + b, 0)) / (items.length - 1);
  const cols = []; let x = xL;
  items.forEach(([t, m, c, dsh, w, tc], i) => {
    cols.push(x);
    if (c) L.line(p, [x, ly], [x + sw, ly], { stroke: c, width: w, dash: dsh });
    else L.rect(p, x, ly - 3.2, sw, 6.4, { fill: C.err, fillOpacity: 0.32, stroke: 'none' });
    seq(x + sw + 3.5, ly, 6, [[t, 't', tc], ...(m ? [[m, 'm', tc]] : [])]);
    x += widths[i] + g;
  });
  // row 2 is centred on the figure, independent of row 1's columns
  const w1 = sw + 3.5 + meas(6, [['Prediction flow', 't']]), w2 = sw + 3.5 + meas(6, [['Training only', 't']]);
  const cc = cols.map((c0, i) => c0 + widths[i] / 2);       // row-1 item centres; row 2 sits midway between them
  const q0 = (cc[0] + cc[1]) / 2 - w1 / 2, q1 = (cc[1] + cc[2]) / 2 - w2 / 2;
  L.arrow(p, [q0, ly2], [q0 + sw, ly2], { stroke: AX, width: 0.9, head: 2.6 });
  L.text(p, q0 + sw + 3.5, ly2, 'Prediction flow', { size: 6, color: C.ink, valign: 'middle' });
  L.arrow(p, [q1, ly2], [q1 + sw, ly2], { ...ds, head: 2.6 });
  L.text(p, q1 + sw + 3.5, ly2, 'Training only', { size: 6, color: C.ink, valign: 'middle' });
}
