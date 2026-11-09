import * as K from './parts.js';
// Hero: none, flat schematic of a pipeline plus an illustrative plot; no physical object, route = flat.
// Notation follows the paper (Karaca et al., DSP 2026): §3, Algorithm 2, Eqs. (4) and (7).
// One source, two versions: { lite: true } is the web hero, { lite: false } adds the search bound and the error band.

const NB = ' ';
// a hat over character i of a text element (the math font places U+0302 badly over α)
function hat(L, p, t, i, color, size) {
  const a = t.getStartPositionOfChar(i), b = t.getEndPositionOfChar(i);
  const cx = (a.x + b.x) / 2 + size * 0.07, y0 = a.y - size * 0.5, hw = size * 0.2, hh = size * 0.15;
  const h = L.path(p, `M${cx - hw},${y0} L${cx},${y0 - hh} L${cx + hw},${y0}`, { stroke: color, width: size * 0.065, fill: 'none', join: 'miter', cap: 'butt' });
  const tr = t.getAttribute('transform'); if (tr) (h.node || h).setAttribute('transform', tr);
}

// a five-point star as the superscript after character i (the font's own star glyph is too small at web size)
const STAR_GAP = '\u2002';
function star(L, p, t, i, color, size) {
  const b = t.getEndPositionOfChar(i), cx = b.x + size * 0.27, cy = b.y - size * 0.6, R = size * 0.19, r = R * 0.42;
  const pts = Array.from({ length: 10 }, (_, k) => { const a = -Math.PI / 2 + (k * Math.PI) / 5, rr = k % 2 ? r : R; return [cx + rr * Math.cos(a), cy + rr * Math.sin(a)]; });
  const s = L.polygon(p, pts, { fill: color, stroke: 'none' });
  const tr = t.getAttribute('transform'); if (tr) (s.node || s).setAttribute('transform', tr);
}

export function drawHero(F, L, o = {}) {
  const lite = !!o.lite, legendAt = o.legend ?? 'top';
  const p = F.root, C = K.COL;
  const AX = '#555b6b', CO = C.ink, COT = '#F3F3F6', DASH = '2.2 1.4';
  const ds = { stroke: CO, width: 0.8, dash: DASH };
  const W = 340.16, xL = 3, xR = W - 3;

  // text laid out piece by piece: [string, 'm' math | 't' text | 'b' bold, colour, hatIndex?]
  const sty = (k, size, c) => (k === 'm' ? { size, math: true, color: c, valign: 'middle' } : { size, color: c, weight: k === 'b' ? 'bold' : 'normal', valign: 'middle' });
  const nb = (t) => t.replace(/^ /, NB).replace(/ $/, NB);
  const meas = (size, parts) => parts.reduce((w, [t, k]) => { const e = L.text(p, 0, -80, nb(t), sty(k, size, '#000')); const v = e.getBBox().width; e.remove(); return w + v; }, 0);
  const seq = (x0, y, size, parts, anchor = 'start') => {
    const els = parts.map(([t, k, c, h, st]) => ({ h, st, c, el: L.text(p, 0, y, nb(t), sty(k, size, c)) }));
    const ws = els.map((e) => e.el.getBBox().width), tot = ws.reduce((a, b) => a + b, 0);
    let x = anchor === 'end' ? x0 - tot : anchor === 'middle' ? x0 - tot / 2 : x0;
    const left = x;
    els.forEach((e, i) => { e.el.setAttribute('transform', `translate(${x},0)`); if (e.h !== undefined) hat(L, p, e.el, e.h, e.c, size); if (e.st !== undefined) star(L, p, e.el, e.st, e.c, size); x += ws[i]; });
    return { left, right: x };
  };
  // a training-only tag: light fill, dashed outline (same language as the target-construction box)
  const tag = (x, y, side, size, parts, r = 2.2) => {
    const w = meas(size, parts) + 7, h = 11, x0 = side === 'l' ? x : side === 'r' ? x - w : x - w / 2;
    L.rect(p, x0, y - h / 2, w, h, { r, fill: COT, stroke: CO, width: 0.6, dash: '1.6 1.1' });
    seq(x0 + w / 2, y + 0.9, size, parts, 'middle');               // optical centre: the text's x-height, not its em box
    return { l: x0, r: x0 + w, t: y - h / 2, b: y + h / 2 };
  };

  // ---- layout: base model | compute correction targets | second-layer model ----
  const py = 98, ph = 70, gap = 23, mw = 94, lx = xL, pw = (xR - lx - mw - 2 * gap) / 2;
  const mxL = lx + pw + gap, rx = mxL + mw + gap;
  const lcx = lx + pw / 2, rcx = rx + pw / 2, mcx = mxL + mw / 2;
  const yP = py + 36, yB = py + 54;                       // target ports (yP) and the base-forecast line (yB)

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
  const sy0 = by + 16;
  L.arrow(p, [cd, sy0], [cd, py - 0.6], { stroke: C.dom, width: 1 });
  L.arrow(p, [cn, sy0], [cn, py - 0.6], { stroke: C.non, width: 1 });
  L.text(p, cd + 5, (sy0 + py) / 2 + 1, 's^{(1)}', { size: 8, math: true, color: C.domTx, valign: 'middle' });
  L.text(p, cn + 5, (sy0 + py) / 2 + 1, 's^{(2)}', { size: 8, math: true, color: C.nonTx, valign: 'middle' });

  // ---- 2. the three blocks ----
  const box = (x, w, h, c, t, op, dash, y0 = py) => L.rect(p, x, y0, w, h, { r: 4, fill: t, fillOpacity: op, stroke: c, width: 0.7, dash });
  box(lx, pw, ph, C.dom, C.domT, 0.55); box(rx, pw, ph, C.non, C.nonT, 0.7);
  const cty = py, ctb = py + 44;
  box(mxL, mw, ctb - cty, CO, COT, 1, DASH, cty);

  // model boxes: title, forward equation, target port on the inner edge, tree ensemble on the outer side
  const model = (cx, title, c, cg, eq, hatIdx, tcx, rows, tcy) => {
    L.text(p, cx, py + 9, title, { anchor: 'middle', size: 6.5, weight: 'bold', color: c, valign: 'middle' });
    const mt = L.text(p, cx, py + 21, eq, { anchor: 'middle', size: 8.5, math: true, color: c, valign: 'middle' });
    if (hatIdx !== undefined) hat(L, p, mt, hatIdx, c, 8.5);
    K.trees(L, p, tcx, tcy, cg, rows);
    L.text(p, tcx + 5.5, py + 63, 'LightGBM', { anchor: 'middle', size: 5.5, color: c, valign: 'middle' });
  };
  // the base model is drawn with a second, staggered row of trees: the larger of the two learners
  model(lcx, 'Base model', C.domTx, C.dom, 'ŷ_{1} = f_{1}(s^{(1)})', undefined, lx + 28, 2, py + 36);
  model(rcx, 'Second-layer model', C.nonTx, C.non, 'α = f_{2}(s^{(2)}, ŷ_{1})', 0, rx + pw - 37, 1, py + 42.5);
  const pB = tag(lx + pw - 2.5, yP, 'r', 6, [['Target ', 't', CO], ['y', 'm', CO]]);
  const pS = tag(rx + 2.5, yP, 'l', 6, [['Target ', 't', CO], ['α' + STAR_GAP, 'm', CO, undefined, 0]]);

  // compute correction targets (conceptual form of Eq. 4; the code searches a finite grid): training only
  L.text(p, mcx, cty + 8.5, 'Compute correction targets', { anchor: 'middle', size: 6.5, weight: 'bold', color: CO, valign: 'middle' });
  const ey = cty + 21, es = 7, mf = 'Cambria, "Times New Roman", serif';
  const eparts = [['α' + STAR_GAP + ' = ', true], ['arg min', false], [' ℓ(y, (1 + a) ŷ_{1})', true]];
  const els = eparts.map(([t, m]) => L.text(p, 0, ey, nb(t), m ? { size: es, math: true, color: CO, valign: 'middle' } : { size: es, family: mf, color: CO, valign: 'middle' }));
  const wid = els.map((e) => e.getBBox().width), tot = wid.reduce((u, v) => u + v, 0);
  let ex0 = mcx - tot / 2;
  els.forEach((e, i) => {
    e.setAttribute('transform', `translate(${ex0},0)`);
    if (i === 0) star(L, p, e, 0, CO, es);
    if (i === 1) L.text(p, ex0 + wid[1] / 2, ey + (lite ? 4.6 : 5.4), lite ? 'a' : 'a ∈ [−β, β]', { anchor: 'middle', size: 5.2, math: true, color: CO, valign: 'middle' });
    ex0 += wid[i];
  });
  L.text(p, mcx, ctb - 6.5, 'Training only', { anchor: 'middle', size: 5.8, weight: 'bold', color: CO, valign: 'middle' });

  // ---- dashed, training only: the observed value y feeds the construction and the base model's target ----
  const pill = tag(mcx, 81, 'c', 6, [['Observed ', 't', CO], ['y', 'm', CO]], 5.25);
  L.arrow(p, [mcx, pill.b + 0.3], [mcx, cty - 0.6], { ...ds, head: 2.6 });
  const tx = lx + pw + 9;
  L.path(p, `M${pill.l - 0.3},81 H${tx} V${yP} H${pB.r + 1.4}`, { ...ds, fill: 'none', join: 'miter' });
  L.chevron(p, [pB.r + 0.5, yP], [-1, 0], { stroke: CO, width: 0.8, len: 2.4 });
  L.arrow(p, [mxL + mw + 0.6, yP], [pS.l - 0.5, yP], { ...ds, head: 2.6 });            // α* supervises the second model
  const bx2 = mcx - 14;                                                                   // ŷ1 branches (dashed) into the construction
  L.arrow(p, [bx2, yB], [bx2, ctb + 0.6], { ...ds, head: 2.6 });

  // ---- solid, prediction flow: ŷ1 leaves the base model once and branches ----
  const jx = mxL + 9;
  L.line(p, [lx + pw, yB], [jx, yB], { stroke: C.dom, width: 0.9 });
  L.arrow(p, [jx, yB], [rx - 0.8, yB], { stroke: C.dom, width: 0.9, head: 2.8 });
  seq((bx2 + rx) / 2 + 2, yB + 6.5, 6.5, [['Base forecast ', 't', C.domTx], ['ŷ_{1}', 'm', C.domTx]], 'middle');

  // ---- 3. combination ŷ = (1 + α̂) ŷ1 ----
  const ny = py + ph + 14, nr = 4.6, nx = mcx;
  L.path(p, `M${rcx},${py + ph} V${ny - 3} Q${rcx},${ny} ${rcx - 3},${ny} H${nx + nr + 0.4}`, { stroke: C.non, width: 0.9, fill: 'none' });
  L.path(p, `M${jx},${yB} V${ny - 3} Q${jx},${ny} ${jx + 3},${ny} H${nx - nr - 0.4}`, { stroke: C.dom, width: 0.9, fill: 'none' });
  [jx, bx2].forEach((xx) => L.circle(p, xx, yB, 1.1, { fill: C.dom }));
  L.chevron(p, [nx - nr - 0.4, ny], [1, 0], { stroke: C.dom, width: 0.9, len: 2.4 });
  L.chevron(p, [nx + nr + 0.4, ny], [-1, 0], { stroke: C.non, width: 0.9, len: 2.4 });
  L.text(p, (jx + nx) / 2, ny - 2, 'ŷ_{1}', { anchor: 'middle', size: 7, math: true, color: C.domTx, valign: 'bottom' });
  hat(L, p, L.text(p, (rcx + nx) / 2, ny - 2, '1 + α', { anchor: 'middle', size: 7, math: true, color: C.nonTx, valign: 'bottom' }), 4, C.nonTx, 7);
  seq(rcx + 4, py + ph + 6.5, 6, [['Predicted', 't', C.nonTx]]);
  seq(rcx + 4, py + ph + 13, 6, [['correction ', 't', C.nonTx], ['α', 'm', C.nonTx, 0]]);
  L.circle(p, nx, ny, nr, { fill: '#fff', stroke: C.ink, width: 0.8 });
  const d = nr * 0.48;
  L.line(p, [nx - d, ny - d], [nx + d, ny + d], { stroke: C.ink, width: 0.8, cap: 'round' });
  L.line(p, [nx - d, ny + d], [nx + d, ny - d], { stroke: C.ink, width: 0.8, cap: 'round' });

  // arrow legend (describes the diagram's arrows, not the plot's curves)
  const legend = (x, y, title) => {
    let r = y;
    if (title) { L.text(p, x, y, 'Diagram arrows', { size: 5.5, weight: 'bold', color: C.mute, valign: 'middle' }); r += 8.5; }
    L.arrow(p, [x, r], [x + 15, r], { stroke: AX, width: 0.9, head: 2.6 });
    L.text(p, x + 19, r, 'Prediction flow', { size: 6, color: C.ink, valign: 'middle' });
    L.arrow(p, [x, r + 8.5], [x + 15, r + 8.5], { ...ds, head: 2.6 });
    L.text(p, x + 19, r + 8.5, 'Training only', { size: 6, color: C.ink, valign: 'middle' });
  };
  if (legendAt === 'top') legend(xR - 19 - meas(6, [['Prediction flow', 't']]), 4.5, true);
  else legend(xL, ny - 8, true);

  // ---- 4. forecast plot ----
  const fy = ny + 20, fh = 54, fw = xR - 3 - xL;
  L.arrow(p, [nx, ny + nr + 0.6], [nx, fy - 1], { stroke: C.non, width: 0.9, head: 2.8 });
  seq(nx - 5, (ny + nr + fy) / 2 + 0.5, 6.5, [['Corrected forecast ', 't', C.nonTx], ['ŷ', 'm', C.nonTx]], 'end');

  const M = K.frame(L, p, xL, fy, fw, fh, [0.25, 0.9]);
  if (!lite) K.errBand(L, p, M, K.base, C.err, 0.32);
  K.history(L, p, M); K.truth(L, p, M);
  L.polyline(p, K.pts(M, K.base, K.iSplit), { stroke: C.dom, width: 1.1, dash: '3 1.5', join: 'round' });
  L.polyline(p, K.pts(M, K.corrected, K.iSplit), { stroke: C.non, width: 1.35, join: 'round', cap: 'round' });

  // plot legend: one row, equal gaps, flush with the plot's edges; same symbols as the diagram
  const ly = fy + fh + 15, sw = 12;
  const items = [['Observed ', 'y', C.ink, null, 1.0, C.ink], ['Base forecast ', 'ŷ_{1}', C.dom, '3 1.5', 1.1, C.domTx], ['Corrected forecast ', 'ŷ', C.non, null, 1.35, C.nonTx]];
  if (!lite) items.push(['Base-forecast error', '', null, null, 0, C.ink]);
  const widths = items.map(([t, m]) => sw + 3.5 + meas(6, [[t, 't'], ...(m ? [[m, 'm']] : [])]));
  const g = (fw + 3 - widths.reduce((a, b) => a + b, 0)) / (items.length - 1);
  let x = xL;
  items.forEach(([t, m, c, dsh, w, tc], i) => {
    if (c) L.line(p, [x, ly], [x + sw, ly], { stroke: c, width: w, dash: dsh });
    else L.rect(p, x, ly - 3.2, sw, 6.4, { fill: C.err, fillOpacity: 0.32, stroke: 'none' });
    seq(x + sw + 3.5, ly, 6, [[t, 't', tc], ...(m ? [[m, 'm', tc]] : [])]);
    x += widths[i] + g;
  });
}
