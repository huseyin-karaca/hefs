import * as K from './parts.js';
// Hero: none, flat schematic of a pipeline plus a result plot; no physical object, route = flat.
// Notation follows the paper (Karaca et al., DSP 2026): §3, Algorithm 2, Eqs. (4) and (7).
export const figure = { width: 120, height: 100, name: 'forecast-hero', format: 'journal' };

// a hat over character i of a text element (the math font places U+0302 badly over α)
function hat(L, p, t, i, color, size) {
  const a = t.getStartPositionOfChar(i), b = t.getEndPositionOfChar(i);
  const cx = (a.x + b.x) / 2 + size * 0.07, y0 = a.y - size * 0.5, hw = size * 0.2, hh = size * 0.15;
  const h = L.path(p, `M${cx - hw},${y0} L${cx},${y0 - hh} L${cx + hw},${y0}`, { stroke: color, width: size * 0.065, fill: 'none', join: 'miter', cap: 'butt' });
  const tr = t.getAttribute('transform'); if (tr) (h.node || h).setAttribute('transform', tr);
}

export default function draw(F, L) {
  const p = F.root, C = K.COL;
  const AX = '#555b6b', lw = 0.6, CO = C.ink, COT = '#F3F3F6';   // cost optimization: neutral, training only
  const W = 340.16, xL = 3, xR = W - 3;           // shared left / right edges of every row

  // ---- 2 (layout first). base model | cost optimization | second-layer model ----
  const T = 11, py = 90 + T, ph = 68, gap = 23, mw = 94, lx = xL, pw = (xR - lx - mw - 2 * gap) / 2;
  const mxL = lx + pw + gap, rx = mxL + mw + gap;
  const lcx = lx + pw / 2, rcx = rx + pw / 2, mcx = mxL + mw / 2;

  // ---- 1. relevance scores: sorted, split at index p into two groups (Algorithm 2, lines 1-7) ----
  const sc = [0.95, 0.82, 0.64, 0.5, 0.3, 0.22, 0.15, 0.09];
  const pitch = 2 * (mcx - xL) / 8, bw = pitch * 0.62, base = 40 + T, hMax = 30;   // split sits on the centre column
  const bx = (i) => xL + i * pitch + (pitch - bw) / 2;
  sc.forEach((v, i) => {
    const dom = i < 4;
    if (i === 6) { for (let k = -1; k <= 1; k++) L.circle(p, bx(i) + bw / 2 + k * 5, base - 2.5, 0.9, { fill: C.mute }); return; }   // ... : there are d features, not eight
    L.rect(p, bx(i), base - v * hMax, bw, v * hMax, { fill: dom ? C.domT : C.nonT, stroke: dom ? C.dom : C.non, width: 0.7 });
    L.text(p, bx(i) + bw / 2, base + 3.5, i === 7 ? 'x_{d}' : `x_{${i + 1}}`, { anchor: 'middle', size: 6.5, color: C.mute, math: true, valign: 'top' });
  });
  L.line(p, [xL, base], [xL + 8 * pitch, base], { stroke: AX, width: lw });
  const xs = xL + 4 * pitch;
  L.line(p, [xs, 4 + T], [xs, 52 + T], { stroke: C.mute, width: 0.5, dash: '1.8 1.4' });
  L.text(p, xs + 5, 4 + T, 'Feature importance', { size: 6, color: C.mute, valign: 'top' });

  const gd = [bx(0), bx(3) + bw], gn = [bx(4), bx(7) + bw], by = 57 + T;
  const bracket = (g, c) => L.path(p, `M${g[0]},${by - 3} V${by} H${g[1]} V${by - 3}`, { stroke: c, width: 0.9, fill: 'none', join: 'round', cap: 'round' });
  bracket(gd, C.dom); bracket(gn, C.non);
  const cd = (gd[0] + gd[1]) / 2, cn = (gn[0] + gn[1]) / 2;
  L.text(p, cd, by + 4, 'Dominant features', { anchor: 'middle', size: 6.5, weight: 'bold', color: C.domTx, valign: 'top' });
  L.text(p, cn, by + 4, 'Non-dominant features', { anchor: 'middle', size: 6.5, weight: 'bold', color: C.nonTx, valign: 'top' });

  // ---- 2. the three blocks: model | compute correction targets (training only, raised) | model ----
  const box = (x, w, h, c, t, op, dash, y0 = py) => L.rect(p, x, y0, w, h, { r: 4, fill: t, fillOpacity: op, stroke: c, width: 0.7, dash });
  box(lx, pw, ph, C.dom, C.domT, 0.55); box(rx, pw, ph, C.non, C.nonT, 0.7);
  const cty = py - 3, cth = 46, ctb = cty + cth;        // construct box: top, height, bottom
  box(mxL, mw, cth, CO, COT, 1, '2.2 1.4', cty);
  const DASH = '2.2 1.4', yT = py + 34, yB = py + 55;    // supervision ports (yT) and the base-forecast line (yB)

  // text laid out piece by piece: [string, 'm' math | 't' text | 'b' bold, colour, hatIndex?]
  const seq = (x0, y, size, parts, anchor = 'start') => {
    const els = parts.map(([t, k, c, h]) => ({ h, c, el: L.text(p, 0, y, t.replace(/^ /, ' ').replace(/ $/, ' '), k === 'm' ? { size, math: true, color: c, valign: 'middle' } : { size, color: c, weight: k === 'b' ? 'bold' : 'normal', valign: 'middle' }) }));
    const ws = els.map((e) => e.el.getBBox().width), tot = ws.reduce((a, b) => a + b, 0);
    let x = anchor === 'end' ? x0 - tot : anchor === 'middle' ? x0 - tot / 2 : x0;
    els.forEach((e, i) => { e.el.setAttribute('transform', `translate(${x},0)`); if (e.h !== undefined) hat(L, p, e.el, e.h, e.c, size); x += ws[i]; });
  };

  // group inputs s(1), s(2): solid, entering the top edges
  L.arrow(p, [cd, 76 + T], [cd, py - 0.6], { stroke: C.dom, width: 1 });
  L.arrow(p, [cn, 76 + T], [cn, py - 0.6], { stroke: C.non, width: 1 });
  L.text(p, cd + 5, 82.5 + T, 's^{(1)}', { size: 8, math: true, color: C.domTx, valign: 'middle' });
  L.text(p, cn + 5, 82.5 + T, 's^{(2)}', { size: 8, math: true, color: C.nonTx, valign: 'middle' });

  // model boxes: title, the forward equation, the supervision port (inner edge), a small tree glyph, LightGBM
  const model = (cx, title, c, cg, eq, hatIdx, tgt, side) => {
    L.text(p, cx, py + 9, title, { anchor: 'middle', size: 6.5, weight: 'bold', color: c, valign: 'middle' });
    const mt = L.text(p, cx, py + 21, eq, { anchor: 'middle', size: 8.5, math: true, color: c, valign: 'middle' });
    if (hatIdx !== undefined) hat(L, p, mt, hatIdx, c, 8.5);
    const ex = side === 'r' ? cx + pw / 2 - 3.5 : cx - pw / 2 + 3.5;
    seq(ex, yT, 6, [['Target: ', 't', CO], ...tgt], side === 'r' ? 'end' : 'start');
    K.trees(L, p, cx - 5, py + 51, cg);
    L.text(p, cx, py + 63.5, 'LightGBM', { anchor: 'middle', size: 5.5, color: c, valign: 'middle' });
  };
  model(lcx, 'Base model', C.domTx, C.dom, 'ŷ_{1} = f_{1}(s^{(1)})', undefined, [['y', 'm', CO]], 'r');
  model(rcx, 'Second-layer model', C.nonTx, C.non, 'α = f_{2}(s^{(2)}, ŷ_{1})', 0, [['α^{\\*}', 'm', CO]], 'l');

  // compute correction targets (conceptual form of Eq. 4; the code searches a finite grid): training only
  L.text(p, mcx, cty + 8, 'Compute correction targets', { anchor: 'middle', size: 6.5, weight: 'bold', color: CO, valign: 'middle' });
  const ey = cty + 21, es = 7, mf = 'Cambria, "Times New Roman", serif';
  const parts = [['α^{\\*} = ', true], ['arg min', false], [' ℓ(y, (1 + a) ŷ_{1})', true]];
  const els = parts.map(([t, m]) => L.text(p, 0, ey, t, m ? { size: es, math: true, color: CO, valign: 'middle' } : { size: es, family: mf, color: CO, valign: 'middle' }));
  const wid = els.map((e) => e.getBBox().width), tot = wid.reduce((u, v) => u + v, 0);
  let ex0 = mcx - tot / 2;
  els.forEach((e, i) => { e.setAttribute('transform', `translate(${ex0},0)`); if (i === 1) L.text(p, ex0 + wid[1] / 2, ey + 6.5, 'a ∈ [−β, β]', { anchor: 'middle', size: 5.2, math: true, color: CO, valign: 'middle' }); ex0 += wid[i]; });
  L.text(p, mcx, ctb - 6, 'Training only', { anchor: 'middle', size: 5.8, weight: 'bold', color: CO, valign: 'middle' });

  // ---- dashed, training only: y, ŷ1 into the construction; y and α* out as targets ----
  const ds = { stroke: CO, width: 0.8, dash: DASH };
  L.arrow(p, [mcx, 82], [mcx, cty - 0.6], { ...ds, head: 2.6 });
  L.text(p, mcx + 4.5, 89, 'y', { size: 7.5, math: true, color: CO, valign: 'middle' });
  L.text(p, mcx + 11.5, 89, 'Observed value', { size: 6, color: C.mute, valign: 'middle' });
  const tx = lx + pw + 6;                                  // y also supervises the base model (entering its inner edge)
  L.path(p, `M${mcx},${cty - 7} H${tx} V${yT} H${lx + pw + 1.4}`, { ...ds, fill: 'none', join: 'miter' });
  L.chevron(p, [lx + pw + 0.6, yT], [-1, 0], { stroke: CO, width: 0.8, len: 2.4 });
  L.circle(p, mcx, cty - 7, 1.0, { fill: CO });
  const bx2 = mcx - 24;                                    // ŷ1 branches (dashed) into the construction box
  L.arrow(p, [bx2, yB], [bx2, ctb + 0.6], { ...ds, head: 2.6 });
  L.arrow(p, [mxL + mw + 0.6, yT], [rx - 0.8, yT], { ...ds, head: 2.6 });   // α* supervises the second model
  L.text(p, mxL + mw + gap / 2 + 0.3, yT - 2.4, 'α^{\\*}', { anchor: 'middle', size: 7.5, math: true, color: CO, valign: 'bottom' });

  // ---- solid, prediction flow: ŷ1 leaves the base model once and branches ----
  const jx = lx + pw + 15;
  L.line(p, [lx + pw, yB], [jx, yB], { stroke: C.dom, width: 0.9 });
  L.arrow(p, [jx, yB], [rx - 0.8, yB], { stroke: C.dom, width: 0.9, head: 2.8 });
  seq(mxL + mw / 2 + 12, yB + 6.5, 6.5, [['Base forecast ', 't', C.domTx], ['ŷ_{1}', 'm', C.domTx]], 'middle');

  // ---- 3. combination ŷ = (1 + α̂) ŷ1 ----
  const ny = py + ph + 14, nr = 4.6, nx = mcx;
  L.path(p, `M${rcx},${py + ph} V${ny - 3} Q${rcx},${ny} ${rcx - 3},${ny} H${nx + nr + 0.4}`, { stroke: C.non, width: 0.9, fill: 'none' });
  L.path(p, `M${jx},${yB} V${ny - 3} Q${jx},${ny} ${jx + 3},${ny} H${nx - nr - 0.4}`, { stroke: C.dom, width: 0.9, fill: 'none' });
  [jx, bx2].forEach((xx) => L.circle(p, xx, yB, 1.1, { fill: C.dom }));
  L.chevron(p, [nx - nr - 0.4, ny], [1, 0], { stroke: C.dom, width: 0.9, len: 2.4 });
  L.chevron(p, [nx + nr + 0.4, ny], [-1, 0], { stroke: C.non, width: 0.9, len: 2.4 });
  L.text(p, (jx + nx) / 2 - 2, ny - 2, 'ŷ_{1}', { anchor: 'middle', size: 7, math: true, color: C.domTx, valign: 'bottom' });
  hat(L, p, L.text(p, (rcx + nx) / 2, ny - 2, '1 + α', { anchor: 'middle', size: 7, math: true, color: C.nonTx, valign: 'bottom' }), 4, C.nonTx, 7);
  seq(rcx + 4, py + ph + 7.5, 6, [['Predicted', 't', C.nonTx]]);
  seq(rcx + 4, py + ph + 14, 6, [['correction ', 't', C.nonTx], ['α', 'm', C.nonTx, 0]]);
  L.circle(p, nx, ny, nr, { fill: '#fff', stroke: C.ink, width: 0.8 });
  const d = nr * 0.48;
  L.line(p, [nx - d, ny - d], [nx + d, ny + d], { stroke: C.ink, width: 0.8, cap: 'round' });
  L.line(p, [nx - d, ny + d], [nx + d, ny - d], { stroke: C.ink, width: 0.8, cap: 'round' });

  // arrow legend, beside the diagram (describes the diagram's arrows, not the plot's curves)
  const lgx = xL, lgy = ny - 8;
  L.text(p, lgx, lgy, 'Diagram arrows', { size: 5.5, weight: 'bold', color: C.mute, valign: 'middle' });
  L.arrow(p, [lgx, lgy + 8.5], [lgx + 17, lgy + 8.5], { stroke: AX, width: 0.9, head: 2.6 });
  L.text(p, lgx + 21, lgy + 8.5, 'Prediction flow', { size: 6, color: C.ink, valign: 'middle' });
  L.arrow(p, [lgx, lgy + 16.5], [lgx + 17, lgy + 16.5], { ...ds, stroke: CO, head: 2.6 });
  L.text(p, lgx + 21, lgy + 16.5, 'Training only', { size: 6, color: C.ink, valign: 'middle' });

  // ---- 4. forecast plot ----
  const fy = ny + 19, fh = 56, fw = xR - 3 - xL;
  L.arrow(p, [nx, ny + nr + 0.6], [nx, fy - 1], { stroke: C.ink, width: 0.9, head: 2.8 });
  seq(nx + 5, (ny + nr + fy) / 2 + 0.5, 6.5, [['Corrected forecast ', 't', C.ink], ['ŷ', 'm', C.ink]]);

  const M = K.frame(L, p, xL, fy, fw, fh, [0.25, 0.88]);
  K.errBand(L, p, M, K.base, C.err, 0.32);
  K.history(L, p, M); K.truth(L, p, M);
  L.polyline(p, K.pts(M, K.base, K.iSplit), { stroke: C.dom, width: 1.1, dash: '3 1.5', join: 'round' });
  L.polyline(p, K.pts(M, K.corrected, K.iSplit), { stroke: C.non, width: 1.35, join: 'round', cap: 'round' });

  // plot legend: one row, equal gaps, flush with the plot's edges; same symbols as the diagram
  const ly = fy + fh + 16, sw = 12;
  const items = [['Observed ', 'y', C.ink, null, 1.0, C.ink], ['Base forecast ', 'ŷ_{1}', C.dom, '3 1.5', 1.1, C.domTx], ['Corrected forecast ', 'ŷ', C.non, null, 1.35, C.nonTx], ['Base-forecast error', '', null, null, 0, C.ink]];
  const meas = items.map(([t, m]) => { const a = L.text(p, 0, -50, t, { size: 6 }); const b = m ? L.text(p, 0, -50, m, { size: 6, math: true }) : null; const w = a.getBBox().width + (b ? b.getBBox().width : 0); a.remove(); if (b) b.remove(); return w; });
  const widths = meas.map((w) => sw + 3.5 + w);
  const g = (fw + 3 - widths.reduce((a, b) => a + b, 0)) / (items.length - 1);
  let x = xL;
  items.forEach(([t, m, c, dsh, w, tc], i) => {
    if (c) L.line(p, [x, ly], [x + sw, ly], { stroke: c, width: w, dash: dsh });
    else L.rect(p, x, ly - 3.2, sw, 6.4, { fill: C.err, fillOpacity: 0.32, stroke: 'none' });
    seq(x + sw + 3.5, ly, 6, [[t, 't', tc], ...(m ? [[m, 'm', tc]] : [])]);
    x += widths[i] + g;
  });

  // what the data are
  seq(xL, 5, 6, [['Data: ', 'b', C.ink], ['each sample has d features ', 't', C.mute], ['x_{1} … x_{d}', 'm', C.ink], [' and an observed value ', 't', C.mute], ['y', 'm', C.ink], [' (to forecast)', 't', C.mute]]);
}
