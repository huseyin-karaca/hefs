import * as K from './parts.js';
// Hero: none, flat schematic of a pipeline plus a result plot; no physical object.
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
  const T = 11, py = 90 + T, ph = 58, gap = 20, mw = 92, lx = xL, pw = (xR - lx - mw - 2 * gap) / 2;
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

  // ---- 2. the three blocks ----
  const box = (x, w, h, c, t, op, dash) => L.rect(p, x, py, w, h, { r: 4, fill: t, fillOpacity: op, stroke: c, width: 0.7, dash });
  box(lx, pw, ph, C.dom, C.domT, 0.55); box(rx, pw, ph, C.non, C.nonT, 0.7);
  const mh = 36;                                      // short, so ŷ1 can pass beneath it to layer 2
  box(mxL, mw, mh, CO, COT, 1, '2.2 1.4');

  // group inputs s(1), s(2)
  L.arrow(p, [cd, 76 + T], [cd, py - 0.6], { stroke: C.dom, width: 1 });
  L.arrow(p, [cn, 76 + T], [cn, py - 0.6], { stroke: C.non, width: 1 });
  L.text(p, cd + 5, 82.5 + T, 's^{(1)}', { size: 8, math: true, color: C.domTx, valign: 'middle' });
  L.text(p, cn + 5, 82.5 + T, 's^{(2)}', { size: 8, math: true, color: C.nonTx, valign: 'middle' });

  // model boxes: title and tree glyph labelled LightGBM; no equations, the arrows carry the notation
  // each box states what it is fitted on: f : inputs → regression target (Algorithm 2, lines 11 and 16-17)
  const model = (cx, title, c, cg, map, hatIt) => {
    L.text(p, cx, py + 9, title, { anchor: 'middle', size: 6.5, weight: 'bold', color: c, valign: 'middle' });
    const mt = L.text(p, cx, py + 20, map, { anchor: 'middle', size: 8, math: true, color: c, valign: 'middle' });
    if (hatIt) hat(L, p, mt, mt.textContent.lastIndexOf('α'), c, 8);
    K.trees(L, p, cx - 5, py + 35, cg);
    L.text(p, cx, py + 48, 'LightGBM', { anchor: 'middle', size: 5.5, color: c, valign: 'middle' });
  };
  model(lcx, 'Base model', C.domTx, C.dom, 's^{(1)} → ŷ_{1}');
  model(rcx, 'Second-layer model', C.nonTx, C.non, '(s^{(2)}, ŷ_{1}) → α', true);

  // cost optimization (Eq. 4 in Algorithm 2's form, as in the code): per sample, the scale α in [−β, β]
  // that brings (1 + α)·ŷ1 closest to y under the final loss ℓ. Inputs y and ŷ1; output α, layer 2's target.
  L.text(p, mcx, py + 8.5, 'Cost optimization', { anchor: 'middle', size: 6.5, weight: 'bold', color: CO, valign: 'middle' });
  // α = arg min ℓ(y, (1 + α) ŷ1) with |α| ≤ β under the operator; laid out piece by piece, operator upright
  const ey = py + 18.5, es = 7, mf = 'Cambria, "Times New Roman", serif';
  const parts = [['α =\u2009', true], ['arg\u2009min', false], ['\u2009ℓ(y, (1 + α) ŷ_{1})', true]];
  const els = parts.map(([t, m]) => L.text(p, 0, ey, t, m ? { size: es, math: true, color: CO, valign: 'middle' } : { size: es, family: mf, color: CO, valign: 'middle' }));
  const wid = els.map((e) => e.getBBox().width), tot = wid.reduce((u, v) => u + v, 0);
  let ex = mcx - tot / 2;
  els.forEach((e, i) => { e.setAttribute('transform', `translate(${ex},0)`); if (i === 1) L.text(p, ex + wid[1] / 2, ey + 5.6, '|α| ≤ β', { anchor: 'middle', size: 5, math: true, color: CO, valign: 'middle' }); ex += wid[i]; });
  L.text(p, mcx, py + mh - 5, 'per sample, training only', { anchor: 'middle', size: 5.5, italic: true, color: C.mute, valign: 'middle' });

  // observed target y enters the search (Algorithm 2, line 14)
  L.arrow(p, [mcx, 77 + T], [mcx, py - 0.6], { stroke: CO, width: 0.8, head: 2.6, dash: '2.2 1.4' });
  L.text(p, mcx - 4.5, 82.5 + T, 'y', { anchor: 'end', size: 8, math: true, color: CO, valign: 'middle' });

  // ŷ1 into the search; α out as the regression target of layer 2 (training only, dashed)
  const ay = py + 18;
  L.arrow(p, [mxL + mw + 1.5, ay], [rx - 1, ay], { stroke: CO, width: 0.8, head: 2.8, dash: '2.2 1.4' });
  L.text(p, mxL + mw + gap / 2, ay - 2.5, 'α', { anchor: 'middle', size: 7, math: true, color: CO, valign: 'bottom' });
  // ŷ1 is also an input of layer 2 (Algorithm 2, line 16): it runs beneath the search block
  const by2 = py + 46;
  L.text(p, mxL + mw + gap / 2, by2 - 2.5, 'ŷ_{1}', { anchor: 'middle', size: 7, math: true, color: C.domTx, valign: 'bottom' });

  // ---- 3. final superposition ŷ = (1 + α̂) ŷ1  (Eq. 7) ----
  const ny = py + ph + 13, nr = 4.6, nx = mcx;
  const elbow = (x0, side, c) => L.path(p, `M${x0},${py + ph} V${ny - 3} Q${x0},${ny} ${x0 + side * 3},${ny} H${nx - side * (nr + 0.4)}`, { stroke: c, width: 0.9, fill: 'none' });
  elbow(rcx, -1, C.non);
  // ŷ1 leaves the base model once, at one point, then branches: to the cost search, to layer 2 and to the product
  const jx = lx + pw + gap / 2;
  L.line(p, [lx + pw, ay], [jx, ay], { stroke: C.dom, width: 0.9 });
  L.path(p, `M${jx},${ay} V${ny - 3} Q${jx},${ny} ${jx + 3},${ny} H${nx - nr - 0.4}`, { stroke: C.dom, width: 0.9, fill: 'none' });
  L.text(p, jx, ay - 2.5, 'ŷ_{1}', { anchor: 'middle', size: 7, math: true, color: C.domTx, valign: 'bottom' });
  L.arrow(p, [jx, ay], [mxL - 1, ay], { stroke: C.dom, width: 0.9, head: 2.8 });
  L.arrow(p, [jx, by2], [rx - 1, by2], { stroke: C.dom, width: 0.9, head: 2.8 });
  [ay, by2].forEach((yy) => L.circle(p, jx, yy, 1.1, { fill: C.dom }));
  L.chevron(p, [nx - nr - 0.4, ny], [1, 0], { stroke: C.dom, width: 0.9, len: 2.4 });
  L.chevron(p, [nx + nr + 0.4, ny], [-1, 0], { stroke: C.non, width: 0.9, len: 2.4 });
  L.text(p, (jx + nx) / 2, ny - 2, 'ŷ_{1}', { anchor: 'middle', size: 7, math: true, color: C.domTx, valign: 'bottom' });
  hat(L, p, L.text(p, (rcx + nx) / 2, ny - 2, '1 + α', { anchor: 'middle', size: 7, math: true, color: C.nonTx, valign: 'bottom' }), 4, C.nonTx, 7);
  L.circle(p, nx, ny, nr, { fill: '#fff', stroke: C.ink, width: 0.8 });
  const d = nr * 0.48;
  L.line(p, [nx - d, ny - d], [nx + d, ny + d], { stroke: C.ink, width: 0.8, cap: 'round' });
  L.line(p, [nx - d, ny + d], [nx + d, ny - d], { stroke: C.ink, width: 0.8, cap: 'round' });

  // ---- 4. forecast plot ----
  const fy = ny + 17, fh = 54, fw = xR - 3 - xL;
  L.arrow(p, [nx, ny + nr + 0.6], [nx, fy - 1], { stroke: C.ink, width: 0.9, head: 2.8 });
  L.text(p, nx + 5, (ny + nr + fy) / 2, 'ŷ', { size: 8, math: true, color: C.ink, valign: 'middle' });

  const M = K.frame(L, p, xL, fy, fw, fh, [0.25, 0.88]);
  K.errBand(L, p, M, K.base, C.err, 0.32);
  K.history(L, p, M); K.truth(L, p, M);
  L.polyline(p, K.pts(M, K.base, K.iSplit), { stroke: C.dom, width: 1.1, dash: '3 1.5', join: 'round' });
  L.polyline(p, K.pts(M, K.corrected, K.iSplit), { stroke: C.non, width: 1.35, join: 'round', cap: 'round' });

  // legend: one row, equal gaps, flush with the plot's edges
  const ly = fy + fh + 16, sw = 12;
  const items = [['Actual', C.ink, null, 1.0], ['Base forecast', C.dom, '3 1.5', 1.1], ['Corrected forecast', C.non, null, 1.35], ['Base-forecast error', null]];
  const TW = { 'Actual': 16.67, 'Base forecast': 36.65, 'Corrected forecast': 49.3, 'Base-forecast error': 51.37 };
  const tw = (s) => TW[s];                             // measured Arial 6 pt widths (out/manifest.json)
  const widths = items.map(([t]) => sw + 3.5 + tw(t));
  const g = (fw + 3 - widths.reduce((a, b) => a + b, 0)) / (items.length - 1);
  let x = xL;
  items.forEach(([t, c, dsh, w], i) => {
    if (c) L.line(p, [x, ly], [x + sw, ly], { stroke: c, width: w, dash: dsh });
    else L.rect(p, x, ly - 3.2, sw, 6.4, { fill: C.err, fillOpacity: 0.32, stroke: 'none' });
    L.text(p, x + sw + 3.5, ly, t, { valign: 'middle', size: 6, color: c === C.dom ? C.domTx : c === C.non ? C.nonTx : C.ink });
    x += widths[i] + g;
  });

  // one line of text built from pieces: [string, 'm' math | 't' text | 'b' bold, hat?]
  const rowParts = (x0, y, size, parts) => {
    let x = x0;
    parts.forEach(([t, k, h]) => {
      const o = k === 'm' ? { size, math: true, color: C.ink, valign: 'middle' } : { size, color: k === 'b' ? C.ink : C.mute, weight: k === 'b' ? 'bold' : 'normal', valign: 'middle' };
      const el = L.text(p, 0, y, t.replace(/^ /, '\u00A0').replace(/ $/, '\u00A0'), o);
      el.setAttribute('transform', `translate(${x},0)`);
      if (h) hat(L, p, el, 0, C.ink, size);
      x += el.getBBox().width;
    });
  };
  // what the data are (top) and what each symbol means (bottom)
  rowParts(xL, 5, 6, [['Data: ', 'b'], ['each sample has d features ', 't'], ['x_{1} … x_{d}', 'm'], [' and a target ', 't'], ['y', 'm'], [' (the value to forecast)', 't']]);
  rowParts(xL, ly + 10, 5.5, [['Key  ', 'b'], ['s^{(1)}', 'm'], [', ', 't'], ['s^{(2)}', 'm'], [': dominant, non-dominant features  ·  ', 't'], ['y', 'm'], [': target  ·  ', 't'], ['ŷ_{1}', 'm'], [': base forecast  ·  ', 't'], ['ŷ', 'm'], [': final forecast', 't']]);
  rowParts(xL, ly + 17, 5.5, [['α', 'm'], [': correction factor found in training  ·  ', 't'], ['α', 'm', true], [': correction predicted by layer 2  ·  ', 't'], ['ŷ', 'm'], [' = (1 + ', 'm'], ['α', 'm', true], [') ŷ_{1}', 'm']]);
}
