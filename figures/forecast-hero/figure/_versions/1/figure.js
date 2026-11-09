import * as K from './parts.js';
// Hero: none, flat schematic of a pipeline plus a result plot; no physical object.
// Notation follows the paper (Karaca et al., DSP 2026): §3, Algorithm 2, Eqs. (4) and (7).
export const figure = { width: 120, height: 90, name: 'forecast-hero', format: 'journal' };

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
  const W = 340.16, xL = 14, xR = W - 4;            // shared left / right edges of every row

  // ---- 2 (layout first). base model | cost optimization | second-layer model ----
  const py = 90, ph = 58, gap = 22, mw = 78, lx = 2, pw = (xR - lx - mw - 2 * gap) / 2;
  const mxL = lx + pw + gap, rx = mxL + mw + gap;
  const lcx = lx + pw / 2, rcx = rx + pw / 2, mcx = mxL + mw / 2;

  // ---- 1. relevance scores: sorted, split at index p into two groups (Algorithm 2, lines 1-7) ----
  const sc = [0.95, 0.82, 0.64, 0.5, 0.3, 0.22, 0.15, 0.09];
  const pitch = 2 * (mcx - xL) / 8, bw = pitch * 0.62, base = 40, hMax = 30;   // split sits on the centre column
  const bx = (i) => xL + i * pitch + (pitch - bw) / 2;
  sc.forEach((v, i) => {
    const dom = i < 4;
    L.rect(p, bx(i), base - v * hMax, bw, v * hMax, { fill: dom ? C.domT : C.nonT, stroke: dom ? C.dom : C.non, width: 0.7 });
    L.text(p, bx(i) + bw / 2, base + 3.5, `x_{${i + 1}}`, { anchor: 'middle', size: 6.5, color: C.mute, math: true, valign: 'top' });
  });
  L.line(p, [xL, base], [xL + 8 * pitch, base], { stroke: AX, width: lw });
  const xs = xL + 4 * pitch;
  L.line(p, [xs, 4], [xs, 52], { stroke: C.mute, width: 0.5, dash: '1.8 1.4' });
  L.text(p, xs + 5, 4, 'Relevance score', { size: 6, color: C.mute, valign: 'top' });

  const gd = [bx(0), bx(3) + bw], gn = [bx(4), bx(7) + bw], by = 57;
  const bracket = (g, c) => L.path(p, `M${g[0]},${by - 3} V${by} H${g[1]} V${by - 3}`, { stroke: c, width: 0.9, fill: 'none', join: 'round', cap: 'round' });
  bracket(gd, C.dom); bracket(gn, C.non);
  const cd = (gd[0] + gd[1]) / 2, cn = (gn[0] + gn[1]) / 2;
  L.text(p, cd, by + 4, 'Dominant features', { anchor: 'middle', size: 6.5, weight: 'bold', color: C.domTx, valign: 'top' });
  L.text(p, cn, by + 4, 'Non-dominant features', { anchor: 'middle', size: 6.5, weight: 'bold', color: C.nonTx, valign: 'top' });

  // ---- 2. the three blocks ----
  const box = (x, w, h, c, t, op, dash) => L.rect(p, x, py, w, h, { r: 4, fill: t, fillOpacity: op, stroke: c, width: 0.7, dash });
  box(lx, pw, ph, C.dom, C.domT, 0.55); box(rx, pw, ph, C.non, C.nonT, 0.7);
  const mh = ph - 10;                                  // shorter, so ŷ1 can pass beneath it to layer 2
  box(mxL, mw, mh, CO, COT, 1, '2.2 1.4');

  // group inputs s(1), s(2)
  L.arrow(p, [cd, 76], [cd, py - 0.6], { stroke: C.dom, width: 1 });
  L.arrow(p, [cn, 76], [cn, py - 0.6], { stroke: C.non, width: 1 });
  L.text(p, cd + 5, 82.5, 's^{(1)}', { size: 8, math: true, color: C.domTx, valign: 'middle' });
  L.text(p, cn + 5, 82.5, 's^{(2)}', { size: 8, math: true, color: C.nonTx, valign: 'middle' });

  // model boxes: title, mapping (on the row where the arrows enter and leave), tree glyph labelled LightGBM
  const ay = py + 23;                                  // the arrow row: ŷ1 out, α in
  const model = (cx, title, c, cg, eq) => {
    L.text(p, cx, py + 8.5, title, { anchor: 'middle', size: 6.5, weight: 'bold', color: c, valign: 'middle' });
    const t = L.text(p, cx, ay, eq, { anchor: 'middle', size: 8, math: true, color: c, valign: 'middle' });
    K.trees(L, p, cx - 5, py + 37, cg);
    L.text(p, cx, py + 49.5, 'LightGBM', { anchor: 'middle', size: 5.5, color: c, valign: 'middle' });
    return t;
  };
  model(lcx, 'Base model', C.domTx, C.dom, 'ŷ_{1} = f_{1}(s^{(1)})');
  // layer 2 regresses the searched α (Algorithm 2, line 17): its prediction α̂ approximates α
  hat(L, p, model(rcx, 'Second-layer model', C.nonTx, C.non, `α = f_{2}(s^{(2)}, ŷ_{1}) {${CO}|≈ α}`), 0, C.nonTx, 8);

  // cost optimization: per-sample grid search of α in [-β, β] for the lowest loss ℓ(y, (1 + α) ŷ1)  (Eq. 4)
  L.text(p, mcx, py + 8.5, 'Cost optimization', { anchor: 'middle', size: 6.5, weight: 'bold', color: CO, valign: 'middle' });
  const g0 = mxL + 13, g1 = mxL + mw - 11, gb = py + 30, gt = py + 14.5, aStar = 0.3;   // α* in units of β
  const GX = (a) => g0 + (a + 1) / 2 * (g1 - g0), GY = (a) => gb - 1.5 - (gb - gt - 1.5) * Math.abs(a - aStar) / (1 + aStar);
  L.line(p, [g0 - 2, gb], [g1 + 2, gb], { stroke: AX, width: 0.5 });
  [-1, 1].forEach((a) => L.line(p, [GX(a), gb], [GX(a), gb + 1.6], { stroke: AX, width: 0.5 }));
  L.polyline(p, [[GX(-1), GY(-1)], [GX(aStar), GY(aStar)], [GX(1), GY(1)]], { stroke: C.mute, width: 0.6, join: 'round' });
  for (let k = 0; k <= 12; k++) {
    const a = -1 + k / 6;
    if (Math.abs(a - aStar) < 0.09) continue;
    L.circle(p, GX(a), GY(a), 0.75, { fill: '#fff', stroke: C.mute, width: 0.45 });
  }
  L.line(p, [GX(aStar), GY(aStar)], [GX(aStar), gb], { stroke: CO, width: 0.45, dash: '0.8 0.8' });
  L.circle(p, GX(aStar), GY(aStar), 1.35, { fill: CO });
  L.text(p, g0 - 3, gt + 1, 'ℓ', { anchor: 'end', size: 6.5, math: true, color: C.mute, valign: 'middle' });
  const tk = { anchor: 'middle', size: 5.5, math: true, color: C.mute, valign: 'top' };
  L.text(p, GX(-1), gb + 2.2, '−β', tk); L.text(p, GX(1), gb + 2.2, 'β', tk);
  L.text(p, GX(aStar), gb + 2.2, 'α', { ...tk, color: CO });
  L.text(p, mcx, py + mh - 3.8, 'training only', { anchor: 'middle', size: 5.5, italic: true, color: C.mute, valign: 'middle' });

  // observed target y enters the search (Algorithm 2, line 14)
  L.arrow(p, [mcx, 77], [mcx, py - 0.6], { stroke: CO, width: 0.8, head: 2.6, dash: '2.2 1.4' });
  L.text(p, mcx - 4.5, 82.5, 'y', { anchor: 'end', size: 8, math: true, color: CO, valign: 'middle' });

  // ŷ1 into the search; α out as the regression target of layer 2 (training only, dashed)
  L.arrow(p, [lx + pw + 1.5, ay], [mxL - 1, ay], { stroke: C.dom, width: 0.9, head: 2.8 });
  L.text(p, lx + pw + gap / 2, ay - 2.5, 'ŷ_{1}', { anchor: 'middle', size: 7, math: true, color: C.domTx, valign: 'bottom' });
  L.arrow(p, [mxL + mw + 1.5, ay], [rx - 1, ay], { stroke: CO, width: 0.8, head: 2.8, dash: '2.2 1.4' });
  L.text(p, mxL + mw + gap / 2, ay - 2.5, 'α', { anchor: 'middle', size: 7, math: true, color: CO, valign: 'bottom' });
  L.text(p, mxL + mw + gap / 2, ay + 2.5, 'target', { anchor: 'middle', size: 5.5, italic: true, color: C.mute, valign: 'top' });
  // ŷ1 is also an input of layer 2 (Algorithm 2, line 16): it runs beneath the search block
  const by2 = py + ph - 4.5;
  L.arrow(p, [lx + pw + 1.5, by2], [rx - 1, by2], { stroke: C.dom, width: 0.9, head: 2.8 });

  // ---- 3. final superposition ŷ = (1 + α̂) ŷ1  (Eq. 7) ----
  const ny = py + ph + 13, nr = 4.6, nx = mcx;
  const elbow = (x0, side, c) => L.path(p, `M${x0},${py + ph} V${ny - 3} Q${x0},${ny} ${x0 + side * 3},${ny} H${nx - side * (nr + 0.4)}`, { stroke: c, width: 0.9, fill: 'none' });
  elbow(lcx, 1, C.dom); elbow(rcx, -1, C.non);
  L.chevron(p, [nx - nr - 0.4, ny], [1, 0], { stroke: C.dom, width: 0.9, len: 2.4 });
  L.chevron(p, [nx + nr + 0.4, ny], [-1, 0], { stroke: C.non, width: 0.9, len: 2.4 });
  L.text(p, (lcx + nx) / 2, ny - 2, 'ŷ_{1}', { anchor: 'middle', size: 7, math: true, color: C.domTx, valign: 'bottom' });
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
  const g = (fw - 6 - widths.reduce((a, b) => a + b, 0)) / (items.length - 1);
  let x = xL;
  items.forEach(([t, c, dsh, w], i) => {
    if (c) L.line(p, [x, ly], [x + sw, ly], { stroke: c, width: w, dash: dsh });
    else L.rect(p, x, ly - 3.2, sw, 6.4, { fill: C.err, fillOpacity: 0.32, stroke: 'none' });
    L.text(p, x + sw + 3.5, ly, t, { valign: 'middle', size: 6, color: c === C.dom ? C.domTx : c === C.non ? C.nonTx : C.ink });
    x += widths[i] + g;
  });
}
