// Shared toy signal + drawing helpers. Illustrative only (no paper data).
export const COL = { dom: '#0072B2', domTx: '#005B8F', non: '#E69F00', nonTx: '#A16F00', err: '#8a90a0', ink: '#2B2B2B', mute: '#7A7A7A', domT: '#DCEBF5', nonT: '#FBECD2' };
const N = 72, SPLIT = 0.34, AX = '#555b6b';
const smooth = (t) => 0.50 + 0.17 * Math.sin(2 * Math.PI * 2.3 * t) + 0.10 * t;
const bump = (t) => [[0.46, 0.2], [0.60, -0.22], [0.74, 0.24], [0.89, -0.18]].reduce((s, [c, a]) => s + a * Math.exp(-((t - c) ** 2) / (2 * 0.016 ** 2)), 0);
const noise = (i, k) => 0.012 * Math.sin(i * 12.9898 * k + k) + 0.008 * Math.sin(i * 4.1 + k);
export const T = Array.from({ length: N }, (_, i) => i / (N - 1));
export const actual = T.map((t, i) => smooth(t) + bump(t) + noise(i, 1));
export const base = T.map((t, i) => smooth(t - 0.012) * 0.985 + noise(i, 2) * 0.5);        // lag-style forecast: no event info
export const corrected = T.map((t, i) => base[i] + 0.88 * bump(t) + noise(i, 3) * 0.6);     // + learned correction
export const iSplit = Math.round(SPLIT * (N - 1));

// plot frame in pt: shaded forecast region, History | Forecast divider, time axis; returns mappers
export function frame(L, p, x, y, w, h, [y0, y1] = [0.1, 1.05]) {
  const X = (t) => x + t * w, Y = (v) => y + h - ((v - y0) / (y1 - y0)) * h, xs = X(SPLIT);
  L.rect(p, xs, y, w * (1 - SPLIT), h, { fill: '#f5f6f9', stroke: 'none' });
  L.line(p, [xs, y], [xs, y + h], { stroke: '#a3a8b4', width: 0.5, dash: '1.6 1.2' });
  L.arrow(p, [x, y + h], [x + w + 3, y + h], { stroke: AX, width: 0.6, head: 2.6 });
  L.line(p, [x, y], [x, y + h], { stroke: AX, width: 0.6 });
  const lab = { size: 6, color: COL.mute, valign: 'top' };
  L.text(p, xs - 4, y + 3, 'History', { ...lab, anchor: 'end' });
  L.text(p, xs + 4, y + 3, 'Forecast', { ...lab, anchor: 'start' });
  L.text(p, x + w + 3, y + h + 3.5, 'Time', { ...lab, anchor: 'end' });
  return { X, Y };
}
export const pts = (M, ys, a = 0, b = ys.length - 1) => ys.slice(a, b + 1).map((v, k) => [M.X(T[a + k]), M.Y(v)]);
export function errBand(L, p, M, f, fill = COL.err, op = 0.3) {
  const a = pts(M, actual, iSplit), b = pts(M, f, iSplit);
  L.polygon(p, [...a, ...b.reverse()], { fill, fillOpacity: op, stroke: 'none' });
}
export function history(L, p, M) { L.polyline(p, pts(M, actual, 0, iSplit), { stroke: COL.ink, width: 1.0 }); }
export function truth(L, p, M) { L.polyline(p, pts(M, actual, iSplit), { stroke: COL.ink, width: 1.0 }); }

// small tree-ensemble glyph (the field's glyph for boosted trees)
export function trees(L, p, cx, cy, color, n = 3) {
  for (let i = 0; i < n; i++) {
    const x = cx + (i - (n - 1) / 2) * 15, w = 0.6;
    const root = [x, cy - 5], mid = [[x - 3.5, cy], [x + 3.5, cy]], leaf = [-5.2, -1.8, 1.8, 5.2];
    mid.forEach((m) => L.line(p, root, m, { stroke: color, width: w }));
    leaf.forEach((lx, j) => L.line(p, mid[j >> 1], [x + lx, cy + 5], { stroke: color, width: w }));
    L.circle(p, root[0], root[1], 1.2, { fill: color });
    mid.forEach((m) => L.circle(p, m[0], m[1], 1.0, { fill: color }));
    leaf.forEach((lx) => L.rect(p, x + lx - 1, cy + 4, 2, 2, { fill: '#fff', stroke: color, width: 0.6 }));
  }
  [0, 1, 2].forEach((d) => L.circle(p, cx + ((n - 1) / 2) * 15 + 10 + d * 3.2, cy + 3, 0.8, { fill: color }));
}
