// Shared toy signal + drawing helpers. Illustrative only (no paper data).
export const COL = { dom: '#0072B2', domTx: '#005B8F', non: '#E69F00', nonTx: '#A16F00', err: '#8a90a0', ink: '#2B2B2B', mute: '#7A7A7A', domT: '#DCEBF5', nonT: '#FBECD2' };
const N = 150, SPLIT = 0.34, AX = '#555b6b';
const smooth = (t) => 0.50 + 0.17 * Math.sin(2 * Math.PI * 2.3 * t) + 0.10 * t;
const bump = (t) => [[0.46, 0.2], [0.60, -0.22], [0.74, 0.24], [0.89, -0.18]].reduce((s, [c, a]) => s + a * Math.exp(-((t - c) ** 2) / (2 * 0.017 ** 2)), 0);
const noise = (i, k) => 0.010 * Math.sin(i * 12.9898 * k + k) + 0.006 * Math.sin(i * 4.1 + k);
export const T = Array.from({ length: N }, (_, i) => i / (N - 1));
export const actual = T.map((t, i) => smooth(t) + bump(t) + noise(i, 1));
// base forecast: a lagged, slightly damped copy of the smooth part; it carries no event information
export const base = T.map((t) => smooth(t - 0.011) * 0.98 + 0.005);
// corrected = (1 + alpha_hat) * base. alpha_hat is a shrunk, smoothed version of the ideal ratio y/base - 1,
// plus its own small model error, so the corrected curve follows the events only partly (never a perfect fit)
const ratio = actual.map((v, i) => v / base[i] - 1);
const ker = (s) => { const r = Math.ceil(3 * s), w = []; for (let k = -r; k <= r; k++) w.push([k, Math.exp(-(k * k) / (2 * s * s))]); return w; };
const blur = (a, s) => a.map((_, i) => { let num = 0, den = 0; for (const [k, w] of ker(s)) { const j = i + k; if (j >= 0 && j < a.length) { num += w * a[j]; den += w; } } return num / den; });
export const alphaHat = blur(ratio, 1.8).map((r, i) => 0.85 * r + 0.03 * Math.sin(2 * Math.PI * 5.3 * T[i] + 0.7));
export const corrected = base.map((b, i) => b * (1 + alphaHat[i]));
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
  return { X, Y };
}
export const pts = (M, ys, a = 0, b = ys.length - 1) => ys.slice(a, b + 1).map((v, k) => [M.X(T[a + k]), M.Y(v)]);
export function errBand(L, p, M, f, fill = COL.err, op = 0.3) {
  const a = pts(M, actual, iSplit), b = pts(M, f, iSplit);
  L.polygon(p, [...a, ...b.reverse()], { fill, fillOpacity: op, stroke: 'none' });
}
export function history(L, p, M) { L.polyline(p, pts(M, actual, 0, iSplit), { stroke: COL.ink, width: 1.0, join: 'round' }); }
export function truth(L, p, M) { L.polyline(p, pts(M, actual, iSplit), { stroke: COL.ink, width: 1.0, join: 'round' }); }

// one tree glyph (root, two splits, four leaves), the field's glyph for a boosted tree
function tree(L, p, x, cy, color) {
  const w = 0.6, root = [x, cy - 5], mid = [[x - 3.5, cy], [x + 3.5, cy]], leaf = [-5.2, -1.8, 1.8, 5.2];
  mid.forEach((m) => L.line(p, root, m, { stroke: color, width: w }));
  leaf.forEach((lx, j) => L.line(p, mid[j >> 1], [x + lx, cy + 5], { stroke: color, width: w }));
  L.circle(p, root[0], root[1], 1.2, { fill: color });
  mid.forEach((m) => L.circle(p, m[0], m[1], 1.0, { fill: color }));
  leaf.forEach((lx) => L.rect(p, x + lx - 1, cy + 4, 2, 2, { fill: '#fff', stroke: color, width: 0.6 }));
}
const dots = (L, p, x, y, color) => [0, 1, 2].forEach((d) => L.circle(p, x + d * 3.2, y, 0.8, { fill: color }));

// a tree ensemble of n trees per row, centred on cx; the first row's roots sit at cy - 5. With rows = 2 a second,
// staggered row (n - 1 trees, half a pitch over) carries the ellipsis instead of the first row.
export function trees(L, p, cx, cy, color, { n = 3, rows = 1, pitch = 15, rowGap = 13 } = {}) {
  const half = 6.2, dotsW = 2 * 3.2 + 0.8;
  const r1 = (n - 1) * pitch + half + (rows === 1 ? 10 - half + dotsW : 0);           // right end of row 1, from x0
  const r2 = rows === 2 ? pitch / 2 + (n - 1) * pitch - 3.2 + dotsW : -Infinity;      // right end of row 2's ellipsis
  const x0 = cx - (Math.max(r1, r2) - half) / 2;                                        // centre the whole glyph
  for (let i = 0; i < n; i++) tree(L, p, x0 + i * pitch, cy, color);
  if (rows === 1) { dots(L, p, x0 + (n - 1) * pitch + 10, cy + 3, color); return cy + 6; }
  const y2 = cy + rowGap;
  for (let i = 0; i < n - 1; i++) tree(L, p, x0 + pitch / 2 + i * pitch, y2, color);
  dots(L, p, x0 + pitch / 2 + (n - 1) * pitch - 3.2, y2 + 3, color);
  return y2 + 6;
}
