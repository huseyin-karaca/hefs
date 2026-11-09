import { drawHero } from './hero.js';
// Detailed version: the web hero plus the target-search bound a ∈ [−β, β] and the base-forecast error band.
export const figure = { width: 120, height: 100, name: 'forecast-hero-detailed', format: 'journal' };
export default function draw(F, L) { drawHero(F, L, { lite: false, legend: 'top' }); }
