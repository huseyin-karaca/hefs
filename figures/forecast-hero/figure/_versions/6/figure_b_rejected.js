import { drawHero } from './hero.js';
// Draft b: arrow legend at the bottom.
export const figure = { width: 120, height: 100, name: 'forecast-hero-draft-b', format: 'journal' };
export default function draw(F, L) { drawHero(F, L, { lite: true, legend: 'bottom' }); }
