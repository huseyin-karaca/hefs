import { drawHero } from './hero.js';
// Web hero (light version): the arrow legend sits top right; the target-search bound and the error band are left to the caption.
export const figure = { width: 120, height: 100, name: 'forecast-hero', format: 'journal' };
export default function draw(F, L) { drawHero(F, L, { lite: true, legend: 'top' }); }
