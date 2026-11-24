import * as THREE from 'three';
import { BLOCKS, BlockId, TileIndex } from '../world/blocks';

export const ATLAS_COLS = 8;
export const ATLAS_ROWS = 8;
export const TILE_PIXELS = 16;
export const TILE_PAD = 8; // 8px edge padding on each side eliminates mipmap seam bleeding
export const SLOT_PIXELS = TILE_PIXELS + TILE_PAD * 2; // 32px per slot
export const ATLAS_WIDTH = ATLAS_COLS * SLOT_PIXELS; // 256
export const ATLAS_HEIGHT = ATLAS_ROWS * SLOT_PIXELS; // 256

interface PixelData {
  r: number;
  g: number;
  b: number;
  a: number;
  h: number; // height 0..1 for normal map generation
  rough: number; // 0..1
  metal: number; // 0..1
  emissive: number; // 0..1
  sss: number; // 0..1 subsurface translucency
}

type TileBuffer = PixelData[];

// Deterministic hash for pixel-perfect authentic Minecraft noise
function hash2(x: number, y: number, seed: number): number {
  let n = Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(seed, 1442695041);
  n = (n ^ (n >>> 13)) * 1274126177;
  return ((n ^ (n >>> 16)) >>> 0) / 4294967295;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function makeBuf(): TileBuffer {
  const buf: TileBuffer = new Array(TILE_PIXELS * TILE_PIXELS);
  for (let i = 0; i < buf.length; i++) {
    buf[i] = { r: 255, g: 0, b: 255, a: 255, h: 0.5, rough: 0.85, metal: 0.0, emissive: 0.0, sss: 0.0 };
  }
  return buf;
}

function setPx(
  buf: TileBuffer,
  x: number,
  y: number,
  r: number,
  g: number,
  b: number,
  a = 255,
  h = 0.5,
  rough = 0.85,
  metal = 0.0,
  emissive = 0.0,
  sss = 0.0
) {
  if (x < 0 || x >= 16 || y < 0 || y >= 16) return;
  const p = buf[y * 16 + x];
  p.r = clamp(Math.round(r), 0, 255);
  p.g = clamp(Math.round(g), 0, 255);
  p.b = clamp(Math.round(b), 0, 255);
  p.a = clamp(Math.round(a), 0, 255);
  p.h = clamp(h, 0, 1);
  p.rough = clamp(rough, 0.02, 1);
  p.metal = clamp(metal, 0, 1);
  p.emissive = clamp(emissive, 0, 1);
  p.sss = clamp(sss, 0, 1);
}

function fillNoise(
  buf: TileBuffer,
  base: [number, number, number],
  variance: number,
  seed: number,
  rough = 0.85,
  hScale = 0.25,
  sss = 0
) {
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      const n = hash2(x, y, seed) * 2 - 1;
      const n2 = hash2(x >> 1, y >> 1, seed + 71) * 2 - 1;
      const d = (n * 0.65 + n2 * 0.35) * variance;
      setPx(
        buf,
        x,
        y,
        base[0] + d,
        base[1] + d * 0.95,
        base[2] + d * 0.9,
        255,
        0.5 + (n * 0.6 + n2 * 0.4) * hScale,
        rough,
        0,
        0,
        sss
      );
    }
  }
}

function drawDirt(buf: TileBuffer, seed = 101) {
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      const n = hash2(x, y, seed);
      const speck = hash2(x, y, seed + 17);
      let r = 134, g = 96, b = 67, h = 0.5;
      if (speck > 0.86) {
        // small pebble
        r = 152; g = 148; b = 142; h = 0.68;
      } else if (n < 0.28) {
        r = 104; g = 72; b = 49; h = 0.35;
      } else if (n > 0.75) {
        r = 150; g = 110; b = 78; h = 0.62;
      }
      setPx(buf, x, y, r, g, b, 255, h, 0.92);
    }
  }
}

function drawStone(buf: TileBuffer, seed = 201) {
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      const band = hash2(x >> 1, y, seed);
      const fine = hash2(x, y, seed + 33);
      const val = 118 + (band - 0.5) * 30 + (fine - 0.5) * 18;
      const crack = fine < 0.08 ? -20 : 0;
      setPx(buf, x, y, val + crack, val + crack, val + crack + 3, 255, 0.5 + (band - 0.5) * 0.35 + (crack ? -0.25 : 0), 0.72);
    }
  }
}

function drawOre(
  buf: TileBuffer,
  oreColor: [number, number, number],
  highlight: [number, number, number],
  seed: number,
  rough: number,
  metal: number,
  emissive = 0
) {
  drawStone(buf, seed);
  // Classic Minecraft irregular crystal pockets
  const spots: [number, number][] = [
    [3, 3], [4, 3], [4, 4], [5, 4],
    [10, 2], [11, 2], [11, 3], [12, 3],
    [2, 9], [3, 9], [3, 10], [4, 10], [2, 10],
    [9, 8], [10, 8], [10, 9], [11, 9], [9, 9],
    [6, 12], [7, 12], [7, 13], [12, 12], [13, 12],
  ];
  for (let i = 0; i < spots.length; i++) {
    const [sx, sy] = spots[i];
    const ox = (sx + Math.floor(hash2(i, seed, 9) * 2)) % 16;
    const oy = (sy + Math.floor(hash2(seed, i, 19) * 2)) % 16;
    const hi = hash2(ox, oy, seed + 5) > 0.6;
    const c = hi ? highlight : oreColor;
    setPx(buf, ox, oy, c[0], c[1], c[2], 255, hi ? 0.92 : 0.78, rough, metal, emissive);
  }
}

function drawPlanks(buf: TileBuffer, base: [number, number, number], dark: [number, number, number], seed: number) {
  for (let y = 0; y < 16; y++) {
    const plankRow = y % 4;
    const isHorizontalSeam = plankRow === 0;
    const plankIdx = y >> 2;
    const vSeamX = (plankIdx * 7 + 3) % 16;
    for (let x = 0; x < 16; x++) {
      const grain = (hash2(x >> 1, y, seed) - 0.5) * 16 + (hash2(x, y, seed + 9) - 0.5) * 8;
      const isVSeam = x === vSeamX;
      if (isHorizontalSeam || isVSeam) {
        setPx(buf, x, y, dark[0], dark[1], dark[2], 255, 0.2, 0.85);
      } else {
        const edgeBevel = plankRow === 1 ? 10 : plankRow === 3 ? -8 : 0;
        setPx(
          buf,
          x,
          y,
          base[0] + grain + edgeBevel,
          base[1] + grain * 0.9 + edgeBevel,
          base[2] + grain * 0.7 + edgeBevel,
          255,
          plankRow === 1 ? 0.7 : 0.58,
          0.74
        );
      }
    }
  }
}

function drawLogSide(
  buf: TileBuffer,
  base: [number, number, number],
  ridge: [number, number, number],
  seed: number,
  birch = false
) {
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      if (birch) {
        const n = (hash2(x, y, seed) - 0.5) * 18;
        const streak = hash2(x >> 2, y, seed + 11);
        const isDark = streak > 0.83 && (x % 4 === 1 || x % 4 === 2);
        if (isDark) {
          setPx(buf, x, y, 42, 40, 38, 255, 0.28, 0.9);
        } else {
          setPx(buf, x, y, 224 + n, 220 + n, 212 + n, 255, 0.62, 0.8);
        }
      } else {
        const colNoise = hash2(x, y >> 2, seed);
        const fine = (hash2(x, y, seed + 13) - 0.5) * 14;
        const isGroove = x % 3 === 0 ? colNoise < 0.7 : colNoise < 0.22;
        const c = isGroove ? ridge : base;
        setPx(buf, x, y, c[0] + fine, c[1] + fine * 0.85, c[2] + fine * 0.7, 255, isGroove ? 0.25 : 0.72, 0.9);
      }
    }
  }
}

function drawLogTop(
  buf: TileBuffer,
  bark: [number, number, number],
  heart: [number, number, number],
  ring: [number, number, number],
  seed: number
) {
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      const dx = Math.abs(x - 7.5);
      const dy = Math.abs(y - 7.5);
      const d = Math.max(dx, dy);
      const fine = (hash2(x, y, seed) - 0.5) * 12;
      if (d > 6.4) {
        setPx(buf, x, y, bark[0] + fine, bark[1] + fine, bark[2] + fine, 255, 0.72, 0.9);
      } else if (Math.floor(d) % 2 === 0) {
        setPx(buf, x, y, heart[0] + fine, heart[1] + fine, heart[2] + fine, 255, 0.55, 0.82);
      } else {
        setPx(buf, x, y, ring[0] + fine, ring[1] + fine, ring[2] + fine, 255, 0.45, 0.85);
      }
    }
  }
}

function drawLeaves(buf: TileBuffer, base: [number, number, number], seed: number, cutoutRate = 0.22) {
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      const n = hash2(x, y, seed);
      const cluster = hash2(x >> 1, y >> 1, seed + 47);
      if (n < cutoutRate && cluster < 0.55) {
        setPx(buf, x, y, 0, 0, 0, 0, 0.0, 0.8, 0, 0, 1.0);
      } else {
        const shade = (n - 0.5) * 42 + (cluster - 0.5) * 28;
        setPx(
          buf,
          x,
          y,
          base[0] + shade,
          base[1] + shade * 1.05,
          base[2] + shade * 0.85,
          255,
          0.4 + n * 0.5,
          0.75,
          0,
          0,
          0.95
        );
      }
    }
  }
}

function generateTile(index: number): TileBuffer {
  const buf = makeBuf();

  switch (index) {
    case TileIndex.GRASS_TOP: {
      // Neutral grayscale-green so biome tinting produces vibrant, authentic Minecraft grass!
      fillNoise(buf, [175, 198, 152], 32, 11, 0.88, 0.35, 0.35);
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          if (hash2(x, y, 19) > 0.72) {
            const p = buf[y * 16 + x];
            p.r = clamp(p.r + 16, 0, 255);
            p.g = clamp(p.g + 22, 0, 255);
            p.b = clamp(p.b + 10, 0, 255);
            p.h = 0.82;
          }
        }
      }
      break;
    }
    case TileIndex.GRASS_SIDE: {
      drawDirt(buf, 102);
      for (let x = 0; x < 16; x++) {
        const hang = 3 + Math.floor(hash2(x, 0, 23) * 4);
        for (let y = 0; y < hang; y++) {
          const n = (hash2(x, y, 29) - 0.5) * 24;
          const dark = y === hang - 1 ? -16 : 0;
          setPx(buf, x, y, 92 + n + dark, 156 + n + dark, 64 + n * 0.6 + dark, 255, 0.78, 0.86, 0, 0, 0.25);
        }
      }
      break;
    }
    case TileIndex.SNOW_GRASS_SIDE: {
      drawDirt(buf, 103);
      for (let x = 0; x < 16; x++) {
        const hang = 4 + Math.floor(hash2(x, 0, 31) * 4);
        for (let y = 0; y < hang; y++) {
          const n = (hash2(x, y, 37) - 0.5) * 14;
          setPx(buf, x, y, 238 + n, 244 + n, 252 + n, 255, 0.82, 0.55, 0.05, 0, 0.4);
        }
      }
      break;
    }
    case TileIndex.DIRT:
      drawDirt(buf, 101);
      break;
    case TileIndex.STONE:
      drawStone(buf, 201);
      break;
    case TileIndex.COBBLESTONE: {
      // Classic cobblestone Voronoi-like rounded stones with dark mortar seams
      const centers: [number, number][] = [
        [2, 2], [8, 2], [13, 3],
        [4, 7], [10, 7], [15, 7],
        [2, 12], [7, 13], [13, 12],
      ];
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          let d1 = 999, d2 = 999, idx = 0;
          for (let i = 0; i < centers.length; i++) {
            const dx = Math.min(Math.abs(x - centers[i][0]), 16 - Math.abs(x - centers[i][0]));
            const dy = Math.min(Math.abs(y - centers[i][1]), 16 - Math.abs(y - centers[i][1]));
            const d = Math.hypot(dx, dy);
            if (d < d1) { d2 = d1; d1 = d; idx = i; }
            else if (d < d2) { d2 = d; }
          }
          const edge = d2 - d1;
          const fine = (hash2(x, y, 211) - 0.5) * 18;
          const tone = 108 + (hash2(idx, 5, 223) - 0.5) * 34;
          if (edge < 0.72) {
            setPx(buf, x, y, 64 + fine, 64 + fine, 66 + fine, 255, 0.18, 0.92);
          } else {
            setPx(buf, x, y, tone + fine, tone + fine, tone + fine + 2, 255, 0.62 + Math.min(0.3, edge * 0.1), 0.76);
          }
        }
      }
      break;
    }
    case TileIndex.BEDROCK: {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const n = hash2(x, y, 307);
          const v = n < 0.3 ? 32 : n < 0.65 ? 64 : 98;
          setPx(buf, x, y, v, v, v + 3, 255, n, 0.85);
        }
      }
      break;
    }
    case TileIndex.SAND:
      fillNoise(buf, [224, 212, 162], 20, 401, 0.9, 0.22);
      break;
    case TileIndex.GRAVEL:
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const n = hash2(x, y, 419);
          const v = n < 0.25 ? 88 : n < 0.6 ? 128 : n < 0.85 ? 152 : 172;
          setPx(buf, x, y, v + 3, v, v - 2, 255, n, 0.92);
        }
      }
      break;
    case TileIndex.CLAY:
      fillNoise(buf, [158, 166, 178], 16, 431, 0.72, 0.18);
      break;
    case TileIndex.SNOW:
      fillNoise(buf, [242, 247, 255], 12, 443, 0.58, 0.18, 0.45);
      break;
    case TileIndex.ICE: {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const streak = (x + y) % 7 === 0 && hash2(x, y, 457) > 0.35;
          const n = (hash2(x, y, 461) - 0.5) * 18;
          if (streak) {
            setPx(buf, x, y, 224, 244, 255, 220, 0.72, 0.08, 0.45);
          } else {
            setPx(buf, x, y, 142 + n, 192 + n, 245 + n, 200, 0.5, 0.06, 0.35);
          }
        }
      }
      break;
    }
    case TileIndex.WATER: {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const wave = Math.sin((x + y * 0.5) * 0.8) * 14;
          setPx(buf, x, y, 36 + wave, 118 + wave, 215 + wave, 215, 0.5, 0.04, 0.6);
        }
      }
      break;
    }
    case TileIndex.LAVA: {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const n = hash2(x >> 1, y >> 1, 503) * 0.6 + hash2(x, y, 509) * 0.4;
          if (n > 0.62) {
            setPx(buf, x, y, 255, 220, 72, 255, 0.65, 0.3, 0, 1.0);
          } else if (n > 0.3) {
            setPx(buf, x, y, 242, 108, 22, 255, 0.5, 0.45, 0, 0.92);
          } else {
            setPx(buf, x, y, 168, 34, 12, 255, 0.35, 0.65, 0, 0.55);
          }
        }
      }
      break;
    }
    case TileIndex.COAL_ORE:
      drawOre(buf, [32, 32, 36], [56, 56, 62], 601, 0.55, 0.15);
      break;
    case TileIndex.IRON_ORE:
      drawOre(buf, [202, 168, 140], [234, 206, 178], 607, 0.32, 0.72);
      break;
    case TileIndex.GOLD_ORE:
      drawOre(buf, [238, 192, 42], [255, 236, 124], 613, 0.16, 0.95);
      break;
    case TileIndex.DIAMOND_ORE:
      drawOre(buf, [68, 222, 218], [164, 255, 252], 619, 0.08, 0.9, 0.22);
      break;
    case TileIndex.EMERALD_ORE:
      drawOre(buf, [34, 210, 88], [134, 255, 166], 625, 0.1, 0.85, 0.18);
      break;
    case TileIndex.OAK_LOG_SIDE:
      drawLogSide(buf, [110, 86, 52], [74, 56, 33], 701);
      break;
    case TileIndex.OAK_LOG_TOP:
      drawLogTop(buf, [96, 74, 44], [186, 152, 102], [148, 118, 76], 703);
      break;
    case TileIndex.OAK_LEAVES:
      drawLeaves(buf, [152, 186, 134], 709, 0.24);
      break;
    case TileIndex.BIRCH_LOG_SIDE:
      drawLogSide(buf, [226, 222, 214], [48, 46, 44], 719, true);
      break;
    case TileIndex.BIRCH_LOG_TOP:
      drawLogTop(buf, [214, 210, 200], [202, 184, 134], [174, 154, 106], 727);
      break;
    case TileIndex.BIRCH_LEAVES:
      drawLeaves(buf, [164, 198, 132], 733, 0.24);
      break;
    case TileIndex.SPRUCE_LOG_SIDE:
      drawLogSide(buf, [68, 48, 30], [44, 30, 18], 739);
      break;
    case TileIndex.SPRUCE_LOG_TOP:
      drawLogTop(buf, [58, 40, 24], [126, 94, 58], [98, 72, 42], 743);
      break;
    case TileIndex.SPRUCE_LEAVES:
      drawLeaves(buf, [68, 104, 72], 751, 0.18);
      break;
    case TileIndex.OAK_PLANKS:
      drawPlanks(buf, [172, 138, 86], [108, 82, 48], 801);
      break;
    case TileIndex.BIRCH_PLANKS:
      drawPlanks(buf, [206, 192, 140], [150, 136, 94], 809);
      break;
    case TileIndex.STONE_BRICKS: {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const row = y >> 3;
          const seamH = y === 0 || y === 8;
          const seamV = (row === 0 && x === 0) || (row === 1 && x === 8);
          const n = (hash2(x, y, 821) - 0.5) * 18;
          if (seamH || seamV) {
            setPx(buf, x, y, 76 + n * 0.5, 76 + n * 0.5, 80 + n * 0.5, 255, 0.16, 0.9);
          } else {
            const bevel = y === 1 || y === 9 ? 14 : y === 7 || y === 15 ? -10 : 0;
            setPx(buf, x, y, 128 + n + bevel, 128 + n + bevel, 132 + n + bevel, 255, 0.68, 0.7);
          }
        }
      }
      break;
    }
    case TileIndex.BRICKS: {
      for (let y = 0; y < 16; y++) {
        const row = y >> 2;
        const seamH = y % 4 === 0;
        const offset = (row % 2) * 4;
        for (let x = 0; x < 16; x++) {
          const seamV = (x + offset) % 8 === 0;
          const n = (hash2(x, y, 829) - 0.5) * 20;
          if (seamH || seamV) {
            setPx(buf, x, y, 198 + n * 0.4, 192 + n * 0.4, 182 + n * 0.4, 255, 0.2, 0.9);
          } else {
            const tone = (hash2(row, (x + offset) >> 3, 839) - 0.5) * 24;
            setPx(buf, x, y, 158 + tone + n, 76 + tone * 0.6 + n * 0.6, 62 + tone * 0.5 + n * 0.5, 255, 0.72, 0.78);
          }
        }
      }
      break;
    }
    case TileIndex.GLASS: {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const border = x === 0 || x === 15 || y === 0 || y === 15;
          const glint = (x === 2 && y === 2) || (x === 3 && y === 2) || (x === 2 && y === 3) || (x === 12 && y === 13) || (x === 13 && y === 12);
          if (border) {
            setPx(buf, x, y, 186, 218, 232, 235, 0.7, 0.12, 0.5);
          } else if (glint) {
            setPx(buf, x, y, 240, 250, 255, 210, 0.6, 0.05, 0.8);
          } else {
            setPx(buf, x, y, 205, 235, 248, 42, 0.5, 0.04, 0.65);
          }
        }
      }
      break;
    }
    case TileIndex.CRAFTING_TOP: {
      drawPlanks(buf, [166, 122, 72], [116, 82, 46], 853);
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const border = x === 0 || x === 15 || y === 0 || y === 15;
          const grid = x === 5 || x === 10 || y === 5 || y === 10;
          if (border) setPx(buf, x, y, 92, 62, 34, 255, 0.75, 0.82);
          else if (grid) setPx(buf, x, y, 72, 48, 26, 255, 0.25, 0.88);
        }
      }
      break;
    }
    case TileIndex.CRAFTING_SIDE:
    case TileIndex.CRAFTING_FRONT: {
      drawPlanks(buf, [164, 124, 74], [108, 76, 42], 859);
      for (let x = 0; x < 16; x++) {
        setPx(buf, x, 0, 92, 62, 34, 255, 0.72, 0.85);
        setPx(buf, x, 1, 112, 78, 44, 255, 0.68, 0.85);
      }
      // Hanging tools silhouette
      for (let y = 4; y <= 11; y++) {
        setPx(buf, 4, y, 64, 68, 74, 255, 0.82, 0.25, 0.8);
        setPx(buf, 11, y, 82, 56, 32, 255, 0.8, 0.75);
      }
      setPx(buf, 3, 4, 152, 158, 168, 255, 0.88, 0.2, 0.9);
      setPx(buf, 5, 4, 152, 158, 168, 255, 0.88, 0.2, 0.9);
      setPx(buf, 10, 5, 162, 168, 176, 255, 0.88, 0.2, 0.9);
      setPx(buf, 12, 5, 162, 168, 176, 255, 0.88, 0.2, 0.9);
      break;
    }
    case TileIndex.FURNACE_SIDE:
    case TileIndex.FURNACE_TOP: {
      drawStone(buf, 863);
      for (let x = 0; x < 16; x++) {
        setPx(buf, x, 0, 74, 74, 78, 255, 0.3, 0.85);
        setPx(buf, x, 15, 74, 74, 78, 255, 0.3, 0.85);
        setPx(buf, 0, x, 74, 74, 78, 255, 0.3, 0.85);
        setPx(buf, 15, x, 74, 74, 78, 255, 0.3, 0.85);
      }
      break;
    }
    case TileIndex.FURNACE_FRONT: {
      drawStone(buf, 877);
      // Upper dark opening
      for (let y = 2; y <= 6; y++) {
        for (let x = 3; x <= 12; x++) {
          setPx(buf, x, y, 28, 24, 24, 255, 0.12, 0.92);
        }
      }
      // Glowing fire hearth at bottom
      for (let y = 9; y <= 13; y++) {
        for (let x = 3; x <= 12; x++) {
          const hot = y >= 11 && x >= 5 && x <= 10;
          if (hot) {
            setPx(buf, x, y, 255, 198, 64, 255, 0.2, 0.4, 0, 1.0);
          } else {
            setPx(buf, x, y, 232, 86, 20, 255, 0.18, 0.5, 0, 0.85);
          }
        }
      }
      break;
    }
    case TileIndex.BOOKSHELF: {
      drawPlanks(buf, [172, 138, 86], [112, 84, 50], 881);
      const bookColors: [number, number, number][] = [
        [168, 42, 38], [46, 98, 168], [52, 134, 62], [186, 146, 44], [128, 56, 144], [156, 92, 48],
      ];
      for (const shelfY of [2, 9]) {
        for (let x = 1; x <= 14; x++) {
          const c = bookColors[(x + shelfY) % bookColors.length];
          const tall = (x % 3 === 0) ? 0 : 1;
          for (let y = shelfY + tall; y <= shelfY + 4; y++) {
            const stripe = y === shelfY + 2 ? 32 : 0;
            setPx(buf, x, y, c[0] + stripe, c[1] + stripe, c[2] + stripe, 255, 0.65, 0.78);
          }
        }
      }
      break;
    }
    case TileIndex.TNT_SIDE: {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          if (y >= 5 && y <= 10) {
            // White band with pixel TNT lettering
            const isLetter =
              (y === 6 && ((x >= 2 && x <= 4) || x === 6 || x === 9 || (x >= 11 && x <= 13))) ||
              (y === 7 && (x === 3 || x === 6 || x === 7 || x === 9 || x === 12)) ||
              (y === 8 && (x === 3 || x === 6 || x === 8 || x === 9 || x === 12)) ||
              (y === 9 && (x === 3 || x === 6 || x === 9 || x === 12));
            if (isLetter) setPx(buf, x, y, 28, 28, 28, 255, 0.62, 0.85);
            else setPx(buf, x, y, 232, 228, 220, 255, 0.6, 0.85);
          } else {
            const stick = x % 4 === 0 ? -24 : 0;
            const n = (hash2(x, y, 887) - 0.5) * 14;
            setPx(buf, x, y, 198 + stick + n, 44 + stick * 0.4 + n * 0.5, 34 + stick * 0.4, 255, stick ? 0.3 : 0.7, 0.82);
          }
        }
      }
      break;
    }
    case TileIndex.TNT_TOP:
    case TileIndex.TNT_BOTTOM: {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const fuse = index === TileIndex.TNT_TOP && x >= 7 && x <= 8 && y >= 7 && y <= 8;
          if (fuse) setPx(buf, x, y, 42, 42, 42, 255, 0.85, 0.9);
          else {
            const ring = (x % 4 === 0 || y % 4 === 0) ? -22 : 0;
            setPx(buf, x, y, 192 + ring, 46 + ring * 0.4, 38 + ring * 0.4, 255, ring ? 0.35 : 0.68, 0.82);
          }
        }
      }
      break;
    }
    case TileIndex.TORCH: {
      // Used on miniature 3D torch post
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          if (y < 4) {
            setPx(buf, x, y, 255, 228, 108, 255, 0.8, 0.3, 0, 1.0);
          } else if (y < 6) {
            setPx(buf, x, y, 255, 134, 34, 255, 0.7, 0.5, 0, 0.9);
          } else {
            const n = (hash2(x, y, 907) - 0.5) * 20;
            setPx(buf, x, y, 138 + n, 102 + n, 58 + n, 255, 0.5, 0.88);
          }
        }
      }
      break;
    }
    case TileIndex.GLOWSTONE: {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const n = hash2(x, y, 919);
          const c = hash2(x >> 1, y >> 1, 929);
          const bright = n * 0.5 + c * 0.5;
          if (bright > 0.65) {
            setPx(buf, x, y, 255, 244, 178, 255, 0.85, 0.25, 0.1, 1.0);
          } else if (bright > 0.32) {
            setPx(buf, x, y, 232, 186, 94, 255, 0.58, 0.45, 0, 0.82);
          } else {
            setPx(buf, x, y, 146, 104, 48, 255, 0.28, 0.72, 0, 0.35);
          }
        }
      }
      break;
    }
    case TileIndex.PUMPKIN_SIDE:
    case TileIndex.PUMPKIN_TOP:
    case TileIndex.JACK_O_LANTERN_FRONT: {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const rib = x % 4 === 0 ? -24 : 0;
          const n = (hash2(x, y, 937) - 0.5) * 14;
          setPx(buf, x, y, 214 + rib + n, 122 + rib * 0.6 + n * 0.7, 26 + n * 0.3, 255, rib ? 0.3 : 0.72, 0.74);
        }
      }
      if (index === TileIndex.PUMPKIN_TOP) {
        for (let y = 6; y <= 9; y++)
          for (let x = 6; x <= 9; x++)
            setPx(buf, x, y, 104, 118, 48, 255, 0.9, 0.85);
      }
      if (index === TileIndex.JACK_O_LANTERN_FRONT) {
        // Carved glowing eyes, nose, toothy grin
        const carve = (x: number, y: number) =>
          (y >= 4 && y <= 6 && ((x >= 3 && x <= 5) || (x >= 10 && x <= 12))) ||
          (y >= 7 && y <= 8 && x >= 7 && x <= 8) ||
          (y >= 10 && y <= 12 && x >= 3 && x <= 12 && !(y === 10 && (x === 5 || x === 10)) && !(y === 12 && (x === 7 || x === 8)));
        for (let y = 0; y < 16; y++) {
          for (let x = 0; x < 16; x++) {
            if (carve(x, y)) {
              setPx(buf, x, y, 255, 232, 112, 255, 0.15, 0.3, 0, 1.0);
            }
          }
        }
      }
      break;
    }
    case TileIndex.WHITE_WOOL:
      fillNoise(buf, [235, 236, 238], 18, 947, 0.96, 0.3);
      break;
    case TileIndex.RED_WOOL:
      fillNoise(buf, [178, 44, 38], 20, 953, 0.96, 0.3);
      break;
    case TileIndex.OBSIDIAN: {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const n = hash2(x, y, 967);
          const vein = n > 0.82;
          if (vein) setPx(buf, x, y, 76, 44, 112, 255, 0.72, 0.2, 0.45);
          else setPx(buf, x, y, 20 + n * 16, 16 + n * 12, 32 + n * 22, 255, 0.4 + n * 0.3, 0.22, 0.35);
        }
      }
      break;
    }
    case TileIndex.TALL_GRASS: {
      for (let i = 0; i < 256; i++) buf[i].a = 0;
      const blades = [
        { x: 2, h: 10 }, { x: 4, h: 13 }, { x: 6, h: 9 }, { x: 8, h: 14 },
        { x: 10, h: 11 }, { x: 12, h: 13 }, { x: 14, h: 8 },
      ];
      for (const b of blades) {
        for (let j = 0; j < b.h; j++) {
          const y = 15 - j;
          const sway = j > 6 ? (b.x > 8 ? 1 : -1) : 0;
          const bright = 148 + j * 5;
          setPx(buf, b.x + sway, y, bright, bright + 24, bright - 16, 255, 0.7, 0.8, 0, 0, 1.0);
        }
      }
      break;
    }
    case TileIndex.POPPY: {
      for (let i = 0; i < 256; i++) buf[i].a = 0;
      for (let y = 7; y <= 15; y++) setPx(buf, 8, y, 56, 132, 46, 255, 0.6, 0.8, 0, 0, 0.9);
      setPx(buf, 7, 12, 62, 146, 52, 255, 0.6, 0.8, 0, 0, 0.9);
      setPx(buf, 9, 10, 62, 146, 52, 255, 0.6, 0.8, 0, 0, 0.9);
      // Red poppy petals
      for (let y = 3; y <= 7; y++) {
        for (let x = 6; x <= 10; x++) {
          if (Math.hypot(x - 8, y - 5) <= 2.3) {
            const center = x === 8 && y === 5;
            setPx(buf, x, y, center ? 42 : 226, center ? 26 : 36, center ? 22 : 34, 255, 0.85, 0.65, 0, 0, 0.85);
          }
        }
      }
      break;
    }
    case TileIndex.DANDELION: {
      for (let i = 0; i < 256; i++) buf[i].a = 0;
      for (let y = 8; y <= 15; y++) setPx(buf, 8, y, 64, 142, 48, 255, 0.6, 0.8, 0, 0, 0.9);
      setPx(buf, 7, 13, 64, 142, 48, 255, 0.6, 0.8, 0, 0, 0.9);
      for (let y = 5; y <= 8; y++) {
        for (let x = 6; x <= 10; x++) {
          if (Math.hypot(x - 8, y - 6.5) <= 2.1) {
            setPx(buf, x, y, 248, 214, 42, 255, 0.85, 0.65, 0, 0, 0.85);
          }
        }
      }
      break;
    }
    case TileIndex.SUGAR_CANE: {
      for (let i = 0; i < 256; i++) buf[i].a = 0;
      for (const sx of [3, 7, 11, 13]) {
        for (let y = 0; y < 16; y++) {
          const node = y % 5 === 0;
          setPx(buf, sx, y, node ? 148 : 114, node ? 204 : 178, node ? 96 : 74, 255, 0.7, 0.72, 0, 0, 0.9);
          setPx(buf, sx + 1, y, node ? 132 : 98, node ? 186 : 160, node ? 84 : 64, 255, 0.65, 0.72, 0, 0, 0.9);
        }
      }
      break;
    }
    case TileIndex.CACTUS_SIDE:
    case TileIndex.CACTUS_TOP: {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const rib = x % 4 === 0 ? -20 : x % 4 === 2 ? 12 : 0;
          const spine = index === TileIndex.CACTUS_SIDE && (x % 4 === 2) && (y % 4 === 1);
          if (spine) setPx(buf, x, y, 224, 218, 178, 255, 0.95, 0.7);
          else setPx(buf, x, y, 68 + rib, 136 + rib, 58 + rib * 0.6, 255, rib > 0 ? 0.72 : 0.45, 0.82);
        }
      }
      break;
    }
    default:
      fillNoise(buf, [200, 50, 200], 20, 999);
  }

  return buf;
}

export class ProceduralAtlas {
  public readonly albedoTexture: THREE.CanvasTexture;
  public readonly normalTexture: THREE.CanvasTexture;
  public readonly pbrTexture: THREE.CanvasTexture;
  public readonly albedoCanvas: HTMLCanvasElement;
  private readonly tiles: Map<number, TileBuffer> = new Map();
  private readonly iconCache: Map<number, string> = new Map();

  constructor() {
    this.albedoCanvas = document.createElement('canvas');
    this.albedoCanvas.width = ATLAS_WIDTH;
    this.albedoCanvas.height = ATLAS_HEIGHT;
    const albedoCtx = this.albedoCanvas.getContext('2d')!;
    const albedoImg = albedoCtx.createImageData(ATLAS_WIDTH, ATLAS_HEIGHT);

    const normalCanvas = document.createElement('canvas');
    normalCanvas.width = ATLAS_WIDTH;
    normalCanvas.height = ATLAS_HEIGHT;
    const normalCtx = normalCanvas.getContext('2d')!;
    const normalImg = normalCtx.createImageData(ATLAS_WIDTH, ATLAS_HEIGHT);

    const pbrCanvas = document.createElement('canvas');
    pbrCanvas.width = ATLAS_WIDTH;
    pbrCanvas.height = ATLAS_HEIGHT;
    const pbrCtx = pbrCanvas.getContext('2d')!;
    const pbrImg = pbrCtx.createImageData(ATLAS_WIDTH, ATLAS_HEIGHT);

    const tileValues = Object.values(TileIndex);
    for (const tileIdx of tileValues) {
      const buf = generateTile(tileIdx);
      this.tiles.set(tileIdx, buf);
      this.writeTileToAtlases(tileIdx, buf, albedoImg.data, normalImg.data, pbrImg.data);
    }

    albedoCtx.putImageData(albedoImg, 0, 0);
    normalCtx.putImageData(normalImg, 0, 0);
    pbrCtx.putImageData(pbrImg, 0, 0);

    this.albedoTexture = new THREE.CanvasTexture(this.albedoCanvas);
    this.albedoTexture.colorSpace = THREE.SRGBColorSpace;
    this.albedoTexture.magFilter = THREE.NearestFilter;
    this.albedoTexture.minFilter = THREE.LinearMipmapLinearFilter;
    this.albedoTexture.anisotropy = 8;
    this.albedoTexture.flipY = false;
    this.albedoTexture.needsUpdate = true;

    this.normalTexture = new THREE.CanvasTexture(normalCanvas);
    this.normalTexture.colorSpace = THREE.NoColorSpace;
    this.normalTexture.magFilter = THREE.NearestFilter;
    this.normalTexture.minFilter = THREE.LinearMipmapLinearFilter;
    this.normalTexture.anisotropy = 8;
    this.normalTexture.flipY = false;
    this.normalTexture.needsUpdate = true;

    this.pbrTexture = new THREE.CanvasTexture(pbrCanvas);
    this.pbrTexture.colorSpace = THREE.NoColorSpace;
    this.pbrTexture.magFilter = THREE.NearestFilter;
    this.pbrTexture.minFilter = THREE.LinearMipmapLinearFilter;
    this.pbrTexture.anisotropy = 8;
    this.pbrTexture.flipY = false;
    this.pbrTexture.needsUpdate = true;
  }

  private writeTileToAtlases(
    tileIdx: number,
    buf: TileBuffer,
    albedo: Uint8ClampedArray,
    normal: Uint8ClampedArray,
    pbr: Uint8ClampedArray
  ) {
    const col = tileIdx % ATLAS_COLS;
    const row = Math.floor(tileIdx / ATLAS_COLS);
    const baseX = col * SLOT_PIXELS;
    const baseY = row * SLOT_PIXELS;

    // Precompute Sobel normals for this 16x16 tile
    const normals: [number, number, number][] = new Array(256);
    const strength = 2.6;
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        const xm = (x + 15) & 15;
        const xp = (x + 1) & 15;
        const ym = (y + 15) & 15;
        const yp = (y + 1) & 15;
        const tl = buf[ym * 16 + xm].h;
        const t = buf[ym * 16 + x].h;
        const tr = buf[ym * 16 + xp].h;
        const l = buf[y * 16 + xm].h;
        const r = buf[y * 16 + xp].h;
        const bl = buf[yp * 16 + xm].h;
        const b = buf[yp * 16 + x].h;
        const br = buf[yp * 16 + xp].h;

        const dX = (tr + 2 * r + br) - (tl + 2 * l + bl);
        const dY = (bl + 2 * b + br) - (tl + 2 * t + tr);
        let nx = -dX * strength;
        let ny = -dY * strength;
        let nz = 1.0;
        const len = Math.hypot(nx, ny, nz);
        nx /= len;
        ny /= len;
        nz /= len;
        normals[y * 16 + x] = [
          Math.round((nx * 0.5 + 0.5) * 255),
          Math.round((ny * 0.5 + 0.5) * 255),
          Math.round((nz * 0.5 + 0.5) * 255),
        ];
      }
    }

    // Write padded 32x32 slot (clamp-to-edge padding eliminates atlas mipmap bleed)
    for (let sy = 0; sy < SLOT_PIXELS; sy++) {
      const ty = clamp(sy - TILE_PAD, 0, TILE_PIXELS - 1);
      for (let sx = 0; sx < SLOT_PIXELS; sx++) {
        const tx = clamp(sx - TILE_PAD, 0, TILE_PIXELS - 1);
        const px = buf[ty * 16 + tx];
        const nm = normals[ty * 16 + tx];
        const off = ((baseY + sy) * ATLAS_WIDTH + (baseX + sx)) * 4;

        albedo[off] = px.r;
        albedo[off + 1] = px.g;
        albedo[off + 2] = px.b;
        albedo[off + 3] = px.a;

        normal[off] = nm[0];
        normal[off + 1] = nm[1];
        normal[off + 2] = nm[2];
        normal[off + 3] = 255;

        pbr[off] = Math.round(px.rough * 255);
        pbr[off + 1] = Math.round(px.metal * 255);
        pbr[off + 2] = Math.round(px.emissive * 255);
        pbr[off + 3] = Math.round(px.sss * 255);
      }
    }
  }

  public getTileUV(tileIdx: number): { u0: number; v0: number; u1: number; v1: number } {
    const col = tileIdx % ATLAS_COLS;
    const row = Math.floor(tileIdx / ATLAS_COLS);
    const eps = 0.02 / ATLAS_WIDTH;
    const u0 = (col * SLOT_PIXELS + TILE_PAD) / ATLAS_WIDTH + eps;
    const v0 = (row * SLOT_PIXELS + TILE_PAD) / ATLAS_HEIGHT + eps;
    const u1 = (col * SLOT_PIXELS + TILE_PAD + TILE_PIXELS) / ATLAS_WIDTH - eps;
    const v1 = (row * SLOT_PIXELS + TILE_PAD + TILE_PIXELS) / ATLAS_HEIGHT - eps;
    return { u0, v0, u1, v1 };
  }

  // Render an authentic isometric 3D Minecraft block icon (or 2D sprite for flowers/torches)
  public getBlockIconDataUrl(blockId: number): string {
    const cached = this.iconCache.get(blockId);
    if (cached) return cached;

    const def = BLOCKS[blockId];
    const c = document.createElement('canvas');
    c.width = 48;
    c.height = 48;
    const ctx = c.getContext('2d')!;
    ctx.imageSmoothingEnabled = false;

    if (!def || blockId === BlockId.AIR) {
      return c.toDataURL();
    }

    const drawTileToTemp = (tileIdx: number, tint?: [number, number, number]): HTMLCanvasElement => {
      const tc = document.createElement('canvas');
      tc.width = 16;
      tc.height = 16;
      const tctx = tc.getContext('2d')!;
      const img = tctx.createImageData(16, 16);
      const buf = this.tiles.get(tileIdx);
      if (buf) {
        for (let i = 0; i < 256; i++) {
          const p = buf[i];
          img.data[i * 4] = tint ? clamp(Math.round((p.r * tint[0]) / 255), 0, 255) : p.r;
          img.data[i * 4 + 1] = tint ? clamp(Math.round((p.g * tint[1]) / 255), 0, 255) : p.g;
          img.data[i * 4 + 2] = tint ? clamp(Math.round((p.b * tint[2]) / 255), 0, 255) : p.b;
          img.data[i * 4 + 3] = p.a;
        }
      }
      tctx.putImageData(img, 0, 0);
      return tc;
    };

    if (def.shape === 'cross') {
      const tint: [number, number, number] | undefined = def.tintAll ? [116, 198, 86] : undefined;
      const sprite = drawTileToTemp(def.tiles.north, tint);
      ctx.drawImage(sprite, 6, 6, 36, 36);
    } else if (def.shape === 'torch') {
      // Iconic Minecraft pixel torch icon
      ctx.fillStyle = '#7c5a32';
      ctx.fillRect(21, 18, 6, 24);
      ctx.fillStyle = '#5c4122';
      ctx.fillRect(25, 18, 2, 24);
      ctx.fillStyle = '#ff881a';
      ctx.fillRect(20, 12, 8, 7);
      ctx.fillStyle = '#fff078';
      ctx.fillRect(22, 10, 4, 6);
    } else {
      const grassTint: [number, number, number] = [118, 206, 86];
      const topCanvas = drawTileToTemp(def.tiles.top, def.tintTop || def.tintAll ? grassTint : undefined);
      const leftCanvas = drawTileToTemp(def.tiles.north, def.tintAll ? grassTint : undefined);
      const rightCanvas = drawTileToTemp(def.tiles.east, def.tintAll ? grassTint : undefined);

      // Draw crisp isometric cube (3 affine-mapped faces)
      // Top diamond
      ctx.save();
      ctx.setTransform(1.18, 0.58, -1.18, 0.58, 24, 5.5);
      ctx.drawImage(topCanvas, 0, 0);
      ctx.restore();

      // Left face (slightly shaded)
      ctx.save();
      ctx.setTransform(1.18, 0.58, 0, 1.32, 5.1, 14.8);
      ctx.drawImage(leftCanvas, 0, 0);
      ctx.fillStyle = 'rgba(0,0,0,0.16)';
      ctx.fillRect(0, 0, 16, 16);
      ctx.restore();

      // Right face (more shaded)
      ctx.save();
      ctx.setTransform(1.18, -0.58, 0, 1.32, 24, 24.1);
      ctx.drawImage(rightCanvas, 0, 0);
      ctx.fillStyle = 'rgba(0,0,0,0.34)';
      ctx.fillRect(0, 0, 16, 16);
      ctx.restore();
    }

    const url = c.toDataURL();
    this.iconCache.set(blockId, url);
    return url;
  }

  public getAverageColor(blockId: number): THREE.Color {
    const def = BLOCKS[blockId];
    if (!def) return new THREE.Color(0x888888);
    const buf = this.tiles.get(def.tiles.north);
    if (!buf) return new THREE.Color(0x888888);
    let r = 0, g = 0, b = 0, n = 0;
    for (const p of buf) {
      if (p.a > 128) {
        r += p.r; g += p.g; b += p.b; n++;
      }
    }
    if (n === 0) return new THREE.Color(0x888888);
    const col = new THREE.Color(r / (n * 255), g / (n * 255), b / (n * 255));
    if (def.tintAll) col.multiply(new THREE.Color(0.55, 0.85, 0.42));
    return col;
  }
}
