import { BiomeType, CHUNK_HEIGHT, CHUNK_SIZE, SEA_LEVEL } from '../types';
import { BlockId } from './blocks';
import { Chunk } from './Chunk';
import { hashCoords, hashCoords3, SeededNoise } from './noise';

export interface ColumnSample {
  height: number;
  biome: BiomeType;
  temperature: number;
  moisture: number;
  grassColor: [number, number, number];
  foliageColor: [number, number, number];
}

export class WorldGen {
  public readonly seed: number;
  private readonly continentNoise: SeededNoise;
  private readonly detailNoise: SeededNoise;
  private readonly mountainNoise: SeededNoise;
  private readonly riverNoise: SeededNoise;
  private readonly tempNoise: SeededNoise;
  private readonly moistNoise: SeededNoise;
  private readonly caveNoiseA: SeededNoise;
  private readonly caveNoiseB: SeededNoise;

  constructor(seed: number) {
    this.seed = seed | 0;
    this.continentNoise = new SeededNoise(this.seed + 101);
    this.detailNoise = new SeededNoise(this.seed + 202);
    this.mountainNoise = new SeededNoise(this.seed + 303);
    this.riverNoise = new SeededNoise(this.seed + 404);
    this.tempNoise = new SeededNoise(this.seed + 505);
    this.moistNoise = new SeededNoise(this.seed + 606);
    this.caveNoiseA = new SeededNoise(this.seed + 707);
    this.caveNoiseB = new SeededNoise(this.seed + 808);
  }

  public sampleColumn(wx: number, wz: number): ColumnSample {
    const temp = this.tempNoise.fbm2D(wx * 0.0032, wz * 0.0032, 3);
    const moist = this.moistNoise.fbm2D(wx * 0.0036 + 50, wz * 0.0036 - 50, 3);

    const cont = this.continentNoise.fbm2D(wx * 0.0045, wz * 0.0045, 4);
    const hills = this.detailNoise.fbm2D(wx * 0.018, wz * 0.018, 4);
    const ridge = this.mountainNoise.ridged2D(wx * 0.0075, wz * 0.0075, 4);
    const riverRaw = Math.abs(this.riverNoise.fbm2D(wx * 0.0042, wz * 0.0042, 3));

    // Scenic spawn valley shaping so spawn (0, 0) always has a picturesque overlook, lake, cabin & mountains in view
    const distFromSpawn = Math.hypot(wx, wz);
    const spawnOverlook = Math.max(0, 1 - distFromSpawn / 24);
    const spawnBowl = Math.max(0, 1 - distFromSpawn / 62);

    // Sculpt a scenic sparkling lake slightly ahead of spawn (+z direction around z = 18..46)
    const lakeDist = Math.hypot((wx - 4) * 0.68, (wz - 30) * 1.0);
    const lakeCarve = Math.max(0, 1 - lakeDist / 24);
    // And an iconic snow-capped mountain range rising behind the lake (z = 62..108)
    const vistaPeakDist = Math.hypot((wx - 10) * 0.58, (wz - 80) * 0.92);
    const vistaPeakBoost = Math.max(0, 1 - vistaPeakDist / 46);

    const mountainMask = Math.max(0, Math.min(1, (ridge - 0.42) * 2.2));
    let baseHeight =
      SEA_LEVEL +
      4.5 +
      cont * 16 +
      hills * 6.5 +
      mountainMask * mountainMask * 38;

    // Carve smooth winding rivers
    if (riverRaw < 0.085) {
      const riverDepth = (1 - riverRaw / 0.085);
      const smooth = riverDepth * riverDepth * (3 - 2 * riverDepth);
      baseHeight = baseHeight * (1 - smooth) + (SEA_LEVEL - 7) * smooth;
    }

    // Smoothly tame mountains in the spawn valley so the player stands on a sunlit grassy ridge overlooking the lake
    if (spawnBowl > 0) {
      const sb = spawnBowl * spawnBowl * (3 - 2 * spawnBowl);
      const gentleMeadow = SEA_LEVEL + 5.5 + hills * 3.8 + cont * 3.5;
      baseHeight = baseHeight * (1 - sb * 0.85) + gentleMeadow * (sb * 0.85);
    }
    if (spawnOverlook > 0) {
      const so = spawnOverlook * spawnOverlook * (3 - 2 * spawnOverlook);
      const overlookTarget = SEA_LEVEL + 6.5 + hills * 1.6;
      baseHeight = baseHeight * (1 - so) + overlookTarget * so;
    }
    if (lakeCarve > 0) {
      const s = lakeCarve * lakeCarve * (3 - 2 * lakeCarve);
      baseHeight = baseHeight * (1 - s * 0.92) + (SEA_LEVEL - 7.5) * (s * 0.92);
    }
    if (vistaPeakBoost > 0) {
      const s = vistaPeakBoost * vistaPeakBoost;
      baseHeight += s * (30 + ridge * 18);
    }

    const height = Math.max(8, Math.min(CHUNK_HEIGHT - 16, Math.round(baseHeight)));

    // Biome classification
    let biome: BiomeType = 'plains';
    if (height < SEA_LEVEL - 1) {
      biome = 'river';
    } else if (height > SEA_LEVEL + 24 || (mountainMask > 0.58 && height > SEA_LEVEL + 16)) {
      biome = 'mountains';
    } else if (temp > 0.34 && moist < -0.05 && distFromSpawn > 48) {
      biome = 'desert';
    } else if (temp < -0.22) {
      biome = 'taiga';
    } else if (moist > 0.16) {
      biome = temp > 0.08 ? 'birch_forest' : 'forest';
    } else if (moist > -0.08 && distFromSpawn < 65) {
      biome = 'forest';
    } else {
      biome = 'plains';
    }

    // Authentic Minecraft biome grass & foliage tint colors (normalized RGB)
    let grassColor: [number, number, number];
    let foliageColor: [number, number, number];
    switch (biome) {
      case 'forest':
        grassColor = [0.48, 0.78, 0.34];
        foliageColor = [0.40, 0.72, 0.28];
        break;
      case 'birch_forest':
        grassColor = [0.54, 0.80, 0.38];
        foliageColor = [0.50, 0.78, 0.34];
        break;
      case 'taiga':
        grassColor = [0.44, 0.68, 0.46];
        foliageColor = [0.36, 0.60, 0.40];
        break;
      case 'desert':
        grassColor = [0.76, 0.74, 0.40];
        foliageColor = [0.68, 0.68, 0.32];
        break;
      case 'mountains':
        grassColor = [0.48, 0.72, 0.44];
        foliageColor = [0.42, 0.66, 0.38];
        break;
      default:
        grassColor = [0.56, 0.84, 0.36];
        foliageColor = [0.46, 0.78, 0.30];
        break;
    }

    return { height, biome, temperature: temp, moisture: moist, grassColor, foliageColor };
  }

  public generateChunk(chunk: Chunk): void {
    const baseX = chunk.cx * CHUNK_SIZE;
    const baseZ = chunk.cz * CHUNK_SIZE;

    const heights = new Int16Array(CHUNK_SIZE * CHUNK_SIZE);
    const biomes: BiomeType[] = new Array(CHUNK_SIZE * CHUNK_SIZE);

    for (let lz = 0; lz < CHUNK_SIZE; lz++) {
      const wz = baseZ + lz;
      for (let lx = 0; lx < CHUNK_SIZE; lx++) {
        const wx = baseX + lx;
        const sample = this.sampleColumn(wx, wz);
        const h = sample.height;
        const idx2 = lz * CHUNK_SIZE + lx;
        heights[idx2] = h;
        biomes[idx2] = sample.biome;

        const isBeach = h <= SEA_LEVEL + 2 && sample.biome !== 'mountains';
        const isSnowPeak = h >= SEA_LEVEL + 29 || (sample.biome === 'taiga' && h >= SEA_LEVEL + 16);

        for (let y = 0; y < CHUNK_HEIGHT; y++) {
          let block: number = BlockId.AIR;

          if (y === 0) {
            block = BlockId.BEDROCK;
          } else if (y <= 2 && hashCoords3(wx, y, wz, this.seed + 19) < (3 - y) * 0.42) {
            block = BlockId.BEDROCK;
          } else if (y <= h) {
            if (y === h) {
              if (sample.biome === 'desert') {
                block = BlockId.SAND;
              } else if (isSnowPeak) {
                block = h >= SEA_LEVEL + 33 ? BlockId.SNOW_BLOCK : BlockId.SNOW_GRASS;
              } else if (h < SEA_LEVEL - 1) {
                const bed = hashCoords(wx, wz, this.seed + 91);
                block = bed < 0.55 ? BlockId.SAND : bed < 0.8 ? BlockId.GRAVEL : BlockId.CLAY;
              } else if (isBeach) {
                block = BlockId.SAND;
              } else if (sample.biome === 'mountains' && h > SEA_LEVEL + 22) {
                block = BlockId.STONE;
              } else {
                block = BlockId.GRASS;
              }
            } else if (y >= h - 3) {
              if (sample.biome === 'desert' || isBeach) {
                block = BlockId.SAND;
              } else if (sample.biome === 'mountains' && h > SEA_LEVEL + 22) {
                block = BlockId.STONE;
              } else {
                block = BlockId.DIRT;
              }
            } else {
              block = this.pickUndergroundBlock(wx, y, wz);
            }

            // 3D Cave carving (keep surface water intact and keep bedrock safe)
            if (y > 4 && y < h - 2 && y < SEA_LEVEL - 2) {
              const ca = this.caveNoiseA.noise3D(wx * 0.048, y * 0.068, wz * 0.048);
              const cb = this.caveNoiseB.noise3D(wx * 0.048 + 100, y * 0.068, wz * 0.048 - 100);
              if (ca * ca + cb * cb < 0.022) {
                block = y <= 9 ? BlockId.LAVA : BlockId.AIR;
              }
            }
          } else if (y <= SEA_LEVEL) {
            block = sample.biome === 'taiga' && y === SEA_LEVEL && h < SEA_LEVEL - 2 && hashCoords(wx, wz, this.seed + 511) < 0.35
              ? BlockId.ICE
              : BlockId.WATER;
          }

          chunk.setBlockLocal(lx, y, lz, block);
        }
      }
    }

    // Populate vegetation (tall grass, flowers, sugar cane, cacti)
    for (let lz = 0; lz < CHUNK_SIZE; lz++) {
      const wz = baseZ + lz;
      for (let lx = 0; lx < CHUNK_SIZE; lx++) {
        const wx = baseX + lx;
        const idx2 = lz * CHUNK_SIZE + lx;
        const h = heights[idx2];
        const biome = biomes[idx2];
        if (h <= SEA_LEVEL || h >= CHUNK_HEIGHT - 12) continue;

        const topBlock = chunk.getBlockLocal(lx, h, lz);
        const r = hashCoords(wx, wz, this.seed + 911);

        if (topBlock === BlockId.GRASS) {
          if (r < 0.095) {
            chunk.setBlockLocal(lx, h + 1, lz, BlockId.TALL_GRASS);
          } else if (r < 0.107) {
            chunk.setBlockLocal(lx, h + 1, lz, BlockId.POPPY);
          } else if (r < 0.12) {
            chunk.setBlockLocal(lx, h + 1, lz, BlockId.DANDELION);
          }
        } else if (topBlock === BlockId.SAND && biome === 'desert') {
          if (r < 0.008 && lx > 1 && lx < 14 && lz > 1 && lz < 14) {
            const ch = 2 + Math.floor(hashCoords(wx, wz, this.seed + 921) * 3);
            for (let i = 1; i <= ch; i++) {
              chunk.setBlockLocal(lx, h + i, lz, BlockId.CACTUS);
            }
          }
        } else if (topBlock === BlockId.SAND && h === SEA_LEVEL + 1) {
          // Sugar cane near water shores
          if (r < 0.032) {
            const sh = 2 + Math.floor(hashCoords(wx, wz, this.seed + 937) * 2);
            for (let i = 1; i <= sh; i++) {
              chunk.setBlockLocal(lx, h + i, lz, BlockId.SUGAR_CANE);
            }
          }
        }
      }
    }

    // Deterministic cross-chunk trees (evaluate 2-block margin around chunk so canopies never clip!)
    this.populateTrees(chunk, baseX, baseZ);

    // Subtle scenic starter campsite lantern near spawn so warm shader torchlight glows immediately!
    this.placeStarterLandmark(chunk, baseX, baseZ);
  }

  private pickUndergroundBlock(wx: number, y: number, wz: number): number {
    const r = hashCoords3(wx, y, wz, this.seed + 1337);
    // Authentic Minecraft ore Y-distribution
    if (y <= 16 && r < 0.0045) return BlockId.DIAMOND_ORE;
    if (y <= 24 && r < 0.006) return BlockId.EMERALD_ORE;
    if (y <= 32 && r < 0.011) return BlockId.GOLD_ORE;
    if (y <= 54 && r < 0.025) return BlockId.IRON_ORE;
    if (y <= 68 && r < 0.044) return BlockId.COAL_ORE;
    if (r > 0.965 && y < 48) return BlockId.GRAVEL;
    return BlockId.STONE;
  }

  private populateTrees(chunk: Chunk, baseX: number, baseZ: number): void {
    const margin = 3;
    for (let wz = baseZ - margin; wz < baseZ + CHUNK_SIZE + margin; wz++) {
      for (let wx = baseX - margin; wx < baseX + CHUNK_SIZE + margin; wx++) {
        // Keep immediate player spawn pad clear so player doesn't spawn inside a tree
        if (Math.hypot(wx, wz) < 5.5) continue;

        // Deterministic grid jitter ensures trees are nicely spaced and never overlap trunks
        const cellX = Math.floor(wx / 5);
        const cellZ = Math.floor(wz / 5);
        const pickX = cellX * 5 + Math.floor(hashCoords(cellX, cellZ, this.seed + 3001) * 5);
        const pickZ = cellZ * 5 + Math.floor(hashCoords(cellX, cellZ, this.seed + 3007) * 5);
        if (wx !== pickX || wz !== pickZ) continue;

        const sample = this.sampleColumn(wx, wz);
        const h = sample.height;
        if (h <= SEA_LEVEL + 1 || h >= CHUNK_HEIGHT - 14) continue;
        if (sample.biome === 'desert' || sample.biome === 'river') continue;

        const chance = hashCoords(cellX, cellZ, this.seed + 3019);
        let density = 0.22;
        if (sample.biome === 'forest' || sample.biome === 'birch_forest') density = 0.62;
        else if (sample.biome === 'taiga') density = 0.56;
        else if (sample.biome === 'mountains') density = h < SEA_LEVEL + 24 ? 0.28 : 0.0;
        if (chance > density) continue;

        let treeType: 'oak' | 'birch' | 'spruce' = 'oak';
        if (sample.biome === 'birch_forest') {
          treeType = hashCoords(wx, wz, this.seed + 3023) < 0.75 ? 'birch' : 'oak';
        } else if (sample.biome === 'taiga') {
          treeType = 'spruce';
        } else if (sample.biome === 'forest' && hashCoords(wx, wz, this.seed + 3029) < 0.28) {
          treeType = 'birch';
        }

        const trunkH =
          treeType === 'spruce'
            ? 6 + Math.floor(hashCoords(wx, wz, this.seed + 3037) * 3)
            : 4 + Math.floor(hashCoords(wx, wz, this.seed + 3041) * 3);

        this.stampTree(chunk, baseX, baseZ, wx, h + 1, wz, trunkH, treeType);
      }
    }
  }

  private stampTree(
    chunk: Chunk,
    baseX: number,
    baseZ: number,
    tx: number,
    ty: number,
    tz: number,
    trunkH: number,
    type: 'oak' | 'birch' | 'spruce'
  ) {
    const logId = type === 'birch' ? BlockId.BIRCH_LOG : type === 'spruce' ? BlockId.SPRUCE_LOG : BlockId.OAK_LOG;
    const leafId = type === 'birch' ? BlockId.BIRCH_LEAVES : type === 'spruce' ? BlockId.SPRUCE_LEAVES : BlockId.OAK_LEAVES;

    const place = (wx: number, wy: number, wz: number, id: number, overwriteSolid = false) => {
      const lx = wx - baseX;
      const lz = wz - baseZ;
      if (lx < 0 || lx >= CHUNK_SIZE || lz < 0 || lz >= CHUNK_SIZE || wy < 1 || wy >= CHUNK_HEIGHT) return;
      const cur = chunk.getBlockLocal(lx, wy, lz);
      if (overwriteSolid || cur === BlockId.AIR || cur === BlockId.TALL_GRASS || cur === BlockId.POPPY || cur === BlockId.DANDELION) {
        chunk.setBlockLocal(lx, wy, lz, id);
      }
    };

    if (type === 'spruce') {
      // Conical pine tree canopy
      for (let dy = 2; dy <= trunkH + 1; dy++) {
        const radius = dy === trunkH + 1 ? 0 : dy === trunkH ? 1 : ((trunkH - dy) % 2 === 0 ? 2 : 1);
        for (let dz = -radius; dz <= radius; dz++) {
          for (let dx = -radius; dx <= radius; dx++) {
            if (radius === 2 && Math.abs(dx) === 2 && Math.abs(dz) === 2) continue;
            place(tx + dx, ty + dy, tz + dz, leafId);
          }
        }
      }
    } else {
      // Classic Minecraft Oak / Birch canopy
      for (let dy = trunkH - 2; dy <= trunkH + 1; dy++) {
        const radius = dy >= trunkH ? 1 : 2;
        for (let dz = -radius; dz <= radius; dz++) {
          for (let dx = -radius; dx <= radius; dx++) {
            if (Math.abs(dx) === radius && Math.abs(dz) === radius) {
              if (dy >= trunkH || hashCoords3(tx + dx, ty + dy, tz + dz, this.seed + 3089) < 0.45) {
                continue;
              }
            }
            place(tx + dx, ty + dy, tz + dz, leafId);
          }
        }
      }
    }

    // Trunk overwrites inner leaves
    for (let dy = 0; dy < trunkH; dy++) {
      place(tx, ty + dy, tz, logId, true);
    }
  }

  private placeStarterLandmark(chunk: Chunk, baseX: number, baseZ: number): void {
    // Cozy rustic wooden dock & lantern posts near spawn overlooking the shimmering lake
    const posts: [number, number][] = [
      [5, 8],
      [-5, 11],
      [9, 15],
    ];
    for (const [px, pz] of posts) {
      const lx = px - baseX;
      const lz = pz - baseZ;
      if (lx < 0 || lx >= CHUNK_SIZE || lz < 0 || lz >= CHUNK_SIZE) continue;
      const sample = this.sampleColumn(px, pz);
      if (sample.height <= SEA_LEVEL) continue;
      const y = sample.height + 1;
      chunk.setBlockLocal(lx, y, lz, BlockId.STONE_BRICKS);
      chunk.setBlockLocal(lx, y + 1, lz, BlockId.OAK_LOG);
      chunk.setBlockLocal(lx, y + 2, lz, BlockId.OAK_LOG);
      chunk.setBlockLocal(lx, y + 3, lz, BlockId.GLOWSTONE);
    }

    // Little rustic oak cabin / workbench terrace at (-8, 4)
    const cx0 = -9;
    const cz0 = 3;
    if (
      cx0 + 4 >= baseX && cx0 <= baseX + CHUNK_SIZE &&
      cz0 + 4 >= baseZ && cz0 <= baseZ + CHUNK_SIZE
    ) {
      const baseH = this.sampleColumn(cx0 + 2, cz0 + 2).height;
      if (baseH > SEA_LEVEL) {
        const setW = (wx: number, wy: number, wz: number, id: number) => {
          const lx = wx - baseX;
          const lz = wz - baseZ;
          if (lx < 0 || lx >= CHUNK_SIZE || lz < 0 || lz >= CHUNK_SIZE || wy < 1 || wy >= CHUNK_HEIGHT) return;
          chunk.setBlockLocal(lx, wy, lz, id);
        };
        // Stone foundation & oak floor
        for (let dx = 0; dx <= 4; dx++) {
          for (let dz = 0; dz <= 4; dz++) {
            setW(cx0 + dx, baseH - 1, cz0 + dz, BlockId.COBBLESTONE);
            setW(cx0 + dx, baseH, cz0 + dz, BlockId.OAK_PLANKS);
            for (let dy = 1; dy <= 4; dy++) setW(cx0 + dx, baseH + dy, cz0 + dz, BlockId.AIR);
          }
        }
        // Corner pillars
        for (const dx of [0, 4]) {
          for (const dz of [0, 4]) {
            for (let dy = 1; dy <= 3; dy++) setW(cx0 + dx, baseH + dy, cz0 + dz, BlockId.OAK_LOG);
          }
        }
        // Walls + glass windows + front doorway
        for (let i = 1; i <= 3; i++) {
          setW(cx0 + i, baseH + 1, cz0, BlockId.OAK_PLANKS);
          setW(cx0 + i, baseH + 2, cz0, i === 2 ? BlockId.GLASS : BlockId.OAK_PLANKS);
          setW(cx0 + i, baseH + 3, cz0, BlockId.OAK_PLANKS);

          setW(cx0, baseH + 1, cz0 + i, BlockId.OAK_PLANKS);
          setW(cx0, baseH + 2, cz0 + i, i === 2 ? BlockId.GLASS : BlockId.OAK_PLANKS);
          setW(cx0, baseH + 3, cz0 + i, BlockId.OAK_PLANKS);

          setW(cx0 + 4, baseH + 1, cz0 + i, i === 2 ? BlockId.AIR : BlockId.OAK_PLANKS);
          setW(cx0 + 4, baseH + 2, cz0 + i, i === 2 ? BlockId.AIR : BlockId.OAK_PLANKS);
          setW(cx0 + 4, baseH + 3, cz0 + i, BlockId.OAK_PLANKS);

          setW(cx0 + i, baseH + 1, cz0 + 4, BlockId.OAK_PLANKS);
          setW(cx0 + i, baseH + 2, cz0 + 4, i === 2 ? BlockId.GLASS : BlockId.OAK_PLANKS);
          setW(cx0 + i, baseH + 3, cz0 + 4, BlockId.OAK_PLANKS);
        }
        // Roof
        for (let dx = 0; dx <= 4; dx++) {
          for (let dz = 0; dz <= 4; dz++) {
            setW(cx0 + dx, baseH + 4, cz0 + dz, BlockId.SPRUCE_LOG);
          }
        }
        // Cozy interior: crafting table, lit furnace, bookshelf, torch
        setW(cx0 + 1, baseH + 1, cz0 + 1, BlockId.CRAFTING_TABLE);
        setW(cx0 + 1, baseH + 1, cz0 + 2, BlockId.FURNACE);
        setW(cx0 + 1, baseH + 1, cz0 + 3, BlockId.BOOKSHELF);
        setW(cx0 + 3, baseH + 1, cz0 + 1, BlockId.TORCH);
        setW(cx0 + 5, baseH + 1, cz0 + 1, BlockId.TORCH);
      }
    }
  }
}
