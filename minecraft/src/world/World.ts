import * as THREE from 'three';
import { BiomeType, CHUNK_HEIGHT, CHUNK_SIZE } from '../types';
import { ProceduralAtlas } from '../textures/ProceduralAtlas';
import { BlockId, getBlockDef, isBlockSolid, isBlockTransparent } from './blocks';
import { Chunk } from './Chunk';
import { buildChunkGeometries } from './ChunkMesher';
import { WorldGen } from './WorldGen';

export interface WorldMaterials {
  opaque: THREE.Material;
  foliage: THREE.Material;
  translucent: THREE.Material;
  water: THREE.Material;
}

export class World {
  public readonly scene: THREE.Scene;
  public readonly atlas: ProceduralAtlas;
  public readonly materials: WorldMaterials;
  public worldGen: WorldGen;
  public readonly chunks: Map<string, Chunk> = new Map();
  public readonly edits: Map<string, number> = new Map();
  public readonly torchPositions: Map<string, THREE.Vector3> = new Map();

  private renderDistance = 6;
  private lastPlayerCx = Number.NaN;
  private lastPlayerCz = Number.NaN;

  constructor(
    scene: THREE.Scene,
    atlas: ProceduralAtlas,
    materials: WorldMaterials,
    seed: number,
    initialEdits?: Record<string, number>
  ) {
    this.scene = scene;
    this.atlas = atlas;
    this.materials = materials;
    this.worldGen = new WorldGen(seed);
    if (initialEdits) {
      for (const [k, v] of Object.entries(initialEdits)) {
        this.edits.set(k, v);
      }
    }
  }

  public setRenderDistance(radius: number): void {
    this.renderDistance = Math.max(3, Math.min(10, Math.round(radius)));
    this.lastPlayerCx = Number.NaN;
  }

  public resetWorld(newSeed: number): void {
    for (const chunk of this.chunks.values()) {
      chunk.disposeMeshes(this.scene);
    }
    this.chunks.clear();
    this.edits.clear();
    this.torchPositions.clear();
    this.worldGen = new WorldGen(newSeed);
    this.lastPlayerCx = Number.NaN;
    this.lastPlayerCz = Number.NaN;
  }

  public static chunkCoord(w: number): number {
    return Math.floor(w / CHUNK_SIZE);
  }

  public static localCoord(w: number): number {
    return ((w % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
  }

  public getChunk(cx: number, cz: number): Chunk | undefined {
    return this.chunks.get(`${cx},${cz}`);
  }

  public getBlock(wx: number, wy: number, wz: number): number {
    if (wy < 0 || wy >= CHUNK_HEIGHT) return BlockId.AIR;
    const cx = World.chunkCoord(wx);
    const cz = World.chunkCoord(wz);
    const chunk = this.chunks.get(`${cx},${cz}`);
    if (!chunk) return BlockId.AIR;
    return chunk.getBlockLocal(World.localCoord(wx), wy, World.localCoord(wz));
  }

  public setBlock(wx: number, wy: number, wz: number, id: number, recordEdit = true): boolean {
    if (wy < 1 || wy >= CHUNK_HEIGHT - 1) return false;
    const cx = World.chunkCoord(wx);
    const cz = World.chunkCoord(wz);
    const chunk = this.chunks.get(`${cx},${cz}`);
    if (!chunk) return false;

    const lx = World.localCoord(wx);
    const lz = World.localCoord(wz);
    const prev = chunk.getBlockLocal(lx, wy, lz);
    if (prev === id) return false;

    chunk.setBlockLocal(lx, wy, lz, id);
    const key = `${wx},${wy},${wz}`;
    if (recordEdit) {
      this.edits.set(key, id);
    }

    if (id === BlockId.TORCH) {
      this.torchPositions.set(key, new THREE.Vector3(wx + 0.5, wy + 0.72, wz + 0.5));
    } else {
      this.torchPositions.delete(key);
    }

    // Mark chunk and any bordering neighbor chunks dirty for lighting + mesh rebuild
    this.markDirtyAround(cx, cz, lx, lz);
    return true;
  }

  private markDirtyAround(cx: number, cz: number, lx: number, lz: number): void {
    const touch = (tcx: number, tcz: number) => {
      const c = this.chunks.get(`${tcx},${tcz}`);
      if (c) {
        c.dirtyLight = true;
        c.dirtyMesh = true;
      }
    };
    touch(cx, cz);
    // Light can propagate up to 15 blocks, so mark immediate 3x3 chunk neighborhood for lighting if near border
    if (lx <= 1) touch(cx - 1, cz);
    if (lx >= CHUNK_SIZE - 2) touch(cx + 1, cz);
    if (lz <= 1) touch(cx, cz - 1);
    if (lz >= CHUNK_SIZE - 2) touch(cx, cz + 1);
    // Also propagate block light changes to immediate 4 neighbors always
    touch(cx - 1, cz);
    touch(cx + 1, cz);
    touch(cx, cz - 1);
    touch(cx, cz + 1);
  }

  public getSkyLight(wx: number, wy: number, wz: number): number {
    if (wy < 0) return 0;
    if (wy >= CHUNK_HEIGHT) return 15;
    const cx = World.chunkCoord(wx);
    const cz = World.chunkCoord(wz);
    const chunk = this.chunks.get(`${cx},${cz}`);
    if (!chunk) return 15;
    return chunk.getSkyLightLocal(World.localCoord(wx), wy, World.localCoord(wz));
  }

  public getBlockLight(wx: number, wy: number, wz: number): number {
    if (wy < 0 || wy >= CHUNK_HEIGHT) return 0;
    const cx = World.chunkCoord(wx);
    const cz = World.chunkCoord(wz);
    const chunk = this.chunks.get(`${cx},${cz}`);
    if (!chunk) return 0;
    return chunk.getBlockLightLocal(World.localCoord(wx), wy, World.localCoord(wz));
  }

  public getBiomeAt(wx: number, wz: number): BiomeType {
    return this.worldGen.sampleColumn(Math.floor(wx), Math.floor(wz)).biome;
  }

  public getSurfaceY(wx: number, wz: number): number {
    const ix = Math.floor(wx);
    const iz = Math.floor(wz);
    for (let y = CHUNK_HEIGHT - 2; y >= 1; y--) {
      const id = this.getBlock(ix, y, iz);
      if (isBlockSolid(id)) return y + 1;
    }
    return this.worldGen.sampleColumn(ix, iz).height + 1;
  }

  // Generate an initial chunk ring around spawn synchronously so first frame is immediately rich
  public generateInitialArea(centerX: number, centerZ: number, radius: number): void {
    const pcx = World.chunkCoord(centerX);
    const pcz = World.chunkCoord(centerZ);
    const coords: [number, number, number][] = [];
    for (let dz = -radius; dz <= radius; dz++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const d2 = dx * dx + dz * dz;
        if (d2 <= radius * radius + 1) {
          coords.push([pcx + dx, pcz + dz, d2]);
        }
      }
    }
    coords.sort((a, b) => a[2] - b[2]);

    for (const [cx, cz] of coords) {
      this.createChunk(cx, cz);
    }
    for (const [cx, cz] of coords) {
      const c = this.chunks.get(`${cx},${cz}`);
      if (c) this.computeChunkLighting(c);
    }
    for (const [cx, cz] of coords) {
      const c = this.chunks.get(`${cx},${cz}`);
      if (c) this.rebuildChunkMesh(c);
    }
    this.lastPlayerCx = pcx;
    this.lastPlayerCz = pcz;
  }

  private createChunk(cx: number, cz: number): Chunk {
    const key = `${cx},${cz}`;
    const existing = this.chunks.get(key);
    if (existing) return existing;

    const chunk = new Chunk(cx, cz);
    this.worldGen.generateChunk(chunk);

    // Replay saved player edits inside this chunk
    const baseX = cx * CHUNK_SIZE;
    const baseZ = cz * CHUNK_SIZE;
    for (const [editKey, blockId] of this.edits.entries()) {
      const [sx, sy, sz] = editKey.split(',').map(Number);
      const lx = sx - baseX;
      const lz = sz - baseZ;
      if (lx >= 0 && lx < CHUNK_SIZE && lz >= 0 && lz < CHUNK_SIZE && sy >= 0 && sy < CHUNK_HEIGHT) {
        chunk.setBlockLocal(lx, sy, lz, blockId);
      }
    }

    // Track torches in this chunk for flame particles
    for (let y = 0; y < CHUNK_HEIGHT; y++) {
      for (let lz = 0; lz < CHUNK_SIZE; lz++) {
        for (let lx = 0; lx < CHUNK_SIZE; lx++) {
          if (chunk.getBlockLocal(lx, y, lz) === BlockId.TORCH) {
            const wx = baseX + lx;
            const wz = baseZ + lz;
            this.torchPositions.set(`${wx},${y},${wz}`, new THREE.Vector3(wx + 0.5, y + 0.72, wz + 0.5));
          }
        }
      }
    }

    this.chunks.set(key, chunk);

    // Mark adjacent chunks to refresh edge culling & light
    const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    for (const [dx, dz] of dirs) {
      const nb = this.chunks.get(`${cx + dx},${cz + dz}`);
      if (nb) {
        nb.dirtyMesh = true;
      }
    }
    return chunk;
  }

  // Incremental streaming & dirty mesh updates per frame (keeps FPS smooth)
  public updateAroundPlayer(px: number, pz: number, maxNewPerFrame = 2, maxRebuildPerFrame = 3): void {
    const pcx = World.chunkCoord(px);
    const pcz = World.chunkCoord(pz);
    const r = this.renderDistance;

    // 1. Unload distant chunks
    if (pcx !== this.lastPlayerCx || pcz !== this.lastPlayerCz) {
      this.lastPlayerCx = pcx;
      this.lastPlayerCz = pcz;
      const unloadR2 = (r + 2) * (r + 2);
      for (const [key, chunk] of this.chunks.entries()) {
        const dx = chunk.cx - pcx;
        const dz = chunk.cz - pcz;
        if (dx * dx + dz * dz > unloadR2) {
          chunk.disposeMeshes(this.scene);
          this.chunks.delete(key);
        }
      }
    }

    // 2. Load closest missing chunks
    let loaded = 0;
    const missing: [number, number, number][] = [];
    for (let dz = -r; dz <= r; dz++) {
      for (let dx = -r; dx <= r; dx++) {
        const d2 = dx * dx + dz * dz;
        if (d2 > r * r + 1) continue;
        const cx = pcx + dx;
        const cz = pcz + dz;
        if (!this.chunks.has(`${cx},${cz}`)) {
          missing.push([cx, cz, d2]);
        }
      }
    }
    missing.sort((a, b) => a[2] - b[2]);
    for (const [cx, cz] of missing) {
      if (loaded >= maxNewPerFrame) break;
      this.createChunk(cx, cz);
      loaded++;
    }

    // 3. Recompute lighting + meshes for closest dirty chunks
    const dirty: [Chunk, number][] = [];
    for (const chunk of this.chunks.values()) {
      if (chunk.dirtyLight || chunk.dirtyMesh) {
        const dx = chunk.cx - pcx;
        const dz = chunk.cz - pcz;
        dirty.push([chunk, dx * dx + dz * dz]);
      }
    }
    dirty.sort((a, b) => a[1] - b[1]);

    let rebuilt = 0;
    for (const [chunk] of dirty) {
      if (rebuilt >= maxRebuildPerFrame) break;
      if (chunk.dirtyLight) {
        this.computeChunkLighting(chunk);
      }
      if (chunk.dirtyMesh) {
        this.rebuildChunkMesh(chunk);
      }
      rebuilt++;
    }
  }

  // Fast Minecraft-authentic sky light + emissive block light BFS flood-fill
  public computeChunkLighting(chunk: Chunk): void {
    const blocks = chunk.blocks;
    const sky = chunk.skyLight;
    const blk = chunk.blockLight;
    sky.fill(0);
    blk.fill(0);

    const skyQueue: number[] = [];
    const blkQueue: number[] = [];

    // Pass 1: Vertical sky light shafts & block light emitters
    for (let lz = 0; lz < CHUNK_SIZE; lz++) {
      for (let lx = 0; lx < CHUNK_SIZE; lx++) {
        let light = 15;
        for (let y = CHUNK_HEIGHT - 1; y >= 0; y--) {
          const idx = (y << 8) | (lz << 4) | lx;
          const id = blocks[idx];
          const def = getBlockDef(id);

          if (def.lightLevel > 0) {
            blk[idx] = def.lightLevel;
            blkQueue.push(idx);
          }

          if (id === BlockId.AIR || def.shape === 'cross' || def.shape === 'torch' || id === BlockId.GLASS) {
            sky[idx] = light;
            if (light === 15) skyQueue.push(idx);
          } else if (id === BlockId.WATER || id === BlockId.ICE || def.windSway) {
            light = Math.max(0, light - 2);
            sky[idx] = light;
            if (light > 1) skyQueue.push(idx);
          } else {
            light = 0;
          }
        }
      }
    }

    // Seed border block lights from immediate neighbor chunks so torch/glowstone light seamlessly crosses chunk seams
    const baseX = chunk.cx * CHUNK_SIZE;
    const baseZ = chunk.cz * CHUNK_SIZE;
    const seedFromNeighbor = (lx: number, lz: number, wx: number, wz: number) => {
      for (let y = 0; y < CHUNK_HEIGHT; y++) {
        const nbBlk = this.getBlockLight(wx, y, wz);
        const idx = (y << 8) | (lz << 4) | lx;
        const id = blocks[idx];
        if (nbBlk > 1 && isBlockTransparent(id) && nbBlk - 1 > blk[idx]) {
          blk[idx] = nbBlk - 1;
          blkQueue.push(idx);
        }
        const nbSky = this.getSkyLight(wx, y, wz);
        if (nbSky > 1 && isBlockTransparent(id) && nbSky - 1 > sky[idx]) {
          sky[idx] = nbSky - 1;
          skyQueue.push(idx);
        }
      }
    };
    for (let i = 0; i < CHUNK_SIZE; i++) {
      seedFromNeighbor(0, i, baseX - 1, baseZ + i);
      seedFromNeighbor(15, i, baseX + 16, baseZ + i);
      seedFromNeighbor(i, 0, baseX + i, baseZ - 1);
      seedFromNeighbor(i, 15, baseX + i, baseZ + 16);
    }

    // Pass 2: BFS horizontal/vertical sky light spread into caves & under trees
    let head = 0;
    while (head < skyQueue.length) {
      const cur = skyQueue[head++];
      const curL = sky[cur];
      if (curL <= 1) continue;
      const nextL = curL - 1;
      const y = cur >> 8;
      const lz = (cur >> 4) & 15;
      const lx = cur & 15;

      const trySpread = (nx: number, ny: number, nz: number) => {
        if (nx < 0 || nx >= 16 || nz < 0 || nz >= 16 || ny < 0 || ny >= CHUNK_HEIGHT) return;
        const ni = (ny << 8) | (nz << 4) | nx;
        if (sky[ni] >= nextL) return;
        if (!isBlockTransparent(blocks[ni])) return;
        sky[ni] = nextL;
        skyQueue.push(ni);
      };

      trySpread(lx + 1, y, lz);
      trySpread(lx - 1, y, lz);
      trySpread(lx, y, lz + 1);
      trySpread(lx, y, lz - 1);
      trySpread(lx, y + 1, lz);
      trySpread(lx, y - 1, lz);
    }

    // Pass 3: BFS warm emissive block light spread
    head = 0;
    while (head < blkQueue.length) {
      const cur = blkQueue[head++];
      const curL = blk[cur];
      if (curL <= 1) continue;
      const nextL = curL - 1;
      const y = cur >> 8;
      const lz = (cur >> 4) & 15;
      const lx = cur & 15;

      const trySpread = (nx: number, ny: number, nz: number) => {
        if (nx < 0 || nx >= 16 || nz < 0 || nz >= 16 || ny < 0 || ny >= CHUNK_HEIGHT) return;
        const ni = (ny << 8) | (nz << 4) | nx;
        if (blk[ni] >= nextL) return;
        if (!isBlockTransparent(blocks[ni])) return;
        blk[ni] = nextL;
        blkQueue.push(ni);
      };

      trySpread(lx + 1, y, lz);
      trySpread(lx - 1, y, lz);
      trySpread(lx, y, lz + 1);
      trySpread(lx, y, lz - 1);
      trySpread(lx, y + 1, lz);
      trySpread(lx, y - 1, lz);
    }

    chunk.dirtyLight = false;
  }

  public rebuildChunkMesh(chunk: Chunk): void {
    chunk.disposeMeshes(this.scene);
    const geoms = buildChunkGeometries(chunk, this, this.atlas);

    if (geoms.opaque) {
      const mesh = new THREE.Mesh(geoms.opaque, this.materials.opaque);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.frustumCulled = true;
      this.scene.add(mesh);
      chunk.opaqueMesh = mesh;
    }

    if (geoms.foliage) {
      const mesh = new THREE.Mesh(geoms.foliage, this.materials.foliage);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.frustumCulled = true;
      this.scene.add(mesh);
      chunk.foliageMesh = mesh;
    }

    if (geoms.translucent) {
      const mesh = new THREE.Mesh(geoms.translucent, this.materials.translucent);
      mesh.castShadow = false;
      mesh.receiveShadow = true;
      mesh.renderOrder = 2;
      mesh.frustumCulled = true;
      this.scene.add(mesh);
      chunk.translucentMesh = mesh;
    }

    if (geoms.water) {
      const mesh = new THREE.Mesh(geoms.water, this.materials.water);
      mesh.castShadow = false;
      mesh.receiveShadow = true;
      mesh.renderOrder = 3;
      mesh.frustumCulled = true;
      this.scene.add(mesh);
      chunk.waterMesh = mesh;
    }

    chunk.dirtyMesh = false;
  }
}
