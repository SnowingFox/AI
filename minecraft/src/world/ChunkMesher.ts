import * as THREE from 'three';
import { CHUNK_HEIGHT, CHUNK_SIZE } from '../types';
import { ProceduralAtlas } from '../textures/ProceduralAtlas';
import { BlockId, getBlockDef, isBlockSolid, isBlockTransparent } from './blocks';
import { Chunk } from './Chunk';
import { World } from './World';

class GeometryBuilder {
  public positions: number[] = [];
  public normals: number[] = [];
  public tangents: number[] = [];
  public uvs: number[] = [];
  public colors: number[] = [];
  public lights: number[] = []; // [skyLight, blockLight, ao, windOrDepth]
  public indices: number[] = [];
  private vertCount = 0;

  public pushQuad(
    p0: [number, number, number],
    p1: [number, number, number],
    p2: [number, number, number],
    p3: [number, number, number],
    n: [number, number, number],
    t: [number, number, number],
    uv0: [number, number],
    uv1: [number, number],
    uv2: [number, number],
    uv3: [number, number],
    col: [number, number, number],
    l0: [number, number, number, number],
    l1: [number, number, number, number],
    l2: [number, number, number, number],
    l3: [number, number, number, number],
    flipDiag = false
  ) {
    const base = this.vertCount;
    const pts = [p0, p1, p2, p3];
    const uvs = [uv0, uv1, uv2, uv3];
    const ls = [l0, l1, l2, l3];
    for (let i = 0; i < 4; i++) {
      this.positions.push(pts[i][0], pts[i][1], pts[i][2]);
      this.normals.push(n[0], n[1], n[2]);
      this.tangents.push(t[0], t[1], t[2]);
      this.uvs.push(uvs[i][0], uvs[i][1]);
      this.colors.push(col[0], col[1], col[2]);
      this.lights.push(ls[i][0], ls[i][1], ls[i][2], ls[i][3]);
    }
    if (flipDiag) {
      this.indices.push(base + 1, base + 2, base + 3, base + 1, base + 3, base);
    } else {
      this.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
    this.vertCount += 4;
  }

  public toGeometry(): THREE.BufferGeometry | null {
    if (this.vertCount === 0) return null;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(this.positions, 3));
    geom.setAttribute('normal', new THREE.Float32BufferAttribute(this.normals, 3));
    geom.setAttribute('aTangent', new THREE.Float32BufferAttribute(this.tangents, 3));
    geom.setAttribute('uv', new THREE.Float32BufferAttribute(this.uvs, 2));
    geom.setAttribute('color', new THREE.Float32BufferAttribute(this.colors, 3));
    geom.setAttribute('aLight', new THREE.Float32BufferAttribute(this.lights, 4));
    geom.setIndex(this.indices);
    geom.computeBoundingSphere();
    return geom;
  }
}

interface FaceSpec {
  dir: [number, number, number];
  tangent: [number, number, number];
  key: 'top' | 'bottom' | 'north' | 'south' | 'east' | 'west';
  // 4 corners in [0..1] cube space, CCW from outside
  corners: [
    [number, number, number],
    [number, number, number],
    [number, number, number],
    [number, number, number],
  ];
  // 4 UV corner indices matching (u0/u1, v0/v1)
  uvMap: [number, number][]; // [uIdx, vIdx] 0=>min, 1=>max
  // Two orthogonal axes in the face plane for AO & smooth lighting sampling
  uAxis: [number, number, number];
  vAxis: [number, number, number];
  // Signs of (uAxis, vAxis) for each of the 4 corners (-1 or +1)
  cornerSigns: [number, number][];
}

const CUBE_FACES: FaceSpec[] = [
  {
    key: 'top',
    dir: [0, 1, 0],
    tangent: [1, 0, 0],
    corners: [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]],
    uvMap: [[0, 1], [1, 1], [1, 0], [0, 0]],
    uAxis: [1, 0, 0],
    vAxis: [0, 0, 1],
    cornerSigns: [[-1, 1], [1, 1], [1, -1], [-1, -1]],
  },
  {
    key: 'bottom',
    dir: [0, -1, 0],
    tangent: [1, 0, 0],
    corners: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    uvMap: [[0, 0], [1, 0], [1, 1], [0, 1]],
    uAxis: [1, 0, 0],
    vAxis: [0, 0, 1],
    cornerSigns: [[-1, -1], [1, -1], [1, 1], [-1, 1]],
  },
  {
    key: 'south',
    dir: [0, 0, 1],
    tangent: [1, 0, 0],
    corners: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    uvMap: [[0, 1], [1, 1], [1, 0], [0, 0]],
    uAxis: [1, 0, 0],
    vAxis: [0, 1, 0],
    cornerSigns: [[-1, -1], [1, -1], [1, 1], [-1, 1]],
  },
  {
    key: 'north',
    dir: [0, 0, -1],
    tangent: [-1, 0, 0],
    corners: [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
    uvMap: [[0, 1], [1, 1], [1, 0], [0, 0]],
    uAxis: [1, 0, 0],
    vAxis: [0, 1, 0],
    cornerSigns: [[1, -1], [-1, -1], [-1, 1], [1, 1]],
  },
  {
    key: 'east',
    dir: [1, 0, 0],
    tangent: [0, 0, -1],
    corners: [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
    uvMap: [[0, 1], [1, 1], [1, 0], [0, 0]],
    uAxis: [0, 0, 1],
    vAxis: [0, 1, 0],
    cornerSigns: [[1, -1], [-1, -1], [-1, 1], [1, 1]],
  },
  {
    key: 'west',
    dir: [-1, 0, 0],
    tangent: [0, 0, 1],
    corners: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    uvMap: [[0, 1], [1, 1], [1, 0], [0, 0]],
    uAxis: [0, 0, 1],
    vAxis: [0, 1, 0],
    cornerSigns: [[-1, -1], [1, -1], [1, 1], [-1, 1]],
  },
];

const AO_CURVE = [0.32, 0.56, 0.78, 1.0];

export interface BuiltChunkGeometries {
  opaque: THREE.BufferGeometry | null;
  foliage: THREE.BufferGeometry | null;
  translucent: THREE.BufferGeometry | null;
  water: THREE.BufferGeometry | null;
}

export function buildChunkGeometries(
  chunk: Chunk,
  world: World,
  atlas: ProceduralAtlas
): BuiltChunkGeometries {
  const opaque = new GeometryBuilder();
  const foliage = new GeometryBuilder();
  const translucent = new GeometryBuilder();
  const water = new GeometryBuilder();

  const baseX = chunk.cx * CHUNK_SIZE;
  const baseZ = chunk.cz * CHUNK_SIZE;

  // Cache biome colors per column for fast tinting
  const grassCols: [number, number, number][] = new Array(CHUNK_SIZE * CHUNK_SIZE);
  const leafCols: [number, number, number][] = new Array(CHUNK_SIZE * CHUNK_SIZE);
  for (let lz = 0; lz < CHUNK_SIZE; lz++) {
    for (let lx = 0; lx < CHUNK_SIZE; lx++) {
      const s = world.worldGen.sampleColumn(baseX + lx, baseZ + lz);
      grassCols[lz * CHUNK_SIZE + lx] = s.grassColor;
      leafCols[lz * CHUNK_SIZE + lx] = s.foliageColor;
    }
  }

  const isOccluder = (wx: number, wy: number, wz: number): number => {
    const id = world.getBlock(wx, wy, wz);
    if (id === BlockId.AIR) return 0;
    const d = getBlockDef(id);
    return d.solid && !d.transparent ? 1 : 0;
  };

  for (let y = 0; y < CHUNK_HEIGHT; y++) {
    for (let lz = 0; lz < CHUNK_SIZE; lz++) {
      const wz = baseZ + lz;
      for (let lx = 0; lx < CHUNK_SIZE; lx++) {
        const blockId = chunk.getBlockLocal(lx, y, lz);
        if (blockId === BlockId.AIR) continue;

        const wx = baseX + lx;
        const def = getBlockDef(blockId);
        const colIdx = lz * CHUNK_SIZE + lx;

        if (def.shape === 'water') {
          emitWater(water, world, atlas, wx, y, wz);
          continue;
        }

        if (def.shape === 'cross') {
          const tint: [number, number, number] = def.tintAll ? grassCols[colIdx] : [1, 1, 1];
          emitCross(foliage, world, atlas, wx, y, wz, def.tiles.north, tint, def.windSway ? 1 : 0);
          continue;
        }

        if (def.shape === 'torch') {
          emitTorch(foliage, world, atlas, wx, y, wz, def.tiles.north);
          continue;
        }

        const isFoliageCube = blockId === BlockId.OAK_LEAVES || blockId === BlockId.BIRCH_LEAVES || blockId === BlockId.SPRUCE_LEAVES;
        const isGlassOrIce = blockId === BlockId.GLASS || blockId === BlockId.ICE;
        const target = isFoliageCube ? foliage : isGlassOrIce ? translucent : opaque;
        const inset = def.shape === 'cactus' ? 0.0625 : 0;

        for (let f = 0; f < 6; f++) {
          const face = CUBE_FACES[f];
          const nx = wx + face.dir[0];
          const ny = y + face.dir[1];
          const nz = wz + face.dir[2];

          const neighborId = world.getBlock(nx, ny, nz);

          // Culling rules matching Minecraft 1:1
          if (neighborId !== BlockId.AIR) {
            if (isGlassOrIce) {
              if (neighborId === blockId) continue;
              if (!isBlockTransparent(neighborId)) continue;
            } else if (isFoliageCube) {
              // Keep leaves lush (render internal leaf faces slightly less aggressively or show when neighbor transparent)
              if (!isBlockTransparent(neighborId)) continue;
            } else {
              if (!isBlockTransparent(neighborId) && getBlockDef(neighborId).shape === 'cube') continue;
            }
          }

          const tileIdx = def.tiles[face.key];
          const uv = atlas.getTileUV(tileIdx);
          const uArr = [uv.u0, uv.u1];
          const vArr = [uv.v0, uv.v1];

          let tint: [number, number, number] = [1, 1, 1];
          if (def.tintAll) {
            tint = leafCols[colIdx];
          } else if (def.tintTop && face.key === 'top') {
            tint = grassCols[colIdx];
          }

          const windVal = def.windSway ? 0.55 : 0.0;

          // Compute exact Minecraft 4-corner smooth AO + smooth light interpolation
          const vertLights: [number, number, number, number][] = [];
          const aoLevels: number[] = [];
          for (let c = 0; c < 4; c++) {
            const [su, sv] = face.cornerSigns[c];
            const ux = face.uAxis[0] * su;
            const uy = face.uAxis[1] * su;
            const uz = face.uAxis[2] * su;
            const vx = face.vAxis[0] * sv;
            const vy = face.vAxis[1] * sv;
            const vz = face.vAxis[2] * sv;

            const s1 = isOccluder(nx + ux, ny + uy, nz + uz);
            const s2 = isOccluder(nx + vx, ny + vy, nz + vz);
            const cr = s1 && s2 ? 1 : isOccluder(nx + ux + vx, ny + uy + vy, nz + uz + vz);
            const aoIdx = s1 && s2 ? 0 : 3 - (s1 + s2 + cr);
            aoLevels.push(aoIdx);

            // Smooth lighting average over 4 cells around vertex
            let skySum = 0;
            let blkSum = 0;
            let count = 0;
            const samplePts: [number, number, number][] = [
              [nx, ny, nz],
              [nx + ux, ny + uy, nz + uz],
              [nx + vx, ny + vy, nz + vz],
              [nx + ux + vx, ny + uy + vy, nz + uz + vz],
            ];
            for (let k = 0; k < 4; k++) {
              const [sx, sy, sz] = samplePts[k];
              if (k === 0 || !isOccluder(sx, sy, sz)) {
                skySum += world.getSkyLight(sx, sy, sz);
                blkSum += world.getBlockLight(sx, sy, sz);
                count++;
              }
            }
            const avgSky = count > 0 ? skySum / (count * 15) : 0;
            const avgBlk = count > 0 ? blkSum / (count * 15) : 0;
            const selfEmit = def.lightLevel / 15;
            vertLights.push([
              avgSky,
              Math.max(avgBlk, selfEmit),
              AO_CURVE[aoIdx],
              windVal,
            ]);
          }

          // Flip quad split diagonal to eliminate AO anisotropy (classic Minecraft fix!)
          const flip = aoLevels[0] + aoLevels[2] < aoLevels[1] + aoLevels[3];

          const pts: [number, number, number][] = [];
          for (let c = 0; c < 4; c++) {
            let cx = face.corners[c][0];
            let cy = face.corners[c][1];
            let cz = face.corners[c][2];
            if (inset > 0 && face.key !== 'top' && face.key !== 'bottom') {
              if (face.key === 'east') cx -= inset;
              if (face.key === 'west') cx += inset;
              if (face.key === 'south') cz -= inset;
              if (face.key === 'north') cz += inset;
            }
            pts.push([wx + cx, y + cy, wz + cz]);
          }

          target.pushQuad(
            pts[0], pts[1], pts[2], pts[3],
            face.dir,
            face.tangent,
            [uArr[face.uvMap[0][0]], vArr[face.uvMap[0][1]]],
            [uArr[face.uvMap[1][0]], vArr[face.uvMap[1][1]]],
            [uArr[face.uvMap[2][0]], vArr[face.uvMap[2][1]]],
            [uArr[face.uvMap[3][0]], vArr[face.uvMap[3][1]]],
            tint,
            vertLights[0], vertLights[1], vertLights[2], vertLights[3],
            flip
          );
        }
      }
    }
  }

  return {
    opaque: opaque.toGeometry(),
    foliage: foliage.toGeometry(),
    translucent: translucent.toGeometry(),
    water: water.toGeometry(),
  };
}

function emitWater(
  builder: GeometryBuilder,
  world: World,
  atlas: ProceduralAtlas,
  wx: number,
  y: number,
  wz: number
) {
  const above = world.getBlock(wx, y + 1, wz);
  const topY = above === BlockId.WATER ? 1.0 : 0.88;
  const uv = atlas.getTileUV(getBlockDef(BlockId.WATER).tiles.top);

  // Measure water depth beneath this column (up to 10 blocks) for realistic turquoise-to-deep-blue gradient
  let depth = 1;
  while (depth < 10 && world.getBlock(wx, y - depth, wz) === BlockId.WATER) {
    depth++;
  }
  const depthNorm = Math.min(1, depth / 8);

  const sky = world.getSkyLight(wx, y + 1, wz) / 15;
  const blk = world.getBlockLight(wx, y + 1, wz) / 15;
  const l: [number, number, number, number] = [Math.max(sky, 0.75), blk, 1.0, depthNorm];

  // Top face of water
  if (above !== BlockId.WATER) {
    builder.pushQuad(
      [wx, y + topY, wz + 1],
      [wx + 1, y + topY, wz + 1],
      [wx + 1, y + topY, wz],
      [wx, y + topY, wz],
      [0, 1, 0],
      [1, 0, 0],
      [uv.u0, uv.v1], [uv.u1, uv.v1], [uv.u1, uv.v0], [uv.u0, uv.v0],
      [1, 1, 1],
      l, l, l, l
    );
  }

  // Side faces where water borders air (e.g. waterfalls or exposed edges)
  for (let f = 2; f < 6; f++) {
    const face = CUBE_FACES[f];
    const nb = world.getBlock(wx + face.dir[0], y + face.dir[1], wz + face.dir[2]);
    if (nb === BlockId.WATER || (isBlockSolid(nb) && !isBlockTransparent(nb))) continue;

    const pts: [number, number, number][] = face.corners.map(([cx, cy, cz]) => [
      wx + cx,
      y + (cy > 0.5 ? topY : 0),
      wz + cz,
    ]);
    builder.pushQuad(
      pts[0], pts[1], pts[2], pts[3],
      face.dir,
      face.tangent,
      [uv.u0, uv.v1], [uv.u1, uv.v1], [uv.u1, uv.v0], [uv.u0, uv.v0],
      [1, 1, 1],
      l, l, l, l
    );
  }
}

function emitCross(
  builder: GeometryBuilder,
  world: World,
  atlas: ProceduralAtlas,
  wx: number,
  y: number,
  wz: number,
  tileIdx: number,
  tint: [number, number, number],
  wind: number
) {
  const uv = atlas.getTileUV(tileIdx);
  const sky = world.getSkyLight(wx, y, wz) / 15;
  const blk = world.getBlockLight(wx, y, wz) / 15;
  const botL: [number, number, number, number] = [sky, blk, 0.78, 0.0];
  const topL: [number, number, number, number] = [sky, blk, 1.0, wind];

  const pad = 0.14;
  // Two intersecting diagonal planes (front & back sides for double-sided cross foliage)
  const diag1: [[number, number, number], [number, number, number], [number, number, number], [number, number, number]] = [
    [wx + pad, y, wz + pad],
    [wx + 1 - pad, y, wz + 1 - pad],
    [wx + 1 - pad, y + 1, wz + 1 - pad],
    [wx + pad, y + 1, wz + pad],
  ];
  const diag2: [[number, number, number], [number, number, number], [number, number, number], [number, number, number]] = [
    [wx + 1 - pad, y, wz + pad],
    [wx + pad, y, wz + 1 - pad],
    [wx + pad, y + 1, wz + 1 - pad],
    [wx + 1 - pad, y + 1, wz + pad],
  ];

  builder.pushQuad(
    diag1[0], diag1[1], diag1[2], diag1[3],
    [0, 1, 0], [0.707, 0, 0.707],
    [uv.u0, uv.v1], [uv.u1, uv.v1], [uv.u1, uv.v0], [uv.u0, uv.v0],
    tint, botL, botL, topL, topL
  );
  builder.pushQuad(
    diag1[1], diag1[0], diag1[3], diag1[2],
    [0, 1, 0], [-0.707, 0, -0.707],
    [uv.u1, uv.v1], [uv.u0, uv.v1], [uv.u0, uv.v0], [uv.u1, uv.v0],
    tint, botL, botL, topL, topL
  );
  builder.pushQuad(
    diag2[0], diag2[1], diag2[2], diag2[3],
    [0, 1, 0], [-0.707, 0, 0.707],
    [uv.u0, uv.v1], [uv.u1, uv.v1], [uv.u1, uv.v0], [uv.u0, uv.v0],
    tint, botL, botL, topL, topL
  );
  builder.pushQuad(
    diag2[1], diag2[0], diag2[3], diag2[2],
    [0, 1, 0], [0.707, 0, -0.707],
    [uv.u1, uv.v1], [uv.u0, uv.v1], [uv.u0, uv.v0], [uv.u1, uv.v0],
    tint, botL, botL, topL, topL
  );
}

function emitTorch(
  builder: GeometryBuilder,
  _world: World,
  atlas: ProceduralAtlas,
  wx: number,
  y: number,
  wz: number,
  tileIdx: number
) {
  const uv = atlas.getTileUV(tileIdx);
  const x0 = wx + 0.43;
  const x1 = wx + 0.57;
  const z0 = wz + 0.43;
  const z1 = wz + 0.57;
  const y0 = y;
  const y1 = y + 0.65;
  const l: [number, number, number, number] = [1.0, 1.0, 1.0, 0.0];

  // 4 sides + top cap of 3D torch post
  builder.pushQuad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1], [1, 0, 0], [uv.u0, uv.v1], [uv.u1, uv.v1], [uv.u1, uv.v0], [uv.u0, uv.v0], [1, 1, 1], l, l, l, l);
  builder.pushQuad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [0, 0, -1], [-1, 0, 0], [uv.u0, uv.v1], [uv.u1, uv.v1], [uv.u1, uv.v0], [uv.u0, uv.v0], [1, 1, 1], l, l, l, l);
  builder.pushQuad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [1, 0, 0], [0, 0, -1], [uv.u0, uv.v1], [uv.u1, uv.v1], [uv.u1, uv.v0], [uv.u0, uv.v0], [1, 1, 1], l, l, l, l);
  builder.pushQuad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [-1, 0, 0], [0, 0, 1], [uv.u0, uv.v1], [uv.u1, uv.v1], [uv.u1, uv.v0], [uv.u0, uv.v0], [1, 1, 1], l, l, l, l);
  // Top glowing ember tip
  const tipV1 = uv.v0 + (uv.v1 - uv.v0) * 0.25;
  builder.pushQuad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], [0, 1, 0], [1, 0, 0], [uv.u0, tipV1], [uv.u1, tipV1], [uv.u1, uv.v0], [uv.u0, uv.v0], [1, 1, 1], l, l, l, l);
}
