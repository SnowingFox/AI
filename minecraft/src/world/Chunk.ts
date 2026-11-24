import * as THREE from 'three';
import { CHUNK_HEIGHT, CHUNK_SIZE } from '../types';
import { BlockId } from './blocks';

export class Chunk {
  public readonly cx: number;
  public readonly cz: number;
  public readonly key: string;
  public readonly blocks: Uint8Array;
  public readonly skyLight: Uint8Array;
  public readonly blockLight: Uint8Array;

  public opaqueMesh: THREE.Mesh | null = null;
  public foliageMesh: THREE.Mesh | null = null;
  public translucentMesh: THREE.Mesh | null = null;
  public waterMesh: THREE.Mesh | null = null;

  public dirtyLight = true;
  public dirtyMesh = true;

  constructor(cx: number, cz: number) {
    this.cx = cx;
    this.cz = cz;
    this.key = `${cx},${cz}`;
    const len = CHUNK_SIZE * CHUNK_HEIGHT * CHUNK_SIZE;
    this.blocks = new Uint8Array(len);
    this.skyLight = new Uint8Array(len);
    this.blockLight = new Uint8Array(len);
  }

  public static index(lx: number, y: number, lz: number): number {
    return (y << 8) | (lz << 4) | lx;
  }

  public getBlockLocal(lx: number, y: number, lz: number): number {
    if (y < 0 || y >= CHUNK_HEIGHT) return BlockId.AIR;
    return this.blocks[(y << 8) | (lz << 4) | lx];
  }

  public setBlockLocal(lx: number, y: number, lz: number, id: number): void {
    if (y < 0 || y >= CHUNK_HEIGHT) return;
    this.blocks[(y << 8) | (lz << 4) | lx] = id;
  }

  public getSkyLightLocal(lx: number, y: number, lz: number): number {
    if (y < 0) return 0;
    if (y >= CHUNK_HEIGHT) return 15;
    return this.skyLight[(y << 8) | (lz << 4) | lx];
  }

  public getBlockLightLocal(lx: number, y: number, lz: number): number {
    if (y < 0 || y >= CHUNK_HEIGHT) return 0;
    return this.blockLight[(y << 8) | (lz << 4) | lx];
  }

  public disposeMeshes(scene: THREE.Scene): void {
    const meshes = [this.opaqueMesh, this.foliageMesh, this.translucentMesh, this.waterMesh];
    for (const m of meshes) {
      if (m) {
        scene.remove(m);
        m.geometry.dispose();
      }
    }
    this.opaqueMesh = null;
    this.foliageMesh = null;
    this.translucentMesh = null;
    this.waterMesh = null;
  }
}
