import * as THREE from 'three';
import { SoundEngine } from '../audio/SoundEngine';
import { MobAndParticleSystem } from '../entities/MobSystem';
import { ProceduralAtlas } from '../textures/ProceduralAtlas';
import { GameMode, HotbarSlot, RaycastHit } from '../types';
import { BlockId, DEFAULT_HOTBAR_BLOCKS, getBlockDef, isBlockSolid } from '../world/blocks';
import { World } from '../world/World';

const PLAYER_HALF_W = 0.3;
const PLAYER_HEIGHT = 1.8;
const EYE_STAND = 1.62;
const EYE_SNEAK = 1.32;

export class PlayerController {
  public readonly camera: THREE.PerspectiveCamera;
  public readonly viewmodelScene = new THREE.Scene();
  public readonly viewmodelCamera: THREE.PerspectiveCamera;

  public readonly position = new THREE.Vector3(0.5, 68, 0.5);
  public readonly velocity = new THREE.Vector3();
  public yaw = 0; // looking toward +Z (the scenic lake & mountain vista!)
  public pitch = -0.14;

  public gameMode: GameMode = 'survival';
  public flying = false;
  public onGround = false;
  public inWater = false;
  public eyesUnderwater = false;
  public sprinting = false;
  public sneaking = false;

  // Survival stats (1:1 Minecraft scales)
  public health = 20;
  public maxHealth = 20;
  public hunger = 20;
  public air = 300;
  public maxAir = 300;
  public xpLevel = 3;
  public xpProgress = 0.42;
  public damageFlash = 0;

  public hotbar: HotbarSlot[] = DEFAULT_HOTBAR_BLOCKS.map((id) => ({ blockId: id, count: 64 }));
  public selectedSlot = 0;

  public targetHit: RaycastHit | null = null;
  public isPointerLocked = false;
  public uiPaused = true;

  // Block mining state
  private leftMouseDown = false;
  private rightMouseDown = false;
  private placeCooldown = 0;
  private breakProgress = 0;
  private breakTargetKey = '';
  private breakTickSound = 0;

  // Input tracking
  private readonly keys: Set<string> = new Set();
  private lastSpaceTap = 0;
  private lastWTap = 0;
  private stepDist = 0;
  private walkBobTime = 0;
  private swingProgress = 0;
  private fallStartY = 68;
  private wasInWater = false;
  private regenTimer = 0;
  private drowningTimer = 0;

  // Visual overlays in world
  private readonly selectionBox: THREE.LineSegments;
  private readonly crackMesh: THREE.Mesh;
  private readonly crackTextures: THREE.CanvasTexture[] = [];

  // First-person held block / arm viewmodel
  private readonly handGroup = new THREE.Group();
  private readonly armMesh: THREE.Mesh;
  private heldBlockMesh: THREE.Mesh | null = null;
  private lastHeldBlockId = -1;
  private readonly vmLight = new THREE.DirectionalLight(0xfff6e5, 1.35);
  private readonly vmAmbient = new THREE.HemisphereLight(0xbbe0ff, 0x555555, 0.9);

  private readonly world: World;
  private readonly atlas: ProceduralAtlas;
  private readonly sound: SoundEngine;
  private readonly mobs: MobAndParticleSystem;
  public onInventoryChanged?: () => void;
  public onWorldEdited?: () => void;
  public onToggleInventory?: () => void;
  public onTogglePause?: () => void;
  public onToggleDebug?: () => void;

  constructor(
    camera: THREE.PerspectiveCamera,
    scene: THREE.Scene,
    world: World,
    atlas: ProceduralAtlas,
    sound: SoundEngine,
    mobs: MobAndParticleSystem
  ) {
    this.camera = camera;
    this.world = world;
    this.atlas = atlas;
    this.sound = sound;
    this.mobs = mobs;

    this.viewmodelCamera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.05, 10);
    this.viewmodelScene.add(this.vmLight);
    this.viewmodelScene.add(this.vmAmbient);
    this.viewmodelScene.add(this.handGroup);
    this.vmLight.position.set(2, 4, 3);

    // Classic Steve forearm (shown when holding Air or subtle behind held block)
    const armGeom = new THREE.BoxGeometry(0.24, 0.68, 0.24);
    const armMat = new THREE.MeshLambertMaterial({ color: 0xc89572 });
    this.armMesh = new THREE.Mesh(armGeom, armMat);
    this.armMesh.position.set(0.56, -0.52, -0.82);
    this.armMesh.rotation.set(0.32, -0.25, 0.25);
    this.handGroup.add(this.armMesh);

    // Exact Minecraft black wireframe target block outline
    const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(1.006, 1.006, 1.006));
    const lineMat = new THREE.LineBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.72 });
    this.selectionBox = new THREE.LineSegments(edges, lineMat);
    this.selectionBox.visible = false;
    this.selectionBox.renderOrder = 6;
    scene.add(this.selectionBox);

    // Generate authentic 10-stage Minecraft crack textures (0..9)
    for (let stage = 0; stage < 10; stage++) {
      this.crackTextures.push(this.createCrackTexture(stage));
    }
    const crackMat = new THREE.MeshBasicMaterial({
      map: this.crackTextures[0],
      transparent: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
    });
    this.crackMesh = new THREE.Mesh(new THREE.BoxGeometry(1.008, 1.008, 1.008), crackMat);
    this.crackMesh.visible = false;
    this.crackMesh.renderOrder = 7;
    scene.add(this.crackMesh);

    this.rebuildHeldViewmodel();
    this.bindEvents();
  }

  private createCrackTexture(stage: number): THREE.CanvasTexture {
    const c = document.createElement('canvas');
    c.width = 16;
    c.height = 16;
    const ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, 16, 16);
    ctx.fillStyle = `rgba(12, 12, 12, ${0.42 + stage * 0.055})`;

    // Radiating pixel fracture lines growing with mining stage
    const rays = [
      [8, 8, 1, 0], [8, 8, -1, 0], [8, 8, 0, 1], [8, 8, 0, -1],
      [8, 8, 1, 1], [8, 8, -1, 1], [8, 8, 1, -1], [8, 8, -1, -1],
    ];
    const len = 1 + Math.floor((stage + 1) * 0.85);
    for (let r = 0; r < rays.length; r++) {
      if (r > stage + 1) break;
      let [x, y, dx, dy] = rays[r];
      for (let s = 0; s < len; s++) {
        ctx.fillRect(Math.floor(x), Math.floor(y), 1, 1);
        if (stage > 5 && s % 2 === 1) {
          ctx.fillRect(Math.floor(x) + dy, Math.floor(y) - dx, 1, 1);
        }
        x += dx + (s % 2 === 0 ? 0 : (r % 2 === 0 ? 0.35 : -0.35));
        y += dy + (s % 2 === 1 ? 0 : (r % 2 === 0 ? -0.35 : 0.35));
      }
    }

    const tex = new THREE.CanvasTexture(c);
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.NearestFilter;
    return tex;
  }

  public resize(aspect: number): void {
    this.viewmodelCamera.aspect = aspect;
    this.viewmodelCamera.updateProjectionMatrix();
  }

  public getSelectedBlockId(): number {
    const slot = this.hotbar[this.selectedSlot];
    if (!slot || slot.count <= 0) return BlockId.AIR;
    return slot.blockId;
  }

  public selectSlot(index: number): void {
    this.selectedSlot = ((index % 9) + 9) % 9;
    this.rebuildHeldViewmodel();
    this.onInventoryChanged?.();
  }

  public setHotbarSlot(index: number, blockId: number, count = 64): void {
    this.hotbar[index] = { blockId, count };
    if (index === this.selectedSlot) this.rebuildHeldViewmodel();
    this.onInventoryChanged?.();
  }

  public toggleGameMode(): void {
    this.gameMode = this.gameMode === 'survival' ? 'creative' : 'survival';
    if (this.gameMode === 'survival') {
      this.flying = false;
    } else {
      this.health = this.maxHealth;
      this.hunger = 20;
      this.air = this.maxAir;
    }
    this.onInventoryChanged?.();
  }

  private rebuildHeldViewmodel(): void {
    const blockId = this.getSelectedBlockId();
    if (blockId === this.lastHeldBlockId) return;
    this.lastHeldBlockId = blockId;

    if (this.heldBlockMesh) {
      this.handGroup.remove(this.heldBlockMesh);
      this.heldBlockMesh.geometry.dispose();
      this.heldBlockMesh = null;
    }

    if (blockId === BlockId.AIR) {
      this.armMesh.visible = true;
      return;
    }
    this.armMesh.visible = false;

    const def = getBlockDef(blockId);
    if (def.shape === 'torch') {
      const g = new THREE.BoxGeometry(0.09, 0.45, 0.09);
      const m = new THREE.MeshBasicMaterial({ color: 0xd99b43 });
      const mesh = new THREE.Mesh(g, m);
      const tip = new THREE.Mesh(
        new THREE.BoxGeometry(0.11, 0.11, 0.11),
        new THREE.MeshBasicMaterial({ color: 0xffdd55 })
      );
      tip.position.y = 0.24;
      mesh.add(tip);
      mesh.position.set(0.46, -0.32, -0.76);
      mesh.rotation.set(0.15, -0.35, 0.12);
      this.heldBlockMesh = mesh;
      this.handGroup.add(mesh);
      return;
    }

    // Build authentic 3D textured cube from ProceduralAtlas with per-face grass/foliage biome tint!
    const geom = new THREE.BoxGeometry(0.36, 0.36, 0.36);
    const uvAttr = geom.getAttribute('uv') as THREE.BufferAttribute;
    const faceTiles = [
      def.tiles.east,
      def.tiles.west,
      def.tiles.top,
      def.tiles.bottom,
      def.tiles.south,
      def.tiles.north,
    ];
    const colors = new Float32Array(24 * 3);
    for (let f = 0; f < 6; f++) {
      const uv = this.atlas.getTileUV(faceTiles[f]);
      const base = f * 4;
      uvAttr.setXY(base + 0, uv.u0, uv.v0);
      uvAttr.setXY(base + 1, uv.u1, uv.v0);
      uvAttr.setXY(base + 2, uv.u0, uv.v1);
      uvAttr.setXY(base + 3, uv.u1, uv.v1);

      const tinted = def.tintAll || (def.tintTop && f === 2);
      const cr = tinted ? 0.52 : 1.0;
      const cg = tinted ? 0.84 : 1.0;
      const cb = tinted ? 0.36 : 1.0;
      for (let v = 0; v < 4; v++) {
        colors[(base + v) * 3] = cr;
        colors[(base + v) * 3 + 1] = cg;
        colors[(base + v) * 3 + 2] = cb;
      }
    }
    uvAttr.needsUpdate = true;
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.MeshLambertMaterial({
      map: this.atlas.albedoTexture,
      vertexColors: true,
      transparent: def.transparent,
      alphaTest: def.shape === 'cross' || def.windSway ? 0.4 : 0,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(0.48, -0.38, -0.78);
    mesh.rotation.set(0.22, -0.68, 0.08);
    this.heldBlockMesh = mesh;
    this.handGroup.add(mesh);
  }

  private bindEvents(): void {
    document.addEventListener('pointerlockchange', () => {
      this.isPointerLocked = document.pointerLockElement !== null;
      if (!this.isPointerLocked) {
        this.keys.clear();
        this.leftMouseDown = false;
        this.rightMouseDown = false;
        this.resetBreaking();
        if (!this.uiPaused) {
          this.onTogglePause?.();
        }
      }
    });

    document.addEventListener('mousemove', (e) => {
      if (!this.isPointerLocked || this.uiPaused) return;
      const sens = 0.0022;
      this.yaw -= e.movementX * sens;
      this.pitch = Math.max(-Math.PI * 0.495, Math.min(Math.PI * 0.495, this.pitch - e.movementY * sens));
    });

    window.addEventListener('keydown', (e) => {
      if (e.code === 'F3' || (e.code === 'KeyI' && !this.uiPaused)) {
        e.preventDefault();
        this.onToggleDebug?.();
        return;
      }
      if (e.code === 'KeyE' && (this.isPointerLocked || !this.uiPaused)) {
        e.preventDefault();
        this.onToggleInventory?.();
        return;
      }
      if (e.code === 'KeyG' && this.isPointerLocked) {
        this.toggleGameMode();
        return;
      }

      if (!this.isPointerLocked || this.uiPaused) return;

      // Number keys 1-9 for hotbar
      if (e.code.startsWith('Digit')) {
        const n = parseInt(e.code.slice(5), 10);
        if (n >= 1 && n <= 9) this.selectSlot(n - 1);
      }

      if (e.code === 'Space' && !e.repeat) {
        const now = performance.now();
        if (this.gameMode === 'creative' && now - this.lastSpaceTap < 280) {
          this.flying = !this.flying;
          this.velocity.y = 0;
        }
        this.lastSpaceTap = now;
      }

      if (e.code === 'KeyW' && !e.repeat) {
        const now = performance.now();
        if (now - this.lastWTap < 280) {
          this.sprinting = true;
        }
        this.lastWTap = now;
      }

      this.keys.add(e.code);
    });

    window.addEventListener('keyup', (e) => {
      this.keys.delete(e.code);
      if (e.code === 'KeyW') this.sprinting = false;
    });

    window.addEventListener('wheel', (e) => {
      if (!this.isPointerLocked || this.uiPaused) return;
      const dir = e.deltaY > 0 ? 1 : -1;
      this.selectSlot(this.selectedSlot + dir);
    }, { passive: true });

    window.addEventListener('mousedown', (e) => {
      if (!this.isPointerLocked || this.uiPaused) return;
      this.sound.unlock();
      if (e.button === 0) {
        this.leftMouseDown = true;
        this.triggerSwing();
        if (this.gameMode === 'creative') {
          this.tryInstantBreak();
        }
      } else if (e.button === 2) {
        this.rightMouseDown = true;
        this.tryPlaceBlock();
        this.placeCooldown = 0.22;
      } else if (e.button === 1 && this.targetHit) {
        // Middle click: Pick Block (1:1 Minecraft Creative feature!)
        e.preventDefault();
        this.setHotbarSlot(this.selectedSlot, this.targetHit.blockId, 64);
      }
    });

    window.addEventListener('mouseup', (e) => {
      if (e.button === 0) {
        this.leftMouseDown = false;
        this.resetBreaking();
      } else if (e.button === 2) {
        this.rightMouseDown = false;
      }
    });

    window.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  private triggerSwing(): void {
    this.swingProgress = 1.0;
  }

  private resetBreaking(): void {
    this.breakProgress = 0;
    this.breakTargetKey = '';
    this.crackMesh.visible = false;
  }

  private tryInstantBreak(): void {
    if (!this.targetHit) return;
    const { x, y, z, blockId } = this.targetHit;
    const def = getBlockDef(blockId);
    if (def.hardness < 0 && this.gameMode !== 'creative') return;
    if (blockId === BlockId.BEDROCK && this.gameMode !== 'creative') return;

    if (this.world.setBlock(x, y, z, BlockId.AIR)) {
      this.sound.playBreak(def.sound);
      this.mobs.spawnBlockBreakParticles(x, y, z, this.atlas.getAverageColor(blockId));
      this.onWorldEdited?.();
    }
  }

  private tryPlaceBlock(): void {
    if (!this.targetHit) return;
    const blockId = this.getSelectedBlockId();
    if (blockId === BlockId.AIR) return;

    const px = this.targetHit.x + this.targetHit.nx;
    const py = this.targetHit.y + this.targetHit.ny;
    const pz = this.targetHit.z + this.targetHit.nz;

    // Prevent placing solid blocks inside the player's own AABB
    const def = getBlockDef(blockId);
    if (def.solid) {
      const minX = this.position.x - PLAYER_HALF_W;
      const maxX = this.position.x + PLAYER_HALF_W;
      const minY = this.position.y;
      const maxY = this.position.y + PLAYER_HEIGHT;
      const minZ = this.position.z - PLAYER_HALF_W;
      const maxZ = this.position.z + PLAYER_HALF_W;

      const overlaps =
        px + 1 > minX && px < maxX &&
        py + 1 > minY && py < maxY &&
        pz + 1 > minZ && pz < maxZ;
      if (overlaps) return;
    }

    if (this.world.setBlock(px, py, pz, blockId)) {
      this.triggerSwing();
      this.sound.playPlace(def.sound);
      if (this.gameMode === 'survival') {
        const slot = this.hotbar[this.selectedSlot];
        if (slot && slot.count > 0) {
          slot.count = Math.max(0, slot.count - 1);
          if (slot.count === 0) slot.blockId = BlockId.AIR;
          this.rebuildHeldViewmodel();
          this.onInventoryChanged?.();
        }
      }
      this.onWorldEdited?.();
    }
  }

  private collectBlockToInventory(blockId: number): void {
    const def = getBlockDef(blockId);
    const drop = def.dropId ?? blockId;
    if (drop === BlockId.AIR) return;
    // Try stacking on existing slot first
    for (const s of this.hotbar) {
      if (s.blockId === drop && s.count < 64) {
        s.count++;
        this.onInventoryChanged?.();
        return;
      }
    }
    // Otherwise fill first empty slot
    for (let i = 0; i < this.hotbar.length; i++) {
      if (this.hotbar[i].blockId === BlockId.AIR || this.hotbar[i].count <= 0) {
        this.hotbar[i] = { blockId: drop, count: 1 };
        if (i === this.selectedSlot) this.rebuildHeldViewmodel();
        this.onInventoryChanged?.();
        return;
      }
    }
  }

  // Exact Amanatides & Woo 3D DDA voxel raycast
  public raycast(maxDist = 5.4): RaycastHit | null {
    const origin = this.camera.position;
    const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion).normalize();

    let x = Math.floor(origin.x);
    let y = Math.floor(origin.y);
    let z = Math.floor(origin.z);

    const stepX = dir.x >= 0 ? 1 : -1;
    const stepY = dir.y >= 0 ? 1 : -1;
    const stepZ = dir.z >= 0 ? 1 : -1;

    const tDeltaX = dir.x !== 0 ? Math.abs(1 / dir.x) : 1e9;
    const tDeltaY = dir.y !== 0 ? Math.abs(1 / dir.y) : 1e9;
    const tDeltaZ = dir.z !== 0 ? Math.abs(1 / dir.z) : 1e9;

    let tMaxX = dir.x !== 0 ? ((stepX > 0 ? x + 1 - origin.x : origin.x - x) * tDeltaX) : 1e9;
    let tMaxY = dir.y !== 0 ? ((stepY > 0 ? y + 1 - origin.y : origin.y - y) * tDeltaY) : 1e9;
    let tMaxZ = dir.z !== 0 ? ((stepZ > 0 ? z + 1 - origin.z : origin.z - z) * tDeltaZ) : 1e9;

    let nx = 0, ny = 0, nz = 0;
    let dist = 0;

    for (let i = 0; i < 64; i++) {
      const id = this.world.getBlock(x, y, z);
      if (id !== BlockId.AIR && id !== BlockId.WATER && id !== BlockId.LAVA) {
        return { x, y, z, nx, ny, nz, blockId: id, distance: dist };
      }

      if (tMaxX < tMaxY && tMaxX < tMaxZ) {
        dist = tMaxX;
        tMaxX += tDeltaX;
        x += stepX;
        nx = -stepX; ny = 0; nz = 0;
      } else if (tMaxY < tMaxZ) {
        dist = tMaxY;
        tMaxY += tDeltaY;
        y += stepY;
        nx = 0; ny = -stepY; nz = 0;
      } else {
        dist = tMaxZ;
        tMaxZ += tDeltaZ;
        z += stepZ;
        nx = 0; ny = 0; nz = -stepZ;
      }

      if (dist > maxDist) break;
    }
    return null;
  }

  private collidesAABB(px: number, py: number, pz: number): boolean {
    const minX = Math.floor(px - PLAYER_HALF_W);
    const maxX = Math.floor(px + PLAYER_HALF_W - 1e-4);
    const minY = Math.floor(py);
    const maxY = Math.floor(py + PLAYER_HEIGHT - 1e-4);
    const minZ = Math.floor(pz - PLAYER_HALF_W);
    const maxZ = Math.floor(pz + PLAYER_HALF_W - 1e-4);

    for (let y = minY; y <= maxY; y++) {
      for (let z = minZ; z <= maxZ; z++) {
        for (let x = minX; x <= maxX; x++) {
          if (isBlockSolid(this.world.getBlock(x, y, z))) return true;
        }
      }
    }
    return false;
  }

  public update(dt: number, baseFov: number, sunColor: THREE.Color, dayFactor: number): void {
    const clampedDt = Math.min(dt, 0.05);

    // Sync viewmodel lighting with world sun
    this.vmLight.color.copy(sunColor);
    this.vmLight.intensity = 0.55 + 0.95 * dayFactor;
    this.vmAmbient.intensity = 0.35 + 0.65 * dayFactor;

    if (this.damageFlash > 0) {
      this.damageFlash = Math.max(0, this.damageFlash - clampedDt * 2.5);
    }

    // Water checks
    const feetBlock = this.world.getBlock(Math.floor(this.position.x), Math.floor(this.position.y + 0.35), Math.floor(this.position.z));
    const eyeH = this.sneaking ? EYE_SNEAK : EYE_STAND;
    const eyeBlock = this.world.getBlock(Math.floor(this.position.x), Math.floor(this.position.y + eyeH), Math.floor(this.position.z));
    this.inWater = feetBlock === BlockId.WATER || eyeBlock === BlockId.WATER;
    this.eyesUnderwater = eyeBlock === BlockId.WATER;

    if (this.inWater && !this.wasInWater && Math.abs(this.velocity.y) > 2.2) {
      this.sound.playSplash();
      this.mobs.spawnWaterSplash(this.position.x, this.position.y + 0.6, this.position.z);
    }
    this.wasInWater = this.inWater;

    // Movement input
    const moveForward = (this.keys.has('KeyW') ? 1 : 0) - (this.keys.has('KeyS') ? 1 : 0);
    const moveRight = (this.keys.has('KeyD') ? 1 : 0) - (this.keys.has('KeyA') ? 1 : 0);
    this.sneaking = (this.keys.has('ShiftLeft') || this.keys.has('ShiftRight')) && !this.flying;
    if (this.keys.has('ControlLeft') && moveForward > 0 && !this.sneaking) {
      this.sprinting = true;
    }
    if (moveForward <= 0 || this.sneaking) {
      this.sprinting = false;
    }

    const sinY = Math.sin(this.yaw);
    const cosY = Math.cos(this.yaw);
    // Forward direction matches Three.js camera facing -Z when yaw=0 (Wait: yaw rotates around Y!)
    // Let's keep camera quaternion YXZ: forward = (-sin(yaw), 0, -cos(yaw)), right = (cos(yaw), 0, -sin(yaw))
    let wishX = -sinY * moveForward + cosY * moveRight;
    let wishZ = -cosY * moveForward - sinY * moveRight;
    const wishLen = Math.hypot(wishX, wishZ);
    if (wishLen > 0) {
      wishX /= wishLen;
      wishZ /= wishLen;
    }

    const baseSpeed = this.flying ? 11.5 : this.sneaking ? 1.85 : this.sprinting ? 6.1 : 4.4;
    const speed = this.inWater && !this.flying ? baseSpeed * 0.56 : baseSpeed;

    const accel = this.onGround || this.flying ? 16 : 6.5;
    this.velocity.x += (wishX * speed - this.velocity.x) * Math.min(1, accel * clampedDt);
    this.velocity.z += (wishZ * speed - this.velocity.z) * Math.min(1, accel * clampedDt);

    if (this.flying) {
      const ascend = (this.keys.has('Space') ? 1 : 0) - ((this.keys.has('ShiftLeft') || this.keys.has('ShiftRight')) ? 1 : 0);
      this.velocity.y += (ascend * 9.0 - this.velocity.y) * Math.min(1, 10 * clampedDt);
      this.fallStartY = this.position.y;
    } else if (this.inWater) {
      if (this.keys.has('Space')) {
        this.velocity.y = Math.min(3.8, this.velocity.y + 15 * clampedDt);
      } else {
        this.velocity.y = Math.max(-2.6, this.velocity.y - 6.5 * clampedDt);
      }
      this.velocity.y *= 0.94;
      this.fallStartY = this.position.y;
    } else {
      // Authentic Minecraft gravity (28 m/s² feels super crisp and responsive)
      this.velocity.y -= 28.0 * clampedDt;
      if (this.keys.has('Space') && this.onGround) {
        this.velocity.y = 8.95; // Clears 1.25 blocks exactly like Minecraft!
        this.onGround = false;
      }
    }

    // Swept axis-separated collision resolution
    const oldX = this.position.x;
    const oldZ = this.position.z;

    // X axis (with sneak edge-clamp!)
    const nextX = this.position.x + this.velocity.x * clampedDt;
    if (!this.collidesAABB(nextX, this.position.y, this.position.z)) {
      if (this.sneaking && this.onGround && !this.collidesAABB(nextX, this.position.y - 0.15, this.position.z)) {
        this.velocity.x = 0;
      } else {
        this.position.x = nextX;
      }
    } else {
      this.velocity.x = 0;
    }

    // Z axis (with sneak edge-clamp!)
    const nextZ = this.position.z + this.velocity.z * clampedDt;
    if (!this.collidesAABB(this.position.x, this.position.y, nextZ)) {
      if (this.sneaking && this.onGround && !this.collidesAABB(this.position.x, this.position.y - 0.15, nextZ)) {
        this.velocity.z = 0;
      } else {
        this.position.z = nextZ;
      }
    } else {
      this.velocity.z = 0;
    }

    // Y axis
    const nextY = this.position.y + this.velocity.y * clampedDt;
    const wasOnGround = this.onGround;
    if (!this.collidesAABB(this.position.x, nextY, this.position.z)) {
      this.position.y = nextY;
      this.onGround = false;
    } else {
      if (this.velocity.y < 0) {
        this.position.y = Math.floor(nextY) + 1.0001;
        // Resolve exact ground snap
        while (this.collidesAABB(this.position.x, this.position.y, this.position.z)) {
          this.position.y += 0.1;
        }
        this.onGround = true;

        // Fall damage in Survival mode
        if (!wasOnGround && this.gameMode === 'survival' && !this.inWater) {
          const fallDist = this.fallStartY - this.position.y;
          if (fallDist > 3.6) {
            const dmg = Math.floor(fallDist - 3);
            this.applyDamage(dmg);
          }
        }
      } else {
        this.position.y = Math.floor(nextY + PLAYER_HEIGHT) - PLAYER_HEIGHT - 0.001;
      }
      this.velocity.y = 0;
    }

    if (this.onGround || this.velocity.y > 0) {
      this.fallStartY = this.position.y;
    }

    // Footstep sound & walk bobbing
    const horizMoved = Math.hypot(this.position.x - oldX, this.position.z - oldZ);
    if ((this.onGround || this.inWater) && horizMoved > 0.001) {
      this.walkBobTime += horizMoved * 2.55;
      this.stepDist += horizMoved;
      if (this.stepDist > 1.95) {
        this.stepDist = 0;
        if (this.inWater) {
          this.sound.playStep('water');
        } else {
          const belowId = this.world.getBlock(Math.floor(this.position.x), Math.floor(this.position.y - 0.1), Math.floor(this.position.z));
          this.sound.playStep(getBlockDef(belowId).sound);
        }
      }
    }

    // Survival oxygen & health regen
    if (this.gameMode === 'survival') {
      if (this.eyesUnderwater) {
        this.air = Math.max(0, this.air - clampedDt * 20);
        if (this.air <= 0) {
          this.drowningTimer += clampedDt;
          if (this.drowningTimer >= 1.0) {
            this.drowningTimer = 0;
            this.applyDamage(2);
          }
        }
      } else {
        this.air = Math.min(this.maxAir, this.air + clampedDt * 90);
        this.drowningTimer = 0;
      }

      if (this.health < this.maxHealth && this.hunger >= 18) {
        this.regenTimer += clampedDt;
        if (this.regenTimer >= 3.2) {
          this.regenTimer = 0;
          this.health = Math.min(this.maxHealth, this.health + 1);
        }
      }
    }

    // Sync camera transform with subtle authentic Minecraft head bob & sprint FOV kick
    const bobAmp = this.onGround && !this.flying ? Math.min(1, horizMoved / (clampedDt * 4.5 + 1e-5)) : 0;
    const bobY = Math.abs(Math.sin(this.walkBobTime)) * 0.055 * bobAmp;
    const bobX = Math.cos(this.walkBobTime) * 0.03 * bobAmp;

    this.camera.position.set(
      this.position.x + bobX * cosY,
      this.position.y + eyeH + bobY,
      this.position.z - bobX * sinY
    );
    this.camera.rotation.order = 'YXZ';
    this.camera.rotation.y = this.yaw;
    this.camera.rotation.x = this.pitch;

    const targetFov = baseFov + (this.sprinting ? 8 : 0);
    if (Math.abs(this.camera.fov - targetFov) > 0.05) {
      this.camera.fov += (targetFov - this.camera.fov) * Math.min(1, clampedDt * 10);
      this.camera.updateProjectionMatrix();
    }

    // Animate first-person held block/arm swing & bob
    if (this.swingProgress > 0) {
      this.swingProgress = Math.max(0, this.swingProgress - clampedDt * 4.8);
    }
    const swingSin = Math.sin(this.swingProgress * Math.PI);
    this.handGroup.position.set(
      Math.cos(this.walkBobTime) * 0.032 * bobAmp - swingSin * 0.14,
      -Math.abs(Math.sin(this.walkBobTime)) * 0.036 * bobAmp + swingSin * 0.08,
      -swingSin * 0.18
    );
    this.handGroup.rotation.set(-swingSin * 0.55, swingSin * 0.25, swingSin * 0.18);

    // Raycast & block interaction
    this.targetHit = this.raycast(5.4);
    if (this.targetHit && !this.uiPaused) {
      this.selectionBox.visible = true;
      this.selectionBox.position.set(this.targetHit.x + 0.5, this.targetHit.y + 0.5, this.targetHit.z + 0.5);
    } else {
      this.selectionBox.visible = false;
    }

    // Survival progressive block breaking with 10-stage crack overlay
    if (this.leftMouseDown && this.isPointerLocked && !this.uiPaused && this.targetHit) {
      if (this.gameMode === 'survival') {
        const key = `${this.targetHit.x},${this.targetHit.y},${this.targetHit.z}`;
        if (key !== this.breakTargetKey) {
          this.breakTargetKey = key;
          this.breakProgress = 0;
        }
        const def = getBlockDef(this.targetHit.blockId);
        if (def.hardness >= 0) {
          const breakTime = Math.max(0.08, def.hardness);
          this.breakProgress += clampedDt / breakTime;
          this.breakTickSound += clampedDt;
          if (this.swingProgress <= 0.1) this.triggerSwing();
          if (this.breakTickSound > 0.22) {
            this.breakTickSound = 0;
            this.sound.playStep(def.sound);
          }

          const stage = Math.min(9, Math.floor(this.breakProgress * 10));
          const mat = this.crackMesh.material as THREE.MeshBasicMaterial;
          mat.map = this.crackTextures[stage];
          this.crackMesh.position.set(this.targetHit.x + 0.5, this.targetHit.y + 0.5, this.targetHit.z + 0.5);
          this.crackMesh.visible = true;

          if (this.breakProgress >= 1.0) {
            const brokenId = this.targetHit.blockId;
            if (this.world.setBlock(this.targetHit.x, this.targetHit.y, this.targetHit.z, BlockId.AIR)) {
              this.sound.playBreak(def.sound);
              this.mobs.spawnBlockBreakParticles(
                this.targetHit.x,
                this.targetHit.y,
                this.targetHit.z,
                this.atlas.getAverageColor(brokenId)
              );
              this.collectBlockToInventory(brokenId);
              this.onWorldEdited?.();
            }
            this.resetBreaking();
          }
        }
      }
    } else {
      this.resetBreaking();
    }

    // Hold right-click to continuously build (classic Minecraft!)
    if (this.rightMouseDown && this.isPointerLocked && !this.uiPaused) {
      this.placeCooldown -= clampedDt;
      if (this.placeCooldown <= 0) {
        this.tryPlaceBlock();
        this.placeCooldown = 0.22;
      }
    }
  }

  private applyDamage(amount: number): void {
    if (amount <= 0) return;
    this.health = Math.max(0, this.health - amount);
    this.damageFlash = 1.0;
    this.sound.playHurt();
    if (this.health <= 0) {
      // Respawn at safe surface height
      this.health = this.maxHealth;
      this.hunger = 20;
      this.air = this.maxAir;
      const y = this.world.getSurfaceY(0.5, 0.5);
      this.position.set(0.5, y + 0.5, 0.5);
      this.velocity.set(0, 0, 0);
    }
  }
}
