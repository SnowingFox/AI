import * as THREE from 'three';
import { BlockId, isBlockSolid } from '../world/blocks';
import { World } from '../world/World';

type MobKind = 'pig' | 'sheep' | 'cow' | 'chicken';

interface MobEntity {
  kind: MobKind;
  group: THREE.Group;
  legs: THREE.Mesh[];
  head: THREE.Object3D;
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  yaw: number;
  targetYaw: number;
  walkTimer: number;
  moving: boolean;
  walkPhase: number;
  width: number;
  height: number;
}

interface Particle {
  mesh: THREE.Mesh;
  vel: THREE.Vector3;
  life: number;
  maxLife: number;
  gravity: number;
  spin: THREE.Vector3;
}

function makeMobFaceTexture(bg: string, eyeColor: string, snoutColor: string, accent?: string): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = 16;
  c.height = 16;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, 16, 16);

  if (accent) {
    ctx.fillStyle = accent;
    ctx.fillRect(0, 0, 5, 5);
    ctx.fillRect(11, 1, 5, 4);
  }

  // Classic Minecraft mob eyes (white sclera + dark pupils)
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(2, 6, 3, 2);
  ctx.fillRect(11, 6, 3, 2);
  ctx.fillStyle = eyeColor;
  ctx.fillRect(3, 6, 2, 2);
  ctx.fillRect(11, 6, 2, 2);

  // Snout / beak
  ctx.fillStyle = snoutColor;
  ctx.fillRect(5, 9, 6, 4);
  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  ctx.fillRect(6, 10, 1, 2);
  ctx.fillRect(9, 10, 1, 2);

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  return tex;
}

export class MobAndParticleSystem {
  private readonly scene: THREE.Scene;
  private readonly world: World;
  private readonly mobs: MobEntity[] = [];
  private readonly particles: Particle[] = [];
  private readonly fireflies: THREE.Points;
  private readonly mobRoot = new THREE.Group();
  private readonly particleRoot = new THREE.Group();
  private readonly sharedBoxGeom = new THREE.BoxGeometry(1, 1, 1);

  private readonly mobAmbientLight = new THREE.HemisphereLight(0xbbe0ff, 0x455a36, 0.85);
  private readonly mobSunLight = new THREE.DirectionalLight(0xfff4dd, 1.35);

  private torchTimer = 0;

  constructor(scene: THREE.Scene, world: World) {
    this.scene = scene;
    this.world = world;
    this.scene.add(this.mobRoot);
    this.scene.add(this.particleRoot);
    this.scene.add(this.mobAmbientLight);
    this.scene.add(this.mobSunLight);

    // Atmospheric dusk/night fireflies
    const ffCount = 90;
    const ffPos = new Float32Array(ffCount * 3);
    for (let i = 0; i < ffCount; i++) {
      ffPos[i * 3] = (Math.random() - 0.5) * 48;
      ffPos[i * 3 + 1] = 2 + Math.random() * 8;
      ffPos[i * 3 + 2] = (Math.random() - 0.5) * 48;
    }
    const ffGeom = new THREE.BufferGeometry();
    ffGeom.setAttribute('position', new THREE.BufferAttribute(ffPos, 3));
    const ffMat = new THREE.PointsMaterial({
      color: 0xdcff72,
      size: 0.22,
      transparent: true,
      opacity: 0.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.fireflies = new THREE.Points(ffGeom, ffMat);
    this.fireflies.frustumCulled = false;
    this.scene.add(this.fireflies);
  }

  public spawnInitialMobs(centerX: number, centerZ: number): void {
    this.clearMobs();
    const kinds: MobKind[] = ['sheep', 'pig', 'cow', 'chicken', 'sheep', 'pig', 'cow', 'chicken', 'sheep', 'pig', 'cow', 'sheep'];
    for (let i = 0; i < kinds.length; i++) {
      const angle = (i / kinds.length) * Math.PI * 2 + Math.random() * 0.4;
      const dist = 9 + Math.random() * 28;
      const x = centerX + Math.cos(angle) * dist;
      const z = centerZ + Math.sin(angle) * dist;
      const y = this.world.getSurfaceY(x, z);
      const below = this.world.getBlock(Math.floor(x), y - 1, Math.floor(z));
      if (below === BlockId.WATER || below === BlockId.LAVA) continue;
      this.spawnMob(kinds[i], x, y + 0.1, z);
    }
  }

  public clearMobs(): void {
    for (const m of this.mobs) {
      this.mobRoot.remove(m.group);
    }
    this.mobs.length = 0;
  }

  private spawnMob(kind: MobKind, x: number, y: number, z: number): void {
    const group = new THREE.Group();
    const legs: THREE.Mesh[] = [];
    let head: THREE.Object3D = new THREE.Group();
    let width = 0.7;
    let height = 1.1;

    if (kind === 'pig') {
      const skin = new THREE.MeshLambertMaterial({ color: 0xf09da8 });
      const faceMat = new THREE.MeshLambertMaterial({ map: makeMobFaceTexture('#f09da8', '#1a1a1a', '#e27d8b') });
      const body = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.52, 0.96), skin);
      body.position.set(0, 0.56, 0);
      group.add(body);

      const hMesh = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.46, 0.46), [skin, skin, skin, skin, faceMat, skin]);
      hMesh.position.set(0, 0.72, 0.56);
      group.add(hMesh);
      head = hMesh;

      for (const [lx, lz] of [[-0.2, 0.3], [0.2, 0.3], [-0.2, -0.3], [0.2, -0.3]]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.34, 0.2), skin);
        leg.position.set(lx, 0.17, lz);
        group.add(leg);
        legs.push(leg);
      }
      height = 0.95;
    } else if (kind === 'sheep') {
      const wool = new THREE.MeshLambertMaterial({ color: 0xf4f4f0 });
      const skin = new THREE.MeshLambertMaterial({ color: 0xdcc3a6 });
      const faceMat = new THREE.MeshLambertMaterial({ map: makeMobFaceTexture('#dcc3a6', '#181818', '#c6a588') });
      const body = new THREE.Mesh(new THREE.BoxGeometry(0.76, 0.66, 1.05), wool);
      body.position.set(0, 0.76, 0);
      group.add(body);

      const hMesh = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.42, 0.48), [skin, skin, wool, skin, faceMat, skin]);
      hMesh.position.set(0, 0.92, 0.62);
      group.add(hMesh);
      head = hMesh;

      for (const [lx, lz] of [[-0.22, 0.34], [0.22, 0.34], [-0.22, -0.34], [0.22, -0.34]]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.46, 0.2), skin);
        leg.position.set(lx, 0.23, lz);
        group.add(leg);
        legs.push(leg);
      }
      height = 1.25;
    } else if (kind === 'cow') {
      const hide = new THREE.MeshLambertMaterial({ color: 0x4c3626 });
      const patch = new THREE.MeshLambertMaterial({ color: 0xd8d2c8 });
      const faceMat = new THREE.MeshLambertMaterial({ map: makeMobFaceTexture('#4c3626', '#111111', '#9c8b82', '#d8d2c8') });
      const body = new THREE.Mesh(new THREE.BoxGeometry(0.76, 0.68, 1.12), hide);
      body.position.set(0, 0.84, 0);
      group.add(body);

      // Iconic cow white spot
      const spot = new THREE.Mesh(new THREE.BoxGeometry(0.78, 0.28, 0.44), patch);
      spot.position.set(0, 0.94, -0.1);
      group.add(spot);

      const hMesh = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.46, 0.44), [hide, hide, hide, hide, faceMat, hide]);
      hMesh.position.set(0, 1.08, 0.64);
      group.add(hMesh);
      head = hMesh;

      for (const [lx, lz] of [[-0.24, 0.38], [0.24, 0.38], [-0.24, -0.38], [0.24, -0.38]]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.54, 0.22), hide);
        leg.position.set(lx, 0.27, lz);
        group.add(leg);
        legs.push(leg);
      }
      width = 0.8;
      height = 1.38;
    } else {
      // Chicken
      const feather = new THREE.MeshLambertMaterial({ color: 0xf8f8f5 });
      const beak = new THREE.MeshLambertMaterial({ color: 0xe6a622 });
      const wattle = new THREE.MeshLambertMaterial({ color: 0xc82626 });
      const faceMat = new THREE.MeshLambertMaterial({ map: makeMobFaceTexture('#f8f8f5', '#111111', '#e6a622') });

      const body = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.34, 0.52), feather);
      body.position.set(0, 0.44, 0);
      group.add(body);

      const hMesh = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.32, 0.28), [feather, feather, feather, feather, faceMat, feather]);
      hMesh.position.set(0, 0.68, 0.28);
      group.add(hMesh);
      head = hMesh;

      const bMesh = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.08, 0.14), beak);
      bMesh.position.set(0, 0.64, 0.46);
      group.add(bMesh);
      const wMesh = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.09, 0.08), wattle);
      wMesh.position.set(0, 0.56, 0.42);
      group.add(wMesh);

      for (const lx of [-0.1, 0.1]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.28, 0.08), beak);
        leg.position.set(lx, 0.14, 0);
        group.add(leg);
        legs.push(leg);
      }
      width = 0.45;
      height = 0.8;
    }

    group.position.set(x, y, z);
    this.mobRoot.add(group);
    this.mobs.push({
      kind,
      group,
      legs,
      head,
      pos: new THREE.Vector3(x, y, z),
      vel: new THREE.Vector3(),
      yaw: Math.random() * Math.PI * 2,
      targetYaw: Math.random() * Math.PI * 2,
      walkTimer: 1 + Math.random() * 4,
      moving: Math.random() > 0.4,
      walkPhase: 0,
      width,
      height,
    });
  }

  public spawnBlockBreakParticles(wx: number, wy: number, wz: number, color: THREE.Color, count = 14): void {
    const mat = new THREE.MeshLambertMaterial({ color });
    for (let i = 0; i < count; i++) {
      const size = 0.07 + Math.random() * 0.09;
      const mesh = new THREE.Mesh(this.sharedBoxGeom, mat);
      mesh.scale.setScalar(size);
      mesh.position.set(
        wx + 0.15 + Math.random() * 0.7,
        wy + 0.15 + Math.random() * 0.7,
        wz + 0.15 + Math.random() * 0.7
      );
      this.particleRoot.add(mesh);
      this.particles.push({
        mesh,
        vel: new THREE.Vector3(
          (Math.random() - 0.5) * 3.8,
          1.4 + Math.random() * 2.8,
          (Math.random() - 0.5) * 3.8
        ),
        life: 0,
        maxLife: 0.45 + Math.random() * 0.35,
        gravity: 13.5,
        spin: new THREE.Vector3((Math.random() - 0.5) * 9, (Math.random() - 0.5) * 9, (Math.random() - 0.5) * 9),
      });
    }
  }

  public spawnWaterSplash(x: number, y: number, z: number): void {
    const mat = new THREE.MeshBasicMaterial({ color: 0xb8ecff, transparent: true, opacity: 0.85 });
    for (let i = 0; i < 10; i++) {
      const mesh = new THREE.Mesh(this.sharedBoxGeom, mat);
      mesh.scale.setScalar(0.06 + Math.random() * 0.06);
      mesh.position.set(x + (Math.random() - 0.5) * 0.5, y + 0.1, z + (Math.random() - 0.5) * 0.5);
      this.particleRoot.add(mesh);
      this.particles.push({
        mesh,
        vel: new THREE.Vector3((Math.random() - 0.5) * 2.8, 2.2 + Math.random() * 2.4, (Math.random() - 0.5) * 2.8),
        life: 0,
        maxLife: 0.35 + Math.random() * 0.25,
        gravity: 12.0,
        spin: new THREE.Vector3(4, 4, 4),
      });
    }
  }

  private spawnTorchSpark(pos: THREE.Vector3): void {
    const isFlame = Math.random() > 0.38;
    const mat = new THREE.MeshBasicMaterial({
      color: isFlame ? (Math.random() > 0.5 ? 0xffbb33 : 0xff6611) : 0x555555,
      transparent: true,
      opacity: isFlame ? 0.95 : 0.45,
    });
    const mesh = new THREE.Mesh(this.sharedBoxGeom, mat);
    mesh.scale.setScalar(isFlame ? 0.055 : 0.085);
    mesh.position.set(
      pos.x + (Math.random() - 0.5) * 0.08,
      pos.y + Math.random() * 0.05,
      pos.z + (Math.random() - 0.5) * 0.08
    );
    this.particleRoot.add(mesh);
    this.particles.push({
      mesh,
      vel: new THREE.Vector3((Math.random() - 0.5) * 0.22, isFlame ? 0.45 : 0.72, (Math.random() - 0.5) * 0.22),
      life: 0,
      maxLife: isFlame ? 0.45 : 0.85,
      gravity: -0.15,
      spin: new THREE.Vector3(2, 3, 1),
    });
  }

  public update(
    dt: number,
    elapsed: number,
    playerPos: THREE.Vector3,
    sunDir: THREE.Vector3,
    sunColor: THREE.Color,
    dayFactor: number
  ): void {
    // Sync mob lighting with live shader sun/moon
    this.mobSunLight.position.copy(playerPos).addScaledVector(sunDir, 40);
    this.mobSunLight.color.copy(sunColor);
    this.mobSunLight.intensity = 0.45 + 1.15 * dayFactor;
    this.mobAmbientLight.intensity = 0.28 + 0.65 * dayFactor;

    // Update passive Minecraft mobs
    for (const m of this.mobs) {
      m.walkTimer -= dt;
      if (m.walkTimer <= 0) {
        m.moving = Math.random() < 0.62;
        m.walkTimer = m.moving ? 2.0 + Math.random() * 3.5 : 1.5 + Math.random() * 3.0;
        m.targetYaw = Math.random() * Math.PI * 2;
      }

      // Smooth yaw turn
      let diff = m.targetYaw - m.yaw;
      while (diff > Math.PI) diff -= Math.PI * 2;
      while (diff < -Math.PI) diff += Math.PI * 2;
      m.yaw += diff * Math.min(1, dt * 4.5);

      const inWater = this.world.getBlock(Math.floor(m.pos.x), Math.floor(m.pos.y + 0.2), Math.floor(m.pos.z)) === BlockId.WATER;
      const speed = m.moving ? (m.kind === 'chicken' ? 1.5 : 1.25) : 0;

      const fx = Math.sin(m.yaw);
      const fz = Math.cos(m.yaw);

      const nextX = m.pos.x + fx * speed * dt;
      const nextZ = m.pos.z + fz * speed * dt;

      // Gravity / water buoyancy
      if (inWater) {
        m.vel.y = Math.min(2.2, m.vel.y + 8.5 * dt);
      } else {
        m.vel.y -= (m.kind === 'chicken' ? 10 : 20) * dt;
      }

      // Horizontal obstacle check & auto-step jump
      const checkX = Math.floor(nextX + fx * m.width * 0.45);
      const checkZ = Math.floor(nextZ + fz * m.width * 0.45);
      const feetY = Math.floor(m.pos.y);
      const blockedLow = isBlockSolid(this.world.getBlock(checkX, feetY, checkZ));
      const blockedHigh = isBlockSolid(this.world.getBlock(checkX, feetY + 1, checkZ));

      if (!blockedLow) {
        m.pos.x = nextX;
        m.pos.z = nextZ;
      } else if (!blockedHigh && m.vel.y <= 0.1) {
        // Hop up 1-block ledge
        m.vel.y = 6.2;
      } else {
        m.targetYaw += Math.PI * (0.6 + Math.random() * 0.8);
      }

      // Vertical resolve
      const nextY = m.pos.y + m.vel.y * dt;
      if (m.vel.y < 0 && isBlockSolid(this.world.getBlock(Math.floor(m.pos.x), Math.floor(nextY), Math.floor(m.pos.z)))) {
        m.pos.y = Math.floor(nextY) + 1;
        m.vel.y = 0;
      } else {
        m.pos.y = nextY;
      }

      // Keep mobs loosely in the player's world neighborhood
      if (m.pos.distanceTo(playerPos) > 65 || m.pos.y < 2) {
        const ang = Math.random() * Math.PI * 2;
        const rx = playerPos.x + Math.cos(ang) * (16 + Math.random() * 14);
        const rz = playerPos.z + Math.sin(ang) * (16 + Math.random() * 14);
        m.pos.set(rx, this.world.getSurfaceY(rx, rz) + 0.2, rz);
        m.vel.set(0, 0, 0);
      }

      m.group.position.copy(m.pos);
      m.group.rotation.y = m.yaw;

      // Animate legs & head bob
      if (m.moving) {
        m.walkPhase += dt * 7.5;
        const swing = Math.sin(m.walkPhase) * 0.55;
        for (let i = 0; i < m.legs.length; i++) {
          m.legs[i].rotation.x = i % 2 === 0 ? swing : -swing;
        }
      } else {
        for (const leg of m.legs) leg.rotation.x *= 0.85;
      }
      m.head.rotation.x = Math.sin(elapsed * 1.4 + m.pos.x) * 0.1;
    }

    // Spawn warm torch embers & smoke
    this.torchTimer += dt;
    if (this.torchTimer > 0.18 && this.particles.length < 180) {
      this.torchTimer = 0;
      for (const tp of this.world.torchPositions.values()) {
        if (tp.distanceToSquared(playerPos) < 38 * 38 && Math.random() < 0.55) {
          this.spawnTorchSpark(tp);
        }
      }
    }

    // Update particles
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.life += dt;
      if (p.life >= p.maxLife) {
        this.particleRoot.remove(p.mesh);
        this.particles.splice(i, 1);
        continue;
      }
      p.vel.y -= p.gravity * dt;
      p.mesh.position.addScaledVector(p.vel, dt);
      p.mesh.rotation.x += p.spin.x * dt;
      p.mesh.rotation.y += p.spin.y * dt;
      const t = 1 - p.life / p.maxLife;
      p.mesh.scale.multiplyScalar(0.985 + 0.015 * t);
    }

    // Fireflies glow gently at twilight & night
    const ffMat = this.fireflies.material as THREE.PointsMaterial;
    const nightGlow = Math.max(0, 1 - dayFactor * 1.4);
    ffMat.opacity = nightGlow * (0.55 + 0.3 * Math.sin(elapsed * 2.5));
    this.fireflies.visible = nightGlow > 0.04;
    if (this.fireflies.visible) {
      this.fireflies.position.set(
        playerPos.x + Math.sin(elapsed * 0.35) * 1.8,
        playerPos.y - 3 + Math.cos(elapsed * 0.6) * 0.6,
        playerPos.z + Math.cos(elapsed * 0.3) * 1.8
      );
    }
  }
}
