import './ui/styles.css';
import * as THREE from 'three';
import { SoundEngine } from './audio/SoundEngine';
import { MobAndParticleSystem } from './entities/MobSystem';
import { PlayerController } from './player/PlayerController';
import { ShaderPipeline } from './shaders/ShaderPipeline';
import { ProceduralAtlas } from './textures/ProceduralAtlas';
import { DEFAULT_SHADER_SETTINGS, ShaderSettings, WorldSaveData } from './types';
import { GameHUD } from './ui/HUD';
import { World } from './world/World';

const SAVE_STORAGE_KEY = 'mc_realistic_shaders_save_v1';

function loadSave(): WorldSaveData | null {
  try {
    const raw = localStorage.getItem(SAVE_STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as WorldSaveData;
  } catch {
    return null;
  }
}

function bootstrap() {
  const canvas = document.getElementById('game-canvas') as HTMLCanvasElement;
  const hudRoot = document.getElementById('hud-root') as HTMLElement;
  const underwaterOverlay = document.getElementById('underwater-overlay') as HTMLElement;
  const damageOverlay = document.getElementById('damage-overlay') as HTMLElement;

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    DEFAULT_SHADER_SETTINGS.fov,
    window.innerWidth / window.innerHeight,
    0.1,
    520
  );

  const settings: ShaderSettings = { ...DEFAULT_SHADER_SETTINGS };
  const saved = loadSave();
  const initialSeed = saved?.seed ?? 20260719;
  if (saved?.timeOfDay !== undefined) {
    settings.timeOfDay = saved.timeOfDay;
  }

  const atlas = new ProceduralAtlas();
  const shaders = new ShaderPipeline(renderer, atlas, settings);
  scene.add(shaders.skyMesh);
  scene.add(shaders.cloudMesh);
  scene.add(shaders.rainSystem);

  const world = new World(scene, atlas, shaders.materials, initialSeed, saved?.edits);
  world.setRenderDistance(settings.renderDistance);

  const sound = new SoundEngine();
  sound.setVolume(settings.masterVolume);
  sound.setAmbientEnabled(settings.ambientMusic);

  const mobs = new MobAndParticleSystem(scene, world);
  const player = new PlayerController(camera, scene, world, atlas, sound, mobs);

  // Build initial terrain ring around spawn synchronously so first frame is immediately rich
  const spawnX = saved?.playerPos?.[0] ?? 0.5;
  const spawnZ = saved?.playerPos?.[2] ?? -2.5;
  world.generateInitialArea(spawnX, spawnZ, Math.min(settings.renderDistance, 4));

  if (saved?.playerPos) {
    player.position.set(saved.playerPos[0], saved.playerPos[1], saved.playerPos[2]);
    player.yaw = saved.playerYaw ?? Math.PI;
    player.pitch = saved.playerPitch ?? -0.14;
    player.gameMode = saved.gameMode ?? 'survival';
    if (saved.hotbar && saved.hotbar.length === 9) {
      player.hotbar = saved.hotbar;
    }
    player.selectSlot(saved.selectedSlot ?? 0);
  } else {
    const sy = world.getSurfaceY(spawnX, spawnZ);
    player.position.set(spawnX, sy + 0.2, spawnZ);
    // Face +Z toward the sunlit lake, rustic cabin, trees & snow-capped mountains!
    player.yaw = Math.PI;
    player.pitch = -0.14;
  }

  mobs.spawnInitialMobs(player.position.x, player.position.z);

  const saveWorldState = () => {
    try {
      const editsObj: Record<string, number> = {};
      for (const [k, v] of world.edits.entries()) editsObj[k] = v;
      const data: WorldSaveData = {
        seed: world.worldGen.seed,
        playerPos: [player.position.x, player.position.y, player.position.z],
        playerYaw: player.yaw,
        playerPitch: player.pitch,
        gameMode: player.gameMode,
        timeOfDay: settings.timeOfDay,
        hotbar: player.hotbar,
        selectedSlot: player.selectedSlot,
        edits: editsObj,
      };
      localStorage.setItem(SAVE_STORAGE_KEY, JSON.stringify(data));
    } catch {
      // ignore storage quota errors
    }
  };

  const hud = new GameHUD(hudRoot, atlas, player, world, settings, sound, {
    onResumeGame: () => {
      canvas.requestPointerLock?.();
    },
    onSettingsChanged: () => {
      shaders.applySettings(settings);
      world.setRenderDistance(settings.renderDistance);
      camera.fov = settings.fov;
      camera.updateProjectionMatrix();
      hud.refreshHotbar();
    },
    onRegenerateWorld: (newSeed: number) => {
      world.resetWorld(newSeed);
      world.generateInitialArea(0.5, -2.5, Math.min(settings.renderDistance, 4));
      const sy = world.getSurfaceY(0.5, -2.5);
      player.position.set(0.5, sy + 0.2, -2.5);
      player.velocity.set(0, 0, 0);
      player.yaw = Math.PI;
      player.pitch = -0.14;
      mobs.spawnInitialMobs(0.5, -2.5);
      saveWorldState();
    },
    onSaveWorld: saveWorldState,
  });

  player.onInventoryChanged = () => {
    hud.refreshHotbar();
    hud.refreshStatusBars();
  };
  player.onWorldEdited = () => {
    saveWorldState();
  };
  player.onToggleInventory = () => hud.toggleInventory();
  player.onTogglePause = () => hud.openPauseMenu();
  player.onToggleDebug = () => hud.toggleDebug();

  // Clicking directly on canvas while menus are closed re-locks pointer
  canvas.addEventListener('click', () => {
    if (!player.uiPaused && !player.isPointerLocked) {
      sound.unlock();
      canvas.requestPointerLock?.();
    }
  });

  window.addEventListener('resize', () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    player.resize(w / h);
    renderer.setSize(w, h);
    shaders.resize(w, h);
  });

  // Expose lightweight runtime hook for automated visual testing / time-of-day showcase
  (window as unknown as Record<string, unknown>).__MINECRAFT__ = {
    settings,
    player,
    world,
    hud,
    applySettings: () => {
      shaders.applySettings(settings);
      world.setRenderDistance(settings.renderDistance);
      hud.refreshHotbar();
    },
    enterLiveView: () => {
      hud.hideAllMenus();
    },
  };

  // Main 60fps Game & Shader Loop
  let lastTime = performance.now();
  let elapsed = 0;
  let fpsSmooth = 60;
  let hudRefreshAccum = 0;
  let panoramaAngle = Math.PI - 0.12;

  const animate = (now: number) => {
    requestAnimationFrame(animate);
    const dt = Math.min((now - lastTime) / 1000, 0.08);
    lastTime = now;
    elapsed += dt;
    if (dt > 0) fpsSmooth = fpsSmooth * 0.9 + (1 / dt) * 0.1;

    // Advance dynamic day/night cycle
    if (settings.dayCycleSpeed > 0) {
      settings.timeOfDay = (settings.timeOfDay + (dt * settings.dayCycleSpeed) / 1200) % 1;
    }

    // Stream chunks around player incrementally
    world.updateAroundPlayer(player.position.x, player.position.z, 2, 3);

    if (player.uiPaused && !player.isPointerLocked) {
      // Live rotating Minecraft panorama camera behind the title/pause menu!
      panoramaAngle += dt * 0.06;
      camera.position.set(player.position.x, player.position.y + 1.9, player.position.z);
      camera.rotation.order = 'YXZ';
      camera.rotation.y = panoramaAngle;
      camera.rotation.x = -0.15;
    } else {
      player.update(dt, settings.fov, shaders.getSunColor(), shaders.getDayFactor());
    }

    shaders.update(dt, elapsed, camera, settings, player.eyesUnderwater);
    mobs.update(dt, elapsed, player.position, shaders.getSunDir(), shaders.getSunColor(), shaders.getDayFactor());
    sound.updateAmbient(dt);

    // Render shadow map + HDR scene + first-person hand + Bloom/God Rays/ACES Post-Process!
    shaders.render(
      scene,
      camera,
      world.chunks.values(),
      settings,
      !player.uiPaused ? player.viewmodelScene : undefined,
      !player.uiPaused ? player.viewmodelCamera : undefined
    );

    // DOM overlays
    underwaterOverlay.style.opacity = player.eyesUnderwater ? '1' : '0';
    damageOverlay.style.opacity = player.damageFlash.toFixed(2);

    hudRefreshAccum += dt;
    if (hudRefreshAccum > 0.15) {
      hudRefreshAccum = 0;
      hud.refreshStatusBars();
      hud.updateDebug(fpsSmooth);
    }
  };

  requestAnimationFrame(animate);
}

bootstrap();
