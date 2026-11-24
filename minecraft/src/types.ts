export type GameMode = 'survival' | 'creative';
export type ShaderQuality = 'ultra' | 'high' | 'medium' | 'vanilla';
export type WeatherType = 'clear' | 'rain';
export type BiomeType = 'plains' | 'forest' | 'birch_forest' | 'taiga' | 'desert' | 'mountains' | 'river';

export type BlockShape = 'cube' | 'cross' | 'torch' | 'water' | 'cactus';
export type BlockSoundGroup = 'grass' | 'stone' | 'wood' | 'sand' | 'gravel' | 'glass' | 'snow' | 'wool' | 'water' | 'leaves';
export type BlockCategory = 'building' | 'natural' | 'ores' | 'lighting' | 'plants';

export interface BlockDefinition {
  id: number;
  key: string;
  name: string;
  nameZh: string;
  solid: boolean;
  transparent: boolean;
  shape: BlockShape;
  hardness: number; // seconds to break by hand in survival (0 = instant, -1 = unbreakable)
  lightLevel: number; // 0..15 emitted block light
  sound: BlockSoundGroup;
  category: BlockCategory;
  // Tile indices in ProceduralAtlas
  tiles: {
    top: number;
    bottom: number;
    north: number;
    south: number;
    east: number;
    west: number;
  };
  tintTop?: boolean; // biome grass/foliage tint
  tintAll?: boolean;
  windSway?: boolean;
  flammable?: boolean;
  dropId?: number;
}

export interface ShaderSettings {
  quality: ShaderQuality;
  shadowsEnabled: boolean;
  shadowResolution: number;
  realisticWater: boolean;
  volumetricClouds: boolean;
  windAnimation: boolean;
  bloomAndGodRays: boolean;
  pbrNormalMapping: boolean;
  renderDistance: number; // in chunks (radius)
  fov: number;
  timeOfDay: number; // 0..1 (0.25 = sunrise, 0.5 = noon, 0.75 = sunset, 0/1 = midnight)
  dayCycleSpeed: number; // 0 = paused, 1 = normal (20min), 10 = fast showcase
  weather: WeatherType;
  ambientMusic: boolean;
  masterVolume: number;
}

export interface HotbarSlot {
  blockId: number;
  count: number;
}

export interface RaycastHit {
  x: number;
  y: number;
  z: number;
  nx: number;
  ny: number;
  nz: number;
  blockId: number;
  distance: number;
}

export interface WorldSaveData {
  seed: number;
  playerPos: [number, number, number];
  playerYaw: number;
  playerPitch: number;
  gameMode: GameMode;
  timeOfDay: number;
  hotbar: HotbarSlot[];
  selectedSlot: number;
  edits: Record<string, number>; // "x,y,z" -> blockId
}

export const CHUNK_SIZE = 16;
export const CHUNK_HEIGHT = 128;
export const SEA_LEVEL = 54;

export const DEFAULT_SHADER_SETTINGS: ShaderSettings = {
  quality: 'ultra',
  shadowsEnabled: true,
  shadowResolution: 2048,
  realisticWater: true,
  volumetricClouds: true,
  windAnimation: true,
  bloomAndGodRays: true,
  pbrNormalMapping: true,
  renderDistance: 6,
  fov: 75,
  timeOfDay: 0.30, // Golden morning sunlight for an iconic opening vista!
  dayCycleSpeed: 3, // Gentle live day-night cycle
  weather: 'clear',
  ambientMusic: true,
  masterVolume: 0.65,
};
