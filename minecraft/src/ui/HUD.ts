import { SoundEngine } from '../audio/SoundEngine';
import { PlayerController } from '../player/PlayerController';
import { ProceduralAtlas } from '../textures/ProceduralAtlas';
import { BlockCategory, ShaderQuality, ShaderSettings } from '../types';
import { BlockId, getBlockDef, PLACEABLE_BLOCKS } from '../world/blocks';
import { World } from '../world/World';

const HEART_FULL_SVG = `<svg class="mc-icon-svg" viewBox="0 0 16 16"><path fill="#140808" d="M2 2h4v1h1v1h2V3h1V2h4v1h1v5h-1v2h-1v1h-1v1h-1v1H9v1H7v-1H5v-1H4v-1H3v-1H2V8H1V3h1z"/><path fill="#d81f1f" d="M2 3h4v1h1v1h2V4h1V3h4v5h-1v2h-1v1h-1v1H9v1H7v-1H5v-1H4v-1H3V8H2z"/><path fill="#ff7878" d="M3 4h2v2H3z"/></svg>`;
const HEART_HALF_SVG = `<svg class="mc-icon-svg" viewBox="0 0 16 16"><path fill="#140808" d="M2 2h4v1h1v1h2V3h1V2h4v1h1v5h-1v2h-1v1h-1v1h-1v1H9v1H7v-1H5v-1H4v-1H3v-1H2V8H1V3h1z"/><path fill="#3a2424" d="M2 3h4v1h1v1h2V4h1V3h4v5h-1v2h-1v1h-1v1H9v1H7v-1H5v-1H4v-1H3V8H2z"/><path fill="#d81f1f" d="M2 3h4v1h1v1h1v8H7v-1H5v-1H4v-1H3V8H2z"/><path fill="#ff7878" d="M3 4h2v2H3z"/></svg>`;
const HEART_EMPTY_SVG = `<svg class="mc-icon-svg" viewBox="0 0 16 16"><path fill="#140808" d="M2 2h4v1h1v1h2V3h1V2h4v1h1v5h-1v2h-1v1h-1v1h-1v1H9v1H7v-1H5v-1H4v-1H3v-1H2V8H1V3h1z"/><path fill="#3a2424" d="M2 3h4v1h1v1h2V4h1V3h4v5h-1v2h-1v1h-1v1H9v1H7v-1H5v-1H4v-1H3V8H2z"/></svg>`;

const HUNGER_FULL_SVG = `<svg class="mc-icon-svg" viewBox="0 0 16 16"><path fill="#1c1108" d="M5 2h6v1h2v5h-1v2h-2v1H8v2H6v2H3v-2h1v-2h2V9H4V4h1z"/><path fill="#c87c3c" d="M6 3h5v5h-1v2H7V8H5V4h1z"/><path fill="#e8ab6c" d="M7 4h3v2H7z"/><path fill="#f0ead6" d="M4 12h2v2H4z"/></svg>`;
const BUBBLE_SVG = `<svg class="mc-icon-svg" viewBox="0 0 16 16"><circle cx="8" cy="8" r="6" fill="#2aa8f2" stroke="#083864" stroke-width="2"/><rect x="5" y="4" width="3" height="2" fill="#d8f6ff"/></svg>`;

export interface HudCallbacks {
  onResumeGame: () => void;
  onSettingsChanged: () => void;
  onRegenerateWorld: (seed: number) => void;
  onSaveWorld: () => void;
}

export class GameHUD {
  private readonly root: HTMLElement;
  private readonly atlas: ProceduralAtlas;
  private readonly player: PlayerController;
  private readonly world: World;
  private readonly settings: ShaderSettings;
  private readonly sound: SoundEngine;
  private readonly callbacks: HudCallbacks;

  private crosshairEl!: HTMLElement;
  private hotbarEl!: HTMLElement;
  private itemLabelEl!: HTMLElement;
  private heartsEl!: HTMLElement;
  private hungerEl!: HTMLElement;
  private airEl!: HTMLElement;
  private statusRowEl!: HTMLElement;
  private xpWrapEl!: HTMLElement;
  private f3El!: HTMLElement;
  private modeBadgeEl!: HTMLElement;

  private menuOverlay!: HTMLElement;
  private menuResumeBtn!: HTMLButtonElement;
  private settingsModal!: HTMLElement;
  private inventoryModal!: HTMLElement;
  private invGridEl!: HTMLElement;
  private invHintEl!: HTMLElement;

  private activeCategory: BlockCategory | 'all' = 'all';
  private isTitleScreen = true;
  private debugVisible = false;
  private itemLabelTimeout = 0;

  constructor(
    root: HTMLElement,
    atlas: ProceduralAtlas,
    player: PlayerController,
    world: World,
    settings: ShaderSettings,
    sound: SoundEngine,
    callbacks: HudCallbacks
  ) {
    this.root = root;
    this.atlas = atlas;
    this.player = player;
    this.world = world;
    this.settings = settings;
    this.sound = sound;
    this.callbacks = callbacks;

    this.buildDOM();
    this.refreshHotbar();
    this.refreshStatusBars();
  }

  private buildDOM(): void {
    this.root.innerHTML = '';

    // Crosshair
    this.crosshairEl = document.createElement('div');
    this.crosshairEl.className = 'mc-crosshair';
    this.crosshairEl.style.display = 'none';
    this.root.appendChild(this.crosshairEl);

    // Top-right badge
    const hints = document.createElement('div');
    hints.className = 'mc-top-hints';
    this.modeBadgeEl = document.createElement('div');
    this.modeBadgeEl.className = 'mc-badge';
    hints.appendChild(this.modeBadgeEl);
    const keyHint = document.createElement('div');
    keyHint.className = 'mc-badge';
    keyHint.innerHTML = '<strong>E</strong> 背包 · <strong>F3</strong> 调试 · <strong>G</strong> 模式 · <strong>ESC</strong> 光影与菜单';
    hints.appendChild(keyHint);
    this.root.appendChild(hints);

    // F3 Debug screen
    this.f3El = document.createElement('div');
    this.f3El.className = 'mc-f3';
    this.root.appendChild(this.f3El);

    // Bottom HUD container
    const bottom = document.createElement('div');
    bottom.className = 'mc-bottom-hud';

    this.itemLabelEl = document.createElement('div');
    this.itemLabelEl.className = 'mc-item-label';
    bottom.appendChild(this.itemLabelEl);

    this.airEl = document.createElement('div');
    this.airEl.className = 'mc-air-row';
    bottom.appendChild(this.airEl);

    this.statusRowEl = document.createElement('div');
    this.statusRowEl.className = 'mc-status-row';
    this.heartsEl = document.createElement('div');
    this.heartsEl.className = 'mc-status-bar';
    this.hungerEl = document.createElement('div');
    this.hungerEl.className = 'mc-status-bar';
    this.statusRowEl.appendChild(this.heartsEl);
    this.statusRowEl.appendChild(this.hungerEl);
    bottom.appendChild(this.statusRowEl);

    this.xpWrapEl = document.createElement('div');
    this.xpWrapEl.className = 'mc-xp-wrap';
    this.xpWrapEl.innerHTML = `<div class="mc-xp-fill" style="width:${Math.round(this.player.xpProgress * 100)}%"></div><div class="mc-xp-level">${this.player.xpLevel}</div>`;
    bottom.appendChild(this.xpWrapEl);

    this.hotbarEl = document.createElement('div');
    this.hotbarEl.className = 'mc-hotbar';
    bottom.appendChild(this.hotbarEl);

    this.root.appendChild(bottom);

    this.buildTitleAndPauseOverlay();
    this.buildSettingsModal();
    this.buildInventoryModal();
  }

  private buildTitleAndPauseOverlay(): void {
    this.menuOverlay = document.createElement('div');
    this.menuOverlay.className = 'mc-overlay';

    const logoWrap = document.createElement('div');
    logoWrap.className = 'mc-logo-wrap';
    logoWrap.innerHTML = `
      <div class="mc-logo">MINECRAFT</div>
      <div class="mc-subtitle">Realistic PBR Shaders · Java 1:1 Web Edition</div>
      <div class="mc-splash">光影实时渲染!</div>
    `;
    this.menuOverlay.appendChild(logoWrap);

    const panel = document.createElement('div');
    panel.className = 'mc-menu-panel';

    this.menuResumeBtn = document.createElement('button');
    this.menuResumeBtn.className = 'mc-btn primary';
    this.menuResumeBtn.textContent = '▶ 开始游戏 (Play World)';
    this.menuResumeBtn.onclick = () => {
      this.sound.unlock();
      this.sound.playUiClick();
      this.isTitleScreen = false;
      this.hideAllMenus();
      this.callbacks.onResumeGame();
    };
    panel.appendChild(this.menuResumeBtn);

    const row1 = document.createElement('div');
    row1.className = 'mc-btn-row';

    const shaderBtn = document.createElement('button');
    shaderBtn.className = 'mc-btn';
    shaderBtn.textContent = '☀ 光影与画质设置';
    shaderBtn.onclick = () => {
      this.sound.unlock();
      this.sound.playUiClick();
      this.openSettings();
    };
    row1.appendChild(shaderBtn);

    const invBtn = document.createElement('button');
    invBtn.className = 'mc-btn';
    invBtn.textContent = '▦ 方块背包 (E)';
    invBtn.onclick = () => {
      this.sound.unlock();
      this.sound.playUiClick();
      this.openInventory();
    };
    row1.appendChild(invBtn);
    panel.appendChild(row1);

    const row2 = document.createElement('div');
    row2.className = 'mc-btn-row';

    const modeBtn = document.createElement('button');
    modeBtn.className = 'mc-btn';
    const syncModeBtn = () => {
      modeBtn.textContent = this.player.gameMode === 'survival' ? '⚔ 模式: 生存 (切换创造)' : '✦ 模式: 创造 (切换生存)';
    };
    syncModeBtn();
    modeBtn.onclick = () => {
      this.sound.unlock();
      this.sound.playUiClick();
      this.player.toggleGameMode();
      syncModeBtn();
      this.refreshStatusBars();
    };
    row2.appendChild(modeBtn);

    const weatherBtn = document.createElement('button');
    weatherBtn.className = 'mc-btn';
    const syncWeatherBtn = () => {
      weatherBtn.textContent = this.settings.weather === 'clear' ? '🌤 天气: 晴朗 (切换雨天)' : '🌧 天气: 降雨 (切换晴朗)';
    };
    syncWeatherBtn();
    weatherBtn.onclick = () => {
      this.sound.unlock();
      this.sound.playUiClick();
      this.settings.weather = this.settings.weather === 'clear' ? 'rain' : 'clear';
      syncWeatherBtn();
      this.callbacks.onSettingsChanged();
    };
    row2.appendChild(weatherBtn);
    panel.appendChild(row2);

    // Quick time-of-day bar right on pause menu for instant shader showcase!
    const timeRow = document.createElement('div');
    timeRow.className = 'mc-chip-row';
    const presets: [string, number][] = [
      ['🌅 黎明', 0.27],
      ['☀ 正午', 0.48],
      ['🌇 黄金日落', 0.73],
      ['🌙 星夜', 0.02],
    ];
    for (const [label, t] of presets) {
      const btn = document.createElement('button');
      btn.className = 'mc-btn';
      btn.style.flex = '1';
      btn.style.padding = '8px 6px';
      btn.style.fontSize = '13px';
      btn.textContent = label;
      btn.onclick = () => {
        this.sound.unlock();
        this.sound.playUiClick();
        this.settings.timeOfDay = t;
        this.callbacks.onSettingsChanged();
      };
      timeRow.appendChild(btn);
    }
    panel.appendChild(timeRow);

    const controls = document.createElement('div');
    controls.className = 'mc-controls-card';
    controls.innerHTML = `
      <div><kbd>W A S D</kbd> 移动 · <kbd>Space</kbd> 跳跃/游泳 (创造双击飞行) · <kbd>Shift</kbd> 潜行防摔 · <kbd>Ctrl</kbd> 疾跑</div>
      <div><kbd>左键</kbd> 挖掘/破坏 · <kbd>右键</kbd> 放置方块 · <kbd>中键</kbd> 吸取方块 · <kbd>1-9</kbd> / 滚轮切换快捷栏</div>
      <div><kbd>E</kbd> 打开完整方块目录 · <kbd>F3</kbd> 调试信息 · <kbd>G</kbd> 切换生存/创造 · 世界改动自动保存</div>
    `;
    panel.appendChild(controls);

    this.menuOverlay.appendChild(panel);
    this.root.appendChild(this.menuOverlay);
  }

  private buildSettingsModal(): void {
    this.settingsModal = document.createElement('div');
    this.settingsModal.className = 'mc-modal hidden';

    const card = document.createElement('div');
    card.className = 'mc-modal-card';
    this.settingsModal.appendChild(card);

    const renderSettings = () => {
      card.innerHTML = '';

      const title = document.createElement('div');
      title.className = 'mc-modal-title';
      title.innerHTML = `<span>光影与世界设置 (Shaders & Video)</span>`;
      card.appendChild(title);

      // Quality Preset Row
      const presetBox = document.createElement('div');
      presetBox.className = 'mc-setting-item';
      presetBox.innerHTML = `<div class="mc-setting-header"><span>光影画质预设</span><span class="mc-setting-val">${this.settings.quality.toUpperCase()}</span></div>`;
      const chipRow = document.createElement('div');
      chipRow.className = 'mc-chip-row';
      const qualities: [ShaderQuality, string][] = [
        ['ultra', '极致 Ultra'],
        ['high', '高光影 High'],
        ['medium', '中等 Medium'],
        ['vanilla', '原版 Vanilla'],
      ];
      for (const [q, label] of qualities) {
        const c = document.createElement('button');
        c.className = `mc-chip ${this.settings.quality === q ? 'active' : ''}`;
        c.textContent = label;
        c.onclick = () => {
          this.sound.playUiClick();
          this.applyQualityPreset(q);
          renderSettings();
          this.callbacks.onSettingsChanged();
        };
        chipRow.appendChild(c);
      }
      presetBox.appendChild(chipRow);
      card.appendChild(presetBox);

      const grid = document.createElement('div');
      grid.className = 'mc-setting-grid';

      // Sliders
      const addSlider = (
        label: string,
        min: number,
        max: number,
        step: number,
        val: number,
        fmt: (v: number) => string,
        onInput: (v: number) => void
      ) => {
        const item = document.createElement('div');
        item.className = 'mc-setting-item';
        const valSpan = document.createElement('span');
        valSpan.className = 'mc-setting-val';
        valSpan.textContent = fmt(val);
        const hdr = document.createElement('div');
        hdr.className = 'mc-setting-header';
        hdr.innerHTML = `<span>${label}</span>`;
        hdr.appendChild(valSpan);
        const inp = document.createElement('input');
        inp.type = 'range';
        inp.min = String(min);
        inp.max = String(max);
        inp.step = String(step);
        inp.value = String(val);
        inp.oninput = () => {
          const nv = parseFloat(inp.value);
          valSpan.textContent = fmt(nv);
          onInput(nv);
          this.callbacks.onSettingsChanged();
        };
        item.appendChild(hdr);
        item.appendChild(inp);
        grid.appendChild(item);
      };

      const formatHour = (t: number) => {
        const totalMin = Math.floor(((t + 0.0) % 1) * 24 * 60);
        const hh = Math.floor(totalMin / 60).toString().padStart(2, '0');
        const mm = (totalMin % 60).toString().padStart(2, '0');
        return `${hh}:${mm}`;
      };

      addSlider('时间 (Time of Day)', 0, 1, 0.005, this.settings.timeOfDay, formatHour, (v) => {
        this.settings.timeOfDay = v;
      });
      addSlider('昼夜流速', 0, 20, 1, this.settings.dayCycleSpeed, (v) => (v === 0 ? '暂停' : `${v}x`), (v) => {
        this.settings.dayCycleSpeed = v;
      });
      addSlider('视距 (Render Distance)', 3, 9, 1, this.settings.renderDistance, (v) => `${v} 区块`, (v) => {
        this.settings.renderDistance = v;
      });
      addSlider('视野 FOV', 60, 105, 1, this.settings.fov, (v) => `${v}°`, (v) => {
        this.settings.fov = v;
      });
      addSlider('主音量', 0, 1, 0.05, this.settings.masterVolume, (v) => `${Math.round(v * 100)}%`, (v) => {
        this.settings.masterVolume = v;
        this.sound.setVolume(v);
      });

      // Toggles
      const addToggle = (label: string, get: () => boolean, set: (v: boolean) => void) => {
        const item = document.createElement('div');
        item.className = 'mc-setting-item';
        const hdr = document.createElement('div');
        hdr.className = 'mc-setting-header';
        hdr.innerHTML = `<span>${label}</span>`;
        const btn = document.createElement('button');
        const sync = () => {
          const on = get();
          btn.className = `mc-toggle-btn ${on ? 'on' : ''}`;
          btn.textContent = on ? '开启 ON' : '关闭 OFF';
        };
        sync();
        btn.onclick = () => {
          this.sound.playUiClick();
          set(!get());
          sync();
          this.callbacks.onSettingsChanged();
        };
        hdr.appendChild(btn);
        item.appendChild(hdr);
        grid.appendChild(item);
      };

      addToggle('实时软阴影 (PCF Shadows)', () => this.settings.shadowsEnabled, (v) => (this.settings.shadowsEnabled = v));
      addToggle('真实水体反射 (Shader Water)', () => this.settings.realisticWater, (v) => (this.settings.realisticWater = v));
      addToggle('PBR 法线与反光材质', () => this.settings.pbrNormalMapping, (v) => (this.settings.pbrNormalMapping = v));
      addToggle('辉光与体积丁达尔光', () => this.settings.bloomAndGodRays, (v) => (this.settings.bloomAndGodRays = v));
      addToggle('3D 体积光影云层', () => this.settings.volumetricClouds, (v) => (this.settings.volumetricClouds = v));
      addToggle('树叶与青草风动', () => this.settings.windAnimation, (v) => (this.settings.windAnimation = v));
      addToggle('环境钢琴背景音乐', () => this.settings.ambientMusic, (v) => {
        this.settings.ambientMusic = v;
        this.sound.setAmbientEnabled(v);
      });

      card.appendChild(grid);

      // World Seed Box
      const seedBox = document.createElement('div');
      seedBox.className = 'mc-setting-item';
      seedBox.innerHTML = `<div class="mc-setting-header"><span>世界种子 (Seed)</span><span class="mc-setting-val">${this.world.worldGen.seed}</span></div>`;
      const seedRow = document.createElement('div');
      seedRow.style.display = 'flex';
      seedRow.style.gap = '8px';
      const seedInput = document.createElement('input');
      seedInput.className = 'mc-seed-input';
      seedInput.value = String(this.world.worldGen.seed);
      const regenBtn = document.createElement('button');
      regenBtn.className = 'mc-btn';
      regenBtn.style.width = '180px';
      regenBtn.style.padding = '8px';
      regenBtn.textContent = '生成新世界';
      regenBtn.onclick = () => {
        this.sound.playUiClick();
        const parsed = parseInt(seedInput.value.trim(), 10);
        const nextSeed = Number.isFinite(parsed) ? parsed : Math.floor(Math.random() * 9999999);
        this.callbacks.onRegenerateWorld(nextSeed);
        renderSettings();
      };
      seedRow.appendChild(seedInput);
      seedRow.appendChild(regenBtn);
      seedBox.appendChild(seedRow);
      card.appendChild(seedBox);

      const closeRow = document.createElement('div');
      closeRow.className = 'mc-btn-row';
      const saveBtn = document.createElement('button');
      saveBtn.className = 'mc-btn';
      saveBtn.textContent = '💾 立即保存世界';
      saveBtn.onclick = () => {
        this.sound.playUiClick();
        this.callbacks.onSaveWorld();
        saveBtn.textContent = '✓ 已保存!';
        setTimeout(() => (saveBtn.textContent = '💾 立即保存世界'), 1200);
      };
      const doneBtn = document.createElement('button');
      doneBtn.className = 'mc-btn primary';
      doneBtn.textContent = '完成并返回';
      doneBtn.onclick = () => {
        this.sound.playUiClick();
        this.settingsModal.classList.add('hidden');
        this.menuOverlay.classList.remove('hidden');
      };
      closeRow.appendChild(saveBtn);
      closeRow.appendChild(doneBtn);
      card.appendChild(closeRow);
    };

    (this.settingsModal as unknown as { _render: () => void })._render = renderSettings;
    this.root.appendChild(this.settingsModal);
  }

  private applyQualityPreset(q: ShaderQuality): void {
    this.settings.quality = q;
    if (q === 'ultra') {
      this.settings.shadowsEnabled = true;
      this.settings.shadowResolution = 2048;
      this.settings.realisticWater = true;
      this.settings.volumetricClouds = true;
      this.settings.windAnimation = true;
      this.settings.bloomAndGodRays = true;
      this.settings.pbrNormalMapping = true;
      this.settings.renderDistance = 6;
    } else if (q === 'high') {
      this.settings.shadowsEnabled = true;
      this.settings.shadowResolution = 1024;
      this.settings.realisticWater = true;
      this.settings.volumetricClouds = true;
      this.settings.windAnimation = true;
      this.settings.bloomAndGodRays = true;
      this.settings.pbrNormalMapping = true;
      this.settings.renderDistance = 5;
    } else if (q === 'medium') {
      this.settings.shadowsEnabled = true;
      this.settings.shadowResolution = 1024;
      this.settings.realisticWater = true;
      this.settings.volumetricClouds = true;
      this.settings.windAnimation = false;
      this.settings.bloomAndGodRays = false;
      this.settings.pbrNormalMapping = false;
      this.settings.renderDistance = 4;
    } else {
      this.settings.shadowsEnabled = false;
      this.settings.realisticWater = false;
      this.settings.volumetricClouds = true;
      this.settings.windAnimation = false;
      this.settings.bloomAndGodRays = false;
      this.settings.pbrNormalMapping = false;
    }
  }

  private buildInventoryModal(): void {
    this.inventoryModal = document.createElement('div');
    this.inventoryModal.className = 'mc-modal hidden';

    const card = document.createElement('div');
    card.className = 'mc-modal-card';

    const title = document.createElement('div');
    title.className = 'mc-modal-title';
    title.innerHTML = `<span>方块目录 / 背包 (点击方块放入当前快捷栏 #${this.player.selectedSlot + 1})</span>`;
    card.appendChild(title);

    const tabs = document.createElement('div');
    tabs.className = 'mc-chip-row';
    const cats: [BlockCategory | 'all', string][] = [
      ['all', '全部'],
      ['building', '建筑'],
      ['natural', '自然地形'],
      ['lighting', '光源与功能'],
      ['ores', '矿石'],
      ['plants', '植物'],
    ];
    for (const [cat, label] of cats) {
      const btn = document.createElement('button');
      btn.className = `mc-chip ${this.activeCategory === cat ? 'active' : ''}`;
      btn.textContent = label;
      btn.onclick = () => {
        this.sound.playUiClick();
        this.activeCategory = cat;
        for (const ch of Array.from(tabs.children)) ch.classList.remove('active');
        btn.classList.add('active');
        this.populateInventoryGrid();
      };
      tabs.appendChild(btn);
    }
    card.appendChild(tabs);

    this.invHintEl = document.createElement('div');
    this.invHintEl.style.fontSize = '13px';
    this.invHintEl.style.color = '#ffe27a';
    this.invHintEl.textContent = '悬停查看名称，点击即可装备到当前选中的快捷栏槽位';
    card.appendChild(this.invHintEl);

    this.invGridEl = document.createElement('div');
    this.invGridEl.className = 'mc-inv-grid';
    card.appendChild(this.invGridEl);

    const closeBtn = document.createElement('button');
    closeBtn.className = 'mc-btn primary';
    closeBtn.textContent = '关闭背包并继续游戏 (E)';
    closeBtn.onclick = () => {
      this.sound.playUiClick();
      this.hideAllMenus();
      this.callbacks.onResumeGame();
    };
    card.appendChild(closeBtn);

    this.inventoryModal.appendChild(card);
    this.root.appendChild(this.inventoryModal);
    this.populateInventoryGrid();
  }

  private populateInventoryGrid(): void {
    this.invGridEl.innerHTML = '';
    const list = PLACEABLE_BLOCKS.filter((id) => {
      if (this.activeCategory === 'all') return true;
      return getBlockDef(id).category === this.activeCategory;
    });

    for (const id of list) {
      const def = getBlockDef(id);
      const cell = document.createElement('div');
      cell.className = 'mc-inv-cell';
      const img = document.createElement('img');
      img.src = this.atlas.getBlockIconDataUrl(id);
      img.alt = def.name;
      cell.appendChild(img);
      cell.onmouseenter = () => {
        this.invHintEl.textContent = `${def.nameZh} (${def.name}) — 点击放入快捷栏 #${this.player.selectedSlot + 1}`;
      };
      cell.onclick = () => {
        this.sound.playUiClick();
        this.player.setHotbarSlot(this.player.selectedSlot, id, 64);
        this.invHintEl.textContent = `✓ 已将 ${def.nameZh} 放入快捷栏 #${this.player.selectedSlot + 1}`;
      };
      this.invGridEl.appendChild(cell);
    }
  }

  public refreshHotbar(): void {
    this.hotbarEl.innerHTML = '';
    for (let i = 0; i < 9; i++) {
      const slot = this.player.hotbar[i];
      const el = document.createElement('div');
      el.className = `mc-slot ${i === this.player.selectedSlot ? 'selected' : ''}`;
      const num = document.createElement('span');
      num.className = 'mc-slot-num';
      num.textContent = String(i + 1);
      el.appendChild(num);

      if (slot && slot.blockId !== BlockId.AIR && slot.count > 0) {
        const img = document.createElement('img');
        img.src = this.atlas.getBlockIconDataUrl(slot.blockId);
        el.appendChild(img);
        if (this.player.gameMode === 'survival' && slot.count > 1) {
          const cnt = document.createElement('span');
          cnt.className = 'mc-slot-count';
          cnt.textContent = String(slot.count);
          el.appendChild(cnt);
        }
      }

      el.onclick = () => {
        this.player.selectSlot(i);
      };
      this.hotbarEl.appendChild(el);
    }

    const sel = this.player.hotbar[this.player.selectedSlot];
    if (sel && sel.blockId !== BlockId.AIR && sel.count > 0) {
      const def = getBlockDef(sel.blockId);
      this.itemLabelEl.textContent = `${def.nameZh} · ${def.name}`;
      this.itemLabelEl.style.opacity = '1';
      window.clearTimeout(this.itemLabelTimeout);
      this.itemLabelTimeout = window.setTimeout(() => {
        this.itemLabelEl.style.opacity = '0.45';
      }, 2200);
    } else {
      this.itemLabelEl.textContent = '';
    }

    this.modeBadgeEl.innerHTML =
      this.player.gameMode === 'survival'
        ? `<strong>生存模式</strong> · 光影: ${this.settings.quality.toUpperCase()}`
        : `<strong>创造模式</strong>${this.player.flying ? ' (飞行中)' : ''} · 光影: ${this.settings.quality.toUpperCase()}`;
  }

  public refreshStatusBars(): void {
    const isSurvival = this.player.gameMode === 'survival';
    this.statusRowEl.style.display = isSurvival ? 'flex' : 'none';
    this.xpWrapEl.style.display = isSurvival ? 'block' : 'none';
    this.airEl.style.display = isSurvival && this.player.air < this.player.maxAir ? 'flex' : 'none';

    if (!isSurvival) return;

    // 10 Hearts
    let hHtml = '';
    for (let i = 0; i < 10; i++) {
      const hp = this.player.health - i * 2;
      hHtml += hp >= 2 ? HEART_FULL_SVG : hp === 1 ? HEART_HALF_SVG : HEART_EMPTY_SVG;
    }
    this.heartsEl.innerHTML = hHtml;

    // 10 Hunger drumsticks
    let fHtml = '';
    for (let i = 0; i < 10; i++) {
      fHtml += HUNGER_FULL_SVG;
    }
    this.hungerEl.innerHTML = fHtml;

    // Underwater bubbles
    if (this.player.air < this.player.maxAir) {
      const bubbles = Math.ceil((this.player.air / this.player.maxAir) * 10);
      this.airEl.innerHTML = BUBBLE_SVG.repeat(Math.max(0, bubbles));
    }
  }

  public openPauseMenu(): void {
    this.player.uiPaused = true;
    this.crosshairEl.style.display = 'none';
    this.settingsModal.classList.add('hidden');
    this.inventoryModal.classList.add('hidden');
    this.menuResumeBtn.textContent = this.isTitleScreen ? '▶ 开始游戏 (Play World)' : '▶ 继续游戏 (Resume)';
    this.menuOverlay.classList.remove('hidden');
  }

  public openSettings(): void {
    this.player.uiPaused = true;
    this.crosshairEl.style.display = 'none';
    this.menuOverlay.classList.add('hidden');
    this.inventoryModal.classList.add('hidden');
    (this.settingsModal as unknown as { _render: () => void })._render();
    this.settingsModal.classList.remove('hidden');
  }

  public openInventory(): void {
    if (document.pointerLockElement) document.exitPointerLock();
    this.player.uiPaused = true;
    this.crosshairEl.style.display = 'none';
    this.menuOverlay.classList.add('hidden');
    this.settingsModal.classList.add('hidden');
    this.populateInventoryGrid();
    this.inventoryModal.classList.remove('hidden');
  }

  public toggleInventory(): void {
    if (!this.inventoryModal.classList.contains('hidden')) {
      this.hideAllMenus();
      this.callbacks.onResumeGame();
    } else {
      this.openInventory();
    }
  }

  public hideAllMenus(): void {
    this.player.uiPaused = false;
    this.menuOverlay.classList.add('hidden');
    this.settingsModal.classList.add('hidden');
    this.inventoryModal.classList.add('hidden');
    this.crosshairEl.style.display = 'block';
  }

  public toggleDebug(): void {
    this.debugVisible = !this.debugVisible;
    this.f3El.classList.toggle('visible', this.debugVisible);
    if (this.debugVisible) {
      this.updateDebug(60);
    }
  }

  public updateDebug(fps: number): void {
    if (!this.debugVisible) return;
    const p = this.player.position;
    const bx = Math.floor(p.x);
    const by = Math.floor(p.y);
    const bz = Math.floor(p.z);
    const cx = World.chunkCoord(bx);
    const cz = World.chunkCoord(bz);
    const biome = this.world.getBiomeAt(bx, bz);
    const sky = this.world.getSkyLight(bx, by + 1, bz);
    const blk = this.world.getBlockLight(bx, by + 1, bz);

    const deg = (((-this.player.yaw * 180) / Math.PI) % 360 + 360) % 360;
    const facing =
      deg >= 315 || deg < 45 ? 'North (-Z)' :
      deg < 135 ? 'East (+X)' :
      deg < 225 ? 'South (+Z)' : 'West (-X)';

    const target = this.player.targetHit
      ? `${getBlockDef(this.player.targetHit.blockId).key} (${this.player.targetHit.x}, ${this.player.targetHit.y}, ${this.player.targetHit.z})`
      : 'none';

    const lines = [
      `Minecraft 1.21 Shaders Web (PBR + PCF Shadows + ACES HDR)`,
      `${Math.round(fps)} fps | Chunks Loaded: ${this.world.chunks.size} | Render Dist: ${this.settings.renderDistance}`,
      `XYZ: ${p.x.toFixed(2)} / ${p.y.toFixed(2)} / ${p.z.toFixed(2)}`,
      `Chunk: ${cx}, ${cz} | Seed: ${this.world.worldGen.seed}`,
      `Facing: ${facing} (${deg.toFixed(1)}°) | Mode: ${this.player.gameMode}`,
      `Biome: ${biome} | Light: ${Math.max(sky, blk)} (${sky} sky, ${blk} block)`,
      `Target Block: ${target}`,
      `Shader: ${this.settings.quality.toUpperCase()} | Time: ${(this.settings.timeOfDay * 24).toFixed(1)}h | Weather: ${this.settings.weather}`,
    ];
    this.f3El.innerHTML = lines.map((l) => `<div class="mc-f3-line">${l}</div>`).join('');
  }
}
