import { BlockSoundGroup } from '../types';

export class SoundEngine {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private noiseBuffer: AudioBuffer | null = null;
  private volume = 0.65;
  private ambientEnabled = true;
  private nextMusicTime = 4.0;
  private musicTimer = 0;

  public unlock(): void {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return;
      this.ctx = new AudioCtx();
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = this.volume;
      this.masterGain.connect(this.ctx.destination);
      this.noiseBuffer = this.createNoiseBuffer(this.ctx);
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }
  }

  public setVolume(v: number): void {
    this.volume = Math.max(0, Math.min(1, v));
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setTargetAtTime(this.volume, this.ctx.currentTime, 0.03);
    }
  }

  public setAmbientEnabled(enabled: boolean): void {
    this.ambientEnabled = enabled;
  }

  private createNoiseBuffer(ctx: AudioContext): AudioBuffer {
    const len = ctx.sampleRate * 1.0;
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < len; i++) {
      data[i] = Math.random() * 2 - 1;
    }
    return buf;
  }

  private playFilteredNoise(freq: number, q: number, duration: number, gainPeak: number, type: BiquadFilterType = 'bandpass') {
    if (!this.ctx || !this.masterGain || !this.noiseBuffer || this.volume <= 0.001) return;
    const now = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noiseBuffer;

    const filter = this.ctx.createBiquadFilter();
    filter.type = type;
    filter.frequency.setValueAtTime(freq * (0.88 + Math.random() * 0.24), now);
    filter.Q.value = q;

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(gainPeak, now + 0.008);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

    src.connect(filter);
    filter.connect(gain);
    gain.connect(this.masterGain);

    src.start(now, Math.random() * 0.5, duration + 0.02);
  }

  private playTone(freq: number, endFreq: number, duration: number, gainPeak: number, oscType: OscillatorType = 'triangle') {
    if (!this.ctx || !this.masterGain || this.volume <= 0.001) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    osc.type = oscType;
    osc.frequency.setValueAtTime(freq, now);
    osc.frequency.exponentialRampToValueAtTime(Math.max(20, endFreq), now + duration);

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(gainPeak, now + 0.008);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

    osc.connect(gain);
    gain.connect(this.masterGain);
    osc.start(now);
    osc.stop(now + duration + 0.02);
  }

  public playStep(group: BlockSoundGroup): void {
    switch (group) {
      case 'grass':
      case 'leaves':
        this.playFilteredNoise(1100, 1.4, 0.075, 0.14);
        break;
      case 'stone':
        this.playFilteredNoise(650, 2.2, 0.06, 0.15);
        this.playTone(170, 90, 0.045, 0.08, 'triangle');
        break;
      case 'wood':
        this.playTone(195, 110, 0.065, 0.14, 'triangle');
        this.playFilteredNoise(500, 2.0, 0.05, 0.08);
        break;
      case 'sand':
      case 'gravel':
      case 'snow':
        this.playFilteredNoise(2100, 0.9, 0.09, 0.13);
        break;
      case 'water':
        this.playFilteredNoise(820, 1.1, 0.14, 0.16);
        break;
      default:
        this.playFilteredNoise(900, 1.5, 0.065, 0.12);
    }
  }

  public playBreak(group: BlockSoundGroup): void {
    switch (group) {
      case 'glass':
        this.playFilteredNoise(4200, 3.0, 0.16, 0.25, 'highpass');
        this.playTone(2200, 950, 0.12, 0.14, 'sine');
        this.playTone(3100, 1600, 0.09, 0.1, 'triangle');
        break;
      case 'stone':
        this.playFilteredNoise(520, 1.8, 0.13, 0.28);
        this.playTone(145, 58, 0.11, 0.2, 'sawtooth');
        break;
      case 'wood':
        this.playTone(240, 95, 0.12, 0.22, 'triangle');
        this.playFilteredNoise(680, 1.5, 0.11, 0.2);
        break;
      case 'grass':
      case 'leaves':
        this.playFilteredNoise(1350, 1.0, 0.14, 0.24);
        this.playTone(210, 120, 0.08, 0.1, 'sine');
        break;
      default:
        this.playFilteredNoise(1600, 1.0, 0.13, 0.22);
        this.playTone(170, 95, 0.09, 0.12, 'triangle');
    }
  }

  public playPlace(group: BlockSoundGroup): void {
    this.playStep(group);
    this.playTone(260, 180, 0.055, 0.12, 'triangle');
  }

  public playSplash(): void {
    this.playFilteredNoise(680, 0.8, 0.28, 0.28, 'lowpass');
    this.playFilteredNoise(1600, 1.2, 0.22, 0.18, 'bandpass');
  }

  public playHurt(): void {
    this.playTone(180, 65, 0.16, 0.26, 'sawtooth');
    this.playFilteredNoise(400, 1.2, 0.12, 0.2);
  }

  public playUiClick(): void {
    this.playTone(680, 420, 0.035, 0.12, 'square');
  }

  // Gentle C418-inspired meditative ambient piano notes
  public updateAmbient(dt: number): void {
    if (!this.ambientEnabled || !this.ctx || !this.masterGain || this.ctx.state !== 'running') return;
    this.musicTimer += dt;
    if (this.musicTimer < this.nextMusicTime) return;
    this.musicTimer = 0;
    this.nextMusicTime = 3.2 + Math.random() * 4.5;

    // Warm C major / F lydian pentatonic pitches (Hz)
    const scale = [261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33, 659.25];
    const root = scale[Math.floor(Math.random() * scale.length)];
    const now = this.ctx.currentTime;

    const playNote = (freq: number, delay: number, peak: number, dur: number) => {
      if (!this.ctx || !this.masterGain) return;
      const osc = this.ctx.createOscillator();
      const osc2 = this.ctx.createOscillator();
      osc.type = 'sine';
      osc2.type = 'triangle';
      osc.frequency.value = freq;
      osc2.frequency.value = freq * 2;

      const filter = this.ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.value = 1100;

      const g = this.ctx.createGain();
      g.gain.setValueAtTime(0.0001, now + delay);
      g.gain.linearRampToValueAtTime(peak, now + delay + 0.08);
      g.gain.exponentialRampToValueAtTime(0.0001, now + delay + dur);

      const g2 = this.ctx.createGain();
      g2.gain.value = 0.18;

      osc.connect(filter);
      osc2.connect(g2);
      g2.connect(filter);
      filter.connect(g);
      g.connect(this.masterGain);

      osc.start(now + delay);
      osc2.start(now + delay);
      osc.stop(now + delay + dur + 0.05);
      osc2.stop(now + delay + dur + 0.05);
    };

    playNote(root, 0, 0.055, 3.6);
    if (Math.random() < 0.65) {
      const harmony = scale[Math.floor(Math.random() * scale.length)];
      playNote(harmony, 0.55 + Math.random() * 0.4, 0.038, 3.2);
    }
  }
}
