import * as THREE from 'three';
import { ProceduralAtlas } from '../textures/ProceduralAtlas';
import { ShaderSettings } from '../types';
import { WorldMaterials } from '../world/World';

// Shared GLSL sky scattering function used by SkyDome, Terrain fog, and Water reflections so everything matches seamlessly!
const GLSL_SKY_COMMON = /* glsl */ `
vec3 computeSkyColor(vec3 dir, vec3 sunDir, float dayFactor, float sunsetFactor, float rainFactor) {
  float y = clamp(dir.y, -0.25, 1.0);
  float horizon = pow(1.0 - max(y, 0.0), 3.6);

  // Day sky palette (vibrant BSL/Complementary atmospheric blue)
  vec3 dayZenith = vec3(0.14, 0.38, 0.82);
  vec3 dayHorizon = vec3(0.62, 0.82, 0.98);

  // Sunset / Sunrise golden-rose palette
  vec3 setZenith = vec3(0.16, 0.20, 0.44);
  vec3 setHorizon = vec3(0.98, 0.46, 0.22);

  // Night deep indigo palette
  vec3 nightZenith = vec3(0.012, 0.020, 0.055);
  vec3 nightHorizon = vec3(0.045, 0.072, 0.145);

  vec3 zenith = mix(nightZenith, dayZenith, dayFactor);
  vec3 horiz = mix(nightHorizon, dayHorizon, dayFactor);
  zenith = mix(zenith, setZenith, sunsetFactor * 0.75);
  horiz = mix(horiz, setHorizon, sunsetFactor * 0.92);

  // Rain overcast desaturation
  vec3 rainSky = mix(vec3(0.06, 0.08, 0.11), vec3(0.42, 0.48, 0.54), dayFactor);
  zenith = mix(zenith, rainSky * 0.78, rainFactor * 0.72);
  horiz = mix(horiz, rainSky, rainFactor * 0.78);

  vec3 col = mix(zenith, horiz, horizon);

  // Mie forward scattering sun halo
  float sunCos = max(dot(normalize(dir), normalize(sunDir)), 0.0);
  float mieGlow = pow(sunCos, 6.0) * 0.35 + pow(sunCos, 42.0) * 0.65;
  vec3 sunTint = mix(vec3(1.0, 0.52, 0.20), vec3(1.0, 0.94, 0.82), clamp((sunDir.y + 0.05) * 2.4, 0.0, 1.0));
  col += sunTint * mieGlow * (1.0 - rainFactor * 0.75) * smoothstep(-0.15, 0.15, sunDir.y);

  // Subtle below-horizon ground reflection gradient
  if (y < 0.0) {
    col = mix(horiz, horiz * 0.45, clamp(-y * 4.0, 0.0, 1.0));
  }
  return col;
}
`;

// Shadow sampling helper with 12-tap rotated Poisson PCF soft shadows
const GLSL_SHADOW_COMMON = /* glsl */ `
uniform sampler2D uShadowMap;
uniform mat4 uShadowMatrix;
uniform float uShadowsEnabled;
uniform float uShadowTexelSize;

const vec2 POISSON_DISK[12] = vec2[12](
  vec2(-0.326212, -0.405810), vec2(-0.840144, -0.073580),
  vec2(-0.695914,  0.457137), vec2(-0.203345,  0.620716),
  vec2( 0.962340, -0.194983), vec2( 0.473434, -0.480026),
  vec2( 0.519456,  0.767022), vec2( 0.185461, -0.893124),
  vec2( 0.507431,  0.064425), vec2( 0.896420,  0.412458),
  vec2(-0.321940, -0.932615), vec2(-0.791559, -0.597710)
);

float sampleSoftShadow(vec3 worldPos, vec3 geoNormal, vec3 lightDir) {
  if (uShadowsEnabled < 0.5) return 1.0;

  float ndotl = max(dot(geoNormal, lightDir), 0.0);
  // Push along geometric normal to eliminate shadow acne on angled/grazing block faces
  vec3 biasedPos = worldPos + geoNormal * (0.045 + 0.08 * (1.0 - ndotl));
  vec4 proj = uShadowMatrix * vec4(biasedPos, 1.0);
  vec3 coords = proj.xyz / proj.w;
  coords = coords * 0.5 + 0.5;

  if (coords.x <= 0.01 || coords.x >= 0.99 || coords.y <= 0.01 || coords.y >= 0.99 || coords.z > 0.999 || coords.z < 0.0) {
    return 1.0;
  }

  // Smooth fade near shadow frustum edges
  float edge = max(abs(coords.x - 0.5), abs(coords.y - 0.5));
  float fade = 1.0 - smoothstep(0.42, 0.5, edge);

  // Per-pixel rotation breaks up PCF banding into cinematic film-like soft shadows
  float angle = fract(sin(dot(floor(worldPos.xz * 16.0), vec2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
  float s = sin(angle), c = cos(angle);
  mat2 rot = mat2(c, -s, s, c);

  float radius = uShadowTexelSize * 2.2;
  float bias = 0.00045 + 0.0008 * (1.0 - ndotl);
  float lit = 0.0;
  for (int i = 0; i < 12; i++) {
    vec2 offset = (rot * POISSON_DISK[i]) * radius;
    float d = texture2D(uShadowMap, coords.xy + offset).r;
    lit += (coords.z - bias <= d) ? 1.0 : 0.0;
  }
  lit /= 12.0;
  return mix(1.0, lit, fade);
}
`;

export class ShaderPipeline {
  public readonly materials: WorldMaterials;
  public readonly skyMesh: THREE.Mesh;
  public readonly cloudMesh: THREE.Mesh;
  public readonly rainSystem: THREE.Points;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly atlas: ProceduralAtlas;

  // Dedicated high-res DepthTexture shadow pass
  private shadowTarget: THREE.WebGLRenderTarget;
  private readonly shadowCamera: THREE.OrthographicCamera;
  private readonly shadowMatrix = new THREE.Matrix4();
  private readonly opaqueDepthMat: THREE.ShaderMaterial;
  private readonly foliageDepthMat: THREE.ShaderMaterial;

  // HDR Scene + Post-process (Bloom, God Rays, ACES Filmic, Underwater)
  private hdrTarget: THREE.WebGLRenderTarget;
  private readonly postScene: THREE.Scene;
  private readonly postCamera: THREE.OrthographicCamera;
  private readonly postMaterial: THREE.ShaderMaterial;

  // Shared shader uniforms
  public readonly sharedUniforms: Record<string, THREE.IUniform>;
  private readonly sunDir = new THREE.Vector3(0.6, 0.65, 0.45).normalize();
  private readonly moonDir = new THREE.Vector3(-0.6, -0.65, -0.45).normalize();
  private readonly lightDir = new THREE.Vector3(0.6, 0.65, 0.45).normalize();
  private readonly sunColor = new THREE.Color(1, 0.95, 0.85);
  private readonly ambientColor = new THREE.Color(0.35, 0.48, 0.68);
  private readonly sunScreenPos = new THREE.Vector2(0.5, 0.5);
  private rainFactor = 0;

  constructor(renderer: THREE.WebGLRenderer, atlas: ProceduralAtlas, settings: ShaderSettings) {
    this.renderer = renderer;
    this.atlas = atlas;

    const shadowRes = settings.shadowResolution;
    this.shadowTarget = this.createShadowTarget(shadowRes);
    this.shadowCamera = new THREE.OrthographicCamera(-64, 64, 64, -64, 1, 260);

    const size = new THREE.Vector2();
    renderer.getDrawingBufferSize(size);
    this.hdrTarget = this.createHdrTarget(Math.max(1, size.x), Math.max(1, size.y));

    this.sharedUniforms = {
      uAlbedoMap: { value: atlas.albedoTexture },
      uNormalMap: { value: atlas.normalTexture },
      uPbrMap: { value: atlas.pbrTexture },
      uShadowMap: { value: this.shadowTarget.depthTexture },
      uShadowMatrix: { value: this.shadowMatrix },
      uShadowTexelSize: { value: 1.0 / shadowRes },
      uShadowsEnabled: { value: settings.shadowsEnabled ? 1.0 : 0.0 },
      uPbrEnabled: { value: settings.pbrNormalMapping ? 1.0 : 0.0 },
      uWindEnabled: { value: settings.windAnimation ? 1.0 : 0.0 },
      uRealisticWater: { value: settings.realisticWater ? 1.0 : 0.0 },
      uVanillaMode: { value: settings.quality === 'vanilla' ? 1.0 : 0.0 },
      uTime: { value: 0.0 },
      uSunDir: { value: this.sunDir },
      uLightDir: { value: this.lightDir },
      uSunColor: { value: this.sunColor },
      uAmbientColor: { value: this.ambientColor },
      uDayFactor: { value: 1.0 },
      uSunsetFactor: { value: 0.0 },
      uRainFactor: { value: 0.0 },
      uFogStart: { value: settings.renderDistance * 11.5 },
      uFogEnd: { value: settings.renderDistance * 15.8 },
      uCameraPos: { value: new THREE.Vector3() },
      uUnderwater: { value: 0.0 },
    };

    // Depth-only shadow materials (with wind sway & leaf cutout!)
    this.opaqueDepthMat = this.createDepthMaterial(false);
    this.foliageDepthMat = this.createDepthMaterial(true);

    // Main world materials
    this.materials = {
      opaque: this.createTerrainMaterial('opaque'),
      foliage: this.createTerrainMaterial('foliage'),
      translucent: this.createTerrainMaterial('translucent'),
      water: this.createWaterMaterial(),
    };

    // Sky dome, 3D blocky Minecraft clouds, and rain particle streaks
    this.skyMesh = this.createSkyDome();
    this.cloudMesh = this.createCloudLayer();
    this.rainSystem = this.createRainSystem();

    // Full-screen post-processing quad
    this.postScene = new THREE.Scene();
    this.postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.postMaterial = this.createPostMaterial(Math.max(1, size.x), Math.max(1, size.y), settings);
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.postMaterial);
    quad.frustumCulled = false;
    this.postScene.add(quad);
  }

  private createShadowTarget(res: number): THREE.WebGLRenderTarget {
    const depthTex = new THREE.DepthTexture(res, res);
    depthTex.type = THREE.UnsignedIntType;
    depthTex.minFilter = THREE.NearestFilter;
    depthTex.magFilter = THREE.NearestFilter;
    depthTex.wrapS = THREE.ClampToEdgeWrapping;
    depthTex.wrapT = THREE.ClampToEdgeWrapping;

    return new THREE.WebGLRenderTarget(res, res, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true,
      stencilBuffer: false,
      depthTexture: depthTex,
    });
  }

  private createHdrTarget(w: number, h: number): THREE.WebGLRenderTarget {
    const isWebGL2 = this.renderer.capabilities.isWebGL2;
    return new THREE.WebGLRenderTarget(w, h, {
      type: isWebGL2 ? THREE.HalfFloatType : THREE.UnsignedByteType,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: true,
      stencilBuffer: false,
    });
  }

  public resize(width: number, height: number): void {
    const dpr = this.renderer.getPixelRatio();
    const w = Math.max(1, Math.floor(width * dpr));
    const h = Math.max(1, Math.floor(height * dpr));
    this.hdrTarget.setSize(w, h);
    this.postMaterial.uniforms.uResolution.value.set(w, h);
    this.postMaterial.uniforms.uSceneTex.value = this.hdrTarget.texture;
  }

  public applySettings(settings: ShaderSettings): void {
    const u = this.sharedUniforms;
    u.uShadowsEnabled.value = settings.shadowsEnabled && settings.quality !== 'vanilla' ? 1.0 : 0.0;
    u.uPbrEnabled.value = settings.pbrNormalMapping && settings.quality !== 'vanilla' ? 1.0 : 0.0;
    u.uWindEnabled.value = settings.windAnimation && settings.quality !== 'vanilla' ? 1.0 : 0.0;
    u.uRealisticWater.value = settings.realisticWater && settings.quality !== 'vanilla' ? 1.0 : 0.0;
    u.uVanillaMode.value = settings.quality === 'vanilla' ? 1.0 : 0.0;
    u.uFogStart.value = settings.renderDistance * 11.0;
    u.uFogEnd.value = settings.renderDistance * 15.6;

    this.cloudMesh.visible = settings.volumetricClouds;
    this.postMaterial.uniforms.uBloomEnabled.value =
      settings.bloomAndGodRays && settings.quality !== 'vanilla' ? 1.0 : 0.0;

    if (this.shadowTarget.width !== settings.shadowResolution) {
      this.shadowTarget.dispose();
      this.shadowTarget = this.createShadowTarget(settings.shadowResolution);
      u.uShadowMap.value = this.shadowTarget.depthTexture;
      u.uShadowTexelSize.value = 1.0 / settings.shadowResolution;
    }
  }

  private createDepthMaterial(alphaCutout: boolean): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      uniforms: {
        uAlbedoMap: this.sharedUniforms.uAlbedoMap,
        uTime: this.sharedUniforms.uTime,
        uWindEnabled: this.sharedUniforms.uWindEnabled,
      },
      vertexShader: /* glsl */ `
        attribute vec4 aLight;
        uniform float uTime;
        uniform float uWindEnabled;
        varying vec2 vUv;
        void main() {
          vUv = uv;
          vec3 pos = position;
          if (uWindEnabled > 0.5 && aLight.w > 0.01) {
            float sway = sin(uTime * 2.1 + pos.x * 0.7 + pos.z * 0.55) * 0.075 * aLight.w;
            float swayZ = cos(uTime * 1.7 + pos.z * 0.8 + pos.y * 0.4) * 0.055 * aLight.w;
            pos.x += sway;
            pos.z += swayZ;
          }
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: alphaCutout
        ? /* glsl */ `
          uniform sampler2D uAlbedoMap;
          varying vec2 vUv;
          void main() {
            if (texture2D(uAlbedoMap, vUv).a < 0.45) discard;
            gl_FragColor = vec4(1.0);
          }
        `
        : /* glsl */ `
          void main() {
            gl_FragColor = vec4(1.0);
          }
        `,
      side: alphaCutout ? THREE.DoubleSide : THREE.FrontSide,
    });
  }

  private createTerrainMaterial(mode: 'opaque' | 'foliage' | 'translucent'): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      uniforms: {
        ...this.sharedUniforms,
      },
      vertexShader: /* glsl */ `
        attribute vec3 aTangent;
        attribute vec3 color;
        attribute vec4 aLight; // sky, block, ao, wind

        uniform float uTime;
        uniform float uWindEnabled;

        varying vec2 vUv;
        varying vec3 vColor;
        varying vec3 vNormal;
        varying vec3 vTangent;
        varying vec3 vWorldPos;
        varying vec4 vLight;

        void main() {
          vUv = uv;
          vColor = color;
          vNormal = normalize(normal);
          vTangent = normalize(aTangent);
          vLight = aLight;

          vec3 pos = position;
          if (uWindEnabled > 0.5 && aLight.w > 0.01) {
            float gust = sin(uTime * 1.9 + pos.x * 0.52 + pos.z * 0.44) + 0.45 * sin(uTime * 3.7 + pos.x * 1.3 - pos.z * 0.9);
            float gustZ = cos(uTime * 1.55 + pos.z * 0.62 + pos.y * 0.35);
            pos.x += gust * 0.055 * aLight.w;
            pos.z += gustZ * 0.045 * aLight.w;
          }

          vec4 wp = modelMatrix * vec4(pos, 1.0);
          vWorldPos = wp.xyz;
          gl_Position = projectionMatrix * viewMatrix * wp;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D uAlbedoMap;
        uniform sampler2D uNormalMap;
        uniform sampler2D uPbrMap;
        uniform float uPbrEnabled;
        uniform float uVanillaMode;
        uniform float uTime;
        uniform vec3 uSunDir;
        uniform vec3 uLightDir;
        uniform vec3 uSunColor;
        uniform vec3 uAmbientColor;
        uniform float uDayFactor;
        uniform float uSunsetFactor;
        uniform float uRainFactor;
        uniform float uFogStart;
        uniform float uFogEnd;
        uniform vec3 uCameraPos;
        uniform float uUnderwater;

        varying vec2 vUv;
        varying vec3 vColor;
        varying vec3 vNormal;
        varying vec3 vTangent;
        varying vec3 vWorldPos;
        varying vec4 vLight;

        ${GLSL_SKY_COMMON}
        ${GLSL_SHADOW_COMMON}

        void main() {
          vec4 tex = texture2D(uAlbedoMap, vUv);
          ${mode === 'foliage' ? 'if (tex.a < 0.42) discard;' : ''}

          // Linearize sRGB albedo for luminous Complementary/BSL-style PBR lighting
          vec3 albedo = pow(tex.rgb, vec3(1.92)) * pow(vColor, vec3(1.65));
          vec4 pbr = texture2D(uPbrMap, vUv);
          float roughness = pbr.r;
          float metalness = pbr.g;
          float emissive = pbr.b;
          float sss = pbr.a;

          vec3 geoN = normalize(vNormal);
          vec3 N = geoN;

          if (uPbrEnabled > 0.5) {
            vec3 T = normalize(vTangent);
            vec3 B = normalize(cross(geoN, T));
            vec3 mapN = texture2D(uNormalMap, vUv).xyz * 2.0 - 1.0;
            mapN.xy *= 0.85;
            N = normalize(T * mapN.x + B * mapN.y + geoN * mapN.z);
          }

          vec3 V = normalize(uCameraPos - vWorldPos);
          vec3 L = normalize(uLightDir);

          float skyL = clamp(vLight.x, 0.0, 1.0);
          float blkL = clamp(vLight.y, 0.0, 1.0);
          float ao = clamp(vLight.z, 0.25, 1.0);

          // Rain wetness makes exposed upward surfaces glossy & slightly darker!
          float wetMask = uRainFactor * smoothstep(0.7, 0.98, skyL) * smoothstep(0.2, 0.85, geoN.y);
          if (wetMask > 0.01 && uVanillaMode < 0.5) {
            albedo *= mix(1.0, 0.72, wetMask);
            roughness = mix(roughness, 0.14, wetMask);
            // Subtle animated rain ripple rings on wet blocks
            float ripple = sin(length(fract(vWorldPos.xz * 0.45) - 0.5) * 32.0 - uTime * 9.0) * 0.03 * wetMask;
            N = normalize(N + vec3(ripple, 0.0, ripple));
          }

          // Directional sunlight / moonlight with soft shadows
          float shadow = sampleSoftShadow(vWorldPos, geoN, L);
          // Only open-sky regions receive direct sun shafts
          float skySunGate = smoothstep(0.35, 0.85, skyL);
          float ndotl = max(dot(N, L), 0.0);

          // Subsurface translucency on leaves & grass when sunlight shines through them
          float backLight = pow(max(dot(-V, L), 0.0), 3.0) * sss * (0.45 + 0.55 * shadow);
          float diffuse = ndotl * shadow * skySunGate;
          if (uVanillaMode > 0.5) {
            // Classic Minecraft directional face shading
            float faceShade = geoN.y > 0.5 ? 1.0 : geoN.y < -0.5 ? 0.55 : abs(geoN.x) > 0.5 ? 0.78 : 0.86;
            vec3 vanillaLight = (uAmbientColor * 0.85 + uSunColor * 0.65) * max(skyL, 0.12) * faceShade;
            vec3 torchLight = vec3(1.0, 0.66, 0.30) * pow(blkL, 1.6) * 1.3;
            vec3 col = albedo * (vanillaLight * ao + torchLight) + albedo * emissive * 1.5;
            gl_FragColor = vec4(col, ${mode === 'translucent' ? 'tex.a' : '1.0'});
            return;
          }

          vec3 directLight = uSunColor * (diffuse * 1.65 + backLight * 0.75 * skySunGate) * (1.0 - uRainFactor * 0.65);

          // Hemisphere sky ambient (sky blue from above, warm earth bounce from below)
          float hemi = N.y * 0.5 + 0.5;
          vec3 skyFill = mix(uAmbientColor * 0.42, uAmbientColor * 1.05, hemi);
          vec3 bounceFill = uSunColor * 0.16 * max(0.0, 1.0 - abs(geoN.y)) * skySunGate;
          vec3 ambient = (skyFill + bounceFill) * (0.06 + 0.94 * pow(skyL, 1.35)) * ao;

          // Warm flickering torch / glowstone / lava voxel block light
          float flicker = 0.95 + 0.05 * sin(uTime * 7.5 + vWorldPos.x * 1.7 + vWorldPos.z * 2.3);
          float blkCurve = pow(blkL, 1.85) * 1.95 * flicker;
          vec3 blockLightCol = vec3(1.0, 0.54, 0.19) * blkCurve * (0.65 + 0.35 * ao);

          // Specular PBR highlights (ores, ice, glass, wet rain surfaces)
          vec3 H = normalize(L + V);
          float ndoth = max(dot(N, H), 0.0);
          float specPower = mix(128.0, 8.0, roughness * roughness);
          float fresnel = pow(1.0 - max(dot(N, V), 0.0), 4.0);
          float specStrength = mix(0.04, 0.95, max(metalness, wetMask * 0.75));
          vec3 specCol = mix(vec3(1.0), albedo, metalness);
          vec3 specular = uSunColor * specCol * pow(ndoth, specPower) * specStrength * (1.0 - roughness * 0.75) * shadow * skySunGate * 1.5;

          // Sky Fresnel reflection on wet blocks, glass & ice
          vec3 reflDir = reflect(-V, N);
          vec3 envRefl = computeSkyColor(reflDir, uSunDir, uDayFactor, uSunsetFactor, uRainFactor);
          vec3 envSpec = envRefl * fresnel * specStrength * (1.0 - roughness) * skyL * 0.65;

          // Underwater caustics shimmer on submerged blocks
          vec3 caustics = vec3(0.0);
          if (vWorldPos.y < 54.0 && skyL > 0.25) {
            float c1 = sin(vWorldPos.x * 2.4 + uTime * 2.0 + sin(vWorldPos.z * 2.1 - uTime * 1.4));
            float c2 = cos(vWorldPos.z * 2.7 - uTime * 1.8 + cos(vWorldPos.x * 1.9 + uTime * 1.6));
            float caust = pow(clamp((c1 + c2) * 0.5 + 0.5, 0.0, 1.0), 4.0) * 0.45 * uDayFactor * skyL;
            caustics = vec3(0.35, 0.85, 1.0) * caust * max(geoN.y, 0.2);
          }

          vec3 color = albedo * (directLight + ambient + blockLightCol + caustics) + specular + envSpec;

          // HDR Emissive glow (triggers Bloom!)
          color += albedo * emissive * (2.2 + 0.25 * sin(uTime * 5.0 + vWorldPos.x));

          // Atmospheric distance fog matching live sky scattering
          float dist = length(uCameraPos - vWorldPos);
          vec3 viewDir = normalize(vWorldPos - uCameraPos);
          vec3 fogCol = computeSkyColor(vec3(viewDir.x, max(viewDir.y, 0.02), viewDir.z), uSunDir, uDayFactor, uSunsetFactor, uRainFactor);
          float fogFactor = smoothstep(uFogStart, uFogEnd, dist);
          // Subtle low-lying valley mist at golden hour
          float valleyMist = exp(-max(vWorldPos.y - 52.0, 0.0) * 0.065) * (1.0 - exp(-dist * 0.009)) * (0.14 + uSunsetFactor * 0.22 + uRainFactor * 0.35);
          color = mix(color, fogCol, clamp(fogFactor + valleyMist * 0.45, 0.0, 0.96));

          // Underwater depth fog
          if (uUnderwater > 0.5) {
            float uwFog = 1.0 - exp(-dist * 0.065);
            vec3 uwCol = mix(vec3(0.02, 0.12, 0.20), vec3(0.06, 0.32, 0.46), uDayFactor);
            color = mix(color, uwCol, clamp(uwFog, 0.0, 0.95));
          }

          float alpha = ${mode === 'translucent' ? 'clamp(tex.a + fresnel * 0.45, 0.18, 0.96)' : '1.0'};
          gl_FragColor = vec4(color, alpha);
        }
      `,
      transparent: mode === 'translucent',
      depthWrite: mode !== 'translucent',
      side: mode === 'foliage' ? THREE.DoubleSide : THREE.FrontSide,
    });
  }

  private createWaterMaterial(): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      uniforms: {
        ...this.sharedUniforms,
      },
      vertexShader: /* glsl */ `
        attribute vec4 aLight; // sky, block, ao, depthNorm
        uniform float uTime;
        uniform float uRealisticWater;
        varying vec3 vWorldPos;
        varying vec3 vNormal;
        varying vec4 vLight;
        varying vec2 vUv;

        void main() {
          vUv = uv;
          vNormal = normalize(normal);
          vLight = aLight;
          vec3 pos = position;
          if (uRealisticWater > 0.5 && vNormal.y > 0.5) {
            float w1 = sin(pos.x * 1.15 + uTime * 1.8 + pos.z * 0.75) * 0.032;
            float w2 = cos(pos.z * 1.45 - uTime * 1.45 + pos.x * 0.6) * 0.024;
            pos.y += w1 + w2;
          }
          vec4 wp = modelMatrix * vec4(pos, 1.0);
          vWorldPos = wp.xyz;
          gl_Position = projectionMatrix * viewMatrix * wp;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        uniform float uRealisticWater;
        uniform vec3 uSunDir;
        uniform vec3 uLightDir;
        uniform vec3 uSunColor;
        uniform float uDayFactor;
        uniform float uSunsetFactor;
        uniform float uRainFactor;
        uniform float uFogStart;
        uniform float uFogEnd;
        uniform vec3 uCameraPos;
        uniform float uUnderwater;

        varying vec3 vWorldPos;
        varying vec3 vNormal;
        varying vec4 vLight;
        varying vec2 vUv;

        ${GLSL_SKY_COMMON}
        ${GLSL_SHADOW_COMMON}

        // Multi-octave procedural water wave height & normal
        float waveHeight(vec2 p, float t) {
          float h = 0.0;
          h += sin(p.x * 1.9 + t * 2.1 + p.y * 0.9) * 0.35;
          h += sin(p.y * 2.4 - t * 1.7 + p.x * 1.2) * 0.28;
          h += sin((p.x - p.y) * 4.2 + t * 3.1) * 0.14;
          h += cos((p.x * 1.7 + p.y * 3.1) * 2.3 - t * 2.6) * 0.10;
          h += sin(p.x * 9.5 + p.y * 8.2 + t * 4.4) * 0.04;
          return h;
        }

        void main() {
          vec3 V = normalize(uCameraPos - vWorldPos);
          vec3 L = normalize(uLightDir);
          float depth = clamp(vLight.w, 0.05, 1.0);

          if (uRealisticWater < 0.5) {
            vec3 simpleCol = mix(vec3(0.08, 0.36, 0.72), vec3(0.03, 0.14, 0.32), depth) * (0.25 + 0.75 * uDayFactor);
            gl_FragColor = vec4(simpleCol, 0.76);
            return;
          }

          // Compute crisp analytical normal from wave gradient
          vec2 p = vWorldPos.xz;
          float eps = 0.08;
          float h0 = waveHeight(p, uTime);
          float hx = waveHeight(p + vec2(eps, 0.0), uTime);
          float hz = waveHeight(p + vec2(0.0, eps), uTime);
          vec3 waveN = normalize(vec3((h0 - hx) * 1.45, 1.0, (h0 - hz) * 1.45));

          // Extra raindrop ripples when raining
          if (uRainFactor > 0.05) {
            float r = sin(length(fract(p * 0.8) - 0.5) * 38.0 - uTime * 11.0) * 0.08 * uRainFactor;
            waveN = normalize(waveN + vec3(r, 0.0, r));
          }

          vec3 N = vNormal.y > 0.5 ? waveN : normalize(vNormal + waveN * 0.25);

          // Physical Schlick Fresnel reflection of sky & sun!
          float cosTheta = clamp(dot(N, V), 0.0, 1.0);
          float fresnel = 0.04 + 0.96 * pow(1.0 - cosTheta, 4.4);

          vec3 reflDir = reflect(-V, N);
          reflDir.y = max(reflDir.y, 0.03);
          vec3 skyRefl = computeSkyColor(reflDir, uSunDir, uDayFactor, uSunsetFactor, uRainFactor);

          // Subtle reflected cloud streaks on water
          float cloudRefl = smoothstep(0.45, 0.85, sin(reflDir.x * 8.0 + uTime * 0.12) * cos(reflDir.z * 7.0 - uTime * 0.08) * 0.5 + 0.5);
          skyRefl = mix(skyRefl, skyRefl * 1.25 + vec3(0.16) * uDayFactor, cloudRefl * 0.22 * (1.0 - uRainFactor * 0.6));

          // Shallow turquoise-emerald to deep ocean sapphire water absorption
          vec3 shallowCol = vec3(0.06, 0.44, 0.48) * (0.22 + 0.95 * uDayFactor);
          vec3 deepCol = vec3(0.012, 0.09, 0.21) * (0.2 + 0.95 * uDayFactor);
          vec3 waterBody = mix(shallowCol, deepCol, pow(depth, 0.72));

          // Subsurface aqua glow in shallow sunlit waves
          float shadow = sampleSoftShadow(vWorldPos, vec3(0.0, 1.0, 0.0), L);
          float waveScatter = pow(max(dot(-V, L), 0.0), 3.0) * (1.0 - depth * 0.6) * shadow;
          waterBody += vec3(0.05, 0.48, 0.44) * waveScatter * uDayFactor * 0.75;

          // Subtle shimmering caustics seen through shallow water
          float caust = pow(clamp(h0 * 0.55 + 0.5, 0.0, 1.0), 3.0) * (1.0 - depth) * uDayFactor * shadow;
          waterBody += vec3(0.24, 0.68, 0.72) * caust * 0.35;

          // Combine refraction body & sky reflection via Fresnel
          vec3 col = mix(waterBody, skyRefl * (0.65 + 0.35 * shadow), clamp(fresnel * 1.12, 0.06, 0.95));

          // Brilliant dancing sun / moon specular glitter path on waves!
          vec3 H = normalize(L + V);
          float specTight = pow(max(dot(N, H), 0.0), 320.0) * 4.2;
          float specBroad = pow(max(dot(N, H), 0.0), 42.0) * 0.48;
          col += uSunColor * (specTight + specBroad) * shadow * (1.0 - uRainFactor * 0.7);

          // Warm torchlight reflection near shoreline lanterns
          col += vec3(1.0, 0.56, 0.20) * pow(vLight.y, 1.6) * 0.7;

          // Atmospheric horizon fog
          float dist = length(uCameraPos - vWorldPos);
          vec3 viewDir = normalize(vWorldPos - uCameraPos);
          vec3 fogCol = computeSkyColor(vec3(viewDir.x, max(viewDir.y, 0.02), viewDir.z), uSunDir, uDayFactor, uSunsetFactor, uRainFactor);
          float fogFactor = smoothstep(uFogStart, uFogEnd, dist);
          col = mix(col, fogCol, fogFactor);

          float alpha = clamp(mix(0.62, 0.93, depth) + fresnel * 0.32, 0.58, 0.97);
          if (uUnderwater > 0.5) alpha = 0.55;
          gl_FragColor = vec4(col, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
  }

  private createSkyDome(): THREE.Mesh {
    const geom = new THREE.SphereGeometry(420, 48, 32);
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uSunDir: this.sharedUniforms.uSunDir,
        uDayFactor: this.sharedUniforms.uDayFactor,
        uSunsetFactor: this.sharedUniforms.uSunsetFactor,
        uRainFactor: this.sharedUniforms.uRainFactor,
        uTime: this.sharedUniforms.uTime,
      },
      vertexShader: /* glsl */ `
        varying vec3 vDir;
        void main() {
          vDir = normalize(position);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform vec3 uSunDir;
        uniform float uDayFactor;
        uniform float uSunsetFactor;
        uniform float uRainFactor;
        uniform float uTime;
        varying vec3 vDir;

        ${GLSL_SKY_COMMON}

        float hash31(vec3 p) {
          p = fract(p * vec3(443.897, 441.423, 437.195));
          p += dot(p, p.yzx + 19.19);
          return fract((p.x + p.y) * p.z);
        }

        void main() {
          vec3 dir = normalize(vDir);
          vec3 sunDir = normalize(uSunDir);
          vec3 moonDir = -sunDir;

          vec3 col = computeSkyColor(dir, sunDir, uDayFactor, uSunsetFactor, uRainFactor);

          // Crisp glowing solar disc + golden corona
          float sunDot = dot(dir, sunDir);
          float sunDisc = smoothstep(0.9985, 0.9993, sunDot);
          float sunCorona = pow(max(sunDot, 0.0), 160.0) * 1.4;
          vec3 sunDiscCol = mix(vec3(1.0, 0.65, 0.25), vec3(1.0, 0.98, 0.90), clamp((sunDir.y + 0.02) * 3.0, 0.0, 1.0));
          float sunVis = smoothstep(-0.08, 0.05, sunDir.y) * (1.0 - uRainFactor * 0.85);
          col += sunDiscCol * (sunDisc * 4.5 + sunCorona) * sunVis;

          // Silver-blue Moon with gentle halo at night
          float moonDot = dot(dir, moonDir);
          float moonDisc = smoothstep(0.9986, 0.9992, moonDot);
          float moonHalo = pow(max(moonDot, 0.0), 64.0) * 0.35;
          float moonVis = smoothstep(-0.08, 0.06, moonDir.y) * (1.0 - uRainFactor * 0.8);
          col += vec3(0.78, 0.86, 1.0) * (moonDisc * 2.2 + moonHalo) * moonVis;

          // Procedural twinkling starfield at night!
          float nightMask = (1.0 - uDayFactor) * smoothstep(0.02, 0.25, dir.y) * (1.0 - uRainFactor);
          if (nightMask > 0.01) {
            vec3 starCell = floor(dir * 260.0);
            float rnd = hash31(starCell);
            if (rnd > 0.9945) {
              float twinkle = 0.65 + 0.35 * sin(uTime * (2.5 + rnd * 6.0) + rnd * 60.0);
              float bright = smoothstep(0.9945, 1.0, rnd) * twinkle * nightMask;
              vec3 starTint = mix(vec3(0.75, 0.85, 1.0), vec3(1.0, 0.92, 0.78), fract(rnd * 91.3));
              col += starTint * bright * 1.9;
            }
          }

          gl_FragColor = vec4(col, 1.0);
        }
      `,
      side: THREE.BackSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.renderOrder = -10;
    mesh.frustumCulled = false;
    return mesh;
  }

  // Iconic 3D extruded blocky Minecraft clouds with golden shader lighting!
  private createCloudLayer(): THREE.Mesh {
    const grid = 28;
    const cellSize = 18;
    const thickness = 4.2;
    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    let vCount = 0;

    const occupied = new Uint8Array(grid * grid);
    for (let z = 0; z < grid; z++) {
      for (let x = 0; x < grid; x++) {
        // Layered deterministic cloud clusters
        const n =
          Math.sin(x * 0.55 + 1.2) * Math.cos(z * 0.52 - 0.7) +
          0.55 * Math.sin(x * 1.15 - z * 0.85) +
          0.35 * Math.cos(x * 0.28 + z * 0.31);
        if (n > 0.35) occupied[z * grid + x] = 1;
      }
    }

    const pushFace = (
      p0: [number, number, number], p1: [number, number, number],
      p2: [number, number, number], p3: [number, number, number],
      n: [number, number, number]
    ) => {
      const b = vCount;
      for (const p of [p0, p1, p2, p3]) {
        positions.push(p[0], p[1], p[2]);
        normals.push(n[0], n[1], n[2]);
      }
      indices.push(b, b + 1, b + 2, b, b + 2, b + 3);
      vCount += 4;
    };

    const half = (grid * cellSize) / 2;
    for (let z = 0; z < grid; z++) {
      for (let x = 0; x < grid; x++) {
        if (!occupied[z * grid + x]) continue;
        const x0 = x * cellSize - half;
        const x1 = x0 + cellSize;
        const z0 = z * cellSize - half;
        const z1 = z0 + cellSize;
        const y0 = 0;
        const y1 = thickness;

        // Top & bottom
        pushFace([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], [0, 1, 0]);
        pushFace([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [0, -1, 0]);
        // Exposed sides
        if (z + 1 >= grid || !occupied[(z + 1) * grid + x])
          pushFace([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1]);
        if (z - 1 < 0 || !occupied[(z - 1) * grid + x])
          pushFace([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [0, 0, -1]);
        if (x + 1 >= grid || !occupied[z * grid + (x + 1)])
          pushFace([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [1, 0, 0]);
        if (x - 1 < 0 || !occupied[z * grid + (x - 1)])
          pushFace([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [-1, 0, 0]);
      }
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geom.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
    geom.setIndex(indices);

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uSunDir: this.sharedUniforms.uSunDir,
        uSunColor: this.sharedUniforms.uSunColor,
        uAmbientColor: this.sharedUniforms.uAmbientColor,
        uDayFactor: this.sharedUniforms.uDayFactor,
        uSunsetFactor: this.sharedUniforms.uSunsetFactor,
        uRainFactor: this.sharedUniforms.uRainFactor,
        uCameraPos: this.sharedUniforms.uCameraPos,
      },
      vertexShader: /* glsl */ `
        varying vec3 vWorldPos;
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normal);
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vWorldPos = wp.xyz;
          gl_Position = projectionMatrix * viewMatrix * wp;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform vec3 uSunDir;
        uniform vec3 uSunColor;
        uniform vec3 uAmbientColor;
        uniform float uDayFactor;
        uniform float uSunsetFactor;
        uniform float uRainFactor;
        uniform vec3 uCameraPos;
        varying vec3 vWorldPos;
        varying vec3 vNormal;

        void main() {
          vec3 N = normalize(vNormal);
          vec3 sunDir = normalize(uSunDir);
          vec3 viewDir = normalize(vWorldPos - uCameraPos);

          // Soft sunlit tops, golden sunset rims, cool atmospheric shadow undersides
          float ndotl = dot(N, sunDir) * 0.5 + 0.5;
          float topShade = N.y > 0.5 ? 1.0 : N.y < -0.5 ? 0.68 : 0.85;
          float silverLining = pow(max(dot(viewDir, sunDir), 0.0), 8.0) * 0.45;

          vec3 brightCol = mix(vec3(0.18, 0.24, 0.36), vec3(1.0, 0.99, 0.96), uDayFactor);
          brightCol = mix(brightCol, vec3(1.0, 0.68, 0.42), uSunsetFactor * 0.85);
          brightCol = mix(brightCol, vec3(0.38, 0.42, 0.48), uRainFactor * 0.72);

          vec3 shadowCol = mix(vec3(0.06, 0.09, 0.16), vec3(0.58, 0.70, 0.86), uDayFactor);
          shadowCol = mix(shadowCol, vec3(0.46, 0.28, 0.38), uSunsetFactor * 0.75);
          shadowCol = mix(shadowCol, vec3(0.24, 0.28, 0.34), uRainFactor * 0.75);

          vec3 col = mix(shadowCol, brightCol, clamp(ndotl * topShade, 0.0, 1.0));
          col += uSunColor * silverLining * (1.0 - uRainFactor * 0.8);

          float horizDist = length(vWorldPos.xz - uCameraPos.xz);
          float alpha = (1.0 - smoothstep(130.0, 240.0, horizDist)) * 0.88;
          if (alpha < 0.02) discard;
          gl_FragColor = vec4(col, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
    });

    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.y = 110;
    mesh.renderOrder = 4;
    mesh.frustumCulled = false;
    return mesh;
  }

  private createRainSystem(): THREE.Points {
    const count = 2400;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 44;
      positions[i * 3 + 1] = Math.random() * 34;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 44;
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      color: 0xaad2ff,
      size: 0.16,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
    });
    const pts = new THREE.Points(geom, mat);
    pts.visible = false;
    pts.frustumCulled = false;
    return pts;
  }

  private createPostMaterial(w: number, h: number, settings: ShaderSettings): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      uniforms: {
        uSceneTex: { value: this.hdrTarget.texture },
        uResolution: { value: new THREE.Vector2(w, h) },
        uSunScreenPos: { value: this.sunScreenPos },
        uSunVisible: { value: 0.0 },
        uSunColor: this.sharedUniforms.uSunColor,
        uBloomEnabled: { value: settings.bloomAndGodRays ? 1.0 : 0.0 },
        uUnderwater: this.sharedUniforms.uUnderwater,
        uTime: this.sharedUniforms.uTime,
      },
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D uSceneTex;
        uniform vec2 uResolution;
        uniform vec2 uSunScreenPos;
        uniform float uSunVisible;
        uniform vec3 uSunColor;
        uniform float uBloomEnabled;
        uniform float uUnderwater;
        uniform float uTime;
        varying vec2 vUv;

        // ACES Filmic Tone Mapping (exact cinema-grade HDR curve)
        vec3 acesFilm(vec3 x) {
          float a = 2.51;
          float b = 0.03;
          float c = 2.43;
          float d = 0.59;
          float e = 0.14;
          return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
        }

        void main() {
          vec2 uv = vUv;
          if (uUnderwater > 0.5) {
            uv.x += sin(uv.y * 22.0 + uTime * 3.5) * 0.004;
            uv.y += cos(uv.x * 18.0 + uTime * 2.8) * 0.003;
          }

          vec3 col = texture2D(uSceneTex, uv).rgb;
          vec2 texel = 1.0 / uResolution;

          if (uBloomEnabled > 0.5) {
            // Soft HDR Bloom around bright lights (sun, glowstone, torches, lava, specular water)
            vec3 bloom = vec3(0.0);
            float weights = 0.0;
            for (int y = -2; y <= 2; y++) {
              for (int x = -2; x <= 2; x++) {
                vec2 off = vec2(float(x), float(y)) * texel * 3.8;
                vec3 s = texture2D(uSceneTex, clamp(uv + off, 0.001, 0.999)).rgb;
                float lum = dot(s, vec3(0.2126, 0.7152, 0.0722));
                vec3 bright = max(s - vec3(0.95), vec3(0.0)) * smoothstep(0.85, 1.6, lum);
                float w = exp(-float(x * x + y * y) * 0.22);
                bloom += bright * w;
                weights += w;
              }
            }
            col += (bloom / weights) * 0.42;

            // Screen-space Volumetric God Rays / Sun Shafts
            if (uSunVisible > 0.01) {
              vec2 delta = (uv - uSunScreenPos) * (0.48 / 12.0);
              vec2 sampleUv = uv;
              vec3 shafts = vec3(0.0);
              float decay = 1.0;
              for (int i = 0; i < 12; i++) {
                sampleUv -= delta;
                vec3 s = texture2D(uSceneTex, clamp(sampleUv, 0.001, 0.999)).rgb;
                float l = dot(s, vec3(0.3, 0.55, 0.15));
                shafts += max(s - vec3(0.92), vec3(0.0)) * smoothstep(0.9, 2.2, l) * decay;
                decay *= 0.87;
              }
              float radialDist = length(uv - uSunScreenPos);
              float mask = smoothstep(1.15, 0.0, radialDist);
              col += shafts * uSunColor * 0.042 * uSunVisible * mask;
            }
          }

          // Subtle cinematic exposure & ACES Filmic tone mapping
          col = acesFilm(col * 1.08);

          // Subtle edge vignette
          vec2 vigUv = (vUv - 0.5) * 2.0;
          float vig = 1.0 - dot(vigUv, vigUv) * 0.14;
          col *= clamp(vig, 0.78, 1.0);

          // Underwater color grading
          if (uUnderwater > 0.5) {
            col *= vec3(0.58, 0.88, 1.02);
          }

          // Gamma encode to sRGB display
          col = pow(col, vec3(1.0 / 2.2));
          gl_FragColor = vec4(col, 1.0);
        }
      `,
      depthTest: false,
      depthWrite: false,
    });
  }

  public update(dt: number, elapsed: number, camera: THREE.PerspectiveCamera, settings: ShaderSettings, underwater: boolean): void {
    const u = this.sharedUniforms;
    u.uTime.value = elapsed;
    u.uCameraPos.value.copy(camera.position);
    u.uUnderwater.value = underwater ? 1.0 : 0.0;

    // Smooth weather transition
    const targetRain = settings.weather === 'rain' ? 1.0 : 0.0;
    this.rainFactor += (targetRain - this.rainFactor) * Math.min(1, dt * 2.2);
    u.uRainFactor.value = this.rainFactor;

    // Compute Sun & Moon orbit from timeOfDay (0.25 = sunrise, 0.5 = noon, 0.75 = sunset)
    const angle = (settings.timeOfDay - 0.25) * Math.PI * 2;
    const sunElev = Math.sin(angle);
    const sunHoriz = Math.cos(angle);
    this.sunDir.set(sunHoriz * 0.92, sunElev, sunHoriz * 0.36 + 0.28).normalize();
    this.moonDir.copy(this.sunDir).negate();

    const dayFactor = THREE.MathUtils.smoothstep(sunElev, -0.14, 0.22);
    const sunsetFactor = Math.exp(-Math.pow((sunElev - 0.04) / 0.19, 2.0));

    u.uDayFactor.value = dayFactor;
    u.uSunsetFactor.value = sunsetFactor;

    // Active shadow-casting light (Sun during day, Moon at night)
    if (sunElev >= -0.04) {
      this.lightDir.copy(this.sunDir);
      // Blend crisp noon sunlight & rich amber sunset light
      const noonCol = new THREE.Color(1.0, 0.97, 0.91);
      const setCol = new THREE.Color(1.0, 0.52, 0.20);
      this.sunColor.copy(noonCol).lerp(setCol, sunsetFactor * 0.85);
      this.sunColor.multiplyScalar(0.35 + 0.85 * THREE.MathUtils.smoothstep(sunElev, -0.04, 0.28));
    } else {
      this.lightDir.copy(this.moonDir);
      this.sunColor.setRGB(0.24, 0.34, 0.58).multiplyScalar(THREE.MathUtils.smoothstep(-sunElev, 0.04, 0.35));
    }

    // Hemisphere ambient color shifts across day/sunset/night
    const dayAmb = new THREE.Color(0.42, 0.62, 0.92);
    const setAmb = new THREE.Color(0.62, 0.36, 0.38);
    const nightAmb = new THREE.Color(0.06, 0.09, 0.18);
    this.ambientColor.copy(nightAmb).lerp(dayAmb, dayFactor).lerp(setAmb, sunsetFactor * 0.65);

    // Keep SkyDome & Clouds centered over the player (clouds drift slowly!)
    this.skyMesh.position.copy(camera.position);
    const drift = (elapsed * 1.4) % 504;
    this.cloudMesh.position.set(
      Math.floor(camera.position.x / 18) * 18 - (drift % 18),
      110,
      Math.floor(camera.position.z / 18) * 18
    );

    // Update rain particles around player
    this.rainSystem.visible = this.rainFactor > 0.05;
    if (this.rainSystem.visible) {
      this.rainSystem.position.set(camera.position.x, camera.position.y - 12, camera.position.z);
      const posAttr = this.rainSystem.geometry.getAttribute('position') as THREE.BufferAttribute;
      for (let i = 0; i < posAttr.count; i++) {
        let y = posAttr.getY(i) - dt * 28;
        if (y < 0) y += 34;
        posAttr.setY(i, y);
      }
      posAttr.needsUpdate = true;
      (this.rainSystem.material as THREE.PointsMaterial).opacity = this.rainFactor * 0.58;
    }

    // Compute Sun screen-space position for God Rays
    const sunWorld = camera.position.clone().addScaledVector(this.sunDir, 300);
    sunWorld.project(camera);
    const camForward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    const lookingAtSun = camForward.dot(this.sunDir);
    this.sunScreenPos.set(sunWorld.x * 0.5 + 0.5, sunWorld.y * 0.5 + 0.5);
    const sunVis =
      THREE.MathUtils.smoothstep(lookingAtSun, 0.15, 0.72) *
      THREE.MathUtils.smoothstep(sunElev, -0.02, 0.12) *
      (1 - this.rainFactor * 0.85);
    this.postMaterial.uniforms.uSunVisible.value = sunVis;
  }

  public getSunDir(): THREE.Vector3 { return this.sunDir; }
  public getSunColor(): THREE.Color { return this.sunColor; }
  public getDayFactor(): number { return this.sharedUniforms.uDayFactor.value as number; }

  public render(
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
    chunks: Iterable<{ opaqueMesh: THREE.Mesh | null; foliageMesh: THREE.Mesh | null }>,
    settings: ShaderSettings,
    viewmodelScene?: THREE.Scene,
    viewmodelCamera?: THREE.PerspectiveCamera
  ): void {
    // 1. Render Real-Time Soft Shadow Map (with texel snapping to eliminate shadow shimmer!)
    if (settings.shadowsEnabled && settings.quality !== 'vanilla') {
      const shadowSpan = 64;
      const texelWorld = (shadowSpan * 2) / this.shadowTarget.width;
      const snapX = Math.round(camera.position.x / texelWorld) * texelWorld;
      const snapY = Math.round(camera.position.y / texelWorld) * texelWorld;
      const snapZ = Math.round(camera.position.z / texelWorld) * texelWorld;

      const target = new THREE.Vector3(snapX, snapY, snapZ);
      this.shadowCamera.position.copy(target).addScaledVector(this.lightDir, 115);
      this.shadowCamera.lookAt(target);
      this.shadowCamera.updateMatrixWorld();
      this.shadowMatrix.multiplyMatrices(this.shadowCamera.projectionMatrix, this.shadowCamera.matrixWorldInverse);

      // Temporarily swap chunk materials for clean fast depth pass (including swaying leaf cutouts!)
      const swaps: [THREE.Mesh, THREE.Material | THREE.Material[]][] = [];
      for (const c of chunks) {
        if (c.opaqueMesh) {
          swaps.push([c.opaqueMesh, c.opaqueMesh.material]);
          c.opaqueMesh.material = this.opaqueDepthMat;
        }
        if (c.foliageMesh) {
          swaps.push([c.foliageMesh, c.foliageMesh.material]);
          c.foliageMesh.material = this.foliageDepthMat;
        }
      }

      const prevSkyVis = this.skyMesh.visible;
      const prevCloudVis = this.cloudMesh.visible;
      const prevRainVis = this.rainSystem.visible;
      this.skyMesh.visible = false;
      this.cloudMesh.visible = false;
      this.rainSystem.visible = false;

      this.renderer.setRenderTarget(this.shadowTarget);
      this.renderer.clear(true, true, false);
      this.renderer.render(scene, this.shadowCamera);

      for (const [mesh, mat] of swaps) {
        mesh.material = mat;
      }
      this.skyMesh.visible = prevSkyVis;
      this.cloudMesh.visible = prevCloudVis;
      this.rainSystem.visible = prevRainVis;
    }

    // 2. Render Main 3D Scene into HDR Target
    this.renderer.setRenderTarget(this.hdrTarget);
    this.renderer.clear(true, true, false);
    this.renderer.render(scene, camera);

    // 2b. Render First-Person Viewmodel Hand/Block on top with fresh depth buffer so it never clips into walls
    if (viewmodelScene && viewmodelCamera) {
      const prevAutoClear = this.renderer.autoClear;
      this.renderer.autoClear = false;
      this.renderer.clearDepth();
      this.renderer.render(viewmodelScene, viewmodelCamera);
      this.renderer.autoClear = prevAutoClear;
    }

    // 3. Render Post-Process Pass (Bloom + Volumetric God Rays + ACES Filmic Tone Mapping) to Screen
    this.renderer.setRenderTarget(null);
    this.renderer.render(this.postScene, this.postCamera);
  }
}
