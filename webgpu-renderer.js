// Copyright (C) <2025> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

import { WebGPUBlur } from './webgpu-blur.js';

class WebGPURenderer {
  constructor(device, segmenter, blurrer, { zeroCopy, directOutput }) {
    console.log("createWebGPUBlurRenderer zeroCopy:", zeroCopy, "directOutput:", directOutput);
    this.device = device;
    this.segmenter = segmenter;
    this.blurrer = blurrer;
    this.zeroCopy = zeroCopy;
    this.directOutput = directOutput;

    // Always use full resolution for processing, regardless of display size
    this.webgpuCanvas = new OffscreenCanvas(1280, 720);
    this.context = this.webgpuCanvas.getContext('webgpu');
    if (!this.context) {
      throw new Error('WebGPU context not available');
    }

    this.format = directOutput ? navigator.gpu.getPreferredCanvasFormat() : 'rgba8unorm';
    this.context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: 'premultiplied',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | (directOutput ? GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST : 0),
    });

    this.segmentationWidth = 256;
    this.segmentationHeight = 144;
    this.downscaledImageData = new ImageData(this.segmentationWidth, this.segmentationHeight);
    this.downscaleSampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    let packShaderUrl;
    if (this.segmenter.segmentGPUBuffer && this.segmenter.deviceType === 'gpu') {
      packShaderUrl = 'blur4/shaders/downscale-and-convert-to-rgb16float.wgsl';
    } else {
      packShaderUrl = 'blur4/shaders/downscale-and-convert-to-rgba8uint.wgsl';
    }
    this.downscaleModule = fetch(packShaderUrl).then(res => res.text())
      .then(code => this.device.createShaderModule({
        code: code.replace(/\${(\w+)}/g, (...groups) => ({
          inputTextureType: zeroCopy ? "texture_external" : "texture_2d<f32>",
          outputTextureFormat: this.format,
        }[groups[1]]))
      }));
    this.downscalePipeline = this.downscaleModule
      .then(module => this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: module, entryPoint: 'main' },
      }));

    // Create a simple render pipeline to copy the compute shader's output (RGBA)
    // to the canvas, which might have a different format (e.g., BGRA).
    this.outputRendererVertexShader = fetch('blur4/shaders/render.vertex.wgsl').then(res => res.text())
      .then(code => this.device.createShaderModule({ code }));

    this.outputRendererFragmentShader = null;
    this.lastDim = null;
    this.renderSampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    this.resourceCache = {};
  }

  async getOrCreateResource(key, createFn) {
    if (!this.resourceCache[key]) {
      console.log("Creating new resource", key);
      this.resourceCache[key] = await createFn();
    }
    return this.resourceCache[key];
  }

  getOrCreateTexture(key, size, usage) {
    const [width, height] = size;
    const cacheKey = `${key}_${width}x${height}_${this.format}_${usage}`;

    let texture = this.resourceCache[cacheKey];
    if (!texture || texture.width !== width || texture.height !== height) {
      console.log("Creating new texture", cacheKey, "with format", this.format, "and usage", usage);
      if (texture) {
        console.log("Destroying old texture", cacheKey);
        texture.destroy();
      }
      texture = this.device.createTexture({ size, format: this.format, usage });
      this.resourceCache[cacheKey] = texture;
    }
    return texture;
  }

  async getOutputRendererFragmentShader(device, width, height) {
    if (!this.outputRendererFragmentShader || this.lastDim !== [width, height]) {
      this.outputRendererFragmentShader = await this.device.createShaderModule({
        code: await fetch('blur4/shaders/render.fragment.wgsl').then(res => res.text())
          .then(code => code.replace(/\${(\w+)}/g, (...groups) => ({ width, height }[groups[1]]))),
      });

      this.lastDim = [width, height];
    }
    return this.outputRendererFragmentShader;
  }

  async segment(sourceTexture) {
    const commandEncoder = this.device.createCommandEncoder();
    const useInterop = this.segmenter.segmentGPUBuffer && this.segmenter.deviceType === 'gpu';

    let downscaledResource;
    if (useInterop) {
      downscaledResource = { buffer: await this.segmenter.getInputBuffer(this.device) };
    } else {
      downscaledResource = {
        texture: this.getOrCreateTexture(
          'downscaledTexture',
          [this.segmentationWidth, this.segmentationHeight, 1],
          GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT
        )
      };
    }
    const packBindGroup = this.device.createBindGroup({
      layout: (await this.downscalePipeline).getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.zeroCopy ? sourceTexture : sourceTexture.createView() },
        { binding: 1, resource: this.downscaleSampler },
        { binding: 2, resource: downscaledResource.texture },
      ],
    });

    const packPass = commandEncoder.beginComputePass();
    packPass.setPipeline(await this.downscalePipeline);
    packPass.setBindGroup(0, packBindGroup);
    packPass.dispatchWorkgroups(Math.ceil(this.segmentationWidth / 8), Math.ceil(this.segmentationHeight / 8));
    packPass.end();

    if (useInterop) {
      this.device.queue.submit([commandEncoder.finish()]);
      downscaledResource.buffer.destroy();

      await this.segmenter.segmentGPUBuffer();

      const outputBuffer = await this.segmenter.getOutputBuffer();
      if (!this.maskTexture) {
        this.maskTexture = this.device.createTexture({
          size: [this.segmentationWidth, this.segmentationHeight, 1],
          format: 'r16float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
      }
      const copyCommandEncoder = this.device.createCommandEncoder();
      copyCommandEncoder.copyBufferToTexture(
        { buffer: outputBuffer, bytesPerRow: this.segmentationWidth * 2 },
        { texture: this.maskTexture },
        [this.segmentationWidth, this.segmentationHeight, 1]
      );
      this.device.queue.submit([copyCommandEncoder.finish()]);
      outputBuffer.destroy();

      return this.maskTexture;
    } else {
      const bufferSize = this.segmentationWidth * this.segmentationHeight * 4;
      const readbackBuffer = await this.getOrCreateResource(`readbackBuffer${bufferSize}`, async () =>
        this.device.createBuffer({
          size: bufferSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }));
      commandEncoder.copyTextureToBuffer(
        downscaledResource,
        { buffer: readbackBuffer, bytesPerRow: this.segmentationWidth * 4 },
        [this.segmentationWidth, this.segmentationHeight]
      );
      this.device.queue.submit([commandEncoder.finish()]);
      await readbackBuffer.mapAsync(GPUMapMode.READ);
      this.downscaledImageData.data.set(new Uint8Array(readbackBuffer.getMappedRange()));
      readbackBuffer.unmap();

      console.log(this.downscaledImageData);
      const maskImageData = await this.segmenter.segment(this.downscaledImageData);
      console.log(maskImageData);

      // Upload
      const maskTexture = this.getOrCreateTexture(
        'maskTexture',
        [maskImageData.width, maskImageData.height, 1],
        GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      );
      this.device.queue.writeTexture(
        { texture: maskTexture },
        maskImageData.data,
        { bytesPerRow: maskImageData.width * 4 },
        [maskImageData.width, maskImageData.height, 1]
      );

      return maskTexture;
    }
  }

  async render(videoFrame) {
    // Import external texture.
    let sourceTexture;
    if (this.zeroCopy) {
      sourceTexture = this.device.importExternalTexture({ source: videoFrame });
    } else {
      sourceTexture = this.getOrCreateTexture(
        'sourceTexture',
        [videoFrame.displayWidth, videoFrame.displayHeight, 1],
        GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      );

      this.device.queue.copyExternalImageToTexture(
        { source: videoFrame },
        { texture: sourceTexture },
        [videoFrame.displayWidth, videoFrame.displayHeight]
      );
    }

    let maskTexture = await this.segment(sourceTexture);

    // Always process at full video resolution, ignore display size
    const processingWidth = videoFrame.displayWidth || 1280;
    const processingHeight = videoFrame.displayHeight || 720;

    // Update canvas size only if video resolution actually changed
    if (this.webgpuCanvas.width !== processingWidth || this.webgpuCanvas.height !== processingHeight) {
      this.webgpuCanvas.width = processingWidth;
      this.webgpuCanvas.height = processingHeight;
      // Reconfigure context with actual video size
      this.context.configure({
        device: this.device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        alphaMode: 'premultiplied',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | (this.directOutput ? GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST : 0),
      });
    }
    const width = this.webgpuCanvas.width;
    const height = this.webgpuCanvas.height;

    const canvasTexture = this.context.getCurrentTexture();
    const outputTexture = this.getOrCreateTexture(
      'outputTexture',
      [width, height, 1],
      GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING
    );

    const commandEncoder = this.device.createCommandEncoder();
    this.blurrer.setInputDimensions(processingWidth, processingHeight);
    this.blurrer.blur(commandEncoder, sourceTexture, maskTexture, this.directOutput ? canvasTexture : outputTexture, 360);

    if (!this.directOutput) {
      const renderPipeline = await this.getOrCreateResource(`renderPipeline_${width}x${height}`, async () =>
        this.device.createRenderPipeline({
          layout: 'auto',
          vertex: {
            module: await this.outputRendererVertexShader,
            entryPoint: 'main',
          },
          fragment: {
            module: await this.getOutputRendererFragmentShader(this.device, width, height),
            entryPoint: 'main',
            targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
          },
          primitive: {
            topology: 'triangle-list',
          },
        }));

      const renderBindGroup = this.device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: outputTexture.createView() },
          { binding: 1, resource: this.renderSampler },
        ],
      });

      // Render to canvas
      const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: canvasTexture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });

      renderPass.setPipeline(renderPipeline);
      renderPass.setBindGroup(0, renderBindGroup);
      renderPass.draw(6);
      renderPass.end();
    }

    this.device.queue.submit([commandEncoder.finish()]);

    // Create a new VideoFrame from the processed WebGPU canvas
    return new VideoFrame(this.webgpuCanvas, {
      timestamp: videoFrame.timestamp,
      duration: videoFrame.duration
    });
  }
}

export async function getWebGPUDevice() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('WebGPU adapter not available');
  }

  // Ensure we're compatible with directOutput
  console.log("Adapter features:");
  console.log([...adapter.features]);
  const requiredFeatures = ['bgra8unorm-storage', 'shader-f16', 'texture-formats-tier1'];
  for (const feature of requiredFeatures) {
    if (!adapter.features.has(feature)) {
      console.log(`${feature} is not supported`);
    }
  }
  const device = await adapter.requestDevice({ requiredFeatures });
  if (!device) {
    console.error('WebGPU adapter does not support the required features:', requiredFeatures);
  }

  return device;
}

// WebGPU blur renderer
export async function createWebGPUBlurRenderer(device, segmenter, zeroCopy, directOutput) {
  const blurrer = new WebGPUBlur(device, zeroCopy, directOutput);
  await blurrer.init();

  return new WebGPURenderer(device, segmenter, blurrer, { zeroCopy, directOutput });
}
