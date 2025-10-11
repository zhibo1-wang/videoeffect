// Copyright (C) <2025> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

import { WebGPUBlur } from './webgpu-blur.js';

function getTextureFormat(directOutput) {
  return directOutput ? navigator.gpu.getPreferredCanvasFormat() : 'rgba8unorm';
}

function getOrCreateResource(cache, key, createFn) {
  if (!cache[key]) {
    console.log("Creating new resource for ", key);
    cache[key] = createFn();
  }
  return cache[key];
}

function getOrCreateTexture(device, cache, key, size, directOutput, usage) {
  const [width, height] = size;
  const format = getTextureFormat(directOutput);
  const cacheKey = `${key}_${width}x${height}_${format}_${usage}`;

  let texture = cache[cacheKey];
  if (!texture || texture.width !== width || texture.height !== height) {
    console.log("Creating new texture for ", cacheKey, " with format ", format, " and usage ", usage);
    if (texture) {
      texture.destroy();
    }
    texture = device.createTexture({
      size,
      format,
      usage,
    });
    cache[cacheKey] = texture;
  }
  return texture;
}

let imageDataCache;

async function renderWithWebGPU(params, videoFrame, resourceCache) {
  const device = params.device;
  const webgpuCanvas = params.webgpuCanvas;
  const width = params.webgpuCanvas.width;
  const height = params.webgpuCanvas.height;
  const segmenter = params.segmenter;
  const blurrer = params.blurrer;

  // Import external texture.
  let sourceTexture;
  if (params.zeroCopy) {
    sourceTexture = device.importExternalTexture({
      source: videoFrame,
    });
  } else {
    sourceTexture = getOrCreateTexture(device, resourceCache, 'sourceTexture',
      [videoFrame.displayWidth, videoFrame.displayHeight, 1],
      params.directOutput,
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT);

    device.queue.copyExternalImageToTexture(
      { source: videoFrame },
      { texture: sourceTexture },
      [videoFrame.displayWidth, videoFrame.displayHeight]
    );
  }
  // Scale down input, segment, read back and upload.
  let maskTexture;
  {
    const segmentationWidth = params.segmentationWidth;
    const segmentationHeight = params.segmentationHeight;
    const destTexture = getOrCreateTexture(device, resourceCache, 'downscaleDest',
      [segmentationWidth, segmentationHeight, 1],
      params.directOutput,
      GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC);

    let downscaleBindGroup;
    try {
      downscaleBindGroup = device.createBindGroup({
        layout: params.downscalePipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0, resource: params.zeroCopy ? sourceTexture : sourceTexture.createView()
          },
          { binding: 1, resource: params.downscaleSampler },
          { binding: 2, resource: destTexture.createView() },
        ],
      });
    } catch (error) {
      console.warn('bind failed:', error);
    }

    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(params.downscalePipeline);
    computePass.setBindGroup(0, downscaleBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(segmentationWidth / 8), Math.ceil(segmentationHeight / 8));
    computePass.end();

    const bufferSize = segmentationWidth * segmentationHeight * 4;
    const readbackBuffer = getOrCreateResource(resourceCache, `readbackBuffer${bufferSize}`, () =>
      device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }));
    commandEncoder.copyTextureToBuffer(
      { texture: destTexture },
      { buffer: readbackBuffer, bytesPerRow: segmentationWidth * 4 },
      [segmentationWidth, segmentationHeight]
    );
    device.queue.submit([commandEncoder.finish()]);
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const downscaledImageData = imageDataCache ?
      imageDataCache :
      new ImageData(segmentationWidth, segmentationHeight);
    downscaledImageData.data.set(new Uint8Array(readbackBuffer.getMappedRange()));
    readbackBuffer.unmap();

    // Segment
    const maskImageData = await segmenter(downscaledImageData);

    // Upload
    maskTexture = getOrCreateTexture(device, resourceCache, 'maskTexture',
      [maskImageData.width, maskImageData.height, 1],
      params.directOutput,
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT);

    device.queue.writeTexture(
      { texture: maskTexture },
      maskImageData.data,
      { bytesPerRow: maskImageData.width * 4 },
      [maskImageData.width, maskImageData.height, 1]
    );
  }

  // Always process at full video resolution, ignore display size
  const processingWidth = videoFrame.displayWidth || 1280;
  const processingHeight = videoFrame.displayHeight || 720;

  // Update canvas size only if video resolution actually changed
  if (webgpuCanvas.width !== processingWidth || webgpuCanvas.height !== processingHeight) {
    webgpuCanvas.width = processingWidth;
    webgpuCanvas.height = processingHeight;
    // Reconfigure context with actual video size
    context.configure({
      device: device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: 'premultiplied',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | (params.directOutput ? GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST : 0),
    });
  }

  const canvasTexture = params.context.getCurrentTexture();
  const outputTexture = getOrCreateTexture(device, resourceCache, 'outputTexture',
    [width, height, 1],
    params.directOutput,
    GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING);

  const commandEncoder = device.createCommandEncoder();
  blurrer.setInputDimensions(processingWidth, processingHeight);
  blurrer.blur(commandEncoder, sourceTexture, maskTexture, params.directOutput ? canvasTexture : outputTexture, 360);

  if (!params.directOutput) {
    const renderPipeline = getOrCreateResource(resourceCache, `renderPipeline_${width}x${height}`, () =>
      device.createRenderPipeline({
        layout: 'auto',
        vertex: {
          module: params.outputRendererVertexShader,
          entryPoint: 'main',
        },
        fragment: {
          module: params.getOutputRendererFragmentShader(device, width, height),
          entryPoint: 'main',
          targets: [{
            format: navigator.gpu.getPreferredCanvasFormat(),
          }],
        },
        primitive: {
          topology: 'triangle-list',
        },
      }));

    const renderBindGroup = device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: outputTexture.createView() },
        { binding: 1, resource: params.renderSampler },
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

  device.queue.submit([commandEncoder.finish()]);

  // Create a new VideoFrame from the processed WebGPU canvas
  const processedVideoFrame = new VideoFrame(params.webgpuCanvas, {
    timestamp: videoFrame.timestamp,
    duration: videoFrame.duration
  });

  return processedVideoFrame;
}

// WebGPU blur renderer
export async function createWebGPUBlurRenderer(segmenter, zeroCopy, directOutput) {
  console.log("createWebGPUBlurRenderer zeroCopy: ", zeroCopy, " directOutput: ", directOutput);
  // Always use full resolution for processing, regardless of display size
  const webgpuCanvas = new OffscreenCanvas(1280, 720);

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('WebGPU adapter not available');
  }

  // Ensure we're compatible with directOutput
  console.log("Adapter features:");
  for (const feature of adapter.features) {
    console.log(`- ${feature}`);
  }
  if (!adapter.features.has('bgra8unorm-storage')) {
    console.log("BGRA8UNORM-STORAGE not supported");
  }
  const device = await adapter.requestDevice({ requiredFeatures: ['bgra8unorm-storage'] });
  const context = webgpuCanvas.getContext('webgpu');

  if (!context) {
    throw new Error('WebGPU context not available');
  }

  context.configure({
    device: device,
    format: navigator.gpu.getPreferredCanvasFormat(),
    alphaMode: 'premultiplied',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | (directOutput ? GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST : 0),
  });

  const segmentationWidth = 256;
  const segmentationHeight = 144;

  const downscaleShaderCode = `
      @group(0) @binding(0) var inputTexture: ${zeroCopy ? "texture_external" : "texture_2d<f32>"};
      @group(0) @binding(1) var textureSampler: sampler;
      @group(0) @binding(2) var outputTexture: texture_storage_2d<${getTextureFormat(directOutput)}, write>;

      @compute @workgroup_size(8, 8)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let outputDims = textureDimensions(outputTexture);
        if (global_id.x >= outputDims.x || global_id.y >= outputDims.y) {
            return;
        }
        let uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5, 0.5)) / vec2<f32>(outputDims);
        let color = textureSampleBaseClampToEdge(inputTexture, textureSampler, uv);
        textureStore(outputTexture, global_id.xy, color);
      }
    `;

  const downscaleShader = device.createShaderModule({ code: downscaleShaderCode });
  const downscalePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: downscaleShader,
      entryPoint: 'main',
    },
  });

  const downscaleSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  const blurrer = new WebGPUBlur(device, zeroCopy, directOutput);
  await blurrer.init();

  // Create a simple render pipeline to copy the compute shader's output (RGBA)
  // to the canvas, which might have a different format (e.g., BGRA).
  const outputRendererVertexShader = device.createShaderModule({
    code: `
        @vertex
        fn main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
          var pos = array<vec2<f32>, 6>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>( 1.0, -1.0),
            vec2<f32>(-1.0,  1.0),
            vec2<f32>( 1.0, -1.0),
            vec2<f32>( 1.0,  1.0),
            vec2<f32>(-1.0,  1.0)
          );
          return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
        }
      `
  });

  let outputRendererFragmentShader;
  let lastDim;

  function getOutputRendererFragmentShader(device, width, height) {
    if (!outputRendererFragmentShader || lastDim !== `${width}x${height}`) {
      outputRendererFragmentShader = device.createShaderModule({
        code: `
            @group(0) @binding(0) var inputTexture: texture_2d<f32>;
            @group(0) @binding(1) var textureSampler: sampler;
          
            @fragment
            fn main(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
              let uv = coord.xy / vec2<f32>(${width}.0, ${height}.0);
              return textureSample(inputTexture, textureSampler, uv);
            }
          `});

      lastDim = `${width}x${height}`;
    }
    return outputRendererFragmentShader;
  }

  const renderSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  const resourceCache = {};

  return {
    render: async (videoFrame) => {
      const params = {
        device,
        context,
        webgpuCanvas,
        outputRendererVertexShader,
        getOutputRendererFragmentShader,
        segmentationWidth,
        segmentationHeight,
        downscalePipeline,
        downscaleSampler,
        renderSampler,
        zeroCopy,
        directOutput,
        segmenter,
        blurrer
      };
      try {
        return await renderWithWebGPU(params, videoFrame, resourceCache);
      } catch (error) {
        console.warn('WebGPU rendering failed:', error);
      }
    }
  };
}
