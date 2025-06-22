
using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using LibDxGeometryRendering.Internal;
using Silk.NET.Core.Native;
using Silk.NET.Direct3D;
using Silk.NET.Direct3D.Compilers;
using Silk.NET.Direct3D11;
using Silk.NET.DXGI;
using Silk.NET.Maths;


namespace LibDxGeometryRendering
{
    public unsafe class QuadsRenderer : IDisposable
    {
        private readonly int _width;
        private readonly int _height;
        private readonly int _maxQuadCount;
        private int _quadCount;

        // 添加抗锯齿相关字段
        private bool _antialiasingEnabled;
        private ComPtr<ID3D11Texture2D> _resolveTexture; // 用于存储多重采样解析结果

        // D3D11 API
        private ComPtr<ID3D11Device> _device;
        private ComPtr<ID3D11DeviceContext> _context;
        private ComPtr<ID3D11RenderTargetView> _renderTargetView;
        private ComPtr<ID3D11Texture2D> _renderTarget;
        private ComPtr<ID3D11Texture2D> _stagingTexture;

        // Shader resources
        private ComPtr<ID3D11VertexShader> _vertexShader;
        private ComPtr<ID3D11GeometryShader> _geometryShader;
        private ComPtr<ID3D11PixelShader> _pixelShader;
        private ComPtr<ID3D11InputLayout> _inputLayout;

        // Buffer resources
        private ComPtr<ID3D11Buffer> _constantBuffer;
        private ComPtr<ID3D11Buffer> _vertexBuffer;
        private bool _disposedValue;

        // Viewport
        private readonly Viewport _viewport;

        // Vertex layout
        private readonly int _quadStride = Unsafe.SizeOf<Quad>();

        // Constants
        private MatrixTransform _transform = MatrixTransform.Identity;
        private float _strokeThicknessFactor = 1;
        private float _widthFactor = 1;
        private float _heightFactor = 1;

        public int Width => _width;
        public int Height => _height;
        public int MaxQuadCount => _maxQuadCount;

        public QuadsRenderer(int width, int height, int maxQuadCount)
        {
            _width = width;
            _height = height;
            _maxQuadCount = maxQuadCount;

            // Create device and context
            CreateDeviceContext(out _device, out _context);

            // Create render targets
            CreateRenderTargets();

            // Compile and create shaders
            CreateShaders();

            // Create input layout
            CreateInputLayout();

            // Create vertex buffer
            CreateVertexBuffer();

            // Create constant buffer
            CreateConstantBuffer();

            // Set up viewport
            _viewport = new Viewport
            {
                Width = (float)width,
                Height = (float)height,
                MinDepth = 0.0f,
                MaxDepth = 1.0f,
                TopLeftX = 0,
                TopLeftY = 0
            };

            // Configure rasterizer state
            ConfigureRasterizer();

            // Configure blend state
            ConfigureBlendState();
        }

        private void CreateDeviceContext(out ComPtr<ID3D11Device> device, out ComPtr<ID3D11DeviceContext> context)
        {
            using var d3d11 = D3D11.GetApi(null, false);

            // Create D3D11 device and context
            D3DFeatureLevel[] featureLevels = { D3DFeatureLevel.Level111, D3DFeatureLevel.Level110, D3DFeatureLevel.Level100 };

            // Create with debug layer in debug builds (奇怪, 只要不带 Debug flag, 就会异常!)
            uint flags = (uint)CreateDeviceFlag.Debug;

            device = default;
            context = default;
            D3DFeatureLevel featureLevel;
            fixed (D3DFeatureLevel* featureLevelsPtr = featureLevels)
            {
                int createDeviceError = d3d11.CreateDevice(
                    ref Unsafe.NullRef<IDXGIAdapter>(),
                    D3DDriverType.Hardware,
                    0,
                    flags,
                    featureLevelsPtr,
                    (uint)featureLevels.Length,
                    D3D11.SdkVersion, ref device, &featureLevel, ref context);
                if (createDeviceError != 0)
                {
                    throw new InvalidOperationException("Failed to create device");
                }
            }
        }

        private void CreateRenderTargets()
        {
            // 创建渲染目标纹理
            Texture2DDesc rtDesc = new Texture2DDesc
            {
                Width = (uint)_width,
                Height = (uint)_height,
                MipLevels = 1,
                ArraySize = 1,
                Format = Format.FormatB8G8R8A8Unorm,
                SampleDesc = _antialiasingEnabled ?
                    new SampleDesc { Count = 4, Quality = 0 } : // 4x MSAA
                    new SampleDesc { Count = 1, Quality = 0 },  // 无抗锯齿
                Usage = Usage.Default,
                BindFlags = (uint)BindFlag.RenderTarget | (uint)BindFlag.ShaderResource,
                CPUAccessFlags = 0,
                MiscFlags = 0
            };
            // 创建渲染目标纹理
            _device.CreateTexture2D(in rtDesc, null, ref _renderTarget);
            // 创建渲染目标视图
            RenderTargetViewDesc rtvDesc = new RenderTargetViewDesc
            {
                Format = Format.FormatB8G8R8A8Unorm,
                ViewDimension = _antialiasingEnabled ?
                    RtvDimension.Texture2Dms :  // 多重采样
                    RtvDimension.Texture2D,     // 无多重采样
                                                // 不需要设置联合字段
            };
            _device.CreateRenderTargetView(_renderTarget, in rtvDesc, ref _renderTargetView);
            // 如果启用了抗锯齿，创建resolve纹理，用于存储解析结果
            if (_antialiasingEnabled)
            {
                // 创建非多重采样的纹理，用于解析
                Texture2DDesc resolveDesc = rtDesc;
                resolveDesc.SampleDesc.Count = 1;
                resolveDesc.SampleDesc.Quality = 0;

                _device.CreateTexture2D(in resolveDesc, null, ref _resolveTexture);
            }

            // 创建staging纹理用于回读（始终是非多重采样的）
            Texture2DDesc stagingDesc = new Texture2DDesc
            {
                Width = (uint)_width,
                Height = (uint)_height,
                MipLevels = 1,
                ArraySize = 1,
                Format = Format.FormatB8G8R8A8Unorm,
                SampleDesc = new SampleDesc { Count = 1, Quality = 0 }, // 始终非多重采样
                Usage = Usage.Staging,
                BindFlags = 0,
                CPUAccessFlags = (uint)CpuAccessFlag.Read,
                MiscFlags = 0
            };
            _device.CreateTexture2D(in stagingDesc, null, ref _stagingTexture);
        }

        private void CreateShaders()
        {
            // Assuming we have shader code as a byte array (compiled offline or embedded resource)
            var vsBytes = CompileShader("VS", "vs_5_0");
            var gsBytes = CompileShader("GS", "gs_5_0");
            var psBytes = CompileShader("PS", "ps_5_0");

            fixed (byte* vsBytesPtr = vsBytes)
            fixed (byte* gsBytesPtr = gsBytes)
            fixed (byte* psBytesPtr = psBytes)
            {
                _device.CreateVertexShader(vsBytesPtr, (nuint)vsBytes.Length, ref Unsafe.NullRef<ID3D11ClassLinkage>(), ref _vertexShader);
                _device.CreateGeometryShader(gsBytesPtr, (nuint)gsBytes.Length, ref Unsafe.NullRef<ID3D11ClassLinkage>(), ref _geometryShader);
                _device.CreatePixelShader(psBytesPtr, (nuint)psBytes.Length, ref Unsafe.NullRef<ID3D11ClassLinkage>(), ref _pixelShader);
            }
        }

        private void CreateConstantBuffer()
        {
            // 创建常量缓冲区 - 现在需要更大空间来存储变换矩阵
            // float2 screenSize + float3x3 transform (需要16字节对齐)
            BufferDesc constBufferDesc = new BufferDesc
            {
                BindFlags = (uint)BindFlag.ConstantBuffer,
                // 常量缓冲区结构：
                // float2 screenSize; (8字节，但会被填充到16字节)
                // float3x3 transform; (9个float，36字节，但会被填充到48字节)
                // float3 strokeWidthFactorAndSizeFactor;  (16bytes
                // 总共 80 字节
                ByteWidth = 80,
                CPUAccessFlags = (uint)CpuAccessFlag.Write,
                Usage = Usage.Dynamic
            };

            ReadOnlySpan<float> initConstBufferData =
            [
                _width, _height, 0, 0,
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                1, 1, 1, 0
            ];

            fixed (float* initConstBufferDataPtr = initConstBufferData)
            {
                SubresourceData subResourceData = new SubresourceData()
                {
                    PSysMem = initConstBufferDataPtr,
                };

                _device.CreateBuffer(in constBufferDesc, ref subResourceData, ref _constantBuffer);
            }
        }

        private void CreateInputLayout()
        {
            // Define input layout elements corresponding to our vertex structure
            InputElementDesc[] layoutDesc = new InputElementDesc[]
            {
                new InputElementDesc
                {
                    SemanticName = (byte*)Marshal.StringToCoTaskMemAnsi("POSITION"),
                    SemanticIndex = 0,
                    Format = Format.FormatR32G32Float,
                    InputSlot = 0,
                    AlignedByteOffset = 0,
                    InputSlotClass = InputClassification.PerVertexData,
                    InstanceDataStepRate = 0
                },
                new InputElementDesc
                {
                    SemanticName = (byte*)Marshal.StringToCoTaskMemAnsi("SIZE"),
                    SemanticIndex = 0,
                    Format = Format.FormatR32G32Float,
                    InputSlot = 0,
                    AlignedByteOffset = 8,
                    InputSlotClass = InputClassification.PerVertexData,
                    InstanceDataStepRate = 0
                },
                new InputElementDesc
                {
                    SemanticName = (byte*)Marshal.StringToCoTaskMemAnsi("ROTATION"),
                    SemanticIndex = 0,
                    Format = Format.FormatR32Float,
                    InputSlot = 0,
                    AlignedByteOffset = 16,
                    InputSlotClass = InputClassification.PerVertexData,
                    InstanceDataStepRate = 0
                },
                new InputElementDesc
                {
                    SemanticName = (byte*)Marshal.StringToCoTaskMemAnsi("STROKE"),
                    SemanticIndex = 0,
                    Format = Format.FormatR32Float,
                    InputSlot = 0,
                    AlignedByteOffset = 20,
                    InputSlotClass = InputClassification.PerVertexData,
                    InstanceDataStepRate = 0
                },
                new InputElementDesc
                {
                    SemanticName = (byte*)Marshal.StringToCoTaskMemAnsi("COLOR"),
                    SemanticIndex = 0,
                    Format = Format.FormatR8G8B8A8Unorm,
                    InputSlot = 0,
                    AlignedByteOffset = 24,
                    InputSlotClass = InputClassification.PerVertexData,
                    InstanceDataStepRate = 0
                },
                new InputElementDesc
                {
                    SemanticName = (byte*)Marshal.StringToCoTaskMemAnsi("COLOR"),
                    SemanticIndex = 1,
                    Format = Format.FormatR8G8B8A8Unorm,
                    InputSlot = 0,
                    AlignedByteOffset = 28,
                    InputSlotClass = InputClassification.PerVertexData,
                    InstanceDataStepRate = 0
                }
            };

            var vsBytes = CompileShader("VS", "vs_5_0");
            fixed (byte* vsBytesPtr = vsBytes)
            {
                _device.CreateInputLayout(
                    in layoutDesc[0],
                    (uint)layoutDesc.Length,
                    vsBytesPtr,
                    (nuint)vsBytes.Length,
                    _inputLayout.GetAddressOf());
            }

            // Free allocated memory for semantic names
            foreach (var element in layoutDesc)
            {
                Marshal.FreeCoTaskMem((nint)element.SemanticName);
            }
        }

        private void CreateVertexBuffer()
        {
            // Create vertex buffer
            BufferDesc vbDesc = new BufferDesc
            {
                ByteWidth = (uint)(_maxQuadCount * _quadStride),
                Usage = Usage.Dynamic,
                BindFlags = (uint)BindFlag.VertexBuffer,
                CPUAccessFlags = (uint)CpuAccessFlag.Write,
                MiscFlags = 0,
                StructureByteStride = (uint)_quadStride
            };

            _device.CreateBuffer(in vbDesc, null, ref _vertexBuffer);
        }

        private void ConfigureRasterizer()
        {
            // Create and set rasterizer state for proper triangle winding
            RasterizerDesc rastDesc = new RasterizerDesc
            {
                FillMode = FillMode.Solid,
                CullMode = CullMode.None,
                FrontCounterClockwise = 0, // FALSE
                DepthBias = 0,
                DepthBiasClamp = 0.0f,
                SlopeScaledDepthBias = 0.0f,
                DepthClipEnable = 1, // TRUE
                ScissorEnable = 0, // FALSE
                MultisampleEnable = 0, // FALSE
                AntialiasedLineEnable = 0 // FALSE
            };

            ComPtr<ID3D11RasterizerState> rasterizerState = new();
            _device.CreateRasterizerState(&rastDesc, rasterizerState.GetAddressOf());
            _context.RSSetState(rasterizerState);
        }

        // 添加以下方法来设置混合状态
        private void ConfigureBlendState()
        {
            // 创建混合状态
            BlendDesc blendDesc = new BlendDesc
            {
                AlphaToCoverageEnable = 0, // FALSE
                IndependentBlendEnable = 0, // FALSE - 所有渲染目标使用相同的混合设置
            };

            // 设置第一个渲染目标的混合属性
            blendDesc.RenderTarget[0].BlendEnable = 1; // TRUE - 启用混合
            blendDesc.RenderTarget[0].SrcBlend = Blend.SrcAlpha; // 源因子为源alpha值
            blendDesc.RenderTarget[0].DestBlend = Blend.InvSrcAlpha; // 目标因子为(1 - 源alpha)
            blendDesc.RenderTarget[0].BlendOp = BlendOp.Add; // 混合操作为加法
            blendDesc.RenderTarget[0].SrcBlendAlpha = Blend.One; // Alpha源因子为1
            blendDesc.RenderTarget[0].DestBlendAlpha = Blend.InvSrcAlpha; // Alpha目标因子为(1 - 源alpha)
            blendDesc.RenderTarget[0].BlendOpAlpha = BlendOp.Add; // Alpha混合操作为加法
            blendDesc.RenderTarget[0].RenderTargetWriteMask = (byte)ColorWriteEnable.All; // 允许写入所有颜色通道
                                                                                          // 创建混合状态对象
            ComPtr<ID3D11BlendState> blendState = new();
            _device.CreateBlendState(&blendDesc, blendState.GetAddressOf());

            // 应用混合状态到管道
            float[] blendFactor = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };
            fixed (float* pBlendFactor = blendFactor)
            {
                _context.OMSetBlendState(blendState, pBlendFactor, 0xffffffff);
            }
        }

        // 释放并重新创建渲染目标资源
        private void RecreateRenderTargets()
        {
            // 释放现有资源
            _renderTargetView.Dispose();
            _renderTarget.Dispose();
            _resolveTexture.Dispose();

            // 保持staging纹理不变，因为它总是非多重采样的

            // 创建新的渲染目标
            CreateRenderTargets();
        }

        private void UpdateConstantBuffer()
        {
            ReadOnlySpan<float> constBufferData =
            [
                _width,                 _height,            0,             0,
                _transform.M11,         _transform.M12,     0,             0,
                _transform.M21,         _transform.M22,     0,             0,
                _transform.OffsetX,     _transform.OffsetY, 1,             0,
                _strokeThicknessFactor, _widthFactor,       _heightFactor, 0
            ];

            // 更新常量缓冲区中的屏幕尺寸和变换矩阵
            MappedSubresource mappedConstBuffer = default;
            _context.Map(_constantBuffer, 0, Map.WriteDiscard, 0, ref mappedConstBuffer);

            fixed (float* constBufferDataPtr = constBufferData)
            {
                Unsafe.CopyBlockUnaligned(
                    mappedConstBuffer.PData,
                    constBufferDataPtr,
                    (uint)(sizeof(float) * constBufferData.Length));
            }

            _context.Unmap(_constantBuffer, 0);
        }

        private void EnsureNotDisposed()
        {
            if (_disposedValue)
            {
                throw new InvalidOperationException("Object disposed");
            }
        }

        public void SetAntialiasing(bool enable)
        {
            EnsureNotDisposed();

            if (_antialiasingEnabled == enable)
                return; // 状态未改变，不需要重建资源

            _antialiasingEnabled = enable;

            // 重新创建渲染目标
            RecreateRenderTargets();
        }

        public void SetTransform(MatrixTransform transform)
        {
            EnsureNotDisposed();

            _transform = transform;
            UpdateConstantBuffer();
        }

        public void SetQuadsFactors(float strokeThicknessFactor, float widthFactor, float heightFactor)
        {
            EnsureNotDisposed();

            _strokeThicknessFactor = strokeThicknessFactor;
            _widthFactor = widthFactor;
            _heightFactor = heightFactor;
            UpdateConstantBuffer();
        }

        public void SetQuads(ReadOnlySpan<Quad> quads)
        {
            EnsureNotDisposed();

            if (quads.Length > _maxQuadCount)
            {
                throw new ArgumentException($"Quad count ({quads.Length}) exceeds maximum capacity ({_maxQuadCount}).");
            }

            // Map the vertex buffer
            MappedSubresource mappedResource;
            _context.Map(_vertexBuffer, 0, Map.WriteDiscard, 0, &mappedResource);

            // Copy data to the vertex buffer
            Unsafe.CopyBlockUnaligned(mappedResource.PData,
                Unsafe.AsPointer(ref MemoryMarshal.GetReference(quads)),
                (uint)(quads.Length * _quadStride));

            // Unmap the vertex buffer
            _context.Unmap(_vertexBuffer, 0);

            _quadCount = quads.Length;
        }

        public void Render(Span<byte> bgraBuffer)
        {
            EnsureNotDisposed();

            if (bgraBuffer.Length < _width * _height * 4)
            {
                throw new ArgumentException("Buffer is too small for the specified dimensions.");
            }

            // Set up render targets
            _context.OMSetRenderTargets(1, ref _renderTargetView, ref Unsafe.NullRef<ID3D11DepthStencilView>());

            // Clear the render target
            float[] clearColor = new float[] { 0.0f, 0.0f, 0.0f, 0.0f }; // Transparent black
            fixed (float* pClearColor = clearColor)
            {
                _context.ClearRenderTargetView(_renderTargetView, pClearColor);
            }

            // Set viewport
            _context.RSSetViewports(1, in _viewport);

            // Set shaders
            _context.VSSetShader(_vertexShader, ref Unsafe.NullRef<ComPtr<ID3D11ClassInstance>>(), 0);
            _context.GSSetShader(_geometryShader, ref Unsafe.NullRef<ComPtr<ID3D11ClassInstance>>(), 0);
            _context.PSSetShader(_pixelShader, ref Unsafe.NullRef<ComPtr<ID3D11ClassInstance>>(), 0);

            // Set input layout
            _context.IASetInputLayout(_inputLayout);

            // Set vertex buffer
            uint stride = (uint)_quadStride;
            uint offset = 0;
            _context.IASetVertexBuffers(0, 1, _vertexBuffer.GetAddressOf(), &stride, &offset);

            // Set primitive topology
            _context.IASetPrimitiveTopology(D3DPrimitiveTopology.D3DPrimitiveTopologyPointlist);

            // Set constant buffer
            _context.GSSetConstantBuffers(0, 1, ref _constantBuffer);

            // Draw quads
            _context.Draw((uint)_quadCount, 0);

            // 处理抗锯齿和回读操作
            if (_antialiasingEnabled)
            {
                // 解析多重采样纹理到非多重采样纹理
                _context.ResolveSubresource(_resolveTexture, 0, _renderTarget, 0, Format.FormatB8G8R8A8Unorm);

                // 从解析纹理复制到staging纹理
                _context.CopyResource(_stagingTexture, _resolveTexture);
            }
            else
            {
                // 直接从渲染目标复制到staging纹理
                _context.CopyResource(_stagingTexture, _renderTarget);
            }

            // Map staging texture to read the pixels
            MappedSubresource mappedResource;
            _context.Map(_stagingTexture, 0, Map.Read, 0, &mappedResource);

            // Copy data from staging resource to our buffer
            for (int y = 0; y < _height; y++)
            {
                IntPtr srcRow = (nint)((nint)mappedResource.PData + y * mappedResource.RowPitch);
                int dstOffset = y * _width * 4;

                // Copy row by row
                Unsafe.CopyBlockUnaligned(
                    ref bgraBuffer[dstOffset],
                    ref Unsafe.AsRef<byte>((void*)srcRow),
                    (uint)(_width * 4));
            }

            // Unmap the staging texture
            _context.Unmap(_stagingTexture, 0);
        }

        private byte[] CompileShader(string entryPoint, string profile)
        {
            string shaderCode = """
                // QuadShader.hlsl

                cbuffer ScreenBuffer : register(b0)
                {
                    float2 screenSize;  // width, height
                    float3x3 transform; // 变换矩阵
                    float3 strokeWidthFactorAndSizeFactor;
                };

                // 顶点着色器输入
                struct VS_INPUT
                {
                    float2 Position : POSITION0;    // 顶点的XY坐标
                    float2 Size : SIZE0;            // 宽高
                    float Rotation : ROTATION0;     // 旋转量
                    float StrokeWidth : STROKE0;    // 描边厚度
                    float4 StrokeColor : COLOR0;    // 描边颜色
                    float4 FillColor : COLOR1;      // 填充颜色
                };

                // 几何着色器输入，与顶点着色器输出相同
                struct GS_INPUT
                {
                    float2 Position : POSITION0;
                    float2 Size : SIZE0;
                    float Rotation : ROTATION0;
                    float StrokeWidth : STROKE0;
                    float4 StrokeColor : COLOR0;
                    float4 FillColor : COLOR1;
                };

                // 像素着色器输入，也是几何着色器输出
                struct PS_INPUT
                {
                    float4 Position : SV_POSITION;  // 裁剪空间位置
                    float4 Color : COLOR0;          // 像素颜色
                };

                float2 ScreenToClipPoint(float2 input)
                {
                    // 应用变换矩阵
                    float3 pos = float3(input, 1.0f);
                    float3 transformedPos = mul(transform, pos);
                    
                    // Convert from screen coordinates (0 to width/height) to clip space (-1 to 1)
                    float2 normalizedPos;
                    normalizedPos.x = (transformedPos.x / screenSize.x) * 2.0f - 1.0f;
                    normalizedPos.y = 1.0f - (transformedPos.y / screenSize.y) * 2.0f; // 反转Y轴，因为DirectX中Y轴向下

                    return normalizedPos;
                }

                // 顶点着色器
                GS_INPUT VS(VS_INPUT input)
                {
                    GS_INPUT output;
                    // 简单地传递数据给几何着色器
                    output.Position = input.Position;
                    output.Size = input.Size;
                    output.Rotation = input.Rotation;
                    output.StrokeWidth = input.StrokeWidth;
                    output.StrokeColor = input.StrokeColor;
                    output.FillColor = input.FillColor;
                    return output;
                }

                // 几何着色器
                [maxvertexcount(15)]
                void GS(point GS_INPUT input[1], inout TriangleStream<PS_INPUT> triStream)
                {
                    GS_INPUT i = input[0];

                    float strokeWidth = i.StrokeWidth * strokeWidthFactorAndSizeFactor.x;
                    float2 size = float2(i.Size.x * strokeWidthFactorAndSizeFactor.y, i.Size.y * strokeWidthFactorAndSizeFactor.z);
                    
                    float halfStrokeWidth = strokeWidth / 2;
                    float2 halfSize = size * 0.5f;
                    float2 outerHalfSize = halfSize + float2(halfStrokeWidth, halfStrokeWidth);
                    float2 innerHalfSize = halfSize - float2(halfStrokeWidth, halfStrokeWidth);
                    
                    // 生成旋转矩阵
                    float sinR = sin(i.Rotation);
                    float cosR = cos(i.Rotation);
                    float2x2 rotMatrix = float2x2(cosR, sinR, -sinR, cosR);
                    
                    // 外部矩形的四个顶点（顺时针）
                    float2 outerCorners[4];
                    outerCorners[0] = ScreenToClipPoint(mul(float2(-outerHalfSize.x, -outerHalfSize.y), rotMatrix) + i.Position); // 左上
                    outerCorners[1] = ScreenToClipPoint(mul(float2(outerHalfSize.x, -outerHalfSize.y), rotMatrix) + i.Position);  // 右上
                    outerCorners[2] = ScreenToClipPoint(mul(float2(outerHalfSize.x, outerHalfSize.y), rotMatrix) + i.Position);   // 右下
                    outerCorners[3] = ScreenToClipPoint(mul(float2(-outerHalfSize.x, outerHalfSize.y), rotMatrix) + i.Position);  // 左下
                    
                    // 内部矩形的四个顶点（顺时针）
                    float2 innerCorners[4];
                    innerCorners[0] = ScreenToClipPoint(mul(float2(-innerHalfSize.x, -innerHalfSize.y), rotMatrix) + i.Position); // 左上
                    innerCorners[1] = ScreenToClipPoint(mul(float2(innerHalfSize.x, -innerHalfSize.y), rotMatrix) + i.Position);  // 右上
                    innerCorners[2] = ScreenToClipPoint(mul(float2(innerHalfSize.x, innerHalfSize.y), rotMatrix) + i.Position);   // 右下
                    innerCorners[3] = ScreenToClipPoint(mul(float2(-innerHalfSize.x, innerHalfSize.y), rotMatrix) + i.Position);  // 左下
                    
                    // 准备顶点
                    PS_INPUT v[12];
                    
                    // 转换为裁剪空间坐标
                    for (int idx = 0; idx < 4; idx++)
                    {
                        // 外部矩形顶点
                        v[idx].Position = float4(outerCorners[idx], 0.0f, 1.0f);
                        v[idx].Color = i.StrokeColor;
                        
                        // 内部矩形顶点
                        v[idx + 4].Position = float4(innerCorners[idx], 0.0f, 1.0f);
                        v[idx + 4].Color = i.StrokeColor;
                        
                        // 内部填充矩形顶点（复用内部矩形顶点，但颜色不同）
                        v[idx + 8].Position = float4(innerCorners[idx], 0.0f, 1.0f);
                        v[idx + 8].Color = i.FillColor;
                    }
                    
                    // 输出描边三角形 - 8个三角形形成描边
                    // 左边描边
                    triStream.Append(v[0]);
                    triStream.Append(v[4]);
                    triStream.Append(v[1]);
                    triStream.Append(v[5]);
                    triStream.Append(v[2]);
                    triStream.Append(v[6]);
                    triStream.Append(v[3]);
                    triStream.Append(v[7]);
                    triStream.Append(v[0]);
                    triStream.Append(v[4]);
                    triStream.RestartStrip();

                    triStream.Append(v[8]);
                    triStream.Append(v[9]);
                    triStream.Append(v[10]);
                    triStream.Append(v[11]);
                    triStream.Append(v[8]);
                    triStream.RestartStrip();
                }

                // 像素着色器
                float4 PS(PS_INPUT input) : SV_Target
                {
                    return input.Color;
                }

                """;

            using var compiler = D3DCompiler.GetApi();

            var shaderCodeBytes = EncodingUtils.EncodeToAscii(shaderCode);

            ComPtr<ID3D10Blob> blobCompiledShader = default;
            ComPtr<ID3D10Blob> blobErrorMsgs = null;
            fixed (byte* pShaderCode = shaderCodeBytes)
            {
                compiler.Compile(pShaderCode, (nuint)(shaderCodeBytes.Length), "shader", ref Unsafe.NullRef<D3DShaderMacro>(), ref Unsafe.NullRef<ID3DInclude>(), entryPoint, profile, 0, 0, ref blobCompiledShader, ref blobErrorMsgs);
            }

            if (blobErrorMsgs.Handle != null)
            {
                string errorMsgs = Marshal.PtrToStringAnsi((nint)blobErrorMsgs.GetBufferPointer(), (int)blobErrorMsgs.GetBufferSize());
                throw new InvalidOperationException(errorMsgs);
            }

            var result = new Span<byte>(blobCompiledShader.GetBufferPointer(), (int)blobCompiledShader.GetBufferSize()).ToArray();
            blobCompiledShader.Dispose();

            return result;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                // Release Direct3D resources
                _renderTargetView.Dispose();
                _renderTarget.Dispose();
                _stagingTexture.Dispose();
                _vertexShader.Dispose();
                _geometryShader.Dispose();
                _pixelShader.Dispose();
                _inputLayout.Dispose();
                _vertexBuffer.Dispose();
                _constantBuffer.Dispose();

                // Release device and context
                if (_context.Handle != null)
                {
                    _context.ClearState();
                    _context.Flush();
                    _context.Dispose();
                }

                if (_device.Handle != null)
                {
                    _device.Dispose();
                }

                _disposedValue = true;
            }
        }

        ~QuadsRenderer()
        {
            // 不要更改此代码。请将清理代码放入“Dispose(bool disposing)”方法中
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            // 不要更改此代码。请将清理代码放入“Dispose(bool disposing)”方法中
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }


}
