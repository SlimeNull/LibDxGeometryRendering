
using System;
using System.IO;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using LibQuadsRenderer.Internal;
using Silk.NET.Core.Native;
using Silk.NET.Direct3D;
using Silk.NET.Direct3D.Compilers;
using Silk.NET.Direct3D11;
using Silk.NET.DXGI;
using Silk.NET.Maths;


namespace LibQuadsRenderer
{
    public unsafe class QuadsRenderer : IDisposable
    {
        private readonly int _width;
        private readonly int _height;
        private readonly int _maxQuadCount;

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
        private ComPtr<ID3D11Buffer> _vertexBuffer;

        // Viewport
        private readonly Viewport _viewport;

        // Vertex layout
        private readonly int _quadStride = Unsafe.SizeOf<Quad>();

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
        }

        private void CreateDeviceContext(out ComPtr<ID3D11Device> device, out ComPtr<ID3D11DeviceContext> context)
        {
            Guid deviceGuid = new Guid("db6f6ddb-ac77-4e88-8253-819df9bbf140"); // IID_ID3D11Device
            Guid contextGuid = new Guid("c0bfa96c-e089-44fb-8eaf-26f8796190da"); // IID_ID3D11DeviceContext

            using var d3d11 = D3D11.GetApi(null, false);

            // Create D3D11 device and context
            D3DFeatureLevel[] featureLevels = { D3DFeatureLevel.Level111, D3DFeatureLevel.Level110, D3DFeatureLevel.Level100 };

            // Create with debug layer in debug builds
#if DEBUG
            uint flags = (uint)CreateDeviceFlag.Debug;
#else
        uint flags = 0;
#endif

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
            // Create the render target texture
            Texture2DDesc rtDesc = new Texture2DDesc
            {
                Width = (uint)_width,
                Height = (uint)_height,
                MipLevels = 1,
                ArraySize = 1,
                Format = Format.FormatB8G8R8A8Unorm,
                SampleDesc = new SampleDesc { Count = 1, Quality = 0 },
                Usage = Usage.Default,
                BindFlags = (uint)BindFlag.RenderTarget | (uint)BindFlag.ShaderResource,
                CPUAccessFlags = 0,
                MiscFlags = 0
            };

            // Create render target texture
            _device.CreateTexture2D(in rtDesc, null, ref _renderTarget);

            // Create render target view
            RenderTargetViewDesc rtvDesc = new RenderTargetViewDesc
            {
                Format = Format.FormatB8G8R8A8Unorm,
                ViewDimension = RtvDimension.Texture2D,
                // No union fields needed for 2D
            };

            _device.CreateRenderTargetView(_renderTarget, in rtvDesc, ref _renderTargetView);

            // Create staging texture for readback
            Texture2DDesc stagingDesc = rtDesc;
            stagingDesc.Usage = Usage.Staging;
            stagingDesc.BindFlags = 0;
            stagingDesc.CPUAccessFlags = (uint)CpuAccessFlag.Read;

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

        public void SetQuads(ReadOnlySpan<Quad> quads)
        {
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
        }

        public void Render(Span<byte> bgraBuffer)
        {
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

            // Draw quads
            _context.Draw((uint)_maxQuadCount, 0);

            // Copy from render target to staging texture for CPU access
            _context.CopyResource(_stagingTexture, _renderTarget);

            // Map staging texture to read the pixels
            MappedSubresource mappedResource;
            _context.Map(_stagingTexture, 0, Map.Read, 0, &mappedResource);

            // Copy data from staging resource to our buffer
            for (int y = 0; y < _height; y++)
            {
                IntPtr srcRow = (nint)((IntPtr)mappedResource.PData + y * mappedResource.RowPitch);
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
                [maxvertexcount(30)]
                void GS(point GS_INPUT input[1], inout TriangleStream<PS_INPUT> triStream)
                {
                    GS_INPUT i = input[0];
                    
                    float halfStrokeWidth = i.StrokeWidth / 2;
                    float2 halfSize = i.Size * 0.5f;
                    float2 outerHalfSize = halfSize + float2(halfStrokeWidth, halfStrokeWidth);
                    float2 innerHalfSize = halfSize - float2(halfStrokeWidth, halfStrokeWidth);
                    
                    // 生成旋转矩阵
                    float sinR = sin(i.Rotation);
                    float cosR = cos(i.Rotation);
                    float2x2 rotMatrix = float2x2(cosR, -sinR, sinR, cosR);
                    
                    // 外部矩形的四个顶点（顺时针）
                    float2 outerCorners[4];
                    outerCorners[0] = mul(float2(-outerHalfSize.x, -outerHalfSize.y), rotMatrix) + i.Position; // 左上
                    outerCorners[1] = mul(float2(outerHalfSize.x, -outerHalfSize.y), rotMatrix) + i.Position;  // 右上
                    outerCorners[2] = mul(float2(outerHalfSize.x, outerHalfSize.y), rotMatrix) + i.Position;   // 右下
                    outerCorners[3] = mul(float2(-outerHalfSize.x, outerHalfSize.y), rotMatrix) + i.Position;  // 左下
                    
                    // 内部矩形的四个顶点（顺时针）
                    float2 innerCorners[4];
                    innerCorners[0] = mul(float2(-innerHalfSize.x, -innerHalfSize.y), rotMatrix) + i.Position; // 左上
                    innerCorners[1] = mul(float2(innerHalfSize.x, -innerHalfSize.y), rotMatrix) + i.Position;  // 右上
                    innerCorners[2] = mul(float2(innerHalfSize.x, innerHalfSize.y), rotMatrix) + i.Position;   // 右下
                    innerCorners[3] = mul(float2(-innerHalfSize.x, innerHalfSize.y), rotMatrix) + i.Position;  // 左下
                    
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
                    triStream.Append(v[3]);
                    
                    triStream.Append(v[3]);
                    triStream.Append(v[4]);
                    triStream.Append(v[7]);
                    triStream.RestartStrip();
                    
                    // 上边描边
                    triStream.Append(v[0]);
                    triStream.Append(v[1]);
                    triStream.Append(v[4]);
                    
                    triStream.Append(v[4]);
                    triStream.Append(v[1]);
                    triStream.Append(v[5]);
                    triStream.RestartStrip();
                    
                    // 右边描边
                    triStream.Append(v[1]);
                    triStream.Append(v[2]);
                    triStream.Append(v[5]);
                    
                    triStream.Append(v[5]);
                    triStream.Append(v[2]);
                    triStream.Append(v[6]);
                    triStream.RestartStrip();
                    
                    // 下边描边
                    triStream.Append(v[3]);
                    triStream.Append(v[7]);
                    triStream.Append(v[2]);
                    
                    triStream.Append(v[2]);
                    triStream.Append(v[7]);
                    triStream.Append(v[6]);
                    triStream.RestartStrip();
                    
                    // 输出填充三角形 - 2个三角形形成填充区域
                    triStream.Append(v[8]);  // 左上
                    triStream.Append(v[9]);  // 右上
                    triStream.Append(v[11]); // 左下
                    
                    triStream.Append(v[9]);  // 右上
                    triStream.Append(v[10]); // 右下
                    triStream.Append(v[11]); // 左下
                }

                // 像素着色器
                float4 PS(PS_INPUT input) : SV_Target
                {
                    return input.Color;
                }

                """;

            using var compiler = D3DCompiler.GetApi();

            var shaderCodeBytes = EncodingUtils.EncodeToAscii(shaderCode);

            ComPtr<ID3D10Blob> compiledShader = default;
            ComPtr<ID3D10Blob> errorMsgs = null;
            fixed (byte* pShaderCode = shaderCodeBytes)
            {
                compiler.Compile(pShaderCode, (nuint)(shaderCodeBytes.Length), "shader", ref Unsafe.NullRef<D3DShaderMacro>(), ref Unsafe.NullRef<ID3DInclude>(), entryPoint, profile, 0, 0, ref compiledShader, ref errorMsgs);
            }

            var result = new Span<byte>(compiledShader.GetBufferPointer(), (int)compiledShader.GetBufferSize()).ToArray();
            compiledShader.Dispose();

            return result;
        }

        public void Dispose()
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
        }
    }


}
