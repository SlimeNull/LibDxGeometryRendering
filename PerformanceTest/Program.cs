using System.Diagnostics;
using System.Drawing;
using System.Numerics;
using System.Runtime.InteropServices;
using LibDxGeometryRendering;

namespace PerformanceTest
{
    internal class Program
    {
        static unsafe void Main(string[] args)
        {
            var canvasWidth = 2000;
            var canvasHeight = 2000;
            var quadCount = 1000000;

            Bitmap bitmap = new Bitmap(1000, 1000, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            PolygonRenderer quadsRenderer = new PolygonRenderer(1000, 1000, quadCount);
            bool firstRender = true;

            var frameCount = 0;
            Stopwatch stopwatch = Stopwatch.StartNew();
            List<Polygon> quads = new List<Polygon>(quadCount);
            quads.Clear();
            for (int i = 0; i < quadCount; i++)
            {
                var x = Random.Shared.Next(0, canvasWidth);
                var y = Random.Shared.Next(0, canvasHeight);
                var size = Random.Shared.Next(Math.Min(canvasWidth, canvasHeight) / 100);
                var stroke = size / 8;
                var rotation = Random.Shared.NextSingle() * 2 * MathF.PI;
                var quad = new Quad(new Vector2(x, y), new Vector2(size, size), rotation, stroke, new ColorRgba(255, 255, 0, 255), new ColorRgba(0, 0, 255, 255));

                quads.Add(quad);
            }

            var quadsSpan = CollectionsMarshal.AsSpan(quads);

            quadsRenderer.SetAntialiasing(true);
            quadsRenderer.SetQuads(quadsSpan);

            while (true)
            {

                var bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                quadsRenderer.Render(new Span<byte>((void*)bitmapData.Scan0, bitmapData.Stride * bitmapData.Height));
                bitmap.UnlockBits(bitmapData);

                if (firstRender)
                {
                    bitmap.Save("output.png");
                    firstRender = false;
                }

                frameCount++;
                if (stopwatch.ElapsedMilliseconds >= 1000)
                {
                    Console.WriteLine($"FPS: {frameCount}");
                    frameCount = 0;
                    stopwatch.Restart();
                }
            }
        }
    }
}
