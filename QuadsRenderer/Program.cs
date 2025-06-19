using LibQuadsRenderer;
using System.Numerics;

namespace QuadsRendererTest
{
    internal class Program
    {
        static unsafe void Main(string[] args)
        {
            Bitmap bitmap = new Bitmap(1000, 1000, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            QuadsRenderer quadsRenderer = new QuadsRenderer(1000, 1000, 10);

            var rotation15 = MathF.PI / 12;
            var testQuad = new Quad(new Vector2(500, 500), new Vector2(100f, 100f), rotation15, 5f, new ColorRgba(255, 255, 0, 255), new ColorRgba(0, 0, 255, 255));
            quadsRenderer.SetAntialiasing(true);
            quadsRenderer.SetQuads([
                testQuad
            ]);

            var bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            quadsRenderer.Render(new Span<byte>((void*)bitmapData.Scan0, bitmapData.Stride * bitmapData.Height));
            bitmap.UnlockBits(bitmapData);

            bitmap.Save("output.png");
        }
    }
}
