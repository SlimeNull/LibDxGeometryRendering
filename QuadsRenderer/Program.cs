using LibDxGeometryRendering;
using System.Numerics;

namespace QuadsRendererTest
{
    internal class Program
    {
        static unsafe void Main(string[] args)
        {
            Bitmap bitmap = new Bitmap(1000, 1000, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            PolygonsRenderer quadsRenderer = new PolygonsRenderer(1000, 1000, 4, 10);

            var rotation15 = (float)(Math.PI / 12);
            var testQuad = new Polygon(new Vector2(500, 500), 100, LinearTransform.Identity, 10f, new ColorRgba(255, 255, 0, 255), new ColorRgba(0, 0, 255, 255));
            quadsRenderer.SetAntialiasing(true);
            quadsRenderer.SetPolygons([
                testQuad
            ]);

            quadsRenderer.SetQuadsFactors(2, 1, 1);

            var bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            quadsRenderer.Render(new Span<byte>((void*)bitmapData.Scan0, bitmapData.Stride * bitmapData.Height));
            bitmap.UnlockBits(bitmapData);
            bitmap.Save("output.png");
        }
    }
}
