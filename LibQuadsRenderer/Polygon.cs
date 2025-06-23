using System.Numerics;
using System.Runtime.InteropServices;

namespace LibDxGeometryRendering
{
    [StructLayout(LayoutKind.Sequential)]
    public record struct Polygon(Vector2 Position, float Radius, LinearTransform Transform, float StrokeWidth, ColorRgba StrokeColor, ColorRgba FillColor);
}
