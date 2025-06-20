using System.Runtime.InteropServices;

namespace LibDxGeometryRendering
{
    [StructLayout(LayoutKind.Sequential)]
    public record struct ColorRgba(byte R, byte G, byte B, byte A);
}
