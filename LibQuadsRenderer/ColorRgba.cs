using System.Runtime.InteropServices;

namespace LibQuadsRenderer
{
    [StructLayout(LayoutKind.Sequential)]
    public record struct ColorRgba(byte R, byte G, byte B, byte A);
}
