﻿using System.Numerics;
using System.Runtime.InteropServices;

namespace LibDxGeometryRendering
{
    [StructLayout(LayoutKind.Sequential)]
    public record struct Quad(Vector2 Position, Vector2 Size, float Rotation, float StrokeWidth, ColorRgba StrokeColor, ColorRgba FillColor);
}
