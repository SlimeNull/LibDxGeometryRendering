﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibDxGeometryRendering
{
    public record struct AffineTransform(float M11, float M12, float M21, float M22, float OffsetX, float OffsetY)
    {
        public static readonly AffineTransform Identity = new AffineTransform(1, 0, 0, 1, 0, 0);
    }
}
