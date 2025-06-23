namespace LibDxGeometryRendering
{
    public record struct LinearTransform(float M11, float M12, float M21, float M22)
    {
        public static readonly LinearTransform Identity = new LinearTransform(1, 0, 0, 1);
    }
}
