using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibDxGeometryRendering.Internal
{
    internal static class EncodingUtils
    {
        /// <summary>
        /// 将字符串转换为ASCII编码，丢弃非ASCII字符
        /// </summary>
        /// <param name="input">输入字符串</param>
        /// <returns>ASCII编码后的字符串</returns>
        public static byte[] EncodeToAscii(string input)
        {
            if (string.IsNullOrEmpty(input))
                return Array.Empty<byte>();

            MemoryStream result = new MemoryStream();

            foreach (char c in input)
            {
                // 只保留ASCII范围内的字符（0-127）
                if (c >= 0 && c <= 127)
                {
                    result.WriteByte((byte)c);
                }
                // 丢弃ASCII范围外的字符
            }

            return result.ToArray();
        }
    }
}
