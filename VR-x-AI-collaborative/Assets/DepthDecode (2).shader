// DepthDecode.shader
// Decodes the camera's hardware depth buffer into LINEAR EYE-SPACE DEPTH (metres),
// written into the RED channel of an RFloat target.
//
// WHY THIS EXISTS:
// Graphics.Blit(_CameraDepthTexture, someRFloatRT) does a raw format copy.
// It does NOT understand reversed-Z, does NOT understand platform-specific
// depth encoding, and will silently produce garbage on HDRP/DX12/Vulkan.
// This shader uses Unity's own LinearEyeDepth() macro, which already knows
// the correct decode for whatever platform/Z-convention you're running on.
// That removes the need to hand-roll reversed-Z math in C# (which is where
// the second bug in the original script came from).

Shader "Hidden/DepthDecode"
{
    SubShader
    {
        Tags { "RenderPipeline" = "HDRenderPipeline" }
        Pass
        {
            ZTest Always
            ZWrite Off
            Cull Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #include "UnityCG.cginc"

            // Plain (non-XR-aware) depth texture + sampler. We don't need
            // single-pass-instanced XR support for this project, so we avoid
            // TEXTURE2D_X / LOAD_TEXTURE2D_X_LOD, which require extra includes
            // that vary between render-pipeline package versions.
            Texture2D _CameraDepthTexture;
            SamplerState sampler_CameraDepthTexture;

            struct Attributes
            {
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv         : TEXCOORD0;
            };

            // Manual fullscreen triangle - avoids needing HDRP's
            // GetFullScreenTriangleVertexPosition/TexCoord helpers, which live
            // in includes that caused the original error.
            Varyings Vert(Attributes input)
            {
                Varyings output;

                float2 uv = float2((input.vertexID << 1) & 2, input.vertexID & 2);
                output.uv = uv;

                float2 pos = uv * 2.0 - 1.0;
#if UNITY_UV_STARTS_AT_TOP
                pos.y = -pos.y;
#endif
                output.positionCS = float4(pos, 0.0, 1.0);
                return output;
            }

            float Frag(Varyings input) : SV_Target
            {
                // Raw hardware depth at this pixel (0..1, reversed-Z on most modern platforms)
                float rawDepth = _CameraDepthTexture.Sample(sampler_CameraDepthTexture, input.uv).r;

                // LinearEyeDepth (from UnityCG.cginc) correctly handles reversed-Z /
                // platform differences for you. Result is distance from camera
                // along view direction, in WORLD UNITS (metres).
                // NOTE: UnityCG.cginc's LinearEyeDepth takes ONE argument only —
                // it reads _ZBufferParams internally as a global. (HDRP's
                // Common.hlsl has a different 2-argument overload; that's not
                // what we're including here, so don't pass _ZBufferParams.)
                float eyeDepth = LinearEyeDepth(rawDepth);

                // Encode "no geometry" (far plane / sky) as exactly 0 so C# can test for it cleanly.
                if (rawDepth <= 0.0)
                    return 0.0;

                return eyeDepth;
            }
            ENDHLSL
        }
    }
    Fallback Off
}
