using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(Camera))]
public class CameraDepthReader : MonoBehaviour
{
    [Tooltip("Assign the DepthDecode shader (Hidden/DepthDecode). " +
             "If left null, the script will try Shader.Find at runtime.")]
    public Shader depthDecodeShader;

    private Camera        _cam;
    private Material      _decodeMat;
    private RenderTexture _depthRT;   // holds LINEAR EYE-SPACE depth in metres, already decoded
    private Texture2D     _readback;
    private bool          _captured = false;

    // ── COORDINATE CONVENTION (read this before touching anything) ──────────
    // All public methods on this class take screen coordinates with
    // (0,0) = TOP-LEFT of the screen, matching image/texture space.
    // This matches yoloDepthImage pixel space directly, so YoloCombinedCollider
    // does NOT need to flip Y at all when talking to this class.
    // Internally we flip once, right where ReadPixels happens, because
    // ReadPixels/GetPixel on a Texture2D use BOTTOM-LEFT origin.
    // --------------------------------------------------------------------

    void OnEnable()
    {
        _cam = GetComponent<Camera>();
        _cam.depthTextureMode = DepthTextureMode.Depth;

        if (depthDecodeShader == null)
            depthDecodeShader = Shader.Find("Hidden/DepthDecode");

        if (depthDecodeShader == null)
        {
            Debug.LogError("[CameraDepthReader] DepthDecode shader not found. " +
                           "Make sure DepthDecode.shader is in the project and assigned.");
            return;
        }

        _decodeMat = new Material(depthDecodeShader) { hideFlags = HideFlags.HideAndDontSave };

        RebuildTextures();
        RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
    }

    void OnDisable()
    {
        RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
        ReleaseResources();
    }

    void OnDestroy()
    {
        RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
        ReleaseResources();
        if (_decodeMat != null) DestroyImmediate(_decodeMat);
    }

    void OnEndCameraRendering(ScriptableRenderContext ctx, Camera cam)
    {
        if (cam != _cam || _decodeMat == null) return;

        if (_depthRT == null ||
            _depthRT.width  != _cam.pixelWidth ||
            _depthRT.height != _cam.pixelHeight)
            RebuildTextures();

        // ── KEY FIX #1 ────────────────────────────────────────────────────
        // Use Graphics.Blit with our decode SHADER (not a raw format blit).
        // This runs DepthDecode.shader per-pixel, which calls LinearEyeDepth()
        // internally — that macro already knows about reversed-Z and platform
        // depth conventions, so we don't hand-roll that math anymore.
        Graphics.Blit(null, _depthRT, _decodeMat);

        var prev = RenderTexture.active;
        RenderTexture.active = _depthRT;
        _readback.ReadPixels(new Rect(0, 0, _depthRT.width, _depthRT.height), 0, 0, false);
        _readback.Apply(false);
        RenderTexture.active = prev;

        _captured = true;
    }

    /// <summary>
    /// Returns metric depth (metres) at the given screen pixel.
    /// (screenX, screenY) use TOP-LEFT origin — i.e. screenY=0 is the TOP of the screen,
    /// same convention as image/texture pixel space. Returns -1 if no geometry (sky / far plane).
    /// </summary>
    public float GetMetricDepth(int screenX, int screenY)
    {
        if (!EnsureReady()) return -1f;

        screenX = Mathf.Clamp(screenX, 0, _readback.width  - 1);
        screenY = Mathf.Clamp(screenY, 0, _readback.height - 1);

        // ── KEY FIX #2 ────────────────────────────────────────────────────
        // Texture2D.GetPixel uses BOTTOM-LEFT origin. Our public contract is
        // TOP-LEFT origin (to match image space). Flip exactly ONCE, here,
        // and nowhere else in the codebase.
        int flippedY = (_readback.height - 1) - screenY;

        float raw = _readback.GetPixel(screenX, flippedY).r;
        return raw <= 0f ? -1f : raw; // already linear metres thanks to the shader
    }

    /// <summary>
    /// Samples a region and returns the median depth (metres) — more robust than one pixel.
    /// Same TOP-LEFT screen convention as GetMetricDepth.
    /// </summary>
    public float GetMetricDepthAveraged(int screenX, int screenY, int radius = 8)
    {
        if (!EnsureReady()) return -1f;

        int w = _readback.width;
        int h = _readback.height;
        var samples = new System.Collections.Generic.List<float>();

        // Convert once to bottom-left (internal) space, then work entirely in that space
        // for the sampling loop to avoid flipping per-sample.
        int centerYFlipped = (h - 1) - Mathf.Clamp(screenY, 0, h - 1);
        int centerX         = Mathf.Clamp(screenX, 0, w - 1);

        for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
        {
            int sx = Mathf.Clamp(centerX + dx, 0, w - 1);
            int sy = Mathf.Clamp(centerYFlipped + dy, 0, h - 1);

            float raw = _readback.GetPixel(sx, sy).r;

            // Decoded by the shader already: 0 means "no geometry / sky / far plane".
            if (raw <= 0f) continue;

            samples.Add(raw); // already linear metres
        }

        if (samples.Count == 0)
        {
            Debug.LogWarning("[CameraDepthReader] No valid depth samples in region. " +
                             $"Centre pixel ({screenX},{screenY}) [top-left space].");
            return -1f;
        }

        samples.Sort();
        return samples[samples.Count / 2]; // median
    }

    bool EnsureReady()
    {
        if (!_captured || _readback == null)
        {
            Debug.LogWarning("[CameraDepthReader] Depth not captured yet.");
            return false;
        }
        return true;
    }

    void RebuildTextures()
    {
        ReleaseResources();
        int w = Mathf.Max(_cam.pixelWidth,  1);
        int h = Mathf.Max(_cam.pixelHeight, 1);

        _depthRT = new RenderTexture(w, h, 0, RenderTextureFormat.RFloat)
        {
            name       = "HDRPDepthRT_Decoded",
            hideFlags  = HideFlags.HideAndDontSave,
            filterMode = FilterMode.Point,
            wrapMode   = TextureWrapMode.Clamp
        };
        _depthRT.Create();

        _readback = new Texture2D(w, h, TextureFormat.RFloat, false)
        {
            name       = "HDRPDepthReadback",
            hideFlags  = HideFlags.HideAndDontSave,
            filterMode = FilterMode.Point
        };
        _captured = false;
    }

    void ReleaseResources()
    {
        if (_depthRT  != null) { _depthRT.Release(); DestroyImmediate(_depthRT);  _depthRT  = null; }
        if (_readback != null) {                     DestroyImmediate(_readback); _readback = null; }
        _captured = false;
    }
}