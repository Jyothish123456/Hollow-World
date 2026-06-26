// SAVE THIS FILE AS EXACTLY: YoloAnchoredCollider.cs
//
// WHY THIS EXISTS:
// The previous approach (YoloCombinedCollider + CameraDepthReader + DepthDecode.shader)
// tried to MEASURE depth from the GPU depth buffer. Any error in that one sampled
// value got multiplied through the entire frustum-size formula, which is why the
// collider ended up huge, rotated, and floating away from the actual object.
//
// This version does not measure depth at all. You manually place an empty
// GameObject ("anchor") at the real center of the target object — something you
// can verify with your own eyes in the Scene view — and the script just measures
// the distance from the camera to THAT point. The YOLO image is only used for its
// pixel bounding-box RATIO (width/height), which is flip-invariant, so none of the
// top-left/bottom-left coordinate juggling from before is needed anymore.
//
// SETUP:
// 1. GameObject > Create Empty, name it e.g. "BottleAnchor".
// 2. Drag it (Move tool) to sit at the real center of the bottle in the Scene view.
// 3. Add a BoxCollider + this script to that SAME empty (simplest), or to a
//    separate object and assign the empty as "Anchor Empty" below.
// 4. Assign yoloDepthImage (the green-box preview PNG) and sourceCamera.
// 5. Press Play, or right-click the component > "Regenerate Collider" in Edit mode.

using UnityEngine;

[RequireComponent(typeof(BoxCollider))]
public class YoloAnchoredCollider : MonoBehaviour
{
    [Header("Data Inputs")]
    public Texture2D yoloDepthImage;
    public Camera sourceCamera;

    [Tooltip("Empty GameObject placed by hand at the real-world center of the target object. " +
             "This replaces the GPU depth reader as the source of truth for distance/position.")]
    public Transform anchorEmpty;

    [Header("Detection Parameters")]
    [Tooltip("Minimum green channel value (with low red/blue) to count as part of the YOLO box outline/fill.")]
    [Range(0.01f, 1f)]
    public float greenThreshold = 0.05f;

    [Tooltip("Scales the computed width/height. 1.0 = exact pixel-ratio fit. <1 shrinks, >1 grows.")]
    [Range(0.1f, 2f)]
    public float tightFitScale = 1.0f;

    [Header("Fine Tuning")]
    [Tooltip("Depth (Z size, front-to-back) as a fraction of the computed width. " +
             "YOLO boxes only give you W/H, not depth, so this is an estimate.")]
    public float symmetryRatio = 0.2f;

    [Tooltip("If true, the collider's rotation matches the camera (so its local X/Y/Z line up " +
             "with the camera's right/up/forward — required for the frustum-based size math to be correct " +
             "when the camera is tilted). If false, the collider stays world-axis-aligned.")]
    public bool alignToCamera = true;

    [Header("Debug (Read Only)")]
    [SerializeField] private string _lastStatus = "Not Run";
    [SerializeField] private float _detectedDepth;
    [SerializeField] private Vector3 _detectedSize;
    [SerializeField] private Vector3 _worldCenter;

    private BoxCollider _col;

    void Start()
    {
        GenerateCollider();
    }

    [ContextMenu("Regenerate Collider")]
    public void GenerateCollider()
    {
        if (_col == null) _col = GetComponent<BoxCollider>();
        if (sourceCamera == null) sourceCamera = Camera.main;

        // ── Validate inputs ────────────────────────────────────────────────
        if (anchorEmpty == null)
        {
            _lastStatus = "ERROR: anchorEmpty not assigned.";
            Debug.LogError("[YoloAnchoredCollider] Assign an empty GameObject placed at the " +
                           "real-world center of the target object.");
            return;
        }

        if (sourceCamera == null)
        {
            _lastStatus = "ERROR: No camera found.";
            Debug.LogError("[YoloAnchoredCollider] No camera assigned and no Camera.main found.");
            return;
        }

        if (yoloDepthImage == null)
        {
            _lastStatus = "ERROR: yoloDepthImage not assigned.";
            Debug.LogError("[YoloAnchoredCollider] Assign a Texture2D to yoloDepthImage.");
            return;
        }

        if (!yoloDepthImage.isReadable)
        {
            _lastStatus = "ERROR: Texture not readable! Enable Read/Write in Import Settings.";
            Debug.LogError("[YoloAnchoredCollider] Texture2D is not readable. " +
                           "Select it in Project → Inspector → enable 'Read/Write'.");
            return;
        }

        // ── 1. Find the green bounding box in pixel space ──────────────────
        // Only the WIDTH and HEIGHT of this box matter (maxX-minX, maxY-minY).
        // Those differences are identical whether the array is top-left or
        // bottom-left origin, so there's no flipping to get wrong here.
        if (!FindGreenBoundingBox(out int minX, out int maxX, out int minY, out int maxY,
                                   out int imgW, out int imgH))
        {
            _lastStatus = $"ERROR: No green pixels found (threshold={greenThreshold:F2}). " +
                          "Lower Green Threshold.";
            Debug.LogWarning("[YoloAnchoredCollider] " + _lastStatus);
            return;
        }

        // ── 2. Depth = distance from camera to the anchor, along the camera's ──
        // own forward axis (eye-space depth, in metres). This is the exact
        // quantity the frustum formula below expects — no GPU sampling needed.
        Vector3 localPos = sourceCamera.transform.InverseTransformPoint(anchorEmpty.position);
        float zDepth = localPos.z;

        if (zDepth <= 0f)
        {
            _lastStatus = "ERROR: anchorEmpty is behind the camera (or exactly on it).";
            Debug.LogError("[YoloAnchoredCollider] " + _lastStatus);
            return;
        }

        // ── 3. Frustum → world dimensions at that known depth ───────────────
        float vFovRad = sourceCamera.fieldOfView * 0.5f * Mathf.Deg2Rad;
        float frustumHeightAtZ = 2f * zDepth * Mathf.Tan(vFovRad);
        float frustumWidthAtZ = frustumHeightAtZ * sourceCamera.aspect;

        float worldW = ((float)(maxX - minX) / imgW) * frustumWidthAtZ * tightFitScale;
        float worldH = ((float)(maxY - minY) / imgH) * frustumHeightAtZ * tightFitScale;
        float worldD = worldW * symmetryRatio;

        // ── 4. Position & orient directly at the anchor ─────────────────────
        // No reprojection (ViewportToWorldPoint) needed — the anchor IS the
        // ground-truth center, placed by hand.
        transform.position = anchorEmpty.position;
        transform.rotation = alignToCamera ? sourceCamera.transform.rotation : Quaternion.identity;

        // ── 5. Collider ───────────────────────────────────────────────────
        _col.center = Vector3.zero;
        _col.size = new Vector3(worldW, worldH, worldD);

        // ── 6. Debug ─────────────────────────────────────────────────────
        _detectedDepth = zDepth;
        _detectedSize = _col.size;
        _worldCenter = transform.position;
        _lastStatus = $"OK | Depth:{zDepth:F2}m | W:{worldW:F2} H:{worldH:F2} D:{worldD:F2}";

        Debug.Log($"[YoloAnchoredCollider] {_lastStatus} | Anchor:{anchorEmpty.position}");
    }

    private bool FindGreenBoundingBox(out int minX, out int maxX, out int minY, out int maxY,
                                       out int imgW, out int imgH)
    {
        imgW = yoloDepthImage.width;
        imgH = yoloDepthImage.height;
        Color[] px = yoloDepthImage.GetPixels();

        minX = imgW; maxX = 0; minY = imgH; maxY = 0;
        bool found = false;

        for (int y = 0; y < imgH; y++)
        for (int x = 0; x < imgW; x++)
        {
            Color c = px[y * imgW + x];
            if (c.g > greenThreshold && c.r < 0.5f && c.b < 0.5f)
            {
                if (x < minX) minX = x; if (x > maxX) maxX = x;
                if (y < minY) minY = y; if (y > maxY) maxY = y;
                found = true;
            }
        }

        return found;
    }

    void OnDrawGizmosSelected()
    {
        if (anchorEmpty == null) return;

        Gizmos.color = Color.yellow;
        Gizmos.DrawSphere(anchorEmpty.position, 0.02f);

        if (sourceCamera != null)
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawLine(sourceCamera.transform.position, anchorEmpty.position);
        }
    }
}