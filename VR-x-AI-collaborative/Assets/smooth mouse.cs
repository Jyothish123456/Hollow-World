using UnityEngine;
using System.Collections;

public class HiResScreenShots : MonoBehaviour
{
    public int resWidth = 2550;
    public int resHeight = 3300;

    private bool takeHiResShot = false;
    private Camera cam; // ✅ FIX: store camera reference

    void Start()
    {
        cam = GetComponent<Camera>(); // ✅ FIX: get camera properly
    }

    public static string ScreenShotName(int width, int height)
    {
        return string.Format("{0}/screenshots/screen_{1}x{2}_{3}.png",
            Application.dataPath,
            width, height,
            System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
    }

    public void TakeHiResShot()
    {
        takeHiResShot = true;
    }

    void LateUpdate()
    {
        takeHiResShot |= Input.GetKeyDown(KeyCode.K); // ✅ FIX: modern input

        if (takeHiResShot)
        {
            RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);

            cam.targetTexture = rt;   // ✅ FIX: use cam instead of camera
            Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);

            cam.Render();             // ✅ FIX
            RenderTexture.active = rt;

            screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);

            cam.targetTexture = null; // ✅ FIX
            RenderTexture.active = null;

            Destroy(rt);

            byte[] bytes = screenShot.EncodeToPNG();
            string filename = ScreenShotName(resWidth, resHeight);

            System.IO.File.WriteAllBytes(filename, bytes);
            Debug.Log("Took screenshot to: " + filename);

            takeHiResShot = false;
        }
    }
}