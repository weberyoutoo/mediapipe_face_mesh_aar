package com.example.mediapipefacemeshapp;

import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;
import android.graphics.SurfaceTexture;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.util.Size;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;

import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.AndroidPacketCreator;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.glutil.EglManager;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import android.widget.TextView;

/**
 * Main activity of MediaPipe example apps.
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final String BINARY_GRAPH_NAME = "face_mesh_mobile_gpu.binarypb";
    private static final String INPUT_VIDEO_STREAM_NAME = "input_video";
    private static final String OUTPUT_VIDEO_STREAM_NAME = "output_video";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME = "multi_face_landmarks";
    private static final String INPUT_NUM_FACES_SIDE_PACKET_NAME = "num_faces";
    private static final int NUM_FACES = 1;
    private static final CameraHelper.CameraFacing CAMERA_FACING = CameraHelper.CameraFacing.FRONT;
    // Flips the camera-preview frames vertically before sending them into FrameProcessor to be
    // processed in a MediaPipe graph, and flips the processed frames back when they are displayed.
    // This is needed because OpenGL represents images assuming the image origin is at the bottom-left
    // corner, whereas MediaPipe in general assumes the image origin is at top-left.
    private static final boolean FLIP_FRAMES_VERTICALLY = true;

    private float ay1, ay2, ry1, ry2, ly1, ly2, rightEyeRatio, leftEyeRatio;
    private float lx, ax, rx, headPoseRatio_x,ty,by, headPoseRatio_y;
    private TextView rightEyetv;
    private TextView leftEyetv;

    private TextView headPoseXtv;
    private TextView headPoseYtv;

    private boolean rightEye_blinked, rightEye_open, leftEye_blinked, leftEye_open;
    private boolean headP_right, headP_left, headPx_center, headP_up, headP_down, headPy_center;

    static {
        // Load all native libraries needed by the app.
        System.loadLibrary("mediapipe_jni");
        System.loadLibrary("opencv_java3");
    }

    // {@link SurfaceTexture} where the camera-preview frames can be accessed.
    private SurfaceTexture previewFrameTexture;
    // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
    private SurfaceView previewDisplayView;
    // Creates and manages an {@link EGLContext}.
    private EglManager eglManager;
    // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
    // frames onto a {@link Surface}.
    private FrameProcessor processor;
    // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
    // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
    private ExternalTextureConverter converter;
    // ApplicationInfo for retrieving metadata defined in the manifest.
    private ApplicationInfo applicationInfo;
    // Handles camera access via the {@link CameraX} Jetpack support library.
    private CameraXPreviewHelper cameraHelper;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(getContentViewLayoutResId());

        try {
            applicationInfo =
                    getPackageManager().getApplicationInfo(getPackageName(), PackageManager.GET_META_DATA);
        } catch (NameNotFoundException e) {
            Log.e(TAG, "Cannot find application info: " + e);
        }

        previewDisplayView = new SurfaceView(this);
        setupPreviewDisplayView();

        // Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
        // binary graphs.
        AndroidAssetUtil.initializeNativeAssetManager(this);
        eglManager = new EglManager(null);
        processor =
                new FrameProcessor(
                        this,
                        eglManager.getNativeContext(),
                        BINARY_GRAPH_NAME,
                        INPUT_VIDEO_STREAM_NAME,
                        OUTPUT_VIDEO_STREAM_NAME);
        processor
                .getVideoSurfaceOutput()
                .setFlipY(FLIP_FRAMES_VERTICALLY);

        PermissionHelper.checkAndRequestCameraPermissions(this);
        AndroidPacketCreator packetCreator = processor.getPacketCreator();
        Map<String, Packet> inputSidePackets = new HashMap<>();
        inputSidePackets.put(INPUT_NUM_FACES_SIDE_PACKET_NAME, packetCreator.createInt32(NUM_FACES));
        processor.setInputSidePackets(inputSidePackets);

        rightEyetv = findViewById(R.id.rightEyetv);
        leftEyetv = findViewById(R.id.leftEyetv);

        headPoseXtv = findViewById(R.id.headPoseXtv);
        headPoseYtv = findViewById(R.id.headPoseYtv);

        rightEye_blinked = true;
        rightEye_open = true;
        leftEye_blinked = true;
        leftEye_open = true;

        headP_right = true;
        headP_left = true;
        headPx_center = true;
        headP_up = true;
        headP_down = true;
        headPy_center = true;

        // To show verbose logging, run:
        // adb shell setprop log.tag.MainActivity VERBOSE
        if (true) {
            processor.addPacketCallback(
                    OUTPUT_LANDMARKS_STREAM_NAME,
                    (packet) -> {
                        Log.v(TAG, "Received multi face landmarks packet.");
                        List<NormalizedLandmarkList> multiFaceLandmarks =
                                PacketGetter.getProtoVector(packet, NormalizedLandmarkList.parser());
                        
                        ay1 = multiFaceLandmarks.get(0).getLandmarkList().get(4).getY()*1920f;
                        ay2 = multiFaceLandmarks.get(0).getLandmarkList().get(5).getY()*1920f;

                        ry1 = multiFaceLandmarks.get(0).getLandmarkList().get(374).getY()*1920f;
                        ry2 = multiFaceLandmarks.get(0).getLandmarkList().get(386).getY()*1920f;

                        ly1 = multiFaceLandmarks.get(0).getLandmarkList().get(145).getY()*1920f;
                        ly2 = multiFaceLandmarks.get(0).getLandmarkList().get(159).getY()*1920f;

                        lx = multiFaceLandmarks.get(0).getLandmarkList().get(36).getX()*1200f;
                        ax = multiFaceLandmarks.get(0).getLandmarkList().get(4).getX()*1200f;
          rx = multiFaceLandmarks.get(0).getLandmarkList().get(266).getX()*1200f;

          ty = multiFaceLandmarks.get(0).getLandmarkList().get(168).getY()*1920f;
          by = multiFaceLandmarks.get(0).getLandmarkList().get(164).getY()*1920f;

          rightEyeRatio = (ry1 - ry2) / (ay1 - ay2);
          leftEyeRatio = (ly1 - ly2) / (ay1 - ay2);

          headPoseRatio_x = (lx - ax) / (ax - rx);
          headPoseRatio_y = (ay2 - ty) / (by - ay2);

          //headPoseYtv.setText(headPoseRatio_y + "");

          if(rightEyeRatio < 0.7){
            if(rightEye_blinked){
                //active:右眼閉眼
              rightEyetv.setText("Eye blinked");
              rightEye_blinked = false;
              rightEye_open = true;
            }
          }
          else{
            if(rightEye_open){
                //active:右眼睜眼
              rightEyetv.setText("Eye is open");
              rightEye_blinked = true;
              rightEye_open = false;
            }
          }

          if(leftEyeRatio < 0.7){
            if(leftEye_blinked){
                //active:左眼閉眼
              leftEyetv.setText("Eye blinked");
              leftEye_blinked = false;
              leftEye_open = true;
            }
          }
          else{
            if(leftEye_open){
                //active:左眼睜眼
              leftEyetv.setText("Eye is open");
              leftEye_blinked = true;
              leftEye_open = false;
            }
          }

          if(headPoseRatio_x > 1.8){
            if(headP_right){
                //active:頭部往右
              headPoseXtv.setText("Right");
              headP_right = false;
              headP_left = true;
              headPx_center = true;
            }
          }
          else if(headPoseRatio_x < 0.75){
            if(headP_left){
                //active:頭部往左
              headPoseXtv.setText("Left");
              headP_right = true;
              headP_left = false;
              headPx_center = true;
            }
          }
          else{
            if(headPx_center){
                //active:頭部置中(水平方向)
              headPoseXtv.setText("Center");
              headP_right = true;
              headP_left = true;
              headPx_center = false;
            }
          }
          if(headPoseRatio_y > 1.15){
            if(headP_down){
                //active:頭部往下
              headPoseYtv.setText("Down");
              headP_up = true;
              headP_down = false;
              headPy_center = true;
            }
          }
          else if(headPoseRatio_y < 0.82){
            if(headP_up){
                //active:頭部往上
              headPoseYtv.setText("Up");
              headP_up = false;
              headP_down = true;
              headPy_center = true;
            }
          }
          else{
            if(headPy_center){
                //active:頭部置中(垂直方向)
              headPoseYtv.setText("Center");
              headP_up = true;
              headP_down = true;
              headPy_center = false;
            }
          }

                        /*Log.v(
                                TAG,
                                "[TS:"
                                        + packet.getTimestamp()
                                        + "] "
                                        + getMultiFaceLandmarksDebugString(multiFaceLandmarks));*/
                    });
        }
    }

    // Used to obtain the content view for this application. If you are extending this class, and
    // have a custom layout, override this method and return the custom layout.
    protected int getContentViewLayoutResId() {
        return R.layout.activity_main;
    }

    @Override
    protected void onResume() {
        super.onResume();
        converter =
                new ExternalTextureConverter(
                        eglManager.getContext());
        converter.setFlipY(FLIP_FRAMES_VERTICALLY);
        converter.setConsumer(processor);
        if (PermissionHelper.cameraPermissionsGranted(this)) {
            startCamera();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        converter.close();

        // Hide preview display until we re-open the camera again.
        previewDisplayView.setVisibility(View.GONE);
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    protected void onCameraStarted(SurfaceTexture surfaceTexture) {
        previewFrameTexture = surfaceTexture;
        // Make the display view visible to start showing the preview. This triggers the
        // SurfaceHolder.Callback added to (the holder of) previewDisplayView.
        previewDisplayView.setVisibility(View.VISIBLE);
    }

    protected Size cameraTargetResolution() {
        return null; // No preference and let the camera (helper) decide.
    }

    public void startCamera() {
        cameraHelper = new CameraXPreviewHelper();
        cameraHelper.setOnCameraStartedListener(
                surfaceTexture -> {
                    onCameraStarted(surfaceTexture);
                });
        CameraHelper.CameraFacing cameraFacing = CameraHelper.CameraFacing.FRONT;
        cameraHelper.startCamera(
                this, cameraFacing, /*unusedSurfaceTexture=*/ null, cameraTargetResolution());
    }

    protected Size computeViewSize(int width, int height) {
        return new Size(width, height);
    }

    protected void onPreviewDisplaySurfaceChanged(
            SurfaceHolder holder, int format, int width, int height) {
        // (Re-)Compute the ideal size of the camera-preview display (the area that the
        // camera-preview frames get rendered onto, potentially with scaling and rotation)
        // based on the size of the SurfaceView that contains the display.
        Size viewSize = computeViewSize(width, height);
        Size displaySize = cameraHelper.computeDisplaySizeFromViewSize(viewSize);
        boolean isCameraRotated = cameraHelper.isCameraRotated();

        // Connect the converter to the camera-preview frames as its input (via
        // previewFrameTexture), and configure the output width and height as the computed
        // display size.
        converter.setSurfaceTextureAndAttachToGLContext(
                previewFrameTexture,
                isCameraRotated ? displaySize.getHeight() : displaySize.getWidth(),
                isCameraRotated ? displaySize.getWidth() : displaySize.getHeight());
    }

    private void setupPreviewDisplayView() {
        previewDisplayView.setVisibility(View.GONE);
        ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
        viewGroup.addView(previewDisplayView);

        previewDisplayView
                .getHolder()
                .addCallback(
                        new SurfaceHolder.Callback() {
                            @Override
                            public void surfaceCreated(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(holder.getSurface());
                            }

                            @Override
                            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                                onPreviewDisplaySurfaceChanged(holder, format, width, height);
                            }

                            @Override
                            public void surfaceDestroyed(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(null);
                            }
                        });
    }

    private static String getMultiFaceLandmarksDebugString(
      List<NormalizedLandmarkList> multiFaceLandmarks) {
    if (multiFaceLandmarks.isEmpty()) {
      return "No face landmarks";
    }
    String multiFaceLandmarksStr = "Number of faces detected: " + multiFaceLandmarks.size() + "\n";
    int faceIndex = 0;
    for (NormalizedLandmarkList landmarks : multiFaceLandmarks) {
      multiFaceLandmarksStr +=
          "\t#Face landmarks for face[" + faceIndex + "]: " + landmarks.getLandmarkCount() + "\n";
      int landmarkIndex = 0;
      for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
        multiFaceLandmarksStr +=
            "\t\tLandmark ["
                + landmarkIndex
                + "]: ("
                + landmark.getX()
                + ", "
                + landmark.getY()
                + ", "
                + landmark.getZ()
                + ")\n";
        ++landmarkIndex;
      }
      ++faceIndex;
    }
    return multiFaceLandmarksStr;
  }
}


