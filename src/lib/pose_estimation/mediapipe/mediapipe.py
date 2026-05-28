import os
import numpy as np
import cv2

try:
    import mediapipe as mp
except Exception:
    mp = None

from utils.annotation import Annotation


class MediaPipePose:
    """Simple MediaPipe Pose wrapper providing a compatible interface
    to the existing pipeline (returns list of `Annotation` objects).

    Notes:
    - MediaPipe Pose is single-person by default; this wrapper will
      return at most one `Annotation` per image.
    - Keypoints are returned with shape (18, 3) to match the expectations
      of downstream code (x, y normalized, score). Missing joints are zeroed.
    """

    _params = dict(
        min_total_joints=5,
        min_leg_joints=1,
        include_head=True,
    )

    def __init__(self, size=(256, 256), model_path=None, **kwargs):
        self.__dict__.update(self._params)
        self.__dict__.update(kwargs)
        if not isinstance(size, (tuple, list)):
            size = (size, size)
        self.height, self.width = size

        if mp is None:
            raise ImportError('mediapipe is not installed')

        # Prefer the new MediaPipe Tasks API (medapipe>=0.10). If available,
        # load or download a bundled task model and create a PoseLandmarker.
        self.use_tasks = False
        try:
            from mediapipe.tasks.python import vision as mv
            from mediapipe.tasks.python.core.base_options import BaseOptions

            # determine model path: allow override via model_path arg (only if a string)
            if model_path and isinstance(model_path, str):
                model_file = model_path
            else:
                model_file = os.path.join(os.getcwd(), '..', 'weights', 'mediapipe', 'pose_landmarker.task')
            model_file = os.path.abspath(model_file)
            model_dir = os.path.dirname(model_file)
            os.makedirs(model_dir, exist_ok=True)

            # download model bundle if missing
            if not os.path.isfile(model_file):
                try:
                    url = 'https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task'
                    print('[SMOKE] Downloading MediaPipe task model to', model_file)
                    import urllib.request
                    urllib.request.urlretrieve(url, model_file)
                except Exception as e:
                    print('[SMOKE] Failed to download MediaPipe model:', e)
                    raise

            opts = mv.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_file),
                                            running_mode=mv.RunningMode.VIDEO,
                                            num_poses=10,
                                            min_pose_detection_confidence=0.35,
                                            min_tracking_confidence=0.3)
            self.pl = mv.PoseLandmarker.create_from_options(opts)
            self.use_tasks = True
        except Exception:
            # Fallback to legacy solutions API if present
            try:
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(static_image_mode=False,
                                              model_complexity=1,
                                              enable_segmentation=False,
                                              min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)
                self.use_tasks = False
            except Exception:
                raise ImportError('Failed to initialize MediaPipe Pose (tasks API and solutions API unavailable)')

        # Mapping: target 18 keypoints (approximate) <- MediaPipe landmarks
        # We'll fill commonly used joints: nose, shoulders, elbows, wrists, hips, knees, ankles
        # Indices below are MediaPipe landmark indices.
        self._mapping = {
            0: 0,    # nose
            1: 11,   # left eye -> use left_shoulder as proxy
            2: 12,   # right eye -> use right_shoulder as proxy
            3: 7,    # left ear
            4: 8,    # right ear
            5: 11,   # neck ~ left_shoulder
            6: 12,   # body
            7: 23,   # left hip
            8: 13,   # left shoulder
            9: 15,   # left elbow
            10: 17,  # left wrist
            11: 12,  # right shoulder
            12: 14,  # right elbow
            13: 16,  # right wrist
            14: 24,  # right hip
            15: 25,  # left knee
            16: 26,  # right knee
            17: 27,  # left ankle
        }

    def _landmark_to_kp(self, lm):
        # MediaPipe landmarks are normalized [0,1] for x and y
        return float(lm.x), float(lm.y), float(lm.visibility) if hasattr(lm, 'visibility') else 1.0

    def predict(self, image, get_bbox=False, frame_index=None, fps=None, timestamp_ms=None):
        """Run MediaPipe pose on an RGB image.
        Returns list of `Annotation` objects with `.keypoints` shaped (18,3).
        """
        if mp is None:
            raise ImportError('mediapipe is not installed')

        img_h, img_w = image.shape[:2]
        lm = None
        # use tasks API
        if getattr(self, 'use_tasks', False):
            # Use top-level MediaPipe Image helper to wrap numpy arrays
            # MediaPipe Image helper does not provide a direct create_from_array
            # helper in some builds; write a temp JPEG and load via create_from_file.
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            tmp_name = tmp.name
            tmp.close()
            # image is RGB; convert to BGR for OpenCV write
            cv2.imwrite(tmp_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            mp_image = mp.Image.create_from_file(tmp_name)
            try:
                # Prefer detect_for_video in VIDEO mode when timestamp is available
                if timestamp_ms is None and frame_index is not None and fps is not None and fps > 0:
                    timestamp_ms = int((frame_index / float(fps)) * 1000.0)
                if timestamp_ms is not None and hasattr(self.pl, 'detect_for_video'):
                    result = self.pl.detect_for_video(mp_image, timestamp_ms)
                else:
                    result = self.pl.detect(mp_image)
            finally:
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass
            # PoseLandmarkerResult stores pose_landmarks as a list in .pose_landmarks
            if not result or not getattr(result, 'pose_landmarks', None):
                return []
            # take first detected pose
            plm = result.pose_landmarks[0]
            # Some mediapipe builds return a wrapper with `.landmark`, others
            # return a plain list of landmarks. Handle both cases.
            if hasattr(plm, 'landmark'):
                lm = plm.landmark
            else:
                lm = plm
        else:
            results = self.pose.process(image)
            if not results or not results.pose_landmarks:
                return []
            lm = results.pose_landmarks.landmark
        kp = np.zeros((18, 3), dtype=np.float64)
        for tgt_idx, mp_idx in self._mapping.items():
            if mp_idx < len(lm):
                x, y, s = self._landmark_to_kp(lm[mp_idx])
                # keep normalized coordinates (x,y) in [0,1]
                kp[tgt_idx, 0] = tgt_idx  # keep id in first col similar to trtpose (index)
                kp[tgt_idx, 1] = x
                kp[tgt_idx, 2] = y
            else:
                kp[tgt_idx] = np.array([tgt_idx, 0., 0.])

        ann = Annotation(kp)
        if get_bbox:
            # compute bbox from non-zero keypoints
            xs = kp[:, 1]
            ys = kp[:, 2]
            valid = (xs != 0) & (ys != 0)
            if valid.any():
                xmin = float(xs[valid].min() * img_w)
                xmax = float(xs[valid].max() * img_w)
                ymin = float(ys[valid].min() * img_h)
                ymax = float(ys[valid].max() * img_h)
                w = xmax - xmin
                h = ymax - ymin
                ann.bbox = [float(max(0, xmin)), float(max(0, ymin)), float(w), float(h)]
        return [ann]

    def close(self):
        try:
            self.pose.close()
        except Exception:
            pass
