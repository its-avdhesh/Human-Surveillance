try:
    import _init_paths
except Exception:
    import sys, os
    # ensure we can import project modules when running from tools/
    repo_root = os.path.join(os.getcwd())
    src_path = os.path.join(repo_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    try:
        import _init_paths
    except Exception:
        pass
import cv2
import traceback
from utils.config import Config
from pose_estimation import get_pose_estimator
from classifier import get_classifier
from utils import utils


def main():
    # resolve project base dir relative to this script so paths work from anywhere
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cfg_path = os.path.join(base_dir, 'configs', 'infer_trtpose_deepsort_dnn.yaml')
    cfg = Config(cfg_path)

    # Use MediaPipe backend for macOS (no NVIDIA GPU required)
    pose_kwargs = dict(cfg.POSE)
    pose_kwargs['name'] = 'mediapipe'
    print('[SMOKE] Pose kwargs (using MediaPipe):', pose_kwargs)

    try:
        pose_estimator = get_pose_estimator(**pose_kwargs)
        print('[SMOKE] MediaPipe estimator initialized')
    except Exception as e:
        print('[ERROR] Failed to init MediaPipe estimator:', e)
        traceback.print_exc()
        return

    # init classifier
    clf_kwargs = dict(cfg.CLASSIFIER)
    try:
        clf = get_classifier(**clf_kwargs)
        print('[SMOKE] Classifier initialized')
    except Exception as e:
        print('[ERROR] Failed to init classifier:', e)
        traceback.print_exc()
        return

    # open test video (resolve relative to project root)
    test_video = os.path.join(base_dir, 'test_data', 'fun_theory.mp4')
    cap = cv2.VideoCapture(test_video)
    if not cap.isOpened():
        print('[ERROR] Failed to open test video')
        return

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    max_frames = 10
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print('[SMOKE] End of video')
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preds = pose_estimator.predict(rgb, get_bbox=True, frame_index=frame_count, fps=fps)
            print(f'[SMOKE] Frame {frame_count}: detected {len(preds)} persons')
            if len(preds) > 0:
                # check keypoints shape
                kp = preds[0].keypoints
                print('[SMOKE] Keypoints shape/type:', getattr(kp, 'shape', type(kp)))
                # convert to openpose format
                preds = utils.convert_to_openpose_skeletons(preds)
                # set ids
                for i, p in enumerate(preds):
                    p.set_tracked_id(i)
                # classify
                preds = clf.classify(preds)
                print('[SMOKE] Classification results:', [p.action for p in preds])
            frame_count += 1
        print('[SMOKE] Completed run without exceptions')
    except Exception as e:
        print('[ERROR] Exception during smoke run:', e)
        traceback.print_exc()
    finally:
        cap.release()
        try:
            pose_estimator.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
