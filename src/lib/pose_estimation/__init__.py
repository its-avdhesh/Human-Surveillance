"""Pose estimator factory with lazy backend imports.

This avoids importing heavy TRT-Pose / TensorRT modules unless the
`trtpose` backend is explicitly requested.
"""

def get_pose_estimator(name, **kwargs):
    if name == 'trtpose':
        from .trtpose.trtpose import TrtPose
        return TrtPose(**kwargs)
    elif name == 'mediapipe':
        from .mediapipe.mediapipe import MediaPipePose
        return MediaPipePose(**kwargs)
    else:
        raise ValueError(f'Unknown pose estimator: {name}')