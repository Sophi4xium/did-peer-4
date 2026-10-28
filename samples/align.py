"""写真から顔を検出し、ArcFace用の 112x112 アライメント済み画像(PNG)を出力する。

使い方:
  pip install insightface onnxruntime opencv-python-headless
  python tools/align.py <buffalo_l のあるディレクトリ> <出力先> <画像1> [画像2 ...]

  ※ <buffalo_l のあるディレクトリ> は、その直下に models/buffalo_l/det_10g.onnx がある場所。
     (buffalo_l.zip を <dir>/models/buffalo_l/ に展開する)

複数の顔が写っている場合は、最も大きい顔を使う。顔が見つからない画像はスキップする。
"""
import os
import sys

import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    root, out_dir, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)

    app = FaceAnalysis(name="buffalo_l", root=root, allowed_modules=["detection"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    for p in paths:
        img = cv2.imread(p)
        faces = app.get(img) if img is not None else []
        if not faces:
            print(f"skip (顔なし/読込失敗): {p}")
            continue
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        crop = face_align.norm_crop(img, f.kps, image_size=112)  # 5点アライメント
        out = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".png")
        cv2.imwrite(out, crop)
        print(f"ok: {p} -> {out} (det_score={f.det_score:.2f}, faces={len(faces)})")


if __name__ == "__main__":
    main()
