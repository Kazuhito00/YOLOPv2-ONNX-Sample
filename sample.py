import os
import copy
import time
import argparse

import cv2
import numpy as np

import onnxruntime

from utils import utils_onnx


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--video',
        type=str,
        default='sample.mp4',
    )

    parser.add_argument(
        '--model',
        type=str,
        default='weight/YOLOPv2.onnx',
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.3,
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.45,
    )

    args = parser.parse_args()

    return args


def run_inference(
    onnx_session,
    image,
    score_th,
    nms_th,
):
    # 前処理
    # パディング処理を実行
    input_image = copy.deepcopy(image)
    input_image, _, (pad_w, pad_h) = utils_onnx.letterbox(input_image)

    # BGR→RGB変換
    input_image = input_image[:, :, ::-1].transpose(2, 0, 1)

    # PyTorch Tensorに変換
    input_image = np.ascontiguousarray(input_image)

    # 正規化
    input_image = input_image.astype('float32')
    input_image /= 255.0

    # NCHWに変換
    input_image = np.expand_dims(input_image, axis=0)

    # 推論
    input_name = onnx_session.get_inputs()[0].name
    results = onnx_session.run(None, {input_name: input_image})

    result_dets = []
    result_dets.append(results[0][0])
    result_dets.append(results[0][1])
    result_dets.append(results[0][2])

    anchor_grid = []
    anchor_grid.append(results[1])
    anchor_grid.append(results[2])
    anchor_grid.append(results[3])

    # 後処理
    # 車検出
    result_dets = utils_onnx.split_for_trace_model(
        result_dets,
        anchor_grid,
    )

    result_dets = utils_onnx.non_max_suppression(
        result_dets,
        conf_thres=score_th,
        iou_thres=nms_th,
    )

    bboxes = []
    scores = []
    class_ids = []
    for result_det in result_dets:
        if len(result_det) > 0:
            # バウンディングボックスのスケールを調整
            result_det[:, :4] = utils_onnx.scale_coords(
                input_image.shape[2:],
                result_det[:, :4],
                image.shape,
            ).round()

            # バウンディングボックス、スコア、クラスIDを取得
            for *xyxy, score, class_id in reversed(result_det):
                x1, y1 = xyxy[0], xyxy[1]
                x2, y2 = xyxy[2], xyxy[3]

                bboxes.append([int(x1), int(y1), int(x2), int(y2)])
                scores.append(float(score))
                class_ids.append(int(class_id))

    # 路面セグメンテーション
    result_road_seg = utils_onnx.driving_area_mask(
        results[4],
        (pad_w, pad_h),
    )

    # レーンセグメンテーション
    result_lane_seg = utils_onnx.lane_line_mask(
        results[5],
        (pad_w, pad_h),
    )

    return (bboxes, scores, class_ids), result_road_seg, result_lane_seg


def main():
    # 引数
    args = get_args()

    video_path = args.video

    model_path = args.model
    score_th = args.score_th
    nms_th = args.nms_th

    # ONNXファイル有無確認
    if not os.path.isfile(model_path):
        import urllib.request
        url = 'https://github.com/Kazuhito00/YOLOPv2-ONNX-Sample/releases/download/v0.0.0/YOLOPv2.onnx'
        save_path = 'weight/YOLOPv2.onnx'

        print('Start Download:YOLOPv2.onnx')
        urllib.request.urlretrieve(url, save_path)
        print('Finish Download')

    # モデルロード
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    # ビデオ読み込み
    video_capture = cv2.VideoCapture(video_path)

    while True:
        start_time = time.time()

        # 画像読み込み
        ret, frame = video_capture.read()
        if not ret:
            break

        # 推論
        (bboxes, scores, class_ids), road_seg, lane_seg = run_inference(
            onnx_session,
            frame,
            score_th,
            nms_th,
        )

        elapsed_time = time.time() - start_time

        # 推論結果可視化
        debug_image = draw_debug_image(
            frame,
            (bboxes, scores, class_ids),
            road_seg,
            lane_seg,
            elapsed_time,
        )

        cv2.imshow("YOLOPv2", debug_image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    video_capture.release()
    cv2.destroyAllWindows()


def draw_debug_image(
    image,
    car_dets,
    road_seg,
    lane_seg,
    elapsed_time,
):
    debug_image = copy.deepcopy(image)

    # 路面セグメンテーション
    image_width, image_height = debug_image.shape[1], debug_image.shape[0]

    # マスク画像を生成
    road_mask = np.stack((road_seg, ) * 3, axis=-1).astype('float32')
    road_mask = cv2.resize(
        road_mask,
        dsize=(image_width, image_height),
        interpolation=cv2.INTER_LINEAR,
    )
    road_mask = np.where(road_mask > 0.5, 0, 1)

    # マスク画像と画像を合成
    bg_image = np.zeros(debug_image.shape, dtype=np.uint8)
    bg_image[:] = [0, 255, 0]
    road_mask_image = np.where(road_mask, debug_image, bg_image)

    # 半透明画像として合成
    debug_image = cv2.addWeighted(debug_image, 0.5, road_mask_image, 0.5, 1.0)

    # レーンセグメンテーション
    # マスク画像を生成
    road_mask = np.stack((lane_seg, ) * 3, axis=-1).astype('float32')
    road_mask = cv2.resize(
        road_mask,
        dsize=(image_width, image_height),
        interpolation=cv2.INTER_LINEAR,
    )
    road_mask = np.where(road_mask > 0.5, 0, 1)

    # マスク画像と画像を合成
    bg_image = np.zeros(debug_image.shape, dtype=np.uint8)
    bg_image[:] = [0, 0, 255]
    road_mask_image = np.where(road_mask, debug_image, bg_image)

    # 半透明画像として合成
    debug_image = cv2.addWeighted(debug_image, 0.5, road_mask_image, 0.5, 1.0)

    # 車検出結果
    for bbox, score, class_id in zip(*car_dets):
        # バウンディングボックス
        cv2.rectangle(
            debug_image,
            pt1=(bbox[0], bbox[1]),
            pt2=(bbox[2], bbox[3]),
            color=(0, 255, 255),
            thickness=2,
        )

        # クラスID、スコア
        text = '%s:%s' % (str(class_id), '%.2f' % score)
        cv2.putText(
            debug_image,
            text,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color=(0, 255, 255),
            thickness=2,
        )

    # 処理時間
    cv2.putText(
        debug_image,
        "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return debug_image


if __name__ == "__main__":
    main()
