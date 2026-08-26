import 'dart:math' as math;
import 'dart:ui' show Rect, Size;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

/// A single detected face in the coordinate space of the original
/// (full-resolution) captured image.
class FaceDetection {
  const FaceDetection(this.box, this.score);

  final Rect box;
  final double score;
}

/// Result of a detection pass: the faces plus the size of the source image
/// they were decoded against (so the caller can map boxes to the preview
/// without decoding the frame a second time).
class DetectionResult {
  const DetectionResult(this.faces, this.imageSize);

  static const DetectionResult empty = DetectionResult([], Size.zero);

  final List<FaceDetection> faces;
  final Size imageSize;
}

/// Runs the YuNet face-detection ONNX model
/// (`assets/models/face_detection_yunet_2023mar.onnx`) via onnxruntime.
///
/// The raw YuNet ONNX graph emits undecoded detection heads at strides 8/16/32
/// (`cls_*`, `obj_*`, `bbox_*`, `kps_*`). This class reproduces the decode that
/// OpenCV's `FaceDetectorYN` performs internally: build per-cell priors, decode
/// boxes, compute `score = sqrt(cls * obj)`, then non-maximum suppression.
class YunetFaceDetector {
  YunetFaceDetector({
    this.inputWidth = 640,
    this.inputHeight = 640,
    this.scoreThreshold = 0.8,
    this.nmsThreshold = 0.3,
    this.topK = 50,
  });

  static const String _modelAsset =
      'assets/models/face_detection_yunet_2023mar.onnx';
  static const List<int> _strides = [8, 16, 32];

  /// Model input size. Must be multiples of 32 (largest stride).
  final int inputWidth;
  final int inputHeight;
  final double scoreThreshold;
  final double nmsThreshold;
  final int topK;

  OrtSession? _session;
  String _inputName = 'input';

  bool get isReady => _session != null;

  Future<void> init() async {
    OrtEnv.instance.init();
    final rawModel = await rootBundle.load(_modelAsset);
    final options = OrtSessionOptions();
    _session = OrtSession.fromBuffer(rawModel.buffer.asUint8List(), options);
    if (_session!.inputNames.isNotEmpty) {
      _inputName = _session!.inputNames.first;
    }
    debugPrint(
      'YuNet loaded. inputs=${_session!.inputNames} '
      'outputs=${_session!.outputNames}',
    );
  }

  /// Decodes [jpegBytes], runs inference, and returns detected faces in the
  /// original image's pixel coordinates along with the source image size.
  /// Returns [DetectionResult.empty] if the model isn't ready or the image
  /// can't be decoded.
  Future<DetectionResult> detect(Uint8List jpegBytes) async {
    final session = _session;
    if (session == null) return DetectionResult.empty;

    final decoded = img.decodeImage(jpegBytes);
    if (decoded == null) return DetectionResult.empty;

    final origW = decoded.width;
    final origH = decoded.height;
    final imageSize = Size(origW.toDouble(), origH.toDouble());
    final resized = img.copyResize(
      decoded,
      width: inputWidth,
      height: inputHeight,
    );

    final input = _buildInputTensor(resized);
    final inputOrt = OrtValueTensor.createTensorWithDataList(input, [
      1,
      3,
      inputHeight,
      inputWidth,
    ]);

    final runOptions = OrtRunOptions();
    List<OrtValue?>? outputs;
    try {
      // runAsync moves the native inference onto a worker isolate so it does
      // not block the UI thread while a face is being processed.
      final future = session.runAsync(runOptions, {_inputName: inputOrt});
      outputs = future == null ? null : await future;
      if (outputs == null) return DetectionResult(const [], imageSize);

      final byName = <String, List<double>>{};
      final names = session.outputNames;
      for (var i = 0; i < names.length; i++) {
        final v = outputs[i]?.value;
        if (v != null) byName[names[i]] = _flattenDoubles(v);
      }

      final candidates = _decode(byName);
      final kept = _nms(candidates);

      final scaleX = origW / inputWidth;
      final scaleY = origH / inputHeight;
      final faces = kept
          .map(
            (d) => FaceDetection(
              Rect.fromLTWH(
                d.box.left * scaleX,
                d.box.top * scaleY,
                d.box.width * scaleX,
                d.box.height * scaleY,
              ),
              d.score,
            ),
          )
          .toList();
      return DetectionResult(faces, imageSize);
    } finally {
      inputOrt.release();
      runOptions.release();
      outputs?.forEach((o) => o?.release());
    }
  }

  void dispose() {
    _session?.release();
    _session = null;
    OrtEnv.instance.release();
  }

  /// Builds a CHW, BGR, 0-255 float32 tensor (YuNet's expected input: no
  /// mean/scale normalization, BGR channel order like OpenCV).
  ///
  /// Reads the whole image as one interleaved BGR byte buffer instead of
  /// per-pixel `getPixel` calls (much faster — no Pixel object per pixel).
  Float32List _buildInputTensor(img.Image image) {
    final plane = image.width * image.height;
    final bgr = image.getBytes(order: img.ChannelOrder.bgr);
    final data = Float32List(3 * plane);
    for (var i = 0; i < plane; i++) {
      final j = i * 3;
      data[i] = bgr[j].toDouble(); // B plane
      data[plane + i] = bgr[j + 1].toDouble(); // G plane
      data[2 * plane + i] = bgr[j + 2].toDouble(); // R plane
    }
    return data;
  }

  List<FaceDetection> _decode(Map<String, List<double>> outputs) {
    final results = <FaceDetection>[];
    for (final stride in _strides) {
      final cls = outputs['cls_$stride'];
      final obj = outputs['obj_$stride'];
      final bbox = outputs['bbox_$stride'];
      if (cls == null || obj == null || bbox == null) continue;

      final cols = inputWidth ~/ stride;
      final rows = inputHeight ~/ stride;
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          final idx = r * cols + c;
          final clsScore = cls[idx].clamp(0.0, 1.0);
          final objScore = obj[idx].clamp(0.0, 1.0);
          final score = math.sqrt(clsScore * objScore);
          if (score < scoreThreshold) continue;

          final cx = (c + bbox[idx * 4 + 0]) * stride;
          final cy = (r + bbox[idx * 4 + 1]) * stride;
          final bw = math.exp(bbox[idx * 4 + 2]) * stride;
          final bh = math.exp(bbox[idx * 4 + 3]) * stride;
          results.add(
            FaceDetection(
              Rect.fromLTWH(cx - bw / 2, cy - bh / 2, bw, bh),
              score,
            ),
          );
        }
      }
    }
    return results;
  }

  /// Greedy non-maximum suppression by IoU.
  List<FaceDetection> _nms(List<FaceDetection> detections) {
    detections.sort((a, b) => b.score.compareTo(a.score));
    final kept = <FaceDetection>[];
    final suppressed = List<bool>.filled(detections.length, false);
    for (var i = 0; i < detections.length && kept.length < topK; i++) {
      if (suppressed[i]) continue;
      final a = detections[i];
      kept.add(a);
      for (var j = i + 1; j < detections.length; j++) {
        if (suppressed[j]) continue;
        if (_iou(a.box, detections[j].box) > nmsThreshold) {
          suppressed[j] = true;
        }
      }
    }
    return kept;
  }

  double _iou(Rect a, Rect b) {
    final interLeft = math.max(a.left, b.left);
    final interTop = math.max(a.top, b.top);
    final interRight = math.min(a.right, b.right);
    final interBottom = math.min(a.bottom, b.bottom);
    final interW = interRight - interLeft;
    final interH = interBottom - interTop;
    if (interW <= 0 || interH <= 0) return 0.0;
    final interArea = interW * interH;
    final union = a.width * a.height + b.width * b.height - interArea;
    if (union <= 0) return 0.0;
    return interArea / union;
  }

  /// Recursively flattens an ONNX tensor `.value` (arbitrarily nested lists)
  /// into a flat list of doubles for index-based access.
  List<double> _flattenDoubles(Object value) {
    final out = <double>[];
    void walk(Object? v) {
      if (v is num) {
        out.add(v.toDouble());
      } else if (v is List) {
        for (final e in v) {
          walk(e);
        }
      }
    }

    walk(value);
    return out;
  }
}
