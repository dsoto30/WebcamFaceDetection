import 'dart:io';
import 'dart:ui' show Rect;

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import 'face_detector.dart';

/// Turns a captured webcam frame plus its detected faces into saved image
/// files: one upper-body crop per person, or the whole frame when no face was
/// detected. Files land in a `captures/` folder next to where the app runs
/// (the repo root under `flutter run`), which is git-ignored.

/// Integer pixel rectangle used for cropping.
class CropRect {
  const CropRect(this.x, this.y, this.width, this.height);
  final int x;
  final int y;
  final int width;
  final int height;
}

/// Estimates an upper-body crop from a face box. YuNet only locates the face,
/// so we widen it ~3x, add a little headroom above, and extend well below to
/// take in the torso — then clamp everything to the image bounds.
CropRect bodyRectForFace(Rect face, int imgW, int imgH) {
  final centerX = face.center.dx;
  final bodyWidth = face.width * 3.0;

  var left = centerX - bodyWidth / 2;
  var top = face.top - face.height * 0.6;
  var right = left + bodyWidth;
  var bottom = top + face.height * 8.0;

  left = left.clamp(0.0, imgW.toDouble());
  top = top.clamp(0.0, imgH.toDouble());
  right = right.clamp(0.0, imgW.toDouble());
  bottom = bottom.clamp(0.0, imgH.toDouble());

  final w = (right - left).round();
  final h = (bottom - top).round();
  return CropRect(left.round(), top.round(), w, h);
}

/// The folder captured pictures are written to (repo root / `captures`).
Directory capturesDir() =>
    Directory('${Directory.current.path}${Platform.pathSeparator}captures');

/// Compact `yyyyMMdd_HHmmss` stamp built without adding an intl dependency.
String _timestamp(DateTime t) {
  String two(int v) => v.toString().padLeft(2, '0');
  return '${t.year}${two(t.month)}${two(t.day)}_'
      '${two(t.hour)}${two(t.minute)}${two(t.second)}';
}

/// Decodes [jpegBytes] once and saves one upper-body crop per detected face.
/// Each capture gets its own timestamped subfolder under `captures/`, and the
/// per-person crops for that shot land inside it. With no faces, saves the full
/// frame in the folder instead so the button always yields a file. Returns the
/// paths written (empty only if the image can't be decoded).
Future<List<String>> saveCrops({
  required Uint8List jpegBytes,
  required List<FaceDetection> faces,
}) async {
  final decoded = img.decodeImage(jpegBytes);
  if (decoded == null) return const [];

  final stamp = _timestamp(DateTime.now());

  // One folder per capture so each shot's bodies stay grouped together.
  final captureDir = Directory(
    '${capturesDir().path}${Platform.pathSeparator}$stamp',
  );
  if (!captureDir.existsSync()) captureDir.createSync(recursive: true);

  final saved = <String>[];

  if (faces.isEmpty) {
    final path = '${captureDir.path}${Platform.pathSeparator}frame.jpg';
    await File(path).writeAsBytes(img.encodeJpg(decoded));
    saved.add(path);
    return saved;
  }

  for (var i = 0; i < faces.length; i++) {
    final rect = bodyRectForFace(faces[i].box, decoded.width, decoded.height);
    if (rect.width <= 0 || rect.height <= 0) continue;
    final crop = img.copyCrop(
      decoded,
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
    );
    final path =
        '${captureDir.path}${Platform.pathSeparator}person_${i + 1}.jpg';
    await File(path).writeAsBytes(img.encodeJpg(crop));
    saved.add(path);
  }
  return saved;
}
