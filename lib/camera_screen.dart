import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'face_detector.dart';

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const CameraScreen({super.key, required this.cameras});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  int _selectedCameraIndex = 0;
  bool _isInitialized = false;
  bool _isTakingPicture = false;
  String? _capturedImagePath;

  final YunetFaceDetector _detector = YunetFaceDetector();
  bool _detectorReady = false;
  bool _liveDetect = false;
  bool _detecting = false; // a detection pass is currently in flight
  List<FaceDetection> _faces = const [];
  Size? _frameSize; // original captured-image dimensions

  @override
  void initState() {
    super.initState();
    final preferredIndex = widget.cameras.indexWhere((c) {
      final name = c.name.toLowerCase();
      return !name.contains('integrated') &&
          !name.contains('built-in') &&
          !name.contains('internal');
    });
    _selectedCameraIndex = preferredIndex != -1 ? preferredIndex : 0;
    _initCamera(_selectedCameraIndex);
    _initDetector();
  }

  Future<void> _initDetector() async {
    try {
      await _detector.init();
      if (mounted) setState(() => _detectorReady = true);
    } catch (e) {
      debugPrint('Face detector init error: $e');
    }
  }

  Future<void> _initCamera(int index) async {
    if (widget.cameras.isEmpty) return;

    final previous = _controller;
    // Mark uninitialized while switching so the preview doesn't flash stale frames.
    if (mounted) setState(() => _isInitialized = false);

    final controller = CameraController(
      widget.cameras[index],
      ResolutionPreset.high,
      enableAudio: false,
    );

    try {
      await controller.initialize();
      await previous?.dispose();
      if (mounted) {
        setState(() {
          _controller = controller;
          _selectedCameraIndex = index;
          _isInitialized = true;
        });
      }
    } on CameraException catch (e) {
      debugPrint('Camera init error: ${e.code} — ${e.description}');
      await controller.dispose();
    }
  }

  @override
  void dispose() {
    _liveDetect = false;
    _controller?.dispose();
    _detector.dispose();
    super.dispose();
  }

  void _toggleLiveDetect(bool value) {
    setState(() {
      _liveDetect = value;
      if (!value) _faces = const [];
    });
    if (value) _detectLoop();
  }

  /// Self-rescheduling detection loop. `camera_windows` has no image-stream API,
  /// so we grab frames with takePicture(), run YuNet, then schedule the next
  /// pass — never more than one in flight at a time.
  Future<void> _detectLoop() async {
    if (!_liveDetect || _detecting) return;
    final controller = _controller;
    if (controller == null || !_isInitialized || !_detectorReady) {
      _scheduleNextTick();
      return;
    }

    _detecting = true;
    try {
      final file = await controller.takePicture();
      final bytes = await file.readAsBytes();
      final result = await _detector.detect(bytes);
      // Clean up the temporary frame file; it's only needed for inference.
      try {
        await File(file.path).delete();
      } catch (_) {}
      if (mounted && _liveDetect) {
        setState(() {
          _faces = result.faces;
          _frameSize = result.imageSize;
        });
      }
    } catch (e) {
      debugPrint('Detection error: $e');
    } finally {
      _detecting = false;
      _scheduleNextTick();
    }
  }

  void _scheduleNextTick() {
    if (!_liveDetect || !mounted) return;
    // Small delay keeps the UI responsive between capture+inference passes.
    Future.delayed(const Duration(milliseconds: 150), _detectLoop);
  }

  Future<void> _takePicture() async {
    final controller = _controller;
    if (controller == null || !_isInitialized || _isTakingPicture) return;

    setState(() => _isTakingPicture = true);

    try {
      final XFile file = await controller.takePicture();
      setState(() {
        _capturedImagePath = file.path;
        _isTakingPicture = false;
      });
    } on CameraException catch (e) {
      debugPrint('Capture error: ${e.code} — ${e.description}');
      setState(() => _isTakingPicture = false);
    }
  }

  String _cameraLabel(CameraDescription cam, int index) {
    final name = cam.name.isNotEmpty ? cam.name : 'Camera ${index + 1}';
    final direction = switch (cam.lensDirection) {
      CameraLensDirection.front => 'Front',
      CameraLensDirection.back => 'Back',
      CameraLensDirection.external => 'External',
    };
    return '$name ($direction)';
  }

  @override
  Widget build(BuildContext context) {
    final hasMultiple = widget.cameras.length > 1;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Camera'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          if (hasMultiple)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              child: DropdownButton<int>(
                value: _selectedCameraIndex,
                underline: const SizedBox.shrink(),
                icon: const Icon(Icons.videocam),
                items: [
                  for (int i = 0; i < widget.cameras.length; i++)
                    DropdownMenuItem(
                      value: i,
                      child: Text(_cameraLabel(widget.cameras[i], i)),
                    ),
                ],
                onChanged: (index) {
                  if (index != null && index != _selectedCameraIndex) {
                    _initCamera(index);
                  }
                },
              ),
            ),
        ],
      ),
      backgroundColor: Colors.black,
      body: Column(
        children: [
          Expanded(child: _buildPreviewArea()),
          _buildControlBar(context),
        ],
      ),
    );
  }

  /// Full-bleed preview area with a black backdrop. The preview keeps its true
  /// aspect ratio (letterboxed) so it never distorts, regardless of the area's
  /// shape. Face overlay + count badge live inside the aspect-locked box, so
  /// the painter's box mapping stays exact. The captured photo floats as a
  /// thumbnail here instead of reflowing the layout.
  Widget _buildPreviewArea() {
    if (!_isInitialized || _controller == null) {
      return const Center(child: CircularProgressIndicator());
    }

    final aspect = (_frameSize != null && !_frameSize!.isEmpty)
        ? _frameSize!.width / _frameSize!.height
        : _controller!.value.aspectRatio;

    return Stack(
      children: [
        Positioned.fill(
          child: Center(
            child: AspectRatio(
              aspectRatio: aspect,
              child: ClipRect(
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    CameraPreview(_controller!),
                    if (_liveDetect && _frameSize != null)
                      LayoutBuilder(
                        builder: (context, constraints) => CustomPaint(
                          painter: FaceBoxPainter(
                            faces: _faces,
                            imageSize: _frameSize!,
                            widgetSize: Size(
                              constraints.maxWidth,
                              constraints.maxHeight,
                            ),
                          ),
                        ),
                      ),
                    if (_liveDetect)
                      Positioned(
                        top: 12,
                        left: 12,
                        child: _CountBadge(count: _faces.length),
                      ),
                  ],
                ),
              ),
            ),
          ),
        ),
        if (_capturedImagePath != null)
          Positioned(
            left: 16,
            bottom: 16,
            child: _CapturedThumbnail(
              path: _capturedImagePath!,
              onTap: () => _showCapturedPhoto(context),
              onDismiss: () => setState(() => _capturedImagePath = null),
            ),
          ),
      ],
    );
  }

  Widget _buildControlBar(BuildContext context) {
    return Material(
      color: Theme.of(context).colorScheme.surface,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
        child: SafeArea(
          top: false,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Detect faces (YuNet)'),
                subtitle: Text(
                  _detectorReady
                      ? 'Live detection on the webcam feed'
                      : 'Loading model…',
                ),
                value: _liveDetect,
                onChanged: _detectorReady ? _toggleLiveDetect : null,
                secondary: const Icon(Icons.face_retouching_natural),
              ),
              const SizedBox(height: 8),
              FilledButton.icon(
                onPressed: _isInitialized && !_isTakingPicture
                    ? _takePicture
                    : null,
                icon: _isTakingPicture
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.camera_alt),
                label: const Text('Take Picture'),
                style: FilledButton.styleFrom(
                  minimumSize: const Size.fromHeight(52),
                  textStyle: const TextStyle(fontSize: 16),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showCapturedPhoto(BuildContext context) {
    final path = _capturedImagePath;
    if (path == null) return;
    showDialog<void>(
      context: context,
      barrierColor: Colors.black,
      builder: (context) => Stack(
        children: [
          Positioned.fill(
            child: InteractiveViewer(
              child: Center(child: Image.file(File(path))),
            ),
          ),
          Positioned(
            top: 40,
            right: 16,
            child: IconButton.filledTonal(
              icon: const Icon(Icons.close),
              onPressed: () => Navigator.of(context).pop(),
            ),
          ),
        ],
      ),
    );
  }
}

/// Floating thumbnail of the most recent captured photo, with a small
/// dismiss button. Tapping opens the full-screen viewer.
class _CapturedThumbnail extends StatelessWidget {
  const _CapturedThumbnail({
    required this.path,
    required this.onTap,
    required this.onDismiss,
  });

  final String path;
  final VoidCallback onTap;
  final VoidCallback onDismiss;

  @override
  Widget build(BuildContext context) {
    return Stack(
      clipBehavior: Clip.none,
      children: [
        GestureDetector(
          onTap: onTap,
          child: Container(
            width: 84,
            height: 84,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.white, width: 2),
              boxShadow: const [
                BoxShadow(color: Colors.black54, blurRadius: 6),
              ],
            ),
            clipBehavior: Clip.antiAlias,
            child: Image.file(File(path), fit: BoxFit.cover),
          ),
        ),
        Positioned(
          top: -8,
          right: -8,
          child: GestureDetector(
            onTap: onDismiss,
            child: Container(
              decoration: const BoxDecoration(
                color: Colors.black87,
                shape: BoxShape.circle,
              ),
              padding: const EdgeInsets.all(2),
              child: const Icon(Icons.close, size: 16, color: Colors.white),
            ),
          ),
        ),
      ],
    );
  }
}

class _CountBadge extends StatelessWidget {
  const _CountBadge({required this.count});

  final int count;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Text(
        '$count ${count == 1 ? 'face' : 'faces'}',
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}

/// Paints face bounding boxes + confidence scores, scaling detections from the
/// original image's pixel space into the preview widget's space. The preview
/// is stretched to fill this box (CameraPreview under StackFit.expand), so we
/// map each axis independently with no centering offset to match exactly.
class FaceBoxPainter extends CustomPainter {
  FaceBoxPainter({
    required this.faces,
    required this.imageSize,
    required this.widgetSize,
    this.mirror = true,
  });

  final List<FaceDetection> faces;
  final Size imageSize;
  final Size widgetSize;

  /// The CameraPreview is shown mirrored (like a mirror), but detection runs on
  /// the un-mirrored captured frame. When true, flip box X so it aligns with
  /// what the user sees on screen.
  final bool mirror;

  @override
  void paint(Canvas canvas, Size size) {
    if (faces.isEmpty || imageSize.isEmpty) return;

    final scaleX = widgetSize.width / imageSize.width;
    final scaleY = widgetSize.height / imageSize.height;

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.greenAccent;

    for (final face in faces) {
      final boxW = face.box.width * scaleX;
      final left = mirror
          ? widgetSize.width - face.box.left * scaleX - boxW
          : face.box.left * scaleX;
      final rect = Rect.fromLTWH(
        left,
        face.box.top * scaleY,
        boxW,
        face.box.height * scaleY,
      );
      canvas.drawRect(rect, boxPaint);

      final label = '${(face.score * 100).toStringAsFixed(0)}%';
      final tp = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.black,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final labelBg = Rect.fromLTWH(
        rect.left,
        rect.top - tp.height - 4,
        tp.width + 8,
        tp.height + 4,
      );
      canvas.drawRect(labelBg, Paint()..color = Colors.greenAccent);
      tp.paint(canvas, Offset(rect.left + 4, rect.top - tp.height - 2));
    }
  }

  @override
  bool shouldRepaint(covariant FaceBoxPainter oldDelegate) {
    return oldDelegate.faces != faces ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.widgetSize != widgetSize;
  }
}
