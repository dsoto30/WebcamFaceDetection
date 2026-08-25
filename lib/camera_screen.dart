import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

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
    _controller?.dispose();
    super.dispose();
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
      body: Column(
        children: [
          Expanded(
            flex: _capturedImagePath != null ? 2 : 3,
            child: _isInitialized && _controller != null
                ? ClipRect(child: CameraPreview(_controller!))
                : const Center(child: CircularProgressIndicator()),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 16),
            child: ElevatedButton.icon(
              onPressed:
                  _isInitialized && !_isTakingPicture ? _takePicture : null,
              icon: _isTakingPicture
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.camera_alt),
              label: const Text('Take Picture'),
              style: ElevatedButton.styleFrom(
                padding:
                    const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                textStyle: const TextStyle(fontSize: 16),
              ),
            ),
          ),
          if (_capturedImagePath != null) ...[
            const Divider(),
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(
                'Captured Photo',
                style: Theme.of(context).textTheme.titleMedium,
              ),
            ),
            Expanded(
              flex: 2,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(8, 0, 8, 8),
                child: Image.file(
                  File(_capturedImagePath!),
                  fit: BoxFit.contain,
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
