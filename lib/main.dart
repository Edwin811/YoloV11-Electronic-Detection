import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(title: 'YOLOv11 Float32', home: HomePage());
  }
}

class HomePage extends StatefulWidget {
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Interpreter? _interpreter;
  List<String> _labels = [];
  img.Image? _selectedImage;
  final picker = ImagePicker();
  List<Map<String, dynamic>> _boxes = [];

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/best_float32.tflite');
      final labelData = await rootBundle.loadString('assets/classes.txt');
      _labels =
          labelData.split('\n').where((e) => e.trim().isNotEmpty).toList();
      if (_labels.isEmpty) {
        _labels = ['Laptop', 'Mobile phone', 'Computer monitor'];
      }
      print("✅ Model & labels loaded");
    } catch (e) {
      print("❌ Failed to load model: $e");
    }
  }

  Future<void> pickImage() async {
    try {
      final picked = await picker.pickImage(source: ImageSource.gallery);
      if (picked != null) {
        print("📷 Gambar dipilih: ${picked.path}");
        final bytes = await picked.readAsBytes();
        final image = img.decodeImage(bytes);
        if (image != null) {
          setState(() {
            _selectedImage = image;
          });
          print("📥 Gambar berhasil dikonversi");
          await runInference(image);
        } else {
          print("⚠️ Gagal mendekode gambar");
        }
      } else {
        print("⚠️ Tidak ada gambar dipilih");
      }
    } catch (e) {
      print("❌ Error saat memilih gambar: $e");
    }
  }

  Future<void> runInference(img.Image image) async {
    if (_interpreter == null) {
      print("❌ Model belum dimuat");
      return;
    }

    try {
      final inputSize = 416;
      final resized = img.copyResize(
        image,
        width: inputSize,
        height: inputSize,
      );
      print("📐 Gambar berhasil di-resize");

      var input = List.generate(
        1,
        (_) => List.generate(
          inputSize,
          (y) => List.generate(inputSize, (x) {
            final pixel = resized.getPixel(x, y);
            return [pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0];
          }),
        ),
      );

      var outputShapes = _interpreter!.getOutputTensors()[0].shape;
      print("📤 Output tensor shape: $outputShapes");

      var output = List.generate(
        outputShapes[0],
        (_) => List.generate(
          outputShapes[1],
          (_) => List.filled(outputShapes[2], 0.0),
        ),
      );

      _interpreter!.run(input, output);
      print("✅ Inferensi selesai.");

      parseYoloOutput(output, 0.3); // <-- PENTING!
    } catch (e) {
      print("❌ Error saat inferensi: $e");
    }
  }

  void parseYoloOutput(List<List<List<double>>> output, double confThreshold) {
    final outputTensor = output[0]; // [7][3549]
    final boxes = <Map<String, dynamic>>[];

    final numAnchors = outputTensor[0].length;
    final numClasses = outputTensor.length - 5;

    for (int i = 0; i < numAnchors; i++) {
      double conf = outputTensor[4][i]; // ambil confidence

      if (conf > confThreshold) {
        double cx = outputTensor[0][i];
        double cy = outputTensor[1][i];
        double w = outputTensor[2][i];
        double h = outputTensor[3][i];

        List<double> classProbs =
            outputTensor.sublist(5).map((e) => e[i]).toList();

        int classId = 0;
        double maxProb = 0.0;
        for (int j = 0; j < classProbs.length; j++) {
          if (classProbs[j] > maxProb) {
            maxProb = classProbs[j];
            classId = j;
          }
        }

        final label = (classId < _labels.length) ? _labels[classId] : "Unknown";

        final x1 = (cx - w / 2) * 416;
        final y1 = (cy - h / 2) * 416;
        final x2 = (cx + w / 2) * 416;
        final y2 = (cy + h / 2) * 416;

        boxes.add({
          "class": classId,
          "label": label,
          "confidence": conf,
          "box": [x1, y1, x2, y2],
        });

        print(
          "🎯 Deteksi [$i]: $label, conf: ${(conf * 100).toStringAsFixed(1)}%",
        );
      }
    }

    print("🔍 Total box valid: ${boxes.length}");
    setState(() {
      _boxes = boxes;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('YOLOv8 Float32 (Offline)')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(onPressed: pickImage, child: Text('Pilih Gambar')),
            if (_selectedImage != null) ...[
              SizedBox(height: 20),
              SizedBox(
                width: 416,
                height: 416,
                child: Stack(
                  children: [
                    Image.memory(
                      Uint8List.fromList(img.encodeJpg(_selectedImage!)),
                      width: 416,
                      height: 416,
                      fit: BoxFit.cover,
                    ),
                    ..._boxes.map((b) {
                      final box = b['box'];
                      return Positioned(
                        left: box[0].clamp(0, 416),
                        top: box[1].clamp(0, 416),
                        width: (box[2] - box[0]).clamp(0, 416),
                        height: (box[3] - box[1]).clamp(0, 416),
                        child: Container(
                          decoration: BoxDecoration(
                            border: Border.all(color: Colors.red, width: 2),
                          ),
                          child: Text(
                            "${b['label']} ${(b['confidence'] * 100).toStringAsFixed(1)}%",
                            style: TextStyle(
                              color: Colors.white,
                              backgroundColor: Colors.red.withOpacity(0.7),
                              fontSize: 12,
                            ),
                          ),
                        ),
                      );
                    }).toList(),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
