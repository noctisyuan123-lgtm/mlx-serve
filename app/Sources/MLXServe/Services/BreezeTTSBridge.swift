import Foundation

/// Runs the installed BF16 Breeze checkpoint in the existing mlx-audio MLX
/// environment.  Each request owns a Python process; no model download occurs.
enum BreezeTTSBridge {
    private static var pythonPath: String {
        let configured = ProcessInfo.processInfo.environment["MLX_CORE_BREEZE_PYTHON"]
        if let configured, !configured.isEmpty { return configured }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("tts-lab/engines/breeze-tts-mlx/venv/bin/python").path
    }

    static func synthesize(modelPath: String, text: String, refAudioPath: String?,
                           refText: String, speed: Double, temperature: Double,
                           outputPath: String,
                           onProgress: ((Int, Int, String) -> Void)? = nil) async throws {
        guard FileManager.default.isExecutableFile(atPath: pythonPath) else {
            throw TTSBridgeRuntime.failure("BreezeTTSBridge",
                "The local Breeze MLX Python environment was not found at \(pythonPath).")
        }
        if refAudioPath != nil && refText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw TTSBridgeRuntime.failure("BreezeTTSBridge",
                "Breeze voice cloning requires the reference transcript.")
        }
        guard speed == 1.0 else {
            throw TTSBridgeRuntime.failure("BreezeTTSBridge",
                "Breeze-TTS 2 does not support playback speed control.")
        }
        let payload: [String: Any] = [
            "model": modelPath,
            "text": text,
            "ref_audio": refAudioPath ?? "",
            "ref_text": refText,
            "speed": speed,
            "temperature": temperature,
            "output": outputPath,
        ]
        try await TTSBridgeRuntime.run(pythonPath: pythonPath, scriptName: "BreezeTTSBridge.py",
                                       label: "Breeze", payload: payload, onProgress: onProgress)
    }
}
