import Foundation

/// Runs IndexTTS 2.5 through the mlx-indextts2 runtime — a separate Python
/// environment from Breeze's mlx-audio one, since mlx-audio has no IndexTTS
/// 2.x implementation.  Zero-shot cloning only: the engine conditions on the
/// reference mel, so a reference clip is required and the transcript is
/// ignored.  Each request owns a Python process; no model download occurs.
enum IndexTTSBridge {
    private static var pythonPath: String {
        let configured = ProcessInfo.processInfo.environment["MLX_CORE_INDEX_PYTHON"]
        if let configured, !configured.isEmpty { return configured }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("tts-lab/engines/indextts25-mlx/mlx-indextts2/.venv/bin/python").path
    }

    static func synthesize(modelPath: String, text: String, refAudioPath: String?,
                           refText: String, speed: Double, temperature: Double,
                           outputPath: String,
                           onProgress: ((Int, Int, String) -> Void)? = nil) async throws {
        guard FileManager.default.isExecutableFile(atPath: pythonPath) else {
            throw TTSBridgeRuntime.failure("IndexTTSBridge",
                "The local IndexTTS MLX Python environment was not found at \(pythonPath).")
        }
        guard let refAudioPath, !refAudioPath.isEmpty else {
            throw TTSBridgeRuntime.failure("IndexTTSBridge",
                "IndexTTS 2.5 needs a reference voice clip — it clones from audio, not text.")
        }
        guard speed > 0 else {
            throw TTSBridgeRuntime.failure("IndexTTSBridge", "Speed must be positive.")
        }
        let payload: [String: Any] = [
            "model": modelPath,
            "text": text,
            "ref_audio": refAudioPath,
            "ref_text": refText,
            "speed": speed,
            "temperature": temperature,
            "output": outputPath,
        ]
        try await TTSBridgeRuntime.run(pythonPath: pythonPath, scriptName: "IndexTTSBridge.py",
                                       label: "IndexTTS", payload: payload, onProgress: onProgress)
    }
}
