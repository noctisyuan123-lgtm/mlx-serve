import Foundation

/// Runs dots.tts (rednote-hilab) through the dots-tts-mlx runtime — its own
/// Python environment, separate from the mlx-audio (Breeze) and mlx-indextts2
/// (IndexTTS) ones.  In-context cloning needs BOTH the reference clip and its
/// transcript; the runtime wants an explicit language code, which the bridge
/// guesses from the script.  Each request owns a Python process; no model
/// download occurs.
enum DotsTTSBridge {
    private static var pythonPath: String {
        let configured = ProcessInfo.processInfo.environment["MLX_CORE_DOTS_PYTHON"]
        if let configured, !configured.isEmpty { return configured }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("tts-lab/engines/dots-tts-mlx/venv/bin/python").path
    }

    static func synthesize(modelPath: String, text: String, refAudioPath: String?,
                           refText: String, speed: Double, temperature: Double,
                           outputPath: String,
                           onProgress: ((Int, Int, String) -> Void)? = nil) async throws {
        guard FileManager.default.isExecutableFile(atPath: pythonPath) else {
            throw TTSBridgeRuntime.failure("DotsTTSBridge",
                "The local dots.tts MLX Python environment was not found at \(pythonPath).")
        }
        guard let refAudioPath, !refAudioPath.isEmpty else {
            throw TTSBridgeRuntime.failure("DotsTTSBridge",
                "dots.tts requires a reference voice clip — it clones from audio, not text.")
        }
        guard refText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false else {
            throw TTSBridgeRuntime.failure("DotsTTSBridge",
                "dots.tts voice cloning requires the reference transcript.")
        }
        guard speed == 1.0 else {
            throw TTSBridgeRuntime.failure("DotsTTSBridge",
                "dots.tts bridge does not support playback speed control.")
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
        try await TTSBridgeRuntime.run(pythonPath: pythonPath, scriptName: "DotsTTSBridge.py",
                                       label: "dots.tts", payload: payload, onProgress: onProgress)
    }
}
