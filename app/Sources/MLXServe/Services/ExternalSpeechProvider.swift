import Foundation

/// Local speech adapters live outside the app bundle. Native models keep their
/// normal weight checks and load lifecycle; adapters never load in mlx-serve.
struct ExternalSpeechProvider {
    let model: String
    let port: UInt16
    let path: String

    static func read(directory: String) -> Self? {
        let file = URL(fileURLWithPath: directory).appendingPathComponent("mlx_tts_plugin.json")
        guard let data = try? Data(contentsOf: file),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = json["plugin"] as? String, !model.isEmpty else { return nil }
        let port = json["port"] as? Int ?? 8096
        let path = json["speech_path"] as? String ?? "/v1/audio/speech/sse"
        guard (1...65535).contains(port), path.hasPrefix("/"), !path.contains("\r"), !path.contains("\n") else { return nil }
        return Self(model: model, port: UInt16(port), path: path)
    }

    @MainActor
    static func resolve(repo: String) -> Self? {
        ServerManager.resolveModelDir(repo: repo).flatMap { read(directory: $0) }
    }

    @MainActor
    static func prepare(repo: String, lanModelId: String?, server: ServerManager) async throws
        -> (port: UInt16, modelId: String, unloadId: String?, path: String) {
        if lanModelId == nil, let provider = resolve(repo: repo) {
            return (provider.port, provider.model, nil, provider.path)
        }
        let (port, modelId, unloadId) = try await server.prepareGenModel(lanModelId: lanModelId, repo: repo)
        return (port, modelId, unloadId, "/v1/audio/speech")
    }
}

/// External engines report completed segments or decoded audio, not native frames.
enum SpeechProgress {
    static func message(_ event: [String: Any]) -> String {
        let stage = event["stage"] as? String ?? "Generating speech"
        var parts = [stage]
        if event["unit"] as? String == "segments", let total = event["total"] as? Int, total > 0 {
            let step = event["step"] as? Int ?? 0
            parts.append("\(step)/\(total) segments")
        }
        if let audio = event["audio_seconds"] as? NSNumber {
            parts.append(String(format: "%.1fs audio generated", audio.doubleValue))
        }
        if let elapsed = event["elapsed_seconds"] as? NSNumber {
            parts.append("\(Int(elapsed.doubleValue))s elapsed")
        }
        return parts.joined(separator: " — ")
    }
}
