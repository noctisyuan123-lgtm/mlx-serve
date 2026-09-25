import Foundation

/// Runs the installed BF16 Breeze checkpoint in the existing mlx-audio MLX
/// environment.  Each request owns a Python process; no model download occurs.
enum BreezeTTSBridge {
    private final class ProcessGate: @unchecked Sendable {
        private let lock = NSLock()
        private var active = false

        func enter() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            if active { return false }
            active = true
            return true
        }

        func leave() {
            lock.lock()
            active = false
            lock.unlock()
        }
    }

    private static let gate = ProcessGate()

    private final class ProcessRun: @unchecked Sendable {
        private let lock = NSLock()
        private var process: Process?
        private var cancelled = false

        func launch(_ next: Process) throws {
            lock.lock()
            defer { lock.unlock() }
            if cancelled { throw CancellationError() }
            try next.run()
            process = next
        }

        func cancel() {
            lock.lock()
            defer { lock.unlock() }
            cancelled = true
            if let process, process.isRunning { process.terminate() }
        }

        var wasCancelled: Bool {
            lock.lock()
            defer { lock.unlock() }
            return cancelled
        }
    }

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
            throw failure("The local Breeze MLX Python environment was not found at \(pythonPath).")
        }
        if refAudioPath != nil && refText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw failure("Breeze voice cloning requires the reference transcript.")
        }
        guard speed == 1.0 else {
            throw failure("Breeze-TTS 2 does not support playback speed control.")
        }
        guard gate.enter() else {
            throw failure("Another Breeze-TTS 2 generation is already running.")
        }
        defer { gate.leave() }
        let script = Bundle.main.resourceURL?.appendingPathComponent("BreezeTTSBridge.py")
        guard let script, FileManager.default.isReadableFile(atPath: script.path) else {
            throw failure("BreezeTTSBridge.py is missing from the app bundle.")
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
        let input = try JSONSerialization.data(withJSONObject: payload)
        let run = ProcessRun()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        let process = Process()
                        process.executableURL = URL(fileURLWithPath: pythonPath)
                        process.arguments = [script.path]
                        let stdin = Pipe()
                        let stderrURL = FileManager.default.temporaryDirectory
                            .appendingPathComponent("breeze-stderr-\(UUID().uuidString).log")
                        try Data().write(to: stderrURL)
                        defer { try? FileManager.default.removeItem(at: stderrURL) }
                        let stderr = try FileHandle(forWritingTo: stderrURL)
                        defer { try? stderr.close() }
                        process.standardInput = stdin
                        // The bridge mirrors the media endpoints' SSE envelope on
                        // stdout (one JSON object per line); stderr stays the
                        // error channel, read once the process has exited.
                        let stdout = Pipe()
                        process.standardOutput = onProgress == nil ? FileHandle.nullDevice : stdout
                        process.standardError = stderr
                        try run.launch(process)
                        if let onProgress {
                            Self.pumpProgress(stdout, into: onProgress)
                        }
                        stdin.fileHandleForWriting.write(input)
                        stdin.fileHandleForWriting.closeFile()
                        process.waitUntilExit()
                        if run.wasCancelled { throw CancellationError() }
                        let errorText = (try? String(contentsOf: stderrURL, encoding: .utf8))?
                            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                        guard process.terminationStatus == 0 else {
                            throw failure(errorText.isEmpty ? "Breeze synthesis failed." : errorText)
                        }
                        guard let size = try? FileManager.default.attributesOfItem(atPath: outputPath)[.size] as? Int,
                              size > 44 else {
                            throw failure("Breeze returned no WAV file.")
                        }
                        continuation.resume()
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        } onCancel: {
            run.cancel()
        }
    }

    /// Reads the bridge's stdout one JSON object per line and forwards
    /// progress events to `sink`. Runs on a background queue for the life of
    /// the process; EOF (empty read) tears the handler down. Non-JSON lines
    /// are skipped rather than fatal — anything the runtime prints on stdout
    /// must not kill a synthesis that would otherwise succeed.
    private static func pumpProgress(_ pipe: Pipe,
                                     into sink: @escaping (Int, Int, String) -> Void) {
        var buffer = Data()
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            buffer.append(data)
            while let nl = buffer.firstIndex(of: 0x0A) {
                let line = buffer.subdata(in: buffer.startIndex..<nl)
                buffer.removeSubrange(buffer.startIndex...nl)
                guard let obj = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any] else {
                    continue
                }
                if case .progress(let step, let total, let stage) = MediaSSE.classify(obj) {
                    sink(step, total, stage)
                }
            }
        }
    }

    private static func failure(_ message: String) -> NSError {
        NSError(domain: "BreezeTTSBridge", code: 1,
                userInfo: [NSLocalizedDescriptionKey: message])
    }
}
