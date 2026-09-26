import Foundation

/// Shared machinery for the local Python TTS bridges (Breeze, IndexTTS): one
/// process per request, stdout pumped for progress lines, stderr buffered for
/// the error message. Each runtime is its own Python environment, and the one
/// GPU serializes them anyway — so one shared gate keeps two bridges from
/// running at once no matter which modality the user pressed.
enum TTSBridgeRuntime {
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
            defer { lock.unlock() }
            active = false
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

    static func failure(_ label: String, _ message: String) -> NSError {
        NSError(domain: label, code: 1,
                userInfo: [NSLocalizedDescriptionKey: message])
    }

    /// Runs one bridge script with the JSON request on stdin and mirrors its
    /// stdout progress lines into `onProgress` (background thread — hop before
    /// touching UI state). Throws with the stderr text when the process fails.
    static func run(pythonPath: String, scriptName: String, label: String,
                    payload: [String: Any],
                    onProgress: ((Int, Int, String) -> Void)? = nil) async throws {
        guard gate.enter() else {
            throw failure(label, "Another \(label) generation is already running.")
        }
        defer { gate.leave() }
        guard let script = Bundle.main.resourceURL?.appendingPathComponent(scriptName),
              FileManager.default.isReadableFile(atPath: script.path) else {
            throw failure(label, "\(scriptName) is missing from the app bundle.")
        }
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
                            .appendingPathComponent("tts-bridge-stderr-\(UUID().uuidString).log")
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
                            pumpProgress(stdout, into: onProgress)
                        }
                        stdin.fileHandleForWriting.write(input)
                        stdin.fileHandleForWriting.closeFile()
                        process.waitUntilExit()
                        if run.wasCancelled { throw CancellationError() }
                        let errorText = (try? String(contentsOf: stderrURL, encoding: .utf8))?
                            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                        guard process.terminationStatus == 0 else {
                            throw failure(label, errorText.isEmpty ? "\(label) synthesis failed." : errorText)
                        }
                        guard let size = try? FileManager.default.attributesOfItem(atPath: payload["output"] as! String)[.size] as? Int,
                              size > 44 else {
                            throw failure(label, "The bridge returned no WAV file.")
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
}
