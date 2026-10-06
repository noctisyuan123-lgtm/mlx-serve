import XCTest
@testable import MLXCore

final class ExternalSpeechProviderTests: XCTestCase {
    func testProgressPreservesEngineUnitsAndHeartbeatTime() {
        let first = SpeechProgress.message(["stage": "Generating speech", "unit": "segments",
                                            "step": 1, "total": 3, "elapsed_seconds": 8.0])
        let later = SpeechProgress.message(["stage": "Generating speech", "unit": "segments",
                                            "step": 1, "total": 3, "elapsed_seconds": 9.0])
        XCTAssertTrue(first.contains("1/3 segments"))
        XCTAssertTrue(first.contains("8s elapsed"))
        XCTAssertTrue(later.contains("9s elapsed"))
        XCTAssertNotEqual(first, later)
        XCTAssertTrue(SpeechProgress.message(["audio_seconds": 2.5]).contains("2.5s audio generated"))
    }
    func testAdapterReadyWithoutWeightsAndNativePartialStillRejected() throws {
        let root = NSTemporaryDirectory() + "speech-provider-" + UUID().uuidString
        let dir = root + "/owner/model"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: root) }
        try Data("{}".utf8).write(to: URL(fileURLWithPath: dir + "/config.json"))
        let audio = MediaBundle.tts(repo: "owner/model", displayName: "Speech", sizeGB: 0).components[0]
        XCTAssertFalse(DownloadManager.componentReady(audio, modelsRoot: root))
        let config = URL(fileURLWithPath: dir + "/mlx_tts_plugin.json")
        try Data(#"{"plugin":"breeze","port":8096}"#.utf8).write(to: config)
        XCTAssertTrue(DownloadManager.componentReady(audio, modelsRoot: root))
        let native = MediaComponent(repo: "owner/model", selection: .chatDefault, readyMarkers: ["config.json"])
        XCTAssertFalse(DownloadManager.componentReady(native, modelsRoot: root))
        XCTAssertEqual(ExternalSpeechProvider.read(directory: dir)?.model, "breeze")
        try Data(#"{"plugin":"breeze","port":70000}"#.utf8).write(to: config)
        XCTAssertFalse(DownloadManager.componentReady(audio, modelsRoot: root))
        try Data(#"{"plugin":""}"#.utf8).write(to: config)
        XCTAssertNil(ExternalSpeechProvider.read(directory: dir))
    }
}
