import Foundation
import Testing

private func snapshot(inputs: [String], event: String = "test") -> AudioRouteSnapshot {
    AudioRouteSnapshot(event: event,
        inputs: inputs.map { AudioPortSnapshot(name: "Test mic", type: $0) },
        outputs: [.init(name: "soundcore A25i", type: "BluetoothA2DPOutput")],
        sampleRate: 48000, category: "playAndRecord", mode: "default", microphoneMode: "standard")
}

@Test func onlyBuiltInInputIsAccepted() {
    #expect(snapshot(inputs: ["MicrophoneBuiltIn"]).usesOnlyBuiltInMic)
    #expect(!snapshot(inputs: []).usesOnlyBuiltInMic)
    #expect(!snapshot(inputs: ["BluetoothHFP"]).usesOnlyBuiltInMic)
    #expect(!snapshot(inputs: ["MicrophoneBuiltIn", "BluetoothHFP"]).usesOnlyBuiltInMic)
    #expect(!snapshot(inputs: ["USBAudio"]).usesOnlyBuiltInMic)
}

@Test func videoOnlyMetadataNeverClaimsAudio() {
    let record = RecordingMetadata(location: "indoor", device: "test", format: "test")
    #expect(!record.audioRequested)
    #expect(record.audioRoutes.isEmpty)
    #expect(record.audioTrackCount == nil)
    #expect(record.audioSource.hasPrefix("none"))
}

@Test func failedAudioSetupRetainsDiagnosticTimeline() throws {
    let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    var record = RecordingMetadata(location: "outdoor", device: "test", format: "test", audioRequested: true)
    var observed = snapshot(inputs: ["BluetoothHFP"], event: "after_activation")
    observed.preferredInput = .init(name: "iPhone Microphone", type: "MicrophoneBuiltIn")
    observed.availableInputs = [observed.preferredInput!]
    record.audioRoutes = [observed]
    record.status = "failed_to_start"
    record.error = "Built-in mic unavailable"
    record.audioCleanupError = "Deactivation failed"
    try record.save(in: directory)
    let data = try Data(contentsOf: directory.appending(path: record.sessionID + ".json"))
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(json["audioRequested"] as? Bool == true)
    #expect(json["schemaVersion"] as? Int == 2)
    #expect(json["status"] as? String == "failed_to_start")
    #expect(json["audioCleanupError"] as? String == "Deactivation failed")
    let routes = try #require(json["audioRoutes"] as? [[String: Any]])
    #expect(routes.count == 1)
    #expect(routes.first?["event"] as? String == "after_activation")
    let preferred = try #require(routes.first?["preferredInput"] as? [String: String])
    #expect(preferred["type"] == "MicrophoneBuiltIn")
    let inputs = try #require(routes.first?["inputs"] as? [[String: String]])
    #expect(inputs.first?["type"] == "BluetoothHFP")
    #expect(json["audioTrackCount"] == nil)
}

@Test func routeSummaryIdentifiesHeadsetAndSampleRate() {
    let route = snapshot(inputs: ["MicrophoneBuiltIn"])
    #expect(route.summary.contains("MicrophoneBuiltIn"))
    #expect(route.summary.contains("soundcore A25i"))
    #expect(route.summary.contains("48000"))
}
