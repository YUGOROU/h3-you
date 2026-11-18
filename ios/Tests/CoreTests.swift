import Foundation
import Testing

@Test func uniqueSafeSessionNames() {
    let records = (0..<100).map { _ in RecordingMetadata(location: "indoor", device: "test", format: "1080p") }
    #expect(Set(records.map(\.sessionID)).count == records.count)
    for record in records {
        #expect(!record.sessionID.contains(":"))
        #expect(record.videoURL.deletingPathExtension() == record.metadataURL.deletingPathExtension())
        #expect(record.status == "recording_unfinalized")
    }
}

@Test func metadataPreservesMillisecondsAndFinalization() throws {
    let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    var record = RecordingMetadata(location: "outdoor", device: "iPhone test", format: "1920×1080 / 30fps")
    try record.save(in: directory)
    let url = directory.appending(path: record.sessionID + ".json")
    let initial = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    #expect(initial["status"] as? String == "recording_unfinalized")
    record.startedAt = Date(timeIntervalSince1970: 1_700_000_000.125)
    record.durationSeconds = 600.5
    record.status = "completed"
    try record.save(in: directory)
    let final = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    #expect(final["status"] as? String == "completed")
    #expect(final["startedAt"] as? String == "2023-11-14T22:13:20.125Z")
    #expect(final["durationSeconds"] as? Double == 600.5)
    #expect(final["mirrored"] as? Bool == false)
    #expect((final["audioSource"] as? String)?.hasPrefix("none") == true)
    #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path).count == 1)
}

@Test func metadataWriteFailureIsNotSilenced() {
    let missing = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString).appending(path: "missing")
    let record = RecordingMetadata(location: "indoor", device: "test", format: "test")
    #expect(throws: (any Error).self) { try record.save(in: missing) }
}

private actor CommandLog {
    var entries: [Int] = []
    func append(_ value: Int) async {
        await Task.yield()
        entries.append(value)
    }
}

@Test @MainActor func commandsPreserveOrderAcrossSuspension() async {
    let commands = SerialCommands()
    let log = CommandLog()
    for value in 0..<100 {
        commands.enqueue { await log.append(value) }
    }
    let final = commands.enqueue { await log.append(100) }
    await final.value
    #expect(await log.entries == Array(0...100))
}
