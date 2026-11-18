import Foundation

struct RecordingMetadata: Encodable, Sendable {
    static var root: URL {
        URL.documentsDirectory.appending(path: "Recordings", directoryHint: .isDirectory)
    }

    let schemaVersion = 2
    let appVersion = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
    let appBuild = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown"
    let sessionID: String
    let requestedAt: Date
    let location: String
    let device: String
    let format: String
    let audioRequested: Bool
    var audioSource = "none; external audio required for lip-sync"
    var audioRoutes: [AudioRouteSnapshot] = []
    var audioTrackCount: Int?
    var stopReason: String?
    var audioCleanupError: String?
    let orientation = "portrait"
    let mirrored = false
    var status = "recording_unfinalized"
    var startedAt: Date?
    var startCallbackUptime: Double?
    var endedAt: Date?
    var durationSeconds: Double?
    var error: String?

    init(location: String, device: String, format: String, audioRequested: Bool = false) {
        self.audioRequested = audioRequested
        requestedAt = Date()
        let stamp = ISO8601DateFormatter().string(from: requestedAt).replacingOccurrences(of: ":", with: "-")
        sessionID = "\(stamp)_\(UUID().uuidString)"
        self.location = location
        self.device = device
        self.format = format
    }

    var videoURL: URL { Self.root.appending(path: sessionID + ".mov") }
    var metadataURL: URL { Self.root.appending(path: sessionID + ".json") }

    func save(in directory: URL = Self.root) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .custom { date, encoder in
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            var container = encoder.singleValueContainer()
            try container.encode(formatter.string(from: date))
        }
        let destination = directory.appending(path: sessionID + ".json")
        try encoder.encode(self).write(to: destination, options: .atomic)
    }
}
