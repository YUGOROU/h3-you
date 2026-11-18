import Foundation

struct AudioPortSnapshot: Codable, Sendable, Equatable {
    let name: String
    let type: String
}

/// Route observed by H3 Camera, not ChatGPT's private audio session.
struct AudioRouteSnapshot: Codable, Sendable {
    var date = Date()
    let event: String
    let inputs: [AudioPortSnapshot]
    let outputs: [AudioPortSnapshot]
    let sampleRate: Double
    let category: String
    let mode: String
    let microphoneMode: String
    var availableInputs: [AudioPortSnapshot] = []
    var preferredInput: AudioPortSnapshot?
    var categoryOptions: UInt = 0

    var usesOnlyBuiltInMic: Bool {
        inputs.count == 1 && inputs[0].type == "MicrophoneBuiltIn"
    }

    var summary: String {
        let input = inputs.map { "\($0.name) [\($0.type)]" }.joined(separator: ", ")
        let output = outputs.map { "\($0.name) [\($0.type)]" }.joined(separator: ", ")
        return "入力: \(input.isEmpty ? "なし" : input)\n出力: \(output.isEmpty ? "なし" : output)\n\(Int(sampleRate)) Hz / Mic mode: \(microphoneMode)"
    }
}
