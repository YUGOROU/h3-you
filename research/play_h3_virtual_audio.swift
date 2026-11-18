// macOS-only, generated diagnostic speech → BlackHole. No recording/global route changes.
import Foundation
import AVFoundation
import CoreAudio
import CryptoKit

func emit(_ value: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
}
func scalar(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> UInt32? {
    var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var value: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value) == noErr else { return nil }
    return value
}
func string(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var value: Unmanaged<CFString>? = nil
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value) == noErr else { return nil }
    return value?.takeRetainedValue() as String?
}
func blackHole() -> (AudioDeviceID, String)? {
    var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr else { return nil }
    var devices = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &devices) == noErr else { return nil }
    for device in devices {
        guard string(device, kAudioObjectPropertyName) == "BlackHole 2ch",
              scalar(device, kAudioDevicePropertyDeviceIsAlive) == 1,
              let uid = string(device, kAudioDevicePropertyDeviceUID) else { continue }
        return (device, uid)
    }
    return nil
}

enum Failure: Error { case unavailable, invalidReceipt, invalidAudio, routeChanged, playbackFailed }
/// Private working area. H3_PRIVATE_ROOT overrides; otherwise ~/Downloads/H3Personal/<session>.
func privateRoot() -> URL {
    let env = ProcessInfo.processInfo.environment
    if let override = env["H3_PRIVATE_ROOT"], !override.isEmpty {
        return URL(fileURLWithPath: override, isDirectory: true).standardizedFileURL.resolvingSymlinksInPath()
    }
    let base = env["H3_PRIVATE_BASE"].flatMap { $0.isEmpty ? nil : $0 } ?? "Downloads/H3Personal"
    let session = env["H3_PRIVATE_SESSION"].flatMap { $0.isEmpty ? nil : $0 } ?? "2026-10-12/production-715"
    let home = FileManager.default.homeDirectoryForCurrentUser
    return home.appendingPathComponent(base).appendingPathComponent(session).standardizedFileURL.resolvingSymlinksInPath()
}

let args = Array(CommandLine.arguments.dropFirst())
let freshTurn = args.count == 2 && args[0] == "--play-turn"
guard args.isEmpty || args == ["--check"] || args == ["--play"] || freshTurn else {
    emit(["error": "usage: --check, --play (diagnostic), --play-turn PRIVATE_VERIFIED_DIRECTORY"]); exit(2)
}
do {
    guard let (device, uid) = blackHole() else {
        emit(["available": false, "played": false, "next": "BlackHole installed but CoreAudio endpoint absent; follow installer reboot instruction"])
        exit(args == ["--play"] || freshTurn ? 1 : 0)
    }
    if args != ["--play"] && !freshTurn {
        emit(["available": true, "device": "BlackHole 2ch", "device_uid": uid, "played": false, "global_route_changed": false]); exit(0)
    }
    let privateArea = privateRoot()
    let file: URL
    let expected: String
    if freshTurn {
        let root = URL(fileURLWithPath: args[1]).resolvingSymlinksInPath()
        guard root.path.hasPrefix(privateArea.path + "/") else { throw Failure.invalidReceipt }
        let receiptBytes = try Data(contentsOf: root.appendingPathComponent("receipt.json"))
        let report = try JSONSerialization.jsonObject(with: receiptBytes) as? [String: Any]
        let download = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("download-receipt.json"))) as? [String: Any]
        let receiptHash = SHA256.hash(data: receiptBytes).map { String(format: "%02x", $0) }.joined()
        guard report?["schema"] as? String == "h3-director-new-irodori-turn-v1",
              report?["source_revision"] as? String == "89f9d8fbd4d51ea019867ee1197725ede1df13c5",
              report?["model_revision"] as? String == "2e0c55428ce97268a507f1feeb2478f8d9148e8b",
              report?["codec_revision"] as? String == "47376ee24834d7a05a48ebabfe3cde29b3c5e214",
              report?["director_model"] as? String == "Qwen/Qwen3.8-27B:cerebras",
              report?["input_origin"] as? String == "user_text_chat_not_discord_asr",
              report?["watermark_applied"] as? Bool == true,
              report?["synthesis_count"] as? Int == 1,
              report?["generated_only"] as? Bool == true,
              report?["h3_executed"] as? Bool == false,
              let revision = download?["output_revision"] as? String,
              revision.range(of: "^[0-9a-f]{40}$", options: .regularExpression) != nil,
              let files = download?["files"] as? [String: String],
              files["receipt.json"] == receiptHash,
              let sha = report?["sha256"] as? String, files["reply.wav"] == sha else { throw Failure.invalidReceipt }
        expected = sha
        file = root.appendingPathComponent("reply.wav").resolvingSymlinksInPath()
        guard file.path.hasPrefix(root.path + "/") else { throw Failure.invalidReceipt }
    } else {
        let root = privateArea.appendingPathComponent("gpu-hf-irodori-diagnostic-v1/verified-audio")
        let report = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("diagnostic-report.json"))) as? [String: Any]
        guard report?["source_revision"] as? String == "89f9d8fbd4d51ea019867ee1197725ede1df13c5",
              report?["model_revision"] as? String == "2e0c55428ce97268a507f1feeb2478f8d9148e8b",
              report?["watermark_required"] as? Bool == true,
              let cases = report?["cases"] as? [[String: Any]],
              let selected = cases.first(where: { $0["case"] as? String == "reference-30s" }),
              selected["file"] as? String == "reference-30s.wav",
              let sha = selected["sha256"] as? String else { throw Failure.invalidReceipt }
        expected = sha; file = root.appendingPathComponent("reference-30s.wav")
    }
    let bytes = try Data(contentsOf: file)
    guard bytes.count < 32_000_000, SHA256.hash(data: bytes).map({ String(format: "%02x", $0) }).joined() == expected else { throw Failure.invalidAudio }
    let previousDefault = scalar(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultOutputDevice)
    let audio = try AVAudioFile(forReading: file)
    let duration = Double(audio.length) / audio.processingFormat.sampleRate
    guard duration > 0, duration <= 30 else { throw Failure.invalidAudio }
    let engine = AVAudioEngine()
    defer { engine.stop() }
    let output = engine.outputNode
    try output.auAudioUnit.setDeviceID(device)
    guard output.auAudioUnit.deviceID == device,
          string(device, kAudioDevicePropertyDeviceUID) == uid,
          scalar(device, kAudioDevicePropertyDeviceIsAlive) == 1 else { throw Failure.routeChanged }
    let player = AVAudioPlayerNode()
    engine.attach(player)
    engine.connect(player, to: engine.mainMixerNode, format: audio.processingFormat)
    engine.prepare()
    try engine.start()
    let completed = DispatchSemaphore(value: 0)
    player.scheduleFile(audio, at: nil, completionCallbackType: .dataPlayedBack) { _ in completed.signal() }
    player.play()
    defer { player.stop() }
    let deadline = Date().addingTimeInterval(35)
    while true {
        guard Date() < deadline, output.auAudioUnit.deviceID == device,
              scalar(device, kAudioDevicePropertyDeviceIsAlive) == 1 else { engine.stop(); throw Failure.routeChanged }
        if completed.wait(timeout: .now()) == .success { break }
        RunLoop.current.run(until: Date().addingTimeInterval(0.02))
    }
    emit(["available": true, "played": true, "device": "BlackHole 2ch", "generated_diagnostic_only": !freshTurn,
          "interactive_tts": freshTurn, "input_origin": freshTurn ? "user_text_chat_not_discord_asr" : "diagnostic", "discord_reception_verified": false,
          "default_output_unchanged": previousDefault == scalar(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultOutputDevice)])
} catch {
    emit(["error": "virtual audio check/playback failed; no fallback to speakers", "played": false]); exit(1)
}
