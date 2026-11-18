// Explicit opt-in macOS default INPUT change only. No capture or playback.
import Foundation
import CoreAudio

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
func targetDevice() -> AudioDeviceID? {
    var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr else { return nil }
    var devices = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &devices) == noErr else { return nil }
    return devices.first { string($0, kAudioObjectPropertyName) == "BlackHole 2ch" && scalar($0, kAudioDevicePropertyDeviceIsAlive) == 1 }
}
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
guard args.isEmpty || args == ["--check"] || args == ["--apply"] else { emit(["error": "use --check (default) or --apply"]); exit(2) }
let system = AudioObjectID(kAudioObjectSystemObject)
guard let target = targetDevice(), let previous = scalar(system, kAudioHardwarePropertyDefaultInputDevice),
      let previousUID = string(previous, kAudioDevicePropertyDeviceUID),
      let targetUID = string(target, kAudioDevicePropertyDeviceUID),
      let previousOutput = scalar(system, kAudioHardwarePropertyDefaultOutputDevice),
      let previousSystemOutput = scalar(system, kAudioHardwarePropertyDefaultSystemOutputDevice)
else { emit(["error": "device metadata unavailable", "changed": false]); exit(1) }
if args != ["--apply"] {
    emit(["available": true, "default_input_is_blackhole": previous == target, "changed": false]); exit(0)
}
do {
    let folder = privateRoot().appendingPathComponent("audio-routing")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    let receipt = folder.appendingPathComponent("default-input-\(UUID().uuidString).json")
    // Save previous input UID BEFORE changing anything, to support manual restoration.
    var record: [String: Any] = ["before_input_uid": previousUID, "target_input_uid": targetUID,
        "operation": "default_input_only", "capture_started": false, "change_verified": false]
    try JSONSerialization.data(withJSONObject: record).write(to: receipt, options: [.atomic])
    try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: receipt.path)
    var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultInputDevice, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var desired = target
    let status = AudioObjectSetPropertyData(system, &address, 0, nil, UInt32(MemoryLayout<AudioDeviceID>.size), &desired)
    let verified = status == noErr && scalar(system, kAudioHardwarePropertyDefaultInputDevice) == target
    let outputsUnchanged = scalar(system, kAudioHardwarePropertyDefaultOutputDevice) == previousOutput && scalar(system, kAudioHardwarePropertyDefaultSystemOutputDevice) == previousSystemOutput
    record["status"] = status; record["change_verified"] = verified; record["outputs_unchanged"] = outputsUnchanged
    try JSONSerialization.data(withJSONObject: record).write(to: receipt, options: [.atomic])
    try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: receipt.path)
    emit(["default_input_is_blackhole": verified, "outputs_unchanged": outputsUnchanged, "capture_started": false, "receipt": receipt.path])
    if !verified || !outputsUnchanged { exit(1) }
} catch { emit(["error": "routing change/receipt failed; inspect private routing receipt", "capture_started": false]); exit(1) }
