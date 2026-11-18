// macOS26: bounded BlackHole-only SpeechAnalyzer probe. No assets/routing/files.
import Foundation
@preconcurrency import AVFoundation
import Speech
import CoreAudio
import AudioToolbox

func emit(_ row: [String:Any]) {
    let data = try! JSONSerialization.data(withJSONObject:row, options:[.sortedKeys])
    print(String(data:data, encoding:.utf8)!); fflush(stdout)
}
func defaultInput() throws -> AudioDeviceID {
    var address = AudioObjectPropertyAddress(mSelector:kAudioHardwarePropertyDefaultInputDevice, mScope:kAudioObjectPropertyScopeGlobal, mElement:kAudioObjectPropertyElementMain)
    var value: AudioDeviceID = 0
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &value) == noErr else { throw ProbeError.route }
    return value
}
func deviceName(_ device: AudioDeviceID) throws -> String {
    var address = AudioObjectPropertyAddress(mSelector:kAudioObjectPropertyName, mScope:kAudioObjectPropertyScopeGlobal, mElement:kAudioObjectPropertyElementMain)
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let status = withUnsafeMutablePointer(to:&value) {
        AudioObjectGetPropertyData(device, &address, 0, nil, &size, $0)
    }
    guard status == noErr, let value else { throw ProbeError.route }
    return value.takeRetainedValue() as String
}
enum ProbeError: Error { case route, unavailable, format, permission }
final class Feed: @unchecked Sendable {
    let lock = NSLock()
    let converter: AVAudioConverter
    let target: AVAudioFormat
    let continuation: AsyncStream<AnalyzerInput>.Continuation
    var buffers = 0
    var frames = 0
    var failures = 0
    var dropped = 0
    var peak: Float = 0
    var peakSamples = 0
    init(_ source: AVAudioFormat, _ target: AVAudioFormat, _ continuation: AsyncStream<AnalyzerInput>.Continuation) throws {
        guard let converter = AVAudioConverter(from:source, to:target) else { throw ProbeError.format }
        self.converter = converter; self.target = target; self.continuation = continuation
    }
    func append(_ buffer: AVAudioPCMBuffer) {
        lock.lock(); defer { lock.unlock() }
        buffers += 1; frames += Int(buffer.frameLength)
        let capacity = AVAudioFrameCount(ceil(Double(buffer.frameLength)*target.sampleRate/buffer.format.sampleRate)+32)
        guard let converted = AVAudioPCMBuffer(pcmFormat:target, frameCapacity:capacity) else { failures += 1; return }
        var error: NSError?
        nonisolated(unsafe) var consumed = false
        nonisolated(unsafe) let input = buffer
        converter.convert(to:converted, error:&error) { _, status in
            if consumed { status.pointee = .noDataNow; return nil }
            consumed = true; status.pointee = .haveData; return input
        }
        guard error == nil, converted.frameLength > 0 else { failures += 1; return }
        if let channels = converted.floatChannelData {
            for c in 0..<Int(target.channelCount) {
                for f in 0..<Int(converted.frameLength) {
                    let value = channels[c][f*converted.stride]
                    if value.isFinite { peak = max(peak, abs(value)); peakSamples += 1 }
                }
            }
        }
        if case .dropped = continuation.yield(AnalyzerInput(buffer:converted)) { dropped += 1 }
    }
    func stats() -> [String:Any] {
        lock.lock(); defer { lock.unlock() }
        return ["buffers":buffers,"input_frames":frames,"conversion_failures":failures,"dropped_buffers":dropped,
                "converted_peak":peakSamples > 0 ? peak as Any : NSNull(), "peak_samples":peakSamples,
                "target_common_format":target.commonFormat.rawValue]
    }
}

let args = Array(CommandLine.arguments.dropFirst())
guard args == ["--check"] || args == ["--listen"] else { emit(["error":"use --check or --listen (12s BlackHole only)"]); exit(2) }
do {
    let locale = Locale(identifier:"ja-JP")
    let installed = await SpeechTranscriber.installedLocales.contains { $0.identifier(.bcp47) == "ja-JP" }
    let device = try defaultInput()
    let name = try deviceName(device)
    if args == ["--check"] {
        emit(["available":SpeechTranscriber.isAvailable,"japanese_installed":installed,
              "default_input":name,"captured":false,"asset_download":false]); exit(0)
    }
    guard name == "BlackHole 2ch" else { throw ProbeError.route }
    guard SpeechTranscriber.isAvailable, installed else { throw ProbeError.unavailable }
    guard AVCaptureDevice.authorizationStatus(for:.audio) == .authorized else { throw ProbeError.permission }
    let transcriber = SpeechTranscriber(locale:locale, transcriptionOptions:[], reportingOptions:[.volatileResults], attributeOptions:[.audioTimeRange])
    let analyzer = SpeechAnalyzer(modules:[transcriber])
    guard let target = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith:[transcriber]) else { throw ProbeError.format }
    let engine = AVAudioEngine()
    let node = engine.inputNode
    guard let unit = node.audioUnit else { throw ProbeError.route }
    // macOS may initially expose a process-local default-device aggregate.
    // Bind this engine only to the explicitly approved BlackHole device.
    var requested = device
    let bindStatus = AudioUnitSetProperty(unit,kAudioOutputUnitProperty_CurrentDevice,kAudioUnitScope_Global,0,
                                         &requested,UInt32(MemoryLayout<AudioDeviceID>.size))
    guard bindStatus == noErr else { emit(["bind_status":bindStatus]); throw ProbeError.route }
    var actual: AudioDeviceID = 0
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    let deviceStatus = AudioUnitGetProperty(unit,kAudioOutputUnitProperty_CurrentDevice,kAudioUnitScope_Global,0,&actual,&size)
    emit(["type":"engine_route_check", "status":deviceStatus, "expected_device_id":device,
          "actual_device_id":actual, "actual_name":(try? deviceName(actual)) ?? "unknown", "captured":false])
    guard deviceStatus == noErr, actual == device else { throw ProbeError.route }
    let source = node.outputFormat(forBus:0)
    let (sequence, continuation) = AsyncStream<AnalyzerInput>.makeStream(bufferingPolicy:.bufferingOldest(64))
    let feed = try Feed(source,target,continuation)
    try await analyzer.prepareToAnalyze(in:target)
    try await analyzer.start(inputSequence:sequence)
    let started = ContinuousClock.now
    let results = Task {
        var count = 0
        var finals = 0
        do {
            for try await result in transcriber.results {
                count += 1; if result.isFinal { finals += 1 }
                emit(["type":"transcript","text":String(result.text.characters),"final":result.isFinal,
                      "elapsed":Double(started.duration(to:.now).components.seconds) + Double(started.duration(to:.now).components.attoseconds) / 1e18,"on_device":true])
            }
        } catch { emit(["type":"recognition_error","error":String(describing:error)]) }
        return (count,finals)
    }
    // Explicit Sendable type prevents the top-level MainActor from leaking
    // into the realtime tap (Swift6 otherwise traps on its callback queue).
    let audioCallback: @Sendable (AVAudioPCMBuffer, AVAudioTime) -> Void = { buffer,_ in feed.append(buffer) }
    node.installTap(onBus:0,bufferSize:4096,format:source,block:audioCallback)
    defer { engine.stop(); node.removeTap(onBus:0); continuation.finish() }
    // Finite in-process safety cutoff, not a background service/auto-retry.
    let cutoff = Task { @MainActor in
        try? await Task.sleep(for:.seconds(30))
        guard !Task.isCancelled else { return }
        engine.stop(); continuation.finish()
        await analyzer.cancelAndFinishNow()
        emit(["error":"30s_hard_cutoff","capture_stopped":true]); exit(4)
    }
    defer { cutoff.cancel() }
    engine.prepare(); try engine.start()
    emit(["type":"listening","seconds":12,"source":name,"source_rate":source.sampleRate,
          "target_rate":target.sampleRate,"api":"SpeechAnalyzer/SpeechTranscriber","audio_saved":false])
    for _ in 0..<120 {
        try await Task.sleep(for:.milliseconds(100))
        guard try defaultInput() == device else { throw ProbeError.route }
    }
    engine.stop(); continuation.finish()
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    let (count,finals) = await results.value
    var stats = feed.stats()
    stats["type"] = "completed"; stats["results"] = count; stats["final_results"] = finals
    stats["capture_stopped"] = !engine.isRunning; stats["audio_saved"] = false
    emit(stats)
} catch {
    emit(["error":String(describing:error),"cloud_fallback":false]); exit(1)
}
