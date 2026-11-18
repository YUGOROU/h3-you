// macOS 14.2+. No microphone, files, routing changes, or cloud fallback.
// Public identifier: Homebrew homebrew-cask Casks/c/comet.rb (ai.perplexity.comet).
import Foundation
import CoreAudio
@preconcurrency import AVFoundation
@preconcurrency import Speech

func emit(_ value: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
       let text = String(data: data, encoding: .utf8) { print(text); fflush(stdout) }
}
struct Failure: Error { let message: String }
func requireOK(_ status: OSStatus) throws {
    if status != noErr { throw Failure(message: "CoreAudio status \(status)") }
}
func address(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                               mElement: kAudioObjectPropertyElementMain)
}
func readValue<T>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector, _ value: inout T) throws {
    var a = address(selector)
    var size = UInt32(MemoryLayout<T>.size)
    try withUnsafeMutablePointer(to: &value) { pointer in
        try requireOK(AudioObjectGetPropertyData(object, &a, 0, nil, &size, pointer))
    }
    guard size == MemoryLayout<T>.size else { throw Failure(message: "ambiguous property size") }
}
struct Process: Equatable {
    let object: AudioObjectID
    let pid: pid_t
    let bundle: String
    var json: [String: Any] { ["object_id": object, "pid": pid, "bundle_id": bundle] }
}
func cometProcesses() throws -> [Process] {
    var a = address(kAudioHardwarePropertyProcessObjectList)
    var size: UInt32 = 0
    try requireOK(AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &size))
    guard size % 4 == 0 else { throw Failure(message: "ambiguous process list") }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / 4)
    let expected = size
    if size == 0 { return [] }
    try ids.withUnsafeMutableBytes { bytes in
        try requireOK(AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &size, bytes.baseAddress!))
    }
    guard size == expected, Set(ids).count == ids.count else { throw Failure(message: "process list changed") }
    var result: [Process] = []
    for id in ids {
        var bundle: CFString? = nil
        try readValue(id, kAudioProcessPropertyBundleID, &bundle)
        guard let bundle else { throw Failure(message: "unidentified audio process") }
        let name = bundle as String
        guard name == "ai.perplexity.comet" || name.hasPrefix("ai.perplexity.comet.") else { continue }
        var pid: pid_t = 0
        try readValue(id, kAudioProcessPropertyPID, &pid)
        guard id != 0, pid > 0, !result.contains(where: { $0.pid == pid }) else {
            throw Failure(message: "ambiguous Comet process")
        }
        result.append(Process(object: id, pid: pid, bundle: name))
    }
    return result.sorted { $0.object < $1.object }
}
// Callback state is shared only under this lock; request append/endAudio are serialized.
final class Sink: @unchecked Sendable {
    private let lock = NSLock()
    private let request: SFSpeechAudioBufferRecognitionRequest
    private let format: AVAudioFormat
    private let preserveEmptyFinal: Bool
    private var ended = false
    private var text = ""
    private var frozenText: String?
    private var textUpdates = 0
    private var final = false
    private var failed = false
    private var buffers = 0
    private var peak: Float = 0
    private var errorCode = 0
    private var errorPresent = false
    private var resultCallbacks = 0
    private var finalCallbacks = 0
    private var totalFrames: UInt64 = 0
    private var endAudioCalls = 0
    private var lastTextTime: ContinuousClock.Instant?
    private var lastVoiceTime: ContinuousClock.Instant?
    private var lastBufferTime: ContinuousClock.Instant?
    private var rms: Float = 0
    init(_ request: SFSpeechAudioBufferRecognitionRequest, _ format: AVAudioFormat,
         preserveEmptyFinal: Bool = false) {
        self.request = request; self.format = format
        self.preserveEmptyFinal = preserveEmptyFinal
    }
    func append(_ input: UnsafePointer<AudioBufferList>) {
        lock.lock(); defer { lock.unlock() }
        guard !ended, !failed else { return }
        let source = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
        let bytes = format.streamDescription.pointee.mBytesPerFrame
        guard bytes > 0, let first = source.first, first.mDataByteSize % bytes == 0 else { failed = true; return }
        let frames = first.mDataByteSize / bytes
        guard frames > 0 else { return }
        guard let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { failed = true; return }
        pcm.frameLength = frames
        let destination = UnsafeMutableAudioBufferListPointer(pcm.mutableAudioBufferList)
        guard source.count == destination.count else { failed = true; return }
        for i in source.indices {
            guard source[i].mNumberChannels == destination[i].mNumberChannels,
                  source[i].mDataByteSize == destination[i].mDataByteSize,
                  let src = source[i].mData, let dst = destination[i].mData else { failed = true; return }
            memcpy(dst, src, Int(source[i].mDataByteSize))
        }
        buffers += 1
        totalFrames += UInt64(frames)
        var bufferPeak: Float = 0
        var sumSquares: Double = 0
        var sampleCount = 0
        if format.commonFormat == .pcmFormatFloat32 {
            for buffer in destination {
                if let data = buffer.mData {
                    let samples = UnsafeBufferPointer(start: data.assumingMemoryBound(to: Float.self), count: Int(buffer.mDataByteSize) / MemoryLayout<Float>.size)
                    for sample in samples where sample.isFinite {
                        bufferPeak = max(bufferPeak, abs(sample))
                        sumSquares += Double(sample) * Double(sample)
                        sampleCount += 1
                    }
                }
            }
        }
        peak = max(peak, bufferPeak)
        rms = sampleCount > 0 ? Float(sqrt(sumSquares / Double(sampleCount))) : 0
        let now = ContinuousClock.now
        lastBufferTime = now
        // Conservative energy gate: continued audible speech delays endpointing even
        // when the recognizer pauses text updates. No audio is retained or emitted.
        if bufferPeak >= 0.01 || rms >= 0.003 { lastVoiceTime = now }
        request.append(pcm)
    }
    func end() {
        lock.lock()
        guard !ended else { lock.unlock(); return }
        ended = true
        endAudioCalls += 1
        lock.unlock()
        // No further appends can start; do not hold the callback lock across Speech.
        request.endAudio()
    }
    func update(_ result: SFSpeechRecognitionResult?, _ error: Error?) {
        lock.lock(); defer { lock.unlock() }
        resultCallbacks += 1
        if let result {
            if result.isFinal { finalCallbacks += 1 }
            let updated = result.bestTranscription.formattedString
            if !(preserveEmptyFinal && result.isFinal && updated.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) {
                if updated != text { lastTextTime = ContinuousClock.now }
                if updated != text { textUpdates += 1 }
                text = updated
            }
            final = result.isFinal
        }
        failed = failed || error != nil
        if let error { errorPresent = true; errorCode = (error as NSError).code }
    }
    var turnEnded: Bool {
        lock.lock(); defer { lock.unlock() }
        let now = ContinuousClock.now
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let lastTextTime, let lastVoiceTime, let lastBufferTime,
              format.commonFormat == .pcmFormatFloat32 else { return false }
        // Require both stable text and quiet audio, with a still-live callback.
        let quiet = now - lastTextTime >= .seconds(1)
            && now - lastVoiceTime >= .seconds(1)
            && now - lastBufferTime < .milliseconds(500)
        // Freeze the SAME text whose stability was checked, under this lock.
        // A callback during capture shutdown must not replace it with newer text.
        if quiet && preserveEmptyFinal && !failed { frozenText = text }
        return quiet && !failed
    }
    // Main-thread consumers assign output revisions; freezing protects the captured
    // endpoint from endAudio callbacks (including an empty Apple final).
    func snapshot(freeze: Bool = false) -> (text: String, updates: Int, failed: Bool) {
        lock.lock(); defer { lock.unlock() }
        if freeze && frozenText == nil { frozenText = text }
        return (frozenText ?? text, textUpdates, failed)
    }
    var result: (String, Bool, Bool) { lock.lock(); defer { lock.unlock() }; return (text, final, failed) }
    var diagnostics: [String: Any] {
        lock.lock(); defer { lock.unlock() }
        let now = ContinuousClock.now
        func age(_ instant: ContinuousClock.Instant?) -> Any {
            guard let instant else { return NSNull() }
            let d = (now - instant).components
            return Double(d.seconds) + Double(d.attoseconds) / 1e18
        }
        // Metadata only: never expose partial text, segments, or raw audio.
        return ["type":"capture_diagnostics", "buffers":buffers, "peak":peak,
                "speech_error_code":errorCode, "error_present":errorPresent,
                "failed":failed, "has_text":!text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "final":final, "result_callbacks":resultCallbacks, "final_callbacks":finalCallbacks,
                "end_audio_calls":endAudioCalls, "total_frames":totalFrames,
                "audio_seconds":Double(totalFrames) / format.sampleRate,
                "sample_rate":format.sampleRate, "channels":format.channelCount,
                "rms":rms, "text_age_seconds":age(lastTextTime),
                "voice_age_seconds":age(lastVoiceTime), "buffer_age_seconds":age(lastBufferTime)]
    }
}

let args = Array(CommandLine.arguments.dropFirst())
guard args.isEmpty || args == ["--check"] || args == ["--authorize"] || args == ["--listen"] || args == ["--listen-turn"] || args == ["--stream-turn"] else {
    emit(["error": "usage: --check (default), --authorize (Speech only), --listen (12 seconds Comet output), --listen-turn (up to 45 seconds + 5 seconds finalization), --stream-turn (opt-in local partials and application-stable endpoint)"]); exit(2)
}
if args == ["--authorize"] {
    let code = await withCheckedContinuation { continuation in
        SFSpeechRecognizer.requestAuthorization { @Sendable status in continuation.resume(returning: status.rawValue) }
    }
    emit(["speech_authorization": code, "captured": false]); exit(code == 3 ? 0 : 3)
}
do {
    let processes = try cometProcesses()
    let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "ja-JP"))
    let streamTurn = args == ["--stream-turn"]
    let listenTurn = args == ["--listen-turn"] || streamTurn
    if args != ["--listen"] && !listenTurn {
        emit(["processes": processes.map(\.json), "warning": "Includes all Comet tabs and matching subprocesses",
              "captured": false, "on_device": recognizer?.supportsOnDeviceRecognition ?? false,
              "speech_authorization": SFSpeechRecognizer.authorizationStatus().rawValue]); exit(0)
    }
    guard !processes.isEmpty, SFSpeechRecognizer.authorizationStatus() == .authorized,
          let recognizer, recognizer.isAvailable, recognizer.supportsOnDeviceRecognition else {
        throw Failure(message: "Comet or authorized on-device ja-JP recognition unavailable")
    }
    var tap: AudioObjectID = 0
    var aggregate: AudioObjectID = 0
    var io: AudioDeviceIOProcID? = nil
    let queue = DispatchQueue(label: "h3.comet.tap")
    // Every acquired resource is released even if a later creation/start fails.
    func cleanup() -> Bool {
        var ok = true
        if let io {
            if AudioDeviceStop(aggregate, io) != noErr { ok = false }
            if AudioDeviceDestroyIOProcID(aggregate, io) != noErr { ok = false }
        }
        io = nil
        queue.sync {}
        if aggregate != 0 { if AudioHardwareDestroyAggregateDevice(aggregate) != noErr { ok = false }; aggregate = 0 }
        if tap != 0 { if AudioHardwareDestroyProcessTap(tap) != noErr { ok = false }; tap = 0 }
        return ok
    }
    defer { _ = cleanup() }
    guard try cometProcesses() == processes else { throw Failure(message: "Comet identities changed") }
    let description = CATapDescription(stereoMixdownOfProcesses: processes.map { $0.object })
    description.isPrivate = true
    description.isExclusive = false
    description.muteBehavior = .unmuted
    try requireOK(AudioHardwareCreateProcessTap(description, &tap))
    var tapFormat = AudioStreamBasicDescription()
    try readValue(tap, kAudioTapPropertyFormat, &tapFormat)
    guard tapFormat.mFormatID == kAudioFormatLinearPCM, let format = AVAudioFormat(streamDescription: &tapFormat) else {
        throw Failure(message: "unsupported tapFormat")
    }
    let specification: [String: Any] = [
        kAudioAggregateDeviceNameKey: "H3 private Comet tap",
        kAudioAggregateDeviceUIDKey: UUID().uuidString,
        kAudioAggregateDeviceIsPrivateKey: true,
        kAudioAggregateDeviceSubDeviceListKey: [], // Never attach physical subdevices; fail if unsupported.
        kAudioAggregateDeviceTapAutoStartKey: true,
        kAudioAggregateDeviceTapListKey: [[kAudioSubTapUIDKey: description.uuid.uuidString]]
    ]
    try requireOK(AudioHardwareCreateAggregateDevice(specification as CFDictionary, &aggregate))
    let request = SFSpeechAudioBufferRecognitionRequest()
    request.requiresOnDeviceRecognition = true
    request.shouldReportPartialResults = listenTurn
    request.taskHint = .dictation
    if listenTurn && format.commonFormat != .pcmFormatFloat32 {
        throw Failure(message: "turn endpointing requires Float32 audio energy measurement")
    }
    let sink = Sink(request, format, preserveEmptyFinal: streamTurn)
    let task = recognizer.recognitionTask(with: request) { @Sendable result, error in sink.update(result, error) }
    defer { sink.end(); task.cancel() }
    try requireOK(AudioDeviceCreateIOProcIDWithBlock(&io, aggregate, queue) { @Sendable _, input, _, _, _ in sink.append(input) })
    guard try cometProcesses() == processes else { throw Failure(message: "Comet identities changed before start") }
    emit(["type": "listening", "seconds": listenTurn ? 45 : 12, "processes": processes.map(\.json),
          "warning": "Includes all Comet tabs and matching subprocesses", "on_device": true])
    try requireOK(AudioDeviceStart(aggregate, io))
    let deadline = ContinuousClock.now.advanced(by: listenTurn ? .seconds(45) : .seconds(12))
    var revision = 0
    var emittedText = ""
    func emitPartial(_ text: String) {
        guard streamTurn, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              text != emittedText else { return }
        revision += 1
        emittedText = text
        emit(["type": "partial", "text": text, "revision": revision,
              "on_device": true, "final": false, "capture_stopped": false])
    }
    var announcedSpeech = false
    var stopReason = "capture_deadline"
    while ContinuousClock.now < deadline && !sink.result.1 && !sink.result.2 {
        if listenTurn && !announcedSpeech && !sink.result.0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            emit(["type": "speech_started", "on_device": true, "text_emitted": false])
            announcedSpeech = true
        }
        if streamTurn { emitPartial(sink.snapshot().text) }
        if listenTurn && sink.turnEnded { stopReason = "quiet_endpoint"; break }
        try await Task.sleep(for: .milliseconds(100))
        guard try cometProcesses() == processes else { throw Failure(message: "Comet identities changed during capture") }
    }
    if sink.result.1 { stopReason = "final" }
    else if sink.result.2 { stopReason = "failure" }
    let captureEndDiagnostics = sink.diagnostics
    let stopped = cleanup()
    let captured = streamTurn ? sink.snapshot(freeze: true) : nil
    sink.end()
    guard stopped else { throw Failure(message: "CoreAudio cleanup failed; transcript suppressed") }
    if streamTurn {
        var diagnostics = sink.diagnostics
        diagnostics["stop_reason"] = stopReason
        diagnostics["capture_end"] = captureEndDiagnostics
        diagnostics["recognition_task_state"] = task.state.rawValue
        diagnostics["finalization_timeout"] = !sink.result.1 && !sink.result.2
        emit(diagnostics)
        guard stopReason == "quiet_endpoint", ContinuousClock.now < deadline,
              let captured, !captured.failed, !sink.result.2,
              !captured.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              (1...160).contains(captured.text.count) else {
            throw Failure(message: "no stable on-device utterance endpoint")
        }
        emitPartial(captured.text)
        emit(["type": "utterance_endpoint", "text": captured.text, "revision": revision,
              "on_device": true, "final": false, "capture_stopped": true,
              "endpoint_source": "application_stable_partial",
              "audio_saved": false, "cloud_fallback": false])
    } else {
    let finish = ContinuousClock.now.advanced(by: .seconds(5))
    while ContinuousClock.now < finish && !sink.result.1 && !sink.result.2 { try await Task.sleep(for: .milliseconds(100)) }
    let result = sink.result
    var diagnostics = sink.diagnostics
    diagnostics["stop_reason"] = stopReason
    diagnostics["capture_end"] = captureEndDiagnostics
    diagnostics["recognition_task_state"] = task.state.rawValue
    diagnostics["finalization_timeout"] = !result.1 && !result.2
    emit(diagnostics)
    guard result.1, !result.2, !result.0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw Failure(message: "no final on-device utterance")
    }
    emit(["type": "transcript", "text": result.0, "final": true, "capture_stopped": true,
          "on_device": true, "audio_saved": false, "cloud_fallback": false])
    }
} catch {
    emit(["error": (error as? Failure)?.message ?? "local recognition failed", "cloud_fallback": false]); exit(1)
}
