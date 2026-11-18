// macOS 14.2+. Explicit --stream only. No Speech, microphones, files or routing changes.
import Foundation
import CoreAudio
import AVFoundation
import Darwin

func emit(_ value: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) {
        FileHandle.standardError.write(data + Data([10]))
    }
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
struct Process: Equatable, Sendable {
    let object: AudioObjectID
    let pid: pid_t
    let bundle: String
    var json: [String: Any] { ["object_id": object, "pid": pid, "bundle_id": bundle] }
}
enum CaptureApp: String {
    case comet, discord, zoom
    var bundle: String {
        switch self {
        case .comet: return "ai.perplexity.comet"
        case .discord: return "com.hnc.Discord"
        case .zoom: return "us.zoom.xos"
        }
    }
    var name: String {
        switch self {
        case .comet: return "Comet"
        case .discord: return "Discord"
        case .zoom: return "Zoom Desktop"
        }
    }
    func matches(_ value: String) -> Bool { value == bundle || value.hasPrefix(bundle + ".") }
    var warning: String {
        let scope = self == .comet
            ? "All Comet tabs and matching application subprocesses; cannot isolate Discord-only audio"
            : "\(name) application and matching subprocesses; bundle selection is a candidate, not runtime verified"
        return scope + "; app scope, not participant isolation"
    }
}
struct Options {
    var app: CaptureApp = .comet
    var stream = false
    var describe = false
    var mute = false
    init(_ arguments: [String]) throws {
        var seen = Set<String>()
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            guard seen.insert(flag).inserted else { throw Failure(message: "duplicate flag: \(flag)") }
            switch flag {
            case "--app":
                index += 1
                guard index < arguments.count, let chosen = CaptureApp(rawValue: arguments[index]) else {
                    throw Failure(message: "--app requires comet|discord|zoom")
                }
                app = chosen
            case "--check": break
            case "--stream": stream = true
            case "--describe": describe = true
            case "--mute-while-capturing": mute = true
            default: throw Failure(message: "unknown flag: \(flag)")
            }
            index += 1
        }
        guard !(seen.contains("--check") && stream) else { throw Failure(message: "--check and --stream conflict") }
        guard !mute || stream else { throw Failure(message: "--mute-while-capturing requires --stream") }
    }
    var warning: String {
        app.warning + (mute ? "; selected app muted only during capture" : "; selected app remains unmuted")
    }
    var json: [String: Any] {
        ["type": "description", "app": app.rawValue, "bundle_id": app.bundle,
         "bundle_match": "exact or dot descendants", "runtime_verified": false,
         "stream_requested": stream, "mute_behavior": mute ? "mutedWhenTapped" : "unmuted",
         "warning": warning, "captured": false, "enumerated": false,
         "encoding": "PCM16LE", "sample_rate": 16000, "channels": 1, "max_seconds": 45]
    }
}

func selectedProcesses(_ app: CaptureApp) throws -> [Process] {
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
        guard app.matches(name) else { continue }
        var pid: pid_t = 0
        try readValue(id, kAudioProcessPropertyPID, &pid)
        guard id != 0, pid > 0, !result.contains(where: { $0.pid == pid }) else {
            throw Failure(message: "ambiguous \(app.name) process")
        }
        result.append(Process(object: id, pid: pid, bundle: name))
    }
    return result.sorted { $0.object < $1.object }
}

// The lock protects every mutable field and slot. Only bounded memcpy occurs
// under it; conversion and stdout writes NEVER occur on the IO callback.
final class PCMQueue: @unchecked Sendable {
    private let lock = NSLock()
    private let slots: [AVAudioPCMBuffer]
    private var head = 0
    private var count = 0
    private var failure: String?
    private var signalNumber: Int32 = 0
    init(format: AVAudioFormat) throws {
        var buffers: [AVAudioPCMBuffer] = []
        for _ in 0..<32 {
            guard let b = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8192) else {
                throw Failure(message: "buffer allocation failed")
            }
            buffers.append(b)
        }
        slots = buffers
    }
    func interrupt(_ value: Int32) { lock.lock(); signalNumber = value; lock.unlock() }
    var status: (String?, Int32) {
        lock.lock(); defer { lock.unlock() }; return (failure, signalNumber)
    }
    func append(_ input: UnsafePointer<AudioBufferList>) {
        lock.lock(); defer { lock.unlock() }
        guard failure == nil, signalNumber == 0 else { return }
        guard count < slots.count else { failure = "dropped buffers: queue full"; return }
        let slot = slots[(head + count) % slots.count]
        let src = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
        let bytes = slot.format.streamDescription.pointee.mBytesPerFrame
        guard bytes > 0, let first = src.first, first.mDataByteSize % bytes == 0 else {
            failure = "invalid input layout"; return
        }
        let frames = first.mDataByteSize / bytes
        if frames == 0 { return }
        guard frames <= slot.frameCapacity else { failure = "dropped oversized buffer"; return }
        slot.frameLength = frames
        let dst = UnsafeMutableAudioBufferListPointer(slot.mutableAudioBufferList)
        guard src.count == dst.count else { failure = "input layout changed"; return }
        for i in src.indices {
            guard src[i].mNumberChannels == dst[i].mNumberChannels,
                  src[i].mDataByteSize == dst[i].mDataByteSize,
                  let source = src[i].mData, let destination = dst[i].mData else {
                failure = "input layout changed"; return
            }
            memcpy(destination, source, Int(src[i].mDataByteSize))
        }
        count += 1
    }
    func pop(into buffer: AVAudioPCMBuffer) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard count > 0 else { return false }
        let slot = slots[head]
        buffer.frameLength = slot.frameLength
        let src = UnsafeMutableAudioBufferListPointer(slot.mutableAudioBufferList)
        let dst = UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList)
        for i in src.indices { memcpy(dst[i].mData!, src[i].mData!, Int(src[i].mDataByteSize)) }
        head = (head + 1) % slots.count; count -= 1
        return true
    }
}

// Converter input ownership stays with this call; lock protects its one-shot
// handoff even if AVFoundation invokes the explicitly Sendable callback elsewhere.
final class ConverterInput: @unchecked Sendable {
    private let lock = NSLock()
    private let buffer: AVAudioPCMBuffer
    private var supplied = false
    init(_ buffer: AVAudioPCMBuffer) { self.buffer = buffer }
    func next(_ status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
        lock.lock(); defer { lock.unlock() }
        if supplied { status.pointee = .noDataNow; return nil }
        supplied = true; status.pointee = .haveData; return buffer
    }
}

func run(options: Options) async throws -> Int32 {
    // Offline configuration inspection must precede every CoreAudio operation.
    if options.describe { emit(options.json); return 0 }
    let app = options.app
    let stream = options.stream
    let processes = try selectedProcesses(app)
    let warning = options.warning
    emit(["type": "selection", "app": app.rawValue, "bundle_id": app.bundle,
          "mute_behavior": options.mute ? "mutedWhenTapped" : "unmuted",
          "processes": processes.map(\.json), "warning": warning,
          "captured": false, "stream_requested": stream])
    guard stream else { return 0 }
    guard !processes.isEmpty else { throw Failure(message: "no \(app.name) processes") }
    var tap: AudioObjectID = 0
    var aggregate: AudioObjectID = 0
    var io: AudioDeviceIOProcID?
    var started = false
    let callbackQueue = DispatchQueue(label: "h3.\(app.rawValue).pcm.io")
    func cleanup() -> Bool {
        var ok = true
        if let proc = io {
            if started && AudioDeviceStop(aggregate, proc) != noErr { ok = false }
            if AudioDeviceDestroyIOProcID(aggregate, proc) != noErr { ok = false }
            io = nil; started = false
        }
        callbackQueue.sync {}
        if aggregate != 0 {
            if AudioHardwareDestroyAggregateDevice(aggregate) != noErr { ok = false }
            aggregate = 0
        }
        if tap != 0 {
            if AudioHardwareDestroyProcessTap(tap) != noErr { ok = false }
            tap = 0
        }
        return ok
    }
    defer { if !cleanup() { emit(["error": "CoreAudio cleanup failed"]) } }
    guard try selectedProcesses(app) == processes else { throw Failure(message: "\(app.name) process list changed") }
    let description = CATapDescription(stereoMixdownOfProcesses: processes.map(\.object))
    description.isPrivate = true; description.isExclusive = false
    description.muteBehavior = options.mute ? .mutedWhenTapped : .unmuted
    try requireOK(AudioHardwareCreateProcessTap(description, &tap))
    var asbd = AudioStreamBasicDescription()
    try readValue(tap, kAudioTapPropertyFormat, &asbd)
    guard asbd.mFormatID == kAudioFormatLinearPCM,
          let format = AVAudioFormat(streamDescription: &asbd),
          format.sampleRate.isFinite, (8000...192000).contains(format.sampleRate),
          (1...2).contains(format.channelCount),
          let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000,
                                     channels: 1, interleaved: true),
          let converter = AVAudioConverter(from: format, to: target),
          let input = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8192),
          let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: 32768) else {
        throw Failure(message: "unsupported conversion format")
    }
    let sink = try PCMQueue(format: format)
    let signals = [SIGINT, SIGTERM].map { number -> any DispatchSourceSignal in
        Darwin.signal(number, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
        source.setEventHandler { @Sendable in sink.interrupt(number) }
        source.resume()
        return source
    }
    defer { for source in signals { source.cancel() } }
    let spec: [String: Any] = [
        kAudioAggregateDeviceNameKey: "H3 private \(app.name) PCM",
        kAudioAggregateDeviceUIDKey: UUID().uuidString,
        kAudioAggregateDeviceIsPrivateKey: true,
        kAudioAggregateDeviceSubDeviceListKey: [],
        kAudioAggregateDeviceTapAutoStartKey: true,
        kAudioAggregateDeviceTapListKey: [[kAudioSubTapUIDKey: description.uuid.uuidString]]
    ]
    try requireOK(AudioHardwareCreateAggregateDevice(spec as CFDictionary, &aggregate))
    try requireOK(AudioDeviceCreateIOProcIDWithBlock(&io, aggregate, callbackQueue) {
        @Sendable _, data, _, _, _ in sink.append(data)
    })
    guard try selectedProcesses(app) == processes else { throw Failure(message: "\(app.name) process list changed") }
    // Nonblocking stdout prevents a stalled reader from delaying stop/cleanup.
    let flags = fcntl(STDOUT_FILENO, F_GETFL)
    guard flags >= 0, fcntl(STDOUT_FILENO, F_SETFL, flags | O_NONBLOCK) == 0 else {
        throw Failure(message: "cannot configure stdout")
    }
    Darwin.signal(SIGPIPE, SIG_IGN)
    defer { _ = fcntl(STDOUT_FILENO, F_SETFL, flags) }
    emit(["type": "stream", "app": app.rawValue, "bundle_id": app.bundle,
          "mute_behavior": options.mute ? "mutedWhenTapped" : "unmuted",
          "encoding": "PCM16LE", "sample_rate": 16000,
          "channels": 1, "max_seconds": 45, "input_sample_rate": format.sampleRate,
          "input_channels": format.channelCount, "warning": warning])
    try requireOK(AudioDeviceStart(aggregate, io)); started = true
    let deadline = ContinuousClock.now.advanced(by: .seconds(45))
    var frames: UInt64 = 0
    var peakPCM16 = 0
    var nonzeroSamples: UInt64 = 0
    var sumSquares = 0.0
    while ContinuousClock.now < deadline && frames < 720000 {
        let state = sink.status
        if let error = state.0 { throw Failure(message: error) }
        if state.1 != 0 { break }
        guard try selectedProcesses(app) == processes else { throw Failure(message: "\(app.name) process list changed") }
        if !sink.pop(into: input) { try await Task.sleep(for: .milliseconds(5)); continue }
        // Input callback is synchronous and converter-local; no shared mutation.
        let provider = ConverterInput(input)
        var conversionError: NSError?
        repeat {
            output.frameLength = 0
            let status = converter.convert(to: output, error: &conversionError) { @Sendable _, status in
                provider.next(status)
            }
            guard conversionError == nil, status != .error else {
                throw Failure(message: "audio conversion failed")
            }
            if output.frameLength > 0 {
                guard let samples = output.int16ChannelData?[0] else { throw Failure(message: "missing PCM16 output") }
                // Apple Silicon/Intel macOS are little-endian.
                let allowed = min(UInt64(output.frameLength), 720000 - frames)
                let size = Int(allowed) * 2
                for index in 0..<Int(allowed) {
                    let value = Int(samples[index])
                    peakPCM16 = max(peakPCM16, abs(value))
                    if value != 0 { nonzeroSamples += 1 }
                    sumSquares += Double(value) * Double(value)
                }
                let written = Darwin.write(STDOUT_FILENO, samples, size)
                guard written == size else { throw Failure(message: "dropped output: stdout blocked, partial write or closed") }
                frames += allowed
            }
            if status != .haveData { break }
            guard ContinuousClock.now < deadline, frames < 720000, sink.status.1 == 0 else { break }
        } while true
    }
    guard cleanup() else { throw Failure(message: "CoreAudio cleanup failed") }
    if let failure = sink.status.0 { throw Failure(message: failure) }
    guard try selectedProcesses(app) == processes else { throw Failure(message: "\(app.name) process list changed") }
    let interrupted = sink.status.1
    emit(["type": "stopped", "app": app.rawValue, "bundle_id": app.bundle,
          "mute_behavior": options.mute ? "mutedWhenTapped" : "unmuted",
          "frames": frames, "signal": interrupted,
          "peak_pcm16": peakPCM16, "nonzero_samples": nonzeroSamples,
          "rms_pcm16": frames > 0 ? sqrt(sumSquares / Double(frames)) : 0,
          "reason": interrupted == 0 ? "deadline" : "signal", "cleanup": true])
    return interrupted == 0 ? 0 : 128 + interrupted
}

let options: Options
do { options = try Options(Array(CommandLine.arguments.dropFirst())) }
catch {
    emit(["error": (error as? Failure)?.message ?? "invalid options"])
    exit(2)
}
do { exit(try await run(options: options)) }
catch {
    emit(["error": (error as? Failure)?.message ?? "local PCM stream failed", "stream_valid": false])
    exit(1)
}
