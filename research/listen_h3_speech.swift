// Bounded local Mac built-in-microphone ASR. No cloud fallback, files, or routing changes.
import Foundation
@preconcurrency import AVFoundation
@preconcurrency import Speech
import CoreAudio

func emit(_ value: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
       let text = String(data: data, encoding: .utf8) { print(text) }
}
final class State: NSObject, AVCaptureAudioDataOutputSampleBufferDelegate, @unchecked Sendable {
    // All mutable state/request writes cross callback queues only under this lock.
    private let lock = NSLock()
    private let request: SFSpeechAudioBufferRecognitionRequest?
    private var ended = false
    private var text = ""
    private var final = false
    private var failed = false
    private var buffers = 0
    private var frames = 0
    private var measuredSamples = 0
    private var pcmCopyFailures = 0
    private var inputFormat: [String: Any] = [:]
    private var peak: Double = 0
    private var callbacks = 0
    private var partialCallbacks = 0
    private var maxTextCharacters = 0
    private var errorDomain = ""
    private var errorCode = 0
    var diagnostics: [String: Any] {
        lock.lock(); defer { lock.unlock() }
        return ["type":"capture_diagnostics", "buffers":buffers, "frames":frames,
                "measured_samples":measuredSamples, "peak":peak,
                "pcm_copy_failures":pcmCopyFailures, "input_format":inputFormat,
                "recognition_callbacks":callbacks, "partial_callbacks":partialCallbacks,
                "max_text_characters":maxTextCharacters, "final":final,
                "error_domain":errorDomain, "error_code":errorCode]
    }
    init(_ request: SFSpeechAudioBufferRecognitionRequest?) { self.request = request }
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        lock.lock(); defer { lock.unlock() }
        if !ended {
            buffers += 1
            frames += CMSampleBufferGetNumSamples(sampleBuffer)
            // Read frame/channel layout via CoreMedia, not CMBlockBuffer byte
            // length: storage may contain padding or a nontrivial PCM layout.
            if let description = CMSampleBufferGetFormatDescription(sampleBuffer) {
                let format = AVAudioFormat(cmAudioFormatDescription: description)
                let count = CMSampleBufferGetNumSamples(sampleBuffer)
                let asbd = format.streamDescription.pointee
                inputFormat = ["sample_rate":asbd.mSampleRate, "channels":asbd.mChannelsPerFrame,
                               "bits_per_channel":asbd.mBitsPerChannel, "format_flags":asbd.mFormatFlags,
                               "bytes_per_frame":asbd.mBytesPerFrame, "interleaved":format.isInterleaved]
                if count > 0, count <= Int(Int32.max), format.commonFormat == .pcmFormatFloat32,
                   let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)) {
                    pcm.frameLength = AVAudioFrameCount(count)
                    let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
                        sampleBuffer, at: 0, frameCount: Int32(count), into: pcm.mutableAudioBufferList)
                    if status == noErr, let channels = pcm.floatChannelData {
                        for channel in 0..<Int(format.channelCount) {
                            for frame in 0..<count {
                                let value = Double(channels[channel][frame * pcm.stride])
                                if value.isFinite { peak = max(peak, abs(value)); measuredSamples += 1 }
                            }
                        }
                    } else { pcmCopyFailures += 1 }
                } else { pcmCopyFailures += 1 }
            } else { pcmCopyFailures += 1 }
            request?.appendAudioSampleBuffer(sampleBuffer)
        }
    }
    func endAudio() { lock.lock(); defer { lock.unlock() }; ended = true; request?.endAudio() }
    func update(_ value: String?, _ isFinal: Bool, _ error: NSError?) {
        lock.lock(); defer { lock.unlock() }
        callbacks += 1
        if let value {
            text = value
            maxTextCharacters = max(maxTextCharacters, value.count)
            if !isFinal { partialCallbacks += 1 }
        }
        if let error { errorDomain = error.domain; errorCode = error.code }
        final = final || isFinal; failed = failed || error != nil
    }
    var result: (String, Bool, Bool) { lock.lock(); defer { lock.unlock() }; return (text, final, failed) }
}
let arguments = Array(CommandLine.arguments.dropFirst())
guard arguments.isEmpty || arguments == ["--check"] || arguments == ["--authorize"] || arguments == ["--listen"] || arguments == ["--meter"] else {
    emit(["error": "usage: --check (default), --authorize (speech OS prompt only), --listen (12s built-in mic), --meter (12s PCM diagnostics without recognition)"]); exit(2)
}
let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "ja-JP"))
let devices = AVCaptureDevice.DiscoverySession(deviceTypes: [.microphone], mediaType: .audio, position: .unspecified).devices.filter {
    UInt32(bitPattern: $0.transportType) == kAudioDeviceTransportTypeBuiltIn
}
if arguments.isEmpty || arguments == ["--check"] {
    emit(["locale": "ja-JP", "available": recognizer?.isAvailable ?? false,
          "on_device": recognizer?.supportsOnDeviceRecognition ?? false,
          "speech_authorization": SFSpeechRecognizer.authorizationStatus().rawValue,
          "mic_authorization": AVCaptureDevice.authorizationStatus(for: .audio).rawValue,
          "builtin_inputs": devices.count,
          "input_devices":devices.map { ["name":$0.localizedName, "unique_id":$0.uniqueID,
                                          "model_id":$0.modelID, "transport":$0.transportType] as [String:Any] },
          "captured": false, "cloud_fallback": false]); exit(0)
}
if arguments == ["--authorize"] {
    let code = await withCheckedContinuation { continuation in
        SFSpeechRecognizer.requestAuthorization { @Sendable status in continuation.resume(returning: status.rawValue) }
    }
    emit(["speech_authorization": code, "captured": false]); exit(code == 3 ? 0 : 3)
}
let meterOnly = arguments == ["--meter"]
guard (meterOnly || SFSpeechRecognizer.authorizationStatus() == .authorized),
      AVCaptureDevice.authorizationStatus(for: .audio) == .authorized,
      let recognizer, recognizer.isAvailable, recognizer.supportsOnDeviceRecognition,
      devices.count == 1, let device = devices.first else {
    emit(["error": "local_ASR_permission_or_builtin_input_unavailable", "captured": false, "cloud_fallback": false]); exit(3)
}
let request = SFSpeechAudioBufferRecognitionRequest()
request.requiresOnDeviceRecognition = true
request.shouldReportPartialResults = true
request.taskHint = .dictation
let state = State(meterOnly ? nil : request)
let task = meterOnly ? nil : recognizer.recognitionTask(with: request) { @Sendable result, error in
    state.update(result?.bestTranscription.formattedString, result?.isFinal ?? false, error as NSError?)
}
let session = AVCaptureSession()
let queue = DispatchQueue(label: "h3.builtin-mic.asr")
do {
    let input = try AVCaptureDeviceInput(device: device)
    let output = AVCaptureAudioDataOutput()
    // Explicit format contract; do not guess the sample layout from raw bytes.
    output.audioSettings = [AVFormatIDKey:kAudioFormatLinearPCM,
                            AVLinearPCMBitDepthKey:32, AVLinearPCMIsFloatKey:true,
                            AVLinearPCMIsBigEndianKey:false, AVLinearPCMIsNonInterleaved:false]
    output.setSampleBufferDelegate(state, queue: queue)
    guard session.canAddInput(input), session.canAddOutput(output) else { throw NSError(domain: "H3ASR", code: 1) }
    session.addInput(input); session.addOutput(output)
    guard input.device.uniqueID == device.uniqueID,
          output.connection(with: .audio) != nil else {
        throw NSError(domain:"H3ASR", code:2)
    }
    session.startRunning()
    let runningAfterStart = session.isRunning
    emit(["type": "listening", "seconds": 12, "source": "mac_builtin_microphone_not_discord_receive",
          "on_device": true, "speech_recognizer_used": !meterOnly, "input_unique_id":input.device.uniqueID])
    fflush(stdout)
    let deadline = Date().addingTimeInterval(12)
    while Date() < deadline && !state.result.1 && !state.result.2 { try await Task.sleep(for: .milliseconds(100)) }
    session.stopRunning(); queue.sync {}; state.endAudio()
    let finish = Date().addingTimeInterval(5)
    while !meterOnly && Date() < finish && !state.result.1 && !state.result.2 { try await Task.sleep(for: .milliseconds(100)) }
    let result = state.result
    var diagnostics = state.diagnostics
    diagnostics["session_running_after_start"] = runningAfterStart
    diagnostics["capture_input"] = device.localizedName
    diagnostics["capture_input_unique_id"] = input.device.uniqueID
    diagnostics["recognition_task_state"] = task?.state.rawValue ?? -1
    diagnostics["microphone_stopped"] = !session.isRunning
    emit(diagnostics)
    task?.cancel()
    if meterOnly { exit(0) }
    guard result.1, !result.0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        emit(["error": "no_final_utterance", "microphone_stopped": true]); exit(4)
    }
    emit(["type": "transcript", "text": result.0, "final": true, "on_device": true,
          "source": "mac_builtin_microphone_not_discord_receive", "microphone_stopped": true,
          "audio_saved": false, "cloud_fallback": false])
} catch {
    session.stopRunning(); queue.sync {}; state.endAudio(); task?.cancel()
    emit(["error": "local_ASR_failed", "microphone_stopped": true]); exit(1)
}
