import AVFoundation
import Foundation

// AVFoundation explicitly supports a preview layer displaying a session managed
// on another executor. Only the engine mutates this session; the main actor may
// attach/detach it to a preview layer. No other cross-executor access is allowed.
final class PreviewSource: @unchecked Sendable {
    let session = AVCaptureSession()
}

enum CaptureEvent: Sendable {
    case ready(String)
    case started(Date)
    case finished(URL?, String)
    case problem(String)
    case audioRoute(AudioRouteSnapshot)
}

private final class MovieDelegate: NSObject, AVCaptureFileOutputRecordingDelegate, Sendable {
    let started: @Sendable () -> Void
    let finished: @Sendable (URL, Bool, String?, Double) -> Void

    init(started: @escaping @Sendable () -> Void,
         finished: @escaping @Sendable (URL, Bool, String?, Double) -> Void) {
        self.started = started
        self.finished = finished
    }

    func fileOutput(_ output: AVCaptureFileOutput, didStartRecordingTo url: URL,
                    from connections: [AVCaptureConnection]) {
        started()
    }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo url: URL,
                    from connections: [AVCaptureConnection], error: (any Error)?) {
        let nsError = error as NSError?
        let usable = error == nil || nsError?.userInfo[AVErrorRecordingSuccessfullyFinishedKey] as? Bool == true
        finished(url, usable, error?.localizedDescription, output.recordedDuration.seconds)
    }
}

actor CaptureEngine {
    nonisolated let preview = PreviewSource()
    nonisolated let events: AsyncStream<CaptureEvent>
    private let continuation: AsyncStream<CaptureEvent>.Continuation
    private let movie = AVCaptureMovieFileOutput()
    private var delegate: MovieDelegate?
    private var configured = false
    private var foreground = false
    private var pending = false
    private var preparing = false
    private var audioStartupInterruption: String?
    private var stopRequested = false
    private var metadata: RecordingMetadata?
    private var formatDescription = ""
    private var observers: [NSObjectProtocol] = []
    private var audioInput: AVCaptureDeviceInput?
    private var audioSessionTouched = false
    private var audioAttempt: UUID?
    private var audioObservers: [NSObjectProtocol] = []

    init() {
        let stream = AsyncStream<CaptureEvent>.makeStream()
        events = stream.stream
        continuation = stream.continuation
    }

    func activate() {
        foreground = true
        do {
            if !configured { try configure() }
            if !preview.session.isRunning { preview.session.startRunning() }
            if preview.session.isRunning && !preview.session.isInterrupted {
                continuation.yield(.ready(formatDescription))
            } else {
                continuation.yield(.problem("カメラを開始できません。他のカメラ利用を終了し、再試行してください。"))
            }
        } catch {
            continuation.yield(.problem(error.localizedDescription))
        }
    }

    private func configure() throws {
        let session = preview.session
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        // Video-only default never activates an audio session. Experimental audio
        // is configured explicitly at recording start and removed at finish.
        session.automaticallyConfiguresApplicationAudioSession = false
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            throw CaptureFailure.message("フロントカメラがありません（シミュレータでは撮影できません）。")
        }
        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input), session.canAddOutput(movie) else {
            throw CaptureFailure.message("カメラ入出力を構成できません。")
        }
        session.addInput(input)
        session.addOutput(movie)
        do {
            let preset: AVCaptureSession.Preset = session.canSetSessionPreset(.hd1920x1080) ? .hd1920x1080 : .hd1280x720
            guard session.canSetSessionPreset(preset) else { throw CaptureFailure.message("1080p/720p非対応です。") }
            session.sessionPreset = preset
            try camera.lockForConfiguration()
            defer { camera.unlockForConfiguration() }
            if camera.activeFormat.videoSupportedFrameRateRanges.contains(where: { $0.minFrameRate <= 30 && $0.maxFrameRate >= 30 }) {
                camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
                camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            }
            let dimensions = CMVideoFormatDescriptionGetDimensions(camera.activeFormat.formatDescription)
            formatDescription = "\(dimensions.width)×\(dimensions.height) / \(Int((1 / camera.activeVideoMinFrameDuration.seconds).rounded()))fps"
            if let connection = movie.connection(with: .video) {
                if connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }
                if connection.isVideoMirroringSupported {
                    connection.automaticallyAdjustsVideoMirroring = false
                    connection.isVideoMirrored = false
                }
                if connection.isVideoStabilizationSupported { connection.preferredVideoStabilizationMode = .standard }
                movie.setOutputSettings([AVVideoCodecKey: AVVideoCodecType.h264], for: connection)
            }
            movie.movieFragmentInterval = CMTime(seconds: 5, preferredTimescale: 600)
            movie.minFreeDiskSpaceLimit = 500 * 1_024 * 1_024
            configured = true
        } catch {
            session.removeInput(input)
            session.removeOutput(movie)
            throw error
        }
        let center = NotificationCenter.default
        for name in [AVCaptureSession.wasInterruptedNotification, AVCaptureSession.runtimeErrorNotification] {
            observers.append(center.addObserver(forName: name, object: session, queue: nil) { [weak self] note in
                let reason = (note.userInfo?[AVCaptureSessionInterruptionReasonKey] as? NSNumber)?.intValue
                let message = (note.userInfo?[AVCaptureSessionErrorKey] as? NSError)?.localizedDescription
                    ?? "カメラが割り込まれました (reason: \(reason ?? -1))。録画を停止します。"
                Task { await self?.interrupted(message) }
            })
        }
        observers.append(center.addObserver(forName: AVCaptureSession.interruptionEndedNotification, object: session, queue: nil) { [weak self] _ in
            Task { await self?.resumePreview() }
        })
    }

    private func resumePreview() {
        if foreground && !pending && !preparing { activate() }
    }

    func start(location: String, device: String, audioEnabled: Bool = false) async {
        guard foreground, configured, !pending, !preparing, preview.session.isRunning, !preview.session.isInterrupted else {
            continuation.yield(.finished(nil, "カメラの準備ができていません。再試行してください。"))
            return
        }
        preparing = true
        stopRequested = false
        audioStartupInterruption = nil
        defer { preparing = false }
        do {
            try Task.checkCancellation()
            // Retry unresolved cleanup before allowing even a video-only recording.
            if audioSessionTouched, let cleanupError = releaseAudio() {
                throw CaptureFailure.message(cleanupError + "\n撮影は開始していません。再試行してください。")
            }
            var root = RecordingMetadata.root
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            var backupPolicy = URLResourceValues()
            backupPolicy.isExcludedFromBackup = true
            try root.setResourceValues(backupPolicy)
            let values = try root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            if let bytes = values.volumeAvailableCapacityForImportantUsage, bytes < 1_000_000_000 {
                throw CaptureFailure.message("空き容量が1GB未満です。録画前に空きを確保してください。")
            }
            let record = RecordingMetadata(location: location, device: device, format: formatDescription, audioRequested: audioEnabled)
            try record.save()
            metadata = record
            if audioEnabled { try await prepareAudio() }
            try Task.checkCancellation()
            guard foreground, !stopRequested else { throw CancellationError() }
            guard !preview.session.isInterrupted else {
                throw CaptureFailure.message("音声設定でカメラが割り込まれました。ChatGPTの会話も確認してください。")
            }
            try metadata?.save()
            pending = true
            let delegate = MovieDelegate(started: { [weak self] in
                let now = Date()
                let uptime = ProcessInfo.processInfo.systemUptime
                Task { await self?.didStart(url: record.videoURL, date: now, uptime: uptime) }
            }, finished: { [weak self] url, usable, error, duration in
                Task { await self?.didFinish(url: url, usable: usable, error: error, duration: duration) }
            })
            self.delegate = delegate
            movie.startRecording(to: record.videoURL, recordingDelegate: delegate)
        } catch {
            let cancelled = error is CancellationError
            var message = cancelled ? "録画開始をキャンセルしました。" : error.localizedDescription
            metadata?.status = cancelled ? "cancelled_before_start" : "failed_to_start"
            metadata?.error = message
            metadata?.endedAt = Date()
            if let cleanupError = releaseAudio() { message += "\n\(cleanupError)" }
            do { try metadata?.save() }
            catch { message += "\n診断JSON保存失敗: \(error.localizedDescription)" }
            metadata = nil
            continuation.yield(.finished(nil, message))
        }
    }

    private func didStart(url: URL, date: Date, uptime: Double) {
        // Delegate callbacks hop executors; ignore late callbacks from old files.
        guard pending, metadata?.videoURL == url, metadata?.startedAt == nil else { return }
        metadata?.startedAt = date
        metadata?.startCallbackUptime = uptime
        do { try metadata?.save() }
        catch {
            stop()
            continuation.yield(.problem("開始メタデータ保存失敗: \(error.localizedDescription)"))
        }
        continuation.yield(.started(date))
        if audioSessionTouched { checkAudioRoute(event: "recording_started") }
        if stopRequested { movie.stopRecording() }
    }

    func stop() {
        guard pending || preparing else { return }
        stopRequested = true
        if movie.isRecording { movie.stopRecording() }
    }

    func deactivate() {
        foreground = false
        stop()
        // Keep the session alive until the file delegate finalizes the movie.
        if !pending && !preparing {
            if let cleanupError = releaseAudio() { continuation.yield(.problem(cleanupError)) }
            if preview.session.isRunning { preview.session.stopRunning() }
        }
    }

    private func interrupted(_ message: String) {
        metadata?.stopReason = message
        if audioSessionTouched { _ = recordAudioRoute(event: "capture_interruption: \(message)") }
        stop()
        continuation.yield(.problem(message))
    }

    private func didFinish(url: URL, usable: Bool, error: String?, duration: Double) async {
        guard pending, metadata?.videoURL == url else { return }
        // Keep pending true across inspection so a new start cannot replace metadata.
        var inspectionError: String?
        if usable {
            do { metadata?.audioTrackCount = try await AVURLAsset(url: url).loadTracks(withMediaType: .audio).count }
            catch { inspectionError = "音声トラック検査失敗: \(error.localizedDescription)" }
        }
        metadata?.endedAt = Date()
        metadata?.durationSeconds = duration.isFinite ? duration : nil
        metadata?.status = usable ? "completed" : "failed"
        metadata?.error = error
        var message = usable ? "保存完了（音声なし）" : "録画保存失敗。残存ファイルは削除していません。"
        if usable && metadata?.audioRequested == true {
            if let count = metadata?.audioTrackCount, count > 0 {
                message = "保存完了（音声トラックあり・内容は再生で確認）"
            } else {
                message = "動画保存済み。ただし音声トラックを確認できません。JSONを確認してください。"
                metadata?.status = "completed_audio_unverified"
            }
        }
        if let inspectionError {
            metadata?.error = [error, inspectionError].compactMap { $0 }.joined(separator: "\n")
            message += "\n\(inspectionError)"
        }
        if let error { message += "\n\(error)" }
        if let stopReason = metadata?.stopReason { message += "\n停止理由: \(stopReason)" }
        if let cleanupError = releaseAudio() { message += "\n\(cleanupError)" }
        do { try metadata?.save() }
        catch { message += "\nメタデータ保存失敗: \(error.localizedDescription)" }
        pending = false
        delegate = nil
        metadata = nil
        if !foreground && preview.session.isRunning { preview.session.stopRunning() }
        continuation.yield(.finished(usable ? url : nil, message))
        if foreground && !preview.session.isInterrupted { activate() }
    }
}

// MARK: - Opt-in audio experiment

private extension CaptureEngine {
    func prepareAudio() async throws {
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else {
            throw CaptureFailure.message("先に「マイク権限を許可」を押してください。")
        }
        let audio = AVAudioSession.sharedInstance()
        audioSessionTouched = true
        let attempt = UUID()
        audioAttempt = attempt
        observeAudio(audio, attempt: attempt)
        _ = recordAudioRoute(event: "before_activation")
        // Never force speaker output or enable HFP input for our recording.
        // A2DP permits built-in input + Bluetooth output when the OS allows it.
        try audio.setCategory(.playAndRecord, mode: .default, options: [.mixWithOthers, .allowBluetoothA2DP])
        try audio.setActive(true)
        guard let builtIn = audio.availableInputs?.first(where: { $0.portType == .builtInMic }) else {
            throw CaptureFailure.message("iPhone内蔵マイクが利用できません。")
        }
        try audio.setPreferredInput(builtIn)
        _ = recordAudioRoute(event: "preferred_input_requested")
        let settled = try await AudioRouteSettlement.wait(sample: { [self] in
            try await sampleStartupRoute()
        })
        guard settled else {
            _ = recordAudioRoute(event: "route_selection_timeout")
            throw CaptureFailure.message("3秒待ちましたが内蔵マイクに切り替わりませんでした。ChatGPT側の入力をiPhone Microphoneに変え、A25iで再生できるか確認してから再試行してください。")
        }
        // Recheck after the cross-actor wait returns: an interruption or route
        // change may have arrived between the last sample and this continuation.
        guard try sampleStartupRoute() else {
            throw CaptureFailure.message("開始直前に内蔵マイク以外へ切り替わりました。録画は開始していません。")
        }
        guard let device = AVCaptureDevice.default(for: .audio) else {
            throw CaptureFailure.message("音声入力デバイスがありません。")
        }
        let input = try AVCaptureDeviceInput(device: device)
        preview.session.beginConfiguration()
        guard preview.session.canAddInput(input) else {
            preview.session.commitConfiguration()
            throw CaptureFailure.message("音声入力を追加できません。ChatGPTとの競合の可能性があります。")
        }
        preview.session.addInput(input)
        audioInput = input
        preview.session.commitConfiguration()
        guard movie.connection(with: .audio) != nil else {
            throw CaptureFailure.message("動画の音声接続を作成できません。")
        }
        metadata?.audioSource = "built_in_microphone_requested; see audioRoutes and audioTrackCount"
    }

    func sampleStartupRoute() throws -> Bool {
        try Task.checkCancellation()
        guard foreground, !stopRequested else { throw CancellationError() }
        if let audioStartupInterruption { throw CaptureFailure.message(audioStartupInterruption) }
        guard !preview.session.isInterrupted else {
            throw CaptureFailure.message("音声経路の準備中にカメラが割り込まれました。")
        }
        return recordAudioRoute(event: "waiting_for_built_in_input").usesOnlyBuiltInMic
    }

    func observeAudio(_ audio: AVAudioSession, attempt: UUID) {
        let center = NotificationCenter.default
        audioObservers.append(center.addObserver(forName: AVAudioSession.routeChangeNotification, object: audio, queue: nil) { [weak self] note in
            let reason = (note.userInfo?[AVAudioSessionRouteChangeReasonKey] as? NSNumber)?.intValue ?? -1
            Task { await self?.audioNotification(attempt: attempt, event: "route_change:\(reason)", shouldStop: false) }
        })
        audioObservers.append(center.addObserver(forName: AVAudioSession.interruptionNotification, object: audio, queue: nil) { [weak self] note in
            let type = (note.userInfo?[AVAudioSessionInterruptionTypeKey] as? NSNumber)?.uintValue
            guard type == AVAudioSession.InterruptionType.began.rawValue else { return }
            Task { await self?.audioNotification(attempt: attempt, event: "audio_interruption_began", shouldStop: true) }
        })
        audioObservers.append(center.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification, object: audio, queue: nil) { [weak self] _ in
            Task { await self?.audioNotification(attempt: attempt, event: "media_services_reset", shouldStop: true) }
        })
    }

    func recordAudioRoute(event: String) -> AudioRouteSnapshot {
        let audio = AVAudioSession.sharedInstance()
        let route = audio.currentRoute
        var snapshot = AudioRouteSnapshot(event: event,
            inputs: route.inputs.map { .init(name: $0.portName, type: $0.portType.rawValue) },
            outputs: route.outputs.map { .init(name: $0.portName, type: $0.portType.rawValue) },
            sampleRate: audio.sampleRate, category: audio.category.rawValue, mode: audio.mode.rawValue,
            microphoneMode: microphoneModeName)
        snapshot.availableInputs = (audio.availableInputs ?? []).map { .init(name: $0.portName, type: $0.portType.rawValue) }
        snapshot.preferredInput = audio.preferredInput.map { .init(name: $0.portName, type: $0.portType.rawValue) }
        snapshot.categoryOptions = audio.categoryOptions.rawValue
        metadata?.audioRoutes.append(snapshot)
        continuation.yield(.audioRoute(snapshot))
        return snapshot
    }

    var microphoneModeName: String {
        switch AVCaptureDevice.activeMicrophoneMode {
        case .standard: return "Standard"
        case .voiceIsolation: return "Voice Isolation"
        case .wideSpectrum: return "Wide Spectrum"
        @unknown default: return String(describing: AVCaptureDevice.activeMicrophoneMode)
        }
    }

    func audioNotification(attempt: UUID, event: String, shouldStop: Bool) {
        guard audioAttempt == attempt else { return }
        if preparing && shouldStop { audioStartupInterruption = "音声準備中の割り込み: \(event)" }
        checkAudioRoute(event: event, shouldStop: shouldStop)
    }

    func checkAudioRoute(event: String, shouldStop: Bool = false) {
        let snapshot = recordAudioRoute(event: event)
        if pending && !stopRequested && (shouldStop || !snapshot.usesOnlyBuiltInMic) {
            interrupted("音声割り込み、または内蔵マイク以外への変更: \(event)。安全のため停止。")
        }
        do { try metadata?.save() }
        catch { interrupted("音声診断JSON保存失敗: \(error.localizedDescription)") }
    }

    /// No-op in video-only mode. Always remove capture input before deactivation.
    func releaseAudio() -> String? {
        guard audioSessionTouched else { return nil }
        audioAttempt = nil
        for observer in audioObservers { NotificationCenter.default.removeObserver(observer) }
        audioObservers.removeAll()
        _ = recordAudioRoute(event: "before_release")
        if let audioInput {
            preview.session.beginConfiguration()
            preview.session.removeInput(audioInput)
            preview.session.commitConfiguration()
            self.audioInput = nil
        }
        let audio = AVAudioSession.sharedInstance()
        var errors: [String] = []
        do { try audio.setPreferredInput(nil) } catch { errors.append(error.localizedDescription) }
        do { try audio.setActive(false, options: .notifyOthersOnDeactivation) }
        catch { errors.append(error.localizedDescription) }
        audioSessionTouched = !errors.isEmpty
        guard !errors.isEmpty else { return nil }
        let message = "音声セッション解放エラー: " + errors.joined(separator: "; ")
        metadata?.audioCleanupError = message
        return message
    }
}

enum CaptureFailure: LocalizedError {
    case message(String)
    var errorDescription: String? { if case .message(let text) = self { text } else { nil } }
}
