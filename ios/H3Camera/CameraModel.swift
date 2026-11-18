import AVFoundation
import Observation
import UIKit

@MainActor @Observable
final class CameraModel {
    let engine = CaptureEngine()
    var status = "ChatGPTのBackground Conversationsを有効にしてから撮影してください。"
    var ready = false
    var busy = false
    var stopping = false
    var startedAt: Date?
    var format = "映像のみ・マイク不使用"
    var files: [URL] = []
    var location = "indoor"
    var audioEnabled = false
    var microphoneGranted = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
    var requestingMicrophone = false
    var audioRoute = "録画開始後に実際の経路を表示します。"
    var audioRouteIsCurrent = false
    var capacity = ""
    var thermal = ""
    private var active = false
    private var lifecycleRevision = 0
    private let commands = SerialCommands()
    private var startCommand: Task<Void, Never>?
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid

    func observe() async {
        refreshFiles()
        for await event in engine.events {
            switch event {
            case .ready(let description):
                ready = active
                format = description
            case .started(let date):
                startedAt = date
                status = stopping ? "保存中…" : (audioEnabled ? "録画中（音声実験・会話継続を確認）" : "録画中（音声なし）")
            case .finished(_, let message):
                startCommand = nil
                busy = false
                stopping = false
                startedAt = nil
                status = message
                UIApplication.shared.isIdleTimerDisabled = false
                endBackgroundTask()
                audioRouteIsCurrent = false
                refreshFiles()
            case .audioRoute(let route):
                audioRoute = route.summary
                audioRouteIsCurrent = route.event != "before_release"
            case .problem(let message):
                status = message
                ready = false
                if busy { stopping = true }
                else { UIApplication.shared.isIdleTimerDisabled = false }
            }
        }
    }

    func setActive(_ value: Bool) async {
        active = value
        lifecycleRevision += 1
        let revision = lifecycleRevision
        let engine = engine
        if value {
            microphoneGranted = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            guard active, revision == lifecycleRevision else { return }
            guard granted else {
                status = "カメラが許可されていません。設定 → H3 Cameraで許可してください。"
                return
            }
            await commands.enqueue { await engine.activate() }.value
            refreshFiles()
        } else {
            startCommand?.cancel()
            ready = false
            if busy {
                stopping = true
                status = "画面を離れたため停止・保存中…"
                if backgroundTask == .invalid {
                    backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "Finalize movie") { [weak self] in
                        self?.endBackgroundTask()
                    }
                }
            }
            UIApplication.shared.isIdleTimerDisabled = false
            await commands.enqueue { await engine.deactivate() }.value
        }
    }

    func toggleRecording() {
        if busy {
            guard !stopping else { return }
            startCommand?.cancel()
            stopping = true
            status = "停止・保存中…"
            let engine = engine
            commands.enqueue { await engine.stop() }
        } else {
            guard ready, active else { return }
            guard !audioEnabled || microphoneGranted else {
                status = "先にマイク権限を許可してください。"
                return
            }
            audioRouteIsCurrent = false
            busy = true
            status = audioEnabled ? "内蔵マイクへの切り替え待ち（最大3秒）…" : "録画開始中…"
            UIApplication.shared.isIdleTimerDisabled = true
            let location = location
            let device = "\(UIDevice.current.model), iOS \(UIDevice.current.systemVersion)"
            let engine = engine
            let audioEnabled = audioEnabled
            startCommand = commands.enqueue { await engine.start(location: location, device: device, audioEnabled: audioEnabled) }
        }
    }

    func requestMicrophone() async {
        guard !busy, !requestingMicrophone else { return }
        requestingMicrophone = true
        microphoneGranted = await AVCaptureDevice.requestAccess(for: .audio)
        requestingMicrophone = false
        status = microphoneGranted ? "マイク許可済み。ChatGPTとの会話を確認してから録画してください。" : "マイクが拒否されています。設定 → H3 Cameraで許可するか、映像のみを選んでください。"
    }

    func updateHealth() {
        if let values = try? RecordingMetadata.root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
           let bytes = values.volumeAvailableCapacityForImportantUsage {
            capacity = "空き \(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file))"
        }
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermal = "温度: 正常"
        case .fair: thermal = "温度: やや高い"
        case .serious: thermal = "高温: 休憩を推奨"
        case .critical:
            thermal = "高温: 撮影停止"
            if busy && !stopping { toggleRecording() }
        @unknown default: thermal = "温度: 不明"
        }
    }

    func refreshFiles() {
        files = ((try? FileManager.default.contentsOfDirectory(at: RecordingMetadata.root,
            includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "mov" || $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
        updateHealth()
    }

    private func endBackgroundTask() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }
}
