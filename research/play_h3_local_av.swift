// macOS unified AV playback. Runtime requires separate operator approval.
import AppKit
import AVKit
import CoreAudio
import Darwin

struct Configuration: Decodable {
    let url: String
    let output_uid: String
    let max_seconds: Double

    static func read() throws -> Configuration {
        guard CommandLine.arguments.count == 1 else { throw Invalid.configuration }
        var bytes = Data()
        while bytes.count < 65536 {
            var byte: UInt8 = 0
            guard Darwin.read(STDIN_FILENO, &byte, 1) == 1 else { throw Invalid.configuration }
            if byte == 10 { break }
            bytes.append(byte)
        }
        guard bytes.count < 65536,
              let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
              Set(object.keys) == Set(["url", "output_uid", "max_seconds"])  else { throw Invalid.configuration }
        let config = try JSONDecoder().decode(Configuration.self, from: bytes)
        guard config.output_uid == "BuiltInSpeakerDevice",
              config.max_seconds.isFinite, config.max_seconds > 0, config.max_seconds <= 240,
              config.url.range(of: #"^http://127\.0\.0\.1:([1-9][0-9]{0,4})/index\.m3u8$"#,
                               options: .regularExpression) != nil,
              let url = URLComponents(string: config.url), let port = url.port,
              (1...65535).contains(port) else { throw Invalid.configuration }
        return config
    }
}
enum Invalid: Error { case configuration, device }

// Pure startup gate. A growing live playlist needs a usable live window,
// unlike an already finished fixture. Do not count this buffer as H3 latency.
func playlistCanStart(_ text: String) -> Bool {
    let lines = text.split(whereSeparator: \.isNewline).map(String.init)
    guard lines.first == "#EXTM3U",
          let targetLine = lines.first(where: { $0.hasPrefix("#EXT-X-TARGETDURATION:") }),
          let target = Double(targetLine.split(separator: ":").last ?? ""),
          target.isFinite, target > 0, target <= 60 else { return false }
    let entries = lines.filter { $0.hasPrefix("#EXTINF:") }
    let segments = lines.filter { $0.range(of: #"^seg-[0-9]{3,6}\.ts$"#, options: .regularExpression) != nil }
    guard !entries.isEmpty, entries.count == segments.count, entries.count <= 256 else { return false }
    var duration = 0.0
    for entry in entries {
        guard let tail = entry.split(separator: ":").last,
              let part = tail.split(separator: ",").first,
              let seconds = Double(part), seconds.isFinite, seconds > 0 else { return false }
        duration += seconds
    }
    return lines.contains("#EXT-X-ENDLIST") || duration >= 3 * target
}

// Read-only CoreAudio enumeration: never consult or modify a default route.
func builtInOutputExists() -> Bool {
    var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    let system = AudioObjectID(kAudioObjectSystemObject)
    guard AudioObjectGetPropertyDataSize(system, &address, 0, nil, &size) == noErr else { return false }
    var devices = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    guard AudioObjectGetPropertyData(system, &address, 0, nil, &size, &devices) == noErr else { return false }
    for device in devices {
        address.mSelector = kAudioDevicePropertyDeviceUID
        var uid: Unmanaged<CFString>?
        size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &uid) == noErr,
              uid?.takeRetainedValue() as String? == "BuiltInSpeakerDevice" else { continue }
        var transport: UInt32 = 0
        address.mSelector = kAudioDevicePropertyTransportType
        size = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &transport) == noErr,
              transport == kAudioDeviceTransportTypeBuiltIn else { return false }
        var alive: UInt32 = 0
        address.mSelector = kAudioDevicePropertyDeviceIsAlive
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &alive) == noErr,
              alive == 1 else { return false }
        address.mSelector = kAudioDevicePropertyStreams
        address.mScope = kAudioDevicePropertyScopeOutput
        size = 0
        return AudioObjectGetPropertyDataSize(device, &address, 0, nil, &size) == noErr && size >= MemoryLayout<AudioStreamID>.size
    }
    return false
}

@MainActor
final class LocalPlayer: NSObject, NSWindowDelegate {
    let config: Configuration
    let started = ProcessInfo.processInfo.systemUptime
    var player: AVPlayer?
    var view: AVPlayerView?
    var window: NSWindow?
    var timer: Timer?
    var termination: DispatchSourceSignal?
    var probeTask: Task<Void, Never>?
    var probeSession: URLSession?
    var eventCount = 0
    var seen = Set<String>()
    var previousState = ""
    var finished = false

    init(_ config: Configuration) { self.config = config }

    func emit(_ value: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) else { return }
        FileHandle.standardOutput.write(data + Data([10])) // unbuffered pipe write
    }
    func event(_ name: String) {
        let terminal = ["ended", "failed", "timeout"].contains(name)
        guard eventCount < (terminal ? 254 : 253) else { return }
        eventCount += 1
        let seconds = player?.currentTime().seconds ?? 0
        emit(["type": "local_player_telemetry", "event": name,
              "elapsed_seconds": max(0, ProcessInfo.processInfo.systemUptime - started),
              "position_seconds": seconds.isFinite ? max(0, seconds) : 0])
    }
    func once(_ name: String) { if seen.insert(name).inserted { event(name) } }

    func start() throws {
        guard builtInOutputExists() else { throw Invalid.device }
        let player = AVPlayer() // Only constructed after the device gate.
        self.player = player
        player.automaticallyWaitsToMinimizeStalling = false
        player.audioOutputDeviceUniqueID = config.output_uid
        guard player.audioOutputDeviceUniqueID == config.output_uid, builtInOutputExists() else { throw Invalid.device }
        // AVPlayer.allowsExternalPlayback is unavailable on macOS. Hide route UI instead.
        let view = AVPlayerView()
        view.controlsStyle = .none
        view.showsSharingServiceButton = false
        view.allowsPictureInPicturePlayback = false
        view.videoGravity = .resizeAspect
        view.player = player
        self.view = view
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 432, height: 768),
                              styleMask: [.titled, .closable, .resizable, .miniaturizable], backing: .buffered, defer: false)
        window.title = "H3 Local AV"
        window.contentView = view
        window.delegate = self
        window.isReleasedWhenClosed = false
        self.window = window
        NotificationCenter.default.addObserver(self, selector: #selector(ended(_:)), name: .AVPlayerItemDidPlayToEndTime, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(failed(_:)), name: .AVPlayerItemFailedToPlayToEndTime, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(stalled(_:)), name: .AVPlayerItemPlaybackStalled, object: nil)
        window.center()
        window.makeKeyAndOrderFront(nil)
        emit(["type": "local_player_ready", "output_uid": config.output_uid])
        timer = Timer.scheduledTimer(timeInterval: 0.1, target: self, selector: #selector(tick), userInfo: nil, repeats: true)
        signal(SIGTERM, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        source.setEventHandler { [weak self] in
            MainActor.assumeIsolated { self?.finish(143, "failed") }
        }
        termination = source
        source.resume()
        beginAvailabilityProbe()
    }
    func beginAvailabilityProbe() {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = [:]
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = 1
        configuration.timeoutIntervalForResource = 1
        let session = URLSession(configuration: configuration)
        probeSession = session
        let url = URL(string: config.url)!
        once("awaiting_playlist")
        probeTask = Task { @MainActor [weak self] in
            guard let self else { return }
            while !Task.isCancelled && !self.finished {
                // Sequential awaits ensure exactly one request in flight. The main
                // timer still enforces the total deadline and verifies the device.
                do {
                    let (data, response) = try await session.data(from: url)
                    guard !Task.isCancelled, !self.finished else { return }
                    if let http = response as? HTTPURLResponse,
                       http.statusCode == 200, http.url == url,
                       String(data: data, encoding: .utf8).map(playlistCanStart) == true {
                        guard let player = self.player else { return }
                        let item = AVPlayerItem(url: url)
                        item.preferredForwardBufferDuration = 0
                        player.replaceCurrentItem(with: item)
                        // Live EVENT playlists need playback intent while loading;
                        // readyToPlay alone must not gate playlist replenishment.
                        // A requested rate is still not proof of a running clock.
                        player.play()
                        session.finishTasksAndInvalidate()
                        return
                    }
                } catch { /* Startup availability only; never restart playback. */ }
                do { try await Task.sleep(for: .milliseconds(150)) } catch { return }
            }
        }
    }
    @objc func tick() {
        guard !finished, let player else { return }
        if ProcessInfo.processInfo.systemUptime - started >= config.max_seconds { finish(124, "timeout"); return }
        guard player.audioOutputDeviceUniqueID == config.output_uid, builtInOutputExists() else { finish(1, "failed"); return }
        if player.status == .failed || player.currentItem?.status == .failed { finish(1, "failed"); return }
        if player.currentItem?.status == .readyToPlay && seen.insert("ready").inserted {
            event("ready")
            player.audioOutputDeviceUniqueID = config.output_uid
            player.playImmediately(atRate: 1)
        }
        if view?.isReadyForDisplay == true { once("first_video") }
        guard player.currentItem != nil else { return }
        let state = player.timeControlStatus == .playing ? "playing" :
            (player.currentItem?.isPlaybackBufferEmpty == true || player.timeControlStatus == .waitingToPlayAtSpecifiedRate ? "buffering" : "waiting")
        if state != previousState { previousState = state; if eventCount < 253 { event(state) } }
    }
    @objc func ended(_ note: Notification) {
        guard let item = note.object as? AVPlayerItem, item === player?.currentItem else { return }
        finish(0, "ended") // AVFoundation signals completion after the item's media tail.
    }
    @objc func failed(_ note: Notification) {
        guard let item = note.object as? AVPlayerItem, item === player?.currentItem else { return }
        finish(1, "failed")
    }
    @objc func stalled(_ note: Notification) {
        guard let item = note.object as? AVPlayerItem, item === player?.currentItem else { return }
        once("stalled")
    }
    func windowWillClose(_ notification: Notification) { finish(130, "failed") }
    func finish(_ code: Int32, _ name: String) {
        guard !finished else { return }
        finished = true
        event(name)
        if code == 0 && name == "ended" { emit(["type": "local_player_done"]) }
        probeTask?.cancel()
        probeSession?.invalidateAndCancel()
        timer?.invalidate()
        termination?.cancel()
        NotificationCenter.default.removeObserver(self)
        player?.pause()
        player?.replaceCurrentItem(with: nil)
        view?.player = nil
        window?.orderOut(nil)
        // Immediate process exit bounds cleanup; no background app or delayed network work.
        Darwin.exit(code)
    }
}

@main
struct Main {
    @MainActor static func main() {
        do {
            let config = try Configuration.read() // bounded blocking constructor protocol, before NSApp
            let app = NSApplication.shared
            app.setActivationPolicy(.regular)
            let controller = LocalPlayer(config)
            do { try controller.start() } catch { controller.finish(1, "failed") }
            withExtendedLifetime(controller) { app.run() }
        } catch {
            let data = Data("{\"type\":\"local_player_telemetry\",\"event\":\"failed\",\"elapsed_seconds\":0,\"position_seconds\":0}\n".utf8)
            FileHandle.standardOutput.write(data)
            Darwin.exit(2)
        }
    }
}
