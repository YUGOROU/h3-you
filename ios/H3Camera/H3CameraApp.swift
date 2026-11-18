import SwiftUI
import AVFoundation

@main
struct H3CameraApp: App {
    var body: some Scene {
        WindowGroup { CameraScreen() }
    }
}

struct CameraScreen: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var camera = CameraModel()
    @State private var showFiles = false

    var body: some View {
        @Bindable var camera = camera
        VStack(spacing: 12) {
            HStack {
                Text("H3 Camera").font(.headline)
                Spacer()
                Text(camera.audioEnabled ? "音声実験" : "マイク不使用")
                    .font(.caption).foregroundStyle(camera.audioEnabled ? .orange : .green)
            }
            CameraPreview(source: camera.engine.preview)
                .aspectRatio(9.0 / 16.0, contentMode: .fit)
                .overlay(alignment: .topLeading) {
                    if let start = camera.startedAt {
                        Text(start, style: .timer)
                            .monospacedDigit().padding(10)
                            .background(.red, in: Capsule()).padding()
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .layoutPriority(-1)
            Text(camera.format).font(.caption)
            Text(camera.status).font(.callout).frame(maxWidth: .infinity, alignment: .leading)
            HStack {
                Text(camera.capacity)
                Spacer()
                Text(camera.thermal)
            }.font(.caption).foregroundStyle(.secondary)
            Picker("収録場所", selection: $camera.location) {
                Text("自宅 / 屋内").tag("indoor")
                Text("外").tag("outdoor")
            }.pickerStyle(.segmented).disabled(camera.busy)
            Toggle("iPhoneマイク録音（実験）", isOn: $camera.audioEnabled)
                .disabled(camera.busy || camera.requestingMicrophone)
            if camera.audioEnabled {
                Text("ChatGPTの会話やBluetooth経路が変わる可能性があります。")
                    .font(.caption).foregroundStyle(.orange)
                if !camera.microphoneGranted {
                    Button("マイク権限を許可") { Task { await camera.requestMicrophone() } }
                        .disabled(camera.requestingMicrophone)
                }
                Text((camera.audioRouteIsCurrent ? "H3 Cameraの経路\n" : "参考・非録音時の最終経路\n") + camera.audioRoute)
                    .font(.caption2).frame(maxWidth: .infinity, alignment: .leading)
            }
            Button(action: camera.toggleRecording) {
                Label(camera.busy ? (camera.stopping ? "保存中…" : "停止して保存") : "録画開始",
                      systemImage: camera.busy ? "stop.fill" : "record.circle")
                    .font(.title3.bold()).frame(maxWidth: .infinity).padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent).tint(.red)
            .disabled(camera.stopping || camera.requestingMicrophone || (!camera.ready && !camera.busy)
                      || (!camera.busy && camera.audioEnabled && !camera.microphoneGranted))
            HStack {
                Button("保存ファイル") { camera.refreshFiles(); showFiles = true }
                    .disabled(camera.busy)
                Spacer()
                if !camera.ready && !camera.busy {
                    Button("カメラを再試行") { Task { await camera.setActive(true) } }
                }
            }
        }
        .padding()
        .preferredColorScheme(.dark)
        .task { await camera.observe() }
        .task(id: scenePhase) { await camera.setActive(scenePhase == .active) }
        .task {
            while !Task.isCancelled {
                camera.updateHealth()
                do { try await Task.sleep(for: .seconds(5)) } catch { break }
            }
        }
        .sheet(isPresented: $showFiles) {
            NavigationStack {
                List {
                    Section {
                        Text("MOVと同名のJSONを一緒にMacへ保存してください。映像は非ミラーです。音声は録画時の実験スイッチによります。Filesアプリ → このiPhone内 → H3 Camera、またはMacのFinder → iPhone → ファイルからも取り出せます。")
                        Text("未完了・失敗ファイルも残します。異常終了後は再生できるか確認してください。音声設定失敗時はJSONだけが残ることがあります。")
                    }
                    ForEach(camera.files, id: \.self) { file in
                        ShareLink(item: file) {
                            Label(file.lastPathComponent, systemImage: file.pathExtension == "mov" ? "film" : "doc.text")
                                .font(.caption).lineLimit(3)
                        }
                    }
                }
                .navigationTitle("保存ファイル")
                .toolbar { Button("閉じる") { showFiles = false } }
            }
        }
    }
}

private final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

    override func layoutSubviews() {
        super.layoutSubviews()
        if let connection = previewLayer.connection {
            if connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = true
            }
        }
    }
}

private struct CameraPreview: UIViewRepresentable {
    let source: PreviewSource
    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = source.session
        view.previewLayer.videoGravity = .resizeAspect
        return view
    }
    func updateUIView(_ uiView: PreviewView, context: Context) { uiView.setNeedsLayout() }
    static func dismantleUIView(_ uiView: PreviewView, coordinator: ()) { uiView.previewLayer.session = nil }
}
