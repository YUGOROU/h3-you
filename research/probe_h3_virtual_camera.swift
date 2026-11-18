// Finite OBS-only frame probe. Never falls back to a physical camera or microphone.
import Foundation
@preconcurrency import AVFoundation
import CoreImage
import ImageIO
import UniformTypeIdentifiers

final class Frames: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var received = 0
    private var saved = false
    let destination: URL
    init(_ destination: URL) { self.destination = destination }
    var count: Int { lock.lock(); defer { lock.unlock() }; return received }
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        lock.lock(); defer { lock.unlock() }
        received += 1
        guard !saved, let pixel = CMSampleBufferGetImageBuffer(sampleBuffer),
              let image = CIContext().createCGImage(CIImage(cvPixelBuffer: pixel), from: CIImage(cvPixelBuffer: pixel).extent),
              let sink = CGImageDestinationCreateWithURL(destination as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(sink, image, nil)
        saved = CGImageDestinationFinalize(sink)
        if saved { try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: destination.path) }
    }
}
let arguments = Array(CommandLine.arguments.dropFirst())
guard arguments.count == 1, ["--probe-obs-only", "--check-formats"].contains(arguments[0]) else {
    print("usage: --check-formats or --probe-obs-only (up to 8s; OBS Virtual Camera only)"); exit(2)
}
let devices = AVCaptureDevice.DiscoverySession(deviceTypes: [.external], mediaType: .video, position: .unspecified).devices.filter { $0.localizedName == "OBS Virtual Camera" }
guard devices.count == 1, let device = devices.first else { print("OBS_DEVICE_UNAVAILABLE; no physical fallback"); exit(3) }
if arguments[0] == "--check-formats" {
    let sizes = device.formats.map { format -> String in
        let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        return "\(d.width)x\(d.height)"
    }
    print("OBS_ADVERTISED_FORMATS \(Array(Set(sizes)).sorted()); metadata only, no capture")
    exit(0)
}
let session = AVCaptureSession()
let frames = Frames(URL(fileURLWithPath: "/tmp/h3-obs-virtual-frame.png"))
do {
    let input = try AVCaptureDeviceInput(device: device)
    let output = AVCaptureVideoDataOutput()
    output.alwaysDiscardsLateVideoFrames = true
    output.setSampleBufferDelegate(frames, queue: DispatchQueue(label: "h3.obs-only.frames"))
    guard session.canAddInput(input), session.canAddOutput(output) else { print("OBS_SESSION_REFUSED"); exit(4) }
    session.addInput(input); session.addOutput(output)
    session.startRunning()
    let deadline = Date().addingTimeInterval(8)
    while Date() < deadline && frames.count < 12 { RunLoop.current.run(until: Date().addingTimeInterval(0.1)) }
    session.stopRunning()
    print("OBS_ONLY_FRAMES \(frames.count); no microphone/physical camera; stopped")
} catch { print("OBS_ONLY_PROBE_FAILED \(type(of: error)); no fallback"); exit(1) }
