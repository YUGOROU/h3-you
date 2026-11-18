"""CPU source contracts and unsigned compile only; never execute the helper."""
import pathlib
import platform
import shutil
import subprocess
import unittest

SOURCE = pathlib.Path(__file__).resolve().parents[1] / 'research/listen_h3_comet.swift'


class CometSpeechTests(unittest.TestCase):
    def test_static_scope(self):
        text = SOURCE.read_text()
        for forbidden in ('AVCapture', 'AVAudioEngine', 'requestRecordPermission',
                          'stereoGlobalTap', 'monoGlobalTap', 'AudioObjectSetPropertyData',
                          'URLSession', 'AVAudioFile', 'Process()', 'ScreenCaptureKit'):
            self.assertNotIn(forbidden, text)
        for required in ('ai.perplexity.comet.', 'kAudioHardwarePropertyProcessObjectList',
                         'kAudioProcessPropertyBundleID', 'kAudioProcessPropertyPID',
                         'description.isPrivate = true', 'description.isExclusive = false',
                         'description.muteBehavior = .unmuted',
                         'request.requiresOnDeviceRecognition = true',
                         'recognizer.supportsOnDeviceRecognition',
                         'kAudioAggregateDeviceSubDeviceListKey: []',
                         'kAudioAggregateDeviceIsPrivateKey: true',
                         'kAudioTapPropertyFormat', 'AVAudioPCMBuffer',
                         'pcm.mutableAudioBufferList', 'memcpy(', 'request.append(pcm)',
                         'AudioDeviceStop', 'AudioDeviceDestroyIOProcID',
                         'AudioHardwareDestroyAggregateDevice', 'AudioHardwareDestroyProcessTap',
                         '.seconds(12)', '.seconds(5)', 'ja-JP'):
            self.assertIn(required, text)
        self.assertLess(text.index('let stopped = cleanup()'), text.index('"type": "transcript"'))
        self.assertLess(text.index('guard args.isEmpty'), text.index('cometProcesses()\n'))
        authorize = text.split('if args == ["--authorize"] {', 1)[1].split('\ndo {', 1)[0]
        self.assertIn('SFSpeechRecognizer.requestAuthorization', authorize)
        self.assertNotIn('AudioHardware', authorize)
        self.assertNotIn('cometProcesses', authorize)

    def test_turn_endpoint_contract(self):
        text = SOURCE.read_text()
        self.assertIn('args == ["--listen-turn"]', text)
        self.assertIn('request.shouldReportPartialResults = listenTurn', text)
        self.assertIn('listenTurn ? .seconds(45) : .seconds(12)', text)
        self.assertIn('if listenTurn && sink.turnEnded { stopReason = "quiet_endpoint"; break }', text)
        endpoint = text.split('var turnEnded: Bool {', 1)[1].split('var result:', 1)[0]
        for required in ('!text.trimmingCharacters', 'let lastTextTime',
                         'let lastVoiceTime', 'now - lastTextTime >= .seconds(1)',
                         'now - lastVoiceTime >= .seconds(1)',
                         'now - lastBufferTime < .milliseconds(500)'):
            self.assertIn(required, endpoint)
        self.assertIn('if quiet && preserveEmptyFinal && !failed { frozenText = text }', endpoint)
        self.assertIn('return quiet && !failed', endpoint)
        self.assertIn('bufferPeak >= 0.01 || rms >= 0.003', text)
        self.assertIn('if updated != text { lastTextTime = ContinuousClock.now }', text)
        self.assertIn('final = result.isFinal', text)
        self.assertIn('guard result.1, !result.2, !result.0.trimmingCharacters', text)
        self.assertEqual(text.count('"type": "transcript"'), 1)
        self.assertNotIn('emit(', text.split('final class Sink:', 1)[1].split('let args =', 1)[0])
        stop = text.index('let stopped = cleanup()')
        end = text.index('sink.end()', stop)
        finish = text.index('let finish =', end)
        output = text.index('emit(["type": "transcript"', finish)
        self.assertLess(stop, end)
        self.assertLess(end, finish)
        self.assertLess(finish, output)
        self.assertIn('let finish = ContinuousClock.now.advanced(by: .seconds(5))', text)
        self.assertIn('{ @Sendable result, error in sink.update(result, error) }', text)
        self.assertIn('{ @Sendable _, input, _, _, _ in sink.append(input) }', text)

    def test_stream_event_contract_and_fail_closed(self):
        text = SOURCE.read_text()
        self.assertIn('let streamTurn = args == ["--stream-turn"]', text)
        partial = text.split('    func emitPartial(', 1)[1].split('    var announcedSpeech', 1)[0]
        for required in ('guard streamTurn', 'text != emittedText', 'revision += 1',
                         '"type": "partial"', '"revision": revision',
                         '"final": false', '"capture_stopped": false'):
            self.assertIn(required, partial)
        self.assertIn('print(text); fflush(stdout)', text)
        stream = text.split('    if streamTurn {\n        var diagnostics', 1)[1].split('    } else {\n    let finish', 1)[0]
        for required in ('stopReason == "quiet_endpoint"', 'ContinuousClock.now < deadline',
                         '!captured.failed', '!sink.result.2', '(1...160).contains(captured.text.count)',
                         '"type": "utterance_endpoint"', '"revision": revision',
                         '"final": false', '"capture_stopped": true',
                         '"endpoint_source": "application_stable_partial"',
                         '"audio_saved": false', '"cloud_fallback": false'):
            self.assertIn(required, stream)
        self.assertLess(stream.index('guard stopReason'), stream.index('emitPartial(captured.text)'))
        self.assertLess(stream.index('emitPartial(captured.text)'), stream.index('"type": "utterance_endpoint"'))
        self.assertNotIn('Task.sleep', stream)
        self.assertNotIn('"type": "transcript"', stream)
        stop = text.index('let stopped = cleanup()')
        freeze = text.index('sink.snapshot(freeze: true)', stop)
        self.assertLess(stop, freeze)
        self.assertLess(freeze, text.index('sink.end()', freeze))
        snapshot = text.split('    func snapshot(', 1)[1].split('    var result:', 1)[0]
        self.assertIn('lock.lock(); defer { lock.unlock() }', snapshot)
        self.assertIn('return (frozenText ?? text, textUpdates, failed)', snapshot)
        self.assertIn('preserveEmptyFinal: streamTurn', text)
        self.assertIn('preserveEmptyFinal && result.isFinal && updated.trimmingCharacters', text)

    def test_private_lifecycle_diagnostics(self):
        text = SOURCE.read_text()
        diagnostic = text.split('var diagnostics: [String: Any] {', 1)[1].split('let args =', 1)[0]
        for key in ('stop_reason', 'capture_end', 'recognition_task_state',
                    'finalization_timeout', 'result_callbacks', 'final_callbacks',
                    'has_text', 'error_present', 'total_frames', 'audio_seconds',
                    'text_age_seconds', 'voice_age_seconds', 'buffer_age_seconds'):
            self.assertIn('"' + key + '"', text)
        self.assertNotIn('"text":', diagnostic)
        self.assertNotIn('formattedString', diagnostic)
        end = text.split('    func end() {', 1)[1].split('    func update(', 1)[0]
        self.assertIn('guard !ended else { lock.unlock(); return }', end)
        self.assertLess(end.index('ended = true'), end.index('request.endAudio()'))
        self.assertLess(end.rindex('lock.unlock()'), end.index('request.endAudio()'))
        self.assertEqual(end.count('request.endAudio()'), 1)
        self.assertNotIn('task.finish()', text)  # instrument without changing finalization policy

    @unittest.skipUnless(platform.system() == 'Darwin' and shutil.which('xcrun'), 'macOS SDK required')
    def test_swift6_compile_only(self):
        binary = pathlib.Path('/tmp/h3-comet-speech-endpoint-build')
        arch = 'arm64' if platform.machine() == 'arm64' else 'x86_64'
        subprocess.run(['xcrun', 'swiftc', '-swift-version', '6',
                        '-strict-concurrency=complete', '-warnings-as-errors',
                        '-Xlinker', '-no_adhoc_codesign',
                        '-target', f'{arch}-apple-macos14.2', str(SOURCE),
                        '-o', str(binary)], check=True, capture_output=True, timeout=120)


if __name__ == '__main__':
    unittest.main()
