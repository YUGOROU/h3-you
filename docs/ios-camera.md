# H3 Camera — 実装と実機確認

## 検証状態

- **映像のみ**: ユーザーがChatGPT Voiceバックグラウンド会話 → 録画・停止・保存 → Mac転送・プレビューのE2E成功を確認。
- **ローカル保存**: 「このiPhone内 / H3 Camera / Recordings」の表示もユーザー確認済み。
- **音声実験**: v0.2で開始失敗を実機確認。H3 Cameraの観測経路はA25i HFP入出力・16kHzのまま。取得したJSONでは音声設定開始から約5msで拒否しており、経路変更待ち不足の可能性を排除できない。
- **v0.2.1（build 3）**: ユーザーがChatGPTの入力を **iPhone Microphone** に変更したことで、希望する会話＋音声付き録画の実現を報告。成功構成は **ChatGPT/H3 CameraともにiPhone内蔵マイク入力、ChatGPT応答はA25i出力**。この実機・設定での確認であり、全経路/全OSでの保証ではない。
- macOS上のSwift Testing 14件成功。対象はメタデータ、経路判定、診断保存、非同期コマンド順序、遅延した経路変更・安定判定・タイムアウト・キャンセル/割り込み。カメラ/音声ハードウェアやChatGPTの自動テストではない。
- 10分以上の連続録画は録画時間の報告待ち。

## 音声実験の最初の手順

ユーザー実機確認済みの構成:

| アプリ | 入力 | 出力 |
|---|---|---|
| ChatGPT | iPhone Microphone | soundcore A25i |
| H3 Camera | iPhone内蔵マイク | 再生なし |

1. 更新版H3 Cameraを開き、**「iPhoneマイク録音（実験）」**をON。
2. **「マイク権限を許可」**を押す。許可ダイアログは事前に済ませる。ONだけでは音声セッションを有効化しない。
3. A25i接続後、ChatGPT Voiceを開始。Background Conversationsを有効にし、コントロールセンターのChatGPT入力を **iPhone Microphone** に変更。応答がA25iから聴こえることを確認する。
4. H3 Cameraに戻って録画開始。切り替え待ち（最大約3秒）の後、**入力が `MicrophoneBuiltIn`** と表示されて録画が始まるか確認。表示される出力はH3 Cameraから見える経路であり、ChatGPTの経路を直接確認したものではない。
5. 20〜30秒会話する。例えば「今から青い傘と言います。何と言ったか返してください」と話し、録画開始後もChatGPTが新しい発話を聞いて応答できるか確かめる。応答がA25iから聴こえることも確認する。
6. 映像に手拍子を入れ、停止・保存。Macで自分の声と手拍子が聴こえるか、口/手の映像と音が合うか確認。ChatGPT応答中に黙る区間も設け、音声の回り込みを確認する。
7. MOV＋同名JSONをセットで保存。開始前/後/録画開始/経路変更/解放前の情報が `audioRoutes` に入る。
8. 問題がなければ1〜2分、次に10分以上へ進む。A25i切断や割り込みは捨て録画で試す。

**重要:** 現仕様ではコントロールセンターを開くなど、アプリが非アクティブになると録画を停止・保存する。録画中に設定を変えて継続する仕様ではない。短いテストごとに区切る。H3 Cameraのマイクモード選択がOSに表示される場合、環境音を残したい収録では「標準」を優先する。アプリはVoice Isolationを強制解除しない。

### 結果として教えてほしいこと

- ChatGPTが**新しい発話を聞き取れているか**、応答がA25iから出るか。
- H3 Cameraの入力/出力表示、停止・エラーメッセージ。
- 保存MOVに本人の音声があるか、同期と相手音声の混入状況。
- 失敗時は同名JSON（動画生成前の設定失敗ならJSONのみ）。私的な会話音声は必要がなければ共有しない。

H3 Cameraの入力が内蔵マイクでも、ChatGPTの聴取停止や経路変化をアプリ側から自動判定することはできない。実機で成功したChatGPT側の「iPhone Microphone」設定を基本手順とする。アプリからChatGPTの設定を自動変更することはしない。

## 実装の範囲

SwiftUI + AVFoundation。iPhone / iOS 18以降。依存ライブラリ、APIキー、ネットワーク通信なし。

- フロント広角カメラ。1080p/30fps優先、720p/既定fpsへフォールバック。画面のfpsは設定値でありドロップ実測値ではない。
- H.264 MOV、縦向き固定。鏡像プレビュー、非鏡像保存。
- 初期設定は**映像のみ**。このモード単独ではマイク権限要求・AVAudioSession取得/設定/有効化なし。音声実験から戻る際はそのセッション解放処理だけが必要。
- 音声実験は明示的なスイッチ＋マイク許可が必要。`playAndRecord` / `default` / `mixWithOthers` / `allowBluetoothA2DP` を設定し、`builtInMic` を優先指定。HFP入力の許可やスピーカー強制はしない。
- `automaticallyConfiguresApplicationAudioSession = false` を両モードで維持。音声実験時のみAVCaptureのaudio入力を追加し、同じMOVへ同期収録する。
- 内蔵マイクの優先指定は即時反映を仮定しない。100ms間隔で最大31回観測し、3回連続で内蔵入力なら開始可能とする（瞬間的に正しくなっただけでは開始しない）。開始直前も再確認。待機中の停止/非アクティブ化は開始タスクをキャンセルして後始末する。内蔵マイク以外のままなら開始拒否。録画開始時/経路変更時にも検査し、入力変更・音声割り込みで停止。通知配送までの短い区間に別入力が含まれる可能性はあるため、そのファイルをそのまま研究用に採用しない。
- 完了時に音声トラック数を検査。「トラックあり」は無音でないことやChatGPT共存の保証ではない。
- 終了/失敗時は音声入力を除去し、優先入力を解除、`notifyOthersOnDeactivation` 付きで音声セッションを解放。解放失敗はJSON/画面に表示し、次の録画開始/非アクティブ化で再試行する。失敗状態で黙って映像専用へ切り替えない。
- 専用actorでセッション操作。UIコマンドは直列化、遅延delegateはファイルID、音声通知は試行IDで検証。
- 容量1GB未満なら開始拒否。録画中500MiB残量制限、5秒MOVフラグメント、開始前JSONと原子的更新。
- 非アクティブ化で停止。短いOS猶予でファイル確定を試みるがバックグラウンド撮影はしない。
- 自動ロック抑止、空き容量/温度表示、critical熱状態で停止。

**制限:** クラッシュ復旧保証・自動修復・ドロップ測定・横撮影・独立WAV保存は未実装。OS/ChatGPT側の競合により録音開始が会話を中断する可能性がある。中断した会話はユーザーがChatGPT側で再開する。イヤホン利用でも音漏れ等による相手音声の完全除去は保証しない。

## 保存とメタデータ

保存先: `Documents/Recordings`。MOVとJSONは同じSession ID。設定失敗時も可能な限りJSONを残す。

- Files → このiPhone内 → H3 Camera → Recordings。
- アプリ「保存ファイル」から共有/AirDrop。
- USB接続MacのFinder → iPhone → ファイル → H3 Camera。

このフォルダはOSバックアップから除外する。**アプリ削除で収録データも消えるため手動でMacへ退避する。** 更新時は同じBundle IDを使い、アプリをアンインストールしない。共有先にユーザーがクラウドを選ぶ場合以外にアップロード処理はない。

JSON schema v2:
- 基本: `appVersion`, `appBuild`, `sessionID`, `requestedAt`, `startedAt`, `startCallbackUptime`, `endedAt`, `durationSeconds`, `location`, `device`, `format`, `orientation`, `mirrored`, `status`, `error`。
- 音声: `audioRequested`, `audioSource`, `audioRoutes`, `audioTrackCount`, `stopReason`, `audioCleanupError`。
- `audioRoutes`: 時刻、イベント名、入出力の名前/ポート種別、sampleRate、カテゴリ、モード、OSのactiveMicrophoneMode。v0.2.1から `availableInputs`, `preferredInput`, `categoryOptions` も保存。`preferredInput` は希望であって実際の経路ではなく、`inputs` と区別する。ChatGPT内部の音声設定は取得しない。
- `audioSource` は設定意図を表す。実測経路とトラック有無を併せて判断する。
- `recording_unfinalized`: 録画中/異常終了。`failed_to_start`: 設定失敗。`cancelled_before_start`: 待機中などの開始キャンセル。`completed_audio_unverified`: 保存できたが要求した音声トラック未確認。
- UTC ISO 8601小数秒。開始時刻はdelegate受信時刻で、厳密な最初の映像フレームUTCではない。deviceは一般機種名＋OS。
- 別端末録音では映像に手拍子を入れ、開始/終了の同期目印でクロックずれも確認する。

## 開発環境・ビルド

Xcode 27.0 (27A266a)、iOS SDK 27.0、XcodeGenあり。対象はiPhone 16 Pro / ユーザー申告iOS 27.2。シミュレータ追加不要。

```sh
cd ios
xcodegen generate
xcodebuild -project H3Camera.xcodeproj -scheme H3Camera \
  -destination 'generic/platform=iOS' \
  -derivedDataPath /tmp/h3-camera-build CODE_SIGNING_ALLOWED=NO build
xcodebuild -project H3Camera.xcodeproj -scheme CoreTests \
  -destination 'platform=macOS' -derivedDataPath /tmp/h3-camera-tests test
```

署名: `ios/Config/Signing.local.xcconfig`（git除外）に `DEVELOPMENT_TEAM` と `H3_BUNDLE_IDENTIFIER` を設定。ユーザーの既存Xcode設定をこのファイルへ移し、再生成で失われないようにした。共有 `Signing.xcconfig` が任意インクルードする。

Xcodeで `ios/H3Camera.xcodeproj` → Scheme H3Camera → 接続iPhone → Run。今回の署名付き成果物は `/tmp/h3-camera-device-build/Build/Products/Debug-iphoneos/H3Camera.app`。インストール済みだがマイク実験は自動実行していない。

## 残りの実機チェック

- [ ] 音声実験の希望経路、双方向会話、音声入りMOVの再生/同期。
- [ ] 音声実験の停止/失敗後、映像のみで従来の会話共存が維持される。
- [ ] 音声許可拒否でも映像のみは使用できる。
- [ ] A25i切断/音声割り込み時の停止・保存・JSON。
- [ ] 10分以上の録画、発熱、容量、重大なドロップ、破損がない。
- [ ] 開始/停止連打、開始直後ホーム移動、ロック、権限復帰。
- [ ] 捨て録画の強制終了後に未完了ファイルが残り区別できる。復旧保証はしない。

## 参照

- [研究計画](h3-you.md)
- [Apple QA1799: マイク選択・HFP入出力連動](https://developer.apple.com/library/archive/qa/qa1799/_index.html)
- [AVCaptureSession自動音声設定](https://developer.apple.com/documentation/avfoundation/avcapturesession/automaticallyconfiguresapplicationaudiosession)
- SDK `AVAudioSessionTypes.h`: A2DP出力と録音の併用は可能だが、他アプリとの入力共存保証ではない。ユーザー提供スクリーンショットではChatGPTの入力選択UIを確認済み。
