# Accepted profile・同期AV・conversation handoff設計

基礎設計は plan 1〜3。USERは1024を理想、512も良いと評価。その後の明示指示でASR/Pi/TTS→Discord E2Eまで拡張し、fresh E2EはUSER成功確認済み。現改善はdelivery診断、増分TTS、傾聴動作、承認済みSP。Endpoint独立化は設計案で、無断deployや新しい高速化探索を許可しない。

2026-10-27：explicit sentence-incremental worker/launcherをCPU接続。原native PCMは1発話1resamplerでcanonical化、未seal ingressへ順次供給しtrueEOSだけでsealする。通常full-WAV defaultは保持し、実TTS/GPU品質は未受入。Mac/GPUのstage timingも接続したが、pipe ACKと実再生／remote受信は別。詳細 `h3-streaming-runtime-integration-and-pi-evaluation-20261027.md`。

## 所有と時計

```text
resident model owner（単一、warm中のreloadなし）
  ├─ immutable quality profile: reference1024 / 明示reference512
  └─ conversation continuation owner［会話経路に接続済み］
       ├─ sourceごとのimmutable Session
       ├─ global committed-latent clock
       ├─ persistent native decoder / emitted-frame clock
       └─ original PCM timeline（source + 明示silence/padding）
              ↓ committed video callback
       AV Delivery（pending+leased合計<=2）
              ↓ lease（video+originalPCMを一組）
       single continuous FFmpeg sink → ACK / terminal failure
```

model常駐とcontinuation ownerの継続は別。sourceを固定した既存Sessionを、source差替え可能にするだけではhandoffにならない。

## Profile固定（実装済み）

`accepted_profiles.py`のfrozen profileに、元参照SHA順序・prompt SHA・loaded processor SHA・prepared geometry/R/T・resize policy・output/fps/window/overlap・BF16/native FP32・SMD4/four forwards・評価済みreview SHAをbindingする。

factoryの主候補はreference1024、512は明示選択。既存runtime/productionのdefaultは変更しない。`LiveVoiceRuntime(quality_profile=...)`が明示的にこの検証profileを使う。warm変更は`configure_quality_profile`で完了Session境界のみ、同じ元参照/prompt/processor/geometryを要求。従来の`configure_conditioning`で無言にedgeだけ変えない。

Qwen解放前に用意する別profile cacheを混用しない。新しいlistening promptは別conditioning identityが必要で、speech品質のUSER承認をlistening動作の承認に拡大しない。

## 基礎AV実装（単一source adapter、現会話はpaired AV）

`av_delivery.py`：immutable Binding/Segment、global frameからFractionでPCM sample境界を導出。原PCM producerは1つ。segment119/102/107の元sample範囲は48kで0:238000、238000:442000、442000:577920。残り78080samplesは明示transport silenceであり、元音声でもaudio-VAE条件でもない。

`incremental_av.py`：既存`render_continuation(on_segment=...)`を接続。まだ未確定なlatentを公開せず、rendererが両rank commit/decode/coverage/segment検査後に出すvideo fileを受ける。Session/index/hash/PTSと元WAV/PCMを検査し、bounded queueへ。consumerがvideo+audioの両pipe受入後にのみlogical ACK。queue上限2、timeout/cancel/consumer errorはterminal、ACK不明の無条件再送なし。

`headless_av_sink.py`：旧`LocalPlaybackSink`（clipごとffplay）は使わない。既存`buffered_stream._asset`のpath/hash保護を再利用し、**1本のFFmpeg rawvideo+S16LE→FFV1/PCM NUT**へ送る。input clipは各々decodeするが、clockと出力encoderは切り替えない。geometry/fpsをprobeし、勝手なscale/resampleなし。元PCMと明示tail padだけを送る。最終review muxは監査用で、配信の二つ目の音声producerではない。

raw入力は全形式を明示し`-nofind_stream_info`を使用する。`analyzeduration=0`はprobeをゼロにせず、future PCM待ちで最初のacceptを止め得るため使わない。`nobuffer`による先頭frame破棄も使わない。dual writerはbounded pipe、期限、cancel/reapを持つ。spawnとcloseを直列化し、reap後の新childを拒否する。

`paced=True`はFFmpegの両raw入力をreadrate=1で消費するheadless試験。最初のvideo/audioのpositive write時刻を別記し、両方が始まった時刻をfirst AV pipe submissionとする。segment全体のACK時刻とは分ける。これはdevice/display clockや外部delivery durabilityではない。

### failureの意味

ACKはin-process pipe acceptance。external crash-safe exactly-onceではない。sink failure時にAVの一方だけcursorをadvanceしないが、既にOSへ送ったbytesは取り消せない。partial outputは成功artifactとして公開しない。

adapterはruntimeをclose/restartしない。ただし中途turnがlocal sink failureで中断した場合の、残るSession/decoderを安全にabandonする復旧契約は未完。正常weightsを保持したことと、そのcontinuation ownerが再利用可能なことを区別する。

### Conversation ownerの終了／retirement

Conversation経路には、serial generate RPC境界でのみ実行する`conversation_control`を実装している。`abandon`（local sink/preparation failure）はmodel health確認後にowner/decoderを破棄し、waveform historyも除去する。native forward中やhealth不明の状態からの安易な再利用は拒否する。単一source adapter全般の復旧を保証するものではない。

正常終了は`action='retire', reason='completed'`。conversation-finalのdecoder ACKが完了し、`emittedF == committedF`のときだけ許可する。前のownerをretireしてもweights/cacheは再ロードしない。退役済みconversation ID再利用を拒否し、既存の有限retired-ID数制限を保持する。native Session IDもconversationごとに一意にしてimmutable receipt衝突を防ぐ。同一warm modelでの2conversation逐次実行は`h3-late-listening-investigation-20261024.md`のGPU試験で確認した。

### Managed live adapter（接続契約と初期CPU検証）

`live_conversation.LiveConversation`はappend-onlyなPCM範囲を所有し、source EOS時にdecoderをflushせず、遅延5frame分の元PCMも次の傾聴windowへ引き継ぐ。傾聴に再入場する際だけ新しいlogical idle epoch/index0を作り、連続idleごとに偽sourceを作らない。native Session filenameはopaque conversation IDのhashから生成する。

`live_native_backend.LiveNativeBackend`は起動済みruntimeを1つだけclaimする。role変更でruntime側profile digestが変わるため、固定bindingとの照合はspeech-roleへ正規化して行う。native/backendまたはcleanupの不確実性はadapterだけでなく親runtimeもquarantineし、通常/independent生成を拒否する。許可するのは同じownerの明示release controlだけで、rank側のhealth/inflight gateを通ったcleanupとreceipt退避が成功するまでreadyへ戻さない。ACK済みnative release後のlocal退避修復はRPCを再送しない。元から別理由でdegraded/closedのruntimeを勝手にreadyへ戻さない。

Sender/receiverはverification receiptとsink pipe ACKを別扱いにし、credit返却はcertainなFIFO sink ACK後のみ。最大2組、固定deadline、partial IO/ambiguous ACK後の再送禁止。ACK readerはforeground lifetime内で終了/joinする。詳細とfake-native→実FFmpeg634frame検証は`h3-live-conversation-sender-integration-report-20261025.md`。これはHF認証経路、実GPU新規入力、OBS/Discord受信の証明ではない。

## 実handoff：設計境界と実装契約

現Sessionはsource/hash固定、first Windowはguideなし、final commitでtailを破棄する。decoderはSession終端で5framesをflushしbuffer/overlapを閉じる。`conversation_state.Handoff`文字列receiptだけではこの問題を解決しない。

次の実装は、新しいconversation-level ownerがtensor-bearing visual tailとnative decoderを持ち、sourceごとのimmutable Sessionを明示的handoff adapterで接続する設計とする。旧sourceを新canonicalへ偽装して混ぜない。既存Sessionをsource-bound lock/commit検査に再利用する設計でも、モデル用global WindowInputとSession自身のlocal WindowInputを区別し、外側adapterが初期guide・suffix discard・global offsetsを実tensorで検証する必要がある。`ctx.token`を差し替えるだけではSessionのcurrent-token／累積検査と矛盾する。このadapterは現在`conversation_continuation.py`と`conversation_decode.py`に実装されている。以下は単一Sessionとの境界の説明であり、最新の検証状態はa local status note。

直前のcommitted endpointをF=17n+5とすると、次sourceの最初のguided targetは実`Window(F-22,F+102,22)`。前7clean video latentsをguideにして、新30latentsだけをappendする。global RoPE temporal offsetは5(F-22)/3。

A(f)=round(5f/3)とし、guide audio長はA(F)-A(F-22)、targetはA(F+102)-A(F-22)。**常に37とhardcodeしない。** 新sourceに先行する実波形区間を新source波形とともにaudio VAEでencodeしてguideをresetする。旧sourceのlatentをそのままcopyしたり、zero latentへ置換したりしない。`conversation_audio.audio_origin`がmodel開始位置・onset・preroll数・容量を定義し、encoderとownerが同じ契約を検証する。`prepare_waveform`は直前sourceのpadded波形末尾だけをCPU上に保持し、次sourceに前置する（傾聴後は無音、発話後は残存する発話末尾とpadding）。prefixは`[A(F)-A(F-22)]*800` samples@32kであり、22/24秒の独立丸めではない。全sourceがprefix込みで容量内に収まらなければ拒否し、切詰めない。外部再生用original PCMは一切前置・移動しない。A(f)の丸めたlatent境界とf/24の元PCM onsetは常に一致するわけではない（例F=124で207×.025=5.175s対124/24=5.1667s）。preroll waveformのsample配置とlatent phaseを明示して検証し、区間ごとの独立丸めやsourceの暗黙切詰めで辻褄を合わせない。

未flush時はcommitted endpoint Fに対してemitted videoはF-5。したがって新sourceの最初のdecoder出力には旧sourceの遅延5framesが含まれる。音声onsetはF/24であって(F-5)/24ではない。単一source AV Bindingを流用せず、`conversation_timeline`/`conversation_av`がsource/synthetic-silence spanを持つ連続clockを管理する。

最低targetは124frames。最初のlisteningを都合よく短く切らない。一例は初期124、speechの3guided windowsで+306、復帰listening+102、最後だけnative final flush。F=124→430→532、video22.1667秒。speech onset124/24、元12.04秒全文と残りpaddingを明示する。これは設計上の算術で、GPU品質/速度の測定値ではない。

## 検証の分離

旧GPU前チェックリスト（実装・検証の品質合格や一般的な障害回復の完成とは別）：

1. listening/speech prompt bankをQwen解放前にprecomputeし、warm miss拒否と役割bindingを実装。
2. encoded-silence waveformを既存audio VAEで処理。自然な閉口/傾聴はGPU+USER確認が必要。
3. 実tensor tail export/import、新source audio preroll/reset、global layouts、両rank failure agreement。
4. source境界でfinal flushせず、同じdecoder stateへsuffixだけfeed。最後だけflush。
5. conversation PCM spans、PREPARING優先度、未commit予約cancel/drain、local failure recovery。
6. finite scenario worker/launcher/allowlistと対象回帰を完成させてから、既存scope内で最小のGPU試験。

機器出力・routing・Discordは別承認。headless通過を実device再生や新規会話E2Eの合格にしない。
