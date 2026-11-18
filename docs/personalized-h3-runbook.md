# Personalized H3: 実装と現在の実行経路

## 2026-10-12 Identity再学習の準備

新旧6原本の方針/実確認範囲は Identity preflight、歴史的loss調査は [loss audit](h3-identity-loss-audit.md)。USERは本番用収録を採用済み、人手採用715/除外130/未分類295。HF CPUで1140×39選別特徴量取得、自動gate0（原本/音声/写真uploadなし）。[guidance-target実装](h3-identity-guidance.md) と [共有cache loader](h3-component-cache.md) はCPU合成検証済みだが、**新GPU未投入、実前処理worker/legacy移行/weight整合性・費用/許可は未完了**。スモークは最小1〜3update、その後本番。以下の旧489件launcher/設定をそのまま再実行しない。

## 動いた範囲（2026-10-11の旧実装）

- `research/h3_personal/preprocess.py`: FFmpegのみでローカル前処理。原本のprobe、packet duration分布、メタデータで拒否済み素材を分離。映像と音声は同一seek原点、rotation焼き込み、timestampベース24 FPS化、cropなしのfit/pad。
- 今回は音声付き正常収録2本、588.696667秒＋1932.4秒＝2521.096667秒（42:01.10）。489クリップ、失敗0。拒否済み2本＋映像のみテスト1本は別管理。非発話も保持。各末尾はfull clipを少し重複して全尺を保持。
- 768×1344、124 frames（5.166667秒）、video-only H.264 MP4＋32 kHz stereo PCM WAV（165600 samples/channel）。音声末尾の8.33msはTrainerの40 Hz latent-gridに合わせたpad。
- 原本 `~/Downloads/Recordings/` は未変更。出力は非公開ローカル `$H3_PRIVATE_ROOT/`（約769 MiB）。`inventory.json`、`plan.json`、`manifest.json`、`summary.json`、`failures.json`を保存。
- `research/trainer_media_smoke.py`: 実際のTrainer `read_manifest/decode_video/extract_audio`の少数load/decode成功。その後、本番ジョブで全量cache作成と120step学習は完了（下記）。本人LoRA推論/品質は別途検証する。
- `research/h3_personal/references.py`: 原本30秒地点の画像＋8秒の音声候補を抽出。明瞭な本人発話であることは未確認。短い候補の試聴・画像確認後に使用。全文書き起こしや手動ラベルは不要。
- `research/h3_personal/e2e.py`: Agentが3回発話文を決定→local VoxCPM2→Ref2VA `lock_source`→元TTS音声mux→逐次ブラウザqueue＋demo.mp4自動結合。実モデル接続は未検証。生成済みsegment単位の逐次配信でありTrue Causal Streamingではない。

## 再実行

```sh
python3 -m research.h3_personal.preprocess \
  --source ~/Downloads/Recordings --output $H3_PRIVATE_ROOT
python3 -m unittest discover -s tests -v
```

同じ設定・原本で再開可能。設定/size/mtimeが変わる場合は別出力先を使う。空き容量1 GiB未満は停止し、原本を削除しない。全出力をprobe・decode検証する。VFR判定はpacket duration分布を記録し、厳密な全PTS分析とは区別する。

少数clipのTrainer読み込み環境はproject依存と分離（現在 `/tmp/h3-media-env`、trainer `/tmp/h3-trainer-inspect`）。再作成例:

```sh
git clone https://github.com/AviadDahan/MiniMax-H3-Trainer /tmp/h3-trainer-inspect
git -C /tmp/h3-trainer-inspect checkout 7a1563a267a17da898df3396acbd6a9d24fcb142
uv venv --python 3.12 /tmp/h3-media-env
uv pip install --python /tmp/h3-media-env/bin/python av numpy torch Pillow safetensors accelerate \
  'transformers>=5.15,<6' 'pydantic>=2.7' \
  'diffusers @ git+https://github.com/huggingface/diffusers.git@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc'
OMP_NUM_THREADS=2 /tmp/h3-media-env/bin/python research/trainer_media_smoke.py \
  --trainer-root /tmp/h3-trainer-inspect \
  --manifest $H3_PRIVATE_ROOT/manifest.json \
  --geometry 768x1344x124 --samples 2 \
  --output $H3_PRIVATE_ROOT/media-smoke.json
```

キャッシュ生成後は同コマンドへ `--loader-cache /path/.precomputed` を追加し、実際の`PrecomputedDataset.__getitem__`を確認。架空tensorのテストで完了扱いにしない。

## 本番学習への切り替え

ユーザー指示で独立した少数学習スモークは省略。`research/launch_hf_train.py`で全489クリップとコードをprivate dataset `YUGOROU/h3-personal-20261011-data`へアップロード済み。private出力repo `YUGOROU/h3-personal-20261011-lora`も作成済み。本番ジョブ `6acbac0d095c578089320054` はCOMPLETED。2026-10-11 15:32UTC開始、20:40UTC学習終了。全量cacheとcheckpoint10..120のprivate保存を確認。最終stateはstep120/samples_seen120/skipped_long0、rank16/alpha16。学習部2.308h、ジョブ全体推定$25.63（請求額ではない）。ジョブ: https://huggingface.co/jobs/YUGOROU/6acbac0d095c578089320054 。

本番構成は全量潜在化→120 updates（約0.245 epoch、rank16）→10stepごとのcheckpoint/private保存。`research/configs/ref2va_audio_lock_train.yaml`、worker `research/hf_train_personal.py`。H200 $5/hr、hard timeout8h＝このジョブ上限$40。潜在キャッシュもprivate保存し、別GPUジョブによる少数確認は実行しない。

ユーザーが公式MiniMaxAI Space/model hostingを根拠にHF上で進めると確認したため投入した。公式Space存在は確認したが、実際のJobs地域は独立検証していない。確認根拠をprivate `production-plan.json`へ記録し、全HF配置への包括的な許諾確認とは扱わない。

以下は再実行用（**現在のジョブを重複起動しない**）:

```sh
/tmp/h3-media-env/bin/python research/launch_hf_train.py --execution-territory-confirmed
```

フラグは地域の未確認を無視するためのものではない。過去H3費用がある場合`--previous-h3-gpu-usd`に指定。最大$40に加えE2E用$50を残せない場合は開始しない。

## 学習前のゲート

1. HF認証は保存済みtokenで解決済み: `export HF_TOKEN="$(< ~/.cache/huggingface/token)"`。ハーネスの環境変数が原因だった。Jobs全件閲覧/private upload成功。トークンは出力・チャット貼り付けしない。
2. [H3 LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE)はUSA/EU/UK/Koreaを対象地域から除外し、域外での実行・出力使用等も制限。利用者の所在地だけでなくHF GPU実行地域も確認する。海外GPUなら自動的に解決すると仮定しない。別モデルを改善するためのH3出力使用にも制限がある。
3. 画像/音声候補の明瞭さ・第三者/個人情報のscreeningは未実施。素材・参照・checkpointはprivate。ユーザーの本番学習指示に基づき派生クリップ/同期音声/参照画像をprivate HFへ送信したが、原本MOV/route metadataは送信していない。
4. 元H3のDiffusers形式（`transformer_ref/`）とvLLM形式（`Ref2VA/transformer/`）は異なる。rank16 attention adapterをvLLMのrank128 Turbo loaderへ渡さない。本人adapterのstrict load→直接diffusers生成は下記の1本で確認済み。vLLM/Turbo serving互換は未確認であり、直接生成成功から互換を推測しない。
5. `research/configs/ref2va_audio_lock_smoke.yaml`はschema-only検証済みの候補。Ref2VA、rank16、画像reference probability=1、video generated/audio frozen。既存default Ref2VA configのaudio生成と区別する。これは以前の少数検証候補。現在はユーザー指示により本番120step構成へ切り替え、途中checkpointの時間/費用を記録する。
6. HF公開hardware listingではH200 $5/hr、RTX PRO 6000 $2.75/hr、H200×2 $10/hr。空き/権限/地域/今回の形状でのメモリ収容は未検証。本番候補はH200、同ジョブで全量VAE/text encodingから学習へ進む。ローカル残容量約3 GiBなので大型モデルをMacへdownloadしない。
7. 学習ジョブは完了、推定GPU費$25.63。累計$150・単価$10/hrを超えず、E2E用予算を残す。学習・推論ジョブにtimeoutを付け、成功/失敗後に停止。今回の単発推論は追加最大$10、学習との合算最大推定$35.63。

Trainerへの原料は `manifest-ref2va.json`（絶対パス、`video/audio/caption/reference_image`）。学習の直接入力はMP4ではなくTrainerの`.precomputed/index.json`とVAE/text/reference safetensors。HFへ移すときはprivateで保存し、manifestの絶対パスを現地パスに書き換える。現在は本番ジョブで`process_dataset.py MANIFEST --model-path MODEL --references --resolution-bucket 768x1344x124`により全量cacheを作り、通常の学習loadへ進む（ユーザー指示により独立少数検証を省略）。clip-ID hashによるtrain/val分割は隣接clipが漏れるため、今回smokeはval無効（quality評価の成功とはしない）。

## 本番監視

ユーザー指示で連続watcherは停止済み。以下の`--once`は一度だけ取得する。watcherモードではジョブstate・概算費用・private checkpoint一覧・直近ログを毎60秒で `$H3_PRIVATE_ROOT/production-monitor.json`へ保存。terminal stateで終了し、自動再試行はしない。Mac停止時もHFの8h timeoutは有効。

```sh
/tmp/h3-media-env/bin/python research/monitor_hf_train.py \
  --job-id 6acbac0d095c578089320054 \
  --output $H3_PRIVATE_ROOT/production-monitor.json --once
```

完了時の実測: media encoding約18秒/clip、全量cache/text処理は18:21UTCまでに完了。120step学習は約2.308h（約69秒/update）。最終checkpoint120を確認済み。

前処理効率のコード確認: batch1・CPU decode/PIL→VAE→audio→referenceの逐次実行で、先読みなし。FP32 VisualVAEの既定256px tile/64px overlapは768×1344で7×4タイル、124framesは17frames×8chunk、計224回の逐次target encoder呼び出し。既定tilingは公開latent契約なので、単なる無効化/大型化やBF16化は安全な高速化とは扱わない。同一tile形状をbatch化する方針には数値整合・実測確認が必要。

今回caption＋参照画像は489件すべて同一だが、現Trainerは画像referenceとQwen条件を毎件再計算。model revision・processor設定・参照内容・caption等を含む同一キーで再利用すればこの重複を除去できる。現ジョブには未適用。HF metrics APIが応答タイムアウトしたため、GPU使用率/VRAM/CPU律速の実測は未取得。実行中のジョブを止める変更は行わず、最終的に全量cacheとcheckpoint120が保存された。

## 単発の目視確認（Agent・サービスなし）

`research/launch_hf_infer.py` / `research/hf_infer_personal.py`を使用。今回ユーザー指定は固定文のVoxCPM2 Zero-shot→本人H3 LoRAの1本のみ。常駐Agent、LLM呼び出し、会話サーバーは不要。

- 学習repo revision `78e0a753755b32138ee2ed48515a8865091bc257` の `runs/20261011-182118/checkpoint-0000120` を明示指定。raw Trainer adapterを同じdiffusers `transformer_ref`へcheckpointのalpha/target指定で適用、全adapter key/shapeを検証して不一致時は失敗。
- Trainerの普通の `generate.py --reference-audio` は追加参照音声であり、外部音声を固定したtarget timelineではない。今回のworkerはsource音声をtarget audio rowsへ置き、全stepでclean/timestep1を維持する。
- 公開`MiniMaxH3Scheduler`の50schedule points（49forward）、seed42、768×1344。TTS全尺＋0.25秒を覆う合法17*n+5フレームbucket（>=124、<=15秒）を選ぶ。BF16 denoiserとFP32 VAEs。TTS/condition/infer/decodeを別プロセスにしてVRAMを解放。最終MP4音声は元VoxCPM2 WAV。
- VoxCPM source/model、H3/Trainer/Diffusersをpin。TTSをprivate保存してからH3を進め、失敗しても保存済み音声を残す。H200$5/hr、HF hard timeout2h/max$10、worker110分。自動retryなし。
- 既存30秒地点の音声候補はmean -49dBでほぼ静かなため、2本目93秒地点の8秒を音量から再選択。本人声/明瞭さ/第三者screeningは未確認。参照とTTSはユーザーが試聴して判定。
- 初回 `6acc0bc0fee2c9007019d90f` はTTS3.04秒生成、condition encode、LoRA416テンソルstrictロードに成功後、noise channelの実装ミス（raw24とpatchified96の取り違え）でdenoise前に失敗。停止確認・約$0.415。`packed_video_channels`とshape検証を追加、20 CPU tests通過。音声は保存・再利用。
- 修正後ジョブ `6acc0d64095c578089324dc3` はCOMPLETED/GPU停止済み。private output prefix `inference/review-20261011T222744Z`、revision `695c8c6831afc2936818812991f275050991faf6`。LoRA416テンソルstrictロード、sequence52084rows、49 denoise forward/772.855秒、FP32 VAE decode、元TTS mux成功。動画768×1344/24FPS/124frames、5.166667秒、音声5.166秒。全decode/A/V検証成功。ユーザー目視評価: 顔は本人に似ておらず、服装特徴をかろうじて維持する程度。背景一貫性はOKだが、デノイズ的な低画質/粗さが目立つ。本人再現成功とは扱わない。本人声・口パク評価は未提示。その後ユーザーは参照画像・LoRA強度・デノイズ設定の比較を承認し、LoRAなし比較は除外するよう明示（次節）。
- この再実行は保存TTSを再利用、初回停止確認後のみ開始。費用はHF started_at/finished_atから初回推定$0.4149＋修正後$1.4006、学習$25.6301と累計推定$27.4456（実請求ではない）。現在ジョブはすべてterminal、ローカル連続watcherなし。
- ローカル完成成果物 `$H3_PRIVATE_ROOT/inference-review/review-20261011T222744Z/`: `review.mp4`、`review.html`、`tts.wav`、参照画像/8秒音声、`preview.png`、`contact-sheet.jpg`、`adapter-load.json`、`review-status.json`、`local-validation.json`。比較ページは静的ローカルファイルで、サービス起動は不要。新規再生成時は累積の両推論費も予算に含める。
- ローカル `$H3_PRIVATE_ROOT/inference-review/latest-submission.json` にID/path保存。Macには生成成果物のみdownloadし、大型モデルは置かない。

再起動用（**既存ジョブが稼働中なら起動しない。新規課金が発生する**）:

```sh
/tmp/h3-media-env/bin/python research/launch_hf_infer.py
```

完了後の取得・ローカルA/V検証・オフライン比較ページ作成（GPU/サーバー起動なし）:

```sh
/tmp/h3-media-env/bin/python research/fetch_hf_review.py
```

`latest-submission.json`の現ジョブがCOMPLETEDかつ成果物status成功・strict LoRAロード済みであることを確認し、該当prefixだけdownloadする。MP4全decodeとA/V時刻・尺を確認し、`review.html`で参照画像、生成動画、TTS、8秒参照音声を比較する。これは人間の品質判定を代替しない。

## 再学習なしの改善比較（LoRAは常時適用）

ユーザー提供のDesktop画像5枚を目視選定。採用は `スクリーンショット 2026-10-12 8.04.42.png`（比較的明瞭な目鼻口・輪郭、少ない左右向きの差、手の遮蔽なし）。次点8.05.48。8.03.24は少し柔らかく、8.06.00は顔の傾きと帽子を持つ手、8.00.33はblurが目立つ。いずれも多少の見上げ角度は残る。原本は未変更、選定画像のみprivateへcopy、filename/SHA256 provenance保存。手動crop・顔修復・enhanceはしない（モデル標準前処理は通常通り）。

| case | LoRA強度 | schedule点数 / forward |
|---|---:|---:|
| reference-scale1-50 | 1.0 | 50 / 49 |
| reference-scale05-50 | 0.5 | 50 / 49 |
| reference-scale15-50 | 1.5 | 50 / 49 |
| reference-scale1-100 | 1.0 | 100 / 99 |

新参照・TTS/drive WAV・prompt・seed42・768×1344/124frames・flow_shift12を全caseで固定。同じencoded conditioningをcacheし、一因子ずつ比較。旧参照の前回LoRA-on動画は参照変更効果の比較用に残す。再学習、LoRA-off、Agentは実行しない。

Workerはpinned diffusers `model.set_adapters('default', weights=scale)`を使い、全LoRA層の実scalingが`scale*alpha/rank`と一致することを確認。Tiny CPU diffusers+PEFT実モデルでAPI/scaling確認済み、22 CPU tests通過。ただしGPU生成/品質向上の証明ではない。

現比較ジョブ `6acc18b0095c5780893255c7`、run `review-20261011T231554Z`。Input private dataset revision `9baf0bb2cc47c6b99e3210d9e53d22ba2339597d`、output model repo prefix `inference/review-20261011T231554Z`。H200$5/hr、HF90分hard timeout/max$7.50、worker82分。既消費推定$27.446に足して最大約$34.946。実測forward~16秒で50点の5秒動画でも~13分。第一case `reference-scale1-50`は完成・private保存されたが、ユーザーが品質を悪いと評価し、残りのcancelを要求。2026-10-11 23:44UTCにHF **CANCELED** を確認、残り3条件は未完了。今回推定GPU費$2.3751、累計~$29.8207（請求額ではない）。リポジトリのstatus JSONはcancel時に更新されずrunningのままでも、HF terminal状態を優先。保存済み第一caseは保持。再実行・自動再試行・連続ローカル監視なし。

新規課金を伴う再実行例（**キャンセル済み比較は自動で再開しない。新規実行にはユーザーの指示が必要。下の既消費額は過去の起動時の値なので必ず更新すること**）:

```sh
/tmp/h3-media-env/bin/python research/launch_hf_infer.py --comparison \
  --reference-image "$HOME/Desktop/スクリーンショット 2026-10-12 8.04.42.png" \
  --reuse-review "$H3_PRIVATE_ROOT.parent/inference-review/review-20261011T222744Z" \
  --previous-inference-usd 1.816
```

完了後は `research/fetch_hf_review.py` で取得。途中の完成caseだけを一度取得する場合 `--allow-partial` を追加（watcher/自動更新ではない）。新runの `review.html` は保存済みcaseの動画と新参照・同じTTS、前回結果へのリンクを表示。今回はcancel済みなので `--allow-partial` が必要。ユーザーの第一case評価は品質不良であり、改善成功とは扱わない。

## E2E実行（サービス準備後・将来の会話デモ）

VoxCPM2は別環境へ `voxcpm soundfile` をinstall。`VoxCPM.from_pretrained(..., load_denoiser=False)`と`reference_wav_path`を使用し、transcriptは不要。サービス契約参照: [vLLM H3 recipe](https://github.com/vllm-project/vllm-omni/blob/47776fd350f55f21e016d58167129476497cb3e5/recipes/MiniMaxAI/MiniMax-H3.md)。

```sh
python3 -m research.h3_personal.e2e \
  --llm-model YOUR_MODEL \
  --reference-image $H3_PRIVATE_ROOT/references/portrait.png \
  --reference-audio $H3_PRIVATE_ROOT/references/voice-candidate.wav \
  --output $H3_PRIVATE_ROOT/demo-001
```

- デフォルトLLM: localhost:8000 `/v1/chat/completions`、H3: localhost:8091 `/v1/videos/sync`。APIキーは`H3_LLM_API_KEY/H3_VIDEO_API_KEY`環境変数。
- ブラウザはlocalhost:8765、Startボタン後に完成segmentが順次再生。最後に`demo.mp4`を保存。音声はH3再構成ではなく元TTSを使用。
- remote personal-data送信は`--allow-remote-personal-data`明示時のみ。元音声参照はlocal TTSだけで使い、画像と生成音声はH3 endpointへ送る。15秒超のTTSは切り捨てず失敗させる。
- 本人LoRAはclientから要求しない。server側adapterも未検証扱いで記録。duration整合検証はlip-sync品質の証明ではない。
- テストはmock model接続＋実FFmpeg合成fixture。**本人会話デモの生成成功ではない**。
