# Ordered person＋surroundings conditioning

> **実native確認（2026-11-03）**：一条件の結果・確認AV。3件のnative境界不具合を局所修正後、既存entrypointで二画像show→returnを完了。USER目視待ち。actual token/memory詳細の手元export不足は同報告に明記。現在Job終了/三Endpoint PAUSED。promptはnative入口のstripで変化しないcanonical文字列、DiT前tag検証はraw消費後のactual packed text span、global sequenceとsource-local indexを区別する。

> **CPU補強（2026-11-02時点）**：残り3点のCPU補強報告v2。実Qwen token内訳のencode前検証/cache/rank証拠、actual guide/packed rows、identity-only EXIF policyを実装・対象CPU検証済み。実Qwen/VAE/品質は未受入、新画像cloud承認は回答待ち。承認前にstage/upload/resume/GPUへ進まない。

既存 `research.launch_h3_direct_conversation` の明示opt-in。既定の単画像・baseモデル/adapter/pins/precision/reference1024/native124-22/decoder/PCMは変更しない。新画像の視覚品質は **unreviewed**。二枚入力が実モデルで有効であることはCPUテストでは証明しない。

## ローカル選択

repo外にJSONを置く（placeholderを実値へ置換）:

```json
{"schema":"h3-ordered-references-v1","references":[
 {"role":"person","path":"/private/person.png","sha256":"LOWERCASE_SHA256","width":9,"height":16},
 {"role":"surroundings","path":"/private/surroundings.png","sha256":"LOWERCASE_SHA256","width":14,"height":11}
]}
```

exact2、person→surroundings、別々のRGB8 PNG。absolute nonsymlink regular file、各64MiB以内、hash/PNG framing・CRC/元寸法をfresh確認する。PNG eXIfをbounded解析し、orientationなし/1のみ許可、非identity/不正/曖昧なmetadataを事前拒否する。このmetadata検証は全pixel decodeではない。元画像・既定画像の上書き、追加crop/retouch、face_crop併用、単画像fallbackはしない。

既存CLIに以下を追加すると**offline検証だけ**を行える:

```
--ordered-references-config /private/ordered-references.json
```

`--visual-intents --headless-interrupt --vox-parent /private/latest-campaign/vox` が必要。`--run`なしではresource/画像stage/upload/playerを開始しない。offline出力はportable identity/hashとunreviewed表示。

## cloud使用の境界

新画像のHF private staging＋既存H3承認地域での処理について、画像と範囲を示す個別承認が必要。Hub保管国は実行国と同じと保証しない。外部vision/公開/別provider利用は含まない。

`--run`と画像選択の併用には、明示的な`--confirm-private-reference-upload`も必要。**単なるgoからこのflagを立ててはいけない。** 未確認時はlease/API/画像copyより前に拒否する。確認後でも既存pre-GPU通知・latest parent・finite owner・独立cleanupを守る。この文書自体は実行許可ではない。

## binding経路

1. `ordered_references.py`: local path込みconfig digestと、content/order/preprocessのみのportable identity digestを区別。
2. `launch_h3_direct_conversation`: immutable rowにordered_reference_identity／ordered_reference_sha256。許可成立後だけexactbytesを`ordered-inputs/{person,surroundings}.png`へ0600保存しallowlistへ束縛。READYは同SHAを返さなければ拒否。
3. `hf_h3_direct_conversation`: identity/hashをstrict検証。staged bytesをsession所有のprivate tmpへcopy/hash/header確認、終了/失敗時cleanup。baseline original assetは保持し、別引数`ordered_references`で渡す。preexisting context envによるambient選択は禁止。
4. `LiveVoiceRuntime`: original AcceptedQualityProfileはbaseline provenanceのまま。candidate_references.pyの別bindingはbase settings snapshotと新contentを結び、旧source/prompt/visual reviewを継承しない。別private context/bindingを公開、同じnative requestへ二枚を渡す。
5. pinned native ordered presentation／Qwen processorをそのまま使う。両roleのpromptに画像1本人/画像2周囲の説明を含め、show/return/interruptで二枚を保持。カメラ意図は別のversioned intent。
6. cache identityは順序hash・前処理policy・actual role/intent prompt・candidate contentへ束縛。既存bounded bank/encoder lifetimeを維持。transport/commandのprofile SHAもbase role digest＋candidate identityを束縛。
7. 両rank conditioning証拠にsource hash、original/prepared geometry、actual latent geometry／native layout segments、R/T、candidate binding、fresh encode、worker memoryを保存。hostは両rankの同一画像・geometry・token証拠を照合する。candidate証拠が欠ける場合の再解釈/fallbackはしない。

## geometryと有限admission

nativeはEXIF transpose→RGB→aspect由来round32 canvasへLANCZOS resize（center cropなし）。元寸法からの予想とactual prepared sizeを照合する。VAEのactual latent `(t,h,w)`とnative `layout.reference_segments`のordered2 image segmentsを照合し、rows=`t*(h//2)*(w//2)`をsumする。継続guide audio/videoは固定画像Rから除外。

Qwenのactual Tはnative text indices/token tags由来であり、**VAE Rと同じ値とは限らない**。legacy singletonのT−R guardを二枚へ流用しない。exact native builder直後/encoder前のhookが同じ呼出しのactual image_grid_thw/merge size/counts/IDs/tagsから画像別内訳とlabel/vision境界/prompt overheadを検証する。再tokenizeしない。cold/refreshed/native memo/role cacheへproofを保持し、hit時は実cached tag hashも検証。両rankのqwen_tokensを照合する。

candidateの固定画像Rは実layoutの先頭二image segmentsから得る。guideはactual target shape、owner/window/token、video/audio tensor coverageと最後のaudio→video segmentsへ照合し、固定112控除を使用しない。packed sequence/index/segment/target row合計をDiT前に検証する。

候補の独立したengineering admission（native能力の受入ではない）:
- edge1024: image rows≤8192、Qwen total tokens≤8703、actual text overhead≤511、packed rows≤32768。
- cold bank edge512: rows≤2048、tokens≤2559、overhead≤511、packed≤16384。

orientation policyと全admission policyはportable identityに含まれ、immutable spec/READY hash・combined command/profile・cache/receiptへ束縛される。

overflowは拒否、truncate/fallbackなし。active candidateは1024固定、warm512へのimplicit switchは禁止。これらはtraining70000やnative text_len1024を流用した上限ではない。実利用のR/T/memoryはnative試験で別確認する。

memoryはworker全modelのcurrent allocated/reserved snapshotで、追加referenceだけのmemory差分ではない。fresh encode時間は既存native API区間内に含まれ、足し直さない。

## 検証範囲

初期接続結果v1（訂正あり）、補強結果v2。fake tensor/processor/rank JSON、exact pinned sourceのAST image/Qwen branch、overlay構文検証を含む。指定実PNGはorientation/hashをfresh照合し既存FFmpegの実pixel decode→nullも成功。ただしnative Pillow・実Qwen/VAE・GPU memory/性能・画像の効き・正面復帰/本人性/口パクは未受入。
