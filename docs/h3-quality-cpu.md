# HF CPUでの顔パーツ・向き・品質解析 (2026-10-12)

## 実行したこと

USERは画像選別の重い処理をHF Jobsへ逃がすことを許可。本人照合ではなく顔ランドマーク/向き推定が目的。30件の追加基準合わせ完了（顔7/動作8/除外15）後、448件の人手判断と未分類review365件を解析した。

- `launch_hf_quality.py`: デフォルトはローカルstageのみ、`--submit`でCPU専用を明示投入。cached tokenを環境変数に設定、値は表示しない。private専用dataset、active重複チェック、1h timeout、価格ガード<=$0.05/h、研究累計<=$150、retryなし。
- private入力はuniformな既存gray320px×4〜5時点filmstrip813件、判断snapshot、必要コードのみ。37,904,202bytes。原動画/音声/Reference写真/絶対パス/route JSONなし。
- `hf_quality_worker.py`: MediaPipe FaceLandmarker IMAGE/CPU delegate、最大2顔。478点からscalar品質feature18種類だけを抽出。顔サイズ/顔・目・口ROIの鮮明さ、露出、画面内率、canonical-face camera-relative姿勢proxy、開眼proxy、中心/面積変動。raw landmarksや本人embeddingは保存・分類しない。顔が見えない時は不確実性扱い。
- モデル: Google公式asset `face_landmarker/face_landmarker/float16/1/face_landmarker.task`、3,758,596bytes、SHA256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`、object generation1683136941916318。小さいassetをローカル取得してhash確認したが推論はHFのみ。Python MediaPipe source/codeはApache-2.0; このジョブはH3重み/Training Adapterを使わず、H3地域許諾に関する証明を兼ねない。公式手順: <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python>。
- Job `6accc076095c57808932d1e5` COMPLETED、cpu-upgrade(8vCPU)、$0.03/hr、2026-10-12 11:12:01.289–11:13:48.962UTC。worker69.035s、Job107.673s、概算CPU$0.000897（課金の分丸め/最低料金/請求額は未検証）。全既知累計概算$29.821597。学習GPUは未投入。
- private入力repo `YUGOROU/h3-quality-quality-20261012T111136Z`、input revision `1cd69ac6749accbaa21906b91a41af8596d01930`。local artifactsは `identity-selection/quality-jobs/quality-20261012T111136Z/`。公開リンク不要。

依存はJob内のみ: Python3.11-slim-bookworm、numpy1.26.4、mediapipe0.10.21、opencv-contrib-python4.11.0.86、scikit-learn1.6.1、huggingface_hub1.33.0。CV/BLAS threads限定、GPU不可、model hash検証、preview byte hashes検証、tar traversal拒否。50件毎にprivate status/features保存。

## 自動判定の検証と結果

`quality_policy.py`: balanced RandomForest200trees/depth6/min_leaf5/seed42、session/time/本人特徴はpredictorに含めない。セッション毎の連続時間帯4fold、heldout片から30秒以内のtrain片をpurge。capture自体が独立した汎化評価ではない。

自動提案条件: detection>=.8、vote>=.8、margin>=.25、training feature範囲/距離内、OOF予測のglobal同class support>=5/precision>=.9、対象session support>=3/precision>=.8。voteは正答確率ではなく、precisionは小さい実測値で保証ではない。

**実結果: 813件処理、処理error0。未分類365件のauto_proposalは0、review365。**

- identityのglobal eligible OOF53件中50正解（.9434）でも、昨日2sessionのidentity eligible supportは0でsession gate失敗。これを残り365件の94%精度と表現しない。
- motion eligible0、reject eligible1だけで不足。365件中355はvote閾値未達、179はfeature support外、52は十分な検出率に届かず（理由は重複）。閾値を後から緩めない。
- このrunから、残りを信用して自動分類できるとは言えない。自動除外・人手ラベル上書き・automatic training approvalは一切行っていない。
- gray previewはUSERが判断したカラーpreviewと情報量が異なる。低角度・帽子・ブレや動作ラベルの意味はscalarモデルだけでは不十分。MediaPipe出力があることも顔細部の正しさを証明しない。

## 採用方針

未判断片を全件手動分類する必要はない。`image_selection.py`で確実な人手採用279identity+66motion=345件（重複tailを含む1782.5s=29:42.5）を暫定 `selected_manifest.json`へ整理済み。人手除外103件、残り692件は未分類（今回review365＋元auto triage327）で**除外ではない**。

このmanifestは原本rangeとhashを持つ。`review_scope=sampled_color_frames`、USER音声安定申告、`full_clip_review_complete=false`、`training_approved=false`を保持。画像だけで全clip/話者/同期の独立検証を名乗らない。採用pool実比279:66≒81:19と学習sampling目安65:35は別物。未実施のresamplingで達成扱いにしない。

次はこの暫定採用分を軸にReference整備・loss対策・cache/loader統合を進める。追加の自動品質探索やGPUを勝手に始めない。残りは必要になった場合だけ少数追加確認する。

## 追加教師30件後の再検証

追加教師はidentity11/motion6/reject13、review側の人手ラベル計478件（identity290/motion72/reject116）。未分類review335件。

`launch_hf_quality_refit.py` / `hf_quality_fit_worker.py` を追加。前回のrevision-pinned813 scalar feature rowsをrange/source hash/feature namesで照合し、そのまま再利用。今回アップロードは小さいfeature JSON・判断snapshot・コードのみ。画像/音声/モデル再uploadやMediaPipe再推論はない。追加教師未完了、range差分、source差分では停止。heavy fittingはHF CPUだけ。既存precision/support gateは変更なし。

Job `6accd5f3095c578089330c65` COMPLETED、run `quality-refit-20261012T124328Z`、private repo `YUGOROU/h3-quality-refit-20261012T124328Z`、input revision `008dee5bdcd53fe3bc52be4faa8d06712790a16b`。cpu-upgrade$0.03/hr、hard20m、elapsed25.234s、追加概算$0.000210、全累計概算$29.821808（請求額ではない）。no retry/GPU。

結果は**auto_proposal0 / review335**。global eligible identity58中56正解でも、昨日2sessionの同class eligible supportは0。motion/reject global eligible0。335中333はvote不足、123はfeature support外、45は検出率不足（重複理由）。教師追加をもって自動化成功とは言えない。閾値を緩めない、保留を除外へ変えない。漫然と教師追加/CPU反復しない。

投入時snapshotには閲覧中の初期327件の人手判断155件もあったが、これらのcached featureはないのでモデルfitには含めていない。この155件を未承認に戻したわけではない。元のgray feature対象813件と初期327承認は別々に管理する。解析結果は人手labelsを書き換えず、USERは承認画面を継続利用できる。

初期327承認が終わった時点で最新labelsから採用manifestを再集計する。現在のselected_manifest345件は以前のsnapshotであり、新しい手動判断の全数を示さない。実行中serverの再起動は閲覧中に行わない。新実装は/dataアクセス時にresultをreload可能で、次回server起動後に有効。

88 tests + DOM key simulation。features再利用stage・不足教師・range違い・proposals更新とhuman label保持を追加確認。

## 追加教師40件完了・temporal-v2特徴量

追加40はidentity15/motion12/reject13。全845判断、採用715（590identity/125motion、末尾重複含め61:34.167）、除外130、残り295。採用manifest更新済み。

`hf_quality_v2_worker.py` / `quality_temporal.py` を追加。legacy18に、可視性の連続/途切れ、sharpness分位/目口median/顔とのratio、瞬き/口開き/pose変動、**隣接valid frameだけ**のcenter/area/eye動作21を追加し39特徴量へ。欠損frameを飛ばして架空の隣接motionにしない。320px grayプレビューの解像度/色/全clip確認限界は残る。identity embedding/raw landmark保存はない。v2では既に手動判断済みの初期327もfeature対象に含める。H3/TrainerやGPUは使用しない。

Job `6acce681134c60eb026065e3` COMPLETED、run `quality-20261012T135351Z`、private repo `YUGOROU/h3-quality-quality-20261012T135351Z`、input revision `b5baadea1f22433bd9c4028d632d204b3e5fbea1`。入力57,164,817bytes、全1140候補の既存uniform gray previews＋判断snapshot/コードのみ。音声/原本/Reference写真uploadなし。cpu-upgrade$0.03/hr、1h timeout/$0.05 guard、no retry。148.852s、追加概算$0.001240、累計概算$29.823048（請求額ではない）。有限foreground wait1回/取得1回、watcherなし。

全1140×39特徴量、処理error0。旧813×18のprefixは以前の特徴量と最大絶対差0で一致。精度gateは維持し、結果**auto0/review295**。global identity eligible193/191correctでも、昨日2session eligible0。残り291低vote、142feature support外、36検出不足（重複）。feature抽出完了は自動選定成功ではない。人手845判断は変更なし、全295追加判断/さらなる同じfitを要求せず、既存715でTrainer実装へ進む。

このunitのroot suite114tests（10torch関連skip）、別環境の実patched guidance12/12、component loader8/8。Model/PEFT forward/GPU training/real-media latentは未実行。詳細 [guidance](h3-identity-guidance.md) / [component cache](h3-component-cache.md)。

## 操作・取得

```sh
# 一回だけ完了resultを取得。RUNNINGなら停止、watch/retryしない。
/tmp/h3-media-env/bin/python research/fetch_hf_quality.py \
  --selection-dir $H3_PRIVATE_ROOT.parent/identity-selection
# 最新人手判断から暫定採用rangeを再作成（heavy処理なし）
python3 -m research.h3_personal.image_selection \
  --selection-dir $H3_PRIVATE_ROOT.parent/identity-selection
```

`quality_proposals.json`には人手判断とは別の結果を保存。viewerはserver再起動＋ページ再読み込みで「自動選別：保留」「自動提案：確認」を表示できる。前者は任意の追加確認であり、全件処理必須ではない。

81 stdlib/FFmpeg tests + JS構文チェック。実際のHF CPU inference・813feature extraction・OOF fittingはこのrunで完了。ただし品質選別の自動化成功とは区別する。
