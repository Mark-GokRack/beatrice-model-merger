# Beatrice Model Merger

## Abstract / 概要

This project trains one multi-speaker student model from source audio and one or more Beatrice 2.0.0-rc.0 teacher models.  
Teacher outputs are generated on demand from the teacher `.bin` files, so no pre-rendered target audio is required.

このプロジェクトは、ソース音声と複数の [Beatrice](https://prj-beatrice.com/) 2.0.0-rc.0 版の 教師モデルから、複数話者に対応する 1 つの生徒モデルを学習します。  
教師の出力は教師 `.bin` ファイルから必要に応じて生成されるため、事前にレンダリングしたターゲット音声は不要です。

This project is experimental, and does not guarantee the quality of the generated student models.

このプロジェクトは実験的なものであり、生成される生徒モデルの品質を保証するものではありません。

## Dataset layout / データの配置方法

Pass a directory of sound source data for training with `-d`; every `.wav` file below it is found recursively and used. Pass a teacher-model directory with `-w`; every subdirectory containing the five Beatrice inference `.bin` files and one TOML file is found recursively. Each voice in each discovered teacher model becomes one voice in the distilled model, in path order followed by original voice-number order.

学習に使用する音源データのディレクトリは `-d` で指定します。配下にあるすべての `.wav` ファイルが再帰的に検出され、使用されます。教師モデルのディレクトリは `-w` で指定します。Beatrice 推論用の 5 つの `.bin` ファイルと 1 つの TOML ファイルを含むすべてのサブディレクトリが再帰的に検出されます。検出された各教師モデルの各ボイスは、パス順、次いで元のボイス番号順で、蒸留モデル内の 1 ボイスになります。

During development of this project, we used [LibriTTS-R](https://www.openslr.org/141/) as the source audio dataset for training and verified that the project runs with it.

このプロジェクトの開発にあたり、学習時の音源ファイルとして [LibriTTS-R](https://www.openslr.org/141/) を使用し、動作確認を行っております。

```text
data/
  corpus_a/
    set_01/phrase_001.wav
  corpus_b/phrase_002.wav
weights/
  teacher_a/
    phone_extractor.bin
    pitch_estimator.bin
    waveform_generator.bin
    embedding_setter.bin
    speaker_embeddings.bin
    teacher_a.toml
  teacher_b/
    ...
```

Each source waveform is converted by every teacher voice with a randomly selected formant shift and four VQ neighbours. The selected value is passed to both the teacher and the student's formant embedding, so the generated output remains an aligned training target.

各ソース波形はランダムに選ばれたフォルマントシフトと VQ 近傍数 4 を使用して、すべての教師ボイスで変換されます。選ばれた値は教師と生徒の両方のフォルマント埋め込みに渡されるため、生成された出力は整合した学習ターゲットとして維持されます。

## PyTorch Teacher Inference / PyTorch による教師推論

Teacher inference uses `beatrice_distiller.PyTorchConverter`; it has no dependency on `libbeatrice`, pybind11, a C++ compiler, or `beatrice-vst`. `phone_extractor` and `pitch_estimator` are loaded from the configured `beatrice-trainer/assets/pretrained/*.pt` checkpoints. The remaining model parameters are reconstructed from each teacher's inference `.bin` files.

教師推論には `beatrice_distiller.PyTorchConverter` を使用します。`phone_extractor` と `pitch_estimator` は設定された `beatrice-trainer/assets/pretrained/*.pt` から読み込み、残りのパラメータは各教師の推論用 `.bin` ファイルから復元します。

`PyTorchConverter.process()` はオフライン推論です。入力の末尾は 160 sample 単位にゼロ埋めされ、入力フレームごとに 240 sample を返します。native runtime との出力を比較するには、native extension を別途保存したうえで次を実行します。

```powershell
python -m beatrice_distiller.compare_inference weights/teacher_a source.wav `
  --native-extension-dir path/to/legacy/extension --speaker 0 --output-wav pytorch.wav
```

このスクリプトは最適な $\pm480$ sample の遅延を選び、波形 MAE/RMSE、signal-to-error ratio、log-STFT magnitude MAE を表示します。vocoder のノイズおよび初期位相は確率的なので、波形完全一致ではなく、これらの分布・スペクトル指標で評価してください。

`process()` はネイティブのストリーミングコンテキストを保持します。最後の 160 サンプル未満の
入力ホップはゼロパディングされ、処理された各ホップに対して 240 サンプルを返します。したがって、
出力には推論エンジンの起動時および末尾の遅延が含まれます。distiller はソース波形ごとに
`process()` を一度呼び出し、このパディングを維持したまま、学習前にソース/ターゲットのペアを
共通のフレーム数までトリミングします。

## Run / 実行

From the repository root, activate the environment and start training:

リポジトリのルートから環境を有効化し、学習を開始します。

```powershell
python -m beatrice_distiller -d data -w weights -o distilled_output
```

Use `-c config.json` to override keys in the distiller's `DEFAULT_CONFIG`. The pretrained generator checkpoint and the two frozen feature extractors must remain available at the configured paths.

`-c config.json` を使用すると、distiller の `DEFAULT_CONFIG` に含まれるキーを上書きできます。事前学習済みジェネレーターのチェックポイントと 2 つの固定済み特徴抽出器は、設定されたパスで利用可能な状態にしておく必要があります。

By default, `generator_init_mode` is `"pretrained"`, which keeps the existing initialization from `pretrained_file`. Set it to `"teacher"` and set `generator_init_model` to a teacher-model directory relative to `-w` (or to an absolute path) to initialize the waveform generator and embedding setter from that model. This mode also imports the selected model's formant embedding and, for every matching teacher voice, its non-codebook speaker embeddings into the corresponding distilled speaker ID. VQ codebooks continue to be initialized from every discovered teacher voice. Set it to `"average"` to initialize the waveform generator and embedding setter with the element-wise mean of every discovered teacher model. The teacher binaries must use the supported Beatrice model version and identical network layouts.

既定の `generator_init_mode` は `"pretrained"` で、従来どおり `pretrained_file` から初期化します。`"teacher"` を指定した場合は、`generator_init_model` に `-w` からの相対パス（または絶対パス）で教師モデルのディレクトリを指定すると、そのモデルの waveform generator と embedding setter を初期値として使用します。このモードでは、指定モデルの formant embedding と、そのモデルに属する各教師話者の codebook 以外の speaker embedding も、蒸留モデル内の対応する話者 ID へ初期値として読み込みます。VQ codebook は従来どおり、検出されたすべての教師話者から初期化されます。`"average"` を指定した場合は、検出されたすべての教師モデルの waveform generator と embedding setter の対応する値の要素ごとの平均を初期値として使用します。教師バイナリはサポート対象の Beatrice モデルバージョンであり、ネットワーク構造が同一である必要があります。

```json
{
  "generator_init_mode": "teacher",
  "generator_init_model": "paraphernalia_000-tatoeba-yomi_00010000"
}
```

```json
{
  "generator_init_mode": "average"
}
```

At each `save_interval`, the distiller writes a training checkpoint and a `paraphernalia_distilled_<step>` directory. The latter contains the inference `.bin` files and a TOML file. The output TOML combines the original teacher voice sections in the same discovery order as the distilled speaker embeddings, including voice descriptions, average pitches, and portraits where their files are available.

各 `save_interval` で、distiller は学習チェックポイントと `paraphernalia_distilled_<step>` ディレクトリを書き出します。後者には推論用の `.bin` ファイルと TOML ファイルが含まれます。出力 TOML には、蒸留された話者埋め込みと同じ検出順で元の教師ボイスのセクションが統合され、利用可能な場合はボイスの説明、平均ピッチ、ポートレートも含まれます。

## Project License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  

このリポジトリは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。

## Third-party Libraries and Licenses

This project uses the following third-party libraries as Git submodules  
このプロジェクトでは、以下のサードパーティライブラリを Git サブモジュールとして使用しています

- [Beatrice-trainer](https://huggingface.co/fierce-cats/beatrice-trainer) : [MIT License](https://huggingface.co/fierce-cats/beatrice-trainer/blob/main/LICENSE)

## Notes / 注釈

- Training feeds each source waveform to the student and uses the teacher's generated waveform as its target, so it learns the actual conversion task rather than self-reconstruction from pseudo voices.
- 学習では各ソース波形を生徒に入力し、教師が生成した波形をターゲットとして使用します。そのため、疑似ボイスからの自己再構成ではなく、実際の変換タスクを学習します。
- VQ codebooks are loaded from each teacher voice's `speaker_embeddings.bin` by default, so codebook K-means initialization is skipped.
- 既定では VQ コードブックが各教師ボイスの `speaker_embeddings.bin` から読み込まれるため、コードブックの K-means 初期化はスキップされます。
- Teacher TOML model versions must match the distiller's Beatrice version (`2.0.0-rc.0`). Set `vq_init_from_bin` to `false` in a config file to use the legacy dynamically generated-output initialization instead.
- 教師 TOML のモデルバージョンは、distiller の Beatrice バージョン（`2.0.0-rc.0`）と一致している必要があります。従来の動的生成出力による初期化を使用する場合は、設定ファイルで `vq_init_from_bin` を `false` に設定してください。
- `formant_shift_candidates` selects the discrete formant conditions used for training. Its default values are the nine formant-embedding positions from $-2.0$ to $+2.0$ semitones in $0.5$-semitone steps. Set it to `[0.0]` to disable formant variation.
- `formant_shift_candidates` は、学習に使用する離散的なフォルマント条件を選択します。既定値は、$-2.0$ から $+2.0$ 半音までを $0.5$ 半音刻みで表す 9 つのフォルマント埋め込み位置です。フォルマント変動を無効化するには `[0.0]` に設定してください。
- The imported `ConverterNetwork` performs the trainer's pitch-shift augmentation automatically while it is in training mode. The teacher target remains at zero explicit pitch shift, matching the regular trainer's pitch-robustness training.
- インポートされた `ConverterNetwork` は、学習モード中にトレーナーのピッチシフト拡張を自動で実行します。教師ターゲットは明示的なピッチシフト 0 のままで、通常のトレーナーにおけるピッチロバスト性学習と一致します。
- Confirm that you have permission to generate derivatives of every teacher model and every source recording used in the dataset.
- 各教師モデルとデータセットで使用する各ソース録音について、派生物を生成する許可があることを確認してください。
