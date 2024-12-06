# autogguf

Automatically convert HuggingFace models to GGUF

rust port of the [python command line utility](https://github.com/brittlewis12/autogguf), originally inspired by [mlabonne](https://twitter.com/maximelabonne/status/1746812715606348138)’s [AutoGGUF.ipynb](https://colab.research.google.com/drive/1P646NEg33BZy4BfLDNpTz0V0lwIU3CHu)

## Pre-requisites

`autogguf` expects a couple programs to already be installed:

- `huggingface-cli`: to download and upload models
  * install:
    - with [pip](https://pip.pypa.io): `pip3 install -U huggingface_hub[cli]`
    - with [Homebrew](https://brew.sh): `brew install huggingface-cli`
    - with [Pkgx](https://pkgx.sh): `pkgx install huggingface-cli`
- `llama.cpp`: to convert & quantize models
  * use `autogguf -u [...]` to install and update if it doesn’t exist in the default path (`~/code/llama.cpp`):
  * use `autogguf -ul <LLAMA_PATH> [...]` to automatically update and recompile with an alternative `llama.cpp` path
  * use `autogguf -l <LLAMA_PATH> [...]` to use an alternative `llama.cpp` path, without updating or recompiling.

## Installation

Build from source with `cargo`:

```sh
cargo install --git https://github.com/brittlewis12/autogguf-rs
```

## Usage

### help

```sh
$ autogguf -h
Convert HuggingFace models to GGUF automatically.

Usage: autogguf [OPTIONS] <MODEL_ID>

Arguments:
  <MODEL_ID>  The HuggingFace model ID to convert. Required

Options:
  -q, --quants <QUANTS>...
          Comma-separated list of quant levels to convert. Defaults to all
          non-imatrix quants [default:
          q2_k,q3_k_s,q3_k_m,q3_k_l,q4_0,q4_1,q4_k_s,q4_k_m,q5_0,q5_1,q5_k_s,q5_k_m,q6_k,q8_0]
  -v, --verbose
          Increase output verbosity
      --full-precision <FULL_PRECISION>
          The full-precision GGUF format to convert to and quantize from
          [default: f16] [possible values: f16, bf16, f32]
      --fp <FP>
          Path to fp16, bf16 or fp32 GGUF file for quantization. Implies
          skipping download and initial conversion to full precision GGUF
      --imatrix <IMATRIX>
          Path to custom imatrix file for imatrix quantization. Skips
          downloading calibration dataset and generating imatrix
      --skip-download
          Skip downloading the model to convert from HuggingFace Hub
      --skip-upload
          Skip uploading converted files to HuggingFace Hub
      --only-upload
          Upload .gguf files in the target model directory to HuggingFace Hub
  -u, --update-llama
          Update the llama.cpp repo before converting. Installs llama.cpp if
          llama-path doesn’t exist
  -l, --llama-path <LLAMA_PATH>
          The path to the llama.cpp repo [default: ~/code/llama.cpp]
      --hf-token <HF_TOKEN>
          Your HuggingFace API token for uploading converted models [env:
          HF_TOKEN]
      --hf-user <HF_USER>
          Your HuggingFace username for uploading converted models [env:
          HF_USER=...]
  -h, --help
          Print help
  -V, --version
          Print version
```

### download, convert, quantize, and upload a model

```sh
$ autogguf meta-llama/meta-llama-3.1-8B-Instruct -uv -q q4_k_m
```
- `MODEL_ID`: `meta-llama/meta-llama-3.1-8B-Instruct`
- with the `-u`/`--update-llama` flag, `autogguf` will update & recompile the `llama.cpp` repo before converting.
  * if the `llama.cpp` repo doesn’t exist, it will be installed.
- will convert the model to `fp16` by default
  * you can instead provide a desired full-precision (`bf16`, `f32`) with the `--full-precision` flag.
  * you can also provide the path to an existing converted model file with the `--fp` flag.
- will read `$HF_USER` & `$HF_TOKEN` from environment to upload to HuggingFace Hub.
  * you can instead provide them with `--hf-user` and `--hf-token` flags.

### download, convert, generate imatrix, quantize, and upload a model

```sh
$ autogguf meta-llama/meta-llama-3.1-8B-Instruct -uv -q iq4_nl
```

- when autogguf detects quant levels requiring an imatrix, it automatically generates one using a battle-tested, general-purpose calibration dataset [discovered and tested by kalomaze](https://github.com/ggerganov/llama.cpp/discussions/5263#discussioncomment-8395384).
  * you can instead provide a custom imatrix file with the `--imatrix` flag.

### download, convert, & quantize a model, without uploading

```sh
$ autogguf meta-llama/meta-llama-3.1-8B-Instruct -uv -q q4_k_m --skip-upload
```

## License

This project is licensed under your choice of the following licenses:
* [Unlicense](https://unlicense.org/) - see [UNLICENSE](./UNLICENSE) for details.
* [MIT license](https://opensource.org/license/MIT) - see [LICENSE-MIT](./LICENSE-MIT) for details.
* [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) - see [LICENSE-Apache-2.0](./LICENSE-Apache-2.0) for details.
