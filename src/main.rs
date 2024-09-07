use clap::{Parser, ValueEnum};
use futures_util::StreamExt;
use shellexpand::tilde;
use std::{
    fmt::Display,
    path::PathBuf,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{
    fs::File,
    io::AsyncWriteExt,
    process::Command,
    select, signal,
    sync::{mpsc, Notify},
    task::JoinHandle,
    time::sleep,
};

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// The HuggingFace model ID to convert. Required.
    model_id: String,

    /// Comma-separated list of quant levels to convert. Defaults to all non-imatrix quants.
    #[clap(
        short,
        long,
        value_delimiter = ',',
        num_args = 1..,
        default_value = "q2_k,q3_k_s,q3_k_m,q3_k_l,q4_0,q4_1,q4_k_s,q4_k_m,q5_0,q5_1,q5_k_s,q5_k_m,q6_k,q8_0"
    )]
    quants: Vec<QuantLevel>,

    #[clap(short, long)]
    /// Increase output verbosity.
    verbose: bool,

    #[clap(long, default_value = "f16")]
    /// The full-precision GGUF format to convert to and quantize from.
    full_precision: Precision,

    #[clap(long)]
    /// Path to fp16, bf16 or fp32 GGUF file for quantization. Implies skipping download and initial conversion to full precision GGUF.
    fp: Option<String>,

    #[clap(long)]
    /// Path to custom imatrix file for imatrix quantization. Skips downloading calibration dataset and generating imatrix.
    imatrix: Option<String>,

    #[clap(long)]
    /// Skip downloading the model to convert from HuggingFace Hub.
    skip_download: bool,

    #[clap(long)]
    /// Skip uploading converted files to HuggingFace Hub.
    skip_upload: bool,

    #[clap(long)]
    /// Upload .gguf files in the target model directory to HuggingFace Hub.
    only_upload: bool,

    #[clap(short, long)]
    /// Update the llama.cpp repo before converting. Installs llama.cpp if llama-path doesn’t exist.
    update_llama: bool,

    #[clap(short, long, default_value = "~/code/llama.cpp")]
    /// The path to the llama.cpp repo.
    llama_path: String,

    #[clap(long)]
    /// Your HuggingFace API token for uploading converted models. Reads from `$HF_TOKEN` by default.
    hf_token: Option<String>,

    #[clap(long)]
    /// Your HuggingFace username for uploading converted models. Reads from `$HF_USER` by default.
    hf_user: Option<String>,
}

#[derive(Debug, Clone, ValueEnum)]
enum Precision {
    F16,
    BF16,
    F32,
}

impl Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Precision::F16 => "f16",
            Precision::BF16 => "bf16",
            Precision::F32 => "f32",
        };
        write!(f, "{label}")
    }
}

macro_rules! quant_level_enum {
    ($($variant:ident => $str:expr),* $(,)?) => {
        #[derive(Debug, Clone)]
        enum QuantLevel {
            $($variant),*
        }

        impl FromStr for QuantLevel {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s.to_lowercase().as_str() {
                    $($str => Ok(QuantLevel::$variant),)*
                    _ => Err(format!("'{}' is not a valid quant level", s)),
                }
            }
        }

        impl Display for QuantLevel {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let label = match self {
                    $(QuantLevel::$variant => $str,)*
                };
                write!(f, "{}", label)
            }
        }
    };
}

quant_level_enum! {
    Q2K => "q2_k",
    Q3KS => "q3_k_s",
    Q3KM => "q3_k_m",
    Q3KL => "q3_k_l",
    Q4_0 => "q4_0",
    Q4_1 => "q4_1",
    Q4KS => "q4_k_s",
    Q4KM => "q4_k_m",
    Q5_0 => "q5_0",
    Q5_1 => "q5_1",
    Q5KS => "q5_k_s",
    Q5KM => "q5_k_m",
    Q6K => "q6_k",
    Q8_0 => "q8_0",
    BF16 => "bf16",
    IQ1S => "iq1_s",
    IQ1M => "iq1_m",
    IQ2XXS => "iq2_xxs",
    IQ2XS => "iq2_xs",
    IQ2S => "iq2_s",
    IQ2M => "iq2_m",
    Q2KS => "q2_k_s",
    IQ3XXS => "iq3_xxs",
    IQ3XS => "iq3_xs",
    IQ3S => "iq3_s",
    IQ3M => "iq3_m",
    IQ4XS => "iq4_xs",
    IQ4NL => "iq4_nl",
}

impl QuantLevel {
    fn requires_imatrix(&self) -> bool {
        matches!(
            self,
            QuantLevel::IQ1S
                | QuantLevel::IQ1M
                | QuantLevel::IQ2XXS
                | QuantLevel::IQ2XS
                | QuantLevel::IQ2S
                | QuantLevel::IQ2M
                | QuantLevel::Q2KS
                | QuantLevel::IQ3XXS
                | QuantLevel::IQ3XS
                | QuantLevel::IQ3S
                | QuantLevel::IQ3M
                | QuantLevel::IQ4XS
                | QuantLevel::IQ4NL
        )
    }
}

async fn update_llama_cpp(
    llama_path: PathBuf,
    verbose: bool,
    cancel_rx: Arc<Notify>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !llama_path.exists() {
        if verbose {
            println!(
                "🐪 llama.cpp not found at {}, installing...",
                llama_path.display()
            );
        }
        let mut clone = Command::new("git")
            .arg("clone")
            .arg("https://github.com/ggerganov/llama.cpp")
            .arg(llama_path.clone())
            .spawn()?;
        select! {
            status = clone.wait() => {
                status?;
            }
            _ = cancel_rx.notified() => {
                clone.kill().await?;
                return Err("Llama.cpp installation process cancelled".into());
            }
        }
    }

    if verbose {
        println!("🐪 compiling llama.cpp...");
    }
    let mut pull = Command::new("git")
        .arg("pull")
        .current_dir(&llama_path)
        .spawn()?;
    select! {
        status = pull.wait() => {
            status?;
        }
        _ = cancel_rx.notified() => {
            pull.kill().await?;
            return Err("Llama.cpp update process cancelled".into());
        }
    }

    let mut clean = Command::new("make")
        .arg("clean")
        .current_dir(&llama_path)
        .spawn()?;
    select! {
        status = clean.wait() => {
            status?;
        }
        _ = cancel_rx.notified() => {
            clean.kill().await?;
            return Err("Llama.cpp build clean process cancelled".into());
        }
    }

    let mut make = Command::new("make").current_dir(&llama_path).spawn()?;
    select! {
        status = make.wait() => {
            status?;
        }
        _ = cancel_rx.notified() => {
            make.kill().await?;
            return Err("Llama.cpp build process cancelled".into());
        }
    }

    if verbose {
        println!("🐪 installing llama.cpp python deps...");
    }
    let mut deps = Command::new("pip3")
        .arg("install")
        .arg("-r")
        .arg("requirements.txt")
        .arg(if verbose { "-v" } else { "-q" })
        .current_dir(&llama_path)
        .spawn()?;

    select! {
        status = deps.wait() => {
            status?;
        }
        _ = cancel_rx.notified() => {
            deps.kill().await?;
            return Err("Llama.cpp build process cancelled".into());
        }
    }

    Ok(())
}

async fn download_model(
    model_id: &str,
    model_name: &str,
    verbose: bool,
    cancel_rx: Arc<Notify>,
) -> Result<(), Box<dyn std::error::Error>> {
    Command::new("mkdir")
        .arg("-p")
        .arg(model_name)
        .spawn()?
        .wait()
        .await?;
    if verbose {
        println!("🤗 downloading {model_name}...");
    }
    let mut args = vec![
        "download".to_string(),
        model_id.to_string(),
        "--local-dir".to_string(),
        format!("./{model_name}"),
    ];
    if !verbose {
        args.push("--quiet".to_string());
    }
    let mut download_task = Command::new("huggingface-cli").args(args).spawn()?;
    select! {
        status = download_task.wait() => {
            status?;
            if verbose {
                println!("🤗 downloaded {model_name}!");
            }
            Ok(())
        }
        _ = cancel_rx.notified() => {
            download_task.kill().await?;
            Err("Download process killed due to interrupt".into())
        }
    }
}

async fn convert_fp(
    precision: Precision,
    llama_path: PathBuf,
    output_path: PathBuf,
    model_name: &str,
    verbose: bool,
    cancel_rx: Arc<Notify>,
) -> Result<(), Box<dyn std::error::Error>> {
    if verbose {
        println!(
            "🪄 converting {model_name} to {}...",
            precision.to_string().to_uppercase()
        );
    }
    let mut convert_fp_task = Command::new("python3")
        .arg(llama_path.join("convert_hf_to_gguf.py"))
        .arg(model_name)
        .arg("--outtype")
        .arg(precision.to_string())
        .arg("--outfile")
        .arg(&output_path)
        .spawn()?;
    select! {
        status = convert_fp_task.wait() => {
            status?;
        }
        _ = cancel_rx.notified() => {
            convert_fp_task.kill().await?;
            return Err("Conversion process killed due to interrupt".into());
        }
    }

    if !tokio::fs::try_exists(output_path).await? {
        return Err("💥 Conversion failed".into());
    };

    if verbose {
        // teeeeeeeechnically this is new and missing from the og autogguf[.py].....
        println!(
            "🪄 {model_name} conversion to {} complete!",
            precision.to_string().to_uppercase()
        );
    }

    Ok(())
}

async fn generate_imatrix(
    llama_path: PathBuf,
    fp: PathBuf,
    output_path: PathBuf,
    model_name: &str,
    verbose: bool,
    cancel_rx: Arc<Notify>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !tokio::fs::try_exists("calibration_data.txt").await? {
        if verbose {
            println!("🌐 downloading calibration dataset...");
        }
        let mut byte_stream =
            reqwest::get("https://github.com/ggerganov/llama.cpp/files/14194570/groups_merged.txt")
                .await?
                .bytes_stream();
        let mut f = File::create("calibration_data.txt").await?;
        while let Some(bytes) = byte_stream.next().await {
            f.write_all(&bytes?).await?;
        }
        f.flush().await?;
    };
    if verbose {
        println!("⚖️ generating imatrix for {model_name}...");
    }
    let mut imatrix_task = Command::new(llama_path.join("llama-imatrix"))
        .arg("-m")
        .arg(fp)
        .arg("-f")
        .arg("calibration_data.txt")
        .arg("-o")
        .arg(output_path)
        .arg("-t")
        .arg("7")
        .arg("-ngl")
        .arg("999")
        .arg("--chunks")
        .arg("2000")
        .spawn()?;
    select! {
        status = imatrix_task.wait() => {
            status?;
        }
        _ = cancel_rx.notified() => {
            imatrix_task.kill().await?;
            return Err("imatrix generation process killed due to interrupt".into());
        }
    }
    if verbose {
        println!("🧹 cleaning up caliration dataset...");
    }
    tokio::fs::remove_file("calibration_data.txt").await?;
    Ok(())
}

async fn quantize(
    q: QuantLevel,
    llama_path: PathBuf,
    fp: PathBuf,
    imatrix: PathBuf,
    model_name: &str,
    verbose: bool,
    cancel_rx: Arc<Notify>,
) -> Result<(), Box<dyn std::error::Error>> {
    if verbose {
        println!(
            "🪄 quantizing {model_name} to {}...",
            q.to_string().to_uppercase()
        );
    }
    let quant_path = format!(
        "{model_name}/{}.{}.gguf",
        model_name.to_lowercase(),
        q.to_string().to_uppercase()
    );
    let default_args = vec![
        fp.to_string_lossy().to_string(),
        format!("{quant_path}.pending"),
        q.to_string(),
    ];
    let mut args = vec![];
    if q.requires_imatrix() {
        args.push("--imatrix".to_string());
        args.push(imatrix.to_string_lossy().to_string());
    }
    args.extend_from_slice(default_args.as_slice());
    let mut quantize = Command::new(llama_path.join("llama-quantize"))
        .args(args)
        .spawn()?;

    select! {
        status = quantize.wait() => {
            status?;
        }
        _ = cancel_rx.notified() => {
            quantize.kill().await?;
            return Err("Quantization process killed due to interrupt".into());
        }
    }

    let mut moov = Command::new("mv")
        .arg(format!("{quant_path}.pending"))
        .arg(quant_path)
        .spawn()?;

    select! {
        status = moov.wait() => {
            status?;
        }
        _ = cancel_rx.notified() => {
            moov.kill().await?;
            return Err("Quantized file rename process killed due to interrupt".into());
        }
    }

    Ok(())
}

async fn upload_ggufs_to_hf(
    hf_user: String,
    hf_token: String,
    verbose: bool,
    model_name: &str,
    cancel_rx: Arc<Notify>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if verbose {
        println!("🤗 uploading {model_name} to HuggingFace Hub...");
    }

    let repo_name = format!("{model_name}-GGUF");
    let repo_id = format!("{hf_user}/{repo_name}");
    let mut upload = Command::new("huggingface-cli")
        .env("HF_USER", hf_user)
        .env("HF_TOKEN", hf_token)
        .arg("upload")
        .arg(repo_id)
        .arg(model_name) // local path
        .arg(".") // remote path
        .arg("--include")
        .arg("*.gguf")
        .arg("*.imatrix")
        .spawn()?;

    select! {
        status = upload.wait() => {
            status?;
            if verbose {
                println!("🤗 uploaded {model_name} to HuggingFace Hub!");
            }
        }
        _ = cancel_rx.notified() => {
            upload.kill().await?;
            return Err("Upload process killed due to interrupt".into());
        }
    }

    Ok(())
}

async fn upload_worker(
    mut receiver: mpsc::Receiver<()>,
    busy: Arc<AtomicBool>,
    hf_user: String,
    hf_token: String,
    verbose: bool,
    model_name: String,
    cancel_rx: Arc<Notify>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    while receiver.recv().await.is_some() {
        if !busy.swap(true, Ordering::Acquire) {
            upload_ggufs_to_hf(
                hf_user.clone(),
                hf_token.clone(),
                verbose,
                &model_name,
                cancel_rx.clone(),
            )
            .await?;

            busy.store(false, Ordering::Release);
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.verbose {
        println!("Got args: {args:?}");
    }

    let notify = Arc::new(Notify::new());
    let notifier = notify.clone();
    tokio::spawn(async move {
        signal::ctrl_c()
            .await
            .expect("failed to register ctrl-c handler");
        notifier.notify_waiters(); // Signal cancellation
    });

    let llama_path = PathBuf::from(tilde(&args.llama_path).into_owned());
    if args.update_llama {
        update_llama_cpp(llama_path.clone(), args.verbose, notify.clone()).await?;
    }

    let model_name = args
        .model_id
        .split('/')
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .get(1)
        .cloned()
        .unwrap_or_default();

    let override_fp = args.fp.is_some();
    if args.skip_download || override_fp || args.only_upload {
        // if args.verbose { // ?
        println!("🤗 skipping download from HuggingFace Hub.");
        // }
    } else {
        download_model(&args.model_id, &model_name, args.verbose, notify.clone()).await?;
    }

    let precision = args.full_precision;
    let fp = if let Some(fp) = args.fp {
        PathBuf::from(tilde(&fp).into_owned())
    } else {
        PathBuf::from(format!(
            "{model_name}/{}.{precision}.gguf",
            model_name.to_lowercase()
        ))
    };
    if override_fp || args.only_upload {
        // if args.verbose { // ?
        println!(
            "skipping {} conversion.",
            precision.to_string().to_uppercase()
        );
        // }
    } else {
        convert_fp(
            precision,
            llama_path.clone(),
            fp.clone(),
            &model_name,
            args.verbose,
            notify.clone(),
        )
        .await?;
    }

    let override_imat = args.imatrix.is_some();
    let imatrix_path = if let Some(imat) = args.imatrix.clone() {
        PathBuf::from(tilde(&imat).into_owned())
    } else {
        PathBuf::from(format!(
            "{model_name}/{}.imatrix",
            model_name.to_lowercase()
        ))
    };
    if !args.only_upload && !override_imat && args.quants.iter().any(QuantLevel::requires_imatrix) {
        generate_imatrix(
            llama_path.clone(),
            fp.clone(),
            imatrix_path.clone(),
            &model_name,
            args.verbose,
            notify.clone(),
        )
        .await?;
    }

    let hf_user = args
        .hf_user
        .clone()
        .unwrap_or_else(|| std::env::var("HF_USER").unwrap_or_default());
    let hf_token = args
        .hf_token
        .clone()
        .unwrap_or_else(|| std::env::var("HF_TOKEN").unwrap_or_default());

    let (upload_tx, upload_rx) = mpsc::channel(10);
    let busy = Arc::new(AtomicBool::new(false));
    let busy_clone = busy.clone();
    let mut upload_handle: Option<JoinHandle<_>> = None;
    if !args.skip_upload {
        upload_handle = Some(tokio::task::spawn(upload_worker(
            upload_rx,
            busy_clone,
            hf_user.clone(),
            hf_token.clone(),
            args.verbose,
            model_name.clone(),
            notify.clone(),
        )));
    }

    if !args.only_upload {
        for q in args.quants {
            quantize(
                q,
                llama_path.clone(),
                fp.clone(),
                imatrix_path.clone(),
                &model_name,
                args.verbose,
                notify.clone(),
            )
            .await?;

            if !args.skip_upload && !busy.load(Ordering::Acquire) {
                upload_tx.send(()).await?;
            }
        }
    }

    if !args.skip_upload {
        while busy.load(Ordering::Acquire) {
            sleep(Duration::from_millis(100)).await;
        }
        upload_tx.send(()).await?;
    }
    drop(upload_tx);
    if let Some(handle) = upload_handle {
        match handle.await? {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error in upload worker: {e:?}");
            }
        }
    }

    println!("🎉 done!");

    Ok(())
}
