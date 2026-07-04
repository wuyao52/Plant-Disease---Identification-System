from __future__ import annotations

import json
import re
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.config import load_config
from plant_disease.data import is_saved_split_dataset
from plant_disease.dataset_validation import validate_dataset
from plant_disease.inference import DiseasePredictor
from plant_disease.pipeline import build_overrides, train_from_config
from plant_disease.recommendations import get_recommendations
from scripts.check_split_leakage import check_leakage
from scripts.prepare_split_dataset import copy_split_dataset


TEXT = {
    "page_title": "\u690d\u7269\u75c5\u5bb3\u8bc6\u522b\u7cfb\u7edf",
    "settings": "\u8fd0\u884c\u8bbe\u7f6e",
    "checkpoint": "\u6a21\u578b\u6743\u91cd",
    "report_dir": "\u8bad\u7ec3\u62a5\u544a\u76ee\u5f55",
    "show_heatmap": "\u663e\u793a\u70ed\u529b\u56fe",
    "demo_notice": "\u5f53\u524d\u9ed8\u8ba4\u52a0\u8f7d\u5408\u6210\u6f14\u793a\u6743\u91cd\uff0c\u4ec5\u7528\u4e8e\u6d41\u7a0b\u9a8c\u8bc1\u3002",
    "caption": "\u53ef\u586b\u5199\u81ea\u5b9a\u4e49\u8bad\u7ec3\u751f\u6210\u7684\u6a21\u578b\u6743\u91cd\u548c\u62a5\u544a\u76ee\u5f55\u3002",
    "training_report": "\u8bad\u7ec3\u62a5\u544a",
    "test_acc": "\u6d4b\u8bd5\u51c6\u786e\u7387",
    "best_val_acc": "\u6700\u4f73\u9a8c\u8bc1\u51c6\u786e\u7387",
    "train_images": "\u8bad\u7ec3\u56fe\u7247\u6570",
    "none": "\u6682\u65e0",
    "download_confusion": "\u4e0b\u8f7d\u6df7\u6dc6\u77e9\u9635 CSV",
    "upload": "\u4e0a\u4f20\u53f6\u7247\u56fe\u7247\uff08\u53ef\u591a\u9009\uff09",
    "select_images": "\u8bf7\u9009\u62e9\u4e00\u5f20\u6216\u591a\u5f20\u53f6\u7247\u56fe\u7247\u3002",
    "preview": "\u9884\u89c8\u56fe\u7247",
    "missing_model": "\u672a\u627e\u5230\u6a21\u578b\u6743\u91cd\uff1a",
    "predicting": "\u6b63\u5728\u8bc6\u522b...",
    "batch_results": "\u6279\u91cf\u8bc6\u522b\u7ed3\u679c",
    "download_results": "\u4e0b\u8f7d\u6279\u91cf\u7ed3\u679c CSV",
    "detail": "\u5355\u56fe\u8be6\u60c5",
    "best_class": "\u6700\u53ef\u80fd\u7c7b\u522b",
    "confidence": "\u7f6e\u4fe1\u5ea6",
    "topk": "Top-K \u7ed3\u679c",
    "model_info": "\u6a21\u578b\u4fe1\u606f",
    "advice": "\u5904\u7406\u5efa\u8bae",
    "heatmap": "\u5173\u6ce8\u533a\u57df",
    "heatmap_caption": "Grad-CAM \u70ed\u529b\u56fe",
    "heatmap_error": "\u5f53\u524d\u6a21\u578b\u6682\u672a\u751f\u6210\u70ed\u529b\u56fe\uff1a",
    "tab_predict": "\u56fe\u7247\u8bc6\u522b",
    "tab_train": "\u81ea\u5b9a\u4e49\u8bad\u7ec3",
    "custom_train": "\u8bad\u7ec3\u81ea\u5df1\u7684\u6570\u636e\u96c6",
    "dataset_path": "\u6570\u636e\u96c6\u8def\u5f84",
    "run_name": "\u8bad\u7ec3\u4efb\u52a1\u540d",
    "prepared_dir": "\u5212\u5206\u540e\u6570\u636e\u4fdd\u5b58\u76ee\u5f55",
    "model_output": "\u6a21\u578b\u4fdd\u5b58\u8def\u5f84",
    "train_output": "\u8bad\u7ec3\u62a5\u544a\u8f93\u51fa\u76ee\u5f55",
    "epochs": "\u8bad\u7ec3\u8f6e\u6570",
    "batch_size": "Batch Size",
    "train_ratio": "\u8bad\u7ec3\u96c6\u6bd4\u4f8b",
    "val_ratio": "\u9a8c\u8bc1\u96c6\u6bd4\u4f8b",
    "max_per_class": "\u6bcf\u7c7b\u6700\u591a\u4f7f\u7528\u56fe\u7247\u6570\uff080 \u4e3a\u4e0d\u9650\u5236\uff09",
    "architecture": "\u6a21\u578b\u7ed3\u6784",
    "no_pretrained": "\u4e0d\u4f7f\u7528\u9884\u8bad\u7ec3\u6743\u91cd",
    "freeze_backbone": "\u53ea\u8bad\u7ec3\u5206\u7c7b\u5934",
    "force": "\u8986\u76d6\u5df2\u6709\u5212\u5206\u6570\u636e",
    "validate_dataset": "\u6821\u9a8c\u6570\u636e\u96c6",
    "prepare_split": "\u751f\u6210\u8bad\u7ec3\u96c6/\u6d4b\u8bd5\u96c6",
    "start_train": "\u5f00\u59cb\u8bad\u7ec3",
    "validation_ok": "\u6570\u636e\u96c6\u6821\u9a8c\u901a\u8fc7",
    "validation_failed": "\u6570\u636e\u96c6\u6821\u9a8c\u5931\u8d25",
    "prepared_ok": "\u6570\u636e\u96c6\u5212\u5206\u5df2\u751f\u6210",
    "training_done": "\u8bad\u7ec3\u5b8c\u6210",
    "training_log": "\u8bad\u7ec3\u65e5\u5fd7",
}


st.set_page_config(page_title=TEXT["page_title"], page_icon=None, layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 1.4rem;}
    [data-testid="stMetricValue"] {font-size: 1.7rem;}
    .result-row {
        border: 1px solid #d8ded9;
        border-radius: 8px;
        padding: 0.65rem 0.75rem;
        margin-bottom: 0.55rem;
        background: #fbfdfb;
    }
    .small-muted {color: #66736b; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_predictor(checkpoint_path: str, config_path: str) -> DiseasePredictor:
    return DiseasePredictor(checkpoint_path, config_path)


def resolve_user_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else {}


def csv_escape(value: object) -> str:
    text = str(value).replace('"', '""')
    return f'"{text}"'


def rows_to_csv(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(csv_escape(row.get(header, "")) for header in headers))
    return "\ufeff" + "\n".join(lines)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("._") or "custom_dataset"


def render_prediction_rows(predictions: list) -> None:
    for index, prediction in enumerate(predictions, start=1):
        probability = prediction.probability
        st.markdown(
            f"""
            <div class="result-row">
                <strong>{index}. {prediction.display_name}</strong>
                <div class="small-muted">{TEXT["confidence"]} {probability:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(probability)


def render_training_report(report_dir: Path) -> None:
    metrics = read_json(report_dir / "metrics.json")
    dataset_summary = read_json(report_dir / "dataset_summary.json")
    if not metrics:
        return

    with st.expander(TEXT["training_report"], expanded=False):
        col1, col2, col3 = st.columns(3)
        test_accuracy = metrics.get("test", {}).get("accuracy")
        best_val_accuracy = metrics.get("best_val_accuracy")
        total_images = dataset_summary.get("total_images")
        col1.metric(TEXT["test_acc"], f"{test_accuracy:.2%}" if isinstance(test_accuracy, float) else TEXT["none"])
        col2.metric(
            TEXT["best_val_acc"],
            f"{best_val_accuracy:.2%}" if isinstance(best_val_accuracy, float) else TEXT["none"],
        )
        col3.metric(TEXT["train_images"], str(total_images) if total_images else TEXT["none"])

        per_class = metrics.get("classification_report", {}).get("per_class", [])
        if per_class:
            st.dataframe(per_class, width="stretch", hide_index=True)

        confusion_path = report_dir / "confusion_matrix.csv"
        if confusion_path.exists():
            st.download_button(
                TEXT["download_confusion"],
                data=confusion_path.read_text(encoding="utf-8"),
                file_name="confusion_matrix.csv",
                mime="text/csv",
            )


def predict_images(predictor: DiseasePredictor, uploaded_files: list, top_k: int) -> tuple[list[dict[str, object]], list]:
    rows: list[dict[str, object]] = []
    details = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        predictions = predictor.predict(image, top_k=top_k)
        best = predictions[0]
        rows.append(
            {
                "file_name": uploaded_file.name,
                "predicted_class": best.class_name,
                "display_name": best.display_name,
                "confidence": round(best.probability, 6),
                "top_k": " | ".join(f"{item.display_name}:{item.probability:.4f}" for item in predictions),
            }
        )
        details.append((uploaded_file.name, image, predictions))
    return rows, details


def render_prediction_tab(
    config,
    checkpoint_path: str,
    report_dir: Path,
    top_k: int,
    enable_gradcam: bool,
) -> None:
    render_training_report(report_dir)

    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        uploaded_files = st.file_uploader(
            TEXT["upload"],
            type=("jpg", "jpeg", "png", "bmp", "webp"),
            accept_multiple_files=True,
        )
        if uploaded_files:
            first_image = Image.open(uploaded_files[0]).convert("RGB")
            st.image(first_image, caption=f'{TEXT["preview"]}: {uploaded_files[0].name}', width="stretch")
            if len(uploaded_files) > 1:
                st.caption(f"{len(uploaded_files)} files selected")
        else:
            st.info(TEXT["select_images"])

    with right:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            st.warning(f"{TEXT['missing_model']}{checkpoint}")
            st.code(
                "python scripts/create_demo_dataset.py --output data/raw/demo_leaf --images-per-class 24\n"
                "python train.py --data-dir data/raw/demo_leaf --epochs 1 --checkpoint models/demo_synthetic.pt",
                language="powershell",
            )
            st.stop()

        if not uploaded_files:
            st.empty()
            return

        with st.spinner(TEXT["predicting"]):
            predictor = load_predictor(str(checkpoint), str(ROOT / "config.yaml"))
            result_rows, details = predict_images(predictor, uploaded_files, top_k)

        st.subheader(TEXT["batch_results"])
        st.dataframe(result_rows, width="stretch", hide_index=True)
        st.download_button(
            TEXT["download_results"],
            data=rows_to_csv(result_rows),
            file_name="plant_disease_predictions.csv",
            mime="text/csv",
        )

        selected_name = st.selectbox(TEXT["detail"], [name for name, _, _ in details])
        selected = next(item for item in details if item[0] == selected_name)
        _, image, predictions = selected
        best = predictions[0]
        st.caption(TEXT["best_class"])
        st.markdown(f"### {best.display_name}")
        st.metric(TEXT["confidence"], f"{best.probability:.2%}")

        st.subheader(TEXT["topk"])
        render_prediction_rows(predictions)

        with st.expander(TEXT["model_info"], expanded=False):
            st.write(
                {
                    "architecture": predictor.model_config.get("architecture"),
                    "classes": len(predictor.class_names),
                    "device": str(predictor.device),
                    "metrics": predictor.metrics,
                }
            )

        st.subheader(TEXT["advice"])
        for advice in get_recommendations(best.class_name):
            st.markdown(f"- {advice}")

        if enable_gradcam:
            st.subheader(TEXT["heatmap"])
            try:
                heatmap = predictor.gradcam(image, class_index=best.index)
                st.image(heatmap, caption=TEXT["heatmap_caption"], width="stretch")
            except Exception as exc:
                st.info(f"{TEXT['heatmap_error']}{exc}")


def custom_training_paths(run_name: str, prepared_value: str, checkpoint_value: str, output_value: str) -> tuple[Path, str, str]:
    prepared_dir = resolve_user_path(prepared_value) if prepared_value else ROOT / "data" / "processed" / f"custom_{run_name}"
    checkpoint = checkpoint_value or f"models/{run_name}.pt"
    output_dir = output_value or f"outputs/runs/{run_name}"
    return prepared_dir, checkpoint, output_dir


def prepare_custom_dataset(
    source: Path,
    prepared_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    max_per_class: int | None,
    force: bool,
) -> Path:
    if is_saved_split_dataset(source):
        return source
    manifest = copy_split_dataset(
        source=source,
        output=prepared_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        max_per_class=max_per_class,
        force=force,
    )
    return Path(str(manifest["output"]))


def render_custom_training_tab(config) -> None:
    st.subheader(TEXT["custom_train"])
    left, right = st.columns(2, gap="large")

    with left:
        dataset_path = st.text_input(TEXT["dataset_path"], value=str(ROOT / "data" / "raw" / "plantvillage"))
        run_name = slugify(st.text_input(TEXT["run_name"], value="my_leaf_model"))
        prepared_value = st.text_input(TEXT["prepared_dir"], value="")
        checkpoint_value = st.text_input(TEXT["model_output"], value=f"models/{run_name}.pt")
        output_value = st.text_input(TEXT["train_output"], value=f"outputs/runs/{run_name}")

    with right:
        architecture = st.selectbox(
            TEXT["architecture"],
            ("mobilenet_v3_small", "resnet18", "efficientnet_b0"),
            index=0,
        )
        epochs = st.number_input(TEXT["epochs"], min_value=1, max_value=100, value=10, step=1)
        batch_size = st.number_input(TEXT["batch_size"], min_value=1, max_value=128, value=16, step=1)
        train_ratio = st.slider(TEXT["train_ratio"], min_value=0.5, max_value=0.9, value=0.7, step=0.05)
        val_ratio = st.slider(TEXT["val_ratio"], min_value=0.05, max_value=0.3, value=0.15, step=0.05)
        max_per_class_value = st.number_input(TEXT["max_per_class"], min_value=0, max_value=10000, value=0, step=10)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=int(config.seed), step=1)
        no_pretrained = st.checkbox(TEXT["no_pretrained"], value=False)
        freeze_backbone = st.checkbox(TEXT["freeze_backbone"], value=False)
        force = st.checkbox(TEXT["force"], value=True)

    source = resolve_user_path(dataset_path)
    prepared_dir, checkpoint, output_dir = custom_training_paths(run_name, prepared_value, checkpoint_value, output_value)
    max_per_class = int(max_per_class_value) if max_per_class_value else None

    col1, col2, col3 = st.columns(3)

    if col1.button(TEXT["validate_dataset"], width="stretch"):
        report = validate_dataset(source)
        if report["valid"]:
            st.success(TEXT["validation_ok"])
        else:
            st.error(TEXT["validation_failed"])
        st.json(report)

    if col2.button(TEXT["prepare_split"], width="stretch"):
        report = validate_dataset(source)
        if not report["valid"]:
            st.error(TEXT["validation_failed"])
            st.json(report)
        else:
            try:
                training_data_dir = prepare_custom_dataset(
                    source=source,
                    prepared_dir=prepared_dir,
                    train_ratio=float(train_ratio),
                    val_ratio=float(val_ratio),
                    seed=int(seed),
                    max_per_class=max_per_class,
                    force=force,
                )
                leakage_report = check_leakage(training_data_dir)
                st.success(f"{TEXT['prepared_ok']}: {training_data_dir}")
                st.json(leakage_report)
            except Exception as exc:
                st.error(str(exc))

    if col3.button(TEXT["start_train"], width="stretch"):
        report = validate_dataset(source)
        if not report["valid"]:
            st.error(TEXT["validation_failed"])
            st.json(report)
            return

        log_buffer = StringIO()
        try:
            with st.spinner(TEXT["start_train"]):
                with redirect_stdout(log_buffer):
                    training_data_dir = prepare_custom_dataset(
                        source=source,
                        prepared_dir=prepared_dir,
                        train_ratio=float(train_ratio),
                        val_ratio=float(val_ratio),
                        seed=int(seed),
                        max_per_class=max_per_class,
                        force=force,
                    )
                    output_path = resolve_user_path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    leakage_report = check_leakage(training_data_dir)
                    (output_path / "leakage_report.json").write_text(
                        json.dumps(leakage_report, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    overrides = build_overrides(
                        data_dir=str(training_data_dir),
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        checkpoint=checkpoint,
                        architecture=str(architecture),
                        output_dir=output_dir,
                        pretrained=False if no_pretrained else None,
                        freeze_backbone=True if freeze_backbone else None,
                    )
                    overrides.setdefault("project", {})["seed"] = int(seed)
                    train_from_config(config.with_overrides(**overrides), allow_auto_download=False)
            st.success(TEXT["training_done"])
            st.write({"model": str(resolve_user_path(checkpoint)), "reports": str(resolve_user_path(output_dir))})
            st.code(log_buffer.getvalue(), language="text")
        except Exception as exc:
            st.error(str(exc))
            if log_buffer.getvalue():
                st.subheader(TEXT["training_log"])
                st.code(log_buffer.getvalue(), language="text")


def main() -> None:
    config = load_config(ROOT / "config.yaml", root_dir=ROOT)
    configured_checkpoint = config.checkpoint_path
    demo_checkpoint = ROOT / "models" / "demo_synthetic.pt"
    use_demo_default = not configured_checkpoint.exists() and demo_checkpoint.exists()
    default_checkpoint = str(demo_checkpoint if use_demo_default else configured_checkpoint)

    with st.sidebar:
        st.header(TEXT["settings"])
        checkpoint_path = st.text_input(TEXT["checkpoint"], value=default_checkpoint)
        report_dir = resolve_user_path(st.text_input(TEXT["report_dir"], value=str(config.output_dir)))
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=int(config.section("inference")["top_k"]))
        enable_gradcam = st.toggle(TEXT["show_heatmap"], value=True)
        if use_demo_default:
            st.info(TEXT["demo_notice"])
        st.caption(TEXT["caption"])

    st.title(TEXT["page_title"])
    predict_tab, train_tab = st.tabs([TEXT["tab_predict"], TEXT["tab_train"]])
    with predict_tab:
        render_prediction_tab(config, checkpoint_path, report_dir, top_k, enable_gradcam)
    with train_tab:
        render_custom_training_tab(config)


if __name__ == "__main__":
    main()
