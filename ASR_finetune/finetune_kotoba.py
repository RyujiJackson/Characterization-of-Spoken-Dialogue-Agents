"""
Finetune KotobaWhisper on the JSUT corpus.

The script is intentionally self contained so you can run:

python ASR_finetune/finetune_kotoba.py \
  --model_name kotoba-tech/kotoba-whisper-v2.0 \
  --output_dir runs/kotoba-jsut \
  --num_train_epochs 5

Assumptions:
- JSUT is available from Hugging Face as `kamo-naoyuki/jsut`.
- Audio is resampled to 16 kHz for Whisper.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Union

import yaml

import torch
from datasets import Audio, DatasetDict, load_dataset
import evaluate
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from dataclasses import dataclass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune KotobaWhisper on JSUT")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file to override defaults.",
    )
    parser.add_argument(
        "--model_name",
        default="kotoba-tech/kotoba-whisper-v2.2",
        help="Base KotobaWhisper checkpoint to finetune.",
    )
    parser.add_argument(
        "--dataset_name",
        default="PlayMaker13/commonvoice_jsut_split",
        help="Dataset identifier for JSUT on Hugging Face.",
    )
    parser.add_argument(
        "--train_split",
        default="train",
        help="Split name to use for training.",
    )
    parser.add_argument(
        "--eval_split",
        default=None,
        help="Optional existing split name to use for eval (uses random split when omitted).",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.02,
        help="Eval holdout ratio when creating a split.",
    )
    parser.add_argument(
        "--text_column",
        default="transcription",
        help="Column in JSUT containing transcripts.",
    )
    parser.add_argument(
        "--audio_column",
        default="audio",
        help="Column in JSUT containing audio.",
    )
    parser.add_argument(
        "--language",
        default="japanese",
        help="Language hint for Whisper processor.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper decoding task.",
    )
    parser.add_argument(
        "--output_dir",
        default="runs/kotoba-jsut",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Eval batch size per device.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Warmup steps for the scheduler.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation to fit memory.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional cap on train samples for quick runs.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional cap on eval samples for quick runs.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the finetuned model to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        default=None,
        help="Hub repo id when pushing to Hub.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path or hub id to resume training from.",
    )
    args = parser.parse_args()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            if not hasattr(args, key):
                raise ValueError(f"Unknown config key: {key}")
            setattr(args, key, value)

    numeric_fields = {
        "learning_rate": float,
        "warmup_steps": int,
        "num_train_epochs": float,
        "eval_steps": int,
        "save_steps": int,
        "logging_steps": int,
        "per_device_train_batch_size": int,
        "per_device_eval_batch_size": int,
        "gradient_accumulation_steps": int,
        "eval_ratio": float,
    }
    for field, caster in numeric_fields.items():
        value = getattr(args, field, None)
        if value is not None:
            setattr(args, field, caster(value))
    return args


def load_and_split_dataset(
    dataset_name: str,
    train_split: str,
    eval_split: Optional[str],
    eval_ratio: float,
    audio_column: str,
    sampling_rate: int,
) -> DatasetDict:
    raw = load_dataset(dataset_name)
    if train_split not in raw:
        raise ValueError(f"Split {train_split} not found in dataset {dataset_name}")

    # Prefer an existing eval split if present; otherwise fall back to a random split.
    resolved_eval_split: Optional[str] = None
    for candidate in [eval_split, "validation", "valid", "eval", "test"]:
        if candidate and candidate in raw:
            resolved_eval_split = candidate
            break

    if resolved_eval_split:
        dataset = DatasetDict(train=raw[train_split], eval=raw[resolved_eval_split])
    else:
        split = raw[train_split].train_test_split(test_size=eval_ratio, seed=42)
        dataset = DatasetDict(train=split["train"], eval=split["test"])

    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sampling_rate))
    return dataset


def prepare_processor(model_name: str, language: str, task: str) -> WhisperProcessor:
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name,
        language=language,
        task=task,
    )
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    return processor


def preprocess_dataset(
    dataset: DatasetDict,
    processor: WhisperProcessor,
    audio_column: str,
    text_column: str,
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
) -> DatasetDict:
    available_cols = set(dataset["train"].column_names)
    if text_column not in available_cols:
        fallback_order = ["sentence", "text", "transcript", "transcription", "kana"]
        for candidate in fallback_order:
            if candidate in available_cols:
                text_column = candidate
                break
        else:
            raise KeyError(
                f"Text column '{text_column}' not found. Available columns: {sorted(available_cols)}"
            )

    def _prepare_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio = batch[audio_column]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch[text_column]).input_ids
        return batch

    processed = DatasetDict()
    train_ds = dataset["train"]
    eval_ds = dataset["eval"]

    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples:
        eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))

    processed["train"] = train_ds.map(
        _prepare_batch,
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count() or 1,
    )
    processed["eval"] = eval_ds.map(
        _prepare_batch,
        remove_columns=dataset["eval"].column_names,
        num_proc=os.cpu_count() or 1,
    )
    return processed


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def build_model_and_collator(
    model_name: str,
    processor: WhisperProcessor,
    language: str,
    task: str,
) -> tuple[WhisperForConditionalGeneration, DataCollatorSpeechSeq2SeqWithPadding]:
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    return model, collator


def build_trainer(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    data: DatasetDict,
    collator: DataCollatorSpeechSeq2SeqWithPadding,
    args: argparse.Namespace,
) -> Seq2SeqTrainer:
    metric = evaluate.load("cer")

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        pred_ids = eval_pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = eval_pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        cer = metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,  # disable to avoid double-backward error
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        label_smoothing_factor=0.0,
    )

    model.config.use_cache = False

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        data_collator=collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )
    return trainer


def main() -> None:
    args = parse_args()

    sampling_rate = 16000
    processor = prepare_processor(args.model_name, args.language, args.task)
    dataset = load_and_split_dataset(
        args.dataset_name,
        args.train_split,
        args.eval_split,
        args.eval_ratio,
        args.audio_column,
        sampling_rate,
    )
    dataset = preprocess_dataset(
        dataset,
        processor,
        args.audio_column,
        args.text_column,
        args.max_train_samples,
        args.max_eval_samples,
    )

    model, collator = build_model_and_collator(
        args.model_name, processor, args.language, args.task
    )
    trainer = build_trainer(model, processor, dataset, collator, args)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
