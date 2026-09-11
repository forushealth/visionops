"""Keras training utilities with CSV-driven input pipelines and MLflow hooks."""

from __future__ import annotations

import copy
import importlib
import os
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import yaml

from .ImageProcessingOptions import build_image_processing_pipelines
from ._quiet import quiet_print
from ._utils import run_cleanup_steps

try:
    import albumentations as A
except ModuleNotFoundError:
    A = None


class KerasFitTrainer:
    """
    Keras trainer with CSV-based dataset loading and optional MLflow logging.

    This trainer reads image/label pairs from train/validation CSV files,
    builds tf.data pipelines, compiles task-aware heads/losses for BC/MCC/MLC,
    and logs training artifacts for downstream evaluation.
    """

    def __init__(self, config_path: str, config: Optional[Dict[str, Any]] = None):
        """Load trainer configuration and initialize runtime/training state."""
        if config is None:
            with open(config_path, encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
        if not isinstance(config, dict):
            raise ValueError("Keras trainer configuration must be a mapping")
        self.config = copy.deepcopy(config)

        # Keep XLA JIT disabled by default; users can opt in through config.
        self.disable_xla_jit = bool(self.config.get("disable_xla_jit", True))
        if self.disable_xla_jit:
            try:
                tf.config.optimizer.set_jit(False)
            except Exception:
                pass

        # Reduce startup overhead and avoid aggressive GPU reservation.
        gpu_devices = []
        try:
            gpu_devices = tf.config.list_physical_devices("GPU")
            for gpu in gpu_devices:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
        except Exception:
            gpu_devices = []

        use_mirrored_cfg = self.config.get("use_mirrored_strategy", None)
        if use_mirrored_cfg is None:
            self.use_mirrored_strategy = len(gpu_devices) > 1
        else:
            self.use_mirrored_strategy = bool(use_mirrored_cfg)

        # Setup distribution strategy
        if self.use_mirrored_strategy:
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            self.strategy = tf.distribute.get_strategy()
        quiet_print(f"[KerasFitTrainer] Number of devices: {self.strategy.num_replicas_in_sync}")
        self.config_path = config_path
        # Enable mixed precision if configured
        if self.config.get('use_mixed_precision', False):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

        # Parse image size from config (e.g., [224, 224, 3])
        self.image_size_tuple = self.config['image_size']
        self.image_height, self.image_width, self.image_channels = self.image_size_tuple
        data_cfg = self.config.get("data", {}) if isinstance(self.config.get("data", {}), dict) else {}
        tfms_cfg = data_cfg.get("tfms", data_cfg.get("transforms", {}))
        if not tfms_cfg and isinstance(self.config.get("transforms"), dict):
            tfms_cfg = self.config.get("transforms")
        self.image_processing = build_image_processing_pipelines(
            tfms_cfg if isinstance(tfms_cfg, dict) else {},
            image_size=self.image_size_tuple,
        )
        if self.image_processing.get("enabled", False):
            names = self.image_processing.get("section_names", {})
            print(
                "[KerasFitTrainer] Using config-driven image processing "
                f"pre={names.get('pre', [])}, aug={names.get('aug', [])}, post={names.get('post', [])}"
            )
        elif tfms_cfg:
            print(
                "[KerasFitTrainer] Image processing config detected but not enabled: "
                f"{self.image_processing.get('reason', 'unknown')}"
            )

        # Internal dataset state
        self.train_ds = None
        self.val_ds = None
        self.num_train_samples = 0
        self.num_val_samples = 0
        self.class_weights = None
        self.model = None

        # Load custom callbacks from config
        self.callbacks = self.load_callbacks()

        # If MLflow logging is enabled, initialize MLflowLogger and add its Keras callback
        self.use_mlflow = self.config.get("logging", {}).get("use_mlflow", False)
        self.mlflow_logger = None
        self.auto_end_mlflow_run = True
        self.train_history_csv_path = None
        self.val_logits_csv_path = None
        self.task_type = str(self.config.get("task_type", "mcc")).lower().strip()
        if self.task_type == "bc":
            self.task_type = "mcc"
        self.num_classes = int(self.config.get("num_classes", 2))
        self.label_to_idx = None
        self.train_csv_path = self._cfg("train_data_csv", "Train_data_csv")
        self.val_csv_path = self._cfg("val_data_csv", "Val_data_csv")
        self.data_directory_path = self._cfg("data_directory_path", "Data_directory_path")
        self.path_column = self._cfg("image_path_column", "Image_path_column", default="path")
        self.label_column = self._cfg(
            "image_label_column",
            "Image_label_column",
            "image_categorical_label_column",
            "Image_categorical_label_column",
            default="labels",
        )

        if self.task_type == "seg":
            raise ValueError("[KerasFitTrainer] Segmentation training is not supported by this trainer.")
        if self.task_type not in {"mcc", "mlc"}:
            raise ValueError("[KerasFitTrainer] task_type must be one of {'bc','mcc','mlc'}.")
        if self.task_type == "mlc" and self.num_classes < 2:
            raise ValueError("[KerasFitTrainer] For mlc task, num_classes must be >= 2.")
        if not self.train_csv_path or not self.val_csv_path:
            raise ValueError(
                "[KerasFitTrainer] Missing train/val CSV paths. Use train_data_csv/val_data_csv or Train_data_csv/Val_data_csv."
            )
        if not self.data_directory_path:
            raise ValueError(
                "[KerasFitTrainer] Missing data directory path. Use data_directory_path or Data_directory_path."
            )
        self.model_builder = self._resolve_model_builder()

    def _cfg(self, *keys, default=None):
        """Return the first available configuration value from candidate keys."""
        for key in keys:
            if key in self.config:
                return self.config[key]
        return default

    @staticmethod
    def _resolve_column_in_df(df: pd.DataFrame, configured: str, fallbacks: List[str], kind: str) -> str:
        """Resolve an existing column name from preferred and fallback candidates."""
        if configured in df.columns:
            return configured
        for col in fallbacks:
            if col in df.columns:
                return col
        raise ValueError(
            f"[KerasFitTrainer] Could not find {kind} column in CSV. "
            f"Tried configured='{configured}' and fallbacks={fallbacks}. "
            f"Available columns: {df.columns.tolist()}"
        )


    def load_callbacks(self) -> list:
        """Load custom callbacks from config."""
        callbacks = []
        if 'callbacks' in self.config:
            for configured_callback in self.config['callbacks']:
                if not isinstance(configured_callback, dict):
                    raise ValueError("Each Keras callback configuration must be a mapping")
                cb_cfg = dict(configured_callback)
                try:
                    cb_name = cb_cfg.pop('name')
                except KeyError as exc:
                    raise ValueError("Keras callback configuration requires a 'name'") from exc
                cb_cls = getattr(tf.keras.callbacks, cb_name)
                cb_instance = cb_cls(**cb_cfg)
                callbacks.append(cb_instance)
                print(f"[KerasFitTrainer] Added callback: {cb_name} with params: {cb_cfg}")
        return callbacks

    def _resolve_model_builder(self) -> Callable[..., Any]:
        model_builder = self.config.get("model_builder")
        if callable(model_builder):
            return model_builder

        if isinstance(model_builder, str) and model_builder.strip():
            ref = model_builder.strip()
            module_name = None
            object_name = None
            if ":" in ref:
                module_name, object_name = ref.split(":", 1)
            elif "." in ref:
                module_name, object_name = ref.rsplit(".", 1)
            if module_name and object_name:
                module = importlib.import_module(module_name)
                builder = getattr(module, object_name)
                if not callable(builder):
                    raise ValueError(f"[KerasFitTrainer] model_builder '{ref}' is not callable.")
                return builder

        return self._default_model_builder

    def _default_model_builder(
        self,
        num_classes: int,
        input_shape: tuple = None,
        task_type: str = None,
    ):
        task = str(task_type or self.task_type).lower().strip()
        shape = input_shape or (self.image_height, self.image_width, self.image_channels)

        inputs = tf.keras.Input(shape=shape)
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        if task == "mlc":
            outputs = tf.keras.layers.Dense(max(1, int(num_classes)), name="logits")(x)
        elif task == "mcc" and int(num_classes) > 2:
            outputs = tf.keras.layers.Dense(int(num_classes), name="logits")(x)
        else:
            outputs = tf.keras.layers.Dense(1, name="logits")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"keras_{task}_model")

    def _build_model(self):
        input_shape = (self.image_height, self.image_width, self.image_channels)
        builder = self.model_builder

        call_trials = [
            {"num_classes": self.num_classes, "input_shape": input_shape, "task_type": self.task_type},
            {"num_classes": self.num_classes, "input_shape": input_shape},
            {"num_classes": self.num_classes},
        ]
        for kwargs in call_trials:
            try:
                return builder(**kwargs)
            except TypeError:
                continue

        try:
            return builder(input_shape, self.num_classes)
        except TypeError as exc:
            raise TypeError(
                "[KerasFitTrainer] model_builder signature is unsupported. "
                "Expected builder(num_classes=..., input_shape=..., task_type=...) or compatible variants."
            ) from exc

    @staticmethod
    def _split_mlc_labels(value: Any) -> List[str]:
        if isinstance(value, (list, tuple, np.ndarray)):
            return [str(x).strip() for x in value if str(x).strip()]
        if value is None:
            return []
        if isinstance(value, float) and np.isnan(value):
            return []
        text = str(value).strip()
        if not text:
            return []
        for sep in ["|", ";", " "]:
            text = text.replace(sep, ",")
        return [t.strip() for t in text.split(",") if t.strip()]

    def _encode_labels(self, df: pd.DataFrame, is_training: bool) -> tuple[list, tf.dtypes.DType]:
        label_col = self._resolve_column_in_df(
            df,
            self.label_column,
            ["labels", "categorical_label", "label", "target"],
            "label",
        )
        if self.task_type == "mlc":
            labels_series = df[label_col]
            if self.label_to_idx is None or is_training:
                vocab = sorted({token for v in labels_series for token in self._split_mlc_labels(v)})
                if not vocab:
                    raise ValueError("[KerasFitTrainer] No labels found for mlc task.")
                self.label_to_idx = {name: idx for idx, name in enumerate(vocab)}
                if self.num_classes != len(vocab):
                    self.num_classes = len(vocab)
                    self.config["num_classes"] = self.num_classes

            encoded = []
            for value in labels_series:
                vec = np.zeros(self.num_classes, dtype=np.float32)
                for token in self._split_mlc_labels(value):
                    idx = self.label_to_idx.get(token)
                    if idx is not None:
                        vec[idx] = 1.0
                encoded.append(vec.tolist())
            return encoded, tf.float32

        labels_series = df[label_col]
        if self.label_to_idx is None or is_training:
            if pd.api.types.is_numeric_dtype(labels_series):
                numeric = pd.to_numeric(labels_series, errors="raise").astype(int)
                classes = sorted(numeric.unique().tolist())
                self.label_to_idx = {int(cls): i for i, cls in enumerate(classes)}
                self.label_to_idx.update({str(cls): i for i, cls in enumerate(classes)})
            else:
                classes = sorted(labels_series.astype(str).unique().tolist())
                self.label_to_idx = {cls: i for i, cls in enumerate(classes)}

            inferred = len(classes)
            if self.task_type == "mcc" and is_training:
                self.num_classes = inferred
                self.config["num_classes"] = self.num_classes
            elif self.num_classes < inferred:
                self.num_classes = inferred
                self.config["num_classes"] = self.num_classes

        encoded: List[int] = []
        unknown = []
        for raw_val in labels_series.tolist():
            if raw_val in self.label_to_idx:
                encoded.append(int(self.label_to_idx[raw_val]))
                continue
            raw_str = str(raw_val)
            if raw_str in self.label_to_idx:
                encoded.append(int(self.label_to_idx[raw_str]))
                continue
            unknown.append(raw_val)

        if unknown:
            unknown_vals = sorted({str(x) for x in unknown})
            raise ValueError(f"[KerasFitTrainer] Unknown label(s) in dataset split: {unknown_vals}")

        return encoded, tf.int64

    def _compile_model(self, model, jit_compile=None, run_eagerly=None):
        optimizer = tf.keras.optimizers.Adam(float(self.config["learning_rate"]))

        if self.task_type == "mlc":
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ]
        elif self.task_type == "mcc" and self.num_classes > 2:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
        else:
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ]

        if jit_compile is None:
            jit_compile = bool(self.config.get("jit_compile", False))
        if run_eagerly is None:
            run_eagerly = bool(self.config.get("run_eagerly", False))

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics,
            jit_compile=bool(jit_compile),
            run_eagerly=bool(run_eagerly),
        )
        return model
    def filter_unreadable_images(self, file_paths: list, labels: list) -> tuple:
        """
        Filter out files that PIL cannot open or verify.

        Any truly corrupted
        JPEG/PNG/GIF/BMP will be skipped (with one Python-side print), and
        this prevents ``tf.image.decode_*`` from receiving unreadable files.

        Returns:
            (valid_paths, valid_labels)

        Raises:
            ValueError if no files remain.
        """
        valid_paths, valid_labels = [], []
        for path, label in zip(file_paths, labels, strict=True):
            try:
                # open + verify integrity
                with Image.open(path) as img:
                    img.verify()
                # if no exception, keep it
                valid_paths.append(path)
                valid_labels.append(label)
            except Exception as e:
                print(f"[KerasFitTrainer] Skipping corrupted image {path}: {e}")

        if not valid_paths:
            raise ValueError(
                "[KerasFitTrainer] After filtering unreadable images, no files remain."
            )
        return valid_paths, valid_labels



    def create_dataset(
        self,
        csv_path: str,
        batch_size: int,
        is_training: bool = True,
        *,
        max_samples: int | None = None,
        validate_images: bool | None = None,
        use_cache: bool | None = None,
        prefetch_buffer: int | Any | None = None,
    ) -> tuple:
        """
        Read a CSV and build a prepared tf.data pipeline.

        Args:
            csv_path (str): Path to CSV file.
            batch_size (int): Batch size.
            is_training (bool): Whether this is a training dataset.

        Returns:
            tuple: (dataset, num_samples)
        """
        if max_samples is None:
            split_key = "train_dataset_max_samples" if is_training else "val_dataset_max_samples"
            max_samples = self.config.get(split_key, self.config.get("dataset_max_samples"))
        try:
            max_samples_int = int(max_samples) if max_samples is not None else None
            if max_samples_int is not None and max_samples_int <= 0:
                max_samples_int = None
        except Exception:
            max_samples_int = None

        if max_samples_int is not None:
            df = pd.read_csv(csv_path, nrows=max_samples_int)
        else:
            df = pd.read_csv(csv_path)
        path_col = self._resolve_column_in_df(
            df,
            self.path_column,
            ["path", "image_path", "filepath", "file_path"],
            "image path",
        )
        if max_samples_int is not None and len(df) > max_samples_int:
            df = df.iloc[:max_samples_int].copy()

        file_labels, label_dtype = self._encode_labels(df, is_training=is_training)

        base_dir = self.data_directory_path
        df['full_path'] = df[path_col].astype(str).apply(lambda x: os.path.join(base_dir, x))

        file_paths = df['full_path'].tolist()

        if validate_images is None:
            validate_images = bool(self.config.get("validate_images", True))
        if validate_images:
            valid_paths, valid_labels = self.filter_unreadable_images(file_paths, file_labels)
        else:
            valid_paths, valid_labels = file_paths, file_labels
        if not valid_paths:
            raise ValueError("[KerasFitTrainer] All images are unreadable or missing!")
        num_samples = len(valid_paths)

        image_paths = tf.constant(valid_paths)
        labels_tf = tf.constant(valid_labels, dtype=label_dtype)

        active_pipeline = None
        post_has_normalization = False
        if isinstance(self.image_processing, dict) and self.image_processing.get("enabled", False):
            active_pipeline = self.image_processing.get("train" if is_training else "eval")
            post_has_normalization = bool(self.image_processing.get("post_has_normalization", False))

        def decode_and_resize(path: tf.Tensor, label: tf.Tensor) -> tuple:
            """
            Load one image via PIL and return float32 tensor in [0, 1].

            Works for both eager + graph because we explicitly
            call .numpy() on the scalar path tensor.
            """
            def _pil_load(path_tensor):
                # path_tensor is a 0-D Tensor; turn into Python bytes → str
                if isinstance(path_tensor, tf.Tensor):
                    path_bytes = path_tensor.numpy()       # b'/full/path/img.jpg'
                else:                                      # safety for eager/bytes case
                    path_bytes = path_tensor
                path_str = path_bytes.decode("utf-8")

                with Image.open(path_str) as img:
                    img = img.convert("RGB")
                    img = img.resize((self.image_width, self.image_height))
                image_np = np.asarray(img, dtype=np.uint8)

                if active_pipeline is not None:
                    out = active_pipeline(image=image_np)
                    image_np = out["image"]
                    if not isinstance(image_np, np.ndarray):
                        image_np = np.asarray(image_np)
                    if np.issubdtype(image_np.dtype, np.integer):
                        image_np = image_np.astype(np.float32) / 255.0
                    else:
                        image_np = image_np.astype(np.float32)
                        if (not post_has_normalization) and image_np.size > 0 and float(np.max(image_np)) > 1.0:
                            image_np = image_np / 255.0
                    return image_np

                return image_np.astype(np.float32) / 255.0

            img = tf.py_function(_pil_load, [path], tf.float32)
            img.set_shape([self.image_height, self.image_width, self.image_channels])
            return img, label



        def augment(img: tf.Tensor, label: tf.Tensor) -> tuple:
            if is_training and active_pipeline is None:
                if self.config.get('use_albumentations', False) and A:
                    transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.Rotate(limit=30, p=0.3),
                    ])
                    aug_img = transform(image=img.numpy())['image']
                    img = tf.convert_to_tensor(aug_img, tf.float32)
                else:
                    img = tf.image.random_flip_left_right(img)
            return img, label

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_tf))
        dataset = dataset.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        if use_cache is None:
            split_cache_key = "train_dataset_cache" if is_training else "val_dataset_cache"
            use_cache = bool(self.config.get(split_cache_key, self.config.get("dataset_cache", False)))
        if use_cache:
            dataset = dataset.cache()
        if is_training:
            dataset = dataset.shuffle(self.config.get('shuffle_buffer_size', 1000))
        dataset = dataset.batch(batch_size)
        if prefetch_buffer is None:
            cfg_prefetch = self.config.get("prefetch_buffer_size", "autotune")
            if isinstance(cfg_prefetch, str) and cfg_prefetch.strip().lower() in {"auto", "autotune"}:
                prefetch_buffer = tf.data.AUTOTUNE
            else:
                try:
                    prefetch_buffer = int(cfg_prefetch)
                except Exception:
                    prefetch_buffer = tf.data.AUTOTUNE
        if prefetch_buffer is tf.data.AUTOTUNE:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        else:
            try:
                dataset = dataset.prefetch(max(1, int(prefetch_buffer)))
            except Exception:
                dataset = dataset.prefetch(tf.data.AUTOTUNE)

        if is_training and self.config.get('use_class_weights', False) and self.task_type == "mcc":
            label_arr = np.array(valid_labels)
            uniq_vals, counts = np.unique(label_arr, return_counts=True)
            total = label_arr.size
            n_classes = len(uniq_vals)
            self.class_weights = {
                int(value): (1.0 / count) * (total / n_classes)
                for value, count in zip(uniq_vals, counts, strict=True)
            }
            print("[KerasFitTrainer] Class Weights:", self.class_weights)
        if not is_training:
            # Save for later so lengths always match
            self.valid_val_df = pd.DataFrame({
                "full_path": valid_paths,
                "label":     valid_labels
            })

        return dataset, num_samples

    def data_visualization(self, dataset_type: str = 'train', num_images: int = 9) -> None:
        """
        Visualizes a batch of images from the training or validation dataset.

        Args:
            dataset_type (str): 'train' or 'val'
            num_images (int): Number of images to display.
        """
        if dataset_type not in ['train', 'val']:
            raise ValueError("[KerasFitTrainer] dataset_type must be 'train' or 'val'.")

        if dataset_type == 'train' and self.train_ds is None:
            self.train_ds, self.num_train_samples = self.create_dataset(
                self.train_csv_path,
                batch_size=self.config['batch_size'],
                is_training=True
            )
        elif dataset_type == 'val' and self.val_ds is None:
            self.val_ds, self.num_val_samples = self.create_dataset(
                self.val_csv_path,
                batch_size=self.config['batch_size'],
                is_training=False
            )
        ds = self.train_ds if dataset_type == 'train' else self.val_ds
        if ds is None:
            raise ValueError(f"[KerasFitTrainer] {dataset_type} dataset is not loaded.")

        idx_to_label = {}
        if isinstance(self.label_to_idx, dict):
            for raw_label, mapped_idx in self.label_to_idx.items():
                try:
                    idx = int(mapped_idx)
                except Exception:
                    continue
                raw_str = str(raw_label)
                prev = idx_to_label.get(idx)
                # Prefer human-readable text labels over numeric aliases.
                if prev is None or (str(prev).isdigit() and not raw_str.isdigit()):
                    idx_to_label[idx] = raw_str

        def _label_to_text(label_value: Any) -> str:
            arr = np.asarray(label_value)
            if arr.ndim == 0:
                value = arr.item()
                if isinstance(value, (np.integer, int)):
                    return str(idx_to_label.get(int(value), int(value)))
                if isinstance(value, (np.floating, float)) and np.isfinite(value):
                    rounded = int(round(float(value)))
                    if abs(float(value) - rounded) < 1e-6:
                        return str(idx_to_label.get(rounded, rounded))
                return str(value)

            flat = arr.reshape(-1)
            if flat.size == 0:
                return "N/A"
            if np.issubdtype(flat.dtype, np.number):
                vals = flat.astype(np.float32)
                if vals.size == 1:
                    return _label_to_text(vals[0])
                if np.all((vals >= 0.0) & (vals <= 1.0)):
                    active = np.where(vals > 0.5)[0].tolist()
                    if active:
                        return ", ".join(str(idx_to_label.get(int(i), int(i))) for i in active)
                if np.isclose(float(np.sum(vals)), 1.0, atol=1e-3):
                    cls_idx = int(np.argmax(vals))
                    return str(idx_to_label.get(cls_idx, cls_idx))
                return np.array2string(np.round(vals, 3), separator=",")

            return ", ".join(str(x) for x in flat.tolist())

        for images, labels in ds.take(1):
            display_count = max(1, min(int(num_images), int(images.shape[0])))
            cols = int(np.ceil(np.sqrt(display_count)))
            rows = int(np.ceil(display_count / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), dpi=100, squeeze=False)

            for i in range(rows * cols):
                ax = axes[i // cols, i % cols]
                if i >= display_count:
                    ax.axis("off")
                    continue
                image_np = images[i].numpy().astype("float32")
                if image_np.size > 0 and (float(np.nanmin(image_np)) < 0.0 or float(np.nanmax(image_np)) > 1.0):
                    mn = float(np.nanmin(image_np))
                    mx = float(np.nanmax(image_np))
                    denom = (mx - mn) if abs(mx - mn) > 1e-6 else 1.0
                    image_np = (image_np - mn) / denom
                ax.imshow(np.clip(image_np, 0.0, 1.0))
                ax.set_title(f"Label: {_label_to_text(labels[i].numpy())}", fontsize=10)
                ax.axis("off")
            fig.tight_layout()
        plt.show()

    def fit(self):
        """
        Train the configured model and emit training artifacts.

        Returns:
            history: The history object from model.fit().
        """
        if self.train_ds is None:
            self.train_ds, self.num_train_samples = self.create_dataset(
                self.train_csv_path,
                self.config['batch_size'],
                is_training=True
            )
        if self.val_ds is None:
            self.val_ds, self.num_val_samples = self.create_dataset(
                self.val_csv_path,
                self.config['batch_size'],
                is_training=False
            )

        with self.strategy.scope():
            model = self._build_model()
            model = self._compile_model(model)
        self.model = model

        class_weight = self.class_weights if (self.config.get('use_class_weights', False) and self.task_type == "mcc") else None
        epochs = self.config.get('epochs',5)
        fit_verbose = int(self.config.get("fit_verbose", self.config.get("verbose", 2)))
        if fit_verbose not in {0, 1, 2}:
            fit_verbose = 2

        print("[KerasFitTrainer] Starting training ...")
        training_error = None
        try:
            if self.use_mlflow:
                from .MLflowLogger import MLflowLogger, create_keras_callback

                print("[KerasFitTrainer] Initializing MLflow logger...")
                self.mlflow_logger = MLflowLogger(self.config_path)
                self.callbacks.append(create_keras_callback(self.mlflow_logger))
                self.mlflow_logger.log_params({
                    "batch_size": self.config["batch_size"],
                    "learning_rate": self.config["learning_rate"],
                    "epochs": self.config["epochs"],
                    "task_type": self.task_type,
                    "num_classes": self.num_classes,
                })

            if self.mlflow_logger is not None:
                from .LivePlotCallback import LivePlotCallback

                self.callbacks.append(
                    LivePlotCallback(
                        mlflow_logger=self.mlflow_logger,
                        metrics=('accuracy', 'val_accuracy'),
                        log_every_n_epochs=1
                    )
                )

            history = model.fit(
                self.train_ds,
                epochs=epochs,
                validation_data=self.val_ds,
                callbacks=self.callbacks,
                class_weight=class_weight,
                verbose=fit_verbose,
            )

            hist_path = os.path.join(self.config.get('tmp_dir', '/tmp'), 'train_history.csv')
            pd.DataFrame(history.history).to_csv(hist_path, index=False)
            self.train_history_csv_path = hist_path
            if self.mlflow_logger:
                self.mlflow_logger.log_artifact(hist_path)
                self.mlflow_logger.log_artifact(self.config_path)

            logits = model.predict(self.val_ds, verbose=0)
            val_df = self.valid_val_df.copy()
            val_df["logits"] = logits.tolist()
            val_csv = os.path.join(self.config.get("tmp_dir", "/tmp"), "val_logits.csv")
            val_df.to_csv(val_csv, index=False)
            self.val_logits_csv_path = val_csv
            if self.mlflow_logger:
                self.mlflow_logger.log_artifact(val_csv)

            if 'final_model_path' in self.config:
                model.save(self.config['final_model_path'])
                if self.mlflow_logger:
                    self.mlflow_logger.log_artifact(self.config['final_model_path'])

            return history
        except BaseException as exc:
            training_error = exc
            raise
        finally:
            if self.mlflow_logger and self.auto_end_mlflow_run:
                run_cleanup_steps(
                    [("MLflow run finalization", self.mlflow_logger.end_run)],
                    primary_error=training_error,
                )
