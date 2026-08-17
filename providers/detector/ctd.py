"""ComicTextDetector (CTD) provider.

dmMaze/comic-text-detector tabanlı manga/comic text detector.
ONNX model ile çalışır (OpenCV DNN backend).

Pipeline:
1. Preprocessing (letterbox, normalize)
2. ONNX inference (3 outputs: YOLO blocks, DBNet lines, segmentation mask)
3. YOLO block postprocessing (NMS)
4. DBNet text-line extraction
5. Segmentation mask processing
6. Grouping
7. Canonical Detection output

Lisans: GPL-3.0
Model kaynağı: https://github.com/zyddnys/manga-image-translator/releases
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from loguru import logger

from core.detection import BBox, Detection, RegionType
from core.system.adaptive_batcher import ElasticAdaptiveBatcher, get_batch_config
from core.system.cuda_init import init_cuda_runtime
from providers.detector.base import DetectorProvider

# Initialize CUDA & cuDNN DLL search paths before loading ONNX Runtime
init_cuda_runtime()

if TYPE_CHECKING:
    import numpy as np


def _native_modules() -> tuple[Any, Any]:
    """Load OpenCV/NumPy only when CTD runtime work is requested."""
    import cv2
    import numpy as np

    return cv2, np


def _make_dynamic_batch_onnx(model_path: Path | str) -> bytes:
    """CTD ONNX modelini dinamik batch [B, 3, 1024, 1024] desteğiyle dönüştürür."""
    import onnx
    from onnx import helper, numpy_helper
    _, np = _native_modules()

    model = onnx.load(str(model_path))

    # 1. Giriş ve çıkış tensörlerinin 0. boyutunu 'batch' yap
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    for out in model.graph.output:
        out.type.tensor_type.shape.dim[0].dim_param = "batch"

    # 2. Shape(images) -> Gather(0) düğümleri ile dinamik batch boyutunu al
    shape_node = helper.make_node("Shape", inputs=["images"], outputs=["img_shape"], name="DynamicBatch_Shape")
    gather_node = helper.make_node(
        "Gather",
        inputs=["img_shape", "dynamic_batch_idx"],
        outputs=["dyn_batch"],
        name="DynamicBatch_Gather",
        axis=0,
    )

    idx_init = numpy_helper.from_array(np.array(0, dtype=np.int64), name="dynamic_batch_idx")
    model.graph.initializer.append(idx_init)

    new_nodes = [shape_node, gather_node]

    # YOLO head reshape sabitleri
    shapes_to_replace = {
        "1317": np.array([3, 7, 128, 128], dtype=np.int64),
        "1356": np.array([3, 7, 64, 64], dtype=np.int64),
        "1395": np.array([3, 7, 32, 32], dtype=np.int64),
        "1350": np.array([-1, 7], dtype=np.int64),
    }

    for init_name, rem_shape in shapes_to_replace.items():
        rem_name = f"rem_{init_name}"
        rem_init = numpy_helper.from_array(rem_shape, name=rem_name)
        model.graph.initializer.append(rem_init)

        unsqueezed_dyn = f"unsqueezed_dyn_{init_name}"
        unsq_node = helper.make_node("Unsqueeze", inputs=["dyn_batch"], outputs=[unsqueezed_dyn], name=f"DynamicUnsq_{init_name}", axes=[0])
        concat_node = helper.make_node("Concat", inputs=[unsqueezed_dyn, rem_name], outputs=[f"dynamic_shape_{init_name}"], name=f"DynamicConcat_{init_name}", axis=0)
        new_nodes.extend([unsq_node, concat_node])

    # Reshape düğümlerinin girdi şeklini dinamik tensörle değiştir
    for node in model.graph.node:
        if node.op_type == "Reshape":
            if len(node.input) >= 2 and node.input[1] in shapes_to_replace:
                orig = node.input[1]
                node.input[1] = f"dynamic_shape_{orig}"

    # Yeni düğümleri grafiğin başına ekle
    for n in reversed(new_nodes):
        model.graph.node.insert(0, n)

    return model.SerializeToString()

MODEL_FILENAME = "comictextdetector.pt.onnx"
MODEL_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"
MODEL_SHA256 = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"


class ComicTextDetector(DetectorProvider):
    """ComicTextDetector provider (Optimized GPU Tile Batching ONNXRuntime / OpenCV DNN backend)."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        input_size: int = 1024,
        tile_batch_size: int | None = None,
    ) -> None:
        logger.debug(f"[THREAD] ComicTextDetector.__init__ thread id: {threading.get_ident()}")
        self._model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent.parent.parent / "models" / "detectors" / "ctd"
        self._model_path = self._model_dir / MODEL_FILENAME
        self._loaded = False
        self._device = "cpu"
        self._net = None
        self._ort_sessions: list[Any] = []
        self._ort_input_name: str = ""
        self._ort_output_names: list[str] = []
        self._input_size = input_size
        self._tile_batch_size = tile_batch_size if tile_batch_size is not None else get_batch_config().detector_tile_batch
        self._adaptive_batcher: ElasticAdaptiveBatcher | None = None
        self._conf_thresh = 0.4
        self._nms_thresh = 0.35
        self._seg_thresh = 0.3
        self._box_thresh = 0.4
        self._debug = False
        self._debug_dir = None
        self.last_output_metadata: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "ComicTextDetector"

    @property
    def version(self) -> str:
        return "beta-0.3"

    @property
    def cache_schema_version(self) -> str:
        """Invalidate pre-geometry cache entries after compact CTD metadata was added."""
        return "ctd-geometry-v2-upstream"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def session(self) -> Any:
        """First active ONNXRuntime session in pool (if loaded)."""
        return self._ort_sessions[0] if self._ort_sessions else None

    @property
    def device(self) -> str:
        return self._device

    @property
    def tile_batch_size(self) -> int:
        return self._tile_batch_size

    @tile_batch_size.setter
    def tile_batch_size(self, val: int) -> None:
        self._tile_batch_size = max(1, int(val))

    def set_debug(self, enabled: bool, output_dir: str | Path | None = None) -> None:
        """Debug modunu ayarlar."""
        self._debug = enabled
        if enabled and output_dir is not None:
            self._debug_dir = Path(output_dir)

    @staticmethod
    def _is_blank_window(img: np.ndarray) -> bool:
        """Hızlı piksel varyans/standart sapma kontrolü ile boş şerit/arka planları atlar."""
        if img is None or img.size == 0:
            return True
        sample = img[::8, ::8]
        if sample.size == 0:
            return True
        min_v = int(sample.min())
        max_v = int(sample.max())
        if max_v - min_v < 8:
            return True
        return float(sample.std()) < 3.0

    def load(self) -> None:
        logger.debug(f"[THREAD] ComicTextDetector.load thread id: {threading.get_ident()}")
        if self._loaded:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"CTD ONNX model not found: {self._model_path}\n"
                f"Download from: {MODEL_URL}\n"
                f"Or run: python scripts/download_detector_models.py --detector ctd"
            )

        # 1. Try accelerated ONNXRuntime with dynamic batch surgery
        try:
            import onnxruntime as ort
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 4
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            available = ort.get_available_providers()
            providers_to_use: list[str | tuple[str, dict[str, Any]]] = []

            if "CUDAExecutionProvider" in available:
                cuda_opts = {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                }
                providers_to_use.append(("CUDAExecutionProvider", cuda_opts))
                self._device = "cuda"
            else:
                self._device = "cpu"

            providers_to_use.append("CPUExecutionProvider")

            model_to_load: str | bytes = str(self._model_path)
            try:
                model_to_load = _make_dynamic_batch_onnx(self._model_path)
                logger.debug("CTD dynamic batch surgery applied to ONNX model.")
            except Exception as dyn_err:
                logger.warning(f"Could not apply dynamic batch surgery to CTD model: {dyn_err}, using static model.")
                model_to_load = str(self._model_path)

            num_workers = 3 if self._device == "cpu" else 1
            self._ort_sessions = [
                ort.InferenceSession(model_to_load, session_options, providers=providers_to_use)
                for _ in range(num_workers)
            ]
            self._ort_input_name = self._ort_sessions[0].get_inputs()[0].name
            self._ort_input_type = self._ort_sessions[0].get_inputs()[0].type
            self._ort_output_names = [o.name for o in self._ort_sessions[0].get_outputs()]
            active_p = self._ort_sessions[0].get_providers()

            if "CUDAExecutionProvider" in available and "CUDAExecutionProvider" not in active_p:
                logger.warning(
                    "CUDAExecutionProvider was available in ORT but session fell back to: {}. "
                    "Verify CUDA 12 and cuDNN DLL dependencies.",
                    active_p,
                )
                self._device = "cpu"
            else:
                self._device = "cuda" if "CUDAExecutionProvider" in active_p else "cpu"

            self._loaded = True
            logger.info(
                "CTD ONNXRuntime model pool ({} workers, device: {}, providers: {}) loaded successfully",
                num_workers,
                self._device,
                active_p,
            )
            return
        except Exception as ort_err:
            logger.warning("ONNXRuntime initialization failed, falling back to OpenCV DNN: {}", ort_err)
            self._ort_sessions = []

        # 2. Fallback to OpenCV DNN
        cv2, _ = _native_modules()
        logger.info(f"Loading CTD ONNX model via OpenCV DNN: {self._model_path}")
        self._net = cv2.dnn.readNetFromONNX(str(self._model_path))
        self._uoln = self._net.getUnconnectedOutLayersNames()
        self._loaded = True
        logger.info("CTD ONNX model loaded successfully via OpenCV DNN")

    def unload(self) -> None:
        if not self._loaded:
            return
        self._net = None
        self._ort_sessions = []
        self._loaded = False

    def _forward_raw(self, img_in: np.ndarray, worker_idx: int = 0) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        """Executes raw inference via ONNXRuntime pool or OpenCV DNN."""
        _, np = _native_modules()
        if self._ort_sessions:
            sess = self._ort_sessions[worker_idx % len(self._ort_sessions)]
            if hasattr(self, "_ort_input_type") and "float16" in self._ort_input_type:
                img_in_typed = img_in.astype(np.float16)
            else:
                img_in_typed = img_in.astype(np.float32) if img_in.dtype != np.float32 else img_in
            outputs = sess.run(None, {self._ort_input_name: img_in_typed})
            output_names = self._ort_output_names
            output_dict = dict(zip(output_names, outputs))
            return outputs, output_dict
        else:
            self._net.setInput(img_in)
            outputs = self._net.forward(self._uoln)
            output_names = self._uoln
            output_dict = dict(zip(output_names, outputs))
            return outputs, output_dict

    def detect(self, image, window_id: int) -> Sequence[Detection]:
        logger.debug(f"[THREAD] ComicTextDetector.detect thread id: {threading.get_ident()}")
        if not self._loaded:
            raise RuntimeError("CTD model not loaded; call load() first")

        cv2, np = _native_modules()
        img = np.array(image)
        if self._is_blank_window(img):
            logger.debug(f"[CTD] Window {window_id} skipped as blank background gutter.")
            return []

        return self._detect_single_array(img, window_id, worker_idx=0)

    def detect_batch(self, items: Sequence[tuple[Any, int]]) -> Sequence[Sequence[Detection]]:
        """Optimized GPU Tile Batching with ElasticAdaptiveBatcher & dynamic fallback."""
        if not self._loaded:
            raise RuntimeError("CTD model not loaded; call load() first")

        if not items:
            return []

        cv2, np = _native_modules()
        results: list[Sequence[Detection]] = [[] for _ in range(len(items))]
        active_jobs: list[tuple[int, np.ndarray, int]] = []  # (original_idx, img_arr, window_id)

        for idx, (img, wid) in enumerate(items):
            arr = np.array(img)
            if self._is_blank_window(arr):
                results[idx] = []
            else:
                active_jobs.append((idx, arr, wid))

        if not active_jobs:
            return results

        # If fallback to OpenCV DNN
        if not self._ort_sessions or self._net is not None:
            for orig_idx, img_arr, wid in active_jobs:
                results[orig_idx] = self._detect_single_array(img_arr, wid, worker_idx=0)
            return results

        if self._adaptive_batcher is None:
            self._adaptive_batcher = ElasticAdaptiveBatcher(
                default_batch_size=self._tile_batch_size,
                min_batch_size=1,
            )

        def _process_chunk(chunk: Sequence[tuple[int, np.ndarray, int]]) -> list[tuple[int, Sequence[Detection]]]:
            sub_jobs = [(job[1], job[2]) for job in chunk]
            chunk_results = self._detect_batch_chunk(sub_jobs)
            return [(chunk[i][0], chunk_results[i]) for i in range(len(chunk))]

        executed_batches = self._adaptive_batcher.execute(
            active_jobs,
            _process_chunk,
            batch_size=self._tile_batch_size,
        )

        for orig_idx, dets in executed_batches:
            results[orig_idx] = dets

        return results

    def _preprocess_batch(self, images: Sequence[np.ndarray]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Vektörize letterbox, normalizasyon ve [B, 3, 1024, 1024] tensör paketleme."""
        cv2, np = _native_modules()
        tensors = []
        meta_list = []

        for img in images:
            im_h, im_w = img.shape[:2]
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 and img.shape[2] == 3 else img
            img_in, ratio, (dw, dh) = self._letterbox(
                img_bgr, new_shape=(self._input_size, self._input_size), auto=False, stride=64
            )
            img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img_in = np.ascontiguousarray(img_in, dtype=np.float32) / 255.0
            tensors.append(img_in)

            scale_x = im_w / (self._input_size - dw)
            scale_y = im_h / (self._input_size - dh)
            meta_list.append({
                "im_w": im_w,
                "im_h": im_h,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "dw": dw,
                "dh": dh,
            })

        batch_tensor = np.stack(tensors, axis=0)  # [B, 3, 1024, 1024]
        return batch_tensor, meta_list

    def _detect_batch_chunk(self, chunk: Sequence[tuple[np.ndarray, int]]) -> list[Sequence[Detection]]:
        """Toplu (batched) forward pass ve tespit koordinat dönüşümü."""
        if not chunk:
            return []

        images = [item[0] for item in chunk]
        window_ids = [item[1] for item in chunk]
        batch_size = len(images)

        batch_tensor, meta_list = self._preprocess_batch(images)
        outputs, _ = self._forward_raw(batch_tensor, worker_idx=0)

        blk_output = next((value for value in outputs if value.ndim == 3 and value.shape[-1] == 7), None)
        dense = [value for value in outputs if value.ndim == 4]
        det_output = next((value for value in dense if value.shape[1] == 2), None)
        seg_output = next((value for value in dense if value.shape[1] == 1), None)

        def _process_tile(b: int) -> list[Detection]:
            wid = window_ids[b]
            meta = meta_list[b]
            im_w = meta["im_w"]
            im_h = meta["im_h"]
            scale_x = meta["scale_x"]
            scale_y = meta["scale_y"]
            dw = meta["dw"]
            dh = meta["dh"]

            blocks = []
            if blk_output is not None:
                blk_b = blk_output[b]
                blocks = self._postprocess_yolo_blocks(blk_b, scale_x, scale_y, im_w, im_h)

            lines = []
            if det_output is not None:
                det_b = det_output[b : b + 1]
                lines = self._postprocess_dbnet_lines(det_b, im_w, im_h, dw, dh)

            seg_mask = None
            if seg_output is not None:
                seg_b = seg_output[b : b + 1]
                seg_mask = self._postprocess_segmentation_mask(seg_b, im_w, im_h, dw, dh)

            grouped_blocks = self._group_blocks_and_lines(blocks, lines, seg_mask, im_w, im_h)

            detections: list[Detection] = []
            for block_index, block in enumerate(grouped_blocks):
                x1, y1, x2, y2 = block["bbox"]
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                line_polygons = [
                    line["polygon"].astype(int).tolist()
                    for line in block.get("lines", [])
                    if not line.get("synthetic", False)
                    and line.get("polygon") is not None
                    and len(line["polygon"]) >= 3
                ]
                metadata: dict[str, Any] = {
                    "geometry_source": "ctd",
                    "ctd_block_id": f"w{wid}:b{block_index}",
                    "ctd_block_bbox": [ix1, iy1, ix2, iy2],
                    "line_polygons": line_polygons,
                    "line_scores": [
                        float(line.get("score", 0.0))
                        for line in block.get("lines", [])
                        if not line.get("synthetic", False)
                    ],
                }
                segmentation_polygons = self._compact_segmentation_polygons(seg_mask, (ix1, iy1, ix2, iy2))
                if segmentation_polygons:
                    metadata["segmentation_polygons"] = segmentation_polygons

                detections.append(
                    Detection(
                        bbox=BBox(x1=ix1, y1=iy1, x2=ix2, y2=iy2),
                        confidence=float(block.get("confidence", 0.5)),
                        type=RegionType.UNKNOWN,
                        source_window_id=wid,
                        metadata=metadata,
                    )
                )
            return detections

        if batch_size == 1:
            chunk_detections = [_process_tile(0)]
        else:
            import concurrent.futures
            max_workers = min(8, batch_size)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                chunk_detections = list(executor.map(_process_tile, range(batch_size)))

        return chunk_detections

    def _detect_single_array(self, img: np.ndarray, window_id: int, worker_idx: int = 0) -> Sequence[Detection]:
        cv2, np = _native_modules()
        im_h, im_w = img.shape[:2]

        # Preprocess
        img_in, ratio, dw, dh = self._preprocess(img)
        scale_x = im_w / (self._input_size - dw)
        scale_y = im_h / (self._input_size - dh)

        if self._debug:
            self._save_debug("01_input.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            lb_vis = (img_in[0].transpose((1, 2, 0)) * 255).astype(np.uint8)
            self._save_debug("02_letterboxed.png", cv2.cvtColor(lb_vis, cv2.COLOR_RGB2BGR))

        # Inference via backend
        outputs, output_dict = self._forward_raw(img_in, worker_idx=worker_idx)

        blk_output = next((value for value in outputs if value.ndim == 3 and value.shape[-1] == 7), None)
        dense = [value for value in outputs if value.ndim == 4]
        det_output = next((value for value in dense if value.shape[1] == 2), None)
        seg_output = next((value for value in dense if value.shape[1] == 1), None)

        if self._debug:
            self._log_output_info("blk", blk_output)
            self._log_output_info("det", det_output)
            self._log_output_info("seg", seg_output)

        # YOLO block postprocessing
        blocks = []
        if blk_output is not None:
            blocks = self._postprocess_yolo_blocks(blk_output, scale_x, scale_y, im_w, im_h)

        # DBNet text-line extraction
        lines = []
        if det_output is not None:
            lines = self._postprocess_dbnet_lines(det_output, im_w, im_h, dw, dh)

        # Segmentation mask
        seg_mask = None
        if seg_output is not None:
            seg_mask = self._postprocess_segmentation_mask(seg_output, im_w, im_h, dw, dh)
            if self._debug:
                mask_vis = (seg_mask * 255).astype(np.uint8) if seg_mask.max() <= 1.0 else seg_mask.astype(np.uint8)
                cv2.imwrite(str(self._debug_dir / "05_segmentation_mask.png"), mask_vis)

        # Grouping
        grouped_blocks = self._group_blocks_and_lines(blocks, lines, seg_mask, im_w, im_h)

        if self._debug:
            self._save_grouped_visualization(img, grouped_blocks, im_w, im_h)

        # Convert to Detection
        detections: list[Detection] = []
        for block_index, block in enumerate(grouped_blocks):
            x1, y1, x2, y2 = block["bbox"]
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            line_polygons = [
                line["polygon"].astype(int).tolist()
                for line in block.get("lines", [])
                if not line.get("synthetic", False)
                and line.get("polygon") is not None and len(line["polygon"]) >= 3
            ]
            metadata: dict[str, Any] = {
                "geometry_source": "ctd",
                "ctd_block_id": f"w{window_id}:b{block_index}",
                "ctd_block_bbox": [ix1, iy1, ix2, iy2],
                "line_polygons": line_polygons,
                "line_scores": [
                    float(line.get("score", 0.0))
                    for line in block.get("lines", [])
                    if not line.get("synthetic", False)
                ],
            }
            segmentation_polygons = self._compact_segmentation_polygons(seg_mask, (ix1, iy1, ix2, iy2))
            if segmentation_polygons:
                metadata["segmentation_polygons"] = segmentation_polygons

            detections.append(
                Detection(
                    bbox=BBox(x1=ix1, y1=iy1, x2=ix2, y2=iy2),
                    confidence=float(block.get("confidence", 0.5)),
                    type=RegionType.UNKNOWN,
                    source_window_id=window_id,
                    metadata=metadata,
                )
            )

        return detections

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, tuple[float, float], int, int]:
        """CTD-style letterbox preprocessing."""
        cv2, np = _native_modules()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_in, ratio, (dw, dh) = self._letterbox(img_bgr, new_shape=(self._input_size, self._input_size), auto=False, stride=64)
        img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_in = np.ascontiguousarray(img_in)
        img_in = img_in.astype(np.float32) / 255.0
        img_in = np.expand_dims(img_in, axis=0)
        return img_in, ratio, dw, dh

    def _letterbox(self, im, new_shape=(640, 640), color=(0, 0, 0), auto=False, scaleFill=False, scaleup=True, stride=128):
        """Letterbox resize with aspect ratio preservation."""
        cv2, np = _native_modules()
        shape = im.shape[:2]
        if not isinstance(new_shape, tuple):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dh, dw = int(dh), int(dw)
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        im = cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (dw, dh)

    def _postprocess_yolo_blocks(self, blk_output, scale_x, scale_y, im_w, im_h):
        """YOLO block output'tan NMS ile filtrelenmiş bbox'lar çıkarır."""
        cv2, np = _native_modules()
        blk_output = np.array(blk_output).reshape(-1, 7)
        if blk_output.size == 0:
            return []

        # Upstream YOLOv5 head format:
        # [center_x, center_y, width, height, objectness, class_0, class_1].
        class_scores = blk_output[:, 5:] * blk_output[:, 4:5]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        mask = scores > self._conf_thresh
        blk_output = blk_output[mask]
        class_ids = class_ids[mask]
        scores = scores[mask]

        if blk_output.shape[0] == 0:
            return []

        boxes_xywh = blk_output[:, :4].copy()
        boxes_xywh[:, 0] -= boxes_xywh[:, 2] / 2
        boxes_xywh[:, 1] -= boxes_xywh[:, 3] / 2

        # OpenCV NMSBoxes is class-agnostic, so run it independently per class
        # to match upstream YOLOv5 non_max_suppression semantics.
        kept_indices: list[int] = []
        for class_id in np.unique(class_ids):
            class_indices = np.flatnonzero(class_ids == class_id)
            indices = cv2.dnn.NMSBoxes(
                boxes_xywh[class_indices].tolist(),
                scores[class_indices].tolist(),
                self._conf_thresh,
                self._nms_thresh,
            )
            if len(indices) > 0:
                kept_indices.extend(class_indices[np.asarray(indices).reshape(-1)].tolist())

        if not kept_indices:
            return []

        kept_indices.sort(key=lambda index: float(scores[index]), reverse=True)
        boxes = boxes_xywh[kept_indices]
        scores = scores[kept_indices]

        # Convert [x1, y1, width, height] to xyxy and scale to original image.
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, im_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, im_h)

        result = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            result.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(score),
                "type": "block",
            })
        return result

    def _postprocess_dbnet_lines(self, det_output, im_w, im_h, dw=0, dh=0):
        """Upstream SegDetectorRepresenter-compatible DBNet box extraction."""
        cv2, np = _native_modules()
        det_output = np.array(det_output)
        if det_output.ndim != 4:
            return []
        shrink_map = det_output[0, 0].astype(np.float32)
        map_h, map_w = shrink_map.shape
        valid_w = max(1, map_w - int(dw))
        valid_h = max(1, map_h - int(dh))
        binary_map = (shrink_map > self._seg_thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for contour in contours[:1000]:
            points, short_side = self._mini_box(contour)
            if short_side < 2:
                continue
            points = np.asarray(points, np.float32)
            score = self._compute_polygon_score(contour.squeeze(1), shrink_map)
            if score < self._box_thresh:
                continue
            expanded = self._unclip_polygon(points, unclip_ratio=1.5)
            if expanded is None:
                continue
            expanded, short_side = self._mini_box(expanded.reshape(-1, 1, 2).astype(np.float32))
            if short_side < 2:
                continue
            expanded = np.asarray(expanded, np.float32)
            expanded[:, 0] = expanded[:, 0] / valid_w * im_w
            expanded[:, 1] = expanded[:, 1] / valid_h * im_h
            expanded[:, 0] = np.clip(expanded[:, 0], 0, im_w - 1)
            expanded[:, 1] = np.clip(expanded[:, 1], 0, im_h - 1)
            lines.append({"polygon": expanded.astype(int), "score": float(score)})
        return lines

    @staticmethod
    def _mini_box(contour):
        cv2, _ = _native_modules()
        box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(box)), key=lambda point: point[0])
        i1, i4 = (0, 1) if points[1][1] > points[0][1] else (1, 0)
        i2, i3 = (2, 3) if points[3][1] > points[2][1] else (3, 2)
        return [points[i1], points[i2], points[i3], points[i4]], min(box[1])

    def _compute_polygon_score(self, polygon, score_map):
        """Polygon içindeki ortalama skoru hesaplar."""
        cv2, np = _native_modules()
        polygon = np.asarray(polygon).reshape(-1, 2).copy()
        xmin, xmax = np.clip([np.floor(polygon[:, 0].min()), np.ceil(polygon[:, 0].max())], 0, score_map.shape[1] - 1).astype(int)
        ymin, ymax = np.clip([np.floor(polygon[:, 1].min()), np.ceil(polygon[:, 1].max())], 0, score_map.shape[0] - 1).astype(int)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
        polygon[:, 0] -= xmin
        polygon[:, 1] -= ymin
        cv2.fillPoly(mask, polygon.reshape(1, -1, 2).astype(np.int32), 1)
        return float(cv2.mean(score_map[ymin:ymax + 1, xmin:xmax + 1], mask)[0])

    def _unclip_polygon(self, polygon, unclip_ratio=1.5):
        """Polygon'u genişletir (unclip)."""
        _, np = _native_modules()
        import pyclipper
        from shapely.geometry import Polygon
        polygon = polygon.reshape(-1, 2)
        if len(polygon) < 3:
            return None
        poly = Polygon(polygon)
        if poly.length <= 0 or poly.area <= 0:
            return None
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(polygon.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(poly.area * unclip_ratio / poly.length)
        if len(expanded) != 1:
            return None
        return np.asarray(expanded[0], np.float32)

    def _postprocess_segmentation_mask(self, seg_output, im_w, im_h, dw=0, dh=0):
        """Segmentation mask'ı original image boyutuna getirir."""
        cv2, np = _native_modules()
        mask = seg_output[0, 0, :, :]
        # Remove padding
        if dh > 0:
            mask = mask[:-dh, :]
        if dw > 0:
            mask = mask[:, :-dw]
        # Threshold
        mask = (mask > self._seg_thresh).astype(np.uint8) * 255
        # Resize to original image size
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        return mask

    def _group_blocks_and_lines(self, blocks, lines, seg_mask, im_w, im_h):
        """YOLO block'ları ve DBNet line'ları gruplayarak canonical block'lar üretir."""
        _, np = _native_modules()
        if not blocks and not lines:
            return []

        if not blocks:
            # Create blocks from lines if no YOLO blocks
            for line in lines:
                x1, y1 = line["polygon"].min(axis=0)
                x2, y2 = line["polygon"].max(axis=0)
                blocks.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": line["score"],
                    "type": "block",
                    "lines": [line],
                })
            return blocks

        # Assign lines only to the original YOLO blocks.  DBNet-only blocks are
        # created after this pass so an early, oversized DBNet polygon cannot
        # steal later lines from a valid YOLO container.
        yolo_block_count = len(blocks)
        for block in blocks:
            block["lines"] = []

        bbox_score_thresh = 0.4
        mask_score_thresh = 0.1

        unmatched_lines = []
        for line in lines:
            bx1, by1, bx2, by2 = line["polygon"].min(axis=0)[0], line["polygon"].min(axis=0)[1], \
                                  line["polygon"].max(axis=0)[0], line["polygon"].max(axis=0)[1]
            line_area = (by2 - by1) * (bx2 - bx1)
            bbox_score = -1
            containment_score = -1
            bbox_idx = -1

            for jj, blk in enumerate(blocks[:yolo_block_count]):
                blk_bbox = blk["bbox"]
                intersection = self._union_area(blk_bbox, [bx1, by1, bx2, by2])
                line_coverage = intersection / (line_area + 1e-6)
                block_area = max(0.0, (blk_bbox[2] - blk_bbox[0]) * (blk_bbox[3] - blk_bbox[1]))
                block_coverage = intersection / (block_area + 1e-6)

                # Normal case: most of the DBNet line lies in the YOLO block.
                # Oversized-DBNet case: the line bbox strongly contains the
                # YOLO block, making the tighter YOLO geometry primary.
                score = line_coverage if line_coverage > bbox_score_thresh else -1
                contained = block_coverage if block_coverage >= 0.9 else -1
                if score > bbox_score or (
                    score == bbox_score and contained > containment_score
                ):
                    bbox_score = score
                    containment_score = contained
                    bbox_idx = jj
                elif bbox_idx < 0 and contained > containment_score:
                    containment_score = contained
                    bbox_idx = jj

            if bbox_idx >= 0 and (
                bbox_score > bbox_score_thresh or containment_score >= 0.9
            ):
                blocks[bbox_idx]["lines"].append(line)
            else:
                unmatched_lines.append((line, bx1, by1, bx2, by2))

        for line, bx1, by1, bx2, by2 in unmatched_lines:
            line_area = (by2 - by1) * (bx2 - bx1)
            dbnet_score = -1
            dbnet_idx = -1
            for jj in range(yolo_block_count, len(blocks)):
                score = self._union_area(
                    blocks[jj]["bbox"], [bx1, by1, bx2, by2]
                ) / (line_area + 1e-6)
                if score > dbnet_score:
                    dbnet_score = score
                    dbnet_idx = jj

            if dbnet_idx >= 0 and dbnet_score > bbox_score_thresh:
                blocks[dbnet_idx]["lines"].append(line)
                dx1, dy1, dx2, dy2 = blocks[dbnet_idx]["bbox"]
                blocks[dbnet_idx]["bbox"] = [
                    min(dx1, float(bx1)),
                    min(dy1, float(by1)),
                    max(dx2, float(bx2)),
                    max(dy2, float(by2)),
                ]
                continue

            if seg_mask is not None:
                mask_score = self._mask_mean(seg_mask, bx1, by1, bx2, by2)
                if mask_score >= mask_score_thresh:
                    blocks.append({
                        "bbox": [float(bx1), float(by1), float(bx2), float(by2)],
                        "confidence": line["score"],
                        "type": "block",
                        "lines": [line],
                    })

        # Filter blocks with no lines (use mask if available)
        filtered_blocks = []
        for block_idx, blk in enumerate(blocks):
            if len(blk.get("lines", [])) == 0:
                bx1, by1, bx2, by2 = blk["bbox"]
                if seg_mask is not None:
                    mask_score = self._mask_mean(seg_mask, bx1, by1, bx2, by2)
                    if mask_score < mask_score_thresh:
                        continue
                # Create synthetic line from bbox
                blk["lines"] = [{
                    "polygon": np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]),
                    "score": blk["confidence"],
                    "synthetic": True,
                }]
            elif block_idx < yolo_block_count:
                real_lines = [
                    line for line in blk["lines"]
                    if not line.get("synthetic", False)
                ]
                if real_lines:
                    lx1 = min(float(line["polygon"][:, 0].min()) for line in real_lines)
                    ly1 = min(float(line["polygon"][:, 1].min()) for line in real_lines)
                    lx2 = max(float(line["polygon"][:, 0].max()) for line in real_lines)
                    ly2 = max(float(line["polygon"][:, 1].max()) for line in real_lines)
                    bx1, by1, bx2, by2 = blk["bbox"]
                    ux1, uy1 = min(bx1, lx1), min(by1, ly1)
                    ux2, uy2 = max(bx2, lx2), max(by2, ly2)
                    block_area = max(1.0, (bx2 - bx1) * (by2 - by1))
                    union_bbox_area = max(0.0, (ux2 - ux1) * (uy2 - uy1))
                    # Permit modest DBNet completion of a clipped YOLO box, but
                    # keep YOLO primary when a coarse polygon would balloon it.
                    if union_bbox_area / block_area <= 2.0:
                        blk["bbox"] = [ux1, uy1, ux2, uy2]
            filtered_blocks.append(blk)

        return filtered_blocks

    @staticmethod
    def _mask_mean(seg_mask, x1, y1, x2, y2) -> float:
        height, width = seg_mask.shape[:2]
        ix1, iy1 = max(0, min(width, int(x1))), max(0, min(height, int(y1)))
        ix2, iy2 = max(0, min(width, int(x2))), max(0, min(height, int(y2)))
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return float(seg_mask[iy1:iy2, ix1:ix2].mean() / 255.0)

    def _compact_segmentation_polygons(self, seg_mask, bbox: tuple[int, int, int, int]) -> list[list[list[int]]]:
        """Store the relevant segmentation hint as simplified contours, never a page-sized array."""
        if seg_mask is None:
            return []
        cv2, np = _native_modules()
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(seg_mask.shape[1], x2), min(seg_mask.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return []
        crop = (seg_mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        compact: list[list[list[int]]] = []
        min_area = max(4.0, crop.size * 0.0005)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:32]:
            if cv2.contourArea(contour) < min_area:
                continue
            epsilon = max(1.0, 0.01 * cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(approx) >= 3:
                compact.append([[int(px + x1), int(py + y1)] for px, py in approx])
        return compact

    def _union_area(self, bbox_a, bbox_b):
        """İki bbox'ın kesişim alanını hesaplar."""
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])
        if y2 < y1 or x2 < x1:
            return 0
        return (y2 - y1) * (x2 - x1)

    def _save_debug(self, filename: str, image: np.ndarray) -> None:
        """Debug görseli kaydeder."""
        if self._debug_dir is None:
            return
        path = self._debug_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2, _ = _native_modules()
        cv2.imwrite(str(path), image)

    def _log_output_info(self, name: str, tensor) -> None:
        """ONNX output bilgisini loglar."""
        if tensor is None:
            logger.warning(f"Output '{name}' is None")
            return
        logger.info(
            f"Output '{name}': shape={tensor.shape}, dtype={tensor.dtype}, "
            f"min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}"
        )

    def _save_grouped_visualization(self, img: np.ndarray, blocks: list[dict], im_w: int, im_h: int) -> None:
        """Gruplanmış block'ları görselleştirir."""
        cv2, _ = _native_modules()
        vis = img.copy()
        for block in blocks:
            x1, y1, x2, y2 = map(int, block["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for line in block.get("lines", []):
                poly = line["polygon"].astype(int)
                cv2.polylines(vis, [poly], True, (0, 0, 255), 1)
        self._save_debug("08_final.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
