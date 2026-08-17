"""Minimal Big-LaMa generator runtime compatible with the official checkpoint."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np


class LaMaLargeInpainter:
    """Lazy CUDA runtime for the official 18-block Big-LaMa generator."""

    def __init__(self, checkpoint_path: str | Path, prefer_bf16: bool = True, max_side: int = 512) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.prefer_bf16 = prefer_bf16
        self.max_side = max(128, int(max_side))
        self._model: Any | None = None
        self._torch: Any | None = None
        self._use_bf16 = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"LaMa checkpoint not found: {self.checkpoint_path}")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("LaMa Large requires CUDA, but torch.cuda.is_available() is false")
        model = _build_big_lama_generator(torch)
        checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        state = checkpoint.get("gen_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state, strict=True)
        model.eval().to("cuda")
        self._torch = torch
        self._model = model
        self._use_bf16 = bool(
            self.prefer_bf16
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )

    def unload(self) -> None:
        model, torch = self._model, self._torch
        self._model = None
        self._torch = None
        self._use_bf16 = False
        if model is not None:
            del model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run LaMa and composite its prediction strictly inside the binary mask."""
        import cv2

        if not np.any(mask):
            return image.copy()
        self.load()
        assert self._model is not None and self._torch is not None
        torch = self._torch
        source_h, source_w = image.shape[:2]
        scale = min(1.0, self.max_side / max(source_h, source_w))
        resized_w = max(8, int(round(source_w * scale)))
        resized_h = max(8, int(round(source_h * scale)))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
        pad_h = (8 - resized_h % 8) % 8
        pad_w = (8 - resized_w % 8) % 8
        padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        padded_mask = cv2.copyMakeBorder(resized_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        image_t = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).float().cuda() / 255.0
        mask_t = torch.from_numpy((padded_mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        model_input = torch.cat((image_t * (1.0 - mask_t), mask_t), dim=1)

        def _forward(use_bf16: bool):
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                return self._model(model_input)

        try:
            prediction = _forward(self._use_bf16)
        except RuntimeError:
            if not self._use_bf16:
                raise
            self._use_bf16 = False
            prediction = _forward(False)
        predicted = prediction[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        predicted = (predicted[:resized_h, :resized_w] * 255.0 + 0.5).astype(np.uint8)
        if (resized_h, resized_w) != (source_h, source_w):
            predicted = cv2.resize(predicted, (source_w, source_h), interpolation=cv2.INTER_CUBIC)
        # Big-LaMa can occasionally produce a strong global color cast for very
        # thin glyph-shaped masks. Calibrate only that cast to the immediate source
        # context while preserving the model's spatial reconstruction/texture.
        selected = mask > 0
        ring = (cv2.dilate(selected.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0) & ~selected
        if np.count_nonzero(ring) >= 24 and np.count_nonzero(selected) > 0:
            context_median = np.median(image[ring].astype(np.float32), axis=0)
            prediction_median = np.median(predicted[selected].astype(np.float32), axis=0)
            color_shift = context_median - prediction_median
            if float(np.linalg.norm(color_shift)) > 35.0:
                corrected = predicted[selected].astype(np.float32) + color_shift
                predicted[selected] = np.clip(corrected, 0, 255).astype(np.uint8)
        result = image.copy()
        result[selected] = predicted[selected]
        return result

    def inpaint_batch(
        self,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        batch_size: int = 24,
    ) -> list[np.ndarray]:
        """Run GPU batched Big-LaMa inference across multiple crops with elastic batch recovery."""
        import cv2

        if not images:
            return []
        if len(images) == 1:
            return [self.inpaint(images[0], masks[0])]

        self.load()
        assert self._model is not None and self._torch is not None
        torch = self._torch

        pairs = list(zip(images, masks))

        def _forward_chunk(chunk_pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
            chunk_imgs = [p[0] for p in chunk_pairs]
            chunk_masks = [p[1] for p in chunk_pairs]
            chunk_results: list[np.ndarray] = []

            try:
                # Check if any item in chunk has an empty mask
                scaled_info = []
                for img, mask in zip(chunk_imgs, chunk_masks):
                    sh, sw = img.shape[:2]
                    scale = min(1.0, self.max_side / max(sh, sw))
                    rw = max(8, int(round(sw * scale)))
                    rh = max(8, int(round(sh * scale)))
                    scaled_info.append((sh, sw, rh, rw, scale))

                max_h = max(info[2] for info in scaled_info)
                max_w = max(info[3] for info in scaled_info)
                target_h = ((max_h + 7) // 8) * 8
                target_w = ((max_w + 7) // 8) * 8

                batch_img_t = []
                batch_mask_t = []

                for (img, mask), (sh, sw, rh, rw, scale) in zip(zip(chunk_imgs, chunk_masks), scaled_info):
                    resized = cv2.resize(
                        img, (rw, rh),
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
                    )
                    resized_mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
                    pad_h = target_h - rh
                    pad_w = target_w - rw
                    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                    padded_mask = cv2.copyMakeBorder(resized_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

                    img_t = torch.from_numpy(padded.transpose(2, 0, 1)).float() / 255.0
                    mask_t = torch.from_numpy((padded_mask > 0).astype(np.float32)).unsqueeze(0)
                    batch_img_t.append(img_t)
                    batch_mask_t.append(mask_t)

                stacked_imgs = torch.stack(batch_img_t).cuda()
                stacked_masks = torch.stack(batch_mask_t).cuda()
                model_input = torch.cat((stacked_imgs * (1.0 - stacked_masks), stacked_masks), dim=1)

                def _forward_batch(use_bf16: bool):
                    with torch.inference_mode(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
                    ):
                        return self._model(model_input)

                try:
                    predictions = _forward_batch(self._use_bf16)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise
                    if not self._use_bf16:
                        raise
                    self._use_bf16 = False
                    predictions = _forward_batch(False)

                preds_np = predictions.float().clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
                if not isinstance(preds_np, np.ndarray):
                    raise RuntimeError("Predictions tensor was not converted to numpy array")

                for k, ((img, mask), (sh, sw, rh, rw, _)) in enumerate(zip(zip(chunk_imgs, chunk_masks), scaled_info)):

                    if not np.any(mask):
                        chunk_results.append(img.copy())
                        continue

                    pred = (preds_np[k, :rh, :rw] * 255.0 + 0.5).astype(np.uint8)
                    if (rh, rw) != (sh, sw):
                        pred = cv2.resize(pred, (sw, sh), interpolation=cv2.INTER_CUBIC)

                    selected = mask > 0
                    ring = (cv2.dilate(selected.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0) & ~selected
                    if np.count_nonzero(ring) >= 24 and np.count_nonzero(selected) > 0:
                        context_median = np.median(img[ring].astype(np.float32), axis=0)
                        prediction_median = np.median(pred[selected].astype(np.float32), axis=0)
                        color_shift = context_median - prediction_median
                        if float(np.linalg.norm(color_shift)) > 35.0:
                            corrected = pred[selected].astype(np.float32) + color_shift
                            pred[selected] = np.clip(corrected, 0, 255).astype(np.uint8)
                    res = img.copy()
                    res[selected] = pred[selected]
                    chunk_results.append(res)

            except Exception as exc:
                if "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError":
                    raise
                # Fallback to single inpaint for non-OOM errors / mock execution
                chunk_results = [self.inpaint(img, mask) for img, mask in zip(chunk_imgs, chunk_masks)]

            return chunk_results

        if not hasattr(self, "_batcher") or self._batcher is None:
            from core.system.adaptive_batcher import ElasticAdaptiveBatcher
            self._batcher = ElasticAdaptiveBatcher(default_batch_size=batch_size, min_batch_size=1, vram_ceiling=0.95)

        return self._batcher.execute(pairs, _forward_chunk, batch_size=batch_size)


def _build_big_lama_generator(torch):
    nn = torch.nn
    F = torch.nn.functional

    class FourierUnit(nn.Module):
        def __init__(self, in_channels, out_channels, groups=1):
            super().__init__()
            self.groups = groups
            self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, 1, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_channels * 2)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            batch, _, height, width = x.shape
            ffted = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho")
            ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
            ffted = ffted.view(batch, -1, height, ffted.shape[-1])
            ffted = self.relu(self.bn(self.conv_layer(ffted)))
            ffted = ffted.view(batch, -1, 2, height, ffted.shape[-1])
            ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])
            return torch.fft.irfftn(ffted, s=(height, width), dim=(-2, -1), norm="ortho")

    class SpectralTransform(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=False):
            super().__init__()
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity()
            self.enable_lfu = enable_lfu
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, 1, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True),
            )
            self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
            if enable_lfu:
                self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
            self.conv2 = nn.Conv2d(out_channels // 2, out_channels, 1, groups=groups, bias=False)

        def forward(self, x):
            x = self.downsample(x)
            x = self.conv1(x)
            output = self.fu(x)
            if self.enable_lfu:
                n, c, h, w = x.shape
                split = 2
                local = x[:, : c // 4]
                local = torch.cat(torch.split(local, h // split, dim=-2), dim=1).contiguous()
                local = torch.cat(torch.split(local, w // split, dim=-1), dim=1).contiguous()
                local = self.lfu(local)
                local = local.repeat(1, 1, split, split).contiguous()
            else:
                local = 0
            return self.conv2(x + output + local)

    class FFC(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                     stride=1, padding=0, dilation=1, groups=1, bias=False, enable_lfu=False,
                     padding_type="reflect", gated=False):
            super().__init__()
            assert stride in (1, 2)
            in_cg = int(in_channels * ratio_gin)
            in_cl = in_channels - in_cg
            out_cg = int(out_channels * ratio_gout)
            out_cl = out_channels - out_cg
            conv_kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation, groups=groups, bias=bias, padding_mode=padding_type)
            self.ratio_gin = ratio_gin
            self.ratio_gout = ratio_gout
            self.global_in_num = in_cg
            module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
            self.convl2l = module(in_cl, out_cl, **conv_kwargs) if module is nn.Conv2d else module()
            module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
            self.convl2g = module(in_cl, out_cg, **conv_kwargs) if module is nn.Conv2d else module()
            module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
            self.convg2l = module(in_cg, out_cl, **conv_kwargs) if module is nn.Conv2d else module()
            module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
            self.convg2g = module(in_cg, out_cg, stride, groups, enable_lfu) if module is SpectralTransform else module()
            self.gated = gated
            self.gate = nn.Conv2d(in_channels, 2, 1) if gated and in_cg > 0 and out_cl > 0 else nn.Identity()

        def forward(self, x):
            x_l, x_g = x if isinstance(x, tuple) else (x, 0)
            out_xl = out_xg = 0
            if self.gated:
                gates = torch.sigmoid(self.gate(torch.cat((x_l, x_g), dim=1)))
                g2l_gate, l2g_gate = gates[:, :1], gates[:, 1:]
            else:
                g2l_gate = l2g_gate = 1
            if self.ratio_gout != 1:
                out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
            if self.ratio_gout != 0:
                out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
            return out_xl, out_xg

    class FFC_BN_ACT(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                     stride=1, padding=0, dilation=1, groups=1, bias=False,
                     norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                     padding_type="reflect", enable_lfu=False, **kwargs):
            super().__init__()
            self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                           stride, padding, dilation, groups, bias, enable_lfu, padding_type)
            global_channels = int(out_channels * ratio_gout)
            self.bn_l = nn.Identity() if ratio_gout == 1 else norm_layer(out_channels - global_channels)
            self.bn_g = nn.Identity() if ratio_gout == 0 else norm_layer(global_channels)
            self.act_l = nn.Identity() if ratio_gout == 1 else activation_layer(inplace=True)
            self.act_g = nn.Identity() if ratio_gout == 0 else activation_layer(inplace=True)

        def forward(self, x):
            x_l, x_g = self.ffc(x)
            return self.act_l(self.bn_l(x_l)), self.act_g(self.bn_g(x_g))

    class FFCResnetBlock(nn.Module):
        def __init__(self, dim, ratio_gin=0.75, ratio_gout=0.75, dilation=1, **kwargs):
            super().__init__()
            self.conv1 = FFC_BN_ACT(dim, dim, 3, ratio_gin, ratio_gout, padding=dilation,
                                    dilation=dilation, activation_layer=nn.ReLU, **kwargs)
            self.conv2 = FFC_BN_ACT(dim, dim, 3, ratio_gin, ratio_gout, padding=dilation,
                                    dilation=dilation, activation_layer=nn.Identity, **kwargs)

        def forward(self, x):
            x_l, x_g = x
            y_l, y_g = self.conv1((x_l, x_g))
            y_l, y_g = self.conv2((y_l, y_g))
            return x_l + y_l, x_g + y_g

    class ConcatTupleLayer(nn.Module):
        def forward(self, x):
            return torch.cat(x, dim=1)

    layers = [nn.ReflectionPad2d(3), FFC_BN_ACT(4, 64, 7, 0, 0, padding=0, activation_layer=nn.ReLU)]
    for index in range(3):
        ratio_in = 0.0
        ratio_out = 0.75 if index == 2 else 0.0
        layers.append(FFC_BN_ACT(64 * (2 ** index), 64 * (2 ** (index + 1)), 3,
                                 ratio_in, ratio_out, stride=2, padding=1, activation_layer=nn.ReLU))
    for _ in range(18):
        layers.append(FFCResnetBlock(512, ratio_gin=0.75, ratio_gout=0.75))
    layers.append(ConcatTupleLayer())
    for index in range(3):
        multiplier = 2 ** (3 - index)
        layers.extend([
            nn.ConvTranspose2d(64 * multiplier, int(64 * multiplier / 2), 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(int(64 * multiplier / 2)),
            nn.ReLU(True),
        ])
    layers.extend([nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7, padding=0), nn.Sigmoid()])

    class FFCResNetGenerator(nn.Module):
        def __init__(self, model_layers):
            super().__init__()
            self.model = nn.Sequential(*model_layers)

        def forward(self, x):
            return self.model(x)

    return FFCResNetGenerator(layers)
