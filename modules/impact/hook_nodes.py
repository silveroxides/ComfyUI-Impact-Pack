import sys
from . import hooks
from . import defs


class SEGSOrderedFilterDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "target": (["area(=w*h)", "width", "height", "x1", "y1", "x2", "y2"], {"tooltip": "The attribute of the segment to use for ordering (e.g., area, width, y1 coordinate)."}),
                        "order": ("BOOLEAN", {"default": True, "label_on": "descending", "label_off": "ascending", "tooltip": "Sort order: 'descending' (True) or 'ascending' (False)."}),
                        "take_start": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1, "tooltip": "Index of the first segment to take from the ordered list (0-based)."}),
                        "take_count": ("INT", {"default": 1, "min": 0, "max": sys.maxsize, "step": 1, "tooltip": "Number of segments to take from the ordered list, starting at `take_start`."}),
                     },
                }

    RETURN_TYPES = ("DETAILER_HOOK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, target, order, take_start, take_count):
        hook = hooks.SEGSOrderedFilterDetailerHook(target, order, take_start, take_count)
        return (hook, )


class SEGSRangeFilterDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "target": (["area(=w*h)", "width", "height", "x1", "y1", "x2", "y2", "length_percent"], {"tooltip": "The attribute of the segment to filter by (e.g., area, width, confidence)."}),
                        "mode": ("BOOLEAN", {"default": True, "label_on": "inside", "label_off": "outside", "tooltip": "Filter mode: 'inside' (True) keeps segments within the range, 'outside' (False) keeps segments outside the range."}),
                        "min_value": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1, "tooltip": "Minimum value for the target attribute for a segment to be kept."}),
                        "max_value": ("INT", {"default": 67108864, "min": 0, "max": sys.maxsize, "step": 1, "tooltip": "Maximum value for the target attribute for a segment to be kept."}),
                     },
                }

    RETURN_TYPES = ("DETAILER_HOOK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, target, mode, min_value, max_value):
        hook = hooks.SEGSRangeFilterDetailerHook(target, mode, min_value, max_value)
        return (hook, )


class SEGSLabelFilterDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS", {"tooltip": "Input SEGS data (Note: this input is not directly used by the hook's logic but might be expected by the pipeline)."}),
                        "preset": (['all'] + defs.detection_labels, {"tooltip": "Select a preset label or 'all'. This primarily influences the UI and is not directly used if 'labels' input is provided."}),
                        "labels": ("STRING", {"multiline": True, "placeholder": "List the types of segments to be allowed, separated by commas", "tooltip": "Comma-separated list of labels to filter segments by. The hook will keep segments matching these labels."}),
                     },
                }

    RETURN_TYPES = ("DETAILER_HOOK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, preset, labels):
        hook = hooks.SEGSLabelFilterDetailerHook(labels)
        return (hook, )


class PreviewDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"quality": ("INT", {"default": 95, "min": 20, "max": 100, "tooltip": "Quality setting for the generated preview images (typically for WEBP format, 20-100)."})},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("DETAILER_HOOK", "UPSCALER_HOOK")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    NOT_IDEMPOTENT = True

    def doit(self, quality, unique_id):
        hook = hooks.PreviewDetailerHook(unique_id, quality)
        return hook, hook
