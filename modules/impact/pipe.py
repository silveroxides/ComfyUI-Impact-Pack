import folder_paths
import impact.wildcards
from impact.utils import any_typ


class ToDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "model": ("MODEL", {"tooltip": "The main model (e.g., SD1.5, SDXL base) for the detailer pipeline."}),
                     "clip": ("CLIP", {"tooltip": "The CLIP model associated with the main model."}),
                     "vae": ("VAE",),
                     "positive": ("CONDITIONING", {"tooltip": "Positive conditioning for the main model."}),
                     "negative": ("CONDITIONING", {"tooltip": "Negative conditioning for the main model."}),
                     "bbox_detector": ("BBOX_DETECTOR", {"tooltip": "Bounding box detector to be included in the pipe."}),
                     "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "Wildcard string/prompt to be associated with this detailer pipe."}),
                     "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
                     "Select to add Wildcard": (["Select the Wildcard to add to the text"], ),
                     },
                "optional": {
                    "sam_model_opt": ("SAM_MODEL", {"tooltip": "Optional SAM model for segmentation refinement."}),
                    "segm_detector_opt": ("SEGM_DETECTOR", {"tooltip": "Optional segmentation detector for more precise masking."}),
                    "detailer_hook": ("DETAILER_HOOK", {"tooltip": "Optional Detailer Hook for custom processing steps."}),
                }}

    RETURN_TYPES = ("DETAILER_PIPE", )
    RETURN_NAMES = ("detailer_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, *args, **kwargs):
        pipe = (kwargs['model'], kwargs['clip'], kwargs['vae'], kwargs['positive'], kwargs['negative'], kwargs['wildcard'], kwargs['bbox_detector'],
                kwargs.get('segm_detector_opt', None), kwargs.get('sam_model_opt', None), kwargs.get('detailer_hook', None),
                kwargs.get('refiner_model', None), kwargs.get('refiner_clip', None),
                kwargs.get('refiner_positive', None), kwargs.get('refiner_negative', None))
        return (pipe, )


class ToDetailerPipeSDXL(ToDetailerPipe):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "model": ("MODEL", {"tooltip": "The SDXL base model."}),
                     "clip": ("CLIP", {"tooltip": "The CLIP model associated with the base model."}),
                     "vae": ("VAE",),
                     "positive": ("CONDITIONING", {"tooltip": "Positive conditioning for the base model."}),
                     "negative": ("CONDITIONING", {"tooltip": "Negative conditioning for the base model."}),
                     "refiner_model": ("MODEL", {"tooltip": "The SDXL refiner model."}),
                     "refiner_clip": ("CLIP", {"tooltip": "The CLIP model associated with the refiner model."}),
                     "refiner_positive": ("CONDITIONING", {"tooltip": "Positive conditioning for the refiner model."}),
                     "refiner_negative": ("CONDITIONING", {"tooltip": "Negative conditioning for the refiner model."}),
                     "bbox_detector": ("BBOX_DETECTOR", {"tooltip": "Bounding box detector to be included in the SDXL pipe."}),
                     "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "Wildcard string/prompt for the SDXL pipe."}),
                     "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
                     "Select to add Wildcard": (["Select the Wildcard to add to the text"],),
                     },
                "optional": {
                    "sam_model_opt": ("SAM_MODEL", {"tooltip": "Optional SAM model for segmentation refinement."}),
                    "segm_detector_opt": ("SEGM_DETECTOR", {"tooltip": "Optional segmentation detector."}),
                    "detailer_hook": ("DETAILER_HOOK", {"tooltip": "Optional Detailer Hook."}),
                }}


class FromDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"detailer_pipe": ("DETAILER_PIPE", {"tooltip": "The detailer pipe to decompose."}), }, }
        return {"required": {"detailer_pipe": ("DETAILER_PIPE", {"tooltip": "The detailer pipe to decompose."}), }, }
        return {"required": {"detailer_pipe": ("DETAILER_PIPE", {"tooltip": "The SDXL detailer pipe to decompose."}), }, }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "BBOX_DETECTOR", "SAM_MODEL", "SEGM_DETECTOR", "DETAILER_HOOK")
    RETURN_NAMES = ("model", "clip", "vae", "positive", "negative", "bbox_detector", "sam_model_opt", "segm_detector_opt", "detailer_hook")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, detailer_pipe):
        model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, _, _, _, _ = detailer_pipe
        return model, clip, vae, positive, negative, bbox_detector, sam_model_opt, segm_detector_opt, detailer_hook


class FromDetailerPipe_v2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"detailer_pipe": ("DETAILER_PIPE",), }, }

    RETURN_TYPES = ("DETAILER_PIPE", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "BBOX_DETECTOR", "SAM_MODEL", "SEGM_DETECTOR", "DETAILER_HOOK")
    RETURN_NAMES = ("detailer_pipe", "model", "clip", "vae", "positive", "negative", "bbox_detector", "sam_model_opt", "segm_detector_opt", "detailer_hook")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, detailer_pipe):
        model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, _, _, _, _ = detailer_pipe
        return detailer_pipe, model, clip, vae, positive, negative, bbox_detector, sam_model_opt, segm_detector_opt, detailer_hook


class FromDetailerPipe_SDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"detailer_pipe": ("DETAILER_PIPE",), }, }

    RETURN_TYPES = ("DETAILER_PIPE", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "BBOX_DETECTOR", "SAM_MODEL", "SEGM_DETECTOR", "DETAILER_HOOK", "MODEL", "CLIP", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("detailer_pipe", "model", "clip", "vae", "positive", "negative", "bbox_detector", "sam_model_opt", "segm_detector_opt", "detailer_hook", "refiner_model", "refiner_clip", "refiner_positive", "refiner_negative")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, detailer_pipe):
        model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, refiner_model, refiner_clip, refiner_positive, refiner_negative = detailer_pipe
        return detailer_pipe, model, clip, vae, positive, negative, bbox_detector, sam_model_opt, segm_detector_opt, detailer_hook, refiner_model, refiner_clip, refiner_positive, refiner_negative


class AnyPipeToBasic:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_pipe": (any_typ, {"tooltip": "Any pipe-like structure (tuple) from which the first 5 elements (model, clip, vae, positive, negative) will be extracted as a BASIC_PIPE."})},
        }

    RETURN_TYPES = ("BASIC_PIPE", )
    RETURN_NAMES = ("basic_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, any_pipe):
        return (any_pipe[:5], )


class ToBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "model": ("MODEL", {"tooltip": "The model component."}),
                     "clip": ("CLIP", {"tooltip": "The CLIP model component."}),
                     "vae": ("VAE",),
                     "positive": ("CONDITIONING", {"tooltip": "The positive conditioning component."}),
                     "negative": ("CONDITIONING", {"tooltip": "The negative conditioning component."}),
                     },
                }

    RETURN_TYPES = ("BASIC_PIPE", )
    RETURN_NAMES = ("basic_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, model, clip, vae, positive, negative):
        pipe = (model, clip, vae, positive, negative)
        return (pipe, )


class FromBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"basic_pipe": ("BASIC_PIPE", {"tooltip": "The basic pipe (model, clip, vae, positive, negative) to decompose."}), }, }
        return {"required": {"basic_pipe": ("BASIC_PIPE", {"tooltip": "The basic pipe to pass through and also decompose."}), }, }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "clip", "vae", "positive", "negative")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, basic_pipe):
        model, clip, vae, positive, negative = basic_pipe
        return model, clip, vae, positive, negative


class FromBasicPipe_v2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"basic_pipe": ("BASIC_PIPE",), }, }

    RETURN_TYPES = ("BASIC_PIPE", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("basic_pipe", "model", "clip", "vae", "positive", "negative")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, basic_pipe):
        model, clip, vae, positive, negative = basic_pipe
        return basic_pipe, model, clip, vae, positive, negative


class BasicPipeToDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"basic_pipe": ("BASIC_PIPE", {"tooltip": "The basic pipe (model, clip, vae, positive, negative) to convert to a detailer pipe."}),
                             "bbox_detector": ("BBOX_DETECTOR", {"tooltip": "Bounding box detector to include in the new detailer pipe."}),
                             "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "Wildcard string/prompt for the new detailer pipe."}),
                             "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
                             "Select to add Wildcard": (["Select the Wildcard to add to the text"],),
                             },
                "optional": {
                    "sam_model_opt": ("SAM_MODEL", {"tooltip": "Optional SAM model."}),
                    "segm_detector_opt": ("SEGM_DETECTOR", {"tooltip": "Optional segmentation detector."}),
                    "detailer_hook": ("DETAILER_HOOK", {"tooltip": "Optional detailer hook."}),
                    },
                }

    RETURN_TYPES = ("DETAILER_PIPE", )
    RETURN_NAMES = ("detailer_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, *args, **kwargs):
        basic_pipe = kwargs['basic_pipe']
        bbox_detector = kwargs['bbox_detector']
        wildcard = kwargs['wildcard']
        sam_model_opt = kwargs.get('sam_model_opt', None)
        segm_detector_opt = kwargs.get('segm_detector_opt', None)
        detailer_hook = kwargs.get('detailer_hook', None)

        model, clip, vae, positive, negative = basic_pipe
        pipe = model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, None, None, None, None
        return (pipe, )


class BasicPipeToDetailerPipeSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"base_basic_pipe": ("BASIC_PIPE", {"tooltip": "Basic pipe for the SDXL base model components."}),
                             "refiner_basic_pipe": ("BASIC_PIPE", {"tooltip": "Basic pipe for the SDXL refiner model components (VAE from base is typically used)."}),
                             "bbox_detector": ("BBOX_DETECTOR", {"tooltip": "Bounding box detector to include in the new SDXL detailer pipe."}),
                             "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "Wildcard string/prompt for the new SDXL detailer pipe."}),
                             "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
                             "Select to add Wildcard": (["Select the Wildcard to add to the text"],),
                             },
                "optional": {
                    "sam_model_opt": ("SAM_MODEL", {"tooltip": "Optional SAM model."}),
                    "segm_detector_opt": ("SEGM_DETECTOR", {"tooltip": "Optional segmentation detector."}),
                    "detailer_hook": ("DETAILER_HOOK", {"tooltip": "Optional detailer hook."}),
                    },
                }

    RETURN_TYPES = ("DETAILER_PIPE", )
    RETURN_NAMES = ("detailer_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, *args, **kwargs):
        base_basic_pipe = kwargs['base_basic_pipe']
        refiner_basic_pipe = kwargs['refiner_basic_pipe']
        bbox_detector = kwargs['bbox_detector']
        wildcard = kwargs['wildcard']
        sam_model_opt = kwargs.get('sam_model_opt', None)
        segm_detector_opt = kwargs.get('segm_detector_opt', None)
        detailer_hook = kwargs.get('detailer_hook', None)

        model, clip, vae, positive, negative = base_basic_pipe
        refiner_model, refiner_clip, refiner_vae, refiner_positive, refiner_negative = refiner_basic_pipe
        pipe = model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, refiner_model, refiner_clip, refiner_positive, refiner_negative
        return (pipe, )


class DetailerPipeToBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"detailer_pipe": ("DETAILER_PIPE", {"tooltip": "The detailer pipe from which to extract base and refiner basic pipes."}), }}

    RETURN_TYPES = ("BASIC_PIPE", "BASIC_PIPE")
    RETURN_NAMES = ("base_basic_pipe", "refiner_basic_pipe")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, detailer_pipe):
        model, clip, vae, positive, negative, _, _, _, _, _, refiner_model, refiner_clip, refiner_positive, refiner_negative = detailer_pipe
        pipe = model, clip, vae, positive, negative
        refiner_pipe = refiner_model, refiner_clip, vae, refiner_positive, refiner_negative
        return (pipe, refiner_pipe)


class EditBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {"basic_pipe": ("BASIC_PIPE", {"tooltip": "The basic pipe to be edited."}), },
                "optional": {
                     "model": ("MODEL", {"tooltip": "Optional new model to replace in the pipe."}),
                     "clip": ("CLIP", {"tooltip": "Optional new CLIP model to replace in the pipe."}),
                     "vae": ("VAE", {"tooltip": "Optional new VAE to replace in the pipe."}),
                     "positive": ("CONDITIONING", {"tooltip": "Optional new positive conditioning to replace in the pipe."}),
                     "negative": ("CONDITIONING", {"tooltip": "Optional new negative conditioning to replace in the pipe."}),
                     },
                }

    RETURN_TYPES = ("BASIC_PIPE", )
    RETURN_NAMES = ("basic_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, basic_pipe, model=None, clip=None, vae=None, positive=None, negative=None):
        res_model, res_clip, res_vae, res_positive, res_negative = basic_pipe

        if model is not None:
            res_model = model

        if clip is not None:
            res_clip = clip

        if vae is not None:
            res_vae = vae

        if positive is not None:
            res_positive = positive

        if negative is not None:
            res_negative = negative

        pipe = res_model, res_clip, res_vae, res_positive, res_negative

        return (pipe, )


class EditDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "detailer_pipe": ("DETAILER_PIPE", {"tooltip": "The detailer pipe to be edited."}),
                "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "New wildcard string. If empty, original is kept."}),
                "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
                "Select to add Wildcard": (["Select the Wildcard to add to the text"],),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Optional new model to replace in the pipe."}),
                "clip": ("CLIP", {"tooltip": "Optional new CLIP model to replace in the pipe."}),
                "vae": ("VAE", {"tooltip": "Optional new VAE to replace in the pipe."}),
                "positive": ("CONDITIONING", {"tooltip": "Optional new positive conditioning to replace in the pipe."}),
                "negative": ("CONDITIONING", {"tooltip": "Optional new negative conditioning to replace in the pipe."}),
                "bbox_detector": ("BBOX_DETECTOR", {"tooltip": "Optional new bounding box detector to replace in the pipe."}),
                "sam_model": ("SAM_MODEL", {"tooltip": "Optional new SAM model to replace in the pipe."}),
                "segm_detector": ("SEGM_DETECTOR", {"tooltip": "Optional new segmentation detector to replace in the pipe."}),
                "detailer_hook": ("DETAILER_HOOK", {"tooltip": "Optional new detailer hook to replace in the pipe."}),
            },
        }

    RETURN_TYPES = ("DETAILER_PIPE",)
    RETURN_NAMES = ("detailer_pipe",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, *args, **kwargs):
        detailer_pipe = kwargs['detailer_pipe']
        wildcard = kwargs['wildcard']
        model = kwargs.get('model', None)
        clip = kwargs.get('clip', None)
        vae = kwargs.get('vae', None)
        positive = kwargs.get('positive', None)
        negative = kwargs.get('negative', None)
        bbox_detector = kwargs.get('bbox_detector', None)
        sam_model = kwargs.get('sam_model', None)
        segm_detector = kwargs.get('segm_detector', None)
        detailer_hook = kwargs.get('detailer_hook', None)
        refiner_model = kwargs.get('refiner_model', None)
        refiner_clip = kwargs.get('refiner_clip', None)
        refiner_positive = kwargs.get('refiner_positive', None)
        refiner_negative = kwargs.get('refiner_negative', None)

        res_model, res_clip, res_vae, res_positive, res_negative, res_wildcard, res_bbox_detector, res_segm_detector, res_sam_model, res_detailer_hook, res_refiner_model, res_refiner_clip, res_refiner_positive, res_refiner_negative = detailer_pipe

        if model is not None:
            res_model = model

        if clip is not None:
            res_clip = clip

        if vae is not None:
            res_vae = vae

        if positive is not None:
            res_positive = positive

        if negative is not None:
            res_negative = negative

        if bbox_detector is not None:
            res_bbox_detector = bbox_detector

        if segm_detector is not None:
            res_segm_detector = segm_detector

        if wildcard != "":
            res_wildcard = wildcard

        if sam_model is not None:
            res_sam_model = sam_model

        if detailer_hook is not None:
            res_detailer_hook = detailer_hook

        if refiner_model is not None:
            res_refiner_model = refiner_model

        if refiner_clip is not None:
            res_refiner_clip = refiner_clip

        if refiner_positive is not None:
            res_refiner_positive = refiner_positive

        if refiner_negative is not None:
            res_refiner_negative = refiner_negative

        pipe = (res_model, res_clip, res_vae, res_positive, res_negative, res_wildcard,
                res_bbox_detector, res_segm_detector, res_sam_model, res_detailer_hook,
                res_refiner_model, res_refiner_clip, res_refiner_positive, res_refiner_negative)

        return (pipe, )


class EditDetailerPipeSDXL(EditDetailerPipe):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "detailer_pipe": ("DETAILER_PIPE", {"tooltip": "The SDXL detailer pipe to be edited."}),
                "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "New wildcard string. If empty, original is kept."}),
                "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
                "Select to add Wildcard": (["Select the Wildcard to add to the text"],),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Optional new base model."}),
                "clip": ("CLIP", {"tooltip": "Optional new base CLIP model."}),
                "vae": ("VAE", {"tooltip": "Optional new VAE."}),
                "positive": ("CONDITIONING", {"tooltip": "Optional new positive conditioning for the base model."}),
                "negative": ("CONDITIONING", {"tooltip": "Optional new negative conditioning for the base model."}),
                "refiner_model": ("MODEL", {"tooltip": "Optional new refiner model."}),
                "refiner_clip": ("CLIP", {"tooltip": "Optional new refiner CLIP model."}),
                "refiner_positive": ("CONDITIONING", {"tooltip": "Optional new positive conditioning for the refiner model."}),
                "refiner_negative": ("CONDITIONING", {"tooltip": "Optional new negative conditioning for the refiner model."}),
                "bbox_detector": ("BBOX_DETECTOR", {"tooltip": "Optional new bounding box detector."}),
                "sam_model": ("SAM_MODEL", {"tooltip": "Optional new SAM model."}),
                "segm_detector": ("SEGM_DETECTOR", {"tooltip": "Optional new segmentation detector."}),
                "detailer_hook": ("DETAILER_HOOK", {"tooltip": "Optional new detailer hook."}),
            },
        }
