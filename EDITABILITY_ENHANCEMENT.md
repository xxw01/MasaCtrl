# MasaCtrl Editability Enhancement

This document describes the enhanced MasaCtrl implementation with three-path processing for improved image editability.

## Overview

The enhanced MasaCtrl introduces a third processing path (Lm) alongside the existing source reconstruction (Ls) and target editing (Lt) paths. This enhancement improves editability by making the editing process less dependent on the original structure while maintaining quality.

## Key Features

### 🚀 Three-Path Processing
- **Ls Path**: Source image reconstruction (unchanged from original)
- **Lt Path**: Target image editing with enhanced noise strategy
- **Lm Path**: Editability enhancement path using source attention features

### 🎛️ Enhanced Control
- **Editability Weight (ws)**: Adjustable parameter controlling editing strength
- **Noise Update Strategy**: `et = em + ws * (et - em)` for improved editability
- **Cross-Attention Control**: Lm path uses target prompt for better editing guidance

### 🔧 Backward Compatibility
- All existing MasaCtrl functionality preserved
- Original two-path processing still available
- Drop-in replacement for existing code

## Installation

No additional dependencies required beyond the original MasaCtrl requirements:

```bash
pip install diffusers transformers torch torchvision
```

## Quick Start

### Basic Three-Path Usage

```python
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_processor import register_attention_processor

# Load pipeline
pipe = MasaCtrlPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Register editability processor
register_attention_processor(
    model=pipe.unet,
    processor_type="MasaCtrlEditabilityProcessor",
    start_step=4,
    start_layer=10,
    total_steps=50,
    model_type="SD"
)

# Generate with three-path processing
results = pipe.__call_three_path__(
    source_prompt="a beautiful landscape with mountains and a lake",
    target_prompt="a beautiful landscape with mountains, a lake, and a sunset sky",
    editability_weight=1.0,  # Adjustable editing strength
    num_inference_steps=50,
    guidance_scale=7.5
)

# Access results
source_image = results['source']        # Ls path output
target_image = results['target']        # Lt path output  
editability_image = results['editability']  # Lm path output
```

### Editability Weight Tuning

The `editability_weight` parameter controls the strength of editing:

```python
# Subtle editing (closer to source)
results_subtle = pipe.__call_three_path__(
    source_prompt="a cat sitting on a chair",
    target_prompt="a dog sitting on a chair", 
    editability_weight=0.5
)

# Strong editing (more divergence from source)
results_strong = pipe.__call_three_path__(
    source_prompt="a cat sitting on a chair",
    target_prompt="a dog sitting on a chair",
    editability_weight=2.0  
)
```

## Technical Details

### Three-Path Attention Control

The enhanced processor implements sophisticated attention control:

1. **Ls Path (Source Reconstruction)**:
   - Standard self-attention: Qs ⊗ Ks → Vs
   - Uses source prompt for cross-attention

2. **Lt Path (Target Editing)**:
   - Modified self-attention: Qt ⊗ Ks → Vs (attends to source keys/values)
   - Uses target prompt for cross-attention
   - Enhanced with editability noise strategy

3. **Lm Path (Editability Enhancement)**:
   - Hybrid self-attention: Qs ⊗ Ks → Vm (source queries/keys, own values)
   - Uses target prompt for cross-attention
   - Provides editability reference for noise update

### Noise Update Strategy

The enhanced noise update formula improves editability:

```
et_enhanced = em + ws * (et - em)
```

Where:
- `et`: Original target path noise
- `em`: Editability path noise  
- `ws`: Editability weight (tunable parameter)
- `et_enhanced`: Final enhanced target noise

This strategy encourages the target editing to diverge from the original structure guided by the editability path.

## API Reference

### MasaCtrlPipeline.__call_three_path__()

```python
def __call_three_path__(
    self,
    source_prompt: str | List[str],
    target_prompt: str | List[str], 
    batch_size: int = 1,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    eta: float = 0.0,
    latents_s: Optional[torch.Tensor] = None,
    latents_t: Optional[torch.Tensor] = None, 
    latents_m: Optional[torch.Tensor] = None,
    editability_weight: float = 1.0,
    return_intermediates: bool = False,
    **kwargs
) -> Dict[str, torch.Tensor]:
```

**Parameters:**
- `source_prompt`: Source description for Ls path
- `target_prompt`: Target description for Lt and Lm paths
- `editability_weight`: Editing strength control (default: 1.0)
- `latents_s/t/m`: Optional initial latents for each path
- `return_intermediates`: Return intermediate latents if True

**Returns:**
Dictionary with keys: `'source'`, `'target'`, `'editability'`

### MasaCtrlEditabilityProcessor

```python
class MasaCtrlEditabilityProcessor(nn.Module):
    def __init__(
        self,
        start_step: int = 4,
        start_layer: int = 10,
        layer_idx: Optional[List[int]] = None,
        step_idx: Optional[List[int]] = None,
        total_layers: int = 32,
        total_steps: int = 50,
        model_type: str = "SD"
    ):
```

**Parameters:**
- `start_step`: Step to begin attention control (default: 4)
- `start_layer`: Layer to begin attention control (default: 10)
- `layer_idx`: Specific layers for control (optional)
- `step_idx`: Specific steps for control (optional)
- `total_layers`: Total transformer layers (16 for SD, 70 for SDXL)
- `total_steps`: Total denoising steps
- `model_type`: "SD" or "SDXL"

## Examples

### Real Image Editing

```python
# Load and invert real image
real_image = Image.open("input.jpg")
inverted_latents, _ = pipe.invert(
    image=real_image, 
    prompt="original description of the image"
)

# Edit with three-path processing
results = pipe.__call_three_path__(
    source_prompt="original description",
    target_prompt="edited description",
    latents_s=inverted_latents,
    latents_t=inverted_latents,
    latents_m=inverted_latents,
    editability_weight=1.2
)
```

### Batch Processing

```python
source_prompts = [
    "a red car in a parking lot",
    "a blue house by the lake"
]
target_prompts = [
    "a red sports car in a parking lot", 
    "a blue modern house by the lake"
]

results = pipe.__call_three_path__(
    source_prompt=source_prompts,
    target_prompt=target_prompts,
    batch_size=2,
    editability_weight=1.0
)
```

## Best Practices

### Editability Weight Selection
- **0.3-0.7**: Subtle edits, preserves original structure
- **0.8-1.2**: Balanced editing (recommended range)
- **1.3-2.0**: Strong edits, significant structural changes
- **>2.0**: Extreme editing, may lose coherence

### Prompt Guidelines
- Use detailed source prompts for better reconstruction
- Target prompts should be similar to source with specific changes
- Avoid completely different concepts for better results

### Performance Optimization
- Use lower `start_step` (e.g., 2-4) for stronger control
- Higher `start_layer` (e.g., 12-15) for efficiency
- Adjust `num_inference_steps` based on quality needs

## Troubleshooting

### Common Issues

**1. Poor editing quality**
- Try adjusting `editability_weight` (0.8-1.5)
- Ensure prompts are well-aligned
- Check `start_step` and `start_layer` values

**2. Memory issues**
- Reduce batch size
- Use mixed precision (fp16)
- Process paths sequentially if needed

**3. Inconsistent results**
- Set fixed random seeds
- Use same latents for comparison
- Ensure proper processor registration

### Bug Fixes Included

- Fixed PIL Image type checking in `image2latent()` method
- Improved numerical stability in denoising steps
- Enhanced error handling for malformed inputs

## Migration Guide

### From Original MasaCtrl

Existing code using original MasaCtrl continues to work unchanged:

```python
# Original code - still works
pipe = MasaCtrlPipeline.from_pretrained("model")
register_attention_processor(pipe.unet, "MasaCtrlProcessor")
result = pipe(prompt="...", num_inference_steps=50)

# Enhanced code - new features
register_attention_processor(pipe.unet, "MasaCtrlEditabilityProcessor") 
results = pipe.__call_three_path__(source_prompt="...", target_prompt="...")
```

### Configuration Updates

Update processor registration for enhanced features:

```python
# Old
register_attention_processor(model, "MasaCtrlProcessor")

# New  
register_attention_processor(model, "MasaCtrlEditabilityProcessor")
```

## Contributing

When contributing to the editability enhancement features:

1. Maintain backward compatibility
2. Add appropriate tests for new functionality
3. Update documentation for API changes
4. Follow existing code style and patterns

## License

This enhancement maintains the same license as the original MasaCtrl project.