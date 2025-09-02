"""
MasaCtrl Editability Enhancement Usage Example

This script demonstrates how to use the new three-path editability enhancement features
in MasaCtrl for improved image synthesis and editing capabilities.

Features demonstrated:
1. Basic three-path pipeline usage
2. Editability processor configuration  
3. Different editability weight settings
4. Comparison with traditional two-path approach
"""

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline

# Import MasaCtrl modules
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_processor import register_attention_processor, MasaCtrlEditabilityProcessor


def setup_editability_pipeline(model_path="runwayml/stable-diffusion-v1-5", device="cuda"):
    """
    Setup MasaCtrl pipeline with editability enhancement processor.
    
    Args:
        model_path: HuggingFace model path or local path to Stable Diffusion model
        device: Device to run on ("cuda" or "cpu")
    
    Returns:
        Configured MasaCtrlPipeline with editability processor
    """
    print("Loading Stable Diffusion model...")
    
    # Load the base Stable Diffusion pipeline
    pipe = MasaCtrlPipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    
    # Register the editability processor
    print("Registering MasaCtrlEditabilityProcessor...")
    register_attention_processor(
        model=pipe.unet,
        processor_type="MasaCtrlEditabilityProcessor",
        start_step=4,       # Start mutual attention control from step 4
        start_layer=10,     # Start from layer 10
        total_steps=50,     # Total denoising steps
        total_layers=16,    # Total transformer layers (16 for SD v1.5)
        model_type="SD"     # Model type
    )
    
    return pipe


def run_three_path_editing(
    pipe, 
    source_prompt, 
    target_prompt,
    editability_weight=1.0,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    height=512,
    width=512
):
    """
    Run three-path editability enhanced image generation.
    
    Args:
        pipe: MasaCtrlPipeline with editability processor
        source_prompt: Source image description for reconstruction (Ls path)
        target_prompt: Target image description for editing (Lt and Lm paths)
        editability_weight: Editability enhancement weight (ws parameter)
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        height: Image height
        width: Image width
    
    Returns:
        Dictionary with generated images for each path
    """
    print(f"Generating images with editability_weight={editability_weight}")
    print(f"Source prompt: '{source_prompt}'")
    print(f"Target prompt: '{target_prompt}'")
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate images using three-path method
    results = pipe.__call_three_path__(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        batch_size=1,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        editability_weight=editability_weight,
        return_intermediates=False
    )
    
    return results


def compare_editability_weights(pipe, source_prompt, target_prompt, weights=[0.5, 1.0, 1.5, 2.0]):
    """
    Compare different editability weight values to demonstrate the effect.
    
    Args:
        pipe: MasaCtrlPipeline
        source_prompt: Source prompt
        target_prompt: Target prompt  
        weights: List of editability weights to test
    
    Returns:
        List of results for each weight
    """
    print("\n" + "="*60)
    print("EDITABILITY WEIGHT COMPARISON")
    print("="*60)
    
    results = []
    
    for weight in weights:
        print(f"\n--- Testing editability_weight = {weight} ---")
        
        result = run_three_path_editing(
            pipe, 
            source_prompt, 
            target_prompt,
            editability_weight=weight,
            seed=42  # Same seed for fair comparison
        )
        
        results.append({
            'weight': weight,
            'images': result
        })
        
        print(f"Generated images for weight {weight}")
    
    return results


def demonstrate_usage():
    """
    Main demonstration of the editability enhancement features.
    """
    print("MasaCtrl Editability Enhancement Demo")
    print("="*50)
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Example prompts
    source_prompt = "a beautiful landscape with mountains and a lake"
    target_prompt = "a beautiful landscape with mountains, a lake, and a sunset sky"
    
    try:
        # Setup pipeline (note: this will fail without actual model access)
        # pipe = setup_editability_pipeline(device=device)
        
        # For demonstration purposes, we'll show the API usage
        print("\nAPI Usage Example:")
        print("-" * 30)
        
        print("1. Setup Pipeline:")
        print("""
pipe = MasaCtrlPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
register_attention_processor(
    model=pipe.unet,
    processor_type="MasaCtrlEditabilityProcessor",
    start_step=4, start_layer=10, total_steps=50, model_type="SD"
)
        """)
        
        print("2. Run Three-Path Generation:")
        print(f"""
results = pipe.__call_three_path__(
    source_prompt="{source_prompt}",
    target_prompt="{target_prompt}",
    editability_weight=1.0,
    num_inference_steps=50,
    guidance_scale=7.5
)

# Access generated images:
source_image = results['source']      # Ls path - source reconstruction
target_image = results['target']      # Lt path - target editing  
editability_image = results['editability']  # Lm path - editability enhancement
        """)
        
        print("3. Editability Weight Effects:")
        print("""
- editability_weight = 0.5: Subtle editing, closer to source
- editability_weight = 1.0: Balanced editing (default)
- editability_weight = 1.5: Stronger editing, more divergence from source
- editability_weight = 2.0: Maximum editing strength
        """)
        
        print("4. Key Benefits:")
        print("""
✓ Three-path processing for enhanced editability
✓ Better control over editing strength via editability_weight
✓ Improved structure preservation with enhanced attention control
✓ Target prompt guidance for cross-attention in Lm path
✓ Backward compatible with existing MasaCtrl functionality
        """)
        
        # If you have access to models, uncomment these lines:
        # results = compare_editability_weights(pipe, source_prompt, target_prompt)
        # save_comparison_results(results)
        
    except Exception as e:
        print(f"Note: Full demo requires Stable Diffusion models. Error: {e}")
        print("The implementation is ready for use when models are available.")


def save_comparison_results(results, output_dir="./editability_comparison"):
    """
    Save comparison results to files.
    
    Args:
        results: Results from compare_editability_weights
        output_dir: Directory to save images
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    for result in results:
        weight = result['weight']
        images = result['images']
        
        # Save each path's image
        for path_name, image_tensor in images.items():
            # Convert tensor to PIL Image (assuming image_tensor is in correct format)
            # image_pil = tensor_to_pil(image_tensor)
            # image_pil.save(f"{output_dir}/weight_{weight}_{path_name}.png")
            print(f"Would save: {output_dir}/weight_{weight}_{path_name}.png")


if __name__ == "__main__":
    demonstrate_usage()