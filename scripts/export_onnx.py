"""
Professional Core Graph Topology Serialization Tool
Converts PyTorch deep learning weights into optimized ONNX graph representations.
"""

import sys
import torch
import torchvision.models as models

def execute_serialization_pipeline():
    print("[PIPELINE HUB] Accessing Vision Framework Weight Repository...")
    
    try:
        # Using string matching 'DEFAULT' guarantees backwards and forwards compatibility
        # across all PyTorch/Torchvision version boundaries.
        model = models.resnet50(weights='DEFAULT')
        
        # Enforce evaluation state (Freezes Dropout layers and Batch Normalization tracking metrics)
        model.eval()
        
        # Enforce MLPerf Industry Standards: Batch=1, RGB Channels=3, Height=224, Width=224
        tensor_dimension_blueprint = torch.randn(1, 3, 224, 224)
        target_output_path = "resnet50.onnx"
        
        print(f"[PIPELINE HUB] Freezing structural graph layers to destination path: '{target_output_path}'...")
        
        # We drop the strict custom naming parameters to ensure the model structures
        # export via the clean default graph path, bypassing legacy parsing alerts.
        torch.onnx.export(
            model,
            tensor_dimension_blueprint,
            target_output_path,
            export_params=True,
            opset_version=13,  # Adjusted to Opset 13 for maximal compatibility with standard trtexec versions
            do_constant_folding=True
        )
        print("[SUCCESS] Operational Model Topology Formed and Documented Successfully.")
        
    except Exception as error_context:
        print(f"[CRITICAL ERROR] Graph Parsing Aborted: {str(error_context)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    execute_serialization_pipeline()
