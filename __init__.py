# Import the CartesianCSVNode from systemlevel.py
from .systemlevel import CartesianCSVNode

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "CartesianCSVNode": CartesianCSVNode
}

# Optional category grouping
NODE_DISPLAY_NAME_MAPPINGS = {
    "CartesianCSVNode": "Custom/Cartesian CSV Node"
}
