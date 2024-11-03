import os
import base64
import json
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Union
import glob
from dotenv import load_dotenv
import random
import colorsys
from litellm import completion

# Load environment variables from .env file
load_dotenv()

class UIVisionProcessor:
    def __init__(self):
        """Initialize the processor with API key from .env."""
        self.output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(self.output_dir, exist_ok=True)

    def encode_image(self, image_path: str) -> tuple[str, str]:
        """Encode an image file to base64 and determine its media type."""
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        ext = image_path.lower().split('.')[-1]
        media_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        media_type = media_types.get(ext, 'image/jpeg')

        return encoded_string, media_type

    def process_images(self, image_paths: Union[str, List[str]]) -> Dict:
        """Process images to detect UI elements and their bounding boxes."""
        if isinstance(image_paths, str):
            if os.path.isdir(image_paths):
                image_paths = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp']:
                    image_paths.extend(glob.glob(os.path.join(image_paths, ext)))
                if not image_paths:
                    raise ValueError(f"No supported images found in directory {image_paths}")
            else:
                if not os.path.exists(image_paths):
                    raise ValueError(f"Image path does not exist: {image_paths}")
                image_paths = [image_paths]

        print(f"Processing {len(image_paths)} images...")

        content = []
        for idx, img_path in enumerate(image_paths, 1):
            print(f"Processing image {idx}: {img_path}")
            content.append({
                "type": "text",
                "text": f"Image {idx}:"
            })

            try:
                # Directly encode the original image
                encoded_image, media_type = self.encode_image(img_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": encoded_image
                    }
                })
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                continue

        system_prompt = """
        You are a precise UI element detection system specialized in identifying EVERY visual component in user interfaces. 

        YOUR TASK:
        Detect and locate every single visual element in the UI with pixel-perfect accuracy.

        OUTPUT FORMAT:
        {
            "image_number": [
                {
                    "element": "specific-element-type",
                    "label": "exact-content-or-purpose",
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence_score
                }
            ]
        }

        BOUNDING BOX GUIDELINES:
        1. Tight Boundaries:
           - Boxes should tightly wrap around elements
           - Include padding/margins only if they're part of the element
           - For text, capture the exact text bounds
           - For buttons, include the full clickable area
           - For icons, include only the icon artwork

        2. Nested Elements:
           - Detect both containers and their contents
           - Menu items within dropdowns
           - Text within buttons
           - Icons within buttons
           - Labels within form fields

        3. Common UI Patterns:
           - Navigation bars: [0.0, 0.0, 1.0, 0.08] (full width, top)
           - Sidebars: [0.0, 0.0, 0.25, 1.0] (full height, left side)
           - Modal dialogs: centered, with padding
           - Buttons: include full padding and borders
           - Text fields: include borders and internal padding

        4. Precision Requirements:
           - Use 4 decimal places for coordinates
           - Ensure x2 > x1 and y2 > y1
           - Coordinates must be normalized (0-1)
           - No overlapping boxes unless elements truly overlap
           - No gaps between adjacent elements

        ELEMENT HIERARCHY:
        1. Page Structure:
           - header
           - main-content
           - sidebar
           - footer

        2. Navigation:
           - nav-bar
           - nav-item
           - nav-dropdown
           - breadcrumb

        3. Content:
           - heading-1 (main title)
           - heading-2 (section titles)
           - heading-3 (subsections)
           - paragraph-text
           - list-item
           - table-cell

        4. Interactive:
           - button-primary (main actions)
           - button-secondary (optional actions)
           - input-field (form inputs)
           - checkbox
           - radio-button
           - dropdown-select

        5. Media:
           - icon (interface icons)
           - image (content images)
           - avatar (user images)
           - logo (brand images)

        6. Status/Feedback:
           - alert-message
           - progress-bar
           - loading-spinner
           - tooltip
           - badge

        CRITICAL RULES:
        1. PRECISION - Coordinates must perfectly match visual boundaries
        2. COMPLETENESS - Detect every element, no matter how small
        3. HIERARCHY - Maintain proper nesting of elements
        4. NO OVERLAP - Unless elements truly overlap in UI
        5. NO GAPS - Adjacent elements should touch exactly
        6. CONSISTENCY - Similar elements should have similar sizes
        7. OUTPUT - Return only valid JSON, no explanations

        Remember: Your coordinate accuracy directly affects the usability of the UI analysis.
        """

        try:
            print("Analyzing images with Claude...")
            response = completion(
                model="claude-3-5-sonnet-20241022",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=8000,
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )

            response_text = response.choices[0].message.content
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                bboxes = json.loads(response_text[json_start:json_end])
                print("Successfully extracted bounding boxes")
                return bboxes
            else:
                raise ValueError("No valid JSON found in response")

        except Exception as e:
            print(f"Error processing images: {str(e)}")
            return None

    def get_random_color(self):
        """Generate a random vibrant color using HSV color space."""
        # Use golden ratio to generate well-distributed hues
        golden_ratio = 0.618033988749895
        hue = random.random()
        hue = (hue + golden_ratio) % 1.0
        
        # Use high saturation and value for vibrant colors
        saturation = 0.85 + random.random() * 0.15  # 0.85-1.00
        value = 0.85 + random.random() * 0.15       # 0.85-1.00
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert to hex color string
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )

    def draw_bounding_boxes(self, image_path: str, bboxes: List[Dict]):
        """Draw bounding boxes on a UI screenshot and save the result."""
        try:
            # Open original image directly
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            width, height = image.size

            font_size = int(height * 0.015)  # Scale font to image height
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                font = ImageFont.load_default()

            color_map = {
                'text': '#FF4444',     # Red for text
                'button': '#44FF44',   # Green for buttons
                'input': '#4444FF',    # Blue for inputs
                'icon': '#FFFF44',     # Yellow for icons
                'container': '#FF44FF', # Purple for containers
                'nav': '#44FFFF',      # Cyan for navigation
                'image': '#FF8844',    # Orange for images
                'status': '#88FF44',   # Lime for status elements
                'modal': '#FF4488',    # Pink for modals
                'list': '#4488FF',     # Light blue for lists
            }

            for bbox in bboxes:
                try:
                    element_type = bbox.get('element', 'unknown').lower()
                    base_type = next((k for k in color_map.keys() if k in element_type), 'unknown')
                    color = color_map.get(base_type, self.get_random_color())

                    # Convert normalized coordinates to pixel coordinates
                    x1, y1, x2, y2 = bbox['bbox']
                    x1, x2 = sorted([x1 * width, x2 * width])
                    y1, y2 = sorted([y1 * height, y2 * height])

                    # Validate coordinates
                    x1 = max(0, min(x1, width))
                    x2 = max(0, min(x2, width))
                    y1 = max(0, min(y1, height))
                    y2 = max(0, min(y2, height))

                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue

                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                    # Create label
                    label_parts = []
                    if 'element' in bbox:
                        label_parts.append(bbox['element'])
                    if 'label' in bbox:
                        label_parts.append(f": {bbox['label']}")
                    if 'confidence' in bbox:
                        label_parts.append(f" ({bbox['confidence']:.2f})")
                    
                    label = "".join(label_parts) if label_parts else "Unknown Element"

                    # Draw label with background
                    label_y = max(font_size, y1 - font_size)
                    text_bbox = draw.textbbox((x1, label_y), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    bg_x1 = min(max(0, x1), width - text_width)
                    bg_x2 = min(bg_x1 + text_width, width)
                    bg_y1 = min(max(0, label_y), height - text_height)
                    bg_y2 = min(bg_y1 + text_height, height)

                    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2],
                                 fill=(0, 0, 0, 180))
                    draw.text((bg_x1, bg_y1), label, fill=color, font=font)

                except Exception as e:
                    print(f"Warning: Skipping invalid bounding box: {str(e)}")
                    continue

            # Save output
            basename = os.path.basename(image_path)
            output_path = os.path.join(self.output_dir, f'ui_analyzed_{basename}')
            image.save(output_path)
            print(f"Saved annotated UI analysis to: {output_path}")

        except Exception as e:
            print(f"Error drawing UI annotations for {image_path}: {str(e)}")
            raise

def main():
    """Process UI screenshots with LiteLLM"""
    print("\n=== LiteLLM UI Analysis ===")
    print("Enter the path to a UI screenshot or a directory containing screenshots.")
    input_path = input("\nPath: ").strip()

    if not os.path.exists(input_path):
        print(f"Error: Path '{input_path}' does not exist!")
        return

    try:
        processor = UIVisionProcessor()
        result = processor.process_images(input_path)

        if result:
            for image_num, bboxes in result.items():
                if os.path.isdir(input_path):
                    images = glob.glob(os.path.join(input_path, f'*{image_num}.*'))
                    if images:
                        image_path = images[0]
                else:
                    image_path = input_path

                processor.draw_bounding_boxes(image_path, bboxes)

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
