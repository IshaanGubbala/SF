from PIL import Image, ImageEnhance
import numpy as np

# Load the image
image_path = "/Users/ishaangubbala/Desktop/Screenshot 2025-03-12 at 8.49.26 PM.png"
image = Image.open(image_path)

# Apply a purple tint to the image
purple_tint = Image.new('RGBA', image.size, (128, 0, 128, 100))
purple_image = Image.alpha_composite(image.convert('RGBA'), purple_tint)

# Save and display the modified image
purple_image_path = "purple_tinted_chart.png"
purple_image.save(purple_image_path)
purple_image.show()
