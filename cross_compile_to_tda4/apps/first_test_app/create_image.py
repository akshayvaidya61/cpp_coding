from PIL import Image

# Define image size
width, height = 1280, 960

# Create a grayscale image with the specified size
image = Image.new('L', (width, height))

# Optionally, you can fill the image with a specific color (e.g., 128 for gray)
image.putdata([128] * (width * height))

# Save the image
image.save('grayscale_image.png')

