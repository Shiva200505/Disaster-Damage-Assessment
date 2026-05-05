import pytest
import numpy as np
import io
from PIL import Image
from member2_siamese.siamese_net import SiameseUNet

@pytest.fixture
def tiny_image():
    """Returns a 64x64x3 uint8 numpy array as a PNG bytes object."""
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@pytest.fixture  
def siamese_model():
    """Returns an untrained SiameseUNet with random weights."""
    return SiameseUNet(in_channels=3, pretrained=False)
