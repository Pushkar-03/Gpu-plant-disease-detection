import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

class TestEnvironment(unittest.TestCase):
    def test_imports(self):
        import src.config
        import src.device
        self.assertTrue(True)

    def test_device_fallback(self):
        from src.device import get_device
        device = get_device()
        if isinstance(device, str):
            self.assertEqual(device, "cpu")
        else:
            self.assertEqual(device.type, "cpu")
            
    def test_config_paths(self):
        from src import config
        self.assertEqual(config.DATA_DIR.name, "data")
        self.assertEqual(config.IMAGE_SIZE, (224, 224))
        
    def test_model_creation(self):
        try:
            import torch
            from src.model import create_model
            model = create_model(num_classes=5, pretrained=False)
            self.assertEqual(model.classifier[1].out_features, 5)
        except ImportError:
            self.skipTest("PyTorch is not installed. Skipping model test.")

if __name__ == '__main__':
    unittest.main()
