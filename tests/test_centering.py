import unittest
import numpy as np
import cv2
from app.features import extract_centering_metrics

class TestCenteringDetection(unittest.TestCase):
    def create_mock_card(self, width=600, height=900, left_border=36, right_border=36, top_border=36, bottom_border=36):
        # Create a mock card image: BGR format (all zeros by default, representing black background)
        # We represent the border area with one intensity, and the inner art area with another intensity
        # to create a sharp transition (high gradient peak).
        card = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Border color: say, light gray (180, 180, 180)
        card[:, :] = [180, 180, 180]
        
        # Inner art coordinates
        start_x = left_border
        end_x = width - right_border
        start_y = top_border
        end_y = height - bottom_border
        
        # Fill inner art area with a different color/intensity (e.g. black) to create the edge
        card[start_y:end_y, start_x:end_x] = [0, 0, 0]
        
        # Add slight blur to simulate realistic image transitions
        card = cv2.GaussianBlur(card, (3, 3), 0)
        return card

    def test_perfectly_centered_card(self):
        # Perfectly centered borders at 6% of width (36px) and 4% of height (36px)
        card_image = self.create_mock_card(left_border=36, right_border=36, top_border=36, bottom_border=36)
        metrics = extract_centering_metrics(card_image)
        
        # Check that centering score is high (perfect centering with sub-pixel shift is >= 9.0)
        self.assertGreaterEqual(metrics["centering_score"], 9.0)
        self.assertEqual(metrics["centering_quality"], "elite")
        
        # Check that the percentages are close to 50%
        self.assertTrue(47.0 <= metrics["left_percent"] <= 53.0)
        self.assertTrue(47.0 <= metrics["top_percent"] <= 53.0)

    def test_off_center_card(self):
        # Off-center: left border is 24px, right border is 48px (ratios roughly 33/67)
        card_image = self.create_mock_card(left_border=24, right_border=48, top_border=36, bottom_border=36)
        metrics = extract_centering_metrics(card_image)
        
        self.assertLess(metrics["centering_score"], 9.0)
        self.assertTrue(30.0 <= metrics["left_percent"] <= 36.0)
        self.assertTrue(47.0 <= metrics["top_percent"] <= 53.0)

    def test_invalid_borders_fallback(self):
        # Create a card with transitions that are impossible/crossed or too deep (e.g. no transitions, plain black)
        # This should trigger the fallback to standard 6% border (50/50)
        card_image = np.zeros((900, 600, 3), dtype=np.uint8)
        metrics = extract_centering_metrics(card_image)
        
        self.assertEqual(metrics["left_right_ratio"], "50/50")
        self.assertEqual(metrics["top_bottom_ratio"], "50/50")
        self.assertEqual(metrics["centering_quality"], "elite")

if __name__ == "__main__":
    unittest.main()
