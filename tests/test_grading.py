import unittest
import numpy as np
import cv2
from app.features import extract_centering_metrics, analyze_edges_and_corners
from app.surface_ai import analyze_surface
from app.grading import estimate_grade

class TestGradingHeuristics(unittest.TestCase):
    def create_mock_perfect_card(self):
        # Standard portrait card dimensions: 900 height, 600 width
        height, width = 900, 600
        card = np.zeros((height, width, 3), dtype=np.uint8)
        
        # solid yellow border
        card[:, :] = [0, 200, 255]
        
        # border width 36
        border_w = 36
        
        # Colorful artwork
        artwork = np.zeros((height - 2*border_w, width - 2*border_w, 3), dtype=np.uint8)
        h_art, w_art = artwork.shape[:2]
        
        for y in range(h_art):
            for x in range(w_art):
                r = int(255 * (y / h_art))
                g = int(255 * (x / w_art))
                b = int(255 * (1.0 - (x * y) / (w_art * h_art)))
                artwork[y, x] = [b, g, r]
                
        card[border_w:height-border_w, border_w:width-border_w] = artwork
        card = cv2.GaussianBlur(card, (3, 3), 0)
        return card

    def test_perfect_card_gets_gem_mint_grade(self):
        card = self.create_mock_perfect_card()
        
        centering = extract_centering_metrics(card)
        edges_corners = analyze_edges_and_corners(card)
        surface = analyze_surface(card)
        grade = estimate_grade(centering, edges_corners, surface)
        
        # The estimated grade range should be '9-10' for a perfect card
        self.assertEqual(grade["estimated_range"], "9-10")
        
        # Centering base score should be high (10.0 or close)
        self.assertEqual(grade["centering_base_score"], 10.0)
        
        # No wear cap and no surface penalty should be applied
        self.assertIsNone(grade["edge_wear_cap"])
        self.assertEqual(grade["surface_penalty"], 0.0)
        
        # Overall estimated score should be Gem Mint (>= 9.5)
        self.assertGreaterEqual(grade["estimated_score"], 9.5)
        self.assertEqual(grade["confidence"], "high")

if __name__ == "__main__":
    unittest.main()
