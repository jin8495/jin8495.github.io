import unittest

import os
from pathlib import Path
from md2post.ObsidianPage import ObsidianPage

class TestObsidianPage(unittest.TestCase):
    def setUp(self):
        # Create a temporary markdown file for testing
        self.temp_file = Path("temp_page.md")
        self.temp_file.write_text(
            "---\n"
            "blog-post: true\n"
            "blog-title: Test Page\n"
            "blog-directory: TestDir\n"
            "blog-date: 2024-11-23\n"
            "---\n"
            "Content here.\n"
        )

    def tearDown(self):
        # Remove the temporary file after each test
        if self.temp_file.exists():
            self.temp_file.unlink()

    def test_is_publishable_page(self):
        # Initialize ObsidianPage object
        page = ObsidianPage(self.temp_file)
        self.assertTrue(page.is_publishable_page())

    def test_write_page(self):
        # Initialize ObsidianPage object
        page = ObsidianPage(self.temp_file)

        # Simulate writing the page
        current_dir = Path(os.path.abspath(__file__)).parent
        output_dir = current_dir / "output_test"
        output_path = output_dir / "output_page.md"
        prefix = "2024-11-23"
        page.write_page(output_path, exclude_prpt_keys=["blog-post"], prefix=prefix)

        # Verify the output file content
        new_output_path = output_dir / f"{prefix}_{output_path.name}"
        with open(new_output_path, "r") as f:
            result = f.read()
        self.assertIn("title: Test Page", result)
        self.assertNotIn("post", result)

        # Clean up
        if new_output_path.exists():
            new_output_path.unlink()
            output_dir.rmdir()

if __name__ == "__main__":
    unittest.main()