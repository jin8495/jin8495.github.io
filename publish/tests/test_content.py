import unittest
from pathlib import Path
from md2post.Content import Content

class TestContent(unittest.TestCase):
    def setUp(self):
        # Create a temporary content file for testing
        self.temp_file = Path("temp_content.md")
        self.temp_file.write_text(
            "---\n"
            "blog-post: true\n"
            "blog-title: Test Content\n"
            "---\n"
            "# blog-comments-start\n"
            "This is a comment.\n"
            "# blog-comments-end\n"
            "Actual content here.\n"
        )

    def tearDown(self):
        # Remove the temporary file after each test
        if self.temp_file.exists():
            self.temp_file.unlink()

    def test_remove_comments(self):
        # Initialize Content object with test file
        content = Content(self.temp_file)

        # Simulate writing content with comments removed
        output_path = Path("output_content.md")
        content.write_contents(output_path, remove_comments=True)

        # Verify the output file content
        with open(output_path, "r") as f:
            result = f.read()
        self.assertNotIn("This is a comment.", result)
        self.assertIn("Actual content here.", result)

        # Clean up
        if output_path.exists():
            output_path.unlink()

    def test_no_remove_comments(self):
        # Initialize Content object with test file
        content = Content(self.temp_file)

        # Simulate writing content without removing comments
        output_path = Path("output_content.md")
        content.write_contents(output_path, remove_comments=False)

        # Verify the output file content
        with open(output_path, "r") as f:
            result = f.read()
        self.assertIn("This is a comment.", result)
        self.assertIn("Actual content here.", result)

        # Clean up
        if output_path.exists():
            output_path.unlink()

if __name__ == "__main__":
    unittest.main()